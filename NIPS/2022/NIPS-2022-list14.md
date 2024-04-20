## [2600] Fair Infinitesimal Jackknife: Mitigating the Influence of Biased Training Data Points Without Refitting

        **Authors**: *Prasanna Sattigeri, Soumya Ghosh, Inkit Padhi, Pierre L. Dognin, Kush R. Varshney*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e94481b99473c83b2e79d91c64eb37d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e94481b99473c83b2e79d91c64eb37d1-Abstract-Conference.html)

        **Abstract**:

        In consequential decision-making applications, mitigating unwanted biases in machine learning models that yield systematic disadvantage to members of groups delineated by sensitive attributes such as race and gender is one key intervention to strive for equity. Focusing on demographic parity and equality of opportunity, in this paper we propose an algorithm that improves the fairness of a pre-trained classifier by simply dropping carefully selected training data points. We select instances based on their influence on the fairness metric of interest, computed using an infinitesimal jackknife-based approach. The dropping of training points is done in principle, but in practice does not require the model to be refit. Crucially, we find that such an intervention does not substantially reduce the predictive performance of the model but drastically improves the fairness metric. Through careful experiments, we evaluate the effectiveness of the proposed approach on diverse tasks and find that it consistently improves upon existing alternatives.

        ----

        ## [2601] JAWS: Auditing Predictive Uncertainty Under Covariate Shift

        **Authors**: *Drew Prinster, Anqi Liu, Suchi Saria*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e944bacecce6b06374ac39b260348db0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e944bacecce6b06374ac39b260348db0-Abstract-Conference.html)

        **Abstract**:

        We propose \textbf{JAWS}, a series of wrapper methods for distribution-free uncertainty quantification tasks under covariate shift, centered on the core method \textbf{JAW}, the \textbf{JA}ckknife+ \textbf{W}eighted with data-dependent likelihood-ratio weights. JAWS also includes computationally efficient \textbf{A}pproximations of JAW using higher-order influence functions: \textbf{JAWA}. Theoretically, we show that JAW relaxes the jackknife+'s assumption of data exchangeability to achieve the same finite-sample coverage guarantee even under covariate shift. JAWA further approaches the JAW guarantee in the limit of the sample size or the influence function order under common regularity assumptions. Moreover, we propose a general approach to repurposing predictive interval-generating methods and their guarantees to the reverse task: estimating the probability that a prediction is erroneous, based on user-specified error criteria such as a safe or acceptable tolerance threshold around the true label. We then propose \textbf{JAW-E} and \textbf{JAWA-E} as the repurposed proposed methods for this \textbf{E}rror assessment task. Practically, JAWS outperform state-of-the-art predictive inference baselines in a variety of biased real world data sets for interval-generation and error-assessment predictive uncertainty auditing tasks.

        ----

        ## [2602] DNA: Proximal Policy Optimization with a Dual Network Architecture

        **Authors**: *Matthew Aitchison, Penny Sweetser*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e95475f5fb8edb9075bf9e25670d4013-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e95475f5fb8edb9075bf9e25670d4013-Abstract-Conference.html)

        **Abstract**:

        This paper explores the problem of simultaneously learning a value function and policy in deep actor-critic reinforcement learning models. We find that the common practice of learning these functions jointly is sub-optimal due to an order-of-magnitude difference in noise levels between the two tasks. Instead, we show that learning these tasks independently, but with a constrained distillation phase, significantly improves performance. Furthermore, we find that policy gradient noise levels decrease when using a lower \textit{variance} return estimate. Whereas, value learning noise level decreases with a lower \textit{bias} estimate. Together these insights inform an extension to Proximal Policy Optimization we call \textit{Dual Network Architecture} (DNA), which significantly outperforms its predecessor. DNA also exceeds the performance of the popular Rainbow DQN algorithm on four of the five environments tested, even under more difficult stochastic control settings.

        ----

        ## [2603] Triangulation candidates for Bayesian optimization

        **Authors**: *Robert B. Gramacy, Annie Sauer, Nathan Wycoff*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e9750610639c3e7a849cff746bf60dbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e9750610639c3e7a849cff746bf60dbd-Abstract-Conference.html)

        **Abstract**:

        Bayesian optimization involves "inner optimization" over a new-data acquisition criterion which is non-convex/highly multi-modal, may be non-differentiable, or may otherwise thwart local numerical optimizers.  In such cases it is common to replace continuous search with a discrete one over random candidates.  Here we propose using candidates based on a Delaunay triangulation of the existing input design.  We detail the construction of these "tricands" and demonstrate empirically how they outperform both numerically optimized acquisitions and random candidate-based alternatives, and are well-suited for hybrid schemes, on benchmark synthetic and real simulation experiments.

        ----

        ## [2604] Masked Autoencoders As Spatiotemporal Learners

        **Authors**: *Christoph Feichtenhofer, Haoqi Fan, Yanghao Li, Kaiming He*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e97d1081481a4017df96b51be31001d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e97d1081481a4017df96b51be31001d3-Abstract-Conference.html)

        **Abstract**:

        This paper studies a conceptually simple extension of Masked Autoencoders (MAE) to spatiotemporal representation learning from videos. We randomly mask out spacetime patches in videos and learn an autoencoder to reconstruct them in pixels. Interestingly, we show that our MAE method can learn strong representations with almost no inductive bias on spacetime (only except for patch and positional embeddings), and spacetime-agnostic random masking performs the best. We observe that the optimal masking ratio is as high as 90% (vs. 75% on images), supporting the hypothesis that this ratio is related to information redundancy of the data. A high masking ratio leads to a large speedup, e.g., > 4x in wall-clock time or even more. We report competitive results on several challenging video datasets using vanilla Vision Transformers. We observe that MAE can outperform supervised pre-training by large margins. We further report encouraging results of training on real-world, uncurated Instagram data. Our study suggests that the general framework of masked autoencoding (BERT, MAE, etc.) can be a unified methodology for representation learning with minimal domain knowledge.

        ----

        ## [2605] PyramidCLIP: Hierarchical Feature Alignment for Vision-language Model Pretraining

        **Authors**: *Yuting Gao, Jinfeng Liu, Zihan Xu, Jun Zhang, Ke Li, Rongrong Ji, Chunhua Shen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e9882f7f7c44a10acc01132302bac9d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e9882f7f7c44a10acc01132302bac9d8-Abstract-Conference.html)

        **Abstract**:

        Large-scale vision-language pre-training has achieved promising results on downstream tasks. Existing methods highly rely on the assumption that the image-text pairs crawled from the Internet are in perfect one-to-one correspondence. However, in real scenarios, this assumption can be difficult to hold: the text description, obtained by crawling the affiliated metadata of the image, often suffers from the semantic mismatch and the mutual compatibility. To address these issues, we introduce PyramidCLIP, which constructs an input pyramid with different semantic levels for each modality, and aligns visual elements and linguistic elements in the form of hierarchy via peer-level semantics alignment and cross-level relation alignment. Furthermore, we soften the loss of negative samples (unpaired samples) so as to weaken the strict constraint during the pre-training stage, thus mitigating the risk of forcing the model to distinguish compatible negative pairs. Experiments on five downstream tasks demonstrate the effectiveness of the proposed PyramidCLIP. In particular, with the same amount of 15 million pre-training image-text pairs, PyramidCLIP exceeds CLIP on ImageNet zero-shot classification top-1 accuracy by 10.6%/13.2%/10.0% with ResNet50/ViT-B32/ViT-B16 based image encoder respectively. When scaling to larger datasets, PyramidCLIP achieves the state-of-the-art results on several downstream tasks. In particular, the results of PyramidCLIP-ResNet50 trained on 143M image-text pairs surpass that of CLIP using 400M data on ImageNet zero-shot classification task, significantly improving the data efficiency of CLIP.

        ----

        ## [2606] On the Parameterization and Initialization of Diagonal State Space Models

        **Authors**: *Albert Gu, Karan Goel, Ankit Gupta, Christopher Ré*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e9a32fade47b906de908431991440f7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e9a32fade47b906de908431991440f7c-Abstract-Conference.html)

        **Abstract**:

        State space models (SSM) have recently been shown to be very effective as a deep learning layer as a promising alternative to sequence models such as RNNs, CNNs, or Transformers.  The first version to show this potential was the S4 model, which is particularly effective on tasks involving long-range dependencies by using a prescribed state matrix called the HiPPO matrix.  While this has an interpretable mathematical mechanism for modeling long dependencies,  it also requires a custom representation and algorithm that makes the model difficult to understand and implement.  On the other hand, a recent variant of S4 called DSS showed that restricting the state matrix to be fully diagonal can still preserve the performance of the original model when using a specific initialization based on approximating S4's matrix.  This work seeks to systematically understand how to parameterize and initialize diagonal state space models.  While it follows from classical results that almost all SSMs have an equivalent diagonal form, we show that the initialization is critical for performance.  First, we explain why DSS works mathematically, as the diagonal approximation to S4 surprisingly recovers the same dynamics in the limit of infinite state dimension.  We then systematically describe various design choices in parameterizing and computing diagonal SSMs, and perform a controlled empirical study ablating the effects of these choices.  Our final model S4D is a simple diagonal version of S4 whose kernel computation requires just 3 lines of code and performs comparably to S4 in almost all settings, with state-of-the-art results in image, audio, and medical time-series domains, and 85\% average on the Long Range Arena benchmark.

        ----

        ## [2607] Implicit Regularization or Implicit Conditioning? Exact Risk Trajectories of SGD in High Dimensions

        **Authors**: *Courtney Paquette, Elliot Paquette, Ben Adlam, Jeffrey Pennington*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e9d89428e0ef0a70913845b3ae812ee0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e9d89428e0ef0a70913845b3ae812ee0-Abstract-Conference.html)

        **Abstract**:

        Stochastic gradient descent (SGD) is a pillar of modern machine learning, serving as the go-to optimization algorithm for a diverse array of problems. While the empirical success of SGD is often attributed to its computational efficiency and favorable generalization behavior, neither effect is well understood and disentangling them remains an open problem. Even in the simple setting of convex quadratic problems, worst-case analyses give an asymptotic convergence rate for SGD that is no better than full-batch gradient descent (GD), and the purported implicit regularization effects of SGD lack a precise explanation. In this work, we study the dynamics of multi-pass SGD on high-dimensional convex quadratics and establish an asymptotic equivalence to a stochastic differential equation, which we call homogenized stochastic gradient descent (HSGD), whose solutions we characterize explicitly in terms of a Volterra integral equation. These results yield precise formulas for the learning and risk trajectories, which reveal a mechanism of implicit conditioning that explains the efficiency of SGD relative to GD. We also prove that the noise from SGD negatively impacts generalization performance, ruling out the possibility of any type of implicit regularization in this context. Finally, we show how to adapt the HSGD formalism to include streaming SGD, which allows us to produce an exact prediction for the excess risk of multi-pass SGD relative to that of streaming SGD (bootstrap risk).

        ----

        ## [2608] Extrapolative Continuous-time Bayesian Neural Network for Fast Training-free Test-time Adaptation

        **Authors**: *Hengguan Huang, Xiangming Gu, Hao Wang, Chang Xiao, Hongfu Liu, Ye Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e9e1a0abc1a5b19a4aeb80dab19c82ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e9e1a0abc1a5b19a4aeb80dab19c82ae-Abstract-Conference.html)

        **Abstract**:

        Human intelligence has shown remarkably lower latency and higher precision than most AI systems when processing non-stationary streaming data in real-time. Numerous neuroscience studies suggest that such abilities may be driven by internal predictive modeling. In this paper, we explore the possibility of introducing such a mechanism in unsupervised domain adaptation (UDA) for handling non-stationary streaming data for real-time streaming applications. We propose to formulate internal predictive modeling as a continuous-time Bayesian filtering problem within a stochastic dynamical system context. Such a dynamical system describes the dynamics of model parameters of a UDA model evolving with non-stationary streaming data. Building on such a dynamical system, we then develop extrapolative continuous-time Bayesian neural networks (ECBNN), which generalize existing Bayesian neural networks to represent temporal dynamics and allow us to extrapolate the distribution of model parameters before observing the incoming data, therefore effectively reducing the latency. Remarkably, our empirical results show that ECBNN is capable of continuously generating better distributions of model parameters along the time axis given historical data only, thereby achieving (1) training-free test-time adaptation with low latency, (2) gradually improved alignment between the source and target features and (3) gradually improved model performance over time during the real-time testing stage.

        ----

        ## [2609] Global Convergence and Stability of Stochastic Gradient Descent

        **Authors**: *Vivak Patel, Shushu Zhang, Bowen Tian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea05e4fc0299c27648c9985266abad47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea05e4fc0299c27648c9985266abad47-Abstract-Conference.html)

        **Abstract**:

        In machine learning, stochastic gradient descent (SGD) is widely deployed to train models using highly non-convex objectives with equally complex noise models. Unfortunately, SGD theory often makes restrictive assumptions that fail to capture the non-convexity of real problems, and almost entirely ignore the complex noise models that exist in practice. In this work, we demonstrate the restrictiveness of these assumptions using three canonical models in machine learning. Then, we develop novel theory to address this shortcoming in two ways. First, we establish that SGD's iterates will either globally converge to a stationary point or diverge under nearly arbitrary nonconvexity and noise models. Under a slightly more restrictive assumption on the joint behavior of the non-convexity and noise model that generalizes current assumptions in the literature, we show that the objective function cannot diverge, even if the iterates diverge. As a consequence of our results, SGD can be applied to a greater range of stochastic optimization problems with confidence about its global convergence behavior and stability.

        ----

        ## [2610] Trap and Replace: Defending Backdoor Attacks by Trapping Them into an Easy-to-Replace Subnetwork

        **Authors**: *Haotao Wang, Junyuan Hong, Aston Zhang, Jiayu Zhou, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea06e6e9e80f1c3d382317fff67041ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea06e6e9e80f1c3d382317fff67041ac-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) are vulnerable to backdoor attacks. Previous works have shown it extremely challenging to unlearn the undesired backdoor behavior from the network, since the entire network can be affected by the backdoor samples. In this paper, we propose a brand-new backdoor defense strategy, which makes it much easier to remove the harmful influence of backdoor samples from the model. Our defense strategy, \emph{Trap and Replace}, consists of two stages. In the first stage, we bait and trap the backdoors in a small and easy-to-replace subnetwork. Specifically, we add an auxiliary image reconstruction head on top of the stem network shared with a light-weighted classification head. The intuition is that the auxiliary image reconstruction task encourages the stem network to keep sufficient low-level visual features that are hard to learn but semantically correct, instead of overfitting to the easy-to-learn but semantically incorrect backdoor correlations.  As a result, when trained on backdoored datasets, the backdoors are easily baited towards the unprotected classification head, since it is much more vulnerable than the shared stem, leaving the stem network hardly poisoned. In the second stage, we replace the poisoned light-weighted classification head with an untainted one, by re-training it from scratch only on a small holdout dataset with clean samples, while fixing the stem network. As a result, both the stem and the classification head in the final network are hardly affected by backdoor training samples. We evaluate our method against ten different backdoor attacks. Our method outperforms previous state-of-the-art methods by up to $20.57\%$, $9.80\%$, and $13.72\%$ attack success rate and on-average $3.14\%$, $1.80\%$, and $1.21\%$ clean classification accuracy on CIFAR10, GTSRB, and ImageNet-12, respectively. Code is available at https://github.com/VITA-Group/Trap-and-Replace-Backdoor-Defense.

        ----

        ## [2611] Reinforcement Learning with Logarithmic Regret and Policy Switches

        **Authors**: *Grigoris Velegkas, Zhuoran Yang, Amin Karbasi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea318cbc405c9803925e188e5d6836c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea318cbc405c9803925e188e5d6836c6-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the problem of regret minimization for episodic Reinforcement Learning (RL) both in the model-free and the model-based setting. We focus on learning with general function classes and general model classes, and we derive results that scale with the eluder dimension of these classes. In contrast to the existing body of work that mainly establishes instance-independent regret guarantees, we focus on the instance-dependent setting and show that the regret scales logarithmically with the horizon $T$, provided that there is a gap between the best and the second best action in every state. In addition, we show that such a logarithmic regret bound is realizable by algorithms with $O(\log T)$ switching cost (also known as adaptivity complexity). In other words, these algorithms rarely switch their policy during the course of their execution. Finally, we complement our results with lower bounds which show that even in the tabular setting, we cannot hope for regret guarantees lower than $O(\log T)$.

        ----

        ## [2612] Cluster and Aggregate: Face Recognition with Large Probe Set

        **Authors**: *Minchul Kim, Feng Liu, Anil K. Jain, Xiaoming Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea35a58ee3da13c01a69df2a819386b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea35a58ee3da13c01a69df2a819386b3-Abstract-Conference.html)

        **Abstract**:

        Feature fusion plays a crucial role in unconstrained face recognition where inputs (probes) comprise of a set of $N$ low quality images whose individual qualities vary. Advances in attention and recurrent modules have led to feature fusion that can model the relationship among the images in the input set. However, attention mechanisms cannot scale to large $N$ due to their quadratic complexity and recurrent modules suffer from input order sensitivity. We propose a two-stage feature fusion paradigm, Cluster and Aggregate, that can both scale to large $N$ and maintain the ability to perform sequential inference with order invariance. Specifically, Cluster stage is a linear assignment of $N$ inputs to $M$ global cluster centers, and Aggregation stage is a fusion over $M$ clustered features. The clustered features play an integral role when the inputs are sequential as they can serve as a summarization of past features. By leveraging the order-invariance of incremental averaging operation, we design an update rule that achieves batch-order invariance, which guarantees that the contributions of early image in the sequence do not diminish as time steps increase. Experiments on IJB-B and IJB-S benchmark datasets show the superiority of the proposed two-stage paradigm in unconstrained face recognition.

        ----

        ## [2613] GLIPv2: Unifying Localization and Vision-Language Understanding

        **Authors**: *Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, Jianfeng Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea370419760b421ce12e3082eb2ae1a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea370419760b421ce12e3082eb2ae1a8-Abstract-Conference.html)

        **Abstract**:

        We present GLIPv2, a grounded VL understanding model, that serves both localization tasks (e.g., object detection, instance segmentation) and Vision-Language (VL) understanding tasks (e.g., VQA, image captioning). GLIPv2 elegantly unifies localization pre-training and Vision-Language Pre-training (VLP) with three pre-training tasks: phrase grounding as a VL reformulation of the detection task, region-word contrastive learning as a novel region-word level contrastive learning task, and the masked language modeling. This unification not only simplifies the previous multi-stage VLP procedure but also achieves mutual benefits between localization and understanding tasks. Experimental results show that a single GLIPv2 model (all model weights are shared) achieves near SoTA performance on various localization and understanding tasks. The model also shows (1) strong zero-shot and few-shot adaption performance on open-vocabulary object detection tasks and (2) superior grounding capability on VL understanding tasks.

        ----

        ## [2614] Rethinking Alignment in Video Super-Resolution Transformers

        **Authors**: *Shuwei Shi, Jinjin Gu, Liangbin Xie, Xintao Wang, Yujiu Yang, Chao Dong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea4d65c59073e8faf79222654d25fbe2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea4d65c59073e8faf79222654d25fbe2-Abstract-Conference.html)

        **Abstract**:

        The alignment of adjacent frames is considered an essential operation in video super-resolution (VSR). Advanced VSR models, including the latest VSR Transformers, are generally equipped with well-designed alignment modules. However, the progress of the self-attention mechanism may violate this common sense. In this paper, we rethink the role of alignment in VSR Transformers and make several counter-intuitive observations. Our experiments show that: (i) VSR Transformers can directly utilize multi-frame information from unaligned videos, and (ii) existing alignment methods are sometimes harmful to VSR Transformers. These observations indicate that we can further improve the performance of VSR Transformers simply by removing the alignment module and adopting a larger attention window. Nevertheless, such designs will dramatically increase the computational burden, and cannot deal with large motions. Therefore, we propose a new and efficient alignment method called patch alignment, which aligns image patches instead of pixels. VSR Transformers equipped with patch alignment could demonstrate state-of-the-art performance on multiple benchmarks. Our work provides valuable insights on how multi-frame information is used in VSR and how to select alignment methods for different networks/datasets. Codes and models will be released at https://github.com/XPixelGroup/RethinkVSRAlignment.

        ----

        ## [2615] Robustness in deep learning: The good (width), the bad (depth), and the ugly (initialization)

        **Authors**: *Zhenyu Zhu, Fanghui Liu, Grigorios Chrysos, Volkan Cevher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea5a63f7ddb82e58623693fd1f4933f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea5a63f7ddb82e58623693fd1f4933f7-Abstract-Conference.html)

        **Abstract**:

        We study the average robustness notion in deep neural networks in (selected) wide and narrow, deep and shallow, as well as lazy and non-lazy training settings. We prove that in the under-parameterized setting, width has a negative effect while it improves robustness in the over-parameterized setting. The effect of depth closely depends on the initialization and the training mode. In particular, when initialized with LeCun initialization, depth helps robustness with the lazy training regime. In contrast, when initialized with Neural Tangent Kernel (NTK) and He-initialization, depth hurts the robustness. Moreover, under the non-lazy training regime, we demonstrate how the width of a two-layer ReLU network benefits robustness. Our theoretical developments improve the results by [Huang et al. NeurIPS21; Wu et al. NeurIPS21] and are consistent with [Bubeck and Sellke NeurIPS21; Bubeck et al. COLT21].

        ----

        ## [2616] Learning to Scaffold: Optimizing Model Explanations for Teaching

        **Authors**: *Patrick Fernandes, Marcos V. Treviso, Danish Pruthi, André Martins, Graham Neubig*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea64883d500d31738cd39eb49a748fa4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea64883d500d31738cd39eb49a748fa4-Abstract-Conference.html)

        **Abstract**:

        Modern machine learning models are opaque, and as a result there is a burgeoning academic subfield on methods that explain these models' behavior.  However, what is the precise goal of providing such explanations, and how can we demonstrate that explanations achieve this goal? Some research argues that explanations should help teach a student (either human or machine) to simulate the model being explained, and that the quality of explanations can be measured by the simulation accuracy of students on unexplained examples. In this work, leveraging meta-learning techniques, we extend this idea to improve the quality of the explanations themselves, specifically by optimizing explanations such that student models more effectively learn to simulate the original model. We train models on three natural language processing and computer vision tasks, and find that students trained with explanations extracted with our framework are able to simulate the teacher significantly more effectively than ones produced with previous methods. Through human annotations and a user study, we further find that these learned explanations more closely align with how humans would explain the required decisions in these tasks. Our code is available at https://github.com/coderpat/learning-scaffold.

        ----

        ## [2617] AutoLink: Self-supervised Learning of Human Skeletons and Object Outlines by Linking Keypoints

        **Authors**: *Xingzhe He, Bastian Wandt, Helge Rhodin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ea96e37a1caab5ca128ac3e15097ce38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ea96e37a1caab5ca128ac3e15097ce38-Abstract-Conference.html)

        **Abstract**:

        Structured representations such as keypoints are widely used in pose transfer, conditional image generation, animation, and 3D reconstruction. However, their supervised learning requires expensive annotation for each target domain. We propose a self-supervised method that learns to disentangle object structure from the appearance with a graph of 2D keypoints linked by straight edges. Both the keypoint location and their pairwise edge weights are learned, given only a collection of images depicting the same object class. The resulting graph is interpretable, for example, AutoLink recovers the human skeleton topology when applied to images showing people. Our key ingredients are i) an encoder that predicts keypoint locations in an input image, ii) a shared graph as a latent variable that links the same pairs of keypoints in every image, iii) an intermediate edge map that combines the latent graph edge weights and keypoint locations in a soft, differentiable manner, and iv) an inpainting objective on randomly masked images. Although simpler, AutoLink outperforms existing self-supervised methods on the established keypoint and pose estimation benchmarks and paves the way for structure-conditioned generative models on more diverse datasets.  Project website: https://xingzhehe.github.io/autolink/.

        ----

        ## [2618] Composite Feature Selection Using Deep Ensembles

        **Authors**: *Fergus Imrie, Alexander Norcliffe, Pietro Lió, Mihaela van der Schaar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eab69250e98b1f9fc54e473cc7a69439-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eab69250e98b1f9fc54e473cc7a69439-Abstract-Conference.html)

        **Abstract**:

        In many real world problems, features do not act alone but in combination with each other. For example, in genomics, diseases might not be caused by any single mutation but require the presence of multiple mutations. Prior work on feature selection either seeks to identify individual features or can only determine relevant groups from a predefined set. We investigate the problem of discovering groups of predictive features without predefined grouping. To do so, we define predictive groups in terms of linear and non-linear interactions between features. We introduce a novel deep learning architecture that uses an ensemble of feature selection models to find predictive groups, without requiring candidate groups to be provided. The selected groups are sparse and exhibit minimum overlap. Furthermore, we propose a new metric to measure similarity between discovered groups and the ground truth. We demonstrate the utility our model on multiple synthetic tasks and semi-synthetic chemistry datasets, where the ground truth structure is known, as well as an image dataset and a real-world cancer dataset.

        ----

        ## [2619] Scaling Multimodal Pre-Training via Cross-Modality Gradient Harmonization

        **Authors**: *Junru Wu, Yi Liang, Feng Han, Hassan Akbari, Zhangyang Wang, Cong Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eacad5b8e67850f2b8dd33d87691d097-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eacad5b8e67850f2b8dd33d87691d097-Abstract-Conference.html)

        **Abstract**:

        Self-supervised pre-training recently demonstrates success on large-scale multimodal data, and state-of-the-art contrastive learning methods often enforce the feature consistency from cross-modality inputs, such as video/audio or video/text pairs. Despite its convenience to formulate and leverage in practice, such cross-modality alignment (CMA) is only a weak and noisy supervision, since two modalities can be semantically misaligned even they are temporally aligned. For example, even in the (often adopted) instructional videos, a speaker can sometimes refer to something that is not visually present in the current frame; and the semantic misalignment would only be more unpredictable for the raw videos collected from unconstrained internet sources. We conjecture that might cause conflicts and biases among modalities, and may hence prohibit CMA from scaling up to training with larger and more heterogeneous data. This paper first verifies our conjecture by observing that, even in the latest VATT pre-training using only narrated videos, there exist strong gradient conflicts between different CMA losses within the same sample triplet (video, audio, text), indicating them as the noisy source of supervision. We then propose to harmonize such gradients during pre-training, via two techniques: (i) cross-modality gradient realignment: modifying different CMA loss gradients for one sample triplet, so that their gradient directions are in more agreement; and (ii) gradient-based curriculum learning: leveraging the gradient conflict information on an indicator of sample noisiness, to develop a curriculum learning strategy to prioritize training with less noisy sample triplets. Applying those gradient harmonization techniques to pre-training VATT on the HowTo100M dataset, we consistently improve its performance on different downstream tasks. Moreover, we are able to scale VATT pre-training to more complicated non-narrative Youtube8M dataset to further improve the state-of-the-arts.

        ----

        ## [2620] Bounded-Regret MPC via Perturbation Analysis: Prediction Error, Constraints, and Nonlinearity

        **Authors**: *Yiheng Lin, Yang Hu, Guannan Qu, Tongxin Li, Adam Wierman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eadeef7c51ad86989cc3b311cb49ec89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eadeef7c51ad86989cc3b311cb49ec89-Abstract-Conference.html)

        **Abstract**:

        We study Model Predictive Control (MPC) and propose a general analysis pipeline to bound its dynamic regret. The pipeline first requires deriving a perturbation bound for a finite-time optimal control problem. Then, the perturbation bound is used to bound the per-step error of MPC, which leads to a bound on the dynamic regret. Thus, our pipeline reduces the study of MPC to the well-studied problem of perturbation analysis, enabling the derivation of regret bounds of MPC under a variety of settings. To demonstrate the power of our pipeline, we use it to generalize existing regret bounds on MPC in linear time-varying (LTV) systems to incorporate prediction errors on costs, dynamics, and disturbances. Further, our pipeline leads to regret bounds on MPC in systems with nonlinear dynamics and constraints.

        ----

        ## [2621] AniFaceGAN: Animatable 3D-Aware Face Image Generation for Video Avatars

        **Authors**: *Yue Wu, Yu Deng, Jiaolong Yang, Fangyun Wei, Qifeng Chen, Xin Tong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eae78bf2712f222f101bd7d12f875a57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eae78bf2712f222f101bd7d12f875a57-Abstract-Conference.html)

        **Abstract**:

        Although 2D generative models have made great progress in face image generation and animation, they often suffer from undesirable artifacts such as 3D inconsistency when rendering images from different camera viewpoints. This prevents them from synthesizing video animations indistinguishable from real ones. Recently, 3D-aware GANs extend 2D GANs for explicit disentanglement of camera pose by leveraging 3D scene representations. These methods can well preserve the 3D consistency of the generated images across different views, yet they cannot achieve fine-grained control over other attributes, among which facial expression control is arguably the most useful and desirable for face animation. In this paper, we propose an animatable 3D-aware GAN for multiview consistent face animation generation. The key idea is to decompose the 3D representation of the 3D-aware GAN into a template field and a deformation field, where the former represents different identities with a canonical expression, and the latter characterizes expression variations of each identity. To achieve meaningful control over facial expressions via deformation, we propose a 3D-level imitative learning scheme between the generator and a parametric 3D face model during adversarial training of the 3D-aware GAN. This helps our method achieve high-quality animatable face image generation with strong visual 3D consistency, even though trained with only unstructured 2D images. Extensive experiments demonstrate our superior performance over prior works. Project page: \url{https://yuewuhkust.github.io/AniFaceGAN/

        ----

        ## [2622] Addressing Resource Scarcity across Sign Languages with Multilingual Pretraining and Unified-Vocabulary Datasets

        **Authors**: *Gokul NC, Manideep Ladi, Sumit Negi, Prem Selvaraj, Pratyush Kumar, Mitesh M. Khapra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb011fd258c763c44d8c6a0e9ce04f17-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb011fd258c763c44d8c6a0e9ce04f17-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        There are over 300 sign languages in the world, many of which have very limited or no labelled sign-to-text datasets. To address low-resource data scenarios, self-supervised pretraining and multilingual finetuning have been shown to be effective in natural language and speech processing. In this work, we apply these ideas to sign language recognition.We make three contributions.- First, we release SignCorpus, a large pretraining dataset on sign languages comprising about 4.6K hours of signing data across 10 sign languages. SignCorpus is curated from sign language videos on the internet, filtered for data quality, and converted into sequences of pose keypoints thereby removing all personal identifiable information (PII).- Second, we release Sign2Vec, a graph-based model with 5.2M parameters that is pretrained on SignCorpus. We envisage Sign2Vec as a multilingual large-scale pretrained model which can be fine-tuned for various sign recognition tasks across languages.- Third, we create MultiSign-ISLR -- a multilingual and label-aligned dataset of sequences of pose keypoints from 11 labelled datasets across 7 sign languages, and MultiSign-FS -- a new finger-spelling training and test set across 7 languages. On these datasets, we fine-tune Sign2Vec to create multilingual isolated sign recognition models. With experiments on multiple benchmarks, we show that pretraining and multilingual transfer are effective giving significant gains over state-of-the-art results.All datasets, models, and code has been made open-source via the OpenHands toolkit.

        ----

        ## [2623] Detection and Localization of Changes in Conditional Distributions

        **Authors**: *Lizhen Nie, Dan Nicolae*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html)

        **Abstract**:

        We study the change point problem that considers alterations in the conditional distribution of an inferential target on a set of covariates. This paired data scenario is in contrast to the standard setting where a sequentially observed variable is analyzed for potential changes in the marginal distribution. We propose new methodology for solving this problem, by starting from a simpler task that analyzes changes in conditional expectation, and generalizing the tools developed for that task to conditional distributions. Large sample properties of the proposed statistics are derived. In empirical studies, we illustrate the performance of the proposed method against baselines adapted from existing tools. Two real data applications are presented to demonstrate its potential.

        ----

        ## [2624] Refining Low-Resource Unsupervised Translation by Language Disentanglement of Multilingual Translation Model

        **Authors**: *Xuan-Phi Nguyen, Shafiq R. Joty, Kui Wu, Ai Ti Aw*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb1a323fa10d4102ff13422476a744ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb1a323fa10d4102ff13422476a744ff-Abstract-Conference.html)

        **Abstract**:

        Numerous recent work on unsupervised machine translation (UMT) implies that competent unsupervised translations of low-resource and unrelated languages, such as Nepali or Sinhala, are only possible if the model is trained in a massive multilingual environment, where these low-resource languages are mixed with high-resource counterparts. Nonetheless, while the high-resource languages greatly help kick-start the target low-resource translation tasks, the language discrepancy between them may hinder their further improvement. In this work, we propose a simple refinement procedure to separate languages from a pre-trained multilingual UMT model for it to focus on only the target low-resource task. Our method achieves the state of the art in the fully unsupervised translation tasks of English to Nepali, Sinhala, Gujarati, Latvian, Estonian and Kazakh, with BLEU score gains of 3.5, 3.5, 3.3, 4.1, 4.2, and 3.3, respectively. Our codebase is available at https://github.com/nxphi47/refineunsupmultilingual_mt

        ----

        ## [2625] A Mean-Field Game Approach to Cloud Resource Management with Function Approximation

        **Authors**: *Weichao Mao, Haoran Qiu, Chen Wang, Hubertus Franke, Zbigniew Kalbarczyk, Ravishankar K. Iyer, Tamer Basar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb3c8135137c8a60425a0320869ad87e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb3c8135137c8a60425a0320869ad87e-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning (RL) has gained increasing popularity for resource management in cloud services such as serverless computing. As self-interested users compete for shared resources in a cluster, the multi-tenancy nature of serverless platforms necessitates multi-agent reinforcement learning (MARL) solutions, which often suffer from severe scalability issues. In this paper, we propose a mean-field game (MFG) approach to cloud resource management that is scalable to a large number of users and applications and incorporates function approximation to deal with the large state-action spaces in real-world serverless platforms. Specifically, we present an online natural actor-critic algorithm for learning in MFGs compatible with various forms of function approximation. We theoretically establish its finite-time convergence to the regularized Nash equilibrium under linear function approximation and softmax parameterization. We further implement our algorithm using both linear and neural-network function approximations, and evaluate our solution on an open-source serverless platform, OpenWhisk, with real-world workloads from production traces. Experimental results demonstrate that our approach is scalable to a large number of users and significantly outperforms various baselines in terms of function latency and resource utilization efficiency.

        ----

        ## [2626] Regret Bounds for Risk-Sensitive Reinforcement Learning

        **Authors**: *Osbert Bastani, Yecheng Jason Ma, Estelle Shen, Wanqiao Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb4898d622e9a48b5f9713ea1fcff2bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb4898d622e9a48b5f9713ea1fcff2bf-Abstract-Conference.html)

        **Abstract**:

        In safety-critical applications of reinforcement learning such as healthcare and robotics, it is often desirable to optimize risk-sensitive objectives that account for tail outcomes rather than expected reward. We prove the first regret bounds for reinforcement learning under a general class of risk-sensitive objectives including the popular CVaR objective. Our theory is based on a novel characterization of the CVaR objective as well as a novel optimistic MDP construction.

        ----

        ## [2627] BILCO: An Efficient Algorithm for Joint Alignment of Time Series

        **Authors**: *Xuelong Mi, Mengfan Wang, Alex Chen, Jing-Xuan Lim, Yizhi Wang, Misha B. Ahrens, Guoqiang Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb5d9195b201ec7ba66c8e20b396d349-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb5d9195b201ec7ba66c8e20b396d349-Abstract-Conference.html)

        **Abstract**:

        Multiple time series data occur in many real applications and the alignment among them is usually a fundamental step of data analysis. Frequently, these multiple time series are inter-dependent, which provides extra information for the alignment task and this information cannot be well utilized in the conventional pairwise alignment methods. Recently, the joint alignment was modeled as a max-flow problem, in which both the profile similarity between the aligned time series and the distance between adjacent warping functions are jointly optimized. However, despite the new model having elegant mathematical formulation and superior alignment accuracy, the long computation time and large memory usage, due to the use of the existing general-purpose max-flow algorithms, limit significantly its well-deserved wide use. In this report, we present BIdirectional pushing with Linear Component Operations (BILCO), a novel algorithm that solves the joint alignment max-flow problems efficiently and exactly. We develop the strategy of linear component operations that integrates dynamic programming technique and the push-relabel approach. This strategy is motivated by the fact that the joint alignment max-flow problem is a generalization of dynamic time warping (DTW) and numerous individual DTW problems are embedded. Further, a bidirectional-pushing strategy is proposed to introduce prior knowledge and reduce unnecessary computation, by leveraging another fact that good initialization can be easily computed for the joint alignment max-flow problem. We demonstrate the efficiency of BILCO using both synthetic and real experiments. Tested on thousands of datasets under various simulated scenarios and in three distinct application categories, BILCO consistently achieves at least 10 and averagely 20-folds increase in speed, and uses at most 1/8 and averagely 1/10 memory compared with the best existing max-flow method. Our source code can be found at https://github.com/yu-lab-vt/BILCO.

        ----

        ## [2628] Giving Feedback on Interactive Student Programs with Meta-Exploration

        **Authors**: *Evan Zheran Liu, Moritz Stephan, Allen Nie, Chris Piech, Emma Brunskill, Chelsea Finn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb6ac5a41d753c35e1c575b350d4116f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb6ac5a41d753c35e1c575b350d4116f-Abstract-Conference.html)

        **Abstract**:

        Developing interactive software, such as websites or games, is a particularly engaging way to learn computer science. However, teaching and giving feedback on such software is time-consuming — standard approaches require instructors to manually grade student-implemented interactive programs. As a result, online platforms that serve millions, like Code.org, are unable to provide any feedback on assignments for implementing interactive programs, which critically hinders students’ ability to learn. One approach toward automatic grading is to learn an agent that interacts with a student’s program and explores states indicative of errors via reinforcement learning. However, existing work on this approach only provides binary feedback of whether a program is correct or not, while students require finer-grained feedback on the specific errors in their programs to understand their mistakes. In this work, we show that exploring to discover errors can be cast as a meta-exploration problem. This enables us to construct a principled objective for discovering errors and an algorithm for optimizing this objective, which provides fine-grained feedback. We evaluate our approach on a set of over 700K real anonymized student programs from a Code.org interactive assignment. Our approach provides feedback with 94.3% accuracy, improving over existing approaches by 17.7% and coming within 1.5% of human-level accuracy. Project web page: https://ezliu.github.io/dreamgrader.

        ----

        ## [2629] Left Heavy Tails and the Effectiveness of the Policy and Value Networks in DNN-based best-first search for Sokoban Planning

        **Authors**: *Dieqiao Feng, Carla P. Gomes, Bart Selman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb7295a8bc613b375726659c2ecd6f14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb7295a8bc613b375726659c2ecd6f14-Abstract-Conference.html)

        **Abstract**:

        Despite the success of practical solvers in various NP-complete domains such as SAT and CSP as well as using deep reinforcement learning to tackle two-player games such as Go, certain classes of PSPACE-hard planning problems have remained out of reach. Even carefully designed domain-specialized solvers can fail quickly due to the exponential search space on hard instances. Recent works that combine traditional search methods, such as best-first search and Monte Carlo tree search, with Deep Neural Networks' (DNN) heuristics have shown promising progress and can solve a significant number of hard planning instances beyond specialized solvers. To better understand why these approaches work, we studied the interplay of the policy and value networks of DNN-based best-first search on Sokoban and show the surprising effectiveness of the policy network, further enhanced by the value network, as a guiding heuristic for the search. To further understand the phenomena, we studied the cost distribution of the search algorithms and found that Sokoban instances can have heavy-tailed runtime distributions, with tails both on the left and right-hand sides. In particular, for the first time, we show the existence of \textit{left heavy tails} and propose an abstract tree model that can empirically explain the appearance of these tails. The experiments show the critical role of the policy network as a powerful heuristic guiding the search, which can lead to left heavy tails with polynomial scaling by avoiding exploring exponentially sized subtrees. Our results also demonstrate the importance of random restarts, as are widely used in traditional combinatorial solvers, for DNN-based search methods to avoid left and right heavy tails.

        ----

        ## [2630] Nonparametric Uncertainty Quantification for Single Deterministic Neural Network

        **Authors**: *Nikita Kotelevskii, Aleksandr Artemenkov, Kirill Fedyanin, Fedor Noskov, Alexander Fishkov, Artem Shelmanov, Artem Vazhentsev, Aleksandr Petiushko, Maxim Panov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb7389b039655fc5c53b11d4a6fa11bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb7389b039655fc5c53b11d4a6fa11bc-Abstract-Conference.html)

        **Abstract**:

        This paper proposes a fast and scalable method for uncertainty quantification of machine learning models' predictions. First, we show the principled way to measure the uncertainty of predictions for a classifier based on Nadaraya-Watson's nonparametric estimate of the conditional label distribution. Importantly, the approach allows to disentangle explicitly \textit{aleatoric} and \textit{epistemic} uncertainties. The resulting method works directly in the feature space. However, one can apply it to any neural network by considering an embedding of the data induced by the network. We demonstrate the strong performance of the method in uncertainty estimation tasks on text classification problems and a variety of real-world image datasets, such as MNIST, SVHN, CIFAR-100 and several versions of ImageNet.

        ----

        ## [2631] Decoupling Features in Hierarchical Propagation for Video Object Segmentation

        **Authors**: *Zongxin Yang, Yi Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eb890c36af87e4ca82e8ef7bcba6a284-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eb890c36af87e4ca82e8ef7bcba6a284-Abstract-Conference.html)

        **Abstract**:

        This paper focuses on developing a more effective method of hierarchical propagation for semi-supervised Video Object Segmentation (VOS). Based on vision transformers, the recently-developed Associating Objects with Transformers (AOT) approach introduces hierarchical propagation into VOS and has shown promising results. The hierarchical propagation can gradually propagate information from past frames to the current frame and transfer the current frame feature from object-agnostic to object-specific. However, the increase of object-specific information will inevitably lead to the loss of object-agnostic visual information in deep propagation layers. To solve such a problem and further facilitate the learning of visual embeddings, this paper proposes a Decoupling Features in Hierarchical Propagation (DeAOT) approach. Firstly, DeAOT decouples the hierarchical propagation of object-agnostic and object-specific embeddings by handling them in two independent branches. Secondly, to compensate for the additional computation from dual-branch propagation, we propose an efficient module for constructing hierarchical propagation, i.e., Gated Propagation Module, which is carefully designed with single-head attention. Extensive experiments show that DeAOT significantly outperforms AOT in both accuracy and efficiency. On YouTube-VOS, DeAOT can achieve 86.0% at 22.4fps and 82.0% at 53.4fps. Without test-time augmentations, we achieve new state-of-the-art performance on four benchmarks, i.e., YouTube-VOS (86.2%), DAVIS 2017 (86.2%), DAVIS 2016 (92.9%), and VOT 2020 (0.622 EAO).  Project page: https://github.com/z-x-yang/AOT.

        ----

        ## [2632] Computationally Efficient Horizon-Free Reinforcement Learning for Linear Mixture MDPs

        **Authors**: *Dongruo Zhou, Quanquan Gu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ebba182cb97864368fdb6ae00773a5e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ebba182cb97864368fdb6ae00773a5e4-Abstract-Conference.html)

        **Abstract**:

        Recent studies have shown that episodic reinforcement learning (RL) is not more difficult than bandits, even with a long planning horizon and unknown state transitions. However, these results are limited to either tabular Markov decision processes (MDPs) or computationally inefficient algorithms for linear mixture MDPs. In this paper, we propose the first computationally efficient horizon-free algorithm for linear mixture MDPs, which achieves the optimal $\tilde O(d\sqrt{K} +d^2)$ regret up to logarithmic factors. Our algorithm adapts a weighted least square estimator for the unknown transitional dynamic, where the weight is both \emph{variance-aware} and \emph{uncertainty-aware}. When applying our weighted least square estimator to heterogeneous linear bandits, we can obtain an $\tilde O(d\sqrt{\sum_{k=1}^K \sigma_k^2} +d)$ regret in the first $K$ rounds, where $d$ is the dimension of the context and $\sigma_k^2$ is the variance of the reward in the $k$-th round. This also improves upon the best known algorithms in this setting when $\sigma_k^2$'s are known.

        ----

        ## [2633] Counterfactual harm

        **Authors**: *Jonathan G. Richens, Rory Beard, Daniel H. Thompson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ebcf1bff7b2fe6dcc3fbe666faaa50f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ebcf1bff7b2fe6dcc3fbe666faaa50f1-Abstract-Conference.html)

        **Abstract**:

        To act safely and ethically in the real world, agents must be able to reason about harm and avoid harmful actions. However, to date there is no statistical method for measuring harm and factoring it into algorithmic decisions. In this paper we propose the first formal definition of harm and benefit using causal models. We show that any factual definition of harm is incapable of identifying harmful actions in certain scenarios, and show that standard machine learning algorithms that cannot perform counterfactual reasoning are guaranteed to pursue harmful policies following distributional shifts. We use our definition of harm to devise a framework for harm-averse decision making using counterfactual objective functions. We demonstrate this framework on the problem of identifying optimal drug doses using a dose-response model learned from randomised control trial data. We find that the standard method of selecting doses using treatment effects results in unnecessarily harmful doses, while our counterfactual approach identifies doses that are significantly less harmful without sacrificing efficacy.

        ----

        ## [2634] Chain of Thought Imitation with Procedure Cloning

        **Authors**: *Mengjiao Yang, Dale Schuurmans, Pieter Abbeel, Ofir Nachum*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ebdb990471f653dffb425eff03c7c980-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ebdb990471f653dffb425eff03c7c980-Abstract-Conference.html)

        **Abstract**:

        Imitation learning aims to extract high-performance policies from logged demonstrations of expert behavior. It is common to frame imitation learning as a supervised learning problem in which one fits a function approximator to the input-output mapping exhibited by the logged demonstrations (input observations to output actions). While the framing of imitation learning as a supervised input-output learning problem allows for applicability in a wide variety of settings, it is also an overly simplistic view of the problem in situations where the expert demonstrations provide much richer insight into expert behavior. For example, applications such as path navigation, robot manipulation, and strategy games acquire expert demonstrations via planning, search, or some other multi-step algorithm, revealing not just the output action to be imitated but also the procedure for how to determine this action. While these intermediate computations may use tools not available to the agent during inference (e.g., environment simulators), they are nevertheless informative as a way to explain an expert’s mapping of state to actions. To properly leverage expert procedure information without relying on the privileged tools the expert may have used to perform the procedure, we propose procedure cloning, which applies supervised sequence prediction to imitate the complete series of expert computations. This way, procedure cloning learns not only what to do (i.e., the output action), but how and why to do it (i.e., the procedure). Through empirical analysis on navigation, simulated robotic manipulation, and game-playing environments, we show that imitating the intermediate computations of an expert’s behavior enables procedure cloning to learn policies exhibiting significant generalization to unseen environment configurations, including those configurations for which running the expert’s procedure directly is infeasible.

        ----

        ## [2635] Revisiting Optimal Convergence Rate for Smooth and Non-convex Stochastic Decentralized Optimization

        **Authors**: *Kun Yuan, Xinmeng Huang, Yiming Chen, Xiaohan Zhang, Yingya Zhang, Pan Pan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec045a5ca2d8cfc528591b4c34296370-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec045a5ca2d8cfc528591b4c34296370-Abstract-Conference.html)

        **Abstract**:

        While numerous effective decentralized algorithms have been proposed with theoretical guarantees and empirical successes, the performance limits in decentralized optimization, especially the influence of network topology and its associated weight matrix on the optimal convergence rate, have not been fully understood. While Lu and Sa have recently provided an optimal rate for non-convex stochastic decentralized optimization using weight matrices associated with linear graphs, the optimal rate with general weight matrices remains unclear. This paper revisits non-convex stochastic decentralized optimization and establishes an optimal convergence rate with general weight matrices. In addition, we also establish the first optimal rate when non-convex loss functions further satisfy the Polyak-Lojasiewicz (PL) condition. Following existing lines of analysis in literature cannot achieve these results. Instead, we leverage the Ring-Lattice graph to admit general weight matrices while maintaining the optimal relation between the graph diameter and weight matrix connectivity. Lastly, we develop a new decentralized algorithm to attain the above two optimal rates up to logarithm factors.

        ----

        ## [2636] Training with More Confidence: Mitigating Injected and Natural Backdoors During Training

        **Authors**: *Zhenting Wang, Hailun Ding, Juan Zhai, Shiqing Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec0c9ca85b4ea49c7ebfb503cf55f2ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec0c9ca85b4ea49c7ebfb503cf55f2ae-Abstract-Conference.html)

        **Abstract**:

        The backdoor or Trojan attack is a severe threat to deep neural networks (DNNs). Researchers find that DNNs trained on benign data and settings can also learn backdoor behaviors, which is known as the natural backdoor. Existing works on anti-backdoor learning are based on weak observations that the backdoor and benign behaviors can differentiate during training. An adaptive attack with slow poisoning can bypass such defenses. Moreover, these methods cannot defend natural backdoors. We found the fundamental differences between backdoor-related neurons and benign neurons: backdoor-related neurons form a hyperplane as the classification surface across input domains of all affected labels. By further analyzing the training process and model architectures, we found that piece-wise linear functions cause this hyperplane surface. In this paper, we design a novel training method that forces the training to avoid generating such hyperplanes and thus remove the injected backdoors. Our extensive experiments on five datasets against five state-of-the-art attacks and also benign training show that our method can outperform existing state-of-the-art defenses. On average, the ASR (attack success rate) of the models trained with NONE is 54.83 times lower than undefended models under standard poisoning backdoor attack and 1.75 times lower under the natural backdoor attack. Our code is available at https://github.com/RU-System-Software-and-Security/NONE.

        ----

        ## [2637] Out-of-Distribution Detection via Conditional Kernel Independence Model

        **Authors**: *Yu Wang, Jingjing Zou, Jingyang Lin, Qing Ling, Yingwei Pan, Ting Yao, Tao Mei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec14daa5c50745f83fb27f685f8dfc22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec14daa5c50745f83fb27f685f8dfc22-Abstract-Conference.html)

        **Abstract**:

        Recently, various methods have been introduced to address the OOD detection problem with training outlier exposure. These methods usually count on discriminative softmax metric or energy method to screen OOD samples. In this paper, we probe an alternative hypothesis on OOD detection by constructing a novel latent variable model based on independent component analysis (ICA) techniques. This novel method named Conditional-i builds upon the probabilistic formulation, and applies the Hilbert-Schmidt Independence Criteria that offers a convenient solution for optimizing variable dependencies. Conditional-i exclusively encodes the useful class condition into the probabilistic model, which provides the desired convenience in delivering theoretical support for the OOD detection task. To facilitate the implementation of the Conditional-i model, we construct unique memory bank architectures that allow for convenient end-to-end training within a tractable budget. Empirical results demonstrate an evident performance boost on benchmarks against SOTA methods. We also provide valuable theoretical justifications that our training strategy is guaranteed to bound the error in the context of OOD detection. Code is available at: https://github.com/OODHSIC/conditional-i.

        ----

        ## [2638] Online Convex Optimization with Hard Constraints: Towards the Best of Two Worlds and Beyond

        **Authors**: *Hengquan Guo, Xin Liu, Honghao Wei, Lei Ying*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec360cb73d322e80a877b7ec7e13c79a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec360cb73d322e80a877b7ec7e13c79a-Abstract-Conference.html)

        **Abstract**:

        This paper considers online convex optimization with hard constraints and analyzes achievable regret and cumulative hard constraint violation (violation for short). The problem distinguishes itself from online convex optimization with soft constraints, where a violation at one round can be compensated/cancelled by a conservative decision at a different round. We propose a RECtified Online Optimization algorithm (RECOO) and consider two settings: fixed constraints and adversarial constraints. Both settings have been considered in the literature. Compared with existing results, {\em RECOO achieves the best of two worlds and beyond.}  For the fixed-constraints setting, RECOO achieves $O\left(\sqrt{T}\right)$ regret and $O(1)$  violation, where $T$ is the learning horizon. The best known results in this case are $O(\sqrt{T})$ regret and $O\left(T^{1/4}\right)$ violation. For the adversarial-constraints setting, it guarantees $O(\sqrt{T})$ regret and $O(T^{3/4})$ violation, which match the best existing results.  When the loss functions are strongly convex,  RECOO can guarantee $O(\log T)$ regret and $O(1)$ violation for fixed constraints, and $O(\log T)$ regret and $O(\sqrt{T\log T})$ violation for adversarial constraints. Both these results are order-wise better than the existing bounds. The regret and violation bounds mentioned above use the best fixed decision in hindsight as the baseline. This paper further considers a dynamic baseline where the comparator sequence is time-varying. This paper shows that RECOO not only improves the existing results in the fixed-constraints setting  but also {\em for the first time,} guarantees dynamic regret and violation bounds in the adversarial-constraints setting. Our experiment results confirm that RECOO outperforms several existing algorithms for both fixed and adversarial constraints.

        ----

        ## [2639] ResT V2: Simpler, Faster and Stronger

        **Authors**: *Qinglong Zhang, Yu-Bin Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec3acc7700fc5be9a8e257b38f870855-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec3acc7700fc5be9a8e257b38f870855-Abstract-Conference.html)

        **Abstract**:

        This paper proposes ResTv2, a simpler, faster, and stronger multi-scale vision Transformer for visual recognition. ResTv2 simplifies the EMSA structure in ResTv1 (i.e., eliminating the multi-head interaction part) and employs an upsample operation to reconstruct the lost medium- and high-frequency information caused by the downsampling operation. In addition, we explore different techniques for better applying ResTv2 backbones to downstream tasks. We find that although combining EMSAv2 and window attention can greatly reduce the theoretical matrix multiply FLOPs, it may significantly decrease the computation density, thus causing lower actual speed. We comprehensively validate ResTv2 on ImageNet classification, COCO detection, and ADE20K semantic segmentation. Experimental results show that the proposed ResTv2 can outperform the recently state-of-the-art backbones by a large margin, demonstrating the potential of ResTv2 as solid backbones. The code and models will be made publicly available at \url{https://github.com/wofmanaf/ResT}.

        ----

        ## [2640] Object-Category Aware Reinforcement Learning

        **Authors**: *Qi Yi, Rui Zhang, Shaohui Peng, Jiaming Guo, Xing Hu, Zidong Du, Xishan Zhang, Qi Guo, Yunji Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec3d49763c653ad7c8d587f52220c129-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec3d49763c653ad7c8d587f52220c129-Abstract-Conference.html)

        **Abstract**:

        Object-oriented reinforcement learning (OORL) is a promising way to improve the sample efficiency and generalization ability over standard RL.  Recent works that try to solve OORL tasks without additional feature engineering mainly focus on learning the object representations and then solving tasks via reasoning based on these object representations. However, none of these works tries to explicitly model the inherent similarity between different object instances of the same category.  Objects of the same category should share similar functionalities; therefore, the category is the most critical property of an object. Following this insight, we propose a novel framework named Object-Category Aware Reinforcement Learning (OCARL), which utilizes the category information of objects to facilitate both perception and reasoning. OCARL consists of three parts: (1) Category-Aware Unsupervised Object Discovery (UOD),  which discovers the objects as well as their corresponding categories; (2) Object-Category Aware Perception, which encodes the category information and is also robust to the incompleteness of (1) at the same time; (3) Object-Centric Modular Reasoning, which adopts multiple independent and object-category-specific networks when reasoning based on objects. Our experiments show that OCARL can improve both the sample efficiency and generalization in the OORL domain.

        ----

        ## [2641] Learning Partial Equivariances From Data

        **Authors**: *David W. Romero, Suhas Lohit*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec51d1fe4bbb754577da5e18eb54e6d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec51d1fe4bbb754577da5e18eb54e6d1-Abstract-Conference.html)

        **Abstract**:

        Group Convolutional Neural Networks (G-CNNs) constrain learned features to respect the symmetries in the selected group, and lead to better generalization when these symmetries appear in the data. If this is not the case, however, equivariance leads to overly constrained models and worse performance. Frequently, transformations occurring in data can be better represented by a subset of a group than by a group as a whole, e.g., rotations in $[-90^{\circ}, 90^{\circ}]$. In such cases, a model that respects equivariance partially is better suited to represent the data. In addition, relevant transformations may differ for low and high-level features. For instance, full rotation equivariance is useful to describe edge orientations in a face, but partial rotation equivariance is better suited to describe face poses relative to the camera. In other words, the optimal level of equivariance may differ per layer. In this work, we introduce Partial G-CNNs: G-CNNs able to learn layer-wise levels of partial and full equivariance to discrete, continuous groups and combinations thereof as part of training. Partial G-CNNs retain full equivariance when beneficial, e.g., for rotated MNIST, but adjust it whenever it becomes harmful, e.g., for classification of 6/9 digits or natural images. We empirically show that partial G-CNNs pair G-CNNs when full equivariance is advantageous, and outperform them otherwise. Our code is publicly available at www.github.com/merlresearch/partial_gcnn .

        ----

        ## [2642] Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

        **Authors**: *Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, Mohammad Norouzi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html)

        **Abstract**:

        We present Imagen, a text-to-image diffusion model with an unprecedented degree of photorealism and a deep level of language understanding. Imagen builds on the power of large transformer language models in understanding text and hinges on the strength of diffusion models in high-fidelity image generation. Our key discovery is that generic large language models (e.g., T5), pretrained on text-only corpora, are surprisingly effective at encoding text for image synthesis: increasing the size of the language model in Imagen boosts both sample fidelity and image-text alignment much more than increasing the size of the image diffusion model. Imagen achieves a new state-of-the-art FID score of 7.27 on the COCO dataset, without ever training on COCO, and human raters find Imagen samples to be on par with the COCO data itself in image-text alignment. To assess text-to-image models in greater depth, we introduce DrawBench, a comprehensive and challenging benchmark for text-to-image models. With DrawBench, we compare Imagen with recent methods including VQ-GAN+CLIP, Latent Diffusion Models, and DALL-E 2, and find that human raters prefer Imagen over other models in side-by-side comparisons, both in terms of sample quality and image-text alignment.

        ----

        ## [2643] A Simple Decentralized Cross-Entropy Method

        **Authors**: *Zichen Zhang, Jun Jin, Martin Jägersand, Jun Luo, Dale Schuurmans*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ec9dc93250548578aa4569aa19acfd81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ec9dc93250548578aa4569aa19acfd81-Abstract-Conference.html)

        **Abstract**:

        Cross-Entropy Method (CEM) is commonly used for planning in model-based reinforcement learning (MBRL) where a centralized approach is typically utilized to update the sampling distribution based on only the top-$k$ operation's results on samples. In this paper, we show that such a centralized approach makes CEM vulnerable to local optima, thus impairing its sample efficiency. To tackle this issue, we propose Decentralized CEM (DecentCEM), a simple but effective improvement over classical CEM, by using an ensemble of CEM instances running independently from one another, and each performing a local improvement of its own sampling distribution. We provide both theoretical and empirical analysis to demonstrate the effectiveness of this simple decentralized approach. We empirically show that, compared to the classical centralized approach using either a single or even a mixture of Gaussian distributions, our DecentCEM finds the global optimum much more consistently thus improves the sample efficiency. Furthermore, we plug in our DecentCEM in the planning problem of MBRL, and evaluate our approach in several continuous control environments, with comparison to the state-of-art CEM based MBRL approaches (PETS and POPLIN). Results show sample efficiency improvement by simply replacing the classical CEM module with our DecentCEM module, while only sacrificing a reasonable amount of computational cost. Lastly, we conduct ablation studies for more in-depth analysis. Code is available at https://github.com/vincentzhang/decentCEM.

        ----

        ## [2644] MSDS: A Large-Scale Chinese Signature and Token Digit String Dataset for Handwriting Verification

        **Authors**: *Peirong Zhang, Jiajia Jiang, Yuliang Liu, Lianwen Jin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eca896a8cf6363c9573a701c8c5c9cc5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/eca896a8cf6363c9573a701c8c5c9cc5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Although online handwriting verification has made great progress recently, the verification performances are still far behind the real usage owing to the small scale of the datasets as well as the limited biometric mediums. Therefore, this paper proposes a new handwriting verification benchmark dataset named Multimodal Signature and Digit String (MSDS), which consists of two subsets: MSDS-ChS (Chinese Signatures) and MSDS-TDS (Token Digit Strings), contributed by 402 users, with 20 genuine samples and 20 skilled forgeries per user per subset. MSDS-ChS consists of handwritten Chinese signatures, which, to the best of our knowledge, is the largest publicly available Chinese signature dataset for handwriting verification, at least eight times larger than existing online datasets. Meanwhile, MSDS-TDS consists of handwritten Token Digit Strings, i.e, the actual phone numbers of users, which have not been explored yet. Extensive experiments with different baselines are respectively conducted for MSDS-ChS and MSDS-TDS. Surprisingly, verification performances of state-of-the-art methods on MSDS-TDS are generally better than those on MSDS-ChS, which indicates that the handwritten Token Digit String could be a more effective biometric than handwritten Chinese signature. This is a promising discovery that could inspire us to explore new biometric traits. The MSDS dataset is available at https://github.com/HCIILAB/MSDS.

        ----

        ## [2645] Do Residual Neural Networks discretize Neural Ordinary Differential Equations?

        **Authors**: *Michael E. Sander, Pierre Ablin, Gabriel Peyré*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ecc38927fe5148c66bee64ee8fed1e76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ecc38927fe5148c66bee64ee8fed1e76-Abstract-Conference.html)

        **Abstract**:

        Neural Ordinary Differential Equations (Neural ODEs) are the continuous analog of Residual Neural Networks (ResNets). We investigate whether the discrete dynamics defined by a ResNet are close to the continuous one of a Neural ODE. We first quantify the distance between the ResNet's hidden state trajectory and the solution of its corresponding Neural ODE. Our bound is tight and, on the negative side, does not go to $0$ with depth $N$ if the residual functions are not smooth with depth. On the positive side, we show that this smoothness is preserved by gradient descent for a ResNet with linear residual functions and small enough initial loss. It ensures an implicit regularization towards a limit Neural ODE at rate $\frac1N$, uniformly with depth and optimization time. As a byproduct of our analysis, we consider the use of a memory-free discrete adjoint method to train a ResNet by recovering the activations on the fly through a backward pass of the network, and show that this method theoretically succeeds at large depth if the residual functions are Lipschitz with the input. We then show that Heun's method, a second order ODE integration scheme, allows for better gradient estimation with the adjoint method when the residual functions are smooth with depth. We experimentally validate that our adjoint method succeeds at large depth, and that Heun’s method needs fewer layers to succeed. We finally use the adjoint method successfully for fine-tuning very deep ResNets without memory consumption in the residual layers.

        ----

        ## [2646] Diffusion-based Molecule Generation with Informative Prior Bridges

        **Authors**: *Lemeng Wu, Chengyue Gong, Xingchao Liu, Mao Ye, Qiang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eccc6e11878857e87ec7dd109eaa9eeb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eccc6e11878857e87ec7dd109eaa9eeb-Abstract-Conference.html)

        **Abstract**:

        AI-based molecule generation provides a promising approach to a large area of biomedical sciences and engineering, such as antibody design, hydrolase engineering, or vaccine development. Because the molecules are governed by physical laws, a key challenge is to incorporate prior information into the training procedure to generate high-quality and realistic molecules. We propose a simple and novel approach to steer the training of diffusion-based generative models with physical and statistics prior information. This is achieved by constructing physically informed diffusion bridges, stochastic processes that guarantee to yield a given observation at the fixed terminal time. We develop a Lyapunov function based method to construct and determine bridges, and propose a number of proposals of informative prior bridges for both high-quality molecule generation and uniformity-promoted 3D point cloud generation. With comprehensive experiments, we show that our method provides a powerful approach to the 3D generation task, yielding molecule structures with better quality and stability scores and more uniformly distributed point clouds of high qualities.

        ----

        ## [2647] Generative multitask learning mitigates target-causing confounding

        **Authors**: *Taro Makino, Krzysztof J. Geras, Kyunghyun Cho*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ece182f93af26c64187ba3f7dfd4309a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ece182f93af26c64187ba3f7dfd4309a-Abstract-Conference.html)

        **Abstract**:

        We propose generative multitask learning (GMTL), a simple and scalable approach to causal machine learning in the multitask setting. Our approach makes a minor change to the conventional multitask inference objective, and improves robustness to target shift. Since GMTL only modifies the inference objective, it can be used with existing multitask learning methods without requiring additional training. The improvement in robustness comes from mitigating unobserved confounders that cause the targets, but not the input. We refer to them as \emph{target-causing confounders}. These confounders induce spurious dependencies between the input and targets. This poses a problem for conventional multitask learning, due to its assumption that the targets are conditionally independent given the input. GMTL mitigates target-causing confounding at inference time, by removing the influence of the joint target distribution, and predicting all targets jointly. This removes the spurious dependencies between the input and targets, where the degree of removal is adjustable via a single hyperparameter. This flexibility is useful for managing the trade-off between in- and out-of-distribution generalization. Our results on the Attributes of People and Taskonomy datasets reflect an improved robustness to target shift across four multitask learning methods.

        ----

        ## [2648] Revisit last-iterate convergence of mSGD under milder requirement on step size

        **Authors**: *Ruinan Jin, Xingkang He, Lang Chen, Difei Cheng, Vijay Gupta*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eceb7374fb94b4efd0fe4bea550d4285-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eceb7374fb94b4efd0fe4bea550d4285-Abstract-Conference.html)

        **Abstract**:

        Understanding convergence of SGD-based optimization algorithms can help deal with enormous machine learning problems. To ensure last-iterate convergence of   SGD and momentum-based SGD (mSGD),  the existing studies  usually constrain the step size $\epsilon_{n}$ to decay as $\sum_{n=1}^{+\infty}\epsilon_{n}^{2}<+\infty$, which however is rather conservative and may lead to slow convergence in the early stage of the iteration. In this paper, we relax this requirement by studying an alternate step size for the mSGD. First, we relax the requirement of the decay on step size to $\sum_{n=1}^{+\infty}\epsilon_{n}^{2+\eta_{0}}<+\infty\ (0\le\eta_{0}<1/2)$. This implies that a larger step size, such as $\epsilon_{n}=\frac{1}{\sqrt{n}}$  can  be utilized for accelerating the mSGD in the early stage.  Under this new step size and some common conditions, we prove that the  gradient norm of mSGD for non-convex loss functions   asymptotically decays to zero. In addition, we show that this step size can indeed help make the  convergence into a neighborhood of the stationary points quicker in the early stage. In addition, we establish the convergence of   mSGD  under a constant step size $\epsilon_n\equiv\epsilon>0$ by removing the common requirement in the literature on the strong convexity of the loss function.   Some experiments are given to illustrate the developed results.

        ----

        ## [2649] Are You Stealing My Model? Sample Correlation for Fingerprinting Deep Neural Networks

        **Authors**: *Jiyang Guan, Jian Liang, Ran He*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ed189de2611f200bd4c2ab30c576e99e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ed189de2611f200bd4c2ab30c576e99e-Abstract-Conference.html)

        **Abstract**:

        An off-the-shelf model as a commercial service could be stolen by model stealing attacks, posing great threats to the rights of the model owner. Model fingerprinting aims to verify whether a suspect model is stolen from the victim model, which gains more and more attention nowadays. Previous methods always leverage the transferable adversarial examples as the model fingerprint, which is sensitive to adversarial defense or transfer learning scenarios. To address this issue, we consider the pairwise relationship between samples instead and propose a novel yet simple model stealing detection method based on SAmple Correlation (SAC). Specifically, we present SAC-w that selects wrongly classified normal samples as model inputs and calculates the mean correlation among their model outputs. To reduce the training time, we further develop SAC-m that selects CutMix Augmented samples as model inputs, without the need for training the surrogate models or generating adversarial examples. Extensive results validate that SAC successfully defends against various model stealing attacks, even including adversarial training or transfer learning, and detects the stolen models with the best performance in terms of AUC across different datasets and model architectures. The codes are available at https://github.com/guanjiyang/SAC.

        ----

        ## [2650] SecureFedYJ: a safe feature Gaussianization protocol for Federated Learning

        **Authors**: *Tanguy Marchand, Boris Muzellec, Constance Beguier, Jean Ogier du Terrail, Mathieu Andreux*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ed3c686f9cda57e56cc859402c775414-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ed3c686f9cda57e56cc859402c775414-Abstract-Conference.html)

        **Abstract**:

        The Yeo-Johnson (YJ) transformation is a standard parametrized per-feature unidimensional transformation often used to Gaussianize features in machine learning. In this paper, we investigate the problem of applying the YJ transformation in a cross-silo Federated Learning setting under privacy constraints. For the first time, we prove that the YJ negative log-likelihood is in fact convex, which allows us to optimize it with exponential search. We numerically show that the resulting algorithm is more stable than the state-of-the-art approach based on the Brent minimization method. Building on this simple algorithm and Secure Multiparty Computation routines, we propose SECUREFEDYJ, a federated algorithm that performs a pooled-equivalent YJ transformation without leaking more information than the final fitted parameters do. Quantitative experiments on real data demonstrate that, in addition to being secure, our approach reliably normalizes features across silos as well as if data were pooled, making it a viable approach for safe federated feature Gaussianization.

        ----

        ## [2651] When to Trust Your Simulator: Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning

        **Authors**: *Haoyi Niu, Shubham Sharma, Yiwen Qiu, Ming Li, Guyue Zhou, Jianming Hu, Xianyuan Zhan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ed3cd2520148b577039adfade82a5566-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ed3cd2520148b577039adfade82a5566-Abstract-Conference.html)

        **Abstract**:

        Learning effective reinforcement learning (RL) policies to solve real-world complex tasks can be quite challenging without a high-fidelity simulation environment. In most cases, we are only given imperfect simulators with simplified dynamics, which inevitably lead to severe sim-to-real gaps in RL policy learning. The recently emerged field of offline RL provides another possibility to learn policies directly from pre-collected historical data. However, to achieve reasonable performance, existing offline RL algorithms need impractically large offline data with sufficient state-action space coverage for training. This brings up a new question: is it possible to combine learning from limited real data in offline RL and unrestricted exploration through imperfect simulators in online RL to address the drawbacks of both approaches? In this study, we propose the Dynamics-Aware Hybrid Offline-and-Online Reinforcement Learning (H2O) framework to provide an affirmative answer to this question. H2O introduces a dynamics-aware policy evaluation scheme, which adaptively penalizes the Q function learning on simulated state-action pairs with large dynamics gaps, while also simultaneously allowing learning from a fixed real-world dataset. Through extensive simulation and real-world tasks, as well as theoretical analysis, we demonstrate the superior performance of H2O against other cross-domain online and offline RL algorithms. H2O provides a brand new hybrid offline-and-online RL paradigm, which can potentially shed light on future RL algorithm design for solving practical real-world tasks.

        ----

        ## [2652] Data-Efficient Structured Pruning via Submodular Optimization

        **Authors**: *Marwa El Halabi, Suraj Srinivas, Simon Lacoste-Julien*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ed5854c456e136afa3faa5e41b1f3509-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ed5854c456e136afa3faa5e41b1f3509-Abstract-Conference.html)

        **Abstract**:

        Structured pruning is an effective approach for compressing large pre-trained neural networks without significantly affecting their performance. However, most current structured pruning methods do not provide any performance guarantees, and often require fine-tuning, which makes them inapplicable in the limited-data regime. We propose a principled data-efficient structured pruning method based on submodular optimization. In particular, for a given layer, we select neurons/channels to prune and corresponding new weights for the next layer, that minimize the change in the next layer's input induced by pruning. We show that this selection problem is a weakly submodular maximization problem, thus it can be provably approximated using an efficient greedy algorithm. Our method is guaranteed to have an exponentially decreasing error between the original model and the pruned model outputs w.r.t the pruned size, under reasonable assumptions. It is also one of the few methods in the literature that uses only a limited-number of training data and no labels. Our experimental results demonstrate that our method outperforms state-of-the-art methods in the limited-data regime.

        ----

        ## [2653] Interpolation and Regularization for Causal Learning

        **Authors**: *Leena Chennuru Vankadara, Luca Rendsburg, Ulrike von Luxburg, Debarghya Ghoshdastidar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ed7b8e1312f6ba8af6e4316dcd28bb3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ed7b8e1312f6ba8af6e4316dcd28bb3d-Abstract-Conference.html)

        **Abstract**:

        Recent work shows that in complex model classes, interpolators can achieve statistical generalization and even be optimal for statistical learning. However, despite increasing interest in learning models with good causal properties, there is no understanding of whether such interpolators can also achieve causal generalization. To address this gap, we study causal learning from observational data through the lens of interpolation and its counterpart---regularization. Under a simple linear causal model, we derive precise asymptotics for the causal risk of the min-norm interpolator and ridge regressors in the high-dimensional regime. We find a large range of behavior that can be precisely characterized by a new measure of confounding strength. When confounding strength is positive, which holds under independent causal mechanisms---a standard assumption in causal learning---we find that interpolators cannot be optimal. Indeed, causal learning requires stronger regularization than statistical learning. Beyond this assumption, when confounding is negative, we observe a phenomenon of self-induced regularization due to positive alignment between statistical and causal signals. Here, causal learning requires weaker regularization than statistical learning, interpolators can be optimal, and optimal regularization can even be negative.

        ----

        ## [2654] A Direct Approximation of AIXI Using Logical State Abstractions

        **Authors**: *Samuel Yang-Zhao, Tianyu Wang, Kee Siong Ng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ed91353f700d113e5d848c7e04a858b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ed91353f700d113e5d848c7e04a858b0-Abstract-Conference.html)

        **Abstract**:

        We propose a practical integration of logical state abstraction with AIXI, a Bayesian optimality notion for reinforcement learning agents, to significantly expand the model class that AIXI agents can be approximated over to complex history-dependent and structured environments. The state representation and reasoning framework is based on higher-order logic, which can be used to define and enumerate complex features on non-Markovian and structured environments. We address the problem of selecting the right subset of features to form state abstractions by adapting the $\Phi$-MDP optimisation criterion from state abstraction theory. Exact Bayesian model learning is then achieved using a suitable generalisation of Context Tree Weighting over abstract state sequences. The resultant architecture can be integrated with different planning algorithms. Experimental results on controlling epidemics on large-scale contact networks validates the agent's performance.

        ----

        ## [2655] Learning Representations via a Robust Behavioral Metric for Deep Reinforcement Learning

        **Authors**: *Jianda Chen, Sinno Jialin Pan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eda9523faa5e7191aee1c2eaff669716-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eda9523faa5e7191aee1c2eaff669716-Abstract-Conference.html)

        **Abstract**:

        Learning an informative representation with behavioral metrics is able to accelerate the deep reinforcement learning process. There are two key research issues on behavioral metric-based representation learning: 1) how to relax the computation of a specific behavioral metric, which is difficult or even intractable to compute, and 2) how to approximate the relaxed metric by learning an embedding space for states. In this paper, we analyze the potential relaxation and/or approximation gaps for existing behavioral metric-based representation learning methods. Based on the analysis, we propose a new behavioral distance, the RAP distance, and develop a practical representation learning algorithm on top of it with a theoretical analysis. We conduct extensive experiments on DeepMind Control Suite with distraction, Robosuite, and autonomous driving simulator CARLA to demonstrate new state-of-the-art results.

        ----

        ## [2656] Finding Second-Order Stationary Points in Nonconvex-Strongly-Concave Minimax Optimization

        **Authors**: *Luo Luo, Yujun Li, Cheng Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/edc79627bd67ccf943bb1d47037922d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/edc79627bd67ccf943bb1d47037922d1-Abstract-Conference.html)

        **Abstract**:

        We study the smooth minimax optimization problem $\min_{\bf x}\max_{\bf y} f({\bf x},{\bf y})$, where $f$ is $\ell$-smooth, strongly-concave in ${\bf y}$ but possibly nonconvex in ${\bf x}$. Most of existing works focus on finding the first-order stationary point of the function $f({\bf x},{\bf y})$ or its primal function $P({\bf x})\triangleq \max_{\bf y} f({\bf x},{\bf y})$, but few of them focus on achieving the second-order stationary point, which is essential to nonconvex problems. In this paper, we propose a novel approach for minimax optimization, called Minimax Cubic Newton (MCN), which could find an ${\mathcal O}\left(\varepsilon,\kappa^{1.5}\sqrt{\rho\varepsilon}\right)$-second-order stationary point of $P({\bf x})$ with calling ${\mathcal O}\left(\kappa^{1.5}\sqrt{\rho}\varepsilon^{-1.5}\right)$ times of second-order oracles and $\tilde{\mathcal O}\left(\kappa^{2}\sqrt{\rho}\varepsilon^{-1.5}\right)$ times of first-order oracles, where $\kappa$ is the condition number and $\rho$ is the Lipschitz continuous constant for the Hessian of $f({\bf x},{\bf y})$. In addition, we propose an inexact variant of MCN for high-dimensional problems to avoid calling the expensive second-order oracles. Instead, our method solves the cubic sub-problem inexactly via gradient descent and matrix Chebyshev expansion. This strategy still obtains the desired approximate second-order stationary point with high probability but only requires $\tilde{\mathcal O}\left(\kappa^{1.5}\ell\varepsilon^{-2}\right)$ Hessian-vector oracle calls and $\tilde{\mathcal O}\left(\kappa^{2}\sqrt{\rho}\varepsilon^{-1.5}\right)$ first-order oracle calls. To the best of our knowledge, this is the first work that considers the non-asymptotic convergence behavior of finding second-order stationary points for minimax problems without the convex-concave assumptions.

        ----

        ## [2657] Contextual Squeeze-and-Excitation for Efficient Few-Shot Image Classification

        **Authors**: *Massimiliano Patacchiola, John Bronskill, Aliaksandra Shysheya, Katja Hofmann, Sebastian Nowozin, Richard E. Turner*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ee1e549d6fb7c58ed06557bfc264335c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ee1e549d6fb7c58ed06557bfc264335c-Abstract-Conference.html)

        **Abstract**:

        Recent years have seen a growth in user-centric applications that require effective knowledge transfer across tasks in the low-data regime. An example is personalization, where a pretrained system is adapted by learning on small amounts of labeled data belonging to a specific user. This setting requires high accuracy under low computational complexity, therefore the Pareto frontier of accuracy vs. adaptation cost plays a crucial role. In this paper we push this Pareto frontier in the few-shot image classification setting with a key contribution: a new adaptive block called Contextual Squeeze-and-Excitation (CaSE) that adjusts a pretrained neural network on a new task to significantly improve performance with a single forward pass of the user data (context). We use meta-trained CaSE blocks to conditionally adapt the body of a network and a fine-tuning routine to adapt a linear head, defining a method called UpperCaSE. UpperCaSE achieves a new state-of-the-art accuracy relative to meta-learners on the 26 datasets of VTAB+MD and on a challenging real-world personalization benchmark (ORBIT), narrowing the gap with leading fine-tuning methods with the benefit of orders of magnitude lower adaptation cost.

        ----

        ## [2658] Fuzzy Learning Machine

        **Authors**: *Junbiao Cui, Jiye Liang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ee26c68c6d62b7d8333815264aa28577-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ee26c68c6d62b7d8333815264aa28577-Abstract-Conference.html)

        **Abstract**:

        Classification is one of the most important problems in machine learning and the nature of it is concept cognition. So far, dozens of different classifiers have been designed. Although their working mechanisms vary widely, few of them fully consider concept cognition. In this paper, a new learning machine, fuzzy learning machine (FLM), is proposed from the perspective of concept cognition. Inspired by cognitive science, its working mechanism is of strong interpretability. At the same time, FLM roots in set theory and fuzzy set theory, so FLM has a solid mathematical foundation. The systematic experimental results on a large number of data sets show that FLM can achieve excellent performance, even with the simple implementation.

        ----

        ## [2659] Learning Structure from the Ground up - Hierarchical Representation Learning by Chunking

        **Authors**: *Shuchen Wu, Noémi Élteto, Ishita Dasgupta, Eric Schulz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ee5bb72130c332c3d4bf8d231e617506-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ee5bb72130c332c3d4bf8d231e617506-Abstract-Conference.html)

        **Abstract**:

        From learning to play the piano to speaking a new language, reusing and recombining previously acquired representations enables us to master complex skills and easily adapt to new environments. Inspired by the Gestalt principle of \textit{grouping by proximity} and theories of chunking in cognitive science, we propose a hierarchical chunking model (HCM). HCM learns representations from non-i.i.d. sequential data from the ground up by first discovering the minimal atomic sequential units as chunks. As learning progresses, a hierarchy of chunk representations is acquired by chunking previously learned representations into more complex representations guided by sequential dependence. We provide learning guarantees on an idealized version of HCM, and demonstrate that HCM learns meaningful and interpretable representations in a human-like fashion. Our model can be extended to learn visual, temporal, and visual-temporal chunks. The interpretability of the learned chunks can be used to assess transfer or interference when the environment changes. Finally, in an fMRI dataset, we demonstrate that HCM learns interpretable chunks of functional coactivation regions and hierarchical modular and sub-modular structures confirmed by the neuroscientific literature. Taken together, our results show how cognitive science in general and theories of chunking in particular can inform novel and more interpretable approaches to representation learning.

        ----

        ## [2660] AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation

        **Authors**: *Yuanfeng Ji, Haotian Bai, Chongjian Ge, Jie Yang, Ye Zhu, Ruimao Zhang, Zhen Li, Lingyan Zhang, Wanling Ma, Xiang Wan, Ping Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ee604e1bedbd069d9fc9328b7b9584be-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/ee604e1bedbd069d9fc9328b7b9584be-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Despite the considerable progress in automatic abdominal multi-organ segmentation from CT/MRI scans in recent years, a comprehensive evaluation of the models' capabilities is hampered by the lack of a large-scale benchmark from diverse clinical scenarios. Constraint by the high cost of collecting and labeling 3D medical data, most of the deep learning models to date are driven by datasets with a limited number of organs of interest or samples, which still limits the power of modern deep models and makes it difficult to provide a fully comprehensive and fair estimate of various methods. To mitigate the limitations, we present AMOS, a large-scale, diverse, clinical dataset for abdominal organ segmentation. AMOS provides 500 CT and 100 MRI scans collected from multi-center, multi-vendor, multi-modality, multi-phase, multi-disease patients, each with voxel-level annotations of 15 abdominal organs, providing challenging examples and test-bed for studying robust segmentation algorithms under diverse targets and scenarios. We further benchmark several state-of-the-art medical segmentation models to evaluate the status of the existing methods on this new challenging dataset. We have made our datasets, benchmark servers, and baselines publicly available, and hope to inspire future research. Information can be found at https://amos22.grand-challenge.org.

        ----

        ## [2661] The price of ignorance: how much does it cost to forget noise structure in low-rank matrix estimation?

        **Authors**: *Jean Barbier, TianQi Hou, Marco Mondelli, Manuel Sáenz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ee74a6ade401e200985e2421b20bbae4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ee74a6ade401e200985e2421b20bbae4-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of estimating a rank-$1$ signal corrupted by structured rotationally invariant noise, and address the following question: \emph{how well do inference algorithms perform when the noise statistics is unknown and hence Gaussian noise is assumed?} While the matched Bayes-optimal setting with unstructured noise is well understood, the analysis of this mismatched problem is only at its premises. In this paper, we make a step towards understanding the effect of the strong source of mismatch which is the noise statistics. Our main technical contribution is the rigorous analysis of a Bayes estimator and of an approximate message passing (AMP) algorithm, both of which incorrectly assume a Gaussian setup. The first result exploits the theory of spherical integrals and of low-rank matrix perturbations; the idea behind the second one is to design and analyze an artificial AMP which, by taking advantage of the flexibility in the denoisers, is able to "correct" the mismatch. Armed with these sharp asymptotic characterizations, we unveil a rich and often unexpected phenomenology. For example, despite AMP is in principle designed to efficiently compute the Bayes estimator, the former is \emph{outperformed} by the latter in terms of mean-square error. We show that this performance gap is due to an incorrect estimation of the signal norm. In fact, when the SNR is large enough, the overlaps of the AMP and the Bayes estimator coincide, and they even match those of optimal estimators taking into account the structure of the noise.

        ----

        ## [2662] Scalable Interpretability via Polynomials

        **Authors**: *Abhimanyu Dubey, Filip Radenovic, Dhruv Mahajan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ee81a23d6b83ac15fbeb5b7a30934e0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ee81a23d6b83ac15fbeb5b7a30934e0b-Abstract-Conference.html)

        **Abstract**:

        Generalized Additive Models (GAMs) have quickly become the leading choice for interpretable machine learning. However, unlike uninterpretable methods such as DNNs, they lack expressive power and easy scalability, and are hence not a feasible alternative for real-world tasks. We present a new class of GAMs that use tensor rank decompositions of polynomials to learn powerful, {\em inherently-interpretable} models. Our approach, titled Scalable Polynomial Additive Models (SPAM) is effortlessly scalable and models {\em all} higher-order feature interactions without a combinatorial parameter explosion. SPAM outperforms all current interpretable approaches, and matches DNN/XGBoost performance on a series of real-world benchmarks with up to hundreds of thousands of features. We demonstrate by human subject evaluations that SPAMs are demonstrably more interpretable in practice, and are hence an effortless replacement for DNNs for creating interpretable and high-performance systems suitable for large-scale machine learning.Source code is available at \href{https://github.com/facebookresearch/nbm-spam}{\ttfamily github.com/facebookresearch/nbm-spam}.

        ----

        ## [2663] DeVRF: Fast Deformable Voxel Radiance Fields for Dynamic Scenes

        **Authors**: *Jiawei Liu, Yan-Pei Cao, Weijia Mao, Wenqiao Zhang, David Junhao Zhang, Jussi Keppo, Ying Shan, Xiaohu Qie, Mike Zheng Shou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html)

        **Abstract**:

        Modeling dynamic scenes is important for many applications such as virtual reality and telepresence. Despite achieving unprecedented fidelity for novel view synthesis in dynamic scenes, existing methods based on Neural Radiance Fields (NeRF) suffer from slow convergence (i.e., model training time measured in days). In this paper, we present DeVRF, a novel representation to accelerate learning dynamic radiance fields. The core of DeVRF is to model both the 3D canonical space and 4D deformation field of a dynamic, non-rigid scene with explicit and discrete voxel-based representations. However, it is quite challenging to train such a representation which has a large number of model parameters, often resulting in overfitting issues. To overcome this challenge, we devise a novel static-to-dynamic learning paradigm together with a new data capture setup that is convenient to deploy in practice. This paradigm unlocks efficient learning of deformable radiance fields via utilizing the 3D volumetric canonical space learnt from multi-view static images to ease the learning of 4D voxel deformation field with only few-view dynamic sequences. To further improve the efficiency of our DeVRF and its synthesized novel view's quality, we conduct thorough explorations and identify a set of strategies. We evaluate DeVRF on both synthetic and real-world dynamic scenes with different types of deformation. Experiments demonstrate that DeVRF achieves two orders of magnitude speedup (100Ã— faster) with on-par high-fidelity results compared to the previous state-of-the-art approaches. The code and dataset are released in https://github.com/showlab/DeVRF.

        ----

        ## [2664] Hiding Images in Deep Probabilistic Models

        **Authors**: *Haoyu Chen, Linqi Song, Zhenxing Qian, Xinpeng Zhang, Kede Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eec7fee9a8595ca964b9a11562767345-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eec7fee9a8595ca964b9a11562767345-Abstract-Conference.html)

        **Abstract**:

        Data hiding with deep neural networks (DNNs) has experienced impressive successes in recent years. A prevailing scheme is to train an autoencoder, consisting of an encoding network to embed (or transform) secret messages in (or into) a carrier, and a decoding network to extract the hidden messages. This scheme may suffer from several limitations regarding practicability, security, and embedding capacity. In this work, we describe a different computational framework to hide images in deep probabilistic models. Specifically, we use a DNN to model the probability density of cover images, and hide a secret image in one particular location of the learned distribution. As an instantiation, we adopt a SinGAN, a pyramid of generative adversarial networks (GANs), to learn the patch distribution of one cover image. We hide the secret image by fitting a deterministic mapping from a fixed set of noise maps (generated by an embedding key) to the secret image during patch distribution learning. The stego SinGAN, behaving as the original SinGAN, is publicly communicated; only the receiver with the embedding key is able to extract the secret image. We demonstrate the feasibility of our SinGAN approach in terms of extraction accuracy and model security. Moreover, we show the flexibility of the proposed method in terms of hiding multiple images for different receivers and obfuscating the secret image.

        ----

        ## [2665] ViewFool: Evaluating the Robustness of Visual Recognition to Adversarial Viewpoints

        **Authors**: *Yinpeng Dong, Shouwei Ruan, Hang Su, Caixin Kang, Xingxing Wei, Jun Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/eee7ae5cf0c4356c2aeca400771791aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/eee7ae5cf0c4356c2aeca400771791aa-Abstract-Conference.html)

        **Abstract**:

        Recent studies have demonstrated that visual recognition models lack robustness to distribution shift. However, current work mainly considers model robustness to 2D image transformations, leaving viewpoint changes in the 3D world less explored. In general, viewpoint changes are prevalent in various real-world applications (e.g., autonomous driving), making it imperative to evaluate viewpoint robustness. In this paper, we propose a novel method called ViewFool to find adversarial viewpoints that mislead visual recognition models. By encoding real-world objects as neural radiance fields (NeRF), ViewFool characterizes a distribution of diverse adversarial viewpoints under an entropic regularizer, which helps to handle the fluctuations of the real camera pose and mitigate the reality gap between the real objects and their neural representations. Experiments validate that the common image classifiers are extremely vulnerable to the generated adversarial viewpoints, which also exhibit high cross-model transferability. Based on ViewFool, we introduce ImageNet-V, a new out-of-distribution dataset for benchmarking viewpoint robustness of image classifiers. Evaluation results on 40 classifiers with diverse architectures, objective functions, and data augmentations reveal a significant drop in model performance when tested on ImageNet-V, which provides a possibility to leverage ViewFool as an effective data augmentation strategy to improve viewpoint robustness.

        ----

        ## [2666] Learning Superpoint Graph Cut for 3D Instance Segmentation

        **Authors**: *Le Hui, Linghua Tang, Yaqi Shen, Jin Xie, Jian Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ef0af61ccfba2bf9fad4f4df6dfcb7c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ef0af61ccfba2bf9fad4f4df6dfcb7c3-Abstract-Conference.html)

        **Abstract**:

        3D instance segmentation is a challenging task due to the complex local geometric structures of objects in point clouds. In this paper, we propose a learning-based superpoint graph cut method that explicitly learns the local geometric structures of the point cloud for 3D instance segmentation. Specifically, we first oversegment the raw point clouds into superpoints and construct the superpoint graph. Then, we propose an edge score prediction network to predict the edge scores of the superpoint graph, where the similarity vectors of two adjacent nodes learned through cross-graph attention in the coordinate and feature spaces are used for regressing edge scores. By forcing two adjacent nodes of the same instance to be close to the instance center in the coordinate and feature spaces, we formulate a geometry-aware edge loss to train the edge score prediction network. Finally, we develop a superpoint graph cut network that employs the learned edge scores and the predicted semantic classes of nodes to generate instances, where bilateral graph attention is proposed to extract discriminative features on both the coordinate and feature spaces for predicting semantic labels and scores of instances. Extensive experiments on two challenging datasets, ScanNet v2 and S3DIS, show that our method achieves new state-of-the-art performance on 3D instance segmentation.

        ----

        ## [2667] Estimating the Arc Length of the Optimal ROC Curve and Lower Bounding the Maximal AUC

        **Authors**: *Song Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ef0c0a23a1a8219c4fc381614664df3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ef0c0a23a1a8219c4fc381614664df3e-Abstract-Conference.html)

        **Abstract**:

        In this paper, we show the arc length of the optimal ROC curve is an $f$-divergence. By leveraging this result, we express the arc length using a variational objective and estimate it accurately using positive and negative samples. We show this estimator has a non-parametric convergence rate $O_p(n^{-\beta/4})$ ($\beta \in (0,1]$ depends on the smoothness). Using the same technique, we show the surface area sandwiched between the optimal ROC curve and the diagonal can be expressed via a similar variational objective. These new insights lead to a novel two-step classification procedure that maximizes an approximate lower bound of the maximal AUC.  Experiments on CIFAR-10 datasets show the proposed two-step procedure achieves good AUC performance in imbalanced binary classification tasks.

        ----

        ## [2668] Acceleration in Distributed Sparse Regression

        **Authors**: *Marie Maros, Gesualdo Scutari*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ef0c1457c4f31c00f460d55ab9d130ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ef0c1457c4f31c00f460d55ab9d130ed-Abstract-Conference.html)

        **Abstract**:

        We study acceleration for distributed sparse regression in   {\it  high-dimensions},  which allows the parameter size  to exceed and grow faster than the sample size. When applicable, existing  distributed algorithms employing acceleration perform poorly  in this setting, theoretically and numerically.  We  propose a new accelerated distributed algorithm suitable for high-dimensions. The method couples  a suitable instance of accelerated Nesterov's proximal gradient  with consensus and gradient-tracking mechanisms, aiming at estimating locally the gradient of the empirical loss while enforcing agreement on the local estimates.  Under standard assumptions on the statistical model and tuning parameters, the proposed method is proved to  globally converge   at {\it linear} rate  to an estimate that is within the {\it statistical precision} of the model. The iteration  complexity scales as $\mathcal{O}(\sqrt{\kappa})$, while the communications per iteration are at most $\widetilde{\mathcal{O}}(\log m/(1-\rho))$,  where $\kappa$ is the restricted condition number of the empirical loss, $m$ is the number of agents, and $\rho\in (0,1)$ measures the network connectivity. As by-product of our design, we also report    an accelerated method for high-dimensional estimations over  master-worker architectures, which is of independent interest and  compares favorably with existing works.

        ----

        ## [2669] Latency-aware Spatial-wise Dynamic Networks

        **Authors**: *Yizeng Han, Zhihang Yuan, Yifan Pu, Chenhao Xue, Shiji Song, Guangyu Sun, Gao Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ef472869c217bf693f2d9bbde66a6b07-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ef472869c217bf693f2d9bbde66a6b07-Abstract-Conference.html)

        **Abstract**:

        Spatial-wise dynamic convolution has become a promising approach to improving the inference efficiency of deep networks. By allocating more computation to the most informative pixels, such an adaptive inference paradigm reduces the spatial redundancy in image features and saves a considerable amount of unnecessary computation. However, the theoretical efficiency achieved by previous methods can hardly translate into a realistic speedup, especially on the multi-core processors (e.g. GPUs). The key challenge is that the existing literature has only focused on designing algorithms with minimal computation, ignoring the fact that the practical latency can also be influenced by scheduling strategies and hardware properties. To bridge the gap between theoretical computation and practical efficiency, we propose a latency-aware spatial-wise dynamic network (LASNet), which performs coarse-grained spatially adaptive inference under the guidance of a novel latency prediction model. The latency prediction model can efficiently estimate the inference latency of dynamic networks by simultaneously considering algorithms, scheduling strategies, and hardware properties. We use the latency predictor to guide both the algorithm design and the scheduling optimization on various hardware platforms. Experiments on image classification, object detection and instance segmentation demonstrate that the proposed framework significantly improves the practical inference efficiency of deep networks. For example, the average latency of a ResNet-101 on the ImageNet validation set could be reduced by 36% and 46% on a server GPU (Nvidia Tesla-V100) and an edge device (Nvidia Jetson TX2 GPU) respectively without sacrificing the accuracy. Code is available at https://github.com/LeapLabTHU/LASNet.

        ----

        ## [2670] Towards Versatile Embodied Navigation

        **Authors**: *Hanqing Wang, Wei Liang, Luc Van Gool, Wenguan Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ef4f2a0232a246b8a502135175e08953-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ef4f2a0232a246b8a502135175e08953-Abstract-Conference.html)

        **Abstract**:

        With the emergence of varied visual navigation tasks (e.g., image-/object-/audio-goal and vision-language navigation) that specify the target in different ways, the community has made appealing advances in training specialized agents capable of handling individual navigation tasks well. Given plenty of embodied navigation tasks and task-specific solutions, we address a more fundamental question: can we learn a single powerful agent that masters not one but multiple navigation tasks concurrently? First, we propose VXN, a large-scale 3D dataset that instantiates~four classic navigation tasks in standardized, continuous, and audiovisual-rich environments. Second, we propose Vienna, a versatile embodied navigation agent that simultaneously learns to perform the four navigation tasks with one model. Building upon a full-attentive architecture, Vienna formulates various navigation tasks as a unified, parse-and-query procedure: the target description, augmented with four task embeddings, is comprehensively interpreted into a set of diversified goal vectors, which are refined as the navigation progresses, and used as queries to retrieve supportive context from episodic history for decision making. This enables the reuse of knowledge across navigation tasks with varying input domains/modalities. We empirically demonstrate that, compared with learning each visual navigation task individually, our multitask agent achieves comparable or even better performance with reduced complexity.

        ----

        ## [2671] Explain My Surprise: Learning Efficient Long-Term Memory by predicting uncertain outcomes

        **Authors**: *Artyom Y. Sorokin, Nazar Buzun, Leonid Pugachev, Mikhail Burtsev*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ef7f6a2f18415e0e89edf50def91ecb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ef7f6a2f18415e0e89edf50def91ecb6-Abstract-Conference.html)

        **Abstract**:

        In many sequential tasks, a model needs to remember relevant events from the distant past to make correct predictions. Unfortunately, a straightforward application of gradient based training requires intermediate computations to be stored for every element of a sequence. This requires to store prohibitively large intermediate data if a sequence consists of thousands or even millions elements, and as a result, makes learning of very long-term dependencies infeasible. However, the majority of sequence elements can usually be predicted by taking into account only temporally local information. On the other hand, predictions affected by long-term dependencies are sparse and characterized by high uncertainty given only local information. We propose \texttt{MemUP}, a new training method that allows to learn long-term dependencies without backpropagating gradients through the whole sequence at a time. This method can potentially be  applied to any recurrent architecture.  LSTM network trained with \texttt{MemUP} performs better or comparable to baselines while requiring to store less intermediate data.

        ----

        ## [2672] Polyhistor: Parameter-Efficient Multi-Task Adaptation for Dense Vision Tasks

        **Authors**: *Yen-Cheng Liu, Chih-Yao Ma, Junjiao Tian, Zijian He, Zsolt Kira*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/efb02f96766a3b599c76852abf4d42dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/efb02f96766a3b599c76852abf4d42dd-Abstract-Conference.html)

        **Abstract**:

        Adapting large-scale pretrained models to various downstream tasks via fine-tuning is a standard method in machine learning. Recently, parameter-efficient fine-tuning methods have shown promise in adapting a pretrained model to different tasks while training only a few parameters. Despite their success, most existing methods are proposed in Natural Language Processing tasks with language Transformers, and adaptation to Computer Vision tasks with Vision Transformers remains under-explored, especially for dense vision tasks. Further, in multi-task settings, individually fine-tuning and storing separate models for different tasks is inefficient. In this work, we provide an extensive single- and multi-task parameter-efficient benchmark and examine existing parameter-efficient fine-tuning NLP methods for vision tasks. Our results on four different dense vision tasks showed that existing methods cannot be efficiently integrated due to the hierarchical nature of the Hierarchical Vision Transformers. To overcome this issue, we propose Polyhistor and Polyhistor-Lite, consisting of Decomposed HyperNetworks and Layer-wise Scaling Kernels, to share information across different tasks with a few trainable parameters. This leads to favorable performance improvements against existing parameter-efficient methods while using fewer trainable parameters. Specifically, Polyhistor achieves competitive accuracy compared to the state-of-the-art while only using less than 10% of their trainable parameters. Furthermore, our methods show larger performance gains when large networks and more pretraining data are used.

        ----

        ## [2673] LAPO: Latent-Variable Advantage-Weighted Policy Optimization for Offline Reinforcement Learning

        **Authors**: *Xi Chen, Ali Ghadirzadeh, Tianhe Yu, Jianhao Wang, Alex Yuan Gao, Wenzhe Li, Liang Bin, Chelsea Finn, Chongjie Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/efb2072a358cefb75886a315a6fcf880-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/efb2072a358cefb75886a315a6fcf880-Abstract-Conference.html)

        **Abstract**:

        Offline reinforcement learning methods hold the promise of learning policies from pre-collected datasets without the need to query the environment for new samples. This setting is particularly well-suited for continuous control robotic applications for which online data collection based on trial-and-error is costly and potentially unsafe. In practice, offline datasets are often heterogeneous, i.e., collected in a variety of scenarios, such as data from several human demonstrators or from policies that act with different purposes. Unfortunately, such datasets often contain action distributions with multiple modes and, in some cases, lack a sufficient number of high-reward trajectories, which render offline policy training inefficient. To address this challenge, we propose to leverage latent-variable generative model to represent high-advantage state-action pairs leading to better adherence to data distributions that contributes to solving the task, while maximizing reward via a policy over the latent variable. As we empirically show on a range of simulated locomotion, navigation, and manipulation tasks, our method referred to as latent-variable advantage-weighted policy optimization (LAPO), improves the average performance of the next best-performing offline reinforcement learning methods by 49\% on heterogeneous datasets, and by 8\% on datasets with narrow and biased distributions.

        ----

        ## [2674] GAMA: Generative Adversarial Multi-Object Scene Attacks

        **Authors**: *Abhishek Aich, Calvin-Khang Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit Roy-Chowdhury*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/efbd571f139d26604e53fe2760e2c073-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/efbd571f139d26604e53fe2760e2c073-Abstract-Conference.html)

        **Abstract**:

        The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code is available here: https://abhishekaich27.github.io/gama.html

        ----

        ## [2675] Predicting Label Distribution from Multi-label Ranking

        **Authors**: *Yunan Lu, Xiuyi Jia*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/efc549c2d22edf2f244b7013387c6251-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/efc549c2d22edf2f244b7013387c6251-Abstract-Conference.html)

        **Abstract**:

        Label distribution can provide richer information about label polysemy than logical labels in multi-label learning. There are currently two strategies including LDL (label distribution learning) and LE (label enhancement) to predict label distributions. LDL requires experts to annotate instances with label distributions and learn a predictive mapping on such a training set. LE requires experts to annotate instances with logical labels and generates label distributions from them. However, LDL requires costly annotation, and the performance of the LE is unstable. In this paper, we study the problem of predicting label distribution from multi-label ranking which is a compromise w.r.t. annotation cost but has good guarantees for performance. On the one hand, we theoretically investigate the relation between multi-label ranking and label distribution. We define the notion of EAE (expected approximation error) to quantify the quality of an annotation, give the bounds of EAE for multi-label ranking, and derive the optimal range of label distribution corresponding to a particular multi-label ranking. On the other hand, we propose a framework of label distribution predicting from multi-label ranking via conditional Dirichlet mixtures. This framework integrates the processes of recovering and learning label distributions end-to-end and allows us to easily encode our knowledge about current tasks by a scoring function. Finally, we implement extensive experiments to validate our proposal.

        ----

        ## [2676] Trajectory of Mini-Batch Momentum: Batch Size Saturation and Convergence in High Dimensions

        **Authors**: *Kiwon Lee, Andrew N. Cheng, Elliot Paquette, Courtney Paquette*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/efcb76ac1df9231a24893a957fcb9001-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/efcb76ac1df9231a24893a957fcb9001-Abstract-Conference.html)

        **Abstract**:

        We analyze the dynamics of large batch stochastic gradient descent with momentum (SGD+M) on the least squares problem when both the number of samples and dimensions are large. In this setting, we show that the dynamics of SGD+M converge to a deterministic discrete Volterra equation as dimension increases, which we analyze.  We identify a stability measurement, the implicit conditioning ratio (ICR), which regulates the ability of SGD+M to accelerate the algorithm.  When the batch size exceeds this ICR, SGD+M converges linearly at a rate of $\mathcal{O}(1/\sqrt{\kappa})$, matching optimal full-batch momentum (in particular performing as well as a full-batch but with a fraction of the size).  For batch sizes smaller than the ICR, in contrast, SGD+M has rates that scale like a multiple of the single batch SGD rate. We give explicit choices for the learning rate and momentum parameter in terms of the Hessian spectra that achieve this performance.

        ----

        ## [2677] Transformers from an Optimization Perspective

        **Authors**: *Yongyi Yang, Zengfeng Huang, David P. Wipf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/efd1e27afcb94addd03b9e14c8d9f78f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/efd1e27afcb94addd03b9e14c8d9f78f-Abstract-Conference.html)

        **Abstract**:

        Deep learning models such as the Transformer are often constructed by heuristics and experience.  To provide a complementary foundation, in this work we study the following problem: Is it possible to find an energy function underlying the Transformer model, such that descent steps along this energy correspond with the Transformer forward pass?  By finding such a function, we can reinterpret Transformers as the unfolding of an interpretable optimization process.  This unfolding perspective has been frequently adopted in the past to elucidate more straightforward deep models such as MLPs and CNNs; however, it has thus far remained elusive obtaining a similar equivalence for more complex models with self-attention mechanisms like the Transformer.  To this end, we first outline several major obstacles before providing companion techniques to at least partially address them, demonstrating for the first time a close association between energy function minimization and deep layers with self-attention.  This interpretation contributes to our intuition and understanding of Transformers, while potentially laying the ground-work for new model designs.

        ----

        ## [2678] PDSketch: Integrated Domain Programming, Learning, and Planning

        **Authors**: *Jiayuan Mao, Tomás Lozano-Pérez, Josh Tenenbaum, Leslie Pack Kaelbling*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/efe36e55d80a94d1726f660b8d237a0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/efe36e55d80a94d1726f660b8d237a0f-Abstract-Conference.html)

        **Abstract**:

        This paper studies a model learning and online planning approach towards building flexible and general robots. Specifically, we investigate how to exploit the locality and sparsity structures in the underlying environmental transition model to improve model generalization, data-efficiency, and runtime-efficiency. We present a new domain definition language, named PDSketch. It allows users to flexibly define high-level structures in the transition models, such as object and feature dependencies, in a way similar to how programmers use TensorFlow or PyTorch to specify kernel sizes and hidden dimensions of a convolutional neural network. The details of the transition model will be filled in by trainable neural networks. Based on the defined structures and learned parameters, PDSketch automatically generates domain-independent planning heuristics without additional training. The derived heuristics accelerate the performance-time planning for novel goals.

        ----

        ## [2679] Amortized Projection Optimization for Sliced Wasserstein Generative Models

        **Authors**: *Khai Nguyen, Nhat Ho*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html)

        **Abstract**:

        Seeking informative projecting directions has been an important task in utilizing sliced Wasserstein distance in applications. However, finding these directions usually requires an iterative optimization procedure over the space of projecting directions, which is computationally expensive. Moreover, the computational issue is even more severe in deep learning applications, where computing the distance between two mini-batch probability measures is repeated several times. This nested-loop has been one of the main challenges that prevent the usage of sliced Wasserstein distances based on good projections in practice. To address this challenge, we propose to utilize the \textit{learning-to-optimize} technique or \textit{amortized optimization} to predict the informative direction of any given two mini-batch probability measures. To the best of our knowledge, this is the first work that bridges amortized optimization and sliced Wasserstein generative models. In particular, we derive linear amortized models, generalized linear amortized models, and non-linear amortized models which are corresponding to three types of novel mini-batch losses, named \emph{amortized sliced Wasserstein}. We demonstrate the favorable performance of the proposed sliced losses in deep generative modeling on standard benchmark datasets.

        ----

        ## [2680] GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks

        **Authors**: *Jinsung Jeon, Jeonghak Kim, Haryong Song, Seunghyeon Cho, Noseong Park*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f03ce573aa8bce26f77b76f1cb9ee979-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f03ce573aa8bce26f77b76f1cb9ee979-Abstract-Conference.html)

        **Abstract**:

        Time series synthesis is an important research topic in the field of deep learning, which can be used for data augmentation. Time series data types can be broadly classified into regular or irregular. However, there are no existing generative models that show good performance for both types without any model changes. Therefore, we present a general purpose model capable of synthesizing regular and irregular time series data. To our knowledge, we are the first designing a general purpose time series synthesis model, which is one of the most challenging settings for time series synthesis. To this end, we design a generative adversarial network-based method, where many related techniques are carefully integrated into a single framework, ranging from neural ordinary/controlled differential equations to continuous time-flow processes. Our method outperforms all existing methods.

        ----

        ## [2681] Heterogeneous Skill Learning for Multi-agent Tasks

        **Authors**: *Yuntao Liu, Yuan Li, Xinhai Xu, Yong Dou, Donghong Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0606b882692637835e8ac981089eccd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0606b882692637835e8ac981089eccd-Abstract-Conference.html)

        **Abstract**:

        Heterogeneous behaviours are widespread in many multi-agent tasks, which have not been paid much attention in the community of multi-agent reinforcement learning. It would be a key factor for improving the learning performance to efficiently characterize and automatically find heterogeneous behaviours. In this paper, we introduce the concept of the skill to explore the ability of heterogeneous behaviours. We propose a novel skill-based multi-agent reinforcement learning framework to enable agents to master diverse skills. Specifically, our framework consists of the skill representation mechanism, the skill selector and the skill-based policy learning mechanism. We design an auto-encoder model to generate the latent variable as the skill representation by incorporating the environment information, which ensures the distinguishable of agents for skill selection and the discriminability for the skill learning. With the representation, a skill selection mechanism is invented to realize the assignment from agents to skills. Meanwhile, diverse skill-based policies are generated through a novel skill-based policy learning method. To promote efficient skill discovery, a mutual information based intrinsic reward function is constructed. Empirical results show that our framework obtains the best performance on three challenging benchmarks, i.e., StarCraft II micromanagement tasks, Google Research Football and GoBigger, over state-of-the-art MARL methods.

        ----

        ## [2682] On Margin Maximization in Linear and ReLU Networks

        **Authors**: *Gal Vardi, Ohad Shamir, Nati Srebro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html)

        **Abstract**:

        The implicit bias of neural networks has been extensively studied in recent years. Lyu and Li (2019) showed that in homogeneous networks trained with the exponential or the logistic loss, gradient flow converges to a KKT point of the max margin problem in parameter space. However, that leaves open the question of whether this point will generally be an actual optimum of the max margin problem. In this paper, we study this question in detail, for several neural network architectures involving linear and ReLU activations. Perhaps surprisingly, we show that in many cases, the KKT point is not even a local optimum of the max margin problem. On the flip side, we identify multiple settings where a local or global optimum can be guaranteed.

        ----

        ## [2683] Unsupervised Causal Generative Understanding of Images

        **Authors**: *Titas Anciukevicius, Patrick Fox-Roberts, Edward Rosten, Paul Henderson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f068c65585985c25c17f221390774ec7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f068c65585985c25c17f221390774ec7-Abstract-Conference.html)

        **Abstract**:

        We present a novel framework for unsupervised object-centric 3D scene understanding that generalizes robustly to out-of-distribution images. To achieve this, we design a causal generative model reflecting the physical process by which an image is produced, when a camera captures a scene containing multiple objects. This model is trained to reconstruct multi-view images via a latent representation describing the shapes, colours and positions of the 3D objects they show. It explicitly represents object instances as separate neural radiance fields, placed into a 3D scene. We then propose an inference algorithm that can infer this latent representation given a single out-of-distribution image as input -- even when it shows an unseen combination of components, unseen spatial compositions or a radically new viewpoint. We conduct extensive experiments applying our approach to test datasets that have zero probability under the training distribution. These show that it accurately reconstructs a scene's geometry, segments objects and infers their positions, despite not receiving any supervision. Our approach significantly out-performs baselines that do not capture the true causal image generation process.

        ----

        ## [2684] DART: Articulated Hand Model with Diverse Accessories and Rich Textures

        **Authors**: *Daiheng Gao, Yuliang Xiu, Kailin Li, Lixin Yang, Feng Wang, Peng Zhang, Bang Zhang, Cewu Lu, Ping Tan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Hand, the bearer of human productivity and intelligence, is receiving much attention due to the recent fever of digital twins. Among different hand morphable models, MANO has been widely used in vision and graphics community. However, MANO disregards textures and accessories, which largely limits its power to synthesize photorealistic hand data. In this paper, we extend MANO with Diverse Accessories and Rich Textures, namely DART. DART is composed of 50 daily 3D accessories which varies in appearance and shape, and 325 hand-crafted 2D texture maps covers different kinds of blemishes or make-ups. Unity GUI is also provided to generate synthetic hand data with user-defined settings, e.g., pose, camera, background, lighting, textures, and accessories. Finally, we release DARTset, which contains large-scale (800K), high-fidelity synthetic hand images, paired with perfect-aligned 3D labels. Experiments demonstrate its superiority in diversity. As a complement to existing hand datasets, DARTset boosts the generalization in both hand pose estimation and mesh recovery tasks. Raw ingredients (textures, accessories), Unity GUI, source code and DARTset are publicly available at dart2022.github.io.

        ----

        ## [2685] BadPrompt: Backdoor Attacks on Continuous Prompts

        **Authors**: *Xiangrui Cai, Haidong Xu, Sihan Xu, Ying Zhang, Xiaojie Yuan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0722b58f02d7793acf7d328928f933a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0722b58f02d7793acf7d328928f933a-Abstract-Conference.html)

        **Abstract**:

        The prompt-based learning paradigm has gained much research attention recently. It has achieved state-of-the-art performance on several NLP tasks, especially in the few-shot scenarios. While steering the downstream tasks, few works have been reported to investigate the security problems of the prompt-based models. In this paper, we conduct the first study on the vulnerability of the continuous prompt learning algorithm to backdoor attacks. We observe that the few-shot scenarios have posed a great challenge to backdoor attacks on the prompt-based models, limiting the usability of existing NLP backdoor methods. To address this challenge, we propose BadPrompt, a lightweight and task-adaptive algorithm, to backdoor attack continuous prompts. Specially, BadPrompt first generates candidate triggers which are indicative for predicting the targeted label and dissimilar to the samples of the non-targeted labels. Then, it automatically selects the most effective and invisible trigger for each sample with an adaptive trigger optimization algorithm. We evaluate the performance of BadPrompt on five datasets and two continuous prompt models. The results exhibit the abilities of BadPrompt to effectively attack continuous prompts while maintaining high performance on the clean test sets, outperforming the baseline models by a large margin. The source code of BadPrompt is publicly available.

        ----

        ## [2686] Off-Policy Evaluation with Policy-Dependent Optimization Response

        **Authors**: *Wenshuo Guo, Michael I. Jordan, Angela Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f074a994e062146561db9cdc63999efa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f074a994e062146561db9cdc63999efa-Abstract-Conference.html)

        **Abstract**:

        The intersection of causal inference and machine learning for decision-making is rapidly expanding, but the default decision criterion remains an average of individual causal outcomes across a population. In practice, various operational restrictions ensure that a decision-maker's utility is not realized as an average but rather as an output of a downstream decision-making problem (such as matching, assignment, network flow, minimizing predictive risk). In this work, we develop a new framework for off-policy evaluation with policy-dependent linear optimization responses: causal outcomes introduce stochasticity in objective function coefficients. Under this framework, a decision-maker's utility depends on the policy-dependent optimization, which introduces a fundamental challenge of optimization bias even for the case of policy evaluation. We construct unbiased estimators for the policy-dependent estimand by a perturbation method, and discuss asymptotic variance properties for a set of adjusted plug-in estimators. Lastly, attaining unbiased policy evaluation allows for policy optimization: we provide a general algorithm for optimizing causal interventions. We corroborate our theoretical results with numerical simulations.

        ----

        ## [2687] FIRE: Semantic Field of Words Represented as Non-Linear Functions

        **Authors**: *Xin Du, Kumiko Tanaka-Ishii*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f08223bc8d177df6807811c32f5acfed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f08223bc8d177df6807811c32f5acfed-Abstract-Conference.html)

        **Abstract**:

        State-of-the-art word embeddings presume a linear vector space, but this approach does not easily incorporate the nonlinearity that is necessary to represent polysemy. We thus propose a novel semantic FIeld REepresentation, called FIRE, which is a $D$-dimensional field in which every word is represented as a set of its locations and a nonlinear function covering the field. The strength of a word's relation to another word at a certain location is measured as the function value at that location. With FIRE, compositionality is represented via functional additivity, whereas polysemy is represented via the set of points and the function's multimodality. By implementing FIRE for English and comparing it with previous representation methods via word and sentence similarity tasks, we show that FIRE produces comparable or even better results. In an evaluation of polysemy to predict the number of word senses, FIRE greatly outperformed BERT and Word2vec, providing evidence of how FIRE represents polysemy. The code is available at https://github.com/kduxin/firelang.

        ----

        ## [2688] Sampling in Constrained Domains with Orthogonal-Space Variational Gradient Descent

        **Authors**: *Ruqi Zhang, Qiang Liu, Xin T. Tong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f092c84221d73387a6a5dd7517c500a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f092c84221d73387a6a5dd7517c500a5-Abstract-Conference.html)

        **Abstract**:

        Sampling methods, as important inference and learning techniques, are typically designed for unconstrained domains. However, constraints are ubiquitous in machine learning problems, such as those on safety, fairness, robustness, and many other properties that must be satisfied to apply sampling results in real-life applications. Enforcing these constraints often leads to implicitly-defined manifolds, making efficient sampling with constraints very challenging. In this paper, we propose a new variational framework with a designed orthogonal-space gradient flow (O-Gradient) for sampling on a manifold $\mathcal{G}_0$ defined by general equality constraints. O-Gradient decomposes the gradient into two parts: one decreases the distance to $\mathcal{G}_0$ and the other decreases the KL divergence in the orthogonal space. While most existing manifold sampling methods require initialization on $\mathcal{G}_0$, O-Gradient does not require such prior knowledge. We prove that O-Gradient converges to the target constrained distribution with rate $\widetilde{O}(1/\text{the number of iterations})$ under mild conditions. Our proof relies on a new Stein characterization of conditional measure which could be of independent interest. We implement O-Gradient through both Langevin dynamics and Stein variational gradient descent and demonstrate its effectiveness in various experiments, including Bayesian deep neural networks.

        ----

        ## [2689] Change-point Detection for Sparse and Dense Functional Data in General Dimensions

        **Authors**: *Carlos Misael Madrid Padilla, Daren Wang, Zifeng Zhao, Yi Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0add74c2f1ac58197173a38c01b2210-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0add74c2f1ac58197173a38c01b2210-Abstract-Conference.html)

        **Abstract**:

        We study the problem of change-point detection and localisation for functional data sequentially observed on a general $d$-dimensional space, where we allow the functional curves to be either sparsely or densely sampled. Data of this form naturally arise in a wide range of applications such as biology, neuroscience, climatology and finance. To achieve such a task, we propose a kernel-based algorithm named functional seeded binary segmentation (FSBS). FSBS is computationally efficient, can handle discretely observed functional data, and is theoretically sound for heavy-tailed and temporally-dependent observations. Moreover, FSBS works for a general $d$-dimensional domain, which is the first in the literature of change-point estimation for functional data.  We show the consistency of FSBS for multiple change-point estimation and further provide a sharp localisation error rate, which reveals an interesting phase transition phenomenon depending on the number of functional curves observed and the sampling frequency for each curve. Extensive numerical experiments illustrate the effectiveness of FSBS and its advantage over existing methods in the literature under various settings. A real data application is further conducted, where FSBS localises change-points of sea surface temperature patterns in the south Pacific attributed to El Ni\~{n}o.

        ----

        ## [2690] ClimbQ: Class Imbalanced Quantization Enabling Robustness on Efficient Inferences

        **Authors**: *Ting-An Chen, De-Nian Yang, Ming-Syan Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html)

        **Abstract**:

        Quantization compresses models to low bits for efficient inferences which has received increasing attentions. However, existing approaches focused on balanced datasets, while imbalanced data is pervasive in the real world. Therefore, in this study, we investigate the realistic problem, quantization on class-imbalanced data. We observe from the analytical results that quantizing imbalanced data tends to obtain a large error due to the differences between separate class distributions, which leads to a significant accuracy loss. To address this issue, we propose a novel quantization framework, Class Imbalanced Quantization (ClimbQ) that focuses on diminishing the inter-class heterogeneity for quantization error reduction. ClimbQ first scales the variance of each class distribution and then projects data through the new distributions to the same space for quantization. To guarantee the homogeneity of class variances after the ClimbQ process, we examine the quantized features and derive that the homogeneity satisfies when data size for each class is restricted (bounded). Accordingly, we design a Homogeneous Variance Loss (HomoVar Loss) which reweights the data losses of each class based on the bounded data sizes to satisfy the homogeneity of class variances. Extensive experiments on class-imbalanced and benchmark balanced datasets reveal that ClimbQ outperforms the state-of-the-art quantization techniques, especially on highly imbalanced data.

        ----

        ## [2691] Look Around and Refer: 2D Synthetic Semantics Knowledge Distillation for 3D Visual Grounding

        **Authors**: *Eslam Mohamed Bakr, Yasmeen Alsaedy, Mohamed Elhoseiny*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0b42291ddab77dcb2ef8a3488301b62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0b42291ddab77dcb2ef8a3488301b62-Abstract-Conference.html)

        **Abstract**:

        3D visual grounding task has been explored with visual and language streams to comprehend referential language for identifying targeted objects in 3D scenes.However, most existing methods devote the visual stream to capture the 3D visual clues using off-the-shelf point clouds encoders. The main question we address is “can we consolidate the 3D visual stream by 2D clues and efficiently utilize them in both training and testing phases?”. The main idea is to assist the 3D encoder by incorporating rich 2D object representations without requiring extra 2D inputs. To this end, we leverage 2D clues, synthetically generated from 3D point clouds, that empirically show their aptitude to boost the quality of the learned visual representations. We validate our approach through comprehensive experiments on Nr3D, Sr3D, and ScanRefer datasets. Our experiments show consistent performance gains against counterparts, where our proposed module, dubbed as LAR, significantly outperforms state-of-the-art 3D visual grounding techniques on three benchmarks.Our code will be made publicly available.

        ----

        ## [2692] Nonstationary Dual Averaging and Online Fair Allocation

        **Authors**: *Luofeng Liao, Yuan Gao, Christian Kroer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0bc367ccb66afc776fcac8d15549516-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0bc367ccb66afc776fcac8d15549516-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of fairly allocating sequentially arriving items to a set of individuals. For this problem, the recently-introduced PACE algorithm leverages the dual averaging algorithm to approximate competitive equilibria and thus generate online fair allocations. PACE is simple, distributed, and parameter-free, making it appealing for practical use in large-scale systems. However, current performance guarantees for PACE require i.i.d. item arrivals. Since real-world data is rarely i.i.d., or even stationary, we study the performance of PACE on nonstationary data. We start by developing new convergence results for the general dual averaging algorithm under three nonstationary input models: adversarially-corrupted stochastic input, ergodic input, and block-independent (including periodic) input. Our results show convergence of dual averaging up to errors caused by nonstationarity of the data, and recover the classical bounds when the input data is i.i.d. Using these results, we show that the PACE algorithm for online fair allocation simultaneously achieves ``best of many worlds'' guarantees against any of these nonstationary input models as well as against i.i.d. input. Finally, numerical experiments show strong empirical performance of PACE against nonstationary inputs.

        ----

        ## [2693] Incentivizing Combinatorial Bandit Exploration

        **Authors**: *Xinyan Hu, Dung Daniel T. Ngo, Aleksandrs Slivkins, Zhiwei Steven Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0d7b528c31bc3f9a0d5bab515ed6ed5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0d7b528c31bc3f9a0d5bab515ed6ed5-Abstract-Conference.html)

        **Abstract**:

        Consider a bandit algorithm that recommends actions to self-interested users in a recommendation system. The users are free to choose other actions and need to be incentivized to follow the algorithm's recommendations. While the users prefer to exploit, the algorithm can incentivize them to explore by leveraging the information collected from the previous users. All published work on this problem, known as incentivized exploration, focuses on small, unstructured action sets and mainly targets the case when the users' beliefs are independent across actions. However, realistic exploration problems often feature large, structured action sets and highly correlated beliefs. We focus on a paradigmatic exploration problem with structure: combinatorial semi-bandits. We prove that Thompson Sampling, when applied to combinatorial semi-bandits, is incentive-compatible when initialized with a sufficient number of samples of each arm (where this number is determined in advance by the Bayesian prior). Moreover, we design incentive-compatible algorithms for collecting the initial samples.

        ----

        ## [2694] Transfer Learning on Heterogeneous Feature Spaces for Treatment Effects Estimation

        **Authors**: *Ioana Bica, Mihaela van der Schaar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0e5cde3850e7dd0db125c0ebae16680-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0e5cde3850e7dd0db125c0ebae16680-Abstract-Conference.html)

        **Abstract**:

        Consider the problem of improving the estimation of conditional average treatment effects (CATE) for a target domain of interest by leveraging related information from a source domain with a different feature space. This heterogeneous transfer learning problem for CATE estimation is ubiquitous in areas such as healthcare where we may wish to evaluate the effectiveness of a treatment for a new patient population for which different clinical covariates and limited data are available. In this paper, we address this problem by introducing several building blocks that use representation learning to handle the heterogeneous feature spaces and a flexible multi-task architecture with shared and private layers to transfer information between potential outcome functions across domains. Then, we show how these building blocks can be used to recover transfer learning equivalents of the standard CATE learners. On a new semi-synthetic data simulation benchmark for heterogeneous transfer learning, we not only demonstrate performance improvements of our heterogeneous transfer causal effect learners across datasets, but also provide insights into the differences between these learners from a transfer perspective.

        ----

        ## [2695] Is Out-of-Distribution Detection Learnable?

        **Authors**: *Zhen Fang, Yixuan Li, Jie Lu, Jiahua Dong, Bo Han, Feng Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0e91b1314fa5eabf1d7ef6d1561ecec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0e91b1314fa5eabf1d7ef6d1561ecec-Abstract-Conference.html)

        **Abstract**:

        Supervised learning aims to train a classifier under the assumption that training and test data are from the same distribution. To ease the above assumption, researchers have studied a more realistic setting: out-of-distribution (OOD) detection, where test data may come from classes that are unknown during training (i.e., OOD data). Due to the unavailability and diversity of OOD data, good generalization ability is crucial for effective OOD detection algorithms. To study the generalization of OOD detection, in this paper, we investigate the probably approximately correct (PAC) learning theory of OOD detection, which is proposed by researchers as an open problem. First, we find a necessary condition for the learnability of OOD detection. Then, using this condition, we prove several impossibility theorems for the learnability of OOD detection under some scenarios. Although the impossibility theorems are frustrating, we find that some conditions of these impossibility theorems may not hold in some practical scenarios. Based on this observation, we next give several necessary and sufficient conditions to characterize the learnability of OOD detection in some practical scenarios. Lastly, we also offer theoretical supports for several representative OOD detection works based on our OOD theory.

        ----

        ## [2696] DMAP: a Distributed Morphological Attention Policy for learning to locomote with a changing body

        **Authors**: *Alberto Silvio Chiappa, Alessandro Marin Vargas, Alexander Mathis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f0fae49cdfab57c41c30c9b0244093cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f0fae49cdfab57c41c30c9b0244093cb-Abstract-Conference.html)

        **Abstract**:

        Biological and artificial agents need to deal with constant changes in the real world. We study this problem in four classical continuous control environments, augmented with morphological perturbations. Learning to locomote when the length and the thickness of different body parts vary is challenging, as the control policy is required to adapt to the morphology to successfully balance and advance the agent. We show that a control policy based on the proprioceptive state performs poorly with highly variable body configurations, while an (oracle) agent with access to a learned encoding of the perturbation performs significantly better. We introduce DMAP, a biologically-inspired, attention-based policy network architecture. DMAP combines independent proprioceptive processing, a distributed policy with individual controllers for each joint, and an attention mechanism, to dynamically gate sensory information from different body parts to different controllers. Despite not having access to the (hidden) morphology information, DMAP can be trained end-to-end in all the considered environments, overall matching or surpassing the performance of an oracle agent. Thus DMAP, implementing principles from biological motor control, provides a strong inductive bias for learning challenging sensorimotor tasks. Overall, our work corroborates the power of these principles in challenging locomotion tasks. The code is available at the following link: https://github.com/amathislab/dmap

        ----

        ## [2697] On the role of overparameterization in off-policy Temporal Difference learning with linear function approximation

        **Authors**: *Valentin Thomas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html)

        **Abstract**:

        Much of the recent successes of deep learning can be attributed to scaling up the size of the networks to the point where they often are vastly overparameterized. Thus, understanding the role of overparameterization is of increasing importance. While predictive theories have been developed for supervised learning, little is known about the Reinforcement Learning case. In this work, we take a theoretical approach and study the role of overparameterization for off-policy Temporal Difference (TD) learning in the linear setting. We leverage tools from Random Matrix Theory and random graph theory to obtain a characterization of the spectrum of the TD operator. We use this result to study the stability and optimization dynamics of TD learning as a function of the number of parameters.

        ----

        ## [2698] Whitening Convergence Rate of Coupling-based Normalizing Flows

        **Authors**: *Felix Draxler, Christoph Schnörr, Ullrich Köthe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f12d4e75bb8c62aba3e88d0586af96d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f12d4e75bb8c62aba3e88d0586af96d3-Abstract-Conference.html)

        **Abstract**:

        Coupling-based normalizing flows (e.g. RealNVP) are a popular family of normalizing flow architectures that work surprisingly well in practice. This calls for theoretical understanding. Existing work shows that such flows weakly converge to arbitrary data distributions. However, they make no statement about the stricter convergence criterion used in practice, the maximum likelihood loss. For the first time, we make a quantitative statement about this kind of convergence: We prove that all coupling-based normalizing flows perform whitening of the data distribution (i.e. diagonalize the covariance matrix) and derive corresponding convergence bounds that show a linear convergence rate in the depth of the flow. Numerical experiments demonstrate the implications of our theory and point at open questions.

        ----

        ## [2699] Neurosymbolic Deep Generative Models for Sequence Data with Relational Constraints

        **Authors**: *Halley Young, Maxwell Du, Osbert Bastani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f13ceb1b94145aad0e54186373cc86d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f13ceb1b94145aad0e54186373cc86d7-Abstract-Conference.html)

        **Abstract**:

        There has been significant recent progress designing deep generative models that generate realistic sequence data such as text or music. Nevertheless, it remains difficult to incorporate high-level structure to guide the generative process, and many such models perform well on local coherence, but less so on global coherence. We propose a novel approach for incorporating global structure in the form of relational constraints between different subcomponents of an example (e.g., lines of a poem or measures of music). Our generative model has two parts: (i) one model to generate a realistic set of relational constraints, and (ii) a second model to generate realistic data satisfying these constraints. For model (i), we propose a constrained optimization algorithm that infers the relational constraints present in the training data, and then learn a generative model based on the resulting constraint data.  In our experiments, we show that our approach significantly improves over state-of-the-art in terms of capturing high-level structure in the data, while performing comparably or better in terms of low-level structure.  We also show that using constrained optimization for part (ii) as well leads to increased controllability with little decrease in quality compared to pure learning-based models.

        ----

        ## [2700] Semi-Supervised Generative Models for Multiagent Trajectories

        **Authors**: *Dennis Fassmeyer, Pascal Fassmeyer, Ulf Brefeld*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f1fb6b2746332167f6670655372186cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f1fb6b2746332167f6670655372186cb-Abstract-Conference.html)

        **Abstract**:

        Analyzing the spatiotemporal behavior of multiple agents is of great interest to many communities. Existing probabilistic models in this realm are formalized either in an unsupervised framework, where the latent space is described by discrete or continuous variables, or in a supervised framework, where weakly preserved labels add explicit information to continuous latent representations. To overcome inherent limitations, we propose a novel objective function for processing multi-agent trajectories based on semi-supervised variational autoencoders, where equivariance and interaction of agents are captured via customized graph networks. The resulting architecture disentangles discrete and continuous latent effects and provides a natural solution for injecting expensive domain knowledge into interactive sequential systems. Empirically, our model not only outperforms various state-of-the-art baselines in trajectory forecasting, but also learns to effectively leverage unsupervised multi-agent sequences for classification tasks on interactive real-world sports datasets.

        ----

        ## [2701] PhysGNN: A Physics-Driven Graph Neural Network Based Model for Predicting Soft Tissue Deformation in Image-Guided Neurosurgery

        **Authors**: *Yasmin Salehi, Dennis Giannacopoulos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f200119a40846e508954abcd61f5f3fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f200119a40846e508954abcd61f5f3fd-Abstract-Conference.html)

        **Abstract**:

        Correctly capturing intraoperative brain shift in image-guided neurosurgical procedures is a critical task for aligning preoperative data with intraoperative geometry for ensuring accurate surgical navigation. While the finite element method (FEM) is a proven technique to effectively approximate soft tissue deformation through biomechanical formulations, their degree of success boils down to a trade-off between accuracy and speed. To circumvent this problem, the most recent works in this domain have proposed leveraging data-driven models obtained by training various machine learning algorithms---e.g., random forests, artificial neural networks (ANNs)---with the results of finite element analysis (FEA) to speed up tissue deformation approximations by prediction. These methods, however, do not account for the structure of the finite element (FE) mesh during training that provides information on node connectivities as well as the distance between them, which can aid with approximating tissue deformation based on the proximity of force load points with the rest of the mesh nodes. Therefore, this work proposes a novel framework, PhysGNN, a data-driven model that approximates the solution of the FEM by leveraging graph neural networks (GNNs), which are capable of accounting for the mesh structural information and inductive learning over unstructured grids and complex topological structures. Empirically, we demonstrate that the proposed architecture, PhysGNN, promises accurate and fast soft tissue deformation approximations, and is competitive with the state-of-the-art (SOTA) algorithms while promising enhanced computational feasibility, therefore suitable for neurosurgical settings.

        ----

        ## [2702] Towards Diverse and Faithful One-shot Adaption of Generative Adversarial Networks

        **Authors**: *Yabo Zhang, Mingshuai Yao, Yuxiang Wei, Zhilong Ji, Jinfeng Bai, Wangmeng Zuo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f2184e55a13b73b89f618ad24abb6ca7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f2184e55a13b73b89f618ad24abb6ca7-Abstract-Conference.html)

        **Abstract**:

        One-shot generative domain adaption aims to transfer a pre-trained generator on one domain to a new domain using one reference image only. However, it remains very challenging for the adapted generator (i) to generate diverse images inherited from the pre-trained generator while (ii) faithfully acquiring the domain-specific attributes and styles of the reference image. In this paper, we present a novel one-shot generative domain adaption method, i.e., DiFa, for diverse generation and faithful adaptation. For global-level adaptation, we leverage the difference between the CLIP embedding of the reference image and the mean embedding of source images to constrain the target generator. For local-level adaptation, we introduce an attentive style loss which aligns each intermediate token of an adapted image with its corresponding token of the reference image. To facilitate diverse generation, selective cross-domain consistency is introduced to select and retain domain-sharing attributes in the editing latent $\mathcal{W}+$ space to inherit the diversity of the pre-trained generator. Extensive experiments show that our method outperforms the state-of-the-arts both quantitatively and qualitatively, especially for the cases of large domain gap. Moreover, our DiFa can easily be extended to zero-shot generative domain adaption with appealing results.

        ----

        ## [2703] Deep Bidirectional Language-Knowledge Graph Pretraining

        **Authors**: *Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D. Manning, Percy Liang, Jure Leskovec*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f224f056694bcfe465c5d84579785761-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f224f056694bcfe465c5d84579785761-Abstract-Conference.html)

        **Abstract**:

        Pretraining a language model (LM) on text has been shown to help various downstream NLP tasks. Recent works show that a knowledge graph (KG) can complement text data, offering structured background knowledge that provides a useful scaffold for reasoning. However, these works are not pretrained to learn a deep fusion of the two modalities at scale, limiting the potential to acquire fully joint representations of text and KG. Here we propose DRAGON (Deep Bidirectional Language-Knowledge Graph Pretraining), a self-supervised approach to pretraining a deeply joint language-knowledge foundation model from text and KG at scale. Specifically, our model takes pairs of text segments and relevant KG subgraphs as input and bidirectionally fuses information from both modalities. We pretrain this model by unifying two self-supervised reasoning tasks, masked language modeling and KG link prediction. DRAGON outperforms existing LM and LM+KG models on diverse downstream tasks including question answering across general and biomedical domains, with +5% absolute gain on average. In particular, DRAGON achieves notable performance on complex reasoning about language and knowledge (+10% on questions involving long contexts or multi-step reasoning) and low-resource QA (+8% on OBQA and RiddleSense), and new state-of-the-art results on various BioNLP tasks. Our code and trained models are available at https://github.com/michiyasunaga/dragon.

        ----

        ## [2704] Theoretical analysis of deep neural networks for temporally dependent observations

        **Authors**: *Mingliang Ma, Abolfazl Safikhani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f23653913d8390cd4fc1bee8a3238e17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f23653913d8390cd4fc1bee8a3238e17-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks are powerful tools to model observations over time with non-linear patterns. Despite the widespread useof neural networks in such settings, most theoretical developments of deep neural networks are under the assumption of independent observations, and theoretical results for temporally dependent observations are scarce. To bridge this gap, we study theoretical properties of deep neural networks on modeling non-linear time series data. Specifically, non-asymptotic bounds for prediction error of (sparse) feed-forward neural network with ReLU activation function is established under mixing-type assumptions. These assumptions are mild such that they include a wide range of time series models including auto-regressive models. Compared to independent observations, established convergence rates have additional logarithmic factors to compensate for additional complexity due to dependence among data points. The theoretical results are supported via various numerical simulation settings as well as an application to a macroeconomic data set.

        ----

        ## [2705] A Theoretical Framework for Inference Learning

        **Authors**: *Nick Alonso, Beren Millidge, Jeffrey L. Krichmar, Emre O. Neftci*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f242c4cba2467637256722cb679642bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f242c4cba2467637256722cb679642bd-Abstract-Conference.html)

        **Abstract**:

        Backpropagation (BP) is the most successful and widely used algorithm in deep learning. However, the computations required by BP are challenging to reconcile with known neurobiology. This difficulty has stimulated interest in more biologically plausible alternatives to BP. One such algorithm is the inference learning algorithm (IL). IL trains predictive coding models of neural circuits and has achieved equal performance to BP on supervised and auto-associative tasks. In contrast to BP, however, the mathematical foundations of IL are not well-understood. Here, we develop a novel theoretical framework for IL. Our main result is that IL closely approximates an optimization method known as implicit stochastic gradient descent (implicit SGD), which is distinct from the explicit SGD implemented by BP. Our results further show how the standard implementation of IL can be altered to better approximate implicit SGD. Our novel implementation considerably improves the stability of IL across learning rates, which is consistent with our theory, as a key property of implicit SGD is its stability. We provide extensive simulation results that further support our theoretical interpretations and find IL achieves quicker convergence when trained with mini-batch size one while performing competitively with BP for larger mini-batches when combined with Adam.

        ----

        ## [2706] Cross-modal Learning for Image-Guided Point Cloud Shape Completion

        **Authors**: *Emanuele Aiello, Diego Valsesia, Enrico Magli*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f2a11632520f4b7473d7838f074a7d25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f2a11632520f4b7473d7838f074a7d25-Abstract-Conference.html)

        **Abstract**:

        In this paper we explore the recent topic of point cloud completion, guided by an auxiliary image. We show how it is possible to effectively combine the information from the two modalities in a localized latent space, thus avoiding the need for complex point cloud reconstruction methods from single views used by the state-of-the-art. We also investigate a novel self-supervised setting where the auxiliary image provides a supervisory signal to the training process by using a differentiable renderer on the completed point cloud to measure fidelity in the image space. Experiments show significant improvements over state-of-the-art supervised methods for both unimodal and multimodal completion. We also show the effectiveness of the self-supervised approach which outperforms a number of supervised methods and is competitive with the latest supervised models only exploiting point cloud information.

        ----

        ## [2707] Coreset for Line-Sets Clustering

        **Authors**: *Sagi Lotan, Ernesto Evgeniy Sanches Shayda, Dan Feldman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f2ce95887c34393af4eb240d60017860-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f2ce95887c34393af4eb240d60017860-Abstract-Conference.html)

        **Abstract**:

        The input to the {line-sets $k$-median} problem is an integer $k \geq 1$, and a set $\mathcal{L} = \{L_1,\dots,L_n\}$that contains $n$ sets of lines in $\mathbb{R}^d$. The goal is to compute a set $C$ of $k$ centers (points in $\mathbb{R}^d$) that minimizes the sum $\sum_{L \in \mathcal{L}}\min_{\ell\in L, c\in C}\mathrm{dist}(\ell,c)$ of Euclidean distances from each set to its closest center, where $\mathrm{dist}(\ell,c):=\min_{x\in \ell}\norm{x-c}_2$.An \emph{$\varepsilon$-coreset} for this problem is a weighted subset of sets in $\mathcal{L}$ that approximates this sum up to $1 \pm \varepsilon$ multiplicative factor, for every set $C$ of $k$ centers. We prove that \emph{every} such input set $\set{L}$ has a small $\varepsilon$-coreset, and provide the first coreset construction for this problem and its variants. The coreset consists of $O(\log^2n)$ weighted line-sets from $\set{L}$, and is constructed in $O(n\log n)$ time for every fixed $d, k\geq 1$ and $\varepsilon \in (0,1)$. The main technique is based on a novel reduction to a ``fair clustering'' of colored points to colored centers. We then provide a coreset for this coloring problem, which may be of independent interest. Open source code and experiments are also provided.

        ----

        ## [2708] Robust On-Policy Sampling for Data-Efficient Policy Evaluation in Reinforcement Learning

        **Authors**: *Rujie Zhong, Duohan Zhang, Lukas Schäfer, Stefano V. Albrecht, Josiah Hanna*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f2dbede0879b9d04ceb30f1b8b476b27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f2dbede0879b9d04ceb30f1b8b476b27-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning (RL) algorithms are often categorized as either on-policy or off-policy  depending on whether they use data from a target policy of interest or from a different behavior policy. In this paper, we study a subtle distinction between on-policy data and on-policy sampling in the context of the RL sub-problem of policy evaluation. We observe that on-policy sampling may fail to match the expected distribution of on-policy data after observing only a finite number of trajectories and this failure hinders data-efficient policy evaluation. Towards improved data-efficiency, we show how non-i.i.d., off-policy sampling can produce data that more closely matches the expected on-policy data distribution and consequently increases the accuracy of the Monte Carlo estimator for policy evaluation. We introduce a method called Robust On-Policy Sampling and demonstrate theoretically and empirically that it produces data that converges faster to the expected on-policy distribution compared to on-policy sampling. Empirically, we show that this faster convergence leads to lower mean squared error policy value estimates.

        ----

        ## [2709] Pre-Trained Model Reusability Evaluation for Small-Data Transfer Learning

        **Authors**: *Yao-Xiang Ding, Xi-Zhu Wu, Kun Zhou, Zhi-Hua Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f308b5f207348484552997c536375654-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f308b5f207348484552997c536375654-Abstract-Conference.html)

        **Abstract**:

        We study {\it model reusability evaluation} (MRE) for source pre-trained models: evaluating their transfer learning performance to new target tasks. In special, we focus on the setting under which the target training datasets are small, making it difficult to produce reliable MRE scores using them. Under this situation, we propose {\it synergistic learning} for building the task-model metric, which can be realized by collecting a set of pre-trained models and asking a group of data providers to participate. We provide theoretical guarantees to show that the learned task-model metric distances can serve as trustworthy MRE scores, and propose synergistic learning algorithms and models for general learning tasks. Experiments show that the MRE models learned by synergistic learning can generate significantly more reliable MRE scores than existing approaches for small-data transfer learning.

        ----

        ## [2710] Tree ensemble kernels for Bayesian optimization with known constraints over mixed-feature spaces

        **Authors**: *Alexander Thebelt, Calvin Tsay, Robert M. Lee, Nathan Sudermann-Merx, David Walz, Behrang Shafei, Ruth Misener*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f3398b76d17792893ce6d4f660546353-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f3398b76d17792893ce6d4f660546353-Abstract-Conference.html)

        **Abstract**:

        Tree ensembles can be well-suited for black-box optimization tasks such as algorithm tuning and neural architecture search, as they achieve good predictive performance with little or no manual tuning, naturally handle discrete feature spaces, and are relatively insensitive to outliers in the training data. Two well-known challenges in using tree ensembles for black-box optimization are (i) effectively quantifying model uncertainty for exploration and (ii) optimizing over the piece-wise constant acquisition function. To address both points simultaneously, we propose using the kernel interpretation of tree ensembles as a Gaussian Process prior to obtain model variance estimates, and we develop a compatible optimization formulation for the acquisition function. The latter further allows us to seamlessly integrate known constraints to improve sampling efficiency by considering domain-knowledge in engineering settings and modeling search space symmetries, e.g., hierarchical relationships in neural architecture search. Our framework performs as well as state-of-the-art methods for unconstrained black-box optimization over continuous/discrete features and outperforms competing methods for problems combining mixed-variable feature spaces and known input constraints.

        ----

        ## [2711] RLIP: Relational Language-Image Pre-training for Human-Object Interaction Detection

        **Authors**: *Hangjie Yuan, Jianwen Jiang, Samuel Albanie, Tao Feng, Ziyuan Huang, Dong Ni, Mingqian Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f37347375d8b54e3203e5d24aeb6c58c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f37347375d8b54e3203e5d24aeb6c58c-Abstract-Conference.html)

        **Abstract**:

        The task of Human-Object Interaction (HOI) detection targets fine-grained visual parsing of humans interacting with their environment, enabling a broad range of applications. Prior work has demonstrated the benefits of effective architecture design and integration of relevant cues for more accurate HOI detection. However, the design of an appropriate pre-training strategy for this task remains underexplored by existing approaches. To address this gap, we propose $\textit{Relational Language-Image Pre-training}$ (RLIP), a strategy for contrastive pre-training that leverages both entity and relation descriptions. To make effective use of such pre-training, we make three technical contributions: (1) a new $\textbf{Par}$allel entity detection and $\textbf{Se}$quential relation inference (ParSe) architecture that enables the use of both entity and relation descriptions during holistically optimized pre-training; (2) a synthetic data generation framework, Label Sequence Extension, that expands the scale of language data available within each minibatch; (3) ambiguity-suppression mechanisms, Relation Quality Labels and Relation Pseudo-Labels, to mitigate the influence of ambiguous/noisy samples in the pre-training data. Through extensive experiments, we demonstrate the benefits of these contributions, collectively termed RLIP-ParSe, for improved zero-shot, few-shot and fine-tuning HOI detection performance as well as increased robustness to learning from noisy annotations. Code will be available at https://github.com/JacobYuan7/RLIP.

        ----

        ## [2712] Skills Regularized Task Decomposition for Multi-task Offline Reinforcement Learning

        **Authors**: *Minjong Yoo, Sangwoo Cho, Honguk Woo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f376f5dff6f6ec6364aea7a46ab49574-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f376f5dff6f6ec6364aea7a46ab49574-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning (RL) with diverse offline datasets can have the advantage of leveraging the relation of multiple tasks and the common skills learned across those tasks, hence allowing us to deal with real-world complex problems efficiently in a data-driven way.  In offline RL where only offline data is used and online interaction with the environment is restricted, it is yet difficult to achieve the optimal policy for multiple tasks, especially when the data quality varies for the tasks. In this paper, we present a skill-based multi-task RL technique on heterogeneous datasets that are generated by behavior policies of different quality. To learn the shareable knowledge across those datasets effectively, we employ a task decomposition method for which common skills are jointly learned and used as guidance to reformulate a task in shared and achievable subtasks. In this joint learning, we use Wasserstein Auto-Encoder (WAE) to represent both skills and tasks on the same latent space and use the quality-weighted loss as a regularization term to induce tasks to be decomposed into subtasks that are more consistent with high-quality skills than others. To improve the performance of offline RL agents learned on the latent space, we also augment datasets with imaginary trajectories relevant to high-quality skills for each task. Through experiments, we show that our multi-task offline RL approach is robust to different-quality datasets and it outperforms other state-of-the-art algorithms for several robotic manipulation tasks and drone navigation tasks.

        ----

        ## [2713] Contextual Dynamic Pricing with Unknown Noise: Explore-then-UCB Strategy and Improved Regrets

        **Authors**: *Yiyun Luo, Will Wei Sun, Yufeng Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f38d1fd25c15a0ad9ba758de4e7b1819-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f38d1fd25c15a0ad9ba758de4e7b1819-Abstract-Conference.html)

        **Abstract**:

        Dynamic pricing is a fast-moving research area in machine learning and operations management. A lot of work has been done for this problem with known noise. In this paper, we consider a contextual dynamic pricing problem under a linear customer valuation model with an unknown market noise distribution $F$. This problem is very challenging due to the difficulty in balancing three tangled tasks of revenue-maximization, estimating the linear valuation parameter $\theta_{0}$, and learning the nonparametric $F$. To address this issue, we develop a novel {\it Explore-then-UCB} (ExUCB) strategy that includes an exploration for $\theta_{0}$-learning and a followed UCB procedure of joint revenue-maximization and $F$-learning. Under Lipschitz and 2nd-order smoothness assumptions on $F$, ExUCB is the first approach to achieve the $\tilde{O}(T^{2/3})$ regret rate. Under the Lipschitz assumption only, ExUCB matches the best existing regret of $\tilde{O}(T^{3/4})$ and is computationally more efficient. Furthermore, for regret lower bounds under the nonparametric $F$, not much work has been done beyond only assuming Lipschitz. To fill this gap, we provide the first $\tilde{\Omega}(T^{3/5})$ lower bound under Lipschitz and 2nd-order smoothness assumptions.

        ----

        ## [2714] Adversarially Robust Learning: A Generic Minimax Optimal Learner and Characterization

        **Authors**: *Omar Montasser, Steve Hanneke, Nati Srebro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f392c6bbb14548df50092f10c9db440f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f392c6bbb14548df50092f10c9db440f-Abstract-Conference.html)

        **Abstract**:

        We present a minimax optimal learner for the problem of learning predictors robust to adversarial examples at test-time. Interestingly, we find that this requires new algorithmic ideas and approaches to adversarially robust learning. In particular, we show, in a strong negative sense, the suboptimality of the robust learner proposed by Montasser, Hanneke, and Srebro [2019] and a broader family of learners we identify as local learners. Our results are enabled by adopting a global perspective, specifically, through a key technical contribution: the  the global one-inclusion graph, which may be of independent interest, that generalizes the classical one-inclusion graph due to Haussler, Littlestone, and Warmuth [1994]. Finally, as a byproduct, we identify a dimension characterizing qualitatively and quantitatively what classes of predictors $\mathcal{H}$ are robustly learnable. This resolves an open problem due to Montasser et al. [2019], and closes a (potentially) infinite gap between the established upper and lower bounds on the sample complexity of adversarially robust learning.

        ----

        ## [2715] The Implicit Delta Method

        **Authors**: *Nathan Kallus, James McInerney*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f3bf2b439c6f235828efdec1e48b72a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f3bf2b439c6f235828efdec1e48b72a3-Abstract-Conference.html)

        **Abstract**:

        Epistemic uncertainty quantification is a crucial part of drawing credible conclusions from predictive models, whether concerned about the prediction at a given point or any downstream evaluation that uses the model as input. When the predictive model is simple and its evaluation differentiable, this task is solved by the delta method, where we propagate the asymptotically-normal uncertainty in the predictive model through the evaluation to compute standard errors and Wald confidence intervals. However, this becomes difficult when the model and/or evaluation becomes more complex. Remedies include the bootstrap, but it can be computationally infeasible when training the model even once is costly. In this paper, we propose an alternative, the implicit delta method, which works by infinitesimally regularizing the training loss of the predictive model to automatically assess downstream uncertainty. We show that the change in the evaluation due to regularization is consistent for the asymptotic variance of the evaluation estimator, even when the infinitesimal change is approximated by a finite difference. This provides both a reliable quantification of uncertainty in terms of standard errors as well as permits the construction of calibrated confidence intervals. We discuss connections to other approaches to uncertainty quantification, both Bayesian and frequentist, and demonstrate our approach empirically.

        ----

        ## [2716] Singular Value Fine-tuning: Few-shot Segmentation requires Few-parameters Fine-tuning

        **Authors**: *Yanpeng Sun, Qiang Chen, Xiangyu He, Jian Wang, Haocheng Feng, Junyu Han, Errui Ding, Jian Cheng, Zechao Li, Jingdong Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f3bfbd65743e60c685a3845bd61ce15f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f3bfbd65743e60c685a3845bd61ce15f-Abstract-Conference.html)

        **Abstract**:

        Freezing the pre-trained backbone has become a standard paradigm to avoid overfitting in few-shot segmentation. In this paper, we rethink the paradigm and explore a new regime: {\em fine-tuning a small part of parameters in the backbone}. We present a solution to overcome the overfitting problem, leading to better model generalization on learning novel classes. Our method decomposes backbone parameters into three successive matrices via the Singular Value Decomposition (SVD), then {\em only fine-tunes the singular values} and keeps others frozen. The above design allows the model to adjust feature representations on novel classes while maintaining semantic clues within the pre-trained backbone. We evaluate our {\em Singular Value Fine-tuning (SVF)} approach on various few-shot segmentation methods with different backbones. We achieve state-of-the-art results on both Pascal-5$^i$ and COCO-20$^i$ across 1-shot and 5-shot settings. Hopefully, this simple baseline will encourage researchers to rethink the role of backbone fine-tuning in few-shot settings.

        ----

        ## [2717] On the relationship between variational inference and auto-associative memory

        **Authors**: *Louis Annabi, Alexandre Pitti, Mathias Quoy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f3d637987f36563fa45f943f8eadc2d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f3d637987f36563fa45f943f8eadc2d0-Abstract-Conference.html)

        **Abstract**:

        In this article, we propose a variational inference formulation of auto-associative memories, allowing us to combine perceptual inference and memory retrieval into the same mathematical framework. In this formulation, the prior probability distribution onto latent representations is made memory dependent, thus pulling the inference process towards previously stored representations. We then study how different neural network approaches to variational inference can be applied in this framework. We compare methods relying on amortized inference such as Variational Auto Encoders and methods relying on iterative inference such as Predictive Coding and suggest combining both approaches to design new auto-associative memory models. We evaluate the obtained algorithms on the CIFAR10 and CLEVR image datasets and compare them with other associative memory models such as Hopfield Networks, End-to-End Memory Networks and Neural Turing Machines.

        ----

        ## [2718] Annihilation of Spurious Minima in Two-Layer ReLU Networks

        **Authors**: *Yossi Arjevani, Michael Field*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f3da4165893c2465fd7e8df453c41ffa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f3da4165893c2465fd7e8df453c41ffa-Abstract-Conference.html)

        **Abstract**:

        We study the optimization problem associated with fitting two-layer ReLU neural networks with respect to the squared loss, where labels are generated by a target network. Use is made of the rich symmetry structure to develop a novel set of tools for studying the mechanism by which over-parameterization annihilates spurious minima through. Sharp analytic estimates are obtained for the loss and the Hessian spectrum at different minima, and it is shown that adding neurons can turn symmetric spurious minima into saddles through a local mechanism that does not generate new spurious minima; minima of smaller symmetry require more neurons. Using Cauchy's interlacing theorem, we prove the existence of descent directions in certain subspaces arising from the symmetry structure of the loss function. This analytic approach uses techniques, new to the field, from algebraic geometry, representation theory and symmetry breaking, and confirms rigorously the effectiveness of over-parameterization in making the associated loss landscape accessible to gradient-based methods. For a fixed number of neurons and inputs, the spectral results remain true under symmetry breaking perturbation of the target.

        ----

        ## [2719] A Closer Look at Weakly-Supervised Audio-Visual Source Localization

        **Authors**: *Shentong Mo, Pedro Morgado*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f3f2ff9579ba6deeb89caa2fe1f0b99c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f3f2ff9579ba6deeb89caa2fe1f0b99c-Abstract-Conference.html)

        **Abstract**:

        Audio-visual source localization is a challenging task that aims to predict the location of visual sound sources in a video. Since collecting ground-truth annotations of sounding objects can be costly, a plethora of weakly-supervised localization methods that can learn from datasets with no bounding-box annotations have been proposed in recent years, by leveraging the natural co-occurrence of audio and visual signals. Despite significant interest, popular evaluation protocols have two major flaws. First, they allow for the use of a fully annotated dataset to perform early stopping, thus significantly increasing the annotation effort required for training. Second, current evaluation metrics assume the presence of sound sources at all times. This is of course an unrealistic assumption, and thus better metrics are necessary to capture the model's performance on (negative) samples with no visible sound sources. To accomplish this, we extend the test set of popular benchmarks, Flickr SoundNet and VGG-Sound Sources, in order to include negative samples, and measure performance using metrics that balance localization accuracy and recall. Using the new protocol, we conducted an extensive evaluation of prior methods, and found that most prior works are not capable of identifying negatives and suffer from significant overfitting problems (rely heavily on early stopping for best results). We also propose a new approach for visual sound source localization that addresses both these problems. In particular, we found that, through extreme visual dropout and the use of momentum encoders, the proposed approach combats overfitting effectively, and establishes a new state-of-the-art performance on both Flickr SoundNet and VGG-Sound Source. Code and pre-trained models are available at https://github.com/stoneMo/SLAVC.

        ----

        ## [2720] Posterior Collapse of a Linear Latent Variable Model

        **Authors**: *Zihao Wang, Ziyin Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f419342f8afd483c781f69c2fabfe4f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f419342f8afd483c781f69c2fabfe4f6-Abstract-Conference.html)

        **Abstract**:

        This work identifies the existence and cause of a type of posterior collapse that frequently occurs in the Bayesian deep learning practice. For a general linear latent variable model that includes linear variational autoencoders as a special case, we precisely identify the nature of posterior collapse to be the competition between the likelihood and the regularization of the mean due to the prior. Our result also suggests that posterior collapse may be a general problem of learning for deeper architectures and deepens our understanding of Bayesian deep learning.

        ----

        ## [2721] Accelerating SGD for Highly Ill-Conditioned Huge-Scale Online Matrix Completion

        **Authors**: *Jialun Zhang, Hong-Ming Chiu, Richard Y. Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f4304cf113235aef5dd0d0330b349940-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f4304cf113235aef5dd0d0330b349940-Abstract-Conference.html)

        **Abstract**:

        The matrix completion problem seeks to recover a $d\times d$ ground truth matrix of low rank $r\ll d$ from observations of its individual elements. Real-world matrix completion is often a huge-scale optimization problem, with $d$ so large that even the simplest full-dimension vector operations with $O(d)$ time complexity become prohibitively expensive. Stochastic gradient descent (SGD) is one of the few algorithms capable of solving matrix completion on a huge scale, and can also naturally handle streaming data over an evolving ground truth. Unfortunately, SGD experiences a dramatic slow-down when the underlying ground truth is ill-conditioned; it requires at least $O(\kappa\log(1/\epsilon))$ iterations to get $\epsilon$-close to ground truth matrix with condition number $\kappa$. In this paper, we propose a preconditioned version of SGD that preserves all the favorable practical qualities of SGD for huge-scale online optimization while also making it agnostic to $\kappa$. For a symmetric ground truth and the Root Mean Square Error (RMSE) loss, we prove that the preconditioned SGD converges to $\epsilon$-accuracy in $O(\log(1/\epsilon))$ iterations, with a rapid linear convergence rate as if the ground truth were perfectly conditioned with $\kappa=1$. In our numerical experiments, we observe a similar acceleration forill-conditioned matrix completion under the root mean square error (RMSE) loss, Euclidean distance matrix (EDM) completion under pairwise square loss, and collaborative filtering under the Bayesian Personalized Ranking (BPR) loss.

        ----

        ## [2722] Lifting Weak Supervision To Structured Prediction

        **Authors**: *Harit Vishwakarma, Frederic Sala*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f463d31ed2fdd7b0ec585c041ec1baa8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f463d31ed2fdd7b0ec585c041ec1baa8-Abstract-Conference.html)

        **Abstract**:

        Weak supervision (WS) is a rich set of techniques that produce pseudolabels by aggregating easily obtained but potentially noisy label estimates from various sources. WS is theoretically well-understood for binary classification, where simple approaches enable consistent estimation of pseudolabel noise rates. Using this result, it has been shown that downstream models trained on the pseudolabels have generalization guarantees nearly identical to those trained on clean labels. While this is exciting, users often wish to use WS for \emph{structured prediction}, where the output space consists of more than a binary or multi-class label set: e.g. rankings, graphs, manifolds, and more. Do the favorable theoretical properties of WS for binary classification lift to this setting? We answer this question in the affirmative for a wide range of scenarios. For labels taking values in a finite metric space, we introduce techniques new to weak supervision based on pseudo-Euclidean embeddings and tensor decompositions, providing a nearly-consistent noise rate estimator. For labels in constant-curvature Riemannian manifolds, we introduce new invariants that also yield consistent noise rate estimation. In both cases, when using the resulting pseudolabels in concert with a flexible downstream model, we obtain generalization guarantees nearly identical to those for models trained on clean data. Several of our results, which can be viewed as robustness guarantees in structured prediction with noisy labels, may be of independent interest.

        ----

        ## [2723] A Lagrangian Duality Approach to Active Learning

        **Authors**: *Juan Elenter, Navid Naderializadeh, Alejandro Ribeiro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f475bdd151d8b5fa01215aeda925e75c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f475bdd151d8b5fa01215aeda925e75c-Abstract-Conference.html)

        **Abstract**:

        We consider the pool-based active learning problem, where only a subset of the training data is labeled, and the goal is to query a batch of unlabeled samples to be labeled so as to maximally improve model performance. We formulate the problem using constrained learning, where a set of constraints bounds the performance of the model on labeled samples. Considering a primal-dual approach, we optimize the primal variables, corresponding to the model parameters, as well as the dual variables, corresponding to the constraints. As each dual variable indicates how significantly the perturbation of the respective constraint affects the optimal value of the objective function, we use it as a proxy of the informativeness of the corresponding training sample. Our approach, which we refer to as Active Learning via Lagrangian dualitY, or ALLY, leverages this fact to select a diverse set of unlabeled samples with the highest estimated dual variables as our query set. We demonstrate the benefits of our approach in a variety of classification and regression tasks and discuss its limitations depending on the capacity of the model used and the degree of redundancy in the dataset. We also examine the impact of the distribution shift induced by active sampling and show that ALLY can be used in a generative mode to create novel, maximally-informative samples.

        ----

        ## [2724] Instance-optimal PAC Algorithms for Contextual Bandits

        **Authors**: *Zhaoqi Li, Lillian J. Ratliff, Houssam Nassif, Kevin G. Jamieson, Lalit Jain*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f4821075019a058700f6e6738eea1365-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f4821075019a058700f6e6738eea1365-Abstract-Conference.html)

        **Abstract**:

        In the stochastic contextual bandit setting, regret-minimizing algorithms have been extensively researched, but their instance-minimizing best-arm identification counterparts remain seldom studied. In this work, we focus on the stochastic bandit problem in the $(\epsilon,\delta)$-PAC setting: given a policy class $\Pi$ the goal of the learner is to return a policy $\pi\in \Pi$ whose expected reward is within $\epsilon$ of the optimal policy with probability greater than $1-\delta$. We characterize the first instance-dependent PAC sample complexity of contextual bandits through a quantity $\rho_{\Pi}$, and provide matching upper and lower bounds in terms of $\rho_{\Pi}$ for the agnostic and linear contextual best-arm identification settings. We show that no algorithm can be simultaneously minimax-optimal for regret minimization and instance-dependent PAC for best-arm identification. Our main result is a new instance-optimal and computationally efficient algorithm that relies on a polynomial number of calls to a cost-sensitive classification oracle.

        ----

        ## [2725] xView3-SAR: Detecting Dark Fishing Activity Using Synthetic Aperture Radar Imagery

        **Authors**: *Fernando Paolo, Tsu-ting Tim Lin, Ritwik Gupta, Bryce Goodman, Nirav Patel, Daniel Kuster, David Kroodsma, Jared Dunnmon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Unsustainable fishing practices worldwide pose a major threat to marine resources and ecosystems. Identifying vessels that do not show up in conventional monitoring systems---known as ``dark vessels''---is key to managing and securing the health of marine environments. With the rise of satellite-based synthetic aperture radar (SAR) imaging and modern machine learning (ML), it is now possible to automate detection of dark vessels day or night, under all-weather conditions. SAR images, however, require a domain-specific treatment and are not widely accessible to the ML community. Maritime objects (vessels and offshore infrastructure) are relatively small and sparse, challenging traditional computer vision approaches. We present the largest labeled dataset for training ML models to detect and characterize vessels and ocean structures in SAR imagery. xView3-SAR consists of nearly 1,000 analysis-ready SAR images from the Sentinel-1 mission that are, on average, 29,400-by-24,400 pixels each. The images are annotated using a combination of automated and manual analysis. Co-located bathymetry and wind state rasters accompany every SAR image. We also provide an overview of the xView3 Computer Vision Challenge, an international competition using xView3-SAR for ship detection and characterization at large scale. We release the data  (\href{https://iuu.xview.us/}{https://iuu.xview.us/}) and code (\href{https://github.com/DIUx-xView}{https://github.com/DIUx-xView}) to support ongoing development and evaluation of ML approaches for this important application.

        ----

        ## [2726] Understanding the Failure of Batch Normalization for Transformers in NLP

        **Authors**: *Jiaxi Wang, Ji Wu, Lei Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f4f2f2b3c67da711df6df557fc870c4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f4f2f2b3c67da711df6df557fc870c4a-Abstract-Conference.html)

        **Abstract**:

        Batch Normalization (BN) is a core and prevalent technique in accelerating the training of deep neural networks and improving the generalization on Computer Vision (CV) tasks. However, it fails to defend its position in Natural Language Processing (NLP), which is dominated by Layer Normalization (LN). In this paper, we are trying to answer why BN usually performs worse than LN in NLP tasks with Transformer models. We find that the inconsistency between training and inference of BN is the leading cause that results in the failure of BN in NLP. We define Training Inference Discrepancy (TID) to quantitatively measure this inconsistency and reveal that TID can indicate BN's performance, supported by extensive experiments, including image classification, neural machine translation, language modeling, sequence labeling, and text classification tasks. We find that BN can obtain much better test performance than LN when TID keeps small through training. To suppress the explosion of TID, we propose Regularized BN (RBN) that adds a simple regularization term to narrow the gap between batch statistics and population statistics of BN. RBN improves the performance of BN consistently and outperforms or is on par with LN on 17 out of 20 settings, including ten datasets and two common variants of Transformer.

        ----

        ## [2727] Exploration via Elliptical Episodic Bonuses

        **Authors**: *Mikael Henaff, Roberta Raileanu, Minqi Jiang, Tim Rocktäschel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f4f79698d48bdc1a6dec20583724182b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f4f79698d48bdc1a6dec20583724182b-Abstract-Conference.html)

        **Abstract**:

        In recent years, a number of reinforcement learning (RL) methods have been pro- posed to explore complex environments which differ across episodes. In this work, we show that the effectiveness of these methods critically relies on a count-based episodic term in their exploration bonus. As a result, despite their success in relatively simple, noise-free settings, these methods fall short in more realistic scenarios where the state space is vast and prone to noise. To address this limitation, we introduce Exploration via Elliptical Episodic Bonuses (E3B), a new method which extends count-based episodic bonuses to continuous state spaces and encourages an agent to explore states that are diverse under a learned embed- ding within each episode. The embedding is learned using an inverse dynamics model in order to capture controllable aspects of the environment. Our method sets a new state-of-the-art across 16 challenging tasks from the MiniHack suite, without requiring task-specific inductive biases. E3B also outperforms existing methods in reward-free exploration on Habitat, demonstrating that it can scale to high-dimensional pixel-based observations and realistic environments.

        ----

        ## [2728] Efficient and Effective Multi-task Grouping via Meta Learning on Task Combinations

        **Authors**: *Xiaozhuang Song, Shun Zheng, Wei Cao, James J. Q. Yu, Jiang Bian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f50f282a3093d36471008b045bd478af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f50f282a3093d36471008b045bd478af-Abstract-Conference.html)

        **Abstract**:

        As a longstanding learning paradigm, multi-task learning has been widely applied into a variety of machine learning applications. Nonetheless, identifying which tasks should be learned together is still a challenging fundamental problem because the possible task combinations grow exponentially with the number of tasks, and existing solutions heavily relying on heuristics may probably lead to ineffective groupings with severe performance degradation. To bridge this gap, we develop a systematic multi-task grouping framework with a new meta-learning problem on task combinations, which is to predict the per-task performance gains of multi-task learning over single-task learning for any combination. Our underlying assumption is that no matter how large the space of task combinations is, the relationships between task combinations and performance gains lie in some low-dimensional manifolds and thus can be learnable. Accordingly, we develop a neural meta learner, MTG-Net, to capture these relationships, and design an active learning strategy to progressively select meta-training samples. In this way, even with limited meta samples, MTG-Net holds the potential to produce reasonable gain estimations on arbitrary task combinations. Extensive experiments on diversified multi-task scenarios demonstrate the efficiency and effectiveness of our method. Specifically, in a large-scale evaluation with $27$ tasks, which produce over one hundred million task combinations, our method almost doubles the performance obtained by the existing best solution given roughly the same computational cost. Data and code are available at https://github.com/ShawnKS/MTG-Net.

        ----

        ## [2729] Learning Fractional White Noises in Neural Stochastic Differential Equations

        **Authors**: *Anh Tong, Thanh Nguyen-Tang, Toan M. Tran, Jaesik Choi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f51df088779c27cbb25b8f094a346544-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f51df088779c27cbb25b8f094a346544-Abstract-Conference.html)

        **Abstract**:

        Differential equations play important roles in modeling complex physical systems. Recent advances present interesting research directions by combining differential equations with neural networks. By including noise, stochastic differential equations (SDEs) allows us to model data with uncertainty and measure imprecision. There are many variants of noises known to exist in many real-world data. For example, previously white noises are idealized and induced by Brownian motions. Nevertheless, there is a lack of machine learning models that can handle such noises. In this paper, we introduce a generalized fractional white noise to existing models and propose an efficient approximation of noise sample paths based on classical integration methods and sparse Gaussian processes. Our experimental results demonstrate that the proposed model can capture noise characteristics such as continuity from various time series data, therefore improving model fittings over existing models. We examine how we can apply our approach to score-based generative models, showing that there exists a case of our generalized noise resulting in a better image generation measure.

        ----

        ## [2730] CryptoGCN: Fast and Scalable Homomorphically Encrypted Graph Convolutional Network Inference

        **Authors**: *Ran Ran, Wei Wang, Quan Gang, Jieming Yin, Nuo Xu, Wujie Wen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f5332c8273d02729730a9c24dec2135e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f5332c8273d02729730a9c24dec2135e-Abstract-Conference.html)

        **Abstract**:

        Recently cloud-based graph convolutional network (GCN) has demonstrated great success and potential in many privacy-sensitive applications such as personal healthcare and financial systems. Despite its high inference accuracy and performance on the cloud, maintaining data privacy in GCN inference, which is of paramount importance to these practical applications, remains largely unexplored. In this paper, we take an initial attempt towards this and develop CryptoGCN--a homomorphic encryption (HE) based GCN inference framework. A key to the success of our approach is to reduce the tremendous computational overhead for HE operations, which can be orders of magnitude higher than its counterparts in the plaintext space. To this end, we develop a solution that can effectively take advantage of the sparsity of matrix operations in GCN inference to significantly reduce the encrypted computational overhead. Specifically, we propose a novel Adjacency Matrix-Aware (AMA) data formatting method along with the AMA assisted patterned sparse matrix partitioning, to exploit the complex graph structure and perform efficient matrix-matrix multiplication in HE computation. In this way, the number of HE operations can be significantly reduced.  We also develop a co-optimization framework that can explore the trade-offs among the accuracy, security level, and computational overhead by judicious pruning and polynomial approximation of activation modules in GCNs. Based on the NTU-XVIEW skeleton joint dataset, i.e., the largest dataset evaluated homomorphically by far as we are aware of, our experimental results demonstrate that CryptoGCN outperforms state-of-the-art solutions in terms of the latency and number of homomorphic operations, i.e., achieving as much as a 3.10$\times$ speedup on latency and reduces the total Homomorphic Operation Count (HOC) by 77.4\% with a small accuracy loss of 1-1.5$\%$. Our code is publicly available at https://github.com/ranran0523/CryptoGCN.

        ----

        ## [2731] UniGAN: Reducing Mode Collapse in GANs using a Uniform Generator

        **Authors**: *Ziqi Pan, Li Niu, Liqing Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f5537b8d8fd126c7fe9d7429b181b1eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f5537b8d8fd126c7fe9d7429b181b1eb-Abstract-Conference.html)

        **Abstract**:

        Despite the significant progress that has been made in the training of Generative Adversarial Networks (GANs), the mode collapse problem remains a major challenge in training GANs, which refers to a lack of diversity in generative samples. In this paper, we propose a new type of generative diversity named uniform diversity, which relates to a newly proposed type of mode collapse named $u$-mode collapse where the generative samples distribute nonuniformly over the data manifold. From a geometric perspective, we show that the uniform diversity is closely related with the generator uniformity property, and the maximum uniform diversity is achieved if the generator is uniform. To learn a uniform generator, we propose UniGAN, a generative framework with a Normalizing Flow based generator and a simple yet sample efficient generator uniformity regularization, which can be easily adapted to any other generative framework. A new type of diversity metric named udiv is also proposed to estimate the uniform diversity given a set of generative samples in practice. Experimental results verify the effectiveness of our UniGAN in learning a uniform generator and improving uniform diversity.

        ----

        ## [2732] UMIX: Improving Importance Weighting for Subpopulation Shift via Uncertainty-Aware Mixup

        **Authors**: *Zongbo Han, Zhipeng Liang, Fan Yang, Liu Liu, Lanqing Li, Yatao Bian, Peilin Zhao, Bingzhe Wu, Changqing Zhang, Jianhua Yao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f593c9c251d4d7cf14d4ab9861dfb7eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f593c9c251d4d7cf14d4ab9861dfb7eb-Abstract-Conference.html)

        **Abstract**:

        Subpopulation shift widely exists in many real-world machine learning applications, referring to the training and test distributions containing the same subpopulation groups but varying in subpopulation frequencies. Importance reweighting is a normal way to handle the subpopulation shift issue by imposing constant or adaptive sampling weights on each sample in the training dataset.  However, some recent studies have recognized that most of these approaches fail to improve the performance over empirical risk minimization especially when applied to over-parameterized neural networks. In this work, we propose a simple yet practical framework, called uncertainty-aware mixup (UMIX), to mitigate the overfitting issue in over-parameterized models by reweighting the ''mixed'' samples according to the sample uncertainty. The training-trajectories-based uncertainty estimation is equipped in the proposed UMIX for each sample to flexibly characterize the subpopulation distribution. We also provide insightful theoretical analysis to verify that UMIX achieves better generalization bounds over prior works. Further, we conduct extensive empirical studies across a wide range of tasks to  validate the effectiveness of our method both qualitatively and quantitatively. Code is available at https://github.com/TencentAILabHealthcare/UMIX.

        ----

        ## [2733] Exploit Reward Shifting in Value-Based Deep-RL: Optimistic Curiosity-Based Exploration and Conservative Exploitation via Linear Reward Shaping

        **Authors**: *Hao Sun, Lei Han, Rui Yang, Xiaoteng Ma, Jian Guo, Bolei Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f600d1a3f6a63f782680031f3ce241a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f600d1a3f6a63f782680031f3ce241a7-Abstract-Conference.html)

        **Abstract**:

        In this work, we study the simple yet universally applicable case of reward shaping in value-based Deep Reinforcement Learning (DRL). We show that reward shifting in the form of a linear transformation is equivalent to changing the initialization of the $Q$-function in function approximation. Based on such an equivalence, we bring the key insight that a positive reward shifting leads to conservative exploitation, while a negative reward shifting leads to curiosity-driven exploration. Accordingly, conservative exploitation improves offline RL value estimation, and optimistic value estimation improves exploration for online RL. We validate our insight on a range of RL tasks and show its improvement over baselines: (1) In offline RL, the conservative exploitation leads to improved performance based on off-the-shelf algorithms; (2) In online continuous control, multiple value functions with different shifting constants can be used to tackle the exploration-exploitation dilemma for better sample efficiency; (3) In discrete control tasks, a negative reward shifting yields an improvement over the curiosity-based exploration method.

        ----

        ## [2734] Stability and Generalization for Markov Chain Stochastic Gradient Methods

        **Authors**: *Puyu Wang, Yunwen Lei, Yiming Ying, Ding-Xuan Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f61538f83b0f19f9306d9d801c15f41c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f61538f83b0f19f9306d9d801c15f41c-Abstract-Conference.html)

        **Abstract**:

        Recently there is a large amount of work devoted to the study of Markov chain stochastic gradient methods (MC-SGMs)  which mainly focus on their convergence analysis for solving minimization problems. In this paper, we provide a comprehensive generalization analysis of MC-SGMs for both minimization and minimax problems through the lens of algorithmic stability in the framework of statistical learning theory. For empirical risk minimization (ERM) problems, we establish the optimal excess population risk bounds for both smooth and non-smooth cases by introducing on-average argument stability. For minimax problems, we develop a quantitative connection between on-average argument stability and generalization error which extends the existing results for uniform stability (Lei et al., 2021). We further develop the first nearly optimal convergence rates for convex-concave problems both in expectation and with high probability, which, combined with our stability results, show that the optimal generalization bounds can be attained for both smooth and non-smooth cases. To the best of our knowledge, this is the first generalization analysis of SGMs when the gradients are sampled from a Markov process.

        ----

        ## [2735] Degradation-Aware Unfolding Half-Shuffle Transformer for Spectral Compressive Imaging

        **Authors**: *Yuanhao Cai, Jing Lin, Haoqian Wang, Xin Yuan, Henghui Ding, Yulun Zhang, Radu Timofte, Luc Van Gool*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f621c2ead473ca36763696b712ffda01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f621c2ead473ca36763696b712ffda01-Abstract-Conference.html)

        **Abstract**:

        In coded aperture snapshot spectral compressive imaging (CASSI) systems, hyperspectral image (HSI) reconstruction methods are employed to recover the spatial-spectral signal from a compressed measurement. Among these algorithms, deep unfolding methods demonstrate promising performance but suffer from two issues. Firstly, they do not estimate the degradation patterns and ill-posedness degree from CASSI to guide the iterative learning. Secondly, they are mainly CNN-based, showing limitations in capturing long-range dependencies. In this paper, we propose a principled Degradation-Aware Unfolding Framework (DAUF) that estimates parameters from the compressed image and physical mask, and then uses these parameters to control each iteration. Moreover, we customize a novel Half-Shuffle Transformer (HST) that simultaneously captures local contents and non-local dependencies. By plugging HST into DAUF, we establish the first Transformer-based deep unfolding method, Degradation-Aware Unfolding Half-Shuffle Transformer (DAUHST), for HSI reconstruction. Experiments show that DAUHST surpasses state-of-the-art methods while requiring cheaper computational and memory costs. Code and models are publicly available at https://github.com/caiyuanhao1998/MST

        ----

        ## [2736] Deep Surrogate Assisted Generation of Environments

        **Authors**: *Varun Bhatt, Bryon Tjanaka, Matthew C. Fontaine, Stefanos Nikolaidis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f649556471416b35e60ae0de7c1e3619-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f649556471416b35e60ae0de7c1e3619-Abstract-Conference.html)

        **Abstract**:

        Recent progress in reinforcement learning (RL) has started producing generally capable agents that can solve a distribution of complex environments. These agents are typically tested on fixed, human-authored environments. On the other hand, quality diversity (QD) optimization has been proven to be an effective component of environment generation algorithms, which can generate collections of high-quality environments that are diverse in the resulting agent behaviors. However, these algorithms require potentially expensive simulations of agents on newly generated environments. We propose Deep Surrogate Assisted Generation of Environments (DSAGE), a sample-efficient QD environment generation algorithm that maintains a deep surrogate model for predicting agent behaviors in new environments. Results in two benchmark domains show that DSAGE significantly outperforms existing QD environment generation algorithms in discovering collections of environments that elicit diverse behaviors of a state-of-the-art RL agent and a planning agent. Our source code and videos are available at https://dsagepaper.github.io/.

        ----

        ## [2737] Toward Equation of Motion for Deep Neural Networks: Continuous-time Gradient Descent and Discretization Error Analysis

        **Authors**: *Taiki Miyagawa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f6499ab2a923fa691accdc0077af9677-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f6499ab2a923fa691accdc0077af9677-Abstract-Conference.html)

        **Abstract**:

        We derive and solve an ``Equation of Motion'' (EoM) for deep neural networks (DNNs), a differential equation that precisely describes the discrete learning dynamics of DNNs. Differential equations are continuous but have played a prominent role even in the study of discrete optimization (gradient descent (GD) algorithms). However, there still exist gaps between differential equations and the actual learning dynamics of DNNs due to discretization error. In this paper, we start from gradient flow (GF) and derive a counter term that cancels the discretization error between GF and GD. As a result, we obtain EoM, a continuous differential equation that precisely describes the discrete learning dynamics of GD. We also derive discretization error to show to what extent EoM is precise. In addition, we apply EoM to two specific cases: scale- and translation-invariant layers. EoM highlights differences between continuous and discrete GD, indicating the importance of the counter term for a better description of the discrete learning dynamics of GD. Our experimental results support our theoretical findings.

        ----

        ## [2738] FLAIR: Federated Learning Annotated Image Repository

        **Authors**: *Congzheng Song, Filip Granqvist, Kunal Talwar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f64e55d03e2fe61aa4114e49cb654acb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/f64e55d03e2fe61aa4114e49cb654acb-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Cross-device federated learning is an emerging machine learning (ML) paradigm where a large population of devices collectively train an ML model while the data remains on the devices.This research field has a unique set of practical challenges, and to systematically make advances, new datasets curated to be compatible with this paradigm are needed.Existing federated learning benchmarks in the image domain do not accurately capture the scale and heterogeneity of many real-world use cases. We introduce FLAIR, a challenging large-scale annotated image dataset for multi-label classification suitable for federated learning.FLAIR has 429,078 images from  51,414  Flickr users and captures many of the intricacies typically encountered in federated learning, such as heterogeneous user data and a long-tailed label distribution.We implement multiple baselines in different learning setups for different tasks on this dataset. We believe FLAIR can serve as a challenging benchmark for advancing the state-of-the art in federated learning.Dataset access and the code for the benchmark are available at https://github.com/apple/ml-flair.

        ----

        ## [2739] Self-Supervised Pretraining for Large-Scale Point Clouds

        **Authors**: *Zaiwei Zhang, Min Bai, Li Erran Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f670ef96387d9a5a8a51e2ed80cb148d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f670ef96387d9a5a8a51e2ed80cb148d-Abstract-Conference.html)

        **Abstract**:

        Pretraining on large unlabeled datasets has been proven to improve the down-stream task performance on many computer vision tasks, such as 2D object detection and video classification. However, for large-scale 3D scenes, such as outdoor LiDAR point clouds, pretraining is not widely used. Due to the special data characteristics of large 3D point clouds, 2D pretraining frameworks tend to not generalize well. In this paper, we propose a new self-supervised pretraining method that targets large-scale 3D scenes. We pretrain commonly used point-based and voxel-based model architectures and show the transfer learning performance on 3D object detection and also semantic segmentation. We demonstrate the effectiveness of our approach on both dense 3D indoor point clouds and also sparse outdoor lidar point clouds.

        ----

        ## [2740] Vision Transformers provably learn spatial structure

        **Authors**: *Samy Jelassi, Michael E. Sander, Yuanzhi Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f69707de866eb0805683d3521756b73f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f69707de866eb0805683d3521756b73f-Abstract-Conference.html)

        **Abstract**:

        Vision Transformers (ViTs) have recently achieved comparable or superior performance to Convolutional neural networks (CNNs) in computer vision. This empirical breakthrough is even more remarkable since ViTs discards spatial information by mixing patch embeddings and positional encodings and do not embed any visual inductive bias (e.g.\ spatial locality). Yet, recent work showed that while minimizing their training loss, ViTs specifically learn spatially delocalized patterns. This raises a central question: how do ViTs learn this pattern by solely minimizing their training loss using gradient-based methods from \emph{random initialization}? We propose a structured classification dataset and a simplified ViT model to provide preliminary theoretical justification of this phenomenon. Our model relies on a simplified attention mechanism --the positional attention mechanism-- where the attention matrix solely depends on the positional encodings. While the problem admits multiple solutions that generalize, we show that our model implicitly learns the spatial structure of the dataset while generalizing. We finally prove that learning the structure helps to  sample-efficiently transfer to downstream datasets that share the same structure as the pre-training one but with different  features. We empirically verify that ViTs using only the positional attention mechanism perform similarly to the original one on CIFAR-10/100, SVHN and ImageNet.

        ----

        ## [2741] Unsupervised Learning of Shape Programs with Repeatable Implicit Parts

        **Authors**: *Boyang Deng, Sumith Kulal, Zhengyang Dong, Congyue Deng, Yonglong Tian, Jiajun Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f6adf61977467560f79b95485d1f3a79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f6adf61977467560f79b95485d1f3a79-Abstract-Conference.html)

        **Abstract**:

        Shape programs encode shape structures by representing object parts as subroutines and constructing the overall shape by composing these subroutines. This usually involves the reuse of subroutines for repeatable parts, enabling the modeling of correlations among shape elements such as geometric similarity. However, existing learning-based shape programs suffer from limited representation capacity, because they use coarse geometry representations such as geometric primitives and low-resolution voxel grids. Further, their training requires manually annotated ground-truth programs, which are expensive to attain. We address these limitations by proposing Shape Programs with Repeatable Implicit Parts (ProGRIP). Using implicit functions to represent parts, ProGRIP greatly boosts the representation capacity of shape programs while preserving the higher-level structure of repetitions and symmetry. Meanwhile, we free ProGRIP from any inaccessible supervised training via devising a matching-based unsupervised training objective. Our empirical studies show that ProGRIP outperforms existing structured representations in both shape reconstruction fidelity and segmentation accuracy of semantic parts.

        ----

        ## [2742] Detecting Abrupt Changes in Sequential Pairwise Comparison Data

        **Authors**: *Wanshan Li, Alessandro Rinaldo, Daren Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f6ccfa588d2a95bef5a3b101c02524c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f6ccfa588d2a95bef5a3b101c02524c9-Abstract-Conference.html)

        **Abstract**:

        The Bradley-Terry-Luce (BTL) model is a classic and very popular statistical approach for eliciting a global ranking among a collection of items using pairwise comparison data. In applications in which the comparison outcomes are observed as a time series, it is often the case that data are non-stationary, in the sense that the true underlying ranking changes over time. In this paper we are concerned with localizing the change points in a high-dimensional BTL model with piece-wise constant parameters. We propose novel and practicable algorithms based on dynamic programming that can consistently estimate the unknown locations of the change points. We provide consistency rates for our methodology that depend explicitly on the model parameters, the temporal spacing between two consecutive change points and the magnitude of the change. We corroborate our findings with extensive numerical experiments and a real-life example.

        ----

        ## [2743] Rethinking Resolution in the Context of Efficient Video Recognition

        **Authors**: *Chuofan Ma, Qiushan Guo, Yi Jiang, Ping Luo, Zehuan Yuan, Xiaojuan Qi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f723b0024f2b843572420b42312a9ed4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f723b0024f2b843572420b42312a9ed4-Abstract-Conference.html)

        **Abstract**:

        In this paper, we empirically study how to make the most of low-resolution frames for efficient video recognition. Existing methods mainly focus on developing compact networks or alleviating temporal redundancy of video inputs to increase efficiency, whereas compressing frame resolution has rarely been considered a promising solution. A major concern is the poor recognition accuracy on low-resolution frames. We thus start by analyzing the underlying causes of performance degradation on low-resolution frames. Our key finding is that the major cause of degradation is not information loss in the down-sampling process, but rather the mismatch between network architecture and input scale. Motivated by the success of knowledge distillation (KD), we propose to bridge the gap between network and input size via cross-resolution KD (ResKD). Our work shows that ResKD is a simple but effective method to boost recognition accuracy on low-resolution frames. Without bells and whistles, ResKD considerably surpasses all competitive methods in terms of efficiency and accuracy on four large-scale benchmark datasets, i.e., ActivityNet, FCVID, Mini-Kinetics, Something-Something V2. In addition, we extensively demonstrate its effectiveness over state-of-the-art architectures, i.e., 3D-CNNs and Video Transformers, and scalability towards super low-resolution frames. The results suggest ResKD can serve as a general inference acceleration method for state-of-the-art video recognition. Our code will be available at https://github.com/CVMI-Lab/ResKD.

        ----

        ## [2744] The Effects of Regularization and Data Augmentation are Class Dependent

        **Authors**: *Randall Balestriero, Léon Bottou, Yann LeCun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f73c04538a5e1cad40ba5586b4b517d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f73c04538a5e1cad40ba5586b4b517d3-Abstract-Conference.html)

        **Abstract**:

        Regularization is a fundamental technique to prevent over-fitting and to improve generalization performances by constraining a model's complexity. Current Deep Networks heavily rely on regularizers such as Data-Augmentation (DA) or weight-decay, and employ structural risk minimization, i.e. cross-validation, to select the optimal regularization hyper-parameters. In this study, we demonstrate that techniques such as DA or weight decay produce a model with a reduced complexity that is unfair across classes. The optimal amount of DA or weight decay found from cross-validation over all classes leads to disastrous model performances on some classes e.g. on Imagenet with a resnet50, the ``barn spider'' classification test accuracy falls from $68\%$ to $46\%$ only by introducing random crop DA during training. Even more surprising, such performance drop also appears when introducing uninformative regularization techniques such as weight decay. Those results demonstrate that our search for ever increasing generalization performance ---averaged over all classes and samples--- has left us with models and regularizers that silently sacrifice performances on some classes. This scenario can become dangerous when deploying a model on downstream tasks e.g. an Imagenet pre-trained resnet50 deployed on INaturalist sees its performances fall from $70\%$ to $30\%$ on class \#8889 when introducing random crop DA during the Imagenet pre-training phase. Those results demonstrate that finding a correct measure of a model's complexity without class-dependent preference remains an open research question.

        ----

        ## [2745] Learning to Mitigate AI Collusion on Economic Platforms

        **Authors**: *Gianluca Brero, Eric Mibuari, Nicolas Lepore, David C. Parkes*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f746974abd33c0015ca583a267dac1fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f746974abd33c0015ca583a267dac1fd-Abstract-Conference.html)

        **Abstract**:

        Algorithmic pricing on online e-commerce platforms raises the concern of tacit collusion, where reinforcement learning algorithms learn to set collusive prices in a decentralized manner and through nothing more than profit feedback. This raises the question as to whether collusive pricing can be prevented through the design of suitable "buy boxes," i.e., through the design of the rules that govern the elements of e-commerce sites that promote particular products and prices to consumers. In this paper, we demonstrate that reinforcement learning (RL) can also be used by platforms to learn buy box rules that are effective in preventing collusion by RL sellers. For this, we adopt the methodology of Stackelberg POMDPs, and demonstrate success in learning robust rules that continue to provide high consumer welfare together with sellers employing different behavior models or having out-of-distribution costs for goods.

        ----

        ## [2746] The Query Complexity of Cake Cutting

        **Authors**: *Simina Brânzei, Noam Nisan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7a7bb369e48f10e85fce85b67d8c516-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7a7bb369e48f10e85fce85b67d8c516-Abstract-Conference.html)

        **Abstract**:

        We consider the query complexity of cake cutting in the standard query model and give lower and upper bounds for computing approximately envy-free, perfect, and equitable allocations with the minimum number of cuts. The lower bounds are tight for computing contiguous  envy-free allocations among $n=3$ players and for computing perfect and equitable allocations with minimum number of cuts between $n=2$ players. For $\epsilon$-envy-free  allocations with contiguous pieces, we also give an upper bound of $O(n/\epsilon)$ and  lower bound of $\Omega(\log(1/\epsilon))$ queries for any number $n \geq 3$ of players.We also formalize moving knife procedures and show that a large subclass of this family, which captures all the known moving knife procedures, can be simulated efficiently with arbitrarily small error in the Robertson-Webb query model.

        ----

        ## [2747] PAC Prediction Sets for Meta-Learning

        **Authors**: *Sangdon Park, Edgar Dobriban, Insup Lee, Osbert Bastani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7bc3ee2dade037a4d2f9e85f4519370-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7bc3ee2dade037a4d2f9e85f4519370-Abstract-Conference.html)

        **Abstract**:

        Uncertainty quantification is a key component of machine learning models targeted at safety-critical systems such as in healthcare or autonomous vehicles. We study this problem in the context of meta learning, where the goal is to quickly adapt a predictor to new tasks. In particular, we propose a novel algorithm to construct \emph{PAC prediction sets}, which capture uncertainty via sets of labels, that can be adapted to new tasks with only a few training examples. These prediction sets satisfy an extension of the typical PAC guarantee to the meta learning setting; in particular, the PAC guarantee holds with high probability over future tasks. We demonstrate the efficacy of our approach on four datasets across three application domains: mini-ImageNet and CIFAR10-C in the visual domain, FewRel in the language domain, and the CDC Heart Dataset in the medical domain. In particular, our prediction sets satisfy the PAC guarantee while having smaller size compared to other baselines that also satisfy this guarantee.

        ----

        ## [2748] High-dimensional Asymptotics of Feature Learning: How One Gradient Step Improves the Representation

        **Authors**: *Jimmy Ba, Murat A. Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, Greg Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7e7fabd73b3df96c54a320862afcb78-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7e7fabd73b3df96c54a320862afcb78-Abstract-Conference.html)

        **Abstract**:

        We study the first gradient descent step on the first-layer parameters $\boldsymbol{W}$ in a two-layer neural network: $f(\boldsymbol{x}) = \frac{1}{\sqrt{N}}\boldsymbol{a}^\top\sigma(\boldsymbol{W}^\top\boldsymbol{x})$, where $\boldsymbol{W}\in\mathbb{R}^{d\times N}, \boldsymbol{a}\in\mathbb{R}^{N}$ are randomly initialized, and the training objective is the empirical MSE loss: $\frac{1}{n}\sum_{i=1}^n (f(\boldsymbol{x}_i)-y_i)^2$. In the proportional asymptotic limit where $n,d,N\to\infty$ at the same rate, and an idealized student-teacher setting where the teacher $f^*$ is a single-index model, we compute the prediction risk of ridge regression on the conjugate kernel after one gradient step on $\boldsymbol{W}$ with learning rate $\eta$. We consider two scalings of the first step learning rate $\eta$. For small $\eta$, we establish a Gaussian equivalence property for the trained feature map, and prove that the learned kernel improves upon the initial random features model, but cannot defeat the best linear model on the input. Whereas for sufficiently large $\eta$, we prove that for certain $f^*$, the same ridge estimator on trained features can go beyond this ``linear regime'' and outperform a wide range of (fixed) kernels. Our results demonstrate that even one gradient step can lead to a considerable advantage over random features, and highlight the role of learning rate scaling in the initial phase of training.

        ----

        ## [2749] Pruning's Effect on Generalization Through the Lens of Training and Regularization

        **Authors**: *Tian Jin, Michael Carbin, Daniel M. Roy, Jonathan Frankle, Gintare Karolina Dziugaite*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7ede9414083fceab9e63d9100a80b36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7ede9414083fceab9e63d9100a80b36-Abstract-Conference.html)

        **Abstract**:

        Practitioners frequently observe that pruning improves model generalization. A long-standing hypothesis based on bias-variance trade-off attributes this generalization improvement to model size reduction. However, recent studies on over-parameterization characterize a new model size regime, in which larger models achieve better generalization. Pruning models in this over-parameterized regime leads to a contradiction -- while theory predicts that reducing model size harms generalization, pruning to a range of sparsities nonetheless improves it. Motivated by this contradiction, we re-examine pruning’s effect on generalization empirically.We show that size reduction cannot fully account for the generalization-improving effect of standard pruning algorithms. Instead, we find that pruning leads to better training at specific sparsities, improving the training loss over the dense model. We find that pruning also leads to additional regularization at other sparsities, reducing the accuracy degradation due to noisy examples over the dense model. Pruning extends model training time and reduces model size. These two factors improve training and add regularization respectively. We empirically demonstrate that both factors are essential to fully explaining pruning's impact on generalization.

        ----

        ## [2750] Cluster Randomized Designs for One-Sided Bipartite Experiments

        **Authors**: *Jennifer Brennan, Vahab Mirrokni, Jean Pouget-Abadie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7f043c3438ba9e385c51bcf50ed007e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7f043c3438ba9e385c51bcf50ed007e-Abstract-Conference.html)

        **Abstract**:

        The conclusions of randomized controlled trials may be biased when the outcome of one unit depends on the treatment status of other units, a problem known as \textit{interference}. In this work, we study interference in the setting of one-sided bipartite experiments in which the experimental units---where treatments are randomized and outcomes are measured---do not interact directly. Instead, their interactions are mediated through their connections to \textit{interference units} on the other side of the graph. Examples of this type of interference are common in marketplaces and two-sided platforms. The \textit{cluster-randomized design} is a popular method to mitigate interference when the graph is known, but it has not been well-studied in the one-sided bipartite experiment setting. In this work, we formalize a natural model for interference in one-sided bipartite experiments using the exposure mapping framework. We first exhibit settings under which existing cluster-randomized designs fail to properly mitigate interference under this model. We then show that minimizing the bias of the difference-in-means estimator under our model results in a balanced partitioning clustering objective with a natural interpretation. We further prove that our design is minimax optimal over the class of linear potential outcomes models with bounded interference. We conclude by providing theoretical and experimental evidence of the robustness of our design to a variety of interference graphs and potential outcomes models.

        ----

        ## [2751] Deep Equilibrium Approaches to Diffusion Models

        **Authors**: *Ashwini Pokle, Zhengyang Geng, J. Zico Kolter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7f47a73d631c0410cbc2748a8015241-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7f47a73d631c0410cbc2748a8015241-Abstract-Conference.html)

        **Abstract**:

        Diffusion-based generative models are extremely effective in generating high-quality images, with generated samples often surpassing the quality of those produced by other models under several metrics.  One distinguishing feature of these models, however, is that they typically require long sampling chains in order to produce high-fidelity images.  This presents a challenge not only from the lenses of sampling time, but also from the inherent difficulty in backpropagating through these chains in order to accomplish tasks such as model inversion, i.e., approximately finding latent states that generate known images.  In this paper, we look at diffusion models through a different perspective, that of a (deep) equilibrium (DEQ) fixed point model. Specifically, we extend the recent denoising diffusion implicit model (DDIM), and model the entire sampling chain as a joint, multi-variate fixed point system. This setup provides an elegant unification of diffusion and equilibrium models, and shows benefits in 1) single-shot image sampling, as it replaces the fully-serial typical sampling process with a parallel one; and 2) model inversion, where we can leverage fast gradients in the DEQ setting to much more quickly find the noise that generates a given image.  The approach is also orthogonal and thus complementary to other methods used to reduce the sampling time, or improve model inversion.  We demonstrate our method's strong performance across several datasets, including CIFAR10, CelebA, and LSUN Bedroom and Churches.

        ----

        ## [2752] Inducing Neural Collapse in Imbalanced Learning: Do We Really Need a Learnable Classifier at the End of Deep Neural Network?

        **Authors**: *Yibo Yang, Shixiang Chen, Xiangtai Li, Liang Xie, Zhouchen Lin, Dacheng Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7f5f501282771c96bb3fedcc96bedfe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7f5f501282771c96bb3fedcc96bedfe-Abstract-Conference.html)

        **Abstract**:

        Modern deep neural networks for classification usually jointly learn a backbone for representation and a linear classifier to output the logit of each class. A recent study has shown a phenomenon called neural collapse that the within-class means of features and the classifier vectors converge to the vertices of a simplex equiangular tight frame (ETF) at the terminal phase of training on a balanced dataset. Since the ETF geometric structure maximally separates the pair-wise angles of all classes in the classifier, it is natural to raise the question, why do we spend an effort to learn a classifier when we know its optimal geometric structure? In this paper, we study the potential of learning a neural network for classification with the classifier randomly initialized as an ETF and fixed during training. Our analytical work based on the layer-peeled model indicates that the feature learning with a fixed ETF classifier naturally leads to the neural collapse state even when the dataset is imbalanced among classes. We further show that in this case the cross entropy (CE) loss is not necessary and can be replaced by a simple squared loss that shares the same global optimality but enjoys a better convergence property. Our experimental results show that our method is able to bring significant improvements with faster convergence on multiple imbalanced datasets.

        ----

        ## [2753] Pruning Neural Networks via Coresets and Convex Geometry: Towards No Assumptions

        **Authors**: *Murad Tukan, Loay Mualem, Alaa Maalouf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7fc38fdd95fd146a471791b93ff9f12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7fc38fdd95fd146a471791b93ff9f12-Abstract-Conference.html)

        **Abstract**:

        Pruning is one of the predominant approaches for compressing deep neural networks (DNNs). Lately, coresets (provable data summarizations) were leveraged for pruning DNNs, adding the advantage of theoretical guarantees on the trade-off between the compression rate and the approximation error. However, coresets in this domain were either data dependant or generated under restrictive assumptions on both the model's weights and inputs. In real-world scenarios, such assumptions are rarely satisfied, limiting the applicability of coresets. To this end, we suggest a novel and robust framework for computing such coresets under mild assumptions on the model's weights and without any assumption on the training data. The idea is to compute the importance of each neuron in each layer with respect to the output of the following layer. This is achieved by an elegant combination of L\"{o}wner ellipsoid and Caratheodory theorem.Our method is simultaneously data-independent, applicable to various networks and datasets (due to the simplified assumptions), and theoretically supported. Experimental results show that our method outperforms existing coreset based neural pruning approaches across a wide range of networks and datasets. For example, our method achieved a $62\%$ compression rate on ResNet50 on ImageNet with $1.09\%$ drop in accuracy.

        ----

        ## [2754] Intermediate Prototype Mining Transformer for Few-Shot Semantic Segmentation

        **Authors**: *Yuanwei Liu, Nian Liu, Xiwen Yao, Junwei Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f7fef21d1fb3e950b12b50ad7f395e31-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f7fef21d1fb3e950b12b50ad7f395e31-Abstract-Conference.html)

        **Abstract**:

        Few-shot semantic segmentation aims to segment the target objects in query under the condition of a few annotated support images. Most previous works strive to mine more effective category information from the support to match with the corresponding objects in query. However, they all ignored the category information gap between query and support images. If the objects in them show large intra-class diversity, forcibly migrating the category information from the support to the query is ineffective. To solve this problem, we are the first to introduce an intermediate prototype for mining both deterministic category information from the support and adaptive category knowledge from the query. Specifically, we design an Intermediate Prototype Mining Transformer (IPMT) to learn the prototype in an iterative way. In each IPMT layer, we propagate the object information in both support and query features to the prototype and then use it to activate the query feature map. By conducting this process iteratively, both the intermediate prototype and the query feature can be progressively improved. At last, the final query feature is used to yield precise segmentation prediction. Extensive experiments on both PASCAL-5i and COCO-20i datasets clearly verify the effectiveness of our IPMT and show that it outperforms previous state-of-the-art methods by a large margin. Code is available at https://github.com/LIUYUANWEI98/IPMT

        ----

        ## [2755] Long-Form Video-Language Pre-Training with Multimodal Temporal Contrastive Learning

        **Authors**: *Yuchong Sun, Hongwei Xue, Ruihua Song, Bei Liu, Huan Yang, Jianlong Fu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f8290ccc2905538be1a7f7914ccef629-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f8290ccc2905538be1a7f7914ccef629-Abstract-Conference.html)

        **Abstract**:

        Large-scale video-language pre-training has shown significant improvement in video-language understanding tasks. Previous studies of video-language pretraining mainly focus on short-form videos (i.e., within 30 seconds) and sentences, leaving long-form video-language pre-training rarely explored. Directly learning representation from long-form videos and language may benefit many long-formvideo-language understanding tasks. However, it is challenging due to the difficulty of modeling long-range relationships and the heavy computational burden caused by more frames. In this paper, we introduce a Long-Form VIdeo-LAnguage pre-training model (LF-VILA) and train it on a large-scale long-form video and paragraph dataset constructed from an existing public dataset. To effectively capturethe rich temporal dynamics and to better align video and language in an efficient end-to-end manner, we introduce two novel designs in our LF-VILA model. We first propose a Multimodal Temporal Contrastive (MTC) loss to learn the temporal relation across different modalities by encouraging fine-grained alignment between long-form videos and paragraphs. Second, we propose a Hierarchical Temporal Window Attention (HTWA) mechanism to effectively capture long-range dependency while reducing computational cost in Transformer. We fine-tune the pre-trained LF-VILA model on seven downstream long-form video-language understanding tasks of paragraph-to-video retrieval and long-form video question-answering, and achieve new state-of-the-art performances. Specifically, our model achieves 16.1% relative improvement on ActivityNet paragraph-to-video retrieval task and 2.4% on How2QA task, respectively. We release our code, dataset, and pre-trained models at https://github.com/microsoft/XPretrain.

        ----

        ## [2756] A new dataset for multilingual keyphrase generation

        **Authors**: *Frédéric Piedboeuf, Philippe Langlais*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f88709551258331f9ab31b33c71021a4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/f88709551258331f9ab31b33c71021a4-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Keyphrases  are an important tool for efficiently dealing with the ever-increasing amount of information present on the internet. While there are many recent papers on English keyphrase generation, keyphrase generation for other languages remains vastly understudied, mostly due to the absence of datasets. To address this, we present a novel dataset called Papyrus, composed of 16427 pairs of abstracts and keyphrases. We release four versions of this dataset, corresponding to different subtasks. Papyrus-e considers only English keyphrases, Papyrus-f considers French keyphrases, Papyrus-m considers keyphrase generation in any language (mostly French and English), and Papyrus-a considers keyphrase generation in several languages. We train a state-of-the-art model on all four tasks and show that they lead to better results for non-English languages, with an average improvement of 14.2\% on keyphrase extraction and 2.0\% on generation. We also show an improvement of 0.4\% on extraction and 0.7\% on generation over English state-of-the-art results by concatenating Papyrus-e with the Kp20K training set.

        ----

        ## [2757] Model Preserving Compression for Neural Networks

        **Authors**: *Jerry Chee, Megan Flynn, Anil Damle, Christopher De Sa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f8928b073ccbec15d35f2a9d39430bfd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f8928b073ccbec15d35f2a9d39430bfd-Abstract-Conference.html)

        **Abstract**:

        After training complex deep learning models, a common task is to compress the model to reduce compute and storage demands. When compressing, it is desirable to preserve the original model's per-example decisions (e.g., to go beyond top-1 accuracy or preserve robustness), maintain the network's structure, automatically determine per-layer compression levels, and eliminate the need for fine tuning. No existing compression methods simultaneously satisfy these criteria---we introduce a principled approach that does by leveraging interpolative decompositions. Our approach simultaneously selects and eliminates channels (analogously, neurons), then constructs an interpolation matrix that propagates a correction into the next layer, preserving the network's structure. Consequently, our method achieves good performance even without fine tuning and admits theoretical analysis. Our theoretical generalization bound for a one layer network lends itself naturally to a heuristic that allows our method to automatically choose per-layer sizes for deep networks. We demonstrate the efficacy of our approach with strong empirical performance on a variety of tasks, models, and datasets---from simple one-hidden-layer networks to deep networks on ImageNet.

        ----

        ## [2758] Neural Conservation Laws: A Divergence-Free Perspective

        **Authors**: *Jack Richter-Powell, Yaron Lipman, Ricky T. Q. Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f8d39584f87944e5dbe46ec76f19e20a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f8d39584f87944e5dbe46ec76f19e20a-Abstract-Conference.html)

        **Abstract**:

        We investigate the parameterization of deep neural networks that by design satisfy the continuity equation, a fundamental conservation law. This is enabled by the observation that any solution of the continuity equation can be represented as a divergence-free vector field. We hence propose building divergence-free neural networks through the concept of differential forms, and with the aid of automatic differentiation, realize two practical constructions. As a result, we can parameterize pairs of densities and vector fields that always satisfy the continuity equation by construction, foregoing the need for extra penalty methods or expensive numerical simulation. Furthermore, we prove these models are universal and so can be used to represent any divergence-free vector field. Finally, we experimentally validate our approaches by computing neural network-based solutions to fluid equations, solving for the Hodge decomposition, and learning dynamical optimal transport maps.

        ----

        ## [2759] Towards Effective Multi-Modal Interchanges in Zero-Resource Sounding Object Localization

        **Authors**: *Yang Zhao, Chen Zhang, Haifeng Huang, Haoyuan Li, Zhou Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f8de10c9ff056ae3d1eef43ad1762351-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f8de10c9ff056ae3d1eef43ad1762351-Abstract-Conference.html)

        **Abstract**:

        Aiming to locate the object that emits a specified sound in complex scenes, the task of sounding object localization bridges two perception-oriented modalities of vision and acoustics, and brings enormous research value to the comprehensive perceptual understanding of machine intelligence. Although there are massive training data collected in this field, few of them contain accurate bounding box annotations, hindering the learning process and further application of proposed models. In order to address this problem, we try to explore an effective multi-modal knowledge transfer strategy to obtain precise knowledge from other similar tasks and transfer it through well-aligned multi-modal data to deal with this task in a zero-resource manner. Concretely, we design and propose a novel \textit{Two-stream Universal Referring localization Network} (TURN), which is composed of a localization stream and an alignment stream to carry out different functions. The former is utilized to extract the knowledge related to referring object localization from the image grounding task, while the latter is devised to learn a universal semantic space shared between texts and audios. Moreover, we further develop an adaptive sampling strategy to automatically identify the overlap between different data domains, thus boosting the performance and stability of our model. The extensive experiments on various publicly-available benchmarks demonstrate that TURN can achieve competitive performance compared with the state-of-the-art approaches without using any data in this field, which verifies the feasibility of our proposed mechanisms and strategies.

        ----

        ## [2760] On the Convergence of Stochastic Multi-Objective Gradient Manipulation and Beyond

        **Authors**: *Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, Wenwu Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract-Conference.html)

        **Abstract**:

        The conflicting gradients problem is one of the major bottlenecks for the effective training of machine learning models that deal with multiple objectives. To resolve this problem, various gradient manipulation techniques, such as PCGrad, MGDA, and CAGrad, have been developed, which directly alter the conflicting gradients to refined ones with alleviated or even no conflicts. However, the existing design and analysis of these techniques are mainly conducted under the full-batch gradient setting, ignoring the fact that they are primarily applied with stochastic mini-batch gradients. In this paper, we illustrate that the stochastic gradient manipulation algorithms may fail to converge to Pareto optimal solutions. Firstly, we show that these different algorithms can be summarized into a unified algorithmic framework, where the descent direction is given by the composition of the gradients of the multiple objectives. Then we provide an explicit two-objective convex optimization instance to explicate the non-convergence issue under the unified framework, which suggests that the non-convergence results from the determination of the composite weights solely by the instantaneous stochastic gradients. To fix the non-convergence issue, we propose a novel composite weights determination scheme that exponentially averages the past calculated weights. Finally, we show the resulting new variant of stochastic gradient manipulation converges to Pareto optimal or critical solutions and yield comparable or improved empirical performance.

        ----

        ## [2761] Decentralized Local Stochastic Extra-Gradient for Variational Inequalities

        **Authors**: *Aleksandr Beznosikov, Pavel E. Dvurechensky, Anastasia Koloskova, Valentin Samokhin, Sebastian U. Stich, Alexander V. Gasnikov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f9379afacdbabfdc6b060972b60f9ab8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f9379afacdbabfdc6b060972b60f9ab8-Abstract-Conference.html)

        **Abstract**:

        We consider distributed stochastic variational inequalities (VIs) on unbounded domains with the problem data that is heterogeneous (non-IID) and distributed across many devices. We make a very general assumption on the computational network that, in particular, covers the settings of fully decentralized calculations with time-varying networks and centralized topologies commonly used in Federated Learning. Moreover, multiple local updates on the workers can be made for reducing the communication frequency between the workers.We extend the stochastic extragradient method to this very general setting and theoretically analyze its convergence rate in the strongly-monotone, monotone, and non-monotone (when a Minty solution exists) settings. The provided rates explicitly exhibit the dependence on network characteristics (e.g., mixing time), iteration counter, data heterogeneity, variance, number of devices, and other standard parameters. As a special case, our method and analysis apply to distributed stochastic saddle-point problems (SPP), e.g., to the training of Deep Generative Adversarial Networks (GANs) for which decentralized training has been reported to be extremely challenging. In experiments for the decentralized training of GANs we demonstrate the effectiveness of our proposed approach.

        ----

        ## [2762] Model Zoos: A Dataset of Diverse Populations of Neural Network Models

        **Authors**: *Konstantin Schürholt, Diyar Taskiran, Boris Knyazev, Xavier Giró-i-Nieto, Damian Borth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f94d5edb5c01715d879693ddbfdc1b98-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/f94d5edb5c01715d879693ddbfdc1b98-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In the last years, neural networks (NN) have evolved from laboratory environments to the state-of-the-art for many real-world problems. It was shown that NN models (i.e., their weights and biases) evolve on unique trajectories in weight space during training. Following, a population of such neural network models (referred to as model zoo) would form structures in weight space. We think that the geometry, curvature and smoothness of these structures contain information about the state of training and can reveal latent properties of individual models. With such model zoos, one could investigate novel approaches for (i) model analysis, (ii) discover unknown learning dynamics, (iii) learn rich representations of such populations, or (iv) exploit the model zoos for generative modelling of NN weights and biases. Unfortunately, the lack of standardized model zoos and available benchmarks significantly increases the friction for further research about populations of NNs. With this work, we publish a novel dataset of model zoos containing systematically generated and diverse populations of NN models for further research. In total the proposed model zoo dataset is based on eight image datasets, consists of 27 model zoos trained with varying hyperparameter combinations and includes 50’360 unique NN models as well as their sparsified twins, resulting in over 3’844’360 collected model states. Additionally, to the model zoo data we provide an in-depth analysis of the zoos and provide benchmarks for multiple downstream tasks. The dataset can be found at www.modelzoos.cc.

        ----

        ## [2763] Weakly-Supervised Multi-Granularity Map Learning for Vision-and-Language Navigation

        **Authors**: *Peihao Chen, Dongyu Ji, Kunyang Lin, Runhao Zeng, Thomas H. Li, Mingkui Tan, Chuang Gan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f959b05dd74ba8a735276c3df4ae8b71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f959b05dd74ba8a735276c3df4ae8b71-Abstract-Conference.html)

        **Abstract**:

        We address a practical yet challenging problem of training robot agents to navigate in an environment following a path described by some language instructions. The instructions often contain descriptions of objects in the environment. To achieve accurate and efficient navigation, it is critical to build a map that accurately represents both spatial location and the semantic information of the environment objects. However, enabling a robot to build a map that well represents the environment is extremely challenging as the environment often involves diverse objects with various attributes. In this paper, we propose a multi-granularity map, which contains both object fine-grained details (\eg, color, texture) and semantic classes, to represent objects more comprehensively. Moreover, we propose a weakly-supervised auxiliary task, which requires the agent to localize instruction-relevant objects on the map. Through this task, the agent not only learns to localize the instruction-relevant objects for navigation but also is encouraged to learn a better map representation that reveals object information. We then feed the learned map and instruction to a waypoint predictor to determine the next navigation goal. Experimental results show our method outperforms the state-of-the-art by 4.0% and 4.6% w.r.t. success rate both in seen and unseen environments, respectively on VLN-CE dataset. The code is available at https://github.com/PeihaoChen/WS-MGMap.

        ----

        ## [2764] Thinned random measures for sparse graphs with overlapping communities

        **Authors**: *Federica Zoe Ricci, Michele Guindani, Erik Sudderth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f9668d223e713943634dce9c66e8f2c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f9668d223e713943634dce9c66e8f2c1-Abstract-Conference.html)

        **Abstract**:

        Network models for exchangeable arrays, including most stochastic block models, generate dense graphs with a limited ability to capture many characteristics of real-world social and biological networks. A class of models based on completely random measures like the generalized gamma process (GGP) have recently addressed some of these limitations. We propose a framework for thinning edges from realizations of GGP random graphs that models observed links via nodes' overall propensity to interact, as well as the similarity of node memberships within a large set of latent communities. Our formulation allows us to learn the number of communities from data, and enables efficient Monte Carlo methods that scale linearly with the number of observed edges, and thus (unlike dense block models) sub-quadratically with the number of entities or nodes. We compare to alternative models for both dense and sparse networks, and demonstrate effective recovery of latent community structure for real-world networks with thousands of nodes.

        ----

        ## [2765] Fine-tuning language models to find agreement among humans with diverse preferences

        **Authors**: *Michiel A. Bakker, Martin J. Chadwick, Hannah Sheahan, Michael Henry Tessler, Lucy Campbell-Gillingham, Jan Balaguer, Nat McAleese, Amelia Glaese, John Aslanides, Matt M. Botvinick, Christopher Summerfield*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f978c8f3b5f399cae464e85f72e28503-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f978c8f3b5f399cae464e85f72e28503-Abstract-Conference.html)

        **Abstract**:

        Recent work in large language modeling (LLMs) has used fine-tuning to align outputs with the preferences of a prototypical user. This work assumes that human preferences are static and homogeneous across individuals, so that aligning to a single "generic" user will confer more general alignment. Here, we embrace the heterogeneity of human preferences to consider a different challenge: how might a machine help people with diverse views find agreement? We fine-tune a 70 billion parameter LLM to generate statements that maximize the expected approval for a group of people with potentially diverse opinions. Human participants provide written opinions on thousands of questions touching on moral and political issues (e.g., "should we raise taxes on the rich?"), and rate the LLM's generated candidate consensus statements for agreement and quality. A reward model is then trained to predict individual preferences, enabling it to quantify and rank consensus statements in terms of their appeal to the overall group, defined according to different aggregation (social welfare) functions. The model produces consensus statements that are preferred by human users over those from prompted LLMs ($>70\%$) and significantly outperforms a tight fine-tuned baseline that lacks the final ranking step. Further, our best model's consensus statements are preferred over the best human-generated opinions ($>65\%$). We find that when we silently constructed consensus statements from only a subset of group members, those who were excluded were more likely to dissent, revealing the sensitivity of the consensus to individual contributions. These results highlight the potential to use LLMs to help groups of humans align their values with one another.

        ----

        ## [2766] What is a Good Metric to Study Generalization of Minimax Learners?

        **Authors**: *Asuman E. Ozdaglar, Sarath Pattathil, Jiawei Zhang, Kaiqing Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f9b8853ea81731f9bfc11820b064de96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f9b8853ea81731f9bfc11820b064de96-Abstract-Conference.html)

        **Abstract**:

        Minimax optimization has served as the backbone of many machine learning problems. Although the convergence behavior of optimization algorithms has been extensively studied in minimax settings, their generalization guarantees, i.e., how the model trained on empirical data performs on the unseen testing data, have been relatively under-explored. A fundamental question remains elusive: What is a good metric to study generalization of minimax learners? In this paper, we aim to answer this question by first showing that primal risk, a universal metric to study generalization in minimization problems, fails in simple examples of minimax problems. Furthermore, another popular metric, the primal-dual risk, also fails to characterize the generalization behavior for minimax problems with nonconvexity, due to non-existence of saddle points. We thus propose a new metric to study generalization of minimax learners: the primal gap, to circumvent these issues. Next, we derive generalization bounds for the primal gap in nonconvex-concave settings. As byproducts of our analysis, we also solve two open questions: establishing generalization bounds for primal risk and primal-dual risk in this setting, and in the strong sense, i.e., without assuming that the maximization and expectation can be interchanged. Finally, we leverage this new metric to compare the generalization behavior of two popular algorithms - gradient descent-ascent (GDA) and gradient descent-max (GDMax) in minimax optimization.

        ----

        ## [2767] Sequencer: Deep LSTM for Image Classification

        **Authors**: *Yuki Tatsunami, Masato Taki*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f9d7d6c695bc983fcfb5b70a5fbdfd2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f9d7d6c695bc983fcfb5b70a5fbdfd2f-Abstract-Conference.html)

        **Abstract**:

        In recent computer vision research, the advent of the Vision Transformer (ViT) has rapidly revolutionized various architectural design efforts: ViT achieved state-of-the-art image classification performance using self-attention found in natural language processing, and MLP-Mixer achieved competitive performance using simple multi-layer perceptrons. In contrast, several studies have also suggested that carefully redesigned convolutional neural networks (CNNs) can achieve advanced performance comparable to ViT without resorting to these new ideas. Against this background, there is growing interest in what inductive bias is suitable for computer vision. Here we propose Sequencer, a novel and competitive architecture alternative to ViT that provides a new perspective on these issues. Unlike ViTs, Sequencer models long-range dependencies using LSTMs rather than self-attention layers. We also propose a two-dimensional version of Sequencer module, where an LSTM is decomposed into vertical and horizontal LSTMs to enhance performance. Despite its simplicity, several experiments demonstrate that Sequencer performs impressively well: Sequencer2D-L, with 54M parameters, realizes 84.6% top-1 accuracy on only ImageNet-1K. Not only that, we show that it has good transferability and the robust resolution adaptability on double resolution-band. solution-band. Our source code is available at https://github.com/okojoalg/sequencer.

        ----

        ## [2768] Double Check Your State Before Trusting It: Confidence-Aware Bidirectional Offline Model-Based Imagination

        **Authors**: *Jiafei Lyu, Xiu Li, Zongqing Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f9e2800a251fa9107a008104f47c45d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f9e2800a251fa9107a008104f47c45d1-Abstract-Conference.html)

        **Abstract**:

        The learned policy of model-free offline reinforcement learning (RL) methods is often constrained to stay within the support of datasets to avoid possible dangerous out-of-distribution actions or states, making it challenging to handle out-of-support region. Model-based RL methods offer a richer dataset and benefit generalization by generating imaginary trajectories with either trained forward or reverse dynamics model. However, the imagined transitions may be inaccurate, thus downgrading the performance of the underlying offline RL method. In this paper, we propose to augment the offline dataset by using trained bidirectional dynamics models and rollout policies with double check. We introduce conservatism by trusting samples that the forward model and backward model agree on. Our method, confidence-aware bidirectional offline model-based imagination, generates reliable samples and can be combined with any model-free offline RL method. Experimental results on the D4RL benchmarks demonstrate that our method significantly boosts the performance of existing model-free offline RL algorithms and achieves competitive or better scores against baseline methods.

        ----

        ## [2769] Learning on Arbitrary Graph Topologies via Predictive Coding

        **Authors**: *Tommaso Salvatori, Luca Pinchetti, Beren Millidge, Yuhang Song, Tianyi Bao, Rafal Bogacz, Thomas Lukasiewicz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f9f54762cbb4fe4dbffdd4f792c31221-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/f9f54762cbb4fe4dbffdd4f792c31221-Abstract-Conference.html)

        **Abstract**:

        Training with backpropagation (BP) in standard deep learning consists of two main steps: a forward pass that maps a data point to its prediction, and a backward pass that propagates the error of this prediction back through the network. This process is highly effective when the goal is to minimize a specific objective function. However, it does not allow training on networks with cyclic or backward connections. This is an obstacle to reaching brain-like capabilities, as the highly complex heterarchical structure of the neural connections in the neocortex are potentially fundamental for its effectiveness. In this paper, we show how predictive coding (PC), a theory of information processing in the cortex, can be used to perform inference and learning on arbitrary graph topologies. We experimentally show how this formulation, called PC graphs, can be used to flexibly perform different tasks with the same network by simply stimulating specific neurons. This enables the model to be queried on stimuli with different structures, such as partial images, images with labels, or images without labels. We conclude by investigating how the topology of the graph influences the final performance, and comparing against simple baselines trained with BP.

        ----

        ## [2770] Robustness Disparities in Face Detection

        **Authors**: *Samuel Dooley, George Z. Wei, Tom Goldstein, John Dickerson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/f9faef4e1b4dbbd48ef60056ffe14c90-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/f9faef4e1b4dbbd48ef60056ffe14c90-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Facial analysis systems have been deployed by large companies and critiqued by scholars and activists for the past decade. Many existing algorithmic audits examine the performance of these systems on later stage elements of facial analysis systems like facial recognition and age, emotion, or perceived gender prediction; however, a core component to these systems has been vastly understudied from a fairness perspective: face detection, sometimes called face localization. Since face detection is a pre-requisite step in facial analysis systems, the bias we observe in face detection will flow downstream to the other components like facial recognition and emotion prediction. Additionally, no prior work has focused on the robustness of these systems under various perturbations and corruptions, which leaves open the question of how various people are impacted by these phenomena. We present the first of its kind detailed benchmark of face detection systems, specifically examining the robustness to noise of commercial and academic models. We use both standard and recently released academic facial datasets to quantitatively analyze trends in face detection robustness. Across all the datasets and systems, we generally find that photos of individuals who are masculine presenting, older, of darker skin type, or have dim lighting are more susceptible to errors than their counterparts in other identities.

        ----

        ## [2771] Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class

        **Authors**: *Khoa D. Doan, Yingjie Lao, Ping Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa0126bb7ebad258bf4ffdbbac2dd787-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa0126bb7ebad258bf4ffdbbac2dd787-Abstract-Conference.html)

        **Abstract**:

        In recent years, machine learning models have been shown to be vulnerable to backdoor attacks. Under such attacks, an adversary embeds a stealthy backdoor into the trained model such that the compromised models will behave normally on clean inputs but will misclassify according to the adversary's control on maliciously constructed input with a trigger. While these existing attacks are very effective, the adversary's capability is limited: given an input, these attacks can only cause the model to misclassify toward a single pre-defined or target class. In contrast, this paper exploits a novel backdoor attack with a much more powerful payload, denoted as Marksman, where the adversary can arbitrarily choose which target class the model will misclassify given any input during inference. To achieve this goal, we propose to represent the trigger function as a class-conditional generative model and to inject the backdoor in a constrained optimization framework, where the trigger function learns to generate an optimal trigger pattern to attack any target class at will while simultaneously embedding this generative backdoor into the trained model. Given the learned trigger-generation function, during inference, the adversary can specify an arbitrary backdoor attack target class, and an appropriate trigger causing the model to classify toward this target class is created accordingly. We show empirically that the proposed framework achieves high attack performance (e.g., 100% attack success rates in several experiments) while preserving the clean-data performance in several benchmark datasets, including MNIST, CIFAR10, GTSRB, and TinyImageNet. The proposed Marksman backdoor attack can also easily bypass existing backdoor defenses that were originally designed against backdoor attacks with a single target class. Our work takes another significant step toward understanding the extensive risks of backdoor attacks in practice.

        ----

        ## [2772] Memorization Without Overfitting: Analyzing the Training Dynamics of Large Language Models

        **Authors**: *Kushal Tirumala, Aram H. Markosyan, Luke Zettlemoyer, Armen Aghajanyan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa0509f4dab6807e2cb465715bf2d249-Abstract-Conference.html)

        **Abstract**:

        Despite their wide adoption, the underlying training and memorization dynamics of very large language models is not well understood. We empirically study exact memorization in causal and masked language modeling, across model sizes and throughout the training process. We measure the effects of dataset size, learning rate, and model size on memorization, finding that larger language models memorize training data faster across all settings. Surprisingly, we show that larger models can memorize a larger portion of the data before over-fitting and tend to forget less throughout the training process. We also analyze the memorization dynamics of different parts of speech and find that models memorize nouns and numbers first; we hypothesize and provide empirical evidence that nouns and numbers act as a unique identifier for memorizing individual training examples. Together, these findings present another piece of the broader puzzle of trying to understand what actually improves as models get bigger.

        ----

        ## [2773] Bandit Theory and Thompson Sampling-Guided Directed Evolution for Sequence Optimization

        **Authors**: *Hui Yuan, Chengzhuo Ni, Huazheng Wang, Xuezhou Zhang, Le Cong, Csaba Szepesvári, Mengdi Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa3c139cf8084de7bfd944f1c90c8695-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa3c139cf8084de7bfd944f1c90c8695-Abstract-Conference.html)

        **Abstract**:

        Directed Evolution (DE), a landmark wet-lab method originated in 1960s, enables discovery of novel protein designs via evolving a population of candidate sequences. Recent advances in biotechnology has made it possible to collect high-throughput data, allowing the use of machine learning to map out a protein's sequence-to-function relation. There is a growing interest in machine learning-assisted DE for accelerating protein optimization. Yet the theoretical understanding of DE, as well as the use of machine learning in DE, remains limited.In this paper, we connect DE with the bandit learning theory and make a first attempt to study regret minimization in DE. We propose a Thompson Sampling-guided Directed Evolution (TS-DE) framework for sequence optimization, where the sequence-to-function mapping is unknown and querying a single value is subject to costly and noisy measurements. TS-DE updates a posterior of the function based on collected measurements. It uses a posterior-sampled function estimate to guide the crossover recombination and mutation steps in DE. In the case of a linear model, we show that TS-DE enjoys a Bayesian regret of order $\tilde O(d^{2}\sqrt{MT})$, where $d$ is feature dimension, $M$ is population size and $T$ is number of rounds. This regret bound is nearly optimal, confirming that bandit learning can provably accelerate DE. It may have implications for more general sequence optimization and evolutionary algorithms.

        ----

        ## [2774] Scalable and Efficient Training of Large Convolutional Neural Networks with Differential Privacy

        **Authors**: *Zhiqi Bu, Jialin Mao, Shiyun Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa5617c176e76fee83f3f9947fdf9f3f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa5617c176e76fee83f3f9947fdf9f3f-Abstract-Conference.html)

        **Abstract**:

        Large convolutional neural networks (CNN) can be difficult to train in the differentially private (DP) regime, since the optimization algorithms require a computationally expensive operation, known as the per-sample gradient clipping. We propose an efficient and scalable implementation of this clipping on convolutional layers, termed as the mixed ghost clipping, that significantly eases the private training in terms of both time and space complexities, without affecting the accuracy. The improvement in efficiency is rigorously studied through the first complexity analysis for the mixed ghost clipping and existing DP training algorithms.Extensive experiments on vision classification tasks, with large ResNet, VGG, and Vision Transformers (ViT), demonstrate that DP training with mixed ghost clipping adds $1\sim 10\%$ memory overhead and $<2\times$ slowdown to the standard non-private training. Specifically, when training VGG19 on CIFAR10, the mixed ghost clipping is $3\times$ faster than state-of-the-art Opacus library with $18\times$ larger maximum batch size. To emphasize the significance of efficient DP training on convolutional layers, we achieve 96.7\% accuracy on CIFAR10 and 83.0\% on CIFAR100 at $\epsilon=1$ using BEiT, while the previous best results are 94.8\% and 67.4\%, respectively. We open-source a privacy engine (\url{https://github.com/woodyx218/private_vision}) that implements DP training of CNN (including convolutional ViT) with a few lines of code.

        ----

        ## [2775] Weakly supervised causal representation learning

        **Authors**: *Johann Brehmer, Pim de Haan, Phillip Lippe, Taco S. Cohen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa567e2b2c870f8f09a87b6e73370869-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa567e2b2c870f8f09a87b6e73370869-Abstract-Conference.html)

        **Abstract**:

        Learning high-level causal representations together with a causal model from unstructured low-level data such as pixels is impossible from observational data alone. We prove under mild assumptions that this representation is however identifiable in a weakly supervised setting. This involves a dataset with paired samples before and after random, unknown interventions, but no further labels. We then introduce implicit latent causal models, variational autoencoders that represent causal variables and causal structure without having to optimize an explicit discrete graph structure. On simple image data, including a novel dataset of simulated robotic manipulation, we demonstrate that such models can reliably identify the causal structure and disentangle causal variables.

        ----

        ## [2776] Zeroth-Order Negative Curvature Finding: Escaping Saddle Points without Gradients

        **Authors**: *Hualin Zhang, Huan Xiong, Bin Gu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa5ddd6bac0d665c72969d79221b680a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa5ddd6bac0d665c72969d79221b680a-Abstract-Conference.html)

        **Abstract**:

        We consider escaping saddle points of nonconvex problems where only the function evaluations can be accessed. Although a variety of works have been proposed, the majority of them require either second or first-order information, and only a few of them have exploited zeroth-order methods, particularly the technique of negative curvature finding with zeroth-order methods which has been proven to be the most efficient method for escaping saddle points. To fill this gap,  in this paper, we propose two zeroth-order negative curvature finding frameworks that can replace Hessian-vector product computations without increasing the iteration complexity. We apply the proposed frameworks to ZO-GD, ZO-SGD, ZO-SCSG, ZO-SPIDER and prove that these ZO algorithms can converge to $(\epsilon,\delta)$-approximate second-order stationary points with less query complexity compared with prior zeroth-order works for finding local minima.

        ----

        ## [2777] Exposing and Exploiting Fine-Grained Block Structures for Fast and Accurate Sparse Training

        **Authors**: *Peng Jiang, Lihan Hu, Shihui Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa69e968b7319fd42524febd41475fb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa69e968b7319fd42524febd41475fb3-Abstract-Conference.html)

        **Abstract**:

        Sparse training is a popular technique to reduce the overhead of training large models. Although previous work has shown promising results for nonstructured sparse models, it is still unclear whether a sparse model with structural constraints can be trained from scratch to high accuracy. In this work, we study the dynamic sparse training for a class of sparse models with shuffled block structures. Compared to nonstructured models, such fine-grained structured models are more hardware-friendly and can effectively accelerate the training process. We propose an algorithm that keeps adapting the sparse model while maintaining the active parameters in shuffled blocks. We conduct experiments on a variety of networks and datasets and obtain positive results. In particular, on ImageNet, we achieve dense accuracy for ResNet50 and ResNet18 at 0.5 sparsity. On CIFAR10/100, we show that dense accuracy can be recovered at 0.6 sparsity for various models. At higher sparsity, our algorithm can still match the accuracy of nonstructured sparse training in most cases, while reducing the training time by up to 5x due to the fine-grained block structures in the models.

        ----

        ## [2778] DABS 20: Improved Datasets and Algorithms for Universal Self-Supervision

        **Authors**: *Alex Tamkin, Gaurab Banerjee, Mohamed Owda, Vincent Liu, Shashank Rammoorthy, Noah D. Goodman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa73aca7b2af724fafbd4852957cd3e0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa73aca7b2af724fafbd4852957cd3e0-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Universal self-supervised (SSL) algorithms hold enormous promise for making machine learning accessible to high-impact domains such as protein biology, manufacturing, and genomics. We present DABS 2.0: a set of improved datasets and algorithms for advancing research on universal SSL. We extend the recently-introduced DABS benchmark with the addition of five real-world science and engineering domains: protein biology, bacterial genomics, multispectral satellite imagery, semiconductor wafers, and particle physics, bringing the total number of domains in the benchmark to twelve. We also propose a new universal SSL algorithm, Capri, and a generalized version of masked autoencoding, and apply both on all twelve domains---the most wide-ranging exploration of SSL yet. We find that multiple algorithms show gains across domains, outperforming previous baselines. In addition, we demonstrate the usefulness of DABS for scientific study of SSL by investigating the optimal corruption rate for each algorithm, showing that the best setting varies based on the domain. Code will be released at http://github.com/alextamkin/dabs}{http://github.com/alextamkin/dabs

        ----

        ## [2779] Operator Splitting Value Iteration

        **Authors**: *Amin Rakhsha, Andrew Wang, Mohammad Ghavamzadeh, Amir-massoud Farahmand*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa809df3ec53cc5781e5078b7d500a5d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa809df3ec53cc5781e5078b7d500a5d-Abstract-Conference.html)

        **Abstract**:

        We introduce new planning and reinforcement learning algorithms for discounted MDPs that utilize an approximate model of the environment to accelerate the convergence of the value function. Inspired by the splitting approach in numerical linear algebra, we introduce \emph{Operator Splitting Value Iteration} (OS-VI) for both Policy Evaluation and Control problems. OS-VI achieves a much faster convergence rate when the model is accurate enough. We also introduce a sample-based version of the algorithm called OS-Dyna. Unlike the traditional Dyna architecture, OS-Dyna still converges to the correct value function in presence of model approximation error.

        ----

        ## [2780] Enhanced Latent Space Blind Model for Real Image Denoising via Alternative Optimization

        **Authors**: *Chao Ren, Yizhong Pan, Jie Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fa93d7bfb48450e1af63c8fa647d317f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fa93d7bfb48450e1af63c8fa647d317f-Abstract-Conference.html)

        **Abstract**:

        Motivated by the achievements in model-based methods and the advances in deep networks, we propose a novel enhanced latent space blind model based deep unfolding network, namely ScaoedNet, for complex real image denoising. It is derived by introducing latent space, noise information, and guidance constraint into the denoising cost function. A self-correction alternative optimization algorithm is proposed to split the novel cost function into three alternative subproblems, i.e., guidance representation (GR), degradation estimation (DE) and reconstruction (RE) subproblems. Finally, we implement the optimization process by a deep unfolding network consisting of GR, DE and RE networks. For higher performance of the DE network, a novel parameter-free noise feature adaptive enhancement (NFAE) layer is proposed. To synchronously and dynamically realize internal-external feature information mining in the RE network, a novel feature multi-modulation attention (FM2A) module is proposed. Our approach thereby leverages the advantages of deep learning, while also benefiting from the principled denoising provided by the classical model-based formulation. To the best of our knowledge, our enhanced latent space blind model, optimization scheme, NFAE and FM2A have not been reported in the previous literature. Experimental results show the promising performance of ScaoedNet on real image denoising. Code is available at https://github.com/chaoren88/ScaoedNet.

        ----

        ## [2781] Self-Explaining Deviations for Coordination

        **Authors**: *Hengyuan Hu, Samuel Sokota, David J. Wu, Anton Bakhtin, Andrei Lupu, Brandon Cui, Jakob N. Foerster*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/faa6276ea12d7afeb3e42b210c86f688-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/faa6276ea12d7afeb3e42b210c86f688-Abstract-Conference.html)

        **Abstract**:

        Fully cooperative, partially observable multi-agent problems are ubiquitous in the real world. In this paper, we focus on a specific subclass of coordination problems in which humans are able to discover self-explaining deviations (SEDs). SEDs are actions that deviate from the common understanding of what reasonable behavior would be in normal circumstances. They are taken with the intention of causing another agent or other agents to realize, using theory of mind, that the circumstance must be abnormal. We motivate this idea with a real world example and formalize its definition. Next, we introduce an algorithm for improvement maximizing SEDs (IMPROVISED). Lastly, we evaluate IMPROVISED both in an illustrative toy setting and the popular benchmark setting Hanabi, where we show that it can produce so called finesse plays.

        ----

        ## [2782] Communication Efficient Federated Learning for Generalized Linear Bandits

        **Authors**: *Chuanhao Li, Hongning Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/faa8be9311811ba7c36fa1ceec13b862-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/faa8be9311811ba7c36fa1ceec13b862-Abstract-Conference.html)

        **Abstract**:

        Contextual bandit algorithms have been recently studied under the federated learning setting to satisfy the demand of keeping data decentralized and pushing the learning of bandit models to the client side. But limited by the required communication efficiency, existing solutions are restricted to linear models to exploit their closed-form solutions for parameter estimation. Such a restricted model choice greatly hampers these algorithms' practical utility. In this paper, we take the first step to addressing this challenge by studying generalized linear bandit models under the federated learning setting. We propose a communication-efficient solution framework that employs online regression for local update and offline regression for global update. We rigorously proved, though the setting is more general and challenging, our algorithm can attain sub-linear rate in both regret and communication cost, which is also validated by our extensive empirical evaluations.

        ----

        ## [2783] Active Learning for Multiple Target Models

        **Authors**: *Ying-Peng Tang, Sheng-Jun Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/faacb7a4827b4d51e201666b93ab5fa7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/faacb7a4827b4d51e201666b93ab5fa7-Abstract-Conference.html)

        **Abstract**:

        We describe and explore a novel setting of active learning (AL), where there are multiple target models to be learned simultaneously. In many real applications, the machine learning system is required to be deployed on diverse devices with varying computational resources (e.g., workstation, mobile phone, edge devices, etc.), which leads to the demand of training multiple target models on the same labeled dataset. However, it is generally believed that AL is model-dependent and untransferable, i.e., the data queried by one model may be less effective for training another model. This phenomenon naturally raises a question "Does there exist an AL method that is effective for multiple target models?" In this paper, we answer this question by theoretically analyzing the label complexity of active and passive learning under the setting with multiple target models, and conclude that AL does have potential to achieve better label complexity under this novel setting. Based on this insight, we further propose an agnostic AL sampling strategy to select the examples located in the joint disagreement regions of different target models. The experimental results on the OCR benchmarks show that the proposed method can significantly surpass the traditional active and passive learning methods under this challenging setting.

        ----

        ## [2784] Descent Steps of a Relation-Aware Energy Produce Heterogeneous Graph Neural Networks

        **Authors**: *Hongjoon Ahn, Yongyi Yang, Quan Gan, Taesup Moon, David P. Wipf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/facaa170287a034cf99cf0489a7f8430-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/facaa170287a034cf99cf0489a7f8430-Abstract-Conference.html)

        **Abstract**:

        Heterogeneous graph neural networks (GNNs) achieve strong performance on node classification tasks in a semi-supervised learning setting. However, as in the simpler homogeneous GNN case, message-passing-based heterogeneous GNNs may struggle to balance between resisting the oversmoothing that may occur in deep models, and capturing long-range dependencies of graph structured data. Moreover, the complexity of this trade-off is compounded in the heterogeneous graph case due to the disparate heterophily relationships between nodes of different types. To address these issues, we propose a novel heterogeneous GNN architecture in which layers are derived from optimization steps that descend a novel relation-aware energy function. The corresponding minimizer is fully differentiable with respect to the energy function parameters, such that bilevel optimization can be applied to effectively learn a functional form whose minimum provides optimal node representations for subsequent classification tasks.  In particular, this methodology allows us to model diverse heterophily relationships between different node types while avoiding oversmoothing effects.  Experimental results on 8 heterogeneous graph benchmarks demonstrates that our proposed method can achieve competitive node classification accuracy.

        ----

        ## [2785] Multi-agent Performative Prediction with Greedy Deployment and Consensus Seeking Agents

        **Authors**: *Qiang Li, Chung-Yiu Yau, Hoi-To Wai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fad7c708dda11f3e72cc1629bb130379-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fad7c708dda11f3e72cc1629bb130379-Abstract-Conference.html)

        **Abstract**:

        We consider a scenario where multiple agents are learning a common decision vector from data which can be influenced by the agents’ decisions. This leads to the problem of multi-agent performative prediction (Multi-PfD). In this paper, we formulate Multi-PfD as a decentralized optimization problem that minimizes a sum of loss functions, where each loss function is based on a distribution influenced by the local decision vector. We first prove the necessary and sufficient condition for the Multi-PfD problem to admit a unique multi-agent performative stable (Multi-PS) solution. We show that enforcing consensus leads to a laxer condition for existence of Multi-PS solution with respect to the distributions’ sensitivities, compared to the single agent case. Then, we study a decentralized extension to  the greedy deployment scheme [Mendler-Dünner et al., 2020], called the DSGD-GD   scheme. We show that DSGD-GD converges to the Multi-PS solution and analyze its non asymptotic convergence rate. Numerical results validate our analysis.

        ----

        ## [2786] Preservation of the Global Knowledge by Not-True Distillation in Federated Learning

        **Authors**: *Gihun Lee, Minchan Jeong, Yongjin Shin, Sangmin Bae, Se-Young Yun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fadec8f2e65f181d777507d1df69b92f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fadec8f2e65f181d777507d1df69b92f-Abstract-Conference.html)

        **Abstract**:

        In federated learning, a strong global model is collaboratively learned by aggregating clients' locally trained models. Although this precludes the need to access clients' data directly, the global model's convergence often suffers from data heterogeneity. This study starts from an analogy to continual learning and suggests that forgetting could be the bottleneck of federated learning. We observe that the global model forgets the knowledge from previous rounds, and the local training induces forgetting the knowledge outside of the local distribution. Based on our findings, we hypothesize that tackling down forgetting will relieve the data heterogeneity problem. To this end, we propose a novel and effective algorithm, Federated Not-True Distillation (FedNTD), which preserves the global perspective on locally available data only for the not-true classes. In the experiments, FedNTD shows state-of-the-art performance on various setups without compromising data privacy or incurring additional communication costs.

        ----

        ## [2787] Finite-Time Regret of Thompson Sampling Algorithms for Exponential Family Multi-Armed Bandits

        **Authors**: *Tianyuan Jin, Pan Xu, Xiaokui Xiao, Anima Anandkumar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fb23cf87a9e04d7677b73c47acd060ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fb23cf87a9e04d7677b73c47acd060ef-Abstract-Conference.html)

        **Abstract**:

        We study the regret of Thompson sampling (TS) algorithms for exponential family bandits, where the reward distribution is from a one-dimensional exponential family, which covers many common reward distributions including Bernoulli, Gaussian, Gamma, Exponential, etc. We propose a Thompson sampling algorithm, termed ExpTS, which uses a novel sampling distribution to avoid the under-estimation of the optimal arm. We provide a tight regret analysis for ExpTS, which simultaneously yields both the finite-time regret bound as well as the asymptotic regret bound. In particular, for a $K$-armed bandit with exponential family rewards, ExpTS over a horizon $T$ is sub-UCB (a strong criterion for the finite-time regret that is problem-dependent), minimax optimal up to a factor $\sqrt{\log K}$, and asymptotically optimal, for exponential family rewards. Moreover, we propose ExpTS$^+$, by adding a greedy exploitation step in addition to the sampling distribution used in ExpTS, to avoid the over-estimation of sub-optimal arms. ExpTS$^+$ is an anytime bandit algorithm and achieves the minimax optimality and asymptotic optimality simultaneously for exponential family reward distributions. Our proof techniques are general and conceptually simple and can be easily applied to analyze standard Thompson sampling with specific reward distributions.

        ----

        ## [2788] Graph Reordering for Cache-Efficient Near Neighbor Search

        **Authors**: *Benjamin Coleman, Santiago Segarra, Alexander J. Smola, Anshumali Shrivastava*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html)

        **Abstract**:

        Graph search is one of the most successful algorithmic trends in near neighbor search. Several of the most popular and empirically successful algorithms are, at their core, a greedy walk along a pruned near neighbor graph. However, graph traversal applications often suffer from poor memory access patterns, and near neighbor search is no exception to this rule. Our measurements show that popular search indices such as the hierarchical navigable small-world graph (HNSW) can have poor cache miss performance. To address this issue, we formulate the graph traversal problem as a cache hit maximization task and propose multiple graph reordering as a solution. Graph reordering is a memory layout optimization that groups commonly-accessed nodes together in memory. We mathematically formalize the connection between the graph layout and the cache complexity of search. We present exhaustive experiments applying several reordering algorithms to a leading graph-based near neighbor method based on the HNSW index. We find that reordering improves the query time by up to 40%, we present analysis and improvements for existing graph layout methods, and we demonstrate that the time needed to reorder the graph is negligible compared to the time required to construct the index.

        ----

        ## [2789] MetaMask: Revisiting Dimensional Confounder for Self-Supervised Learning

        **Authors**: *Jiangmeng Li, Wenwen Qiang, Yanan Zhang, Wenyi Mo, Changwen Zheng, Bing Su, Hui Xiong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fb575ab4d882a4c734641155a5f30911-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fb575ab4d882a4c734641155a5f30911-Abstract-Conference.html)

        **Abstract**:

        As a successful approach to self-supervised learning, contrastive learning aims to learn invariant information shared among distortions of the input sample. While contrastive learning has yielded continuous advancements in sampling strategy and architecture design, it still remains two persistent defects: the interference of task-irrelevant information and sample inefficiency, which are related to the recurring existence of trivial constant solutions. From the perspective of dimensional analysis, we find out that the dimensional redundancy and dimensional confounder are the intrinsic issues behind the phenomena, and provide experimental evidence to support our viewpoint. We further propose a simple yet effective approach MetaMask, short for the dimensional Mask learned by Meta-learning, to learn representations against dimensional redundancy and confounder. MetaMask adopts the redundancy-reduction technique to tackle the dimensional redundancy issue and innovatively introduces a dimensional mask to reduce the gradient effects of specific dimensions containing the confounder, which is trained by employing a meta-learning paradigm with the objective of improving the performance of masked representations on a typical self-supervised task. We provide solid theoretical analyses to prove MetaMask can obtain tighter risk bounds for downstream classification compared to typical contrastive methods. Empirically, our method achieves state-of-the-art performance on various benchmarks.

        ----

        ## [2790] On Feature Learning in the Presence of Spurious Correlations

        **Authors**: *Pavel Izmailov, Polina Kirichenko, Nate Gruver, Andrew Gordon Wilson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fb64a552feda3d981dbe43527a80a07e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fb64a552feda3d981dbe43527a80a07e-Abstract-Conference.html)

        **Abstract**:

        Deep classifiers are known to rely on spurious features â€” patterns which are correlated with the target on the training data but not inherently relevant to the learning problem, such as the image backgrounds when classifying the foregrounds. In this paper we evaluate the amount of information about the core (non-spurious) features that can be decoded from the representations learned by standard empirical risk minimization (ERM) and specialized group robustness training. Following recent work on Deep Feature Reweighting (DFR), we evaluate the feature representations by re-training the last layer of the model on a held-out set where the spurious correlation is broken. On multiple vision and NLP problems, we show that the features learned by simple ERM are highly competitive with the features learned by specialized group robustness methods targeted at reducing the effect of spurious correlations. Moreover, we show that the quality of learned feature representations is greatly affected by the design decisions beyond the training method, such as the model architecture and pre-training strategy. On the other hand, we find that strong regularization is not necessary for learning high-quality feature representations.Finally, using insights from our analysis, we significantly improve upon the best results reported in the literature on the popular Waterbirds, CelebA hair color prediction and WILDS-FMOW problems, achieving 97\%, 92\% and 50\% worst-group accuracies, respectively.

        ----

        ## [2791] Sparse2Dense: Learning to Densify 3D Features for 3D Object Detection

        **Authors**: *Tianyu Wang, Xiaowei Hu, Zhengzhe Liu, Chi-Wing Fu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fb71332951af4ae27fbd457daadc5341-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fb71332951af4ae27fbd457daadc5341-Abstract-Conference.html)

        **Abstract**:

        LiDAR-produced point clouds are the major source for most state-of-the-art 3D object detectors. Yet, small, distant, and incomplete objects with sparse or few points are often hard to detect. We present Sparse2Dense, a new framework to efficiently boost 3D detection performance by learning to densify point clouds in latent space. Specifically, we first train a dense point 3D detector (DDet) with a dense point cloud as input and design a sparse point 3D detector (SDet) with a regular point cloud as input. Importantly, we formulate the lightweight plug-in S2D module and the point cloud reconstruction module in SDet to densify 3D features and train SDet to produce 3D features, following the dense 3D features in DDet. So, in inference, SDet can simulate dense 3D features from regular (sparse) point cloud inputs without requiring dense inputs. We evaluate our method on the large-scale Waymo Open Dataset and the Waymo Domain Adaptation Dataset, showing its high performance and efficiency over the state of the arts.

        ----

        ## [2792] Exploring Length Generalization in Large Language Models

        **Authors**: *Cem Anil, Yuhuai Wu, Anders Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay V. Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, Behnam Neyshabur*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fb7451e43f9c1c35b774bcfad7a5714b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fb7451e43f9c1c35b774bcfad7a5714b-Abstract-Conference.html)

        **Abstract**:

        The ability to extrapolate from short problem instances to longer ones is an important form of out-of-distribution generalization in reasoning tasks, and is crucial when learning from datasets where longer problem instances are rare. These include theorem proving, solving quantitative mathematics problems, and reading/summarizing novels. In this paper, we run careful empirical studies exploring the length generalization capabilities of transformer-based language models. We first establish that naively finetuning transformers on length generalization tasks shows significant generalization deficiencies independent of model scale. We then show that combining pretrained large language models' in-context learning abilities with scratchpad prompting (asking the model to output solution steps before producing an answer) results in a dramatic improvement in length generalization. We run careful failure analyses on each of the learning modalities and identify common sources of mistakes that highlight opportunities in equipping language models with the ability to generalize to longer problems.

        ----

        ## [2793] Stability and Generalization Analysis of Gradient Methods for Shallow Neural Networks

        **Authors**: *Yunwen Lei, Rong Jin, Yiming Ying*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fb8fe6b79288f3d83696a5d276f4fc9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fb8fe6b79288f3d83696a5d276f4fc9d-Abstract-Conference.html)

        **Abstract**:

        While significant theoretical progress has been achieved,  unveiling the generalization mystery of overparameterized neural networks still remains largely elusive. In this paper, we study the generalization behavior of shallow neural networks (SNNs) by leveraging the concept of algorithmic stability. We consider gradient descent (GD) and stochastic gradient descent (SGD) to train SNNs, for both of which we develop consistent excess risk bounds by balancing the optimization and generalization via early-stopping. As compared to existing analysis on GD, our new analysis requires a relaxed overparameterization assumption and also  applies to SGD. The key for the improvement is a better estimation of the smallest eigenvalues of the Hessian matrices of the empirical risks and the loss function along the trajectories of GD and SGD by providing a refined estimation of their iterates.

        ----

        ## [2794] ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation

        **Authors**: *Yufei Xu, Jing Zhang, Qiming Zhang, Dacheng Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fbb10d319d44f8c3b4720873e4177c65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fbb10d319d44f8c3b4720873e4177c65-Abstract-Conference.html)

        **Abstract**:

        Although no specific domain knowledge is considered in the design, plain vision transformers have shown excellent performance in visual recognition tasks. However, little effort has been made to reveal the potential of such simple structures for pose estimation tasks. In this paper, we show the surprisingly good capabilities of plain vision transformers for pose estimation from various aspects, namely simplicity in model structure, scalability in model size, flexibility in training paradigm, and transferability of knowledge between models, through a simple baseline model called ViTPose. Specifically, ViTPose employs plain and non-hierarchical vision transformers as backbones to extract features for a given person instance and a lightweight decoder for pose estimation. It can be scaled up from 100M to 1B parameters by taking the advantages of the scalable model capacity and high parallelism of transformers, setting a new Pareto front between throughput and performance. Besides, ViTPose is very flexible regarding the attention type, input resolution, pre-training and finetuning strategy, as well as dealing with multiple pose tasks. We also empirically demonstrate that the knowledge of large ViTPose models can be easily transferred to small ones via a simple knowledge token. Experimental results show that our basic ViTPose model outperforms representative methods on the challenging MS COCO Keypoint Detection benchmark, while the largest model sets a new state-of-the-art. The code and models are available at https://github.com/ViTAE-Transformer/ViTPose.

        ----

        ## [2795] Re-Analyze Gauss: Bounds for Private Matrix Approximation via Dyson Brownian Motion

        **Authors**: *Oren Mangoubi, Nisheeth K. Vishnoi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fbc9981dd6316378aee7fd5975250f21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fbc9981dd6316378aee7fd5975250f21-Abstract-Conference.html)

        **Abstract**:

        Given a symmetric matrix $M$ and a vector $\lambda$, we present new bounds on the Frobenius-distance utility of the Gaussian mechanism for  approximating $M$ by a matrix whose spectrum is $\lambda$, under $(\varepsilon,\delta)$-differential privacy. Our bounds depend on both $\lambda$ and the gaps in the eigenvalues of $M$, and hold whenever the top $k+1$ eigenvalues of $M$ have sufficiently large gaps. When applied to the problems of private rank-$k$ covariance matrix approximation and subspace recovery, our bounds yield improvements over previous bounds. Our bounds are obtained by viewing the addition of Gaussian noise as a continuous-time matrix Brownian motion. This viewpoint allows us to track the evolution of eigenvalues and eigenvectors of the matrix, which are governed by  stochastic differential equations discovered by Dyson. These equations allow us to bound the utility as the square-root of a sum-of-squares of perturbations to the eigenvectors, as opposed to a sum of perturbation bounds obtained via Davis-Kahan-type theorems.

        ----

        ## [2796] ASPiRe: Adaptive Skill Priors for Reinforcement Learning

        **Authors**: *Mengda Xu, Manuela Veloso, Shuran Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fbd8e65962da06f83f3f28b52774ffd0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fbd8e65962da06f83f3f28b52774ffd0-Abstract-Conference.html)

        **Abstract**:

        We introduce ASPiRe (Adaptive Skill Prior for RL), a new approach that leverages prior experience to accelerate reinforcement learning. Unlike existing methods that learn a single skill prior from a large and diverse dataset, our framework learns a library of different distinction skill priors (i.e., behavior priors) from a collection of specialized datasets, and learns how to combine them to solve a new task. This formulation allows the algorithm to acquire a set of specialized skill priors that are more reusable for downstream tasks; however, it also brings up additional challenges of how to effectively combine these unstructured sets of skill priors to form a new prior for new tasks. Specifically, it requires the agent not only to identify which skill prior(s) to use but also how to combine them (either sequentially or concurrently) to form a new prior. To achieve this goal, ASPiRe includes Adaptive Weight Module (AWM) that learns to infer an adaptive weight assignment between different skill priors and uses them to guide policy learning for downstream tasks via weighted Kullback-Leibler divergences. Our experiments demonstrate that ASPiRe can significantly accelerate the learning of new downstream tasks in the presence of multiple priors and show improvement on competitive baselines.

        ----

        ## [2797] Neural Differential Equations for Learning to Program Neural Nets Through Continuous Learning Rules

        **Authors**: *Kazuki Irie, Francesco Faccio, Jürgen Schmidhuber*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fc09b26b85ab3abb2832bd555a2e4215-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fc09b26b85ab3abb2832bd555a2e4215-Abstract-Conference.html)

        **Abstract**:

        Neural ordinary differential equations (ODEs) have attracted much attention as continuous-time counterparts of deep residual neural networks (NNs), and numerous extensions for recurrent NNs have been proposed. Since the 1980s, ODEs have also been used to derive theoretical results for NN learning rules, e.g., the famous connection between Oja's rule and principal component analysis. Such rules are typically expressed as additive iterative update processes which have  straightforward ODE counterparts. Here we introduce a novel combination of learning rules and Neural ODEs to build continuous-time sequence processing nets that learn to manipulate short-term memory in rapidly changing synaptic connections of other nets. This yields continuous-time counterparts of Fast Weight Programmers and linear Transformers. Our novel models outperform the best existing Neural Controlled Differential Equation based models on various time series classification tasks, while also addressing their fundamental scalability limitations. Our code is public.

        ----

        ## [2798] MEMO: Test Time Robustness via Adaptation and Augmentation

        **Authors**: *Marvin Zhang, Sergey Levine, Chelsea Finn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html)

        **Abstract**:

        While deep neural networks can attain good accuracy on in-distribution test points, many applications require robustness even in the face of unexpected perturbations in the input, changes in the domain, or other sources of distribution shift. We study the problem of test time robustification, i.e., using the test input to improve model robustness. Recent prior works have proposed methods for test time adaptation, however, they each introduce additional assumptions, such as access to multiple test points, that prevent widespread adoption. In this work, we aim to study and devise methods that make no assumptions about the model training process and are broadly applicable at test time. We propose a simple approach that can be used in any test setting where the model is probabilistic and adaptable: when presented with a test example, perform different data augmentations on the data point, and then adapt (all of) the model parameters by minimizing the entropy of the model's average, or marginal, output distribution across the augmentations. Intuitively, this objective encourages the model to make the same prediction across different augmentations, thus enforcing the invariances encoded in these augmentations, while also maintaining confidence in its predictions. In our experiments, we evaluate two baseline ResNet models, two robust ResNet-50 models, and a robust vision transformer model, and we demonstrate that this approach achieves accuracy gains of 1-8% over standard model evaluation and also generally outperforms prior augmentation and adaptation strategies. For the setting in which only one test point is available, we achieve state-of-the-art results on the ImageNet-C, ImageNet-R, and, among ResNet-50 models, ImageNet-A distribution shift benchmarks.

        ----

        ## [2799] Learning-Augmented Algorithms for Online Linear and Semidefinite Programming

        **Authors**: *Elena Grigorescu, Young-San Lin, Sandeep Silwal, Maoyuan Song, Samson Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fc5a1845bee1f5405ef99ba25c2d44e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fc5a1845bee1f5405ef99ba25c2d44e1-Abstract-Conference.html)

        **Abstract**:

        Semidefinite programming (SDP) is a unifying framework that generalizes both linear programming and quadratically-constrained  quadratic programming, while also yielding efficient solvers, both in theory and in practice. However, there exist known impossibility results for approximating the optimal solution when constraints for covering SDPs arrive in an online fashion. In this paper, we study online covering linear and semidefinite programs in which the algorithm is augmented with advice from a possibly erroneous predictor. We show that if the predictor is accurate, we can efficiently bypass these impossibility results and achieve a constant-factor approximation to the optimal solution, i.e., consistency. On the other hand, if the predictor is inaccurate, under some technical conditions, we achieve results that match both the classical optimal upper bounds and the tight lower bounds up to constant factors, i.e., robustness. More broadly, we introduce a framework that extends both (1) the online set cover problem augmented with machine-learning predictors, studied by Bamas, Maggiori, and Svensson (NeurIPS 2020), and (2) the online covering SDP problem, initiated by Elad, Kale, and Naor (ICALP 2016).  Specifically, we obtain general online learning-augmented algorithms for covering linear programs with fractional advice and constraints, and initiate the study of learning-augmented algorithms for covering SDP problems. Our techniques are based on the primal-dual framework of Buchbinder and Naor (Mathematics of Operations Research, 34, 2009) and can be further adjusted to handle constraints where the variables lie in a bounded region, i.e., box constraints.

        ----

        

[Go to the previous page](NIPS-2022-list13.md)

[Go to the next page](NIPS-2022-list15.md)

[Go to the catalog section](README.md)