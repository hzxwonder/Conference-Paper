## [0] On the Generalization Properties of Diffusion Models

**Authors**: *Puheng Li, Zhong Li, Huishuai Zhang, Jiang Bian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06abed94583030dd50abe6767bd643b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06abed94583030dd50abe6767bd643b1-Abstract-Conference.html)

**Abstract**:

Diffusion models are a class of generative models that serve to establish a stochastic transport map between an empirically observed, yet unknown, target distribution and a known prior. Despite their remarkable success in real-world applications, a theoretical understanding of their generalization capabilities remains underdeveloped. This work embarks on a comprehensive theoretical exploration of the generalization attributes of diffusion models. We establish the theoretical estimates of the generalization gap that evolves in tandem with the training dynamics of score-based diffusion models, suggesting a polynomially small generalization error ($O(n^{-2/5}+m^{-4/5})$) on both the sample size $n$ and the model capacity $m$, evading the curse of dimensionality (i.e., independent of the data dimension) when *early-stopped*. Furthermore, we extend our quantitative analysis to a *data-dependent* scenario, wherein target distributions are portrayed as a succession of densities with progressively increasing distances between modes. This precisely elucidates the *adverse* effect of "*modes shift*'' in ground truths on the model generalization. Furthermore, these estimates are not solely theoretical constructs but have also been confirmed through numerical simulations. Our findings contribute to the rigorous understanding of diffusion models' generalization properties and provide insights that may guide practical applications.

----

## [0] Regularized Behavior Cloning for Blocking the Leakage of Past Action Information

**Authors**: *Seokin Seo, HyeongJoo Hwang, Hongseok Yang, Kee-Eung Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06b71ad997f7e3e4b2e2f2ea12e5a759-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06b71ad997f7e3e4b2e2f2ea12e5a759-Abstract-Conference.html)

**Abstract**:

For partially observable environments, imitation learning with observation histories (ILOH) assumes that control-relevant information is sufficiently captured in the observation histories for imitating the expert actions. In the offline setting wherethe agent is required to learn to imitate without interaction with the environment, behavior cloning (BC) has been shown to be a simple yet effective method for imitation learning. However, when the information about the actions executed in the past timesteps leaks into the observation histories, ILOH via BC often ends up imitating its own past actions. In this paper, we address this catastrophic failure by proposing a principled regularization for BC, which we name Past Action Leakage Regularization (PALR). The main idea behind our approach is to leverage the classical notion of conditional independence to mitigate the leakage. We compare different instances of our framework with natural choices of conditional independence metric and its estimator. The result of our comparison advocates the use of a particular kernel-based estimator for the conditional independence metric. We conduct an extensive set of experiments on benchmark datasets in order to assess the effectiveness of our regularization method. The experimental results show that our method significantly outperforms prior related approaches, highlighting its potential to successfully imitate expert actions when the past action information leaks into the observation histories.

----

## [0] The Distortion of Binomial Voting Defies Expectation

**Authors**: *Yannai A. Gonczarowski, Gregory Kehne, Ariel D. Procaccia, Ben Schiffer, Shirley Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06cb881ec90a657a8f949a62f1b4ee5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06cb881ec90a657a8f949a62f1b4ee5f-Abstract-Conference.html)

**Abstract**:

In computational social choice, the distortion of a voting rule quantifies the degree to which the rule overcomes limited preference information to select a socially desirable outcome. This concept has been investigated extensively, but only through a worst-case lens. Instead, we study the expected distortion of voting rules with respect to an underlying distribution over voter utilities. Our main contribution is the design and analysis of a novel and intuitive rule, binomial voting, which provides strong distribution-independent guarantees for both expected distortion and expected welfare.

----

## [0] UP-DP: Unsupervised Prompt Learning for Data Pre-Selection with Vision-Language Models

**Authors**: *Xin Li, Sima Behpour, Thang Long Doan, Wenbin He, Liang Gou, Liu Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06d5f1fe6509b001e6d4e0ec1afd83dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06d5f1fe6509b001e6d4e0ec1afd83dd-Abstract-Conference.html)

**Abstract**:

In this study, we investigate the task of data pre-selection, which aims to select instances for labeling from an unlabeled dataset through a single pass, thereby optimizing performance for undefined downstream tasks with a limited annotation budget. Previous approaches to data pre-selection relied solely on visual features extracted from foundation models, such as CLIP and BLIP-2, but largely ignored the powerfulness of text features. In this work, we argue that, with proper design, the joint feature space of both vision and text can yield a better representation for data pre-selection. To this end, we introduce UP-DP, a simple yet effective unsupervised prompt learning approach that adapts vision-language models, like BLIP-2, for data pre-selection. Specifically, with the BLIP-2 parameters frozen, we train text prompts to extract the joint features with improved representation, ensuring a diverse cluster structure that covers the entire dataset. We extensively compare our method with the state-of-the-art using seven benchmark datasets in different settings, achieving up to a performance gain of 20\%. Interestingly, the prompts learned from one dataset demonstrate significant generalizability and can be applied directly to enhance the feature extraction of BLIP-2 from other datasets. To the best of our knowledge, UP-DP is the first work to incorporate unsupervised prompt learning in a vision-language model for data pre-selection.

----

## [0] Optimistic Rates for Multi-Task Representation Learning

**Authors**: *Austin Watkins, Enayat Ullah, Thanh Nguyen-Tang, Raman Arora*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06e3c330d140f3a25671acf2dc2d6357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06e3c330d140f3a25671acf2dc2d6357-Abstract-Conference.html)

**Abstract**:

We study the problem of transfer learning via Multi-Task Representation Learning (MTRL), wherein multiple source tasks are used to learn a good common representation, and a predictor is trained on top of it for the target task. Under standard regularity assumptions on the loss function and task diversity, we provide new statistical rates on the excess risk of the target task, which demonstrate the benefit of representation learning. Importantly, our rates are optimistic, i.e., they interpolate between the standard $O(m^{-1/2})$ rate and the fast $O(m^{-1})$ rate, depending on the difficulty of the learning task, where $m$ is the number of samples for the target task. Besides the main result, we make several new contributions, including giving optimistic rates for excess risk of source tasks (multi-task learning (MTL)), a local Rademacher complexity theorem for MTRL and MTL, as well as a chain rule for local Rademacher complexity for composite predictor classes.

----

## [0] Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

**Authors**: *Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M. Alabdulmohsin, Avital Oliver, Piotr Padlewski, Alexey A. Gritsenko, Mario Lucic, Neil Houlsby*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html)

**Abstract**:

The ubiquitous and demonstrably suboptimal choice of resizing images to a fixed resolution before processing them with computer vision models has not yet been successfully challenged. However, models such as the Vision Transformer (ViT) offer flexible sequence-based modeling, and hence varying input sequence lengths.  We take advantage of this with NaViT (Native Resolution ViT) which uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios. Alongside flexible model usage, we demonstrate improved training efficiency for large-scale supervised and contrastive image-text pretraining.NaViT can be efficiently transferred to standard tasks such as image and video classification, object detection, and semantic segmentation and leads to improved results on robustness and fairness benchmarks. At inference time, the input resolution flexibility can be used to smoothly navigate the test-time cost-performance trade-off. We believe that NaViTmarks a departure from the standard, CNN-designed, input and modelling pipeline used by most computer vision models, and represents a promising direction for ViTs.

----

## [0] The Benefits of Being Distributional: Small-Loss Bounds for Reinforcement Learning

**Authors**: *Kaiwen Wang, Kevin Zhou, Runzhe Wu, Nathan Kallus, Wen Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06fc38f5c21ae66ef955e28b7a78ece5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06fc38f5c21ae66ef955e28b7a78ece5-Abstract-Conference.html)

**Abstract**:

While distributional reinforcement learning (DistRL) has been empirically effective, the question of when and why it is better than vanilla, non-distributional RL has remained unanswered.This paper explains the benefits of DistRL through the lens of small-loss bounds, which are instance-dependent bounds that scale with optimal achievable cost.Particularly, our bounds converge much faster than those from non-distributional approaches if the optimal cost is small.As warmup, we propose a distributional contextual bandit (DistCB) algorithm, which we show enjoys small-loss regret bounds and empirically outperforms the state-of-the-art on three real-world tasks.In online RL, we propose a DistRL algorithm that constructs confidence sets using maximum likelihood estimation. We prove that our algorithm enjoys novel small-loss PAC bounds in low-rank MDPs.As part of our analysis, we introduce the $\ell_1$ distributional eluder dimension which may be of independent interest. Then, in offline RL, we show that pessimistic DistRL enjoys small-loss PAC bounds that are novel to the offline setting and are more robust to bad single-policy coverage.

----

## [0] Honesty Is the Best Policy: Defining and Mitigating AI Deception

**Authors**: *Francis Ward, Francesca Toni, Francesco Belardinelli, Tom Everitt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06fc7ae4a11a7eb5e20fe018db6c036f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06fc7ae4a11a7eb5e20fe018db6c036f-Abstract-Conference.html)

**Abstract**:

Deceptive agents are a challenge for the safety, trustworthiness, and cooperation of AI systems. We focus on the problem that agents might deceive in order to achieve their goals (for instance, in our experiments with language models, the goal of being evaluated as truthful).There are a number of existing definitions of deception in the literature on game theory and symbolic AI, but there is no overarching theory of deception for learning agents in games. We introduce a formaldefinition of deception in structural causal games, grounded in the philosophyliterature, and applicable to real-world machine learning systems.Several examples and results illustrate that our formal definition aligns with the philosophical and commonsense meaning of deception.Our main technical result is to provide graphical criteria for deception. We show, experimentally, that these results can be used to mitigate deception in reinforcement learning agents and language models.

----

## [0] Improving *day-ahead* Solar Irradiance Time Series Forecasting by Leveraging Spatio-Temporal Context

**Authors**: *Oussama Boussif, Ghait Boukachab, Dan Assouline, Stefano Massaroli, Tianle Yuan, Loubna Benabbou, Yoshua Bengio*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/070a57c5ef1e58cc90201b11d369b3c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/070a57c5ef1e58cc90201b11d369b3c2-Abstract-Conference.html)

**Abstract**:

Solar power harbors immense potential in mitigating climate change by substantially reducing CO$_{2}$ emissions. Nonetheless, the inherent variability of solar irradiance poses a significant challenge for seamlessly integrating solar power into the electrical grid. While the majority of prior research has centered on employing purely time series-based methodologies for solar forecasting, only a limited number of studies have taken into account factors such as cloud cover or the surrounding physical context.In this paper, we put forth a deep learning architecture designed to harness spatio-temporal context using satellite data, to attain highly accurate day-ahead time-series forecasting for any given station, with a particular emphasis on forecasting Global Horizontal Irradiance (GHI). We also suggest a methodology to extract a distribution for each time step prediction, which can serve as a very valuable measure of uncertainty attached to the forecast. When evaluating models, we propose a testing scheme in which we separate particularly difficult examples from easy ones, in order to capture the model performances in crucial situations, which in the case of this study are the days suffering from varying cloudy conditions. Furthermore, we present a new multi-modal dataset gathering satellite imagery over a large zone and time series for solar irradiance and other related physical variables from multiple geographically diverse solar stations. Our approach exhibits robust performance in solar irradiance forecasting, including zero-shot generalization tests at unobserved solar stations, and holds great promise in promoting the effective integration of solar power into the grid.

----

## [0] Uncovering and Quantifying Social Biases in Code Generation

**Authors**: *Yan Liu, Xiaokang Chen, Yan Gao, Zhe Su, Fengji Zhang, Daoguang Zan, Jian-Guang Lou, Pin-Yu Chen, Tsung-Yi Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/071a637d41ea290ac4360818a8323f33-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/071a637d41ea290ac4360818a8323f33-Abstract-Conference.html)

**Abstract**:

With the popularity of automatic code generation tools, such as Copilot, the study of the potential hazards of these tools is gaining importance. In this work, we explore the social bias problem in pre-trained code generation models. We propose a new paradigm to construct code prompts and successfully uncover social biases in code generation models. To quantify the severity of social biases in generated code, we develop a dataset along with three metrics to evaluate the overall social bias and fine-grained unfairness across different demographics. Experimental results on three pre-trained code generation models (Codex, InCoder, and CodeGen) with varying sizes, reveal severe social biases. Moreover, we conduct analysis to provide useful insights for further choice of code generation models with low social bias.

----

## [0] A Bounded Ability Estimation for Computerized Adaptive Testing

**Authors**: *Yan Zhuang, Qi Liu, Guanhao Zhao, Zhenya Huang, Weizhe Huang, Zachary A. Pardos, Enhong Chen, Jinze Wu, Xin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0730b81dbc16cce7e85b519cb7fe5a8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0730b81dbc16cce7e85b519cb7fe5a8d-Abstract-Conference.html)

**Abstract**:

Computerized adaptive testing (CAT), as a tool that can efficiently measure student's ability, has been widely used in various standardized tests (e.g., GMAT and GRE). The adaptivity of CAT refers to the selection of the most informative questions for each student, reducing test length. Existing CAT methods do not explicitly target ability estimation accuracy since there is no student's true ability as ground truth; therefore, these methods cannot be guaranteed to make the estimate converge to the true with such limited responses. In this paper, we analyze the statistical properties of estimation and find a theoretical approximation of the true ability: the ability estimated by full responses to question bank. Based on this, a Bounded Ability Estimation framework for CAT (BECAT) is proposed in a data-summary manner, which selects a question subset that closely matches the gradient of the full responses. Thus, we develop an expected gradient difference approximation to design a simple greedy selection algorithm, and show the rigorous theoretical and error upper-bound guarantees of its ability estimate. Experiments on both real-world and synthetic datasets, show that it can reach the same estimation accuracy using 15\% less questions on average, significantly reducing test length.

----

## [0] ForecastPFN: Synthetically-Trained Zero-Shot Forecasting

**Authors**: *Samuel Dooley, Gurnoor Singh Khurana, Chirag Mohapatra, Siddartha V. Naidu, Colin White*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0731f0e65559059eb9cd9d6f44ce2dd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0731f0e65559059eb9cd9d6f44ce2dd8-Abstract-Conference.html)

**Abstract**:

The vast majority of time-series forecasting approaches require a substantial training dataset. However, many real-life forecasting applications have very little initial observations, sometimes just 40 or fewer. Thus, the applicability of most forecasting methods is restricted in data-sparse commercial applications. While there is recent work in the setting of very limited initial data (so-called `zero-shot' forecasting), its performance is inconsistent depending on the data used for pretraining. In this work, we take a different approach and devise ForecastPFN, the first zero-shot forecasting model trained purely on a novel synthetic data distribution. ForecastPFN is a prior-data fitted network, trained to approximate Bayesian inference, which can make predictions on a new time series dataset in a single forward pass. Through extensive experiments, we show that zero-shot predictions made by ForecastPFN are more accurate and faster compared to state-of-the-art forecasting methods, even when the other methods are allowed to train on hundreds of additional in-distribution data points.

----

## [0] Exact Bayesian Inference on Discrete Models via Probability Generating Functions: A Probabilistic Programming Approach

**Authors**: *Fabian Zaiser, Andrzej S. Murawski, Chih-Hao Luke Ong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0747af6f877c0cb555fea595f01b0e83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0747af6f877c0cb555fea595f01b0e83-Abstract-Conference.html)

**Abstract**:

We present an exact Bayesian inference method for discrete statistical models, which can find exact solutions to a large class of discrete inference problems, even with infinite support and continuous priors.To express such models, we introduce a probabilistic programming language that supports discrete and continuous sampling, discrete observations, affine functions, (stochastic) branching, and conditioning on discrete events.Our key tool is probability generating functions:they provide a compact closed-form representation of distributions that are definable by programs, thus enabling the exact computation of posterior probabilities, expectation, variance, and higher moments.Our inference method is provably correct and fully automated in a tool called Genfer, which uses automatic differentiation (specifically, Taylor polynomials), but does not require computer algebra.Our experiments show that Genfer is often faster than the existing exact inference tools PSI, Dice, and Prodigy.On a range of real-world inference problems that none of these exact tools can solve, Genfer's performance is competitive with approximate Monte Carlo methods, while avoiding approximation errors.

----

## [0] SE(3) Equivariant Convolution and Transformer in Ray Space

**Authors**: *Yinshuang Xu, Jiahui Lei, Kostas Daniilidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html)

**Abstract**:

3D reconstruction and novel view rendering can greatly benefit from geometric priors when the input views are not sufficient in terms of coverage and inter-view baselines. Deep learning of geometric priors from 2D images requires each image to be represented in a $2D$ canonical frame and the prior to be learned in a given or learned $3D$ canonical frame. In this paper, given only the relative poses of the cameras, we show how to learn priors from multiple views equivariant to coordinate frame transformations by proposing an $SE(3)$-equivariant convolution and transformer in the space of rays in 3D. We model the ray space as a homogeneous space of $SE(3)$ and introduce the $SE(3)$-equivariant convolution in ray space. Depending on the output domain of the convolution, we present convolution-based $SE(3)$-equivariant maps from ray space to ray space and to $\mathbb{R}^3$. Our mathematical framework allows us to go beyond convolution to $SE(3)$-equivariant attention in the ray space. We showcase how to tailor and adapt the equivariant convolution and transformer in the tasks of equivariant $3D$ reconstruction and equivariant neural rendering from multiple views. We demonstrate $SE(3)$-equivariance by obtaining robust results in roto-translated datasets without performing transformation augmentation.

----

## [0] Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision

**Authors**: *Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David D. Cox, Yiming Yang, Chuang Gan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0764db1151b936aca59249e2c1386101-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0764db1151b936aca59249e2c1386101-Abstract-Conference.html)

**Abstract**:

Recent AI-assistant agents, such as ChatGPT, predominantly rely on supervised fine-tuning (SFT) with human annotations and reinforcement learning from human feedback (RLHF) to align the output of large language models (LLMs) with human intentions, ensuring they are helpful, ethical, and reliable. However, this dependence can significantly constrain the true potential of AI-assistant agents due to the high cost of obtaining human supervision and the related issues on quality, reliability, diversity, self-consistency, and undesirable biases. To address these challenges, we propose a novel approach called SELF-ALIGN, which combines principle-driven reasoning and the generative power of LLMs for the self-alignment of AI agents with minimal human supervision. Our approach encompasses four stages: first, we use an LLM to generate synthetic prompts, and a topic-guided method to augment the prompt diversity; second, we use a small set of human-written principles for AI models to follow, and guide the LLM through in-context learning from demonstrations (of principles application) to produce helpful, ethical, and reliable responses to user's queries; third, we fine-tune the original LLM with the high-quality self-aligned responses so that the resulting model can generate desirable responses for each query directly without the principle set and the demonstrations anymore; and finally, we offer a refinement step to address the issues of overly-brief or indirect responses. Applying SELF-ALIGN to the LLaMA-65b base language model, we develop an AI assistant named Dromedary. With fewer than 300 lines of human annotations (including < 200 seed prompts, 16 generic principles, and 5 exemplars for in-context learning). Dromedary significantly surpasses the performance of several state-of-the-art AI systems, including Text-Davinci-003 and Alpaca, on benchmark datasets with various settings.

----

## [0] Prototypical Variational Autoencoder for 3D Few-shot Object Detection

**Authors**: *Weiliang Tang, Biqi Yang, Xianzhi Li, Yun-Hui Liu, Pheng-Ann Heng, Chi-Wing Fu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/076a93fd42aa85f5ccee921a01d77dd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/076a93fd42aa85f5ccee921a01d77dd5-Abstract-Conference.html)

**Abstract**:

Few-Shot 3D Point Cloud Object Detection (FS3D) is a challenging task, aiming to detect 3D objects of novel classes using only limited annotated samples for training. Considering that the detection performance highly relies on the quality of the latent features, we design a VAE-based prototype learning scheme, named prototypical VAE (P-VAE), to learn a probabilistic latent space for enhancing the diversity and distinctiveness of the sampled features. The network encodes a multi-center GMM-like posterior, in which each distribution centers at a prototype. For regularization, P-VAE incorporates a reconstruction task to preserve geometric information. To adopt P-VAE for the detection framework, we formulate Geometric-informative Prototypical VAE (GP-VAE) to handle varying geometric components and Class-specific Prototypical VAE (CP-VAE) to handle varying object categories. In the first stage, we harness GP-VAE to aid feature extraction from the input scene. In the second stage, we cluster the geometric-informative features into per-instance features and use CP-VAE to refine each instance feature with category-level guidance. Experimental results show the top performance of our approach over the state of the arts on two FS3D benchmarks. Quantitative ablations and qualitative prototype analysis further demonstrate that our probabilistic modeling can significantly boost prototype learning for FS3D.

----

## [0] Double Gumbel Q-Learning

**Authors**: *David Yu-Tung Hui, Aaron C. Courville, Pierre-Luc Bacon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07956d40074d6523bad11112b3225c6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07956d40074d6523bad11112b3225c6e-Abstract-Conference.html)

**Abstract**:

We show that Deep Neural Networks introduce two heteroscedastic Gumbel noise sources into Q-Learning.  To account for these noise sources, we propose Double Gumbel Q-Learning, a Deep Q-Learning algorithm applicable for both discrete and continuous control.  In discrete control, we derive a closed-form expression for the loss function of our algorithm.  In continuous control, this loss function is intractable and we therefore derive an approximation with a hyperparameter whose value regulates pessimism in Q-Learning.  We present a default value for our pessimism hyperparameter that enables DoubleGum to outperform DDPG, TD3, SAC, XQL, quantile regression, and Mixture-of-Gaussian Critics in aggregate over 33 tasks from DeepMind Control, MuJoCo, MetaWorld, and Box2D and show that tuning this hyperparameter may further improve sample efficiency.

----

## [0] Mutual-Information Regularized Multi-Agent Policy Iteration

**Authors**: *Jiangxing Wang, Deheng Ye, Zongqing Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0799492e7be38b66d10ead5e8809616d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0799492e7be38b66d10ead5e8809616d-Abstract-Conference.html)

**Abstract**:

Despite the success of cooperative multi-agent reinforcement learning algorithms, most of them focus on a single team composition, which prevents them from being used in more realistic scenarios where dynamic team composition is possible. While some studies attempt to solve this problem via multi-task learning in a fixed set of team compositions, there is still a risk of overfitting to the training set, which may lead to catastrophic performance when facing dramatically varying team compositions during execution. To address this problem, we propose to use mutual information (MI) as an augmented reward to prevent individual policies from relying too much on team-related information and encourage agents to learn policies that are robust in different team compositions. Optimizing this MI-augmented objective in an off-policy manner can be intractable due to the existence of dynamic marginal distribution. To alleviate this problem, we first propose a multi-agent policy iteration algorithm with a fixed marginal distribution and prove its convergence and optimality. Then, we propose to employ the Blahutâ€“Arimoto algorithm and an imaginary team composition distribution for optimization with approximate marginal distribution as the practical implementation. Empirically, our method demonstrates strong zero-shot generalization to dynamic team compositions in complex cooperative tasks.

----

## [0] An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination

**Authors**: *Xue Yan, Jiaxian Guo, Xingzhou Lou, Jun Wang, Haifeng Zhang, Yali Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07a363fd2263091c2063998e0034999c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07a363fd2263091c2063998e0034999c-Abstract-Conference.html)

**Abstract**:

The goal of zero-shot human-AI coordination is to develop an agent that can collaborate with humans without relying on human data. Prevailing two-stage population-based methods require a diverse population of mutually distinct policies to simulate diverse human behaviors. The necessity of such populations severely limits their computational efficiency. To address this issue, we propose E3T, an Efficient End-to-End Training approach for zero-shot human-AI coordination. E3T employs a mixture of ego policy and random policy to construct the partner policy, making it both coordination-skilled and diverse. In this way, the ego agent is end-to-end trained with this mixture policy without the need of a pre-trained population, thus significantly improving the training efficiency.  In addition, a partner modeling module is proposed to predict the partner's action from historical information. With the predicted partner's action, the ego policy is able to adapt its policy and take actions accordingly when collaborating with humans of different behavior patterns. Empirical results on the Overcooked environment show that our method significantly improves the training efficiency while preserving comparable or superior performance than the population-based baselines. Demo videos are available at https://sites.google.com/view/e3t-overcooked.

----

## [0] Computing Optimal Equilibria and Mechanisms via Learning in Zero-Sum Extensive-Form Games

**Authors**: *Brian Hu Zhang, Gabriele Farina, Ioannis Anagnostides, Federico Cacciamani, Stephen McAleer, Andreas A. Haupt, Andrea Celli, Nicola Gatti, Vincent Conitzer, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07be1a0850e58ca29e2b6ce31fc0c791-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07be1a0850e58ca29e2b6ce31fc0c791-Abstract-Conference.html)

**Abstract**:

We introduce a new approach for computing optimal equilibria via learning in games. It applies to extensive-form settings with any number of players, including mechanism design, information design, and solution concepts such as correlated, communication, and certification equilibria. We observe that optimal equilibria are minimax equilibrium strategies of a player in an extensive-form zero-sum game. This reformulation allows to apply techniques for learning in zero-sum games, yielding the first learning dynamics that converge to optimal equilibria, not only in empirical averages, but also in iterates. We demonstrate the practical scalability and flexibility of our approach by attaining state-of-the-art performance in benchmark tabular games, and by computing an optimal mechanism for a sequential auction design problem using deep reinforcement learning.

----

## [0] Parts of Speech-Grounded Subspaces in Vision-Language Models

**Authors**: *James Oldfield, Christos Tzelepis, Yannis Panagakis, Mihalis Nicolaou, Ioannis Patras*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07cf32cf61224da628157b7ed0ce994a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07cf32cf61224da628157b7ed0ce994a-Abstract-Conference.html)

**Abstract**:

Latent image representations arising from vision-language models have proved immensely useful for a variety of downstream tasks. However, their utility is limited by their entanglement with respect to different visual attributes. For instance, recent work has shown that CLIP image representations are often biased toward specific visual properties (such as objects or actions) in an unpredictable manner. In this paper, we propose to separate representations of the different visual modalities in CLIP’s joint vision-language space by leveraging the association between parts of speech and specific visual modes of variation (e.g. nouns relate to objects, adjectives describe appearance). This is achieved by formulating an appropriate component analysis model that learns subspaces capturing variability corresponding to a specific part of speech, while jointly minimising variability to the rest. Such a subspace yields disentangled representations of the different visual properties of an image or text in closed form while respecting the underlying geometry of the manifold on which the representations lie. What’s more, we show the proposed model additionally facilitates learning subspaces corresponding to specific visual appearances (e.g. artists’ painting styles), which enables the selective removal of entire visual themes from CLIP-based text-to-image synthesis. We validate the model both qualitatively, by visualising the subspace projections with a text-to-image model and by preventing the imitation of artists’ styles, and quantitatively, through class invariance metrics and improvements to baseline zero-shot classification.

----

## [0] Searching for Optimal Per-Coordinate Step-sizes with Multidimensional Backtracking

**Authors**: *Frederik Kunstner, Victor Sanches Portella, Mark Schmidt, Nicholas J. A. Harvey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07e436cdeb48e2a67618274f5d5eff85-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07e436cdeb48e2a67618274f5d5eff85-Abstract-Conference.html)

**Abstract**:

The backtracking line-search is an effective technique to automatically tune the step-size in smooth optimization. It guarantees similar performance to using the theoretically optimal step-size. Many approaches have been developed to instead tune per-coordinate step-sizes, also known as diagonal preconditioners, but none of the existing methods are provably competitive with the optimal per-coordinate step-sizes. We propose multidimensional backtracking, an extension of the backtracking line-search to find good diagonal preconditioners for smooth convex problems. Our key insight is that the gradient with respect to the step-sizes, also known as hyper-gradients, yields separating hyperplanes that let us search for good preconditioners using cutting-plane methods. As black-box cutting-plane approaches like the ellipsoid method are computationally prohibitive, we develop an efficient algorithm tailored to our setting. Multidimensional backtracking is provably competitive with the best diagonal preconditioner and requires no manual tuning.

----

## [0] Estimating the Rate-Distortion Function by Wasserstein Gradient Descent

**Authors**: *Yibo Yang, Stephan Eckstein, Marcel Nutz, Stephan Mandt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07eea3fb833c905c5edf46f914231f15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07eea3fb833c905c5edf46f914231f15-Abstract-Conference.html)

**Abstract**:

In the theory of lossy compression, the rate-distortion (R-D) function $R(D)$ describes how much a data source can be compressed (in bit-rate) at any given level of fidelity (distortion). Obtaining $R(D)$ for a given data source establishes the fundamental performance limit for all compression algorithms.  We propose a new method to estimate $R(D)$ from the perspective of optimal transport. Unlike the classic Blahut--Arimoto algorithm which fixes the support of the reproduction distribution in advance, our Wasserstein gradient descent algorithm learns the support of the optimal reproduction distribution by moving particles. We prove its local convergence and analyze the sample complexity of our R-D estimator based on a connection to entropic optimal transport.  Experimentally, we obtain comparable or tighter bounds than state-of-the-art neural network methods on low-rate sources while requiring considerably less tuning and computation effort.  We also highlight a connection to maximum-likelihood deconvolution and introduce a new class of sources that can be used as test cases with known solutions to the R-D problem.

----

## [0] Epistemic Neural Networks

**Authors**: *Ian Osband, Zheng Wen, Seyed Mohammad Asghari, Vikranth Dwaracherla, Morteza Ibrahimi, Xiuyuan Lu, Benjamin Van Roy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07fbde96bee50f4e09303fd4f877c2f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07fbde96bee50f4e09303fd4f877c2f3-Abstract-Conference.html)

**Abstract**:

Intelligence relies on an agent's knowledge of what it does not know.This capability can be assessed based on the quality of joint predictions of labels across multiple inputs.In principle, ensemble-based approaches can produce effective joint predictions, but the computational costs of large ensembles become prohibitive.We introduce the epinet: an architecture that can supplement any conventional neural network, including large pretrained models, and can be trained with modest incremental computation to estimate uncertainty.With an epinet, conventional neural networks outperform very large ensembles, consisting of hundreds or more particles, with orders of magnitude less computation.The epinet does not fit the traditional framework of Bayesian neural networks.To accommodate development of approaches beyond BNNs, such as the epinet, we introduce the epistemic neural network (ENN) as a general interface for models that produce joint predictions.

----

## [0] HotBEV: Hardware-oriented Transformer-based Multi-View 3D Detector for BEV Perception

**Authors**: *Peiyan Dong, Zhenglun Kong, Xin Meng, Pinrui Yu, Yifan Gong, Geng Yuan, Hao Tang, Yanzhi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/081b08068e4733ae3e7ad019fe8d172f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/081b08068e4733ae3e7ad019fe8d172f-Abstract-Conference.html)

**Abstract**:

The bird's-eye-view (BEV) perception plays a critical role in autonomous driving systems, involving the accurate and efficient detection and tracking of objects from a top-down perspective. To achieve real-time decision-making in self-driving scenarios, low-latency computation is essential. While recent approaches to BEV detection have focused on improving detection precision using Lift-Splat-Shoot (LSS)-based or transformer-based schemas, the substantial computational and memory burden of these approaches increases the risk of system crashes when multiple on-vehicle tasks run simultaneously. Unfortunately, there is a dearth of literature on efficient BEV detector paradigms, let alone achieving realistic speedups.Unlike existing works that focus on reducing computation costs, this paper focuses on developing an efficient model design that prioritizes actual on-device latency.To achieve this goal, we propose a latency-aware design methodology that considers key hardware properties, such as memory access cost and degree of parallelism.Given the prevalence of GPUs as the main computation platform for autonomous driving systems, we develop a theoretical latency prediction model and introduce efficient building operators.By leveraging these operators and following an effective local-to-global visual modeling process, we propose a hardware-oriented backbone that is also optimized for strong feature capturing and fusing.Using these insights, we present a new hardware-oriented framework for efficient yet accurate camera-view BEV detectors.Experiments show that HotBEV achieves a 2\%$\sim$23\% NDS gain, and 2\%$\sim$7.8\% mAP gain with a 1.1$\times$$\sim$3.4$\times$ speedups compared to existing works on V100;On multiple GPU devices such as GPU GTX 2080 and the low-end GTX 1080, HotBEV achieves 1.1$\times$$\sim$6.3$\times$ faster than others.

----

## [0] Mip-Grid: Anti-aliased Grid Representations for Neural Radiance Fields

**Authors**: *Seungtae Nam, Daniel Rho, Jong Hwan Ko, Eunbyung Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/082d3d795520c43214da5123e56a3a34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/082d3d795520c43214da5123e56a3a34-Abstract-Conference.html)

**Abstract**:

Despite the remarkable achievements of neural radiance fields (NeRF) in representing 3D scenes and generating novel view images, the aliasing issue, rendering 'jaggies' or 'blurry' images at varying camera distances, remains unresolved in most existing approaches. The recently proposed mip-NeRF has effectively addressed this challenge by introducing integrated positional encodings (IPE). However, it relies on MLP architecture to represent the radiance fields, missing out on the fast training speed offered by the latest grid-based methods. In this work, we present mip-Grid, a novel approach that integrates anti-aliasing techniques into grid-based representations for radiance fields, mitigating the aliasing artifacts while enjoying fast training time. Notably, the proposed method uses a single-scale shared grid representation and a single-sampling approach, which only introduces minimal additions to the model parameters and computational costs. To handle scale ambiguity, mip-Grid generates multiple grids by applying simple convolution operations over the shared grid and uses the scale-aware coordinate to retrieve the appropriate features from the generated multiple grids. To test the effectiveness, we incorporated the proposed approach into the two recent representative grid-based methods, TensoRF and K-Planes. The experimental results demonstrated that mip-Grid greatly improved the rendering performance of both methods and showed comparable performance to mip-NeRF on multi-scale datasets while achieving significantly faster training time.

----

## [0] Theoretically Guaranteed Bidirectional Data Rectification for Robust Sequential Recommendation

**Authors**: *Yatong Sun, Bin Wang, Zhu Sun, Xiaochun Yang, Yan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08309150af77fc7c79ade0bf8bb6a562-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08309150af77fc7c79ade0bf8bb6a562-Abstract-Conference.html)

**Abstract**:

Sequential recommender systems (SRSs) are typically trained to predict the next item as the target given its preceding (and succeeding) items as the input. Such a paradigm assumes that every input-target pair is reliable for training. However, users can be induced to click on items that are inconsistent with their true preferences, resulting in unreliable instances, i.e., mismatched input-target pairs. Current studies on mitigating this issue suffer from two limitations: (i) they discriminate instance reliability according to models trained with unreliable data, yet without theoretical guarantees that such a seemingly contradictory solution can be effective; and (ii) most methods can only tackle either unreliable input or targets but fail to handle both simultaneously. To fill the gap, we theoretically unveil the relationship between SRS predictions and instance reliability, whereby two error-bounded strategies are proposed to rectify unreliable targets and input, respectively. On this basis, we devise a model-agnostic Bidirectional Data Rectification (BirDRec) framework, which can be flexibly implemented with most existing SRSs for robust training against unreliable data. Additionally, a rectification sampling strategy is devised and a self-ensemble mechanism is adopted to reduce the (time and space) complexity of BirDRec. Extensive experiments on four real-world datasets verify the generality, effectiveness, and efficiency of our proposed BirDRec.

----

## [0] Consistent Aggregation of Objectives with Diverse Time Preferences Requires Non-Markovian Rewards

**Authors**: *Silviu Pitis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08342dc6ab69f23167b4123086ad4d38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08342dc6ab69f23167b4123086ad4d38-Abstract-Conference.html)

**Abstract**:

As the capabilities of artificial agents improve, they are being increasingly deployed to service multiple diverse objectives and stakeholders. However, the composition of these objectives is often performed ad hoc, with no clear justification. This paper takes a normative approach to multi-objective agency: from a set of intuitively appealing axioms, it is shown that Markovian aggregation of Markovian reward functions is not possible when the time preference (discount factor) for each objective may vary. It follows that optimal multi-objective agents must admit rewards that are non-Markovian with respect to the individual objectives. To this end, a practical non-Markovian aggregation scheme is proposed, which overcomes the impossibility with only one additional parameter for each objective. This work offers new insights into sequential, multi-objective agency and intertemporal choice, and has practical implications for the design of AI systems deployed to serve multiple generations of principals with varying time preference.

----

## [0] Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability

**Authors**: *Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/088463cd3126aef2002ffc69da42ec59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/088463cd3126aef2002ffc69da42ec59-Abstract-Conference.html)

**Abstract**:

Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberatelymislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g. content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods.

----

## [0] InstanT: Semi-supervised Learning with Instance-dependent Thresholds

**Authors**: *Muyang Li, Runze Wu, Haoyu Liu, Jun Yu, Xun Yang, Bo Han, Tongliang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/088d99765bc121c6df215da7d45bc4e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/088d99765bc121c6df215da7d45bc4e9-Abstract-Conference.html)

**Abstract**:

Semi-supervised learning (SSL) has been a fundamental challenge in machine learning for decades. The primary family of SSL algorithms, known as pseudo-labeling, involves assigning pseudo-labels to confident unlabeled instances and incorporating them into the training set. Therefore, the selection criteria of confident instances are crucial to the success of SSL. Recently, there has been growing interest in the development of SSL methods that use dynamic or adaptive thresholds. Yet, these methods typically apply the same threshold to all samples, or use class-dependent thresholds for instances belonging to a certain class, while neglecting instance-level information. In this paper, we propose the study of instance-dependent thresholds, which has the highest degree of freedom compared with existing methods. Specifically, we devise a novel instance-dependent threshold function for all unlabeled instances by utilizing their instance-level ambiguity and the instance-dependent error rates of pseudo-labels, so instances that are more likely to have incorrect pseudo-labels will have higher thresholds. Furthermore, we demonstrate that our instance-dependent threshold function provides a bounded probabilistic guarantee for the correctness of the pseudo-labels it assigns.

----

## [0] Neural Lyapunov Control for Discrete-Time Systems

**Authors**: *Junlin Wu, Andrew Clark, Yiannis Kantaros, Yevgeniy Vorobeychik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08bf1773e94763b6cc366ee7c6582f27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08bf1773e94763b6cc366ee7c6582f27-Abstract-Conference.html)

**Abstract**:

While ensuring stability for linear systems is well understood, it remains a major challenge for nonlinear systems. A general approach in such cases is to compute a combination of a Lyapunov function and an associated control policy. However, finding Lyapunov functions for general nonlinear systems is a challenging task. To address this challenge, several methods have been proposed that represent Lyapunov functions using neural networks. However, such approaches either focus on continuous-time systems, or highly restricted classes of nonlinear dynamics. We propose the first approach for learning neural Lyapunov control in a broad class of discrete-time systems. Three key ingredients enable us to effectively learn provably stable control policies. The first is a novel mixed-integer linear programming approach for verifying the discrete-time Lyapunov stability conditions, leveraging the particular structure of these conditions. The second is a novel approach for computing verified sublevel sets. The third is a heuristic gradient-based method for quickly finding counterexamples to significantly speed up Lyapunov function learning. Our experiments on four standard benchmarks demonstrate that our approach significantly outperforms state-of-the-art baselines. For example, on the path tracking benchmark, we outperform recent neural Lyapunov control baselines by an order of magnitude in both running time and the size of the region of attraction, and on two of the four benchmarks (cartpole and PVTOL), ours is the first automated approach to return a provably stable controller. Our code is available at: https://github.com/jlwu002/nlc_discrete.

----

## [0] Information Maximization Perspective of Orthogonal Matching Pursuit with Applications to Explainable AI

**Authors**: *Aditya Chattopadhyay, Ryan Pilgrim, René Vidal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08eac13583b310ec55d755f99c549be3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08eac13583b310ec55d755f99c549be3-Abstract-Conference.html)

**Abstract**:

Information Pursuit (IP) is a classical active testing algorithm for predicting an output by sequentially and greedily querying the input in order of information gain. However, IP is computationally intensive since it involves estimating mutual information in high-dimensional spaces. This paper explores Orthogonal Matching Pursuit (OMP) as an alternative to IP for greedily selecting the queries. OMP is a classical signal processing algorithm for sequentially encoding a signal in terms of dictionary atoms chosen in order of correlation gain. In each iteration, OMP selects the atom that is most correlated with the signal residual (the signal minus its reconstruction thus far). Our first contribution is to establish a fundamental connection between IP and OMP, where we prove that IP with random projections of dictionary atoms as queries ``almost'' reduces to OMP, with the difference being that IP selects atoms in order of normalized correlation gain. We call this version IP-OMP and present simulations indicating that this difference does not have any appreciable effect on the sparse code recovery rate of IP-OMP compared to that of OMP for random Gaussian dictionaries. Inspired by this connection, our second contribution is to explore the utility of IP-OMP for generating explainable predictions, an area in which IP has recently gained traction. More specifically, we propose a simple explainable AI algorithm which encodes an image as a sparse combination of semantically meaningful dictionary atoms that are defined as text embeddings of interpretable concepts. The final prediction is made using the weights of this sparse combination, which serve as an explanation. Empirically, our proposed algorithm is not only competitive with existing explainability methods but also computationally less expensive.

----

## [0] Evolving Connectivity for Recurrent Spiking Neural Networks

**Authors**: *Guan Wang, Yuhao Sun, Sijie Cheng, Sen Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08f9de0232c0b485110237f6e6cf88f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08f9de0232c0b485110237f6e6cf88f1-Abstract-Conference.html)

**Abstract**:

Recurrent spiking neural networks (RSNNs) hold great potential for advancing artificial general intelligence, as they draw inspiration from the biological nervous system and show promise in modeling complex dynamics.However, the widely-used surrogate gradient-based training methods for RSNNs are inherently inaccurate and unfriendly to neuromorphic hardware.To address these limitations, we propose the evolving connectivity (EC) framework, an inference-only method for training RSNNs.The EC framework reformulates weight-tuning as a search into parameterized connection probability distributions, and employs Natural Evolution Strategies (NES) for optimizing these distributions.Our EC framework circumvents the need for gradients and features hardware-friendly characteristics, including sparse boolean connections and high scalability.We evaluate EC on a series of standard robotic locomotion tasks, where it achieves comparable performance with deep neural networks and outperforms gradient-trained RSNNs, even solving the complex 17-DoF humanoid task.Additionally, the EC framework demonstrates a two to three fold speedup in efficiency compared to directly evolving parameters.By providing a performant and hardware-friendly alternative, the EC framework lays the groundwork for further energy-efficient applications of RSNNs and advances the development of neuromorphic devices.Our code is publicly available at https://github.com/imoneoi/EvolvingConnectivity.

----

## [0] Bayesian Optimization with Cost-varying Variable Subsets

**Authors**: *Sebastian Tay, Chuan Sheng Foo, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/090b23d52bc2722eef2fbf79c5ebf9ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/090b23d52bc2722eef2fbf79c5ebf9ec-Abstract-Conference.html)

**Abstract**:

We introduce the problem of Bayesian optimization with cost-varying variable subsets (BOCVS) where in each iteration, the learner chooses a subset of query variables and specifies their values while the rest are randomly sampled. Each chosen subset has an associated cost. This presents the learner with the novel challenge of balancing between choosing more informative subsets for more directed learning versus leaving some variables to be randomly sampled to reduce incurred costs. This paper presents a novel Gaussian process upper confidence bound-based algorithm for solving the BOCVS problem that is provably no-regret. We analyze how the availability of cheaper control sets helps in exploration and reduces overall regret. We empirically show that our proposed algorithm can find significantly better solutions than comparable baselines with the same budget.

----

## [0] Transformed Low-Rank Parameterization Can Help Robust Generalization for Tensor Neural Networks

**Authors**: *Andong Wang, Chao Li, Mingyuan Bai, Zhong Jin, Guoxu Zhou, Qibin Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html)

**Abstract**:

Multi-channel learning has gained significant attention in recent applications, where neural networks with t-product layers (t-NNs) have shown promising performance through novel feature mapping in the transformed domain. However, despite the practical success of t-NNs, the theoretical analysis of their generalization remains unexplored. We address this gap by deriving upper bounds on the generalization error of t-NNs in both standard and adversarial settings. Notably, it reveals that t-NNs compressed with exact transformed low-rank parameterization can achieve tighter adversarial generalization bounds compared to non-compressed models. While exact transformed low-rank weights are rare in practice, the analysis demonstrates that through adversarial training with gradient flow, highly over-parameterized t-NNs with the ReLU activation can be implicitly regularized towards a transformed low-rank parameterization under certain conditions. Moreover, this paper establishes sharp adversarial generalization bounds for t-NNs with approximately transformed low-rank weights. Our analysis highlights the potential of transformed low-rank parameterization in enhancing the robust generalization of t-NNs, offering valuable insights for further research and development.

----

## [0] Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples

**Authors**: *Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Mehran Kazemi, Najoung Kim, He He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09425891e393e64b0535194a81ba15b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/09425891e393e64b0535194a81ba15b7-Abstract-Conference.html)

**Abstract**:

Given the intractably large size of the space of proofs, any model that is capable of general deductive reasoning must generalize to proofs of greater complexity. Recent studies have shown that large language models (LLMs) possess some abstract deductive reasoning ability given chain-of-thought prompts. However, they have primarily been tested on proofs using modus ponens or of a specific size, and from the same distribution as the in-context examples. To measure the general deductive reasoning ability of LLMs, we test on a broad set of deduction rules and measure their ability to generalize to more complex proofs from simpler demonstrations from multiple angles: depth-, width-, and compositional generalization. To facilitate systematic exploration, we construct a new synthetic and programmable reasoning dataset that enables control over deduction rules and proof complexity. Our experiments on four LLMs of various sizes and training objectives show that they are able to generalize to compositional proofs. However, they have difficulty generalizing to longer proofs, and they require explicit demonstrations to produce hypothetical subproofs, specifically in proof by cases and proof by contradiction.

----

## [0] MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining

**Authors**: *Jacob Portes, Alexander Trott, Sam Havens, Daniel King, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, Jonathan Frankle*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/095a6917768712b7ccc61acbeecad1d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/095a6917768712b7ccc61acbeecad1d8-Abstract-Conference.html)

**Abstract**:

Although BERT-style encoder models are heavily used in NLP research, many researchers do not pretrain their own BERTs from scratch due to the high cost of training. In the past half-decade since BERT first rose to prominence, many advances have been made with other transformer architectures and training configurations that have yet to be systematically incorporated into BERT. Here, we introduce MosaicBERT, a BERT-style encoder architecture and training recipe that is empirically optimized for fast pretraining. This efficient architecture incorporates FlashAttention, Attention with Linear Biases (ALiBi), Gated Linear Units (GLU), a module to dynamically remove padded tokens, and low precision LayerNorm into the classic transformer encoder block. The training recipe includes a 30% masking ratio for the Masked Language Modeling (MLM) objective, bfloat16 precision, and vocabulary size optimized for GPU throughput, in addition to best-practices from RoBERTa and other encoder models. When pretrained from scratch on the C4 dataset, this base model achieves a downstream average GLUE (dev) score of 79.6 in 1.13 hours on 8 A100 80 GB GPUs at a cost of roughly $20. We plot extensive accuracy vs. pretraining speed Pareto curves and show that MosaicBERT base and large are consistently Pareto optimal when compared to a competitive BERT base and large. This empirical speed up in pretraining enables researchers and engineers to pretrain custom BERT-style models at low cost instead of finetune on existing generic models. We open source our model weights and code.

----

## [0] GraphMP: Graph Neural Network-based Motion Planning with Efficient Graph Search

**Authors**: *Xiao Zang, Miao Yin, Jinqi Xiao, Saman A. Zonouz, Bo Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/096961cae3c3423c44ea045aeb584e05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/096961cae3c3423c44ea045aeb584e05-Abstract-Conference.html)

**Abstract**:

Motion planning, which aims to find a high-quality collision-free path in the configuration space, is a fundamental task in robotic systems. Recently, learning-based motion planners, especially the graph neural network-powered, have shown promising planning performance. However, though the state-of-the-art GNN planner can efficiently extract and learn graph information, its inherent mechanism is not well suited for graph search process, hindering its further performance improvement. To address this challenge and fully unleash the potential of GNN in motion planning, this paper proposes GraphMP, a neural motion planner for both low and high-dimensional planning tasks. With the customized model architecture and training mechanism design, GraphMP can simultaneously perform efficient graph pattern extraction and graph search processing, leading to strong planning performance. Experiments on a variety of environments, ranging from 2D Maze to 14D dual KUKA robotic arm, show that our proposed GraphMP achieves significant improvement on path quality and planning speed over the state-of-the-art learning-based and classical planners; while preserving the competitive success rate.

----

## [0] Accountability in Offline Reinforcement Learning: Explaining Decisions with a Corpus of Examples

**Authors**: *Hao Sun, Alihan Hüyük, Daniel Jarrett, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/096b1019463f34eb241e87cfce8dfe16-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/096b1019463f34eb241e87cfce8dfe16-Abstract-Conference.html)

**Abstract**:

Learning controllers with offline data in decision-making systems is an essential area of research due to its potential to reduce the risk of applications in real-world systems. However, in responsibility-sensitive settings such as healthcare, decision accountability is of paramount importance, yet has not been adequately addressed by the literature.This paper introduces the Accountable Offline Controller (AOC) that employs the offline dataset as the Decision Corpus and performs accountable control based on a tailored selection of examples, referred to as the Corpus Subset. AOC operates effectively in low-data scenarios, can be extended to the strictly offline imitation setting, and displays qualities of both conservation and adaptability.We assess AOC's performance in both simulated and real-world healthcare scenarios, emphasizing its capability to manage offline control tasks with high levels of performance while maintaining accountability.

----

## [0] Synthcity: a benchmark framework for diverse use cases of tabular synthetic data

**Authors**: *Zhaozhi Qian, Robert Davis, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09723c9f291f6056fd1885081859c186-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/09723c9f291f6056fd1885081859c186-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Accessible high-quality data is the bread and butter of machine learning research, and the demand for data has exploded as larger and more advanced ML models are built across different domains. Yet, real data often contain sensitive information, are subject to various biases, and are costly to acquire, which compromise their quality and accessibility. Synthetic data have thus emerged as a complement to, sometimes even a replacement for, real data for ML training. However, the landscape of synthetic data research has been fragmented due to the diverse range of data modalities, such as tabular, time series, and images, and the wide array of use cases, including privacy preservation, fairness considerations, and data augmentation. This fragmentation poses practical challenges when comparing and selecting synthetic data generators in for different problem settings. To this end, we develop Synthcity, an open-source Python library that allows researchers and practitioners to perform one-click benchmarking of synthetic data generators across data modalities and use cases. Beyond benchmarking, Synthcity serves as a centralized toolkit for accessing cutting-edge data generators. In addition, Synthcityâ€™s flexible plug-in style API makes it easy to incorporate additional data generators into the framework. Using examples of tabular data generation and data augmentation, we illustrate the general applicability of Synthcity, and the insight one can obtain.

----

## [0] SOAR: Improved Indexing for Approximate Nearest Neighbor Search

**Authors**: *Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, Sanjiv Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0973524e02a712af33325d0688ae6f49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0973524e02a712af33325d0688ae6f49-Abstract-Conference.html)

**Abstract**:

This paper introduces SOAR: Spilling with Orthogonality-Amplified Residuals, a novel data indexing technique for approximate nearest neighbor (ANN) search. SOAR extends upon previous approaches to ANN search, such as spill trees, that utilize multiple redundant representations while partitioning the data to reduce the probability of missing a nearest neighbor during search. Rather than training and computing these redundant representations independently, however, SOAR uses an orthogonality-amplified residual loss, which optimizes each representation to compensate for cases where other representations perform poorly. This drastically improves the overall index quality, resulting in state-of-the-art ANN benchmark performance while maintaining fast indexing times and low memory consumption.

----

## [0] Type-to-Track: Retrieve Any Object via Prompt-based Tracking

**Authors**: *Pha Nguyen, Kha Gia Quach, Kris Kitani, Khoa Luu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html)

**Abstract**:

One of the recent trends in vision problems is to use natural language captions to describe the objects of interest. This approach can overcome some limitations of traditional methods that rely on bounding boxes or category annotations. This paper introduces a novel paradigm for Multiple Object Tracking called Type-to-Track, which allows users to track objects in videos by typing natural language descriptions. We present a new dataset for that Grounded Multiple Object Tracking task, called GroOT, that contains videos with various types of objects and their corresponding textual captions describing their appearance and action in detail. Additionally, we introduce two new evaluation protocols and formulate evaluation metrics specifically for this task. We develop a new efficient method that models a transformer-based eMbed-ENcoDE-extRact framework (MENDER) using the third-order tensor decomposition. The experiments in five scenarios show that our MENDER approach outperforms another two-stage design in terms of accuracy and efficiency, up to 14.7\% accuracy and $4\times$ speed faster.

----

## [0] Finding Counterfactually Optimal Action Sequences in Continuous State Spaces

**Authors**: *Stratis Tsirtsis, Manuel Rodriguez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09ae6beae5f1ff38f05c05979097ea0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/09ae6beae5f1ff38f05c05979097ea0f-Abstract-Conference.html)

**Abstract**:

Whenever a clinician reflects on the efficacy of a sequence of treatment decisions for a patient, they may try to identify critical time steps where, had they made different decisions, the patient's health would have improved. While recent methods at the intersection of causal inference and reinforcement learning promise to aid human experts, as the clinician above, to retrospectively analyze sequential decision making processes, they have focused on environments with finitely many discrete states. However, in many practical applications, the state of the environment is inherently continuous in nature. In this paper, we aim to fill this gap. We start by formally characterizing a sequence of discrete actions and continuous states using finite horizon Markov decision processes and a broad class of bijective structural causal models. Building upon this characterization, we formalize the problem of finding counterfactually optimal action sequences and show that, in general, we cannot expect to solve it in polynomial time. Then, we develop a search method based on the A* algorithm that, under a natural form of Lipschitz continuity of the environmentâ€™s dynamics, is guaranteed to return the optimal solution to the problem. Experiments on real clinical data show that our method is very efficient in practice, and it has the potential to offer interesting insights for sequential decision making tasks.

----

## [0] Reusing Pretrained Models by Multi-linear Operators for Efficient Training

**Authors**: *Yu Pan, Ye Yuan, Yichun Yin, Zenglin Xu, Lifeng Shang, Xin Jiang, Qun Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09d9a13f7018110cfb439c06b07940a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/09d9a13f7018110cfb439c06b07940a2-Abstract-Conference.html)

**Abstract**:

Training large models from scratch usually costs a substantial amount of resources. Towards this problem, recent studies such as bert2BERT and LiGO have reused small pretrained models to initialize a large model (termed the ``target model''), leading to a considerable acceleration in training. Despite the successes of these previous studies, they  grew pretrained models by mapping partial weights only, ignoring potential correlations across the entire model. As we show in this paper, there are inter- and intra-interactions among the weights of both the pretrained and the target models. As a result, the partial mapping may not capture the complete information and lead to inadequate growth. In this paper, we propose a method that linearly correlates each weight of the target model to all the weights of the pretrained model to further enhance acceleration ability. We utilize multi-linear operators to reduce computational and spacial complexity, enabling acceptable resource requirements. Experiments demonstrate that our method can save 76\% computational costs on DeiT-base transferred from DeiT-small, which outperforms bert2BERT by +12\% and LiGO by +21\%, respectively.

----

## [0] Tartarus: A Benchmarking Platform for Realistic And Practical Inverse Molecular Design

**Authors**: *AkshatKumar Nigam, Robert Pollice, Gary Tom, Kjell Jorner, John Willes, Luca A. Thiede, Anshul Kundaje, Alán Aspuru-Guzik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09f8b2469a3d1089a7c60d9ef1983271-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/09f8b2469a3d1089a7c60d9ef1983271-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The efficient exploration of chemical space to design molecules with intended properties enables the accelerated discovery of drugs, materials, and catalysts, and is one of the most important outstanding challenges in chemistry. Encouraged by the recent surge in computer power and artificial intelligence development, many algorithms have been developed to tackle this problem. However, despite the emergence of many new approaches in recent years, comparatively little progress has been made in developing realistic benchmarks that reflect the complexity of molecular design for real-world applications. In this work, we develop a set of practical benchmark tasks relying on physical simulation of molecular systems mimicking real-life molecular design problems for materials, drugs, and chemical reactions. Additionally, we demonstrate the utility and ease of use of our new benchmark set by demonstrating how to compare the performance of several well-established families of algorithms. Overall, we believe that our benchmark suite will help move the field towards more realistic molecular design benchmarks, and move the development of inverse molecular design algorithms closer to the practice of designing molecules that solve existing problems in both academia and industry alike.

----

## [0] DreamSparse: Escaping from Plato's Cave with 2D Diffusion Model Given Sparse Views

**Authors**: *Paul Yoo, Jiaxian Guo, Yutaka Matsuo, Shixiang Shane Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a003511b09274348b8117f5f3b94c93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a003511b09274348b8117f5f3b94c93-Abstract-Conference.html)

**Abstract**:

Synthesizing novel view images from a few views is a challenging but practical problem. Existing methods often struggle with producing high-quality results or necessitate per-object optimization in such few-view settings due to the insufficient information provided.  In this work, we explore leveraging the strong 2D priors in pre-trained diffusion models for synthesizing novel view images. 2D diffusion models, nevertheless, lack 3D awareness, leading to distorted image synthesis and compromising the identity. To address these problems, we propose $\textit{DreamSparse}$, a framework that enables the frozen pre-trained diffusion model to generate geometry and identity-consistent novel view images. Specifically, DreamSparse incorporates a geometry module designed to capture features about spatial information from sparse views as a 3D prior. Subsequently, a spatial guidance model is introduced to convert rendered feature maps as spatial information for the generative process. This information is then used to guide the pre-trained diffusion model toencourage the synthesis of geometrically consistent images without further tuning. Leveraging the strong image priors in the pre-trained diffusion models, DreamSparse is capable of synthesizing high-quality novel views for both object and object-centric scene-level images and generalising to open-set images.Experimental results demonstrate that our framework can effectively synthesize novel view images from sparse views and outperforms baselines in both trained and open-set category images. More results can be found on our project page: https://sites.google.com/view/dreamsparse-webpage.

----

## [0] Sample Complexity Bounds for Score-Matching: Causal Discovery and Generative Modeling

**Authors**: *Zhenyu Zhu, Francesco Locatello, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a3dc35a2391cabcb59a6b123544e3db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a3dc35a2391cabcb59a6b123544e3db-Abstract-Conference.html)

**Abstract**:

This paper provides statistical sample complexity bounds for score-matching and its applications in causal discovery. We demonstrate that accurate estimation of the score function is achievable by training a standard deep ReLU neural network using stochastic gradient descent. We establish bounds on the error rate of recovering causal relationships using the score-matching-based causal discovery method of Rolland et al. [2022], assuming a sufficiently good estimation of the score function. Finally, we analyze the upper bound of score-matching estimation within the score-based generative modeling, which has been applied for causal discovery but is also of independent interest within the domain of generative models.

----

## [0] Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach

**Authors**: *Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a443a000e1cb2281480b3bac395b3b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a443a000e1cb2281480b3bac395b3b8-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) are vulnerable to adversarial perturbations, including those that affect both node features and graph topology. This paper investigates GNNs derived from diverse neural flows, concentrating on their connection to various stability notions such as BIBO stability, Lyapunov stability, structural stability, and conservative stability. We argue that Lyapunov stability, despite its common use, does not necessarily ensure adversarial robustness. Inspired by physics principles, we advocate for the use of conservative Hamiltonian neural flows to construct GNNs that are robust to adversarial attacks. The adversarial robustness of different neural flow GNNs is empirically compared on several benchmark datasets under a variety of adversarial attacks. Extensive numerical experiments demonstrate that GNNs leveraging conservative Hamiltonian flows with Lyapunov stability substantially improve robustness against adversarial perturbations. The implementation code of experiments  is available at \url{https://github.com/zknus/NeurIPS-2023-HANG-Robustness}.

----

## [0] A Path to Simpler Models Starts With Noise

**Authors**: *Lesia Semenova, Harry Chen, Ronald Parr, Cynthia Rudin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a49935d2b3d3342ca08d6db0adcfa34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a49935d2b3d3342ca08d6db0adcfa34-Abstract-Conference.html)

**Abstract**:

The Rashomon set is the set of models that perform approximately equally well on a given dataset, and the Rashomon ratio is the fraction of all models in a given hypothesis space that are in the Rashomon set. Rashomon ratios are often large for tabular datasets in criminal justice, healthcare, lending, education, and in other areas, which has practical implications about whether simpler models can attain the same level of accuracy as more complex models. An open question is why Rashomon ratios often tend to be large. In this work, we propose and study a mechanism of the data generation process, coupled with choices usually made by the analyst during the learning process, that determines the size of the Rashomon ratio. Specifically, we demonstrate that noisier datasets lead to larger Rashomon ratios through the way that practitioners train models. Additionally, we introduce a measure called pattern diversity, which captures the average difference in predictions between distinct classification patterns in the Rashomon set, and motivate why it tends to increase with label noise. Our results explain a key aspect of why simpler models often tend to perform as well as black box models on complex, noisier datasets.

----

## [0] Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model

**Authors**: *Zirui Liu, Guanchu Wang, Shaochen (Henry) Zhong, Zhaozhuo Xu, Daochen Zha, Ruixiang (Ryan) Tang, Zhimeng Stephen Jiang, Kaixiong Zhou, Vipin Chaudhary, Shuai Xu, Xia Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html)

**Abstract**:

As the model size grows rapidly, fine-tuning the large pre-trained language model has become increasingly difficult due to its extensive memory usage. Previous works usually focus on reducing the number of trainable parameters in the network. While the model parameters do contribute to memory usage, the primary memory bottleneck during training arises from storing feature maps, also known as activations, as they are crucial for gradient calculation. Notably, machine learning models are typically trained using stochastic gradient descent.We argue that in stochastic optimization, models can handle noisy gradients as long as the gradient estimator is unbiased with reasonable variance.Following this motivation, we propose a new family of unbiased estimators called \sas, for matrix production with reduced variance, which only requires storing the sub-sampled activations for calculating the gradient.Our work provides both theoretical and experimental evidence that, in the context of tuning transformers, our proposed estimators exhibit lower variance compared to existing ones.By replacing the linear operation with our approximated one in transformers, we can achieve up to 2.7X peak memory reduction with almost no accuracy drop and enables up to $6.4\times$ larger batch size.Under the same hardware, \sas enables better down-streaming task performance by applying larger models and/or faster training speed with larger batch sizes.The code is available at https://anonymous.4open.science/r/WTACRS-A5C5/.

----

## [0] Zeroth-Order Methods for Nondifferentiable, Nonconvex, and Hierarchical Federated Optimization

**Authors**: *Yuyang Qiu, Uday V. Shanbhag, Farzad Yousefian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a70c9cd8179fe6f8f6135fafa2a8798-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a70c9cd8179fe6f8f6135fafa2a8798-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) has emerged as an enabling framework for communication-efficient decentralized training. We study three broadly applicable problem classes in FL: (i) Nondifferentiable nonconvex federated optimization; (ii) Federated bilevel optimization; (iii) Federated minimax problems. Notably, in an implicit sense, both (ii) and (iii) are instances of (i). However, the hierarchical problems in (ii) and (iii) are often complicated by the absence of a closed-form expression for the implicit objective function. Unfortunately, research on these problems has been limited and afflicted by reliance on strong assumptions, including the need for differentiability and L-smoothness of the implicit function. We address this shortcoming by making the following contributions. In (i), by leveraging convolution-based smoothing and Clarkeâ€™s subdifferential calculus, we devise a randomized smoothing-enabled zeroth-order FL method and derive communication and iteration complexity guarantees for computing an approximate Clarke stationary point.  To contend with (ii) and (iii), we devise a unified randomized implicit zeroth-order FL framework, equipped with explicit communication and iteration complexities. Importantly, our method utilizes delays during local steps to skip making calls to the inexact lower-level FL oracle. This results in significant reduction in communication overhead when addressing hierarchical problems. We empirically validate the theory on nonsmooth and hierarchical ML problems.

----

## [0] Language Model Alignment with Elastic Reset

**Authors**: *Michael Noukhovitch, Samuel Lavoie, Florian Strub, Aaron C. Courville*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a980183c520446f6b8afb6fa2a2c70e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a980183c520446f6b8afb6fa2a2c70e-Abstract-Conference.html)

**Abstract**:

Finetuning language models with reinforcement learning (RL), e.g. from human feedback (HF), is a prominent method for alignment. But optimizing against a reward model can improve on reward while degrading performance in other areas, a phenomenon known as reward hacking, alignment tax, or language drift. First, we argue that commonly-used test metrics are insufficient and instead measure how different algorithms tradeoff between reward and drift. The standard method modified the reward with a Kullback-Lieber (KL) penalty between the online and initial model. We propose Elastic Reset, a new algorithm that achieves higher reward with less drift without explicitly modifying the training objective. We periodically reset the online model to an exponentially moving average (EMA) of itself, then reset the EMA model to the initial model. Through the use of an EMA, our model recovers quickly after resets and achieves higher reward with less drift in the same number of steps. We demonstrate that fine-tuning language models with Elastic Reset leads to state-of-the-art performance on a small scale pivot-translation benchmark, outperforms all baselines in a medium-scale RLHF-like IMDB mock sentiment task and leads to a more performant and more aligned technical QA chatbot with LLaMA-7B. Code available https://github.com/mnoukhov/elastic-reset

----

## [0] Resolving the Tug-of-War: A Separation of Communication and Learning in Federated Learning

**Authors**: *Junyi Li, Heng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0aa800df4298539770b57824afc77a89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0aa800df4298539770b57824afc77a89-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is a promising privacy-preserving machine learning paradigm over distributed data. In this paradigm, each client trains the parameter of a model locally and the server aggregates the parameter from clients periodically. Therefore, we perform the learning and communication over the same set of parameters. However, we find that learning and communication have fundamentally divergent requirements for parameter selection, akin to two opposite teams in a tug-of-war game. To mitigate this discrepancy, we introduce FedSep, a novel two-layer federated learning framework. FedSep consists of separated communication and learning layers for each client and the two layers are connected through decode/encode operations. In particular, the decoding operation is formulated as a minimization problem. We view FedSep as a federated bilevel optimization problem and propose an efficient algorithm to solve it. Theoretically, we demonstrate that its convergence matches that of the standard FL algorithms. The separation of communication and learning in FedSep offers innovative solutions to various challenging problems in FL, such as Communication-Efficient FL and Heterogeneous-Model FL. Empirical validation shows the superior performance of FedSep over various baselines in these tasks.

----

## [0] GlucoSynth: Generating Differentially-Private Synthetic Glucose Traces

**Authors**: *Josephine Lamp, Mark Derdzinski, Christopher Hannemann, Joost van der Linden, Lu Feng, Tianhao Wang, David E. Evans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ab51646ca369140c3c3ece011b66587-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ab51646ca369140c3c3ece011b66587-Abstract-Conference.html)

**Abstract**:

We focus on the problem of generating high-quality, private synthetic glucose traces, a task generalizable to many other time series sources. Existing methods for time series data synthesis, such as those using Generative Adversarial Networks (GANs), are not able to capture the innate characteristics of glucose data and cannot provide any formal privacy guarantees without severely degrading the utility of the synthetic data. In this paper we present GlucoSynth, a novel privacy-preserving GAN framework to generate synthetic glucose traces. The core intuition behind our approach is to conserve relationships amongst motifs (glucose events) within the traces, in addition to temporal dynamics. Our framework incorporates differential privacy mechanisms to provide strong formal privacy guarantees. We provide a comprehensive evaluation on the real-world utility of the data using 1.2 million glucose traces; GlucoSynth outperforms all previous methods in its ability to generate high-quality synthetic glucose traces with strong privacy guarantees.

----

## [0] OBJECT 3DIT: Language-guided 3D-aware Image Editing

**Authors**: *Oscar Michel, Anand Bhattad, Eli VanderBilt, Ranjay Krishna, Aniruddha Kembhavi, Tanmay Gupta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b0153a91f827b14e8bfea4e211362f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b0153a91f827b14e8bfea4e211362f3-Abstract-Conference.html)

**Abstract**:

Existing image editing tools, while powerful, typically disregard the underlying 3D geometry from which the image is projected. As a result, edits made using these tools may become detached from the geometry and lighting conditions that are at the foundation of the image formation process; such edits break the portrayal of a coherent 3D world. 3D-aware generative models are a promising solution, but currently only succeed on small datasets or at the level of a single object. In this work, we formulate the new task of language-guided 3D-aware editing, where objects in an image should be edited according to a language instruction while remaining consistent with the underlying 3D scene. To promote progress towards this goal, we release OBJect: a benchmark dataset of 400K editing examples created from procedurally generated 3D scenes. Each example consists of an input image, editing instruction in language, and the edited image. We also introduce 3DIT: single and multi-task models for four editing tasks. Our models show impressive abilities to understand the 3D composition of entire scenes, factoring in surrounding objects, surfaces, lighting conditions, shadows, and physically-plausible object configurations. Surprisingly, training on only synthetic scenes from \dataset, editing capabilities of 3DIT generalize to real-world images.

----

## [0] Learning Rule-Induced Subgraph Representations for Inductive Relation Prediction

**Authors**: *Tianyu Liu, Qitan Lv, Jie Wang, Shuling Yang, Hanzhu Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b06c8673ebb453e5e468f7743d8f54e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b06c8673ebb453e5e468f7743d8f54e-Abstract-Conference.html)

**Abstract**:

Inductive relation prediction (IRP)---where entities can be different during training and inference---has shown great power for completing evolving knowledge graphs. Existing works mainly focus on using graph neural networks (GNNs) to learn the representation of the subgraph induced from the target link, which can be seen as an implicit rule-mining process to measure the plausibility of the target link. However, these methods are not able to differentiate the target link and other links during message passing, hence the final subgraph representation will contain irrelevant rule information to the target link, which reduces the reasoning performance and severely hinders the applications for real-world scenarios. To tackle this problem, we propose a novel $\textit{single-source edge-wise}$ GNN model to learn the $\textbf{R}$ule-induc$\textbf{E}$d $\textbf{S}$ubgraph represen$\textbf{T}$ations $(\textbf{REST}$), which encodes relevant rules and eliminates irrelevant rules within the subgraph. Specifically, we propose a $\textit{single-source}$ initialization approach to initialize edge features only for the target link, which guarantees the relevance of mined rules and target link. Then we propose several RNN-based functions for $\textit{edge-wise}$ message passing to model the sequential property of mined rules. REST is a simple and effective approach with theoretical support to learn the $\textit{rule-induced subgraph representation}$. Moreover, REST does not need node labeling, which significantly accelerates the subgraph preprocessing time by up to $\textbf{11.66}\times$. Experiments on inductive relation prediction benchmarks demonstrate the effectiveness of our REST.

----

## [0] Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment

**Authors**: *Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, Gal Chechik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b08d733a5d45a547344c4e9d88bb8bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b08d733a5d45a547344c4e9d88bb8bc-Abstract-Conference.html)

**Abstract**:

Text-conditioned image generation models often generate incorrect associations between entities and their visual attributes. This reflects an impaired mapping between linguistic binding of entities and modifiers in the prompt and visual binding of the corresponding elements in the generated image. As one example, a query like ``a pink sunflower and a yellow flamingo'' may incorrectly produce an image of a yellow sunflower and a pink flamingo. To remedy this issue, we propose SynGen, an approach which first syntactically analyses the prompt to identify entities and their modifiers, and then uses a novel loss function that encourages the cross-attention maps to agree with the linguistic binding reflected by the syntax. Specifically, we encourage large overlap between attention maps of entities and their modifiers, and small overlap with other entities and modifier words. The loss is optimized during inference, without retraining or fine-tuning the model. Human evaluation on three datasets, including one new and challenging set, demonstrate significant improvements of SynGen compared with current state of the art methods. This work highlights how making use of sentence structure during inference can efficiently and substantially improve the faithfulness of text-to-image generation.

----

## [0] Optimistic Natural Policy Gradient: a Simple Efficient Policy Optimization Framework for Online RL

**Authors**: *Qinghua Liu, Gellért Weisz, András György, Chi Jin, Csaba Szepesvári*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b13c22ca208bc08f3fd13793292f25f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b13c22ca208bc08f3fd13793292f25f-Abstract-Conference.html)

**Abstract**:

While policy optimization algorithms have played an important role in recent empirical success of Reinforcement Learning (RL), the existing theoretical understanding of policy optimization remains rather limited---they are either restricted to tabular MDPs or suffer from highly suboptimal sample complexity, especial in online RL where exploration is necessary. This paper proposes a simple efficient policy optimization framework---Optimistic NPG for online RL. Optimistic NPG can be viewed as simply combining of the classic natural policy gradient (NPG) algorithm [Kakade, 2001]  with optimistic policy evaluation subroutines to encourage exploration. For $d$-dimensional linear MDPs, Optimistic NPG is computationally efficient, and learns an $\epsilon$-optimal policy within  $\tilde{\mathcal{O}}(d^2/\epsilon^3)$ samples, which is the first computationally efficient algorithm whose sample complexity has the optimal dimension dependence $\tilde{\Theta}(d^2)$. It also improves over state-of-the-art results of policy optimization algorithms [Zanette et al., 2021] by a factor of $d$. For general function approximation that subsumes linear MDPs, Optimistic NPG, to our best knowledge, is also the first policy optimization algorithm that achieves the polynomial sample complexity for learning near-optimal policies.

----

## [0] Two-Stage Learning to Defer with Multiple Experts

**Authors**: *Anqi Mao, Christopher Mohri, Mehryar Mohri, Yutao Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b17d256cf1fe1cc084922a8c6b565b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b17d256cf1fe1cc084922a8c6b565b7-Abstract-Conference.html)

**Abstract**:

We study a two-stage scenario for learning to defer with multiple experts, which is crucial in practice for many applications.  In this scenario, a predictor is derived in a first stage by training with a common loss function such as cross-entropy.  In the second stage, a deferral function is learned to assign the most suitable expert to each input.  We design a new family of surrogate loss functions for this scenario both in the score-based and the predictor-rejector settings and prove that they are supported by $H$-consistency bounds, which implies their Bayes-consistency.  Moreover, we show that, for a constant cost function, our two-stage surrogate losses are realizable $H$-consistent. While the main focus of this work is a theoretical analysis, we also report the results of several experiments on CIFAR-10 and SVHN datasets.

----

## [0] A Computationally Efficient Sparsified Online Newton Method

**Authors**: *Devvrit, Sai Surya Duvvuri, Rohan Anil, Vineet Gupta, Cho-Jui Hsieh, Inderjit S. Dhillon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b43289db08ed60edc6451cb2132e203-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b43289db08ed60edc6451cb2132e203-Abstract-Conference.html)

**Abstract**:

Second-order methods hold significant promise for enhancing the convergence of deep neural network training; however, their large memory and computational demands have limited their practicality. Thus there is a need for scalable second-order methods that can efficiently train large models. In this paper, we introduce the Sparsified Online Newton~(SONew) method, a memory-efficient second-order algorithm that yields a sparsified yet effective preconditioner. The algorithm emerges from a novel use of the LogDet matrix divergence measure; we combine it with sparsity constraints to minimize regret in the online convex optimization framework. Empirically, we test our method on large scale benchmarks of up to 1B parameters. We achieve up to $30\\%$ faster convergence, $3.4\\%$ relative improvement in validation performance, and $80\\%$ relative improvement in training loss, in comparison to memory efficient optimizers including first order methods. Powering the method is a surprising fact -- imposing structured sparsity patterns, like tridiagonal and banded structure, requires little to no overhead, making it as efficient and parallelizable as first-order methods. In wall-clock time, tridiagonal SONew is only about $3\\%$ slower per step than first-order methods but gives overall gains due to much faster convergence. In contrast, one of the state-of-the-art (SOTA) memory-intensive second-order methods, Shampoo, is unable to scale to large benchmarks. Additionally, while Shampoo necessitates significant engineering efforts to scale to large benchmarks, SONew offers a more straightforward implementation, increasing its practical appeal. SONew code is available at: https://github.com/devvrit/SONew

----

## [0] SparseProp: Efficient Event-Based Simulation and Training of Sparse Recurrent Spiking Neural Networks

**Authors**: *Rainer Engelken*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b443d358a391166d1fbf551fb53de02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b443d358a391166d1fbf551fb53de02-Abstract-Conference.html)

**Abstract**:

Spiking Neural Networks (SNNs) are biologically-inspired models that are capable of processing information in streams of action potentials. However, simulating and training SNNs is computationally expensive due to the need to solve large systems of coupled differential equations. In this paper, we propose a novel event-based algorithm called SparseProp for simulating and training sparse SNNs. Our algorithm reduces the computational cost of both forward pass and backward pass operations from O(N) to O(log(N)) per network spike, enabling numerically exact simulations of large spiking networks and their efficient training using backpropagation through time. By exploiting the sparsity of the network, SparseProp avoids iterating through all neurons at every spike and uses efficient state updates. We demonstrate the effectiveness of SparseProp for several classical integrate-and-fire neuron models, including simulating a sparse SNN with one million LIF neurons, which is sped up by more than four orders of magnitude compared to previous implementations. Our work provides an efficient and exact solution for training large-scale spiking neural networks and opens up new possibilities for building more sophisticated brain-inspired models.

----

## [0] ConRad: Image Constrained Radiance Fields for 3D Generation from a Single Image

**Authors**: *Senthil Purushwalkam, Nikhil Naik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b68d474baf8dff30f3280c199a32089-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b68d474baf8dff30f3280c199a32089-Abstract-Conference.html)

**Abstract**:

We present a novel method for reconstructing 3D objects from a single RGB image. Our method leverages the latest image generation models to infer the hidden 3D structure while remaining faithful to the input image. While existing methods obtain impressive results in generating 3D models from text prompts, they do not provide an easy approach for conditioning on input RGB data. Naive extensions of these methods often lead to improper alignment in appearance between the input image and the 3D reconstructions. We address these challenges by introducing Image Constrained Radiance Fields (ConRad), a novel variant of neural radiance fields. ConRad is an efficient 3D representation that explicitly captures the appearance of an input image in one viewpoint. We propose a training algorithm that leverages the single RGB image in conjunction with pretrained Diffusion Models to optimize the parameters of a ConRad representation. Extensive experiments show that ConRad representations can simplify preservation of image details while producing a realistic 3D reconstruction. Compared to existing state-of-the-art baselines, we show that our 3D reconstructions remain more faithful to the input and produce more consistent 3D models while demonstrating significantly improved quantitative performance on a ShapeNet object benchmark.

----

## [0] Fair Canonical Correlation Analysis

**Authors**: *Zhuoping Zhou, Davoud Ataee Tarzanagh, Bojian Hou, Boning Tong, Jia Xu, Yanbo Feng, Qi Long, Li Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b8e4c8468273ee3bafb288229c0acbc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b8e4c8468273ee3bafb288229c0acbc-Abstract-Conference.html)

**Abstract**:

This paper investigates fairness and bias in Canonical Correlation Analysis (CCA), a widely used statistical technique for examining the relationship between two sets of variables. We present a framework that alleviates unfairness by minimizing the correlation disparity error associated with protected attributes. Our approach enables CCA to learn global projection matrices from all data points while ensuring that these matrices yield comparable correlation levels to group-specific projection matrices. Experimental evaluation on both synthetic and real-world datasets demonstrates the efficacy of our method in reducing correlation disparity error without compromising CCA accuracy.

----

## [0] DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization

**Authors**: *Zhiqing Sun, Yiming Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html)

**Abstract**:

Neural network-based Combinatorial Optimization (CO) methods have shown promising results in solving various NP-complete (NPC) problems without relying on hand-crafted domain knowledge. This paper broadens the current scope of neural solvers for NPC problems by introducing a new graph-based diffusion framework, namely DIFUSCO. It formulates NPC problems into a discrete {0, 1}-vector space and uses graph-based denoising diffusion models to generate high-quality solutions. Specifically, we explore diffusion models with Gaussian and Bernoulli noise, respectively, and also introduce an effective inference schedule to improve the generation quality. We evaluate our methods on two well-studied combinatorial optimization problems: Traveling Salesman Problem (TSP) and Maximal Independent Set (MIS). Experimental results show that DIFUSCO strongly outperforms the previous state-of-the-art neural solvers, improving the performance gap between ground-truth and neural solvers from 1.76% to 0.46% on TSP-500, from 2.46% to 1.17% on TSP-1000, and from 3.19% to 2.58% on TSP-10000. For the MIS problem, DIFUSCO outperforms the previous state-of-the-art neural solver on the challenging SATLIB benchmark. Our code is available at this url.

----

## [0] Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models

**Authors**: *George Stein, Jesse C. Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Leigh Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L. Caterini, J. Eric T. Taylor, Gabriel Loaiza-Ganem*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0bc795afae289ed465a65a3b4b1f4eb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0bc795afae289ed465a65a3b4b1f4eb7-Abstract-Conference.html)

**Abstract**:

We systematically study a wide variety of generative models spanning semantically-diverse image datasets to understand and improve the feature extractors and metrics used to evaluate them.Using best practices in psychophysics, we measure human perception of image realism for generated samples by conducting the largest experiment evaluating generative models to date, and find that no existing metric strongly correlates with human evaluations.Comparing to 17 modern metrics for evaluating the overall performance, fidelity, diversity, rarity, and memorization of generative models, we find that the state-of-the-art perceptual realism of diffusion models as judged by humans is not reflected in commonly reported metrics such as FID. This discrepancy is not explained by diversity in generated samples, though one cause is over-reliance on Inception-V3.We address these flaws through a study of alternative self-supervised feature extractors, find that the semantic information encoded by individual networks strongly depends on their training procedure, and show that DINOv2-ViT-L/14 allows for much richer evaluation of generative models. Next, we investigate data memorization, and find that generative models do memorize training examples on simple, smaller datasets like CIFAR10, but not necessarily on more complex datasets like ImageNet. However, our experiments show that current metrics do not properly detect memorization: none in the literature is able to separate memorization from other phenomena such as underfitting or mode shrinkage. To facilitate further development of generative models and their evaluation we release all generated image datasets, human evaluation data, and a modular library to compute 17 common metrics for 9 different encoders at https://github.com/layer6ai-labs/dgm-eval.

----

## [0] Online Clustering of Bandits with Misspecified User Models

**Authors**: *Zhiyong Wang, Jize Xie, Xutong Liu, Shuai Li, John C. S. Lui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0bcd8d153b8c548629eca53f4ebdeb42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0bcd8d153b8c548629eca53f4ebdeb42-Abstract-Conference.html)

**Abstract**:

The contextual linear bandit is an important online learning problem where given arm features, a learning agent selects an arm at each round to maximize the cumulative rewards in the long run. A line of works, called the clustering of bandits (CB), utilize the collaborative effect over user preferences and have shown significant improvements over classic linear bandit algorithms. However, existing CB algorithms require well-specified linear user models and can fail when this critical assumption does not hold. Whether robust CB algorithms can be designed for more practical scenarios with misspecified user models remains an open problem. In this paper, we are the first to present the important problem of clustering of bandits with misspecified user models (CBMUM), where the expected rewards in user models can be perturbed away from perfect linear models. We devise two robust CB algorithms, RCLUMB and RSCLUMB (representing the learned clustering structure with dynamic graph and sets, respectively), that can accommodate the inaccurate user preference estimations and erroneous clustering caused by model misspecifications. We prove regret upper bounds of $O(\epsilon_*T\sqrt{md\log T}  + d\sqrt{mT}\log T)$ for our algorithms under milder assumptions than previous CB works, which match the lower bound asymptotically in $T$ up to logarithmic factors, and also match the state-of-the-art results in several degenerate cases. Our regret analysis is novel and different from the typical proof flow of previous CB works. The techniques in proving the regret caused by misclustering users are quite general and may be of independent interest. Experiments on both synthetic and real-world data show our outperformance over previous algorithms.

----

## [0] Temporal Conditioning Spiking Latent Variable Models of the Neural Response to Natural Visual Scenes

**Authors**: *Gehua Ma, Runhao Jiang, Rui Yan, Huajin Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0bcf9cf6ffe26bba3af99e18be0e1d8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0bcf9cf6ffe26bba3af99e18be0e1d8d-Abstract-Conference.html)

**Abstract**:

Developing computational models of neural response is crucial for understanding sensory processing and neural computations. Current state-of-the-art neural network methods use temporal filters to handle temporal dependencies, resulting in an unrealistic and inflexible processing paradigm. Meanwhile, these methods target trial-averaged firing rates and fail to capture important features in spike trains. This work presents the temporal conditioning spiking latent variable models (TeCoS-LVM) to simulate the neural response to natural visual stimuli. We use spiking neurons to produce spike outputs that directly match the recorded trains. This approach helps to avoid losing information embedded in the original spike trains. We exclude the temporal dimension from the model parameter space and introduce a temporal conditioning operation to allow the model to adaptively explore and exploit temporal dependencies in stimuli sequences in a natural paradigm. We show that TeCoS-LVM models can produce more realistic spike activities and accurately fit spike statistics than powerful alternatives. Additionally, learned TeCoS-LVM models can generalize well to longer time scales. Overall, while remaining computationally tractable, our model effectively captures key features of neural coding systems. It thus provides a useful tool for building accurate predictive computational accounts for various sensory perception circuits.

----

## [0] Double Auctions with Two-sided Bandit Feedback

**Authors**: *Soumya Basu, Abishek Sankararaman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0bcfb525c8f8f07ae10a93d0b2a40e00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0bcfb525c8f8f07ae10a93d0b2a40e00-Abstract-Conference.html)

**Abstract**:

Double Auction enables decentralized transfer of goods between multiple buyers and sellers, thus underpinning functioning of many online marketplaces. Buyers and sellers compete in these markets through bidding, but do not often know their own valuation a-priori. As the allocation and pricing happens through bids, the profitability of participants, hence sustainability of such markets, depends crucially on learning respective valuations through repeated interactions. We initiate the study of Double Auction markets under bandit feedback on both buyers' and sellers' side. We show with confidence bound based bidding, and `Average Pricing' there is an efficient price discovery among the participants.  In particular, the regret on combined valuation of the buyers and the sellers -- a.k.a. the social regret -- is $O(\log(T)/\Delta)$ in $T$ rounds, where $\Delta$ is the minimum price gap. Moreover, the buyers and sellers exchanging goods attain $O(\sqrt{T})$ regret, individually. The buyers and sellers who do not benefit from exchange in turn only experience $O(\log{T}/ \Delta)$  regret individually in $T$ rounds.  We augment our upper bound by showing that $\omega(\sqrt{T})$ individual regret, and $\omega(\log{T})$ social regret is unattainable in certain Double Auction markets. Our paper is the first to provide decentralized learning algorithms in a two-sided market where \emph{both sides have uncertain preference} that need to be learned.

----

## [0] Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking

**Authors**: *Juanhui Li, Harry Shomer, Haitao Mao, Shenglai Zeng, Yao Ma, Neil Shah, Jiliang Tang, Dawei Yin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0be50b4590f1c5fdf4c8feddd63c4f67-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0be50b4590f1c5fdf4c8feddd63c4f67-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Link prediction attempts to predict whether an unseen edge exists based on only a portion of the graph. A flurry of methods has been created in recent years that attempt to make use of graph neural networks (GNNs) for this task. Furthermore, new and diverse datasets have also been created to better evaluate the effectiveness of these new models. However, multiple limitations currently exist that hinders our ability to properly evaluate these new methods. This includes, but is not limited to: (1) The underreporting of performance on multiple baselines, (2) A lack of a unified data split and evaluation metric on some datasets, (3) An unrealistic evaluation setting that produces negative samples that are easy to classify. To overcome these challenges we first conduct a fair comparison across prominent methods and datasets, utilizing the same dataset settings and hyperparameter settings. We then create a new real-world evaluation setting that samples difficult negative samples via multiple heuristics. The new evaluation setting helps promote new challenges and opportunities in link prediction by aligning the evaluation with real-world situations.

----

## [0] EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images

**Authors**: *Seongsu Bae, Daeun Kyung, Jaehee Ryu, Eunbyeol Cho, Gyubok Lee, Sunjun Kweon, Jungwoo Oh, Lei Ji, Eric I-Chao Chang, Tackeun Kim, Edward Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c007ebef1d11fd48da6ce4f54687db6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c007ebef1d11fd48da6ce4f54687db6-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Electronic Health Records (EHRs), which contain patients' medical histories in various multi-modal formats, often overlook the potential for joint reasoning across imaging and table modalities underexplored in current EHR Question Answering (QA) systems. In this paper, we introduce EHRXQA, a novel multi-modal question answering dataset combining structured EHRs and chest X-ray images. To develop our dataset, we first construct two uni-modal resources: 1) The MIMIC- CXR-VQA dataset, our newly created medical visual question answering (VQA) benchmark, specifically designed to augment the imaging modality in EHR QA, and 2) EHRSQL (MIMIC-IV), a refashioned version of a previously established table-based EHR QA dataset. By integrating these two uni-modal resources, we successfully construct a multi-modal EHR QA dataset that necessitates both uni-modal and cross-modal reasoning. To address the unique challenges of multi-modal questions within EHRs, we propose a NeuralSQL-based strategy equipped with an external VQA API. This pioneering endeavor enhances engagement with multi-modal EHR sources and we believe that our dataset can catalyze advances in real-world medical scenarios such as clinical decision-making and research. EHRXQA is available at https://github.com/baeseongsu/ehrxqa.

----

## [0] Enhancing Robot Program Synthesis Through Environmental Context

**Authors**: *Tianyi Chen, Qidi Wang, Zhen Dong, Liwei Shen, Xin Peng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c1e94af650f5c74b1f3da467c2308c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c1e94af650f5c74b1f3da467c2308c2-Abstract-Conference.html)

**Abstract**:

Program synthesis aims to automatically generate an executable program that conforms to the given specification. Recent advancements have demonstrated that deep neural methodologies and large-scale pretrained language models are highly proficient in capturing program semantics.For robot programming, prior works have facilitated program synthesis by incorporating global environments. However, the assumption of acquiring a comprehensive understanding of the entire environment is often excessively challenging to achieve.In this work, we present a framework that learns to synthesize a program by rectifying potentially erroneous code segments, with the aid of partially observed environments. To tackle the issue of inadequate attention to partial observations, we propose to first learn an environment embedding space that can implicitly evaluate the impacts of each program token based on the precondition. Furthermore, by employing a graph structure, the model can aggregate both environmental and syntactic information flow and furnish smooth program rectification guidance.Extensive experimental evaluations and ablation studies on the partially observed VizDoom domain authenticate that our method offers superior generalization capability across various tasks and greater robustness when encountering noises.

----

## [0] ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling

**Authors**: *Quanyi Li, Zhenghao Mark Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo, Bolei Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c26a501df8fb919a0350e2df06b5d39-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c26a501df8fb919a0350e2df06b5d39-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large-scale driving datasets such as Waymo Open Dataset and nuScenes substantially accelerate autonomous driving research, especially for perception tasks such as 3D detection and trajectory forecasting. Since the driving logs in these datasets contain HD maps and detailed object annotations which accurately reflect the real-world complexity of traffic behaviors, we can harvest a massive number of complex traffic scenarios and recreate their digital twins in simulation. Compared to the hand-crafted scenarios often used in existing simulators, data-driven scenarios collected from the real world can facilitate many research opportunities in machine learning and autonomous driving. In this work, we present ScenarioNet, an open-source platform for large-scale traffic scenario modeling and simulation. ScenarioNet defines a unified scenario description format and collects a large-scale repository of real-world traffic scenarios from the heterogeneous data in various driving datasets including Waymo, nuScenes, Lyft L5, and nuPlan datasets. These scenarios can be further replayed and interacted with in multiple views from Bird-Eye-View layout to realistic 3D rendering in MetaDrive simulator. This provides a benchmark for evaluating the safety of autonomous driving stacks in simulation before their real-world deployment. We further demonstrate the strengths of ScenarioNet on large-scale scenario generation, imitation learning, and reinforcement learning in both single-agent and multi-agent settings. Code, demo videos, and website are available at https://github.com/metadriverse/scenarionet

----

## [0] Understanding Deep Gradient Leakage via Inversion Influence Functions

**Authors**: *Haobo Zhang, Junyuan Hong, Yuyang Deng, Mehrdad Mahdavi, Jiayu Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c4dd7e3d9f528f0b4f2aca9fbcdca8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c4dd7e3d9f528f0b4f2aca9fbcdca8d-Abstract-Conference.html)

**Abstract**:

Deep Gradient Leakage (DGL) is a highly effective attack that recovers private training images from gradient vectors.This attack casts significant privacy challenges on distributed learning from clients with sensitive data, where clients are required to share gradients.  Defending against such attacks requires but lacks an understanding of when and how privacy leakage happens, mostly because of the black-box nature of deep networks.  In this paper, we propose a novel Inversion Influence Function (I$^2$F) that establishes a closed-form connection between the recovered images and the private gradients by implicitly solving the DGL problem.  Compared to directly solving DGL, I$^2$F is scalable for analyzing deep networks, requiring only oracle access to gradients and Jacobian-vector products.  We empirically demonstrate that I$^2$F effectively approximated the DGL generally on different model architectures, datasets, modalities, attack implementations, and perturbation-based defenses.  With this novel tool, we provide insights into effective gradient perturbation directions, the unfairness of privacy protection, and privacy-preferred model initialization.  Our codes are provided in https://github.com/illidanlab/inversion-influence-function.

----

## [0] Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization

**Authors**: *Shurui Gui, Meng Liu, Xiner Li, Youzhi Luo, Shuiwang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c6c92a0c5237761168eafd4549f1584-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c6c92a0c5237761168eafd4549f1584-Abstract-Conference.html)

**Abstract**:

We tackle the problem of graph out-of-distribution (OOD) generalization. Existing graph OOD algorithms either rely on restricted assumptions or fail to exploit environment information in training data. In this work, we propose to simultaneously incorporate label and environment causal independence (LECI) to fully make use of label and environment information, thereby addressing the challenges faced by prior methods on identifying causal and invariant subgraphs. We further develop an adversarial training strategy to jointly optimize these two properties for casual subgraph discovery with theoretical guarantees. Extensive experiments and analysis show that LECI significantly outperforms prior methods on both synthetic and real-world datasets, establishing LECI as a practical and effective solution for graph OOD generalization.

----

## [0] Bayesian Learning of Optimal Policies in Markov Decision Processes with Countably Infinite State-Space

**Authors**: *Saghar Adler, Vijay G. Subramanian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html)

**Abstract**:

Models of many real-life applications, such as queueing models of communication networks or computing systems, have a countably infinite state-space. Algorithmic and learning procedures that have been developed to produce optimal policies mainly focus on finite state settings, and do not directly apply to these models. To overcome this lacuna, in this work we study the problem of optimal control of a family of discrete-time countable state-space Markov Decision Processes (MDPs) governed by an unknown parameter $\theta\in\Theta$,  and defined on a countably-infinite state-space $\mathcal X=\mathbb{Z}_+^d$, with finite action space $\mathcal A$, and an unbounded cost function. We take a Bayesian perspective with the random unknown parameter $\boldsymbol{\theta}^*$ generated via a given fixed prior distribution on $\Theta$. To optimally control the unknown MDP, we propose an algorithm based on Thompson sampling with dynamically-sized episodes: at the beginning of each episode, the posterior distribution formed via Bayes' rule is used to produce a parameter estimate, which then decides the policy applied during the episode. To ensure the stability of the Markov chain obtained by following the policy chosen for each parameter, we impose ergodicity assumptions. From this condition and using the solution of the average cost Bellman equation, we establish an $\tilde O(dh^d\sqrt{|\mathcal A|T})$ upper bound on the Bayesian regret of our algorithm, where $T$ is the time-horizon. Finally, to elucidate the applicability of our algorithm, we consider two different queueing models with unknown dynamics, and show that our algorithm can be applied to develop approximately optimal control algorithms.

----

## [0] CARE: Modeling Interacting Dynamics Under Temporal Environmental Variation

**Authors**: *Xiao Luo, Haixin Wang, Zijie Huang, Huiyu Jiang, Abhijeet Gangan, Song Jiang, Yizhou Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c7ca207a051228f978971447a56464a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c7ca207a051228f978971447a56464a-Abstract-Conference.html)

**Abstract**:

Modeling interacting dynamical systems, such as fluid dynamics and intermolecular interactions, is a fundamental research problem for understanding and simulating complex real-world systems. Many of these systems can be naturally represented by dynamic graphs, and graph neural network-based approaches have been proposed and shown promising performance. However, most of these approaches assume the underlying dynamics does not change over time, which is unfortunately untrue. For example, a molecular dynamics can be affected by the environment temperature over the time. In this paper, we take an attempt to provide a probabilistic view for time-varying dynamics and propose a model Context-attended Graph ODE (CARE) for modeling time-varying interacting dynamical systems. In our CARE, we explicitly use a context variable to model time-varying environment and construct an encoder to initialize the context variable from historical trajectories. Furthermore, we employ a neural ODE model to depict the dynamic evolution of the context variable inferred from system states. This context variable is incorporated into a coupled ODE to simultaneously drive the evolution of systems. Comprehensive experiments on four datasets demonstrate the effectiveness of our proposed CARE compared with several state-of-the-art approaches.

----

## [0] Diffused Redundancy in Pre-trained Representations

**Authors**: *Vedant Nanda, Till Speicher, John P. Dickerson, Krishna P. Gummadi, Soheil Feizi, Adrian Weller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c86142265c5e2c900613dd1d031cb90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c86142265c5e2c900613dd1d031cb90-Abstract-Conference.html)

**Abstract**:

Representations learned by pre-training a neural network on a large dataset are increasingly used successfully to perform a variety of downstream tasks. In this work, we take a closer look at how features are encoded in such pre-trained representations. We find that learned representations in a given layer exhibit a degree of diffuse redundancy, ie, any randomly chosen subset of neurons in the layer that is larger than a threshold size shares a large degree of similarity with the full layer and is able to perform similarly as the whole layer on a variety of downstream tasks. For example, a linear probe trained on $20\%$ of randomly picked neurons from the penultimate layer of a ResNet50 pre-trained on ImageNet1k achieves an accuracy within $5\%$ of a linear probe trained on the full layer of neurons for downstream CIFAR10 classification. We conduct experiments on different neural architectures (including CNNs and Transformers) pre-trained on both ImageNet1k and ImageNet21k and evaluate a variety of downstream tasks taken from the VTAB benchmark. We find that the loss \& dataset used during pre-training largely govern the degree of diffuse redundancy and the "critical mass" of neurons needed often depends on the downstream task, suggesting that there is a task-inherent redundancy-performance Pareto frontier. Our findings shed light on the nature of representations learned by pre-trained deep neural networks and suggest that entire layers might not be necessary to perform many downstream tasks. We investigate the potential for exploiting this redundancy to achieve efficient generalization for downstream tasks and also draw caution to certain possible unintended consequences. Our code is available at \url{https://github.com/nvedant07/diffused-redundancy}.

----

## [0] AI for Interpretable Chemistry: Predicting Radical Mechanistic Pathways via Contrastive Learning

**Authors**: *Mohammadamin Tavakoli, Pierre Baldi, Ann Marie Carlton, Yin Ting T. Chiu, Alexander Shmakov, David Van Vranken*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ca70969597da7166128f7755c64ffd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ca70969597da7166128f7755c64ffd5-Abstract-Conference.html)

**Abstract**:

Deep learning-based reaction predictors have undergone significant architectural evolution. However, their reliance on reactions from the US Patent Office results in a lack of interpretable predictions and limited generalizability to other chemistry domains, such as radical and atmospheric chemistry. To address these challenges, we introduce a new reaction predictor system, RMechRP, that leverages contrastive learning in conjunction with mechanistic pathways, the most interpretable representation of chemical reactions. Specifically designed for radical reactions, RMechRP provides different levels of interpretation of chemical reactions. We develop and train multiple deep-learning models using RMechDB, a public database of radical reactions, to establish the first benchmark for predicting radical reactions. Our results demonstrate the effectiveness of RMechRP in providing accurate and interpretable predictions of radical reactions, and its potential for various applications in atmospheric chemistry.

----

## [0] Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks

**Authors**: *Jules Berman, Benjamin Peherstorfer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0cb310ed8121549488fea8e8c2056096-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0cb310ed8121549488fea8e8c2056096-Abstract-Conference.html)

**Abstract**:

Training neural networks sequentially in time to approximate solution fields of time-dependent partial differential equations can be beneficial for preserving causality and other physics properties; however, the sequential-in-time training is numerically challenging because training errors quickly accumulate and amplify over time. This work introduces Neural Galerkin schemes that update randomized sparse subsets of network parameters at each time step. The randomization avoids overfitting locally in time and so helps prevent the error from accumulating quickly over the sequential-in-time training, which is motivated by dropout that addresses a similar issue of overfitting due to neuron co-adaptation. The sparsity of the update reduces the computational costs of training without losing expressiveness because many of the network parameters are redundant locally at each time step. In numerical experiments with a wide range of evolution equations, the proposed scheme with randomized sparse updates is up to two orders of magnitude more accurate at a fixed computational budget and up to two orders of magnitude faster at a fixed accuracy than schemes with dense updates.

----

## [0] Handling Data Heterogeneity via Architectural Design for Federated Visual Recognition

**Authors**: *Sara Pieri, Jose Renato Restom, Samuel Horváth, Hisham Cholakkal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ccd06ff26fd6a7829293ce90e0e7f7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ccd06ff26fd6a7829293ce90e0e7f7d-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) is a promising research paradigm that enables the collaborative training of machine learning models among various parties without the need for sensitive information exchange. Nonetheless, retaining data in individual clients introduces fundamental challenges to achieving performance on par with centrally trained models. Our study provides an extensive review of federated learning applied to visual recognition. It underscores the critical role of thoughtful architectural design choices in achieving optimal performance, a factor often neglected in the FL literature. Many existing FL solutions are tested on shallow or simple networks, which may not accurately reflect real-world applications. This practice restricts the transferability of research findings to large-scale visual recognition models. Through an in-depth analysis of diverse cutting-edge architectures such as convolutional neural networks, transformers, and MLP-mixers,  we experimentally demonstrate that architectural choices can substantially enhance FL systems' performance, particularly when handling heterogeneous data.  We study visual recognition models from five different architectural families on four challenging FL datasets. We also re-investigate the inferior performance convolution-based architectures in the FL setting and analyze the influence of normalization layers on the FL performance. Our findings emphasize the importance of architectural design for computer vision tasks in practical scenarios, effectively narrowing the performance gap between federated and centralized learning.

----

## [0] Spatial-frequency channels, shape bias, and adversarial robustness

**Authors**: *Ajay Subramanian, Elena Sizikova, Najib J. Majaj, Denis G. Pelli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0cdc1e85736d9c01d366cbf9b4b81672-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0cdc1e85736d9c01d366cbf9b4b81672-Abstract-Conference.html)

**Abstract**:

What spatial frequency information do humans and neural networks use to recognize objects? In neuroscience, critical band masking is an established tool that can reveal the frequency-selective filters used for object recognition. Critical band masking measures the sensitivity of recognition performance to noise added at each spatial frequency. Existing critical band masking studies show that humans recognize periodic patterns (gratings) and letters by means of a spatial-frequency filter (or "channel") that has a frequency bandwidth of one octave (doubling of frequency). Here, we introduce critical band masking as a task for network-human comparison and test 14 humans and 76 neural networks on 16-way ImageNet categorization in the presence of narrowband noise. We find that humans recognize objects in natural images using the same one-octave-wide channel that they use for letters and gratings, making it a canonical feature of human object recognition. Unlike humans, the neural network channel is very broad, 2-4 times wider than the human channel. This means that the network channel extends to frequencies higher and lower than those that humans are sensitive to. Thus, noise at those frequencies will impair network performance and spare human performance. Adversarial and augmented-image training are commonly used to increase network robustness and shape bias. Does this training align network and human object recognition channels? Three network channel properties (bandwidth, center frequency, peak noise sensitivity) correlate strongly with shape bias (51% variance explained) and robustness of adversarially-trained networks (66% variance explained). Adversarial training increases robustness but expands the channel bandwidth even further beyond the human bandwidth. Thus, critical band masking reveals that the network channel is more than twice as wide as the human channel, and that adversarial training only makes it worse. Networks with narrower channels might be more robust.

----

## [0] Optimality in Mean Estimation: Beyond Worst-Case, Beyond Sub-Gaussian, and Beyond 1+α Moments

**Authors**: *Trung Dang, Jasper C. H. Lee, Maoyuan Raymond Song, Paul Valiant*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0cddb777d3441326544e21b67f41bdc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0cddb777d3441326544e21b67f41bdc8-Abstract-Conference.html)

**Abstract**:

There is growing interest in improving our algorithmic understanding of fundamental statistical problems such as mean estimation, driven by the goal of understanding the fundamental limits of what we can extract from limited and valuable data.The state of the art results for mean estimation in $\mathbb{R}$ are 1) the optimal sub-Gaussian mean estimator by [Lee and Valiant, 2022], attaining the optimal sub-Gaussian error constant for all distributions with finite but unknown variance, and 2) the analysis of the median-of-means algorithm by [Bubeck, Cesa-Bianchi and Lugosi, 2013] and a matching lower bound by [Devroye, Lerasle, Lugosi, and Oliveira, 2016], characterizing the big-O optimal errors for distributions that have tails heavy enough that only a $1+\alpha$ moment exists for some $\alpha \in (0,1)$.Both of these results, however, are optimal only in the worst case.Motivated by the recent effort in the community to go "beyond the worst-case analysis" of algorithms, we initiate the fine-grained study of the mean estimation problem:Is it possible for algorithms to leverage *beneficial* features/quirks of their input distribution to *beat* the sub-Gaussian rate, without explicit knowledge of these features?We resolve this question, finding an unexpectedly nuanced answer: "Yes in limited regimes, but in general no".Given a distribution $p$, assuming *only* that it has a finite mean and absent any additional assumptions,we show how to construct a distribution $q_{n,\delta}$ such that the means of $p$ and $q$ are well-separated, yet $p$ and $q$ are impossible to distinguish with $n$ samples with probability $1-\delta$, and $q$ further preserves the finiteness of moments of $p$.Moreover, the variance of $q$ is at most twice the variance of $p$ if it exists.The main consequence of our result is that, no reasonable estimator can asymptotically achieve better than the sub-Gaussian error rate for any distribution, up to constant factors, which matches the worst-case result of [Lee and Valiant, 2022].More generally, we introduce a new definitional framework to analyze the fine-grained optimality of algorithms, which we call "neighborhood optimality", interpolating between the unattainably strong "instance optimality" and the trivially weak admissibility/Pareto optimality definitions.As an application of the new framework, we show that the median-of-means algorithm is neighborhood optimal, up to constant factors.It is an open question to find a neighborhood-optimal estimator *without* constant factor slackness.

----

## [0] Provably Efficient Offline Goal-Conditioned Reinforcement Learning with General Function Approximation and Single-Policy Concentrability

**Authors**: *Hanlin Zhu, Amy Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0cfc9404f89400c5ed897035e0d3748c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0cfc9404f89400c5ed897035e0d3748c-Abstract-Conference.html)

**Abstract**:

Goal-conditioned reinforcement learning (GCRL) refers to learning general-purpose skills that aim to reach diverse goals. In particular, offline GCRL only requires purely pre-collected datasets to perform training tasks without additional interactions with the environment. Although offline GCRL has become increasingly prevalent and many previous works have demonstrated its empirical success, the theoretical understanding of efficient offline GCRL algorithms is not well established, especially when the state space is huge and the offline dataset only covers the policy we aim to learn. In this paper, we provide a rigorous theoretical analysis of an existing empirically successful offline GCRL algorithm. We prove that under slight modification, this algorithm enjoys an $\tilde{O}(\text{poly}(1/\epsilon))$ sample complexity (where $\epsilon$ is the desired suboptimality of the learned policy) with general function approximation thanks to the property of (semi-)strong convexity of the objective functions. We only require nearly minimal assumptions on the dataset (single-policy concentrability) and the function class (realizability). Moreover, this algorithm consists of two uninterleaved optimization steps, which we refer to as $V$-learning and policy learning, and is computationally stable since it does not involve minimax optimization. We also empirically validate our theory by showing that the modified algorithm outperforms the previous algorithm in various real-world environments.To the best of our knowledge, this is the first algorithm that is both provably efficient with general function approximation and single-policy concentrability, and empirically successful without requiring solving minimax optimization problems.

----

## [0] SQ Lower Bounds for Non-Gaussian Component Analysis with Weaker Assumptions

**Authors**: *Ilias Diakonikolas, Daniel Kane, Lisheng Ren, Yuxin Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d00a699f60e642b310eb04b76cc7731-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d00a699f60e642b310eb04b76cc7731-Abstract-Conference.html)

**Abstract**:

We study the complexity of Non-Gaussian Component Analysis (NGCA) in the Statistical Query (SQ) model.Prior work developed a methodology to prove SQ lower bounds for NGCA that have been applicable to a wide range of contexts.In particular, it was known that for any univariate distribution $A$ satisfying certain conditions,distinguishing between a standard multivariate Gaussian and a distribution that behaves like $A$ in a random hidden direction and like a standard Gaussian in the orthogonal complement, is SQ-hard.The required conditions were that (1) $A$ matches many low-order moments with a standard Gaussian,and (2) the chi-squared norm of $A$ with respect to the standard Gaussian is finite.While the moment-matching condition is clearly necessary for hardness, the chi-squared condition was only required for technical reasons.In this work, we establish that the latter condition is indeed not necessary.In particular, we prove near-optimal SQ lower bounds for NGCA under the moment-matching condition only.

----

## [0] Efficient Equivariant Transfer Learning from Pretrained Models

**Authors**: *Sourya Basu, Pulkit Katdare, Prasanna Sattigeri, Vijil Chenthamarakshan, Katherine Driggs Campbell, Payel Das, Lav R. Varshney*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d02892a0055c94584f6394f8d069c8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d02892a0055c94584f6394f8d069c8e-Abstract-Conference.html)

**Abstract**:

Efficient transfer learning algorithms are key to the success of foundation models on diverse downstream tasks even with limited data. Recent works of Basu et al. (2023) and Kaba et al. (2022) propose group averaging (equitune) and optimization-based methods, respectively, over features from group-transformed inputs to obtain equivariant outputs from non-equivariant neural networks. While Kaba et al. (2022) are only concerned with training from scratch, we find that equitune performs poorly on equivariant zero-shot tasks despite good finetuning results. We hypothesize that this is because pretrained models provide better quality features for certain transformations than others and simply averaging them is deleterious. Hence, we propose 位-equitune that averages the features using importance weights, 位s. These weights are learned directly from the data using a small neural network, leading to excellent zero-shot and finetuned results that outperform equitune. Further, we prove that 位-equitune is equivariant and a universal approximator of equivariant functions. Additionally, we show that the method of Kaba et al. (2022) used with appropriate loss functions, which we call equizero, also gives excellent zero-shot and finetuned performance. Both equitune and equizero are special cases of 位- equitune. To show the simplicity and generality of our method, we validate on a wide range of diverse applications and models such as 1) image classification using CLIP, 2) deep Q-learning, 3) fairness in natural language generation (NLG), 4) compositional generalization in languages, and 5) image classification using pretrained CNNs such as Resnet and Alexnet.

----

## [0] Kernelized Reinforcement Learning with Order Optimal Regret Bounds

**Authors**: *Sattar Vakili, Julia Olkhovskaya*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d17d033059bacd127f25ab28784f829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d17d033059bacd127f25ab28784f829-Abstract-Conference.html)

**Abstract**:

Modern reinforcement learning (RL) has shown empirical success in various real world settings with complex models and large state-action spaces. The existing analytical results, however, typically focus on settings with a small number of state-actions or simple models such as linearly modeled state-action value functions. To derive RL policies that efficiently handle large state-action spaces with more general value functions, some recent works have considered nonlinear function approximation using kernel ridge regression. We propose $\pi$-KRVI, an optimistic modification of least-squares value iteration, when the action-value function is represented by an RKHS. We prove the first order-optimal regret guarantees under a general setting. Our results show a significant polynomial in the number of episodes improvement over the state of the art. In particular, with highly non-smooth kernels (such as Neural Tangent kernel or some Mat√©rn kernels) the existing results lead to trivial (superlinear in the number of episodes) regret bounds. We show a sublinear regret bound that is order optimal in the cases where a lower bound on regret is known (which includes the kernels mentioned above).

----

## [0] Learning Domain-Aware Detection Head with Prompt Tuning

**Authors**: *Haochen Li, Rui Zhang, Hantao Yao, Xinkai Song, Yifan Hao, Yongwei Zhao, Ling Li, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d18ab3b5fabfa6fe47c62e711af02f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d18ab3b5fabfa6fe47c62e711af02f0-Abstract-Conference.html)

**Abstract**:

Domain adaptive object detection (DAOD) aims to generalize detectors trained on an annotated source domain to an unlabelled target domain.  However, existing methods focus on reducing the domain bias of the detection backbone by inferring a discriminative visual encoder, while ignoring the domain bias in the detection head.  Inspired by the high generalization of vision-language models (VLMs), applying a VLM as the robust detection backbone following a domain-aware detection head is a reasonable way to learn the discriminative detector for each domain, rather than reducing the domain bias in traditional methods.  To achieve the above issue, we thus propose a novel DAOD framework named Domain-Aware detection head with Prompt tuning (DA-Pro), which applies the learnable domain-adaptive prompt to generate the dynamic detection head for each domain.   Formally, the domain-adaptive prompt consists of the domain-invariant tokens, domain-specific tokens, and the domain-related textual description along with the class label.   Furthermore, two constraints between the source and target domains are applied to ensure that the domain-adaptive prompt can capture the domains-shared and domain-specific knowledge.  A prompt ensemble strategy is also proposed to reduce the effect of prompt disturbance.   Comprehensive experiments over multiple cross-domain adaptation tasks demonstrate that using the domain-adaptive prompt can produce an effectively domain-related detection head for boosting domain-adaptive object detection.  Our code is available at https://github.com/Therock90421/DA-Pro.

----

## [0] Parallel Sampling of Diffusion Models

**Authors**: *Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html)

**Abstract**:

Diffusion models are powerful generative models but suffer from slow sampling, often taking 1000 sequential denoising steps for one sample. As a result, considerable efforts have been directed toward reducing the number of denoising steps, but these methods hurt sample quality. Instead of reducing the number of denoising steps (trading quality for speed), in this paper we explore an orthogonal approach: can we run the denoising steps in parallel (trading compute for speed)? In spite of the sequential nature of the denoising steps, we show that surprisingly it is possible to parallelize sampling via Picard iterations, by guessing the solution of future denoising steps and iteratively refining until convergence. With this insight, we present ParaDiGMS, a novel method to accelerate the sampling of pretrained diffusion models by denoising multiple steps in parallel. ParaDiGMS is the first diffusion sampling method that enables trading compute for speed and is even compatible with existing fast sampling techniques such as DDIM and DPMSolver. Using ParaDiGMS, we improve sampling speed by 2-4x across a range of robotics and image generation models, giving state-of-the-art sampling speeds of 0.2s on 100-step DiffusionPolicy and 14.6s on 1000-step StableDiffusion-v2 with no measurable degradation of task reward, FID score, or CLIP score.

----

## [0] Fractal Landscapes in Policy Optimization

**Authors**: *Tao Wang, Sylvia L. Herbert, Sicun Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d21f257b5288385cb6cb8e0ff2ce82e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d21f257b5288385cb6cb8e0ff2ce82e-Abstract-Conference.html)

**Abstract**:

Policy gradient lies at the core of deep reinforcement learning (RL) in continuous domains. Despite much success, it is often observed in practice that RL training with policy gradient can fail for many reasons, even on standard control problems with known solutions. We propose a framework for understanding one inherent limitation of the policy gradient approach: the optimization landscape in the policy space can be extremely non-smooth or fractal for certain classes of MDPs, such that there does not exist gradient to be estimated in the first place. We draw on techniques from chaos theory and non-smooth analysis, and analyze the maximal Lyapunov exponents and H\"older exponents of the policy optimization objectives. Moreover, we develop a practical method that can estimate the local smoothness of objective function from samples to identify when the training process has encountered fractal landscapes. We show experiments to illustrate how some failure cases of policy optimization can be explained by such fractal landscapes.

----

## [0] Moral Responsibility for AI Systems

**Authors**: *Sander Beckers*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d5b7fd8c669fac58d6702188ed63afa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d5b7fd8c669fac58d6702188ed63afa-Abstract-Conference.html)

**Abstract**:

As more and more decisions that have a significant ethical dimension are being outsourced to AI systems, it is important to have a definition of moral responsibility that can be applied to AI systems. Moral responsibility for an outcome of an agent who performs some action is commonly taken to involve both a causal condition and an epistemic condition: the action should cause the outcome, and the agent should have been aware - in some form or other - of the possible moral consequences of their action. This paper presents a formal definition of both conditions within the framework of causal models. I compare my approach to the existing approaches of Braham and van Hees (BvH) and of Halpern and Kleiman-Weiner (HK). I then generalize my definition into a degree of responsibility.

----

## [0] Characterizing the Impacts of Semi-supervised Learning for Weak Supervision

**Authors**: *Jeffrey Li, Jieyu Zhang, Ludwig Schmidt, Alexander J. Ratner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d6270381e018b3d83eb9be7d0b06036-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d6270381e018b3d83eb9be7d0b06036-Abstract-Conference.html)

**Abstract**:

Labeling training data is a critical and expensive step in producing high accuracy ML models, whether training from scratch or fine-tuning. To make labeling more efficient, two major approaches are programmatic weak supervision (WS) and semi-supervised learning (SSL). More recent works have either explicitly or implicitly used techniques at their intersection, but in various complex and ad hoc ways. In this work, we define a simple, modular design space to study the use of SSL techniques for WS more systematically. Surprisingly, we find that fairly simple methods from our design space match the performance of more complex state-of-the-art methods, averaging a 3 p.p. increase in accuracy/F1-score across 8 standard WS benchmarks. Further, we provide practical guidance on when different components are worth their added complexity and training costs. Contrary to current understanding, we find using SSL is not necessary to obtain the best performance on most WS benchmarks but is more effective when: (1) end models are smaller, and (2) WS provides labels for only a small portion of training examples.

----

## [0] Logarithmic Bayes Regret Bounds

**Authors**: *Alexia Atsidakou, Branislav Kveton, Sumeet Katariya, Constantine Caramanis, Sujay Sanghavi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d9057d84a9fc37523bf826232ea6820-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d9057d84a9fc37523bf826232ea6820-Abstract-Conference.html)

**Abstract**:

We derive the first finite-time logarithmic Bayes regret upper bounds for Bayesian bandits. In a multi-armed bandit, we obtain $O(c_\Delta \log n)$ and $O(c_h \log^2 n)$ upper bounds for an upper confidence bound algorithm, where $c_h$ and $c_\Delta$ are constants depending on the prior distribution and the gaps of bandit instances sampled from it, respectively. The latter bound asymptotically matches the lower bound of Lai (1987). Our proofs are a major technical departure from prior works, while being simple and general. To show the generality of our techniques, we apply them to linear bandits. Our results provide insights on the value of prior in the Bayesian setting, both in the objective and as a side information given to the learner. They significantly improve upon existing $\tilde{O}(\sqrt{n})$ bounds, which have become standard in the literature despite the logarithmic lower bound of Lai (1987).

----

## [0] Frequency-Enhanced Data Augmentation for Vision-and-Language Navigation

**Authors**: *Keji He, Chenyang Si, Zhihe Lu, Yan Huang, Liang Wang, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d9e08f247ca7fbbfd5e50b7ff9cf357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d9e08f247ca7fbbfd5e50b7ff9cf357-Abstract-Conference.html)

**Abstract**:

Vision-and-Language Navigation (VLN) is a challenging task that requires an agent to navigate through complex environments based on natural language instructions. In contrast to conventional approaches, which primarily focus on the spatial domain exploration, we propose a paradigm shift toward the Fourier domain. This alternative perspective aims to enhance visual-textual matching, ultimately improving the agent's ability to understand and execute navigation tasks based on the given instructions. In this study, we first explore the significance of high-frequency information in VLN and provide evidence that it is instrumental in bolstering visual-textual matching processes. Building upon this insight, we further propose a sophisticated and versatile Frequency-enhanced Data Augmentation (FDA) technique to improve the VLN model's capability of capturing critical high-frequency information. Specifically, this approach requires the agent to navigate in environments where only a subset of high-frequency visual information corresponds with the provided textual instructions, ultimately fostering the agent's ability to selectively discern and capture pertinent high-frequency features according to the given instructions. Promising results on R2R, RxR, CVDN and REVERIE demonstrate that our FDA can be readily integrated with existing VLN approaches, improving performance without adding extra parameters, and keeping models simple and efficient. The code is available at https://github.com/hekj/FDA.

----

## [0] Building Socio-culturally Inclusive Stereotype Resources with Community Engagement

**Authors**: *Sunipa Dev, Jaya Goyal, Dinesh Tewari, Shachi Dave, Vinodkumar Prabhakaran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

With rapid development and deployment of generative language models in global settings, there is an urgent need to also scale our measurements of harm, not just in the number and types of harms covered, but also how well they account for local cultural contexts, including marginalized identities and the social biases experienced by them.Current evaluation paradigms are limited in their abilities to address this, as they are not representative of diverse, locally situated but global, socio-cultural perspectives. It is imperative that our evaluation resources are enhanced and calibrated by including people and experiences from different cultures and societies worldwide, in order to prevent gross underestimations or skews in measurements of harm. In this work, we demonstrate a socio-culturally aware expansion of evaluation resources in the Indian societal context, specifically for the harm of stereotyping. We devise a community engaged effort to build a resource which contains stereotypes for axes of disparity that are uniquely present in India. The resultant resource increases the number of stereotypes known for and in the Indian context by over 1000 stereotypes across many unique identities. We also demonstrate the utility and effectiveness of such expanded resources for evaluations of language models.CONTENT WARNING: This paper contains examples of stereotypes that may be offensive.

----

## [0] Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment

**Authors**: *Hao Liu, Wilson Yan, Pieter Abbeel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0df1738319f8c6e15b58cb16ea3cfa57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0df1738319f8c6e15b58cb16ea3cfa57-Abstract-Conference.html)

**Abstract**:

Recent progress in scaling up large language models has shown impressive capabilities in performing few-shot learning across a wide range of natural language tasks. However, a key limitation is that these language models fundamentally lack grounding to visual perception - a crucial attribute needed to extend to real world tasks such as in visual-question answering and robotics. While prior works have largely connected image to text through pretraining or fine-tuning, learning such alignments are generally costly due to a combination of curating massive datasets and large computational burdens. In order to resolve these limitations, we propose a simple yet effective approach called Language-Quantized AutoEncoder (LQAE), a modification of VQ-VAE that learns to align text-image data in an unsupervised manner by leveraging pretrained language model denoisers (e.g., BERT). Our main idea is to encode images as sequences of text tokens by directly quantizing image embeddings using a pretrained language codebook. We then feed a masked version of the quantized embeddings into a BERT to reconstruct the original input. By doing so, LQAE learns to represent similar images with similar clusters of text tokens, thereby aligning these two modalities without the use of aligned text-image pairs. We show LQAE learns text-aligned image tokens that enable few-shot multi-modal learning with large language models, outperforming baseline methods in tasks such as image classification and VQA while requiring as few as 1-10 image-text pairs.

----

## [0] QuIP: 2-Bit Quantization of Large Language Models With Guarantees

**Authors**: *Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0df38cd13520747e1e64e5b123a78ef8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0df38cd13520747e1e64e5b123a78ef8-Abstract-Conference.html)

**Abstract**:

This work studies post-training parameter quantization in large language models (LLMs). We introduce quantization with incoherence processing (QuIP), a new method based on the insight that quantization benefits from incoherent weight and Hessian matrices, i.e., from the weights being even in magnitude and the directions in which it is important to round them accurately being unaligned with the coordinate axes. QuIP consists of two steps: (1) an adaptive rounding procedure minimizing a quadratic proxy objective; (2) efficient pre- and post-processing that ensures weight and Hessian incoherence via multiplication by random orthogonal matrices. We complement QuIP with the first theoretical analysis for an LLM-scale quantization algorithm, and show that our theory also applies to an existing method, OPTQ. Empirically, we find that our incoherence preprocessing improves several existing quantization algorithms and yields the first LLM quantization methods that produce viable results using only two bits per weight. Our code can be found at https://github.com/Cornell-RelaxML/QuIP.

----

## [0] Exploiting Correlated Auxiliary Feedback in Parameterized Bandits

**Authors**: *Arun Verma, Zhongxiang Dai, Yao Shu, Bryan Kian Hsiang Low*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e0157ce5ea15831072be4744cbd5334-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e0157ce5ea15831072be4744cbd5334-Abstract-Conference.html)

**Abstract**:

We study a novel variant of the parameterized bandits problem in which the learner can observe additional auxiliary feedback that is correlated with the observed reward. The auxiliary feedback is readily available in many real-life applications, e.g., an online platform that wants to recommend the best-rated services to its users can observe the user's rating of service (rewards) and collect additional information like service delivery time (auxiliary feedback). In this paper, we first develop a method that exploits auxiliary feedback to build a reward estimator with tight confidence bounds, leading to a smaller regret. We then characterize the regret reduction in terms of the correlation coefficient between reward and its auxiliary feedback. Experimental results in different settings also verify the performance gain achieved by our proposed method.

----

## [0] Multi-modal Queried Object Detection in the Wild

**Authors**: *Yifan Xu, Mengdan Zhang, Chaoyou Fu, Peixian Chen, Xiaoshan Yang, Ke Li, Changsheng Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e3af444e7d82d29871804de476d1fbe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e3af444e7d82d29871804de476d1fbe-Abstract-Conference.html)

**Abstract**:

We introduce MQ-Det, an efficient architecture and pre-training strategy design to utilize both textual description with open-set generalization and visual exemplars with rich description granularity as category queries, namely, Multi-modal Queried object Detection, for real-world detection with both open-vocabulary categories and various granularity. MQ-Det incorporates vision queries into existing well-established language-queried-only detectors. A plug-and-play gated class-scalable perceiver module upon the frozen detector is proposed to augment category text with class-wise visual information. To address the learning inertia problem brought by the frozen detector, a vision conditioned masked language prediction strategy is proposed. MQ-Det's simple yet effective architecture and training strategy design is compatible with most language-queried object detectors, thus yielding versatile applications. Experimental results demonstrate that multi-modal queries largely boost open-world detection. For instance, MQ-Det significantly improves the state-of-the-art open-set detector GLIP by +7.8% AP on the LVIS benchmark via multi-modal queries without any downstream finetuning, and averagely +6.3% AP on 13 few-shot downstream tasks, with merely additional 3% modulating time required by GLIP. Code is available at https://github.com/YifanXu74/MQ-Det.

----

## [0] H-Consistency Bounds: Characterization and Extensions

**Authors**: *Anqi Mao, Mehryar Mohri, Yutao Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e441913d4fa486c3eec967d79750b13-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e441913d4fa486c3eec967d79750b13-Abstract-Conference.html)

**Abstract**:

A series of recent publications by Awasthi et al. have introduced the key notion of *$H$-consistency bounds* for surrogate loss functions.  These are upper bounds on the zero-one estimation error of any predictor in a hypothesis set, expressed in terms of its surrogate loss estimation error. They are both non-asymptotic and hypothesis set-specific and thus stronger and more informative than Bayes-consistency. However, determining if they hold and deriving these bounds have required a specific proof and analysis for each surrogate loss. Can we derive more general tools and characterizations? This paper provides both a general characterization and an extension of $H$-consistency bounds for multi-class classification. We present new and tight $H$-consistency bounds for both the family of constrained losses and that of comp-sum losses, which covers the familiar cross-entropy, or logistic loss applied to the outputs of a neural network. We further extend our analysis beyond the completeness assumptions adopted in previous studies and cover more realistic bounded hypothesis sets.  Our characterizations are based on error transformations, which are explicitly defined for each formulation. We illustrate the application of our general results through several special examples. A by-product of our analysis is the observation that a recently derived multi-class $H$-consistency bound for cross-entropy reduces to an excess bound and is not significant. Instead, we prove a much stronger and more significant guarantee.

----

## [0] Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms

**Authors**: *Peiyao Xiao, Hao Ban, Kaiyi Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e5b96f97c1813bb75f6c28532c2ecc7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e5b96f97c1813bb75f6c28532c2ecc7-Abstract-Conference.html)

**Abstract**:

Multi-objective optimization (MOO) has become an influential framework in many machine learning problems with multiple objectives such as learning with multiple criteria and multi-task learning (MTL). In this paper, we propose a new direction-oriented multi-objective formulation by regularizing the common descent direction within a neighborhood of a direction that optimizes a linear combination of objectives such as the average loss in MTL or a weighted loss that places higher emphasis on some tasks than the others. This formulation includes GD and MGDA as special cases, enjoys the direction-oriented benefit as in CAGrad, and facilitates the design of stochastic algorithms. To solve this problem, we propose Stochastic Direction-oriented Multi-objective Gradient descent (SDMGrad) with simple SGD type of updates, and its variant SDMGrad-OS with an efficient objective sampling. We develop a comprehensive convergence analysis for the proposed methods with different loop sizes and regularization coefficients. We show that both SDMGrad and SDMGrad-OS achieve improved sample complexities to find an $\epsilon$-accurate Pareto stationary point while achieving a small $\epsilon$-level distance toward a conflict-avoidant (CA) direction. For a constant-level CA distance, their sample complexities match the best known $\mathcal{O}(\epsilon^{-2})$ without bounded function value assumption. Extensive experiments show that our methods achieve competitive or improved performance compared to existing gradient manipulation approaches in a series of tasks on multi-task supervised learning and reinforcement learning. Code is available at https://github.com/ml-opt-lab/sdmgrad.

----



[Go to the previous page](NIPS-2023-list100.md)

[Go to the next page](NIPS-2023-list102.md)

[Go to the catalog section](README.md)