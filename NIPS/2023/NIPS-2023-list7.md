## [0] Policy Optimization for Continuous Reinforcement Learning

**Authors**: *Hanyang Zhao, Wenpin Tang, David D. Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c53bc01e30711a08f6ac86919193022-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c53bc01e30711a08f6ac86919193022-Abstract-Conference.html)

**Abstract**:

We study reinforcement learning (RL) in the setting of continuous time and space, for an infinite horizon with a discounted objective and the underlying dynamics driven by a stochastic differential equation. Built upon recent advances in the continuous approach to RL, we develop a notion of occupation time (specifically for a discounted objective),  and show how it can be effectively used to derive performance difference and local approximation formulas. We further extend these results to illustrate their applications in the PG (policy gradient) and TRPO/PPO (trust region policy optimization/ proximal policy optimization) methods,  which have been familiar and powerful tools in the discrete RL setting but under-developed in continuous RL. Through numerical experiments, we demonstrate the effectiveness and advantages of our approach.

----

## [0] PrimDiffusion: Volumetric Primitives Diffusion for 3D Human Generation

**Authors**: *Zhaoxi Chen, Fangzhou Hong, Haiyi Mei, Guangcong Wang, Lei Yang, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c575c088de5cfef858b8837251f3027-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c575c088de5cfef858b8837251f3027-Abstract-Conference.html)

**Abstract**:

We present PrimDiffusion, the first diffusion-based framework for 3D human generation. Devising diffusion models for 3D human generation is difficult due to the intensive computational cost of 3D representations and the articulated topology of 3D humans. To tackle these challenges, our key insight is operating the denoising diffusion process directly on a set of volumetric primitives, which models the human body as a number of small volumes with radiance and kinematic information. This volumetric primitives representation marries the capacity of volumetric representations with the efficiency of primitive-based rendering. Our PrimDiffusion framework has three appealing properties: **1)** compact and expressive parameter space for the diffusion model, **2)** flexible representation that incorporates human prior, and **3)** decoder-free rendering for efficient novel-view and novel-pose synthesis. Extensive experiments validate that PrimDiffusion outperforms state-of-the-art methods in 3D human generation. Notably, compared to GAN-based methods, our PrimDiffusion supports real-time rendering of high-quality 3D humans at a resolution of $512\times512$ once the denoising process is done. We also demonstrate the flexibility of our framework on training-free conditional generation such as texture transfer and 3D inpainting.

----

## [0] A Closer Look at the Robustness of Contrastive Language-Image Pre-Training (CLIP)

**Authors**: *Weijie Tu, Weijian Deng, Tom Gedeon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c6be9f09e08ca166cdc0aa26306c61f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c6be9f09e08ca166cdc0aa26306c61f-Abstract-Conference.html)

**Abstract**:

Contrastive Language-Image Pre-training (CLIP) models have demonstrated remarkable generalization capabilities across multiple challenging distribution shifts. However, there is still much to be explored in terms of their robustness to the variations of specific visual factors. In real-world applications, reliable and safe systems must consider other safety measures beyond classification accuracy, such as predictive uncertainty. Yet, the effectiveness of CLIP models on such safety-related objectives is less-explored. Driven by the above, this work comprehensively investigates the safety measures of CLIP models, specifically focusing on three key properties: resilience to visual factor variations, calibrated uncertainty estimations, and the ability to detect anomalous inputs. To this end, we study $83$ CLIP models and $127$ ImageNet classifiers. They are diverse in architecture (pre)training distribution and training strategies. We consider $10$ visual factors (\emph{e.g.}, shape and pattern), $5$ types of out-of-distribution data, and $8$ natural and challenging test conditions with different shift types, such as texture, style, and perturbation shifts. Our study has unveiled several previously unknown insights into CLIP models. For instance, they are not consistently more calibrated than other ImageNet models, which contradicts existing findings. Additionally, our analysis underscores the significance of training source design by showcasing its profound influence on the three key properties. We believe our comprehensive study can shed light on and help guide the development of more robust and reliable CLIP models.

----

## [0] Model Spider: Learning to Rank Pre-Trained Models Efficiently

**Authors**: *Yi-Kai Zhang, Ting-Ji Huang, Yao-Xiang Ding, De-Chuan Zhan, Han-Jia Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c71b14637802ed08eaa3cf50342b2b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c71b14637802ed08eaa3cf50342b2b9-Abstract-Conference.html)

**Abstract**:

Figuring out which Pre-Trained Model (PTM) from a model zoo fits the target task is essential to take advantage of plentiful model resources. With the availability of numerous heterogeneous PTMs from diverse fields, efficiently selecting the most suitable one is challenging due to the time-consuming costs of carrying out forward or backward passes over all PTMs. In this paper, we propose Model Spider, which tokenizes both PTMs and tasks by summarizing their characteristics into vectors to enable efficient PTM selection. By leveraging the approximated performance of PTMs on a separate set of training tasks, Model Spider learns to construct representation and measure the fitness score between a model-task pair via their representation. The ability to rank relevant PTMs higher than others generalizes to new tasks. With the top-ranked PTM candidates, we further learn to enrich task repr. with their PTM-specific semantics to re-rank the PTMs for better selection. Model Spider balances efficiency and selection ability, making PTM selection like a spider preying on a web. Model Spider exhibits promising performance across diverse model zoos, including visual models and Large Language Models (LLMs). Code is available at https://github.com/zhangyikaii/Model-Spider.

----

## [0] Investigating how ReLU-networks encode symmetries

**Authors**: *Georg Bökman, Fredrik Kahl*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c74f005aabbf90a8f1747d99f387321-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c74f005aabbf90a8f1747d99f387321-Abstract-Conference.html)

**Abstract**:

Many data symmetries can be described in terms of group equivariance and the most common way of encoding group equivariances in neural networks is by building linear layers that are group equivariant.In this work we investigate whether equivariance of a network implies that all layers are equivariant.On the theoretical side we find cases where equivariance implies layerwise equivariance, but alsodemonstrate that this is not the case generally.Nevertheless, we conjecture that CNNs that are trained to be equivariant will exhibit layerwise equivariance and explain how this conjecture is a weaker version of the recent permutation conjecture by Entezari et al.\ [2022].We perform quantitative experiments with VGG-nets on CIFAR10 and qualitative experiments with ResNets on ImageNet to illustrate and support our theoretical findings. These experiments are not only of interest for understanding how group equivariance is encoded in ReLU-networks, but they also give a new perspective on Entezari et al.'s permutation conjecture as we find that itis typically easier to merge a network with a group-transformed version of itself than merging two different networks.

----

## [0] Optimal and Fair Encouragement Policy Evaluation and Learning

**Authors**: *Angela Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c7967a442300bff58e9d7b73aa26f24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c7967a442300bff58e9d7b73aa26f24-Abstract-Conference.html)

**Abstract**:

In consequential domains, it is often impossible to compel individuals to take treatment, so that optimal policy rules are merely suggestions in the presence of human non-adherence to treatment recommendations. In these same domains, there may be heterogeneity both in who responds in taking-up treatment, and heterogeneity in treatment efficacy. For example, in social services, a persistent puzzle is the gap in take-up of beneficial services among those who may benefit from them the most. When in addition the decision-maker has distributional preferences over both access and average outcomes, the optimal decision rule changes. We study identification, doubly-robust estimation, and robust estimation under potential violations of positivity. We consider fairness constraints such as demographic parity in treatment take-up, and other constraints, via constrained optimization. Our framework can be extended to handle algorithmic recommendations under an often-reasonable covariate-conditional exclusion restriction, using our robustness checks for lack of positivity in the recommendation. We develop a two-stage, online learning-based algorithm for solving over parametrized policy classes under general constraints to obtain variance-sensitive regret bounds. We assess improved recommendation rules in a stylized case study of optimizing recommendation of supervised release in the PSA-DMF pretrial risk-assessment tool while reducing surveillance disparities.

----

## [0] NICE: NoIse-modulated Consistency rEgularization for Data-Efficient GANs

**Authors**: *Yao Ni, Piotr Koniusz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c8047bf3ed8ef6905351608d641f02f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c8047bf3ed8ef6905351608d641f02f-Abstract-Conference.html)

**Abstract**:

Generative Adversarial Networks (GANs) are powerful tools for image synthesis. However, they require access to vast amounts of training data, which is often costly and prohibitive. Limited data affects GANs, leading to discriminator overfitting and training instability. In this paper, we present a novel approach called NoIse-modulated Consistency rEgularization (NICE) to overcome these challenges. To this end, we introduce an adaptive multiplicative noise into the discriminator to modulate its latent features. We demonstrate the effectiveness of such a modulation in preventing discriminator overfitting by adaptively reducing the Rademacher complexity of the discriminator. However, this modulation leads to an unintended consequence of increased gradient norm, which can undermine the stability of GAN training. To mitigate this undesirable effect, we impose a constraint on the discriminator, ensuring its consistency for the same inputs under different noise modulations. The constraint effectively penalizes the first and second-order gradients of latent features, enhancing GAN stability. Experimental evidence aligns with our theoretical analysis, demonstrating the reduction of generalization error and gradient penalization of NICE. This substantiates the efficacy of NICE in reducing discriminator overfitting and improving stability of GAN training. NICE achieves state-of-the-art results on CIFAR-10, CIFAR-100, ImageNet and FFHQ datasets when trained with limited data, as well as in low-shot generation tasks.

----

## [0] Not All Out-of-Distribution Data Are Harmful to Open-Set Active Learning

**Authors**: *Yang Yang, Yuxuan Zhang, Xin Song, Yi Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c8d9636f74d0207ff4f65956010f450-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c8d9636f74d0207ff4f65956010f450-Abstract-Conference.html)

**Abstract**:

Active learning (AL) methods have been proven to be an effective way to reduce the labeling effort by intelligently selecting valuable instances for annotation. Despite their great success with in-distribution (ID) scenarios, AL methods suffer from performance degradation in many real-world applications because out-of-distribution (OOD) instances are always inevitably contained in unlabeled data, which may lead to inefficient sampling. Therefore, several attempts have been explored open-set AL by strategically selecting pure ID instances while filtering OOD instances. However, concentrating solely on selecting pseudo-ID instances may cause the training constraint of the ID classifier and OOD detector. To address this issue, we propose a simple yet effective sampling scheme, Progressive Active Learning (PAL), which employs a progressive sampling mechanism to leverage the active selection of valuable OOD instances. The proposed PAL measures unlabeled instances by synergistically evaluating instances' informativeness and representativeness, and thus it can balance the pseudo-ID and pseudo-OOD instances in each round to enhance both the capacity of the ID classifier and the OOD detector. %Meanwhile, PAL measures unlabeled instances by synergistically evaluating instances' informativeness and representativeness, which can more effectively estimate the values of instances. Extensive experiments on various open-set AL scenarios demonstrate the effectiveness of the proposed PAL, compared with the state-of-the-art methods. The code is available at \url{https://github.com/njustkmg/PAL}.

----

## [0] Improving the Privacy and Practicality of Objective Perturbation for Differentially Private Linear Learners

**Authors**: *Rachel Redberg, Antti Koskela, Yu-Xiang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ceda49041816da6d5a34eb3b612607f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ceda49041816da6d5a34eb3b612607f-Abstract-Conference.html)

**Abstract**:

In the arena of privacy-preserving machine learning, differentially private stochastic gradient descent (DP-SGD) has outstripped the objective perturbation mechanism in popularity and interest. Though unrivaled in versatility, DP-SGD requires a non-trivial privacy overhead (for privately tuning the modelâ€™s hyperparameters) and a computational complexity which might be extravagant for simple models such as linear and logistic regression. This paper revamps the objective perturbation mechanism with tighter privacy analyses and new computational tools that boost it to perform competitively with DP-SGD on unconstrained convex generalized linear problems.

----

## [0] Implicit Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis

**Authors**: *Zhu Wang, Sourav Medya, Sathya N. Ravi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2cf153951b5e9b39564fc4a0ef6adc1a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2cf153951b5e9b39564fc4a0ef6adc1a-Abstract-Conference.html)

**Abstract**:

Deep network models are often purely inductive during both training and inference on unseen data. When these models are used for prediction, but they may fail to capture important semantic information and implicit dependencies within datasets. Recent advancements have shown that combining multiple modalities in large-scale vision and language settings can improve understanding and generalization performance. However, as the model size increases, fine-tuning and deployment become computationally expensive, even for a small number of downstream tasks. Moreover, it is still unclear how domain or prior modal knowledge can be specified in a backpropagation friendly manner, especially in large-scale and noisy settings. To address these challenges, we propose a simplified alternative of combining features from pretrained deep networks and freely available semantic explicit knowledge. In order to remove irrelevant explicit knowledge that does not correspond well to the images, we introduce an implicit Differentiable Out-of-Distribution (OOD) detection layer. This layer addresses outlier detection by solving for fixed points of a differentiable function and using the last iterate of fixed point solver to backpropagate. In practice, we apply our model on several vision and language downstream tasks including visual question answering, visual reasoning, and image-text retrieval on different datasets. Our experiments show that it is possible to design models that perform similarly to state-of-the-art results but with significantly fewer samples and less training time. Our models and code are available here: https://github.com/ellenzhuwang/implicit_vkood

----

## [0] DSR: Dynamical Surface Representation as Implicit Neural Networks for Protein

**Authors**: *Daiwen Sun, He Huang, Yao Li, Xinqi Gong, Qiwei Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d025936bae21d2c2d4cc74779aa77c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d025936bae21d2c2d4cc74779aa77c7-Abstract-Conference.html)

**Abstract**:

We propose a novel neural network-based approach to modeling protein dynamics using an implicit representation of a proteinâ€™s surface in 3D and time. Our method utilizes the zero-level set of signed distance functions (SDFs) to represent protein surfaces, enabling temporally and spatially continuous representations of protein dynamics. Our experimental results demonstrate that our model accurately captures protein dynamic trajectories and can interpolate and extrapolate in 3D and time. Importantly, this is the first study to introduce this method and successfully model large-scale protein dynamics. This approach offers a promising alternative to current methods, overcoming the limitations of first-principles-based and deep learning methods, and provides a more scalable and efficient approach to modeling protein dynamics. Additionally, our surface representation approach simplifies calculations and allows identifying movement trends and amplitudes of protein domains, making it a useful tool for protein dynamics research. Codes are available at https://github.com/Sundw-818/DSR, and we have a project webpage that shows some video results, https://sundw-818.github.io/DSR/.

----

## [0] A Theory of Transfer-Based Black-Box Attacks: Explanation and Implications

**Authors**: *Yanbo Chen, Weiwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d0842550e6d92b0e27e7e810b1a4792-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d0842550e6d92b0e27e7e810b1a4792-Abstract-Conference.html)

**Abstract**:

Transfer-based attacks are a practical method of black-box adversarial attacks, in which the attacker aims to craft adversarial examples from a source (surrogate) model that is transferable to the target model. A wide range of empirical works has tried to explain the transferability of adversarial examples from different angles. However, these works only provide ad hoc explanations without quantitative analyses. The theory behind transfer-based attacks remains a mystery.This paper studies transfer-based attacks under a unified theoretical framework. We propose an explanatory model, called the manifold attack model, that formalizes popular beliefs and explains the existing empirical results. Our model explains why adversarial examples are transferable even when the source model is inaccurate. Moreover, our model implies that the existence of transferable adversarial examples depends on the “curvature” of the data manifold, which quantitatively explains why the success rates of transfer-based attacks are hard to improve. We also discuss the expressive power and the possible extensions of our model in general applications.

----

## [0] Explaining V1 Properties with a Biologically Constrained Deep Learning Architecture

**Authors**: *Galen Pogoncheff, Jacob Granley, Michael Beyeler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d1ef4aba0503226330661d74fdb236e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d1ef4aba0503226330661d74fdb236e-Abstract-Conference.html)

**Abstract**:

Convolutional neural networks (CNNs) have recently emerged as promising models of the ventral visual stream, despite their lack of biological specificity.While current state-of-the-art models of the primary visual cortex (V1) have surfaced from training with adversarial examples and extensively augmented data, these models are still unable to explain key neural properties observed in V1 that arise from biological circuitry.To address this gap, we systematically incorporated neuroscience-derived architectural components into CNNs to identify a set of mechanisms and architectures that more comprehensively explain V1 activity.Upon enhancing task-driven CNNs with architectural components that simulate center-surround antagonism, local receptive fields, tuned normalization, and cortical magnification, we uncover models with latent representations that yield state-of-the-art explanation of V1 neural activity and tuning properties.Moreover, analyses of the learned parameters of these components and stimuli that maximally activate neurons of the evaluated networks provide support for their role in explaining neural properties of V1.Our results highlight an important advancement in the field of NeuroAI, as we systematically establish a set of architectural components that contribute to unprecedented explanation of V1.The neuroscience insights that could be gleaned from increasingly accurate in-silico models of the brain have the potential to greatly advance the fields of both neuroscience and artificial intelligence.

----

## [0] Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models

**Authors**: *Naman Deep Singh, Francesco Croce, Matthias Hein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d3b007613940def7a5ec9d6d635937b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d3b007613940def7a5ec9d6d635937b-Abstract-Conference.html)

**Abstract**:

While adversarial training has been extensively studied for ResNet architectures and low resolution datasets like CIFAR-10, much less is known for ImageNet. Given the recent debate about whether transformers are more robust than convnets, we revisit adversarial training on ImageNet comparing ViTs and ConvNeXts. Extensive experiments show that minor changes in architecture, most notably replacing PatchStem with ConvStem, and training scheme have a significant impact on the achieved robustness. These changes not only increase robustness in the seen $\ell_\infty$-threat model, but even more so improve generalization to unseen $\ell_1/\ell_2$-attacks. Our modified ConvNeXt, ConvNeXt + ConvStem, yields the most robust $\ell_\infty$-models across different ranges of model parameters and FLOPs, while our ViT + ConvStem yields the best generalization to unseen threat models.

----

## [0] URL: A Representation Learning Benchmark for Transferable Uncertainty Estimates

**Authors**: *Michael Kirchhof, Bálint Mucsányi, Seong Joon Oh, Enkelejda Kasneci*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d421cd0e763f9f01958a30bace955bf-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d421cd0e763f9f01958a30bace955bf-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Representation learning has significantly driven the field to develop pretrained models that can act as a valuable starting point when transferring to new datasets. With the rising demand for reliable machine learning and uncertainty quantification, there is a need for pretrained models that not only provide embeddings but also transferable uncertainty estimates. To guide the development of such models, we propose the Uncertainty-aware Representation Learning (URL) benchmark. Besides the transferability of the representations, it also measures the zero-shot transferability of the uncertainty estimate using a novel metric. We apply URL to evaluate ten uncertainty quantifiers that are pretrained on ImageNet and transferred to eight downstream datasets. We find that approaches that focus on the uncertainty of the representation itself or estimate the prediction risk directly outperform those that are based on the probabilities of upstream classes. Yet, achieving transferable uncertainty quantification remains an open challenge. Our findings indicate that it is not necessarily in conflict with traditional representation learning goals. Code is available at https://github.com/mkirchhof/url.

----

## [0] FineMoGen: Fine-Grained Spatio-Temporal Motion Generation and Editing

**Authors**: *Mingyuan Zhang, Huirong Li, Zhongang Cai, Jiawei Ren, Lei Yang, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d52879ef2ba487445ca2e143b104c3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d52879ef2ba487445ca2e143b104c3b-Abstract-Conference.html)

**Abstract**:

Text-driven motion generation has achieved substantial progress with the emergence of diffusion models. However, existing methods still struggle to generate complex motion sequences that correspond to fine-grained descriptions, depicting detailed and accurate spatio-temporal actions.This lack of fine controllability limits the usage of motion generation to a larger audience. To tackle these challenges, we present FineMoGen, a diffusion-based motion generation and editing framework that can synthesize fine-grained motions, with spatial-temporal composition to the user instructions. Specifically, FineMoGen builds upon diffusion model with a novel transformer architecture dubbed Spatio-Temporal Mixture Attention SAMI. SAMI optimizes the generation of the global attention template from two perspectives: 1) explicitly modeling the constraints of spatio-temporal composition; and 2) utilizing sparsely-activated mixture-of-experts to adaptively extract fine-grained features. To facilitate a large-scale study on this new fine-grained motion generation task, we contribute the HuMMan-MoGen dataset, which consists of 2,968 videos and 102,336 fine-grained spatio-temporal descriptions. Extensive experiments validate that FineMoGen  exhibits superior motion generation quality over state-of-the-art methods. Notably, FineMoGen further enables zero-shot motion editing capabilities with the aid of modern large language models (LLM), which faithfully manipulates motion sequences with fine-grained instructions.

----

## [0] Stabilizing the Optimization of Neural Signed Distance Functions and Finer Shape Representation

**Authors**: *Huizong Yang, Yuxin Sun, Ganesh Sundaramoorthi, Anthony J. Yezzi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d6336c1c2987e9d1d9894edd593478d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d6336c1c2987e9d1d9894edd593478d-Abstract-Conference.html)

**Abstract**:

We present new insights and a novel paradigm for learning implicit neural representations (INR) of shapes. In particular, we shed light on the popular eikonal loss used for imposing a signed distance function constraint in INR. We show analytically that as the representation power of the network increases, the optimization approaches a partial differential equation (PDE) in the continuum limit that is unstable. We show that this instability can manifest in existing network optimization, leading to irregularities in the reconstructed surface and/or convergence to sub-optimal local minima, and thus fails to capture fine geometric and topological structure. We show analytically how other terms added to the loss, currently used in the literature for other purposes, can actually eliminate these instabilities. However, such terms can over-regularize the surface, preventing the representation of fine shape detail. Based on a similar PDE theory for the continuum limit, we introduce a new regularization term that still counteracts the eikonal instability but without over-regularizing. Furthermore, since stability is now guaranteed in the continuum limit, this stabilization also allows for considering new network structures that are able to represent finer shape detail. We introduce such a structure based on quadratic layers. Experiments on multiple benchmark data sets show that our new regularization and network are able to capture more precise shape details and more accurate topology than existing state-of-the-art.

----

## [0] Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale

**Authors**: *Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, Wei-Ning Hsu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d8911db9ecedf866015091b28946e15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d8911db9ecedf866015091b28946e15-Abstract-Conference.html)

**Abstract**:

Large-scale generative models such as GPT and DALL-E have revolutionized the research community. These models not only generate high fidelity outputs, but are also generalists which can solve tasks not explicitly taught. In contrast, speech generative models are still primitive in terms of scale and task generalization. In this paper, we present Voicebox, the most versatile text-guided generative model for speech at scale. Voicebox is a non-autoregressive flow-matching model trained to infill speech, given audio context and text, trained on over 50K hours of speech that are not filtered or enhanced. Similar to GPT, Voicebox can perform many different tasks through in-context learning, but is more flexible as it can also condition on future context. Voicebox can be used for mono or cross-lingual zero-shot text-to-speech synthesis, noise removal, content editing, style conversion, and diverse sample generation. In particular, Voicebox outperforms the state-of-the-art zero-shot TTS model VALL-E on both intelligibility (5.9\% vs 1.9\% word error rates) and audio similarity (0.580 vs 0.681) while being up to 20 times faster. Audio samples can be found in \url{https://voicebox.metademolab.com}.

----

## [0] Optimizing Solution-Samplers for Combinatorial Problems: The Landscape of Policy-Gradient Method

**Authors**: *Constantine Caramanis, Dimitris Fotakis, Alkis Kalavasis, Vasilis Kontonis, Christos Tzamos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d950a2cfd8a75124c178a89545b97fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d950a2cfd8a75124c178a89545b97fd-Abstract-Conference.html)

**Abstract**:

Deep Neural Networks and Reinforcement Learning methods have empirically shown great promise in tackling challenging combinatorial problems. In those methods a deep neural network is used as a solution generator which is then trained by gradient-based methods (e.g., policy gradient) to successively obtain better solution distributions.In this work we introduce a novel theoretical framework for analyzing the effectiveness of such methods. We ask whether there exist generative models that (i) are expressive enough to generate approximately optimal solutions; (ii) have a tractable, i.e, polynomial in the size of the input, number of parameters; (iii) their optimization landscape is benign in the sense that it does not contain sub-optimal stationary points. Our main contribution is a positive answer to this question. Our result holds for a broad class of combinatorial problems including Max- and Min-Cut, Max-$k$-CSP, Maximum-Weight-Bipartite-Matching, and the Traveling Salesman Problem. As a byproduct of our analysis we introduce a novel regularization process over vanilla gradient descent and provide theoretical and experimental evidence that it helps address vanishing-gradient issues and escape bad stationary points.

----

## [0] SoTTA: Robust Test-Time Adaptation on Noisy Data Streams

**Authors**: *Taesik Gong, Yewon Kim, Taeckyung Lee, Sorn Chottananurak, Sung-Ju Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2da53cd1abdae59150e35f4693834f32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2da53cd1abdae59150e35f4693834f32-Abstract-Conference.html)

**Abstract**:

Test-time adaptation (TTA) aims to address distributional shifts between training and testing data using only unlabeled test data streams for continual model adaptation. However, most TTA methods assume benign test streams, while test samples could be unexpectedly diverse in the wild. For instance, an unseen object or noise could appear in autonomous driving. This leads to a new threat to existing TTA algorithms; we found that prior TTA algorithms suffer from those noisy test samples as they blindly adapt to incoming samples. To address this problem, we present Screening-out Test-Time Adaptation (SoTTA), a novel TTA algorithm that is robust to noisy samples. The key enabler of SoTTA is two-fold: (i) input-wise robustness via high-confidence uniform-class sampling that effectively filters out the impact of noisy samples and (ii) parameter-wise robustness via entropy-sharpness minimization that improves the robustness of model parameters against large gradients from noisy samples. Our evaluation with standard TTA benchmarks with various noisy scenarios shows that our method outperforms state-of-the-art TTA methods under the presence of noisy samples and achieves comparable accuracy to those methods without noisy samples. The source code is available at https://github.com/taeckyung/SoTTA.

----

## [0] FouriDown: Factoring Down-Sampling into Shuffling and Superposing

**Authors**: *Qi Zhu, Man Zhou, Jie Huang, Naishan Zheng, Hongzhi Gao, Chongyi Li, Yuan Xu, Feng Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2dae7d1ccf1edf76f8ce7c282bdf4730-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2dae7d1ccf1edf76f8ce7c282bdf4730-Abstract-Conference.html)

**Abstract**:

Spatial down-sampling techniques, such as strided convolution, Gaussian, and Nearest down-sampling, are essential in deep neural networks. In this study, we revisit the working mechanism of the spatial down-sampling family and analyze the biased effects caused by the static weighting strategy employed in previous approaches. To overcome this limitation, we propose a novel  down-sampling paradigm in the Fourier domain, abbreviated as FouriDown, which unifies existing down-sampling techniques. Drawing inspiration from the signal sampling theorem, we parameterize the non-parameter static weighting down-sampling operator as a learnable and context-adaptive operator within a unified Fourier function. Specifically, we organize the corresponding frequency positions of the 2D plane in a physically-closed manner within a single channel dimension. We then perform point-wise channel shuffling based on an indicator that determines whether a channel's signal frequency bin is susceptible to aliasing, ensuring the consistency of the weighting parameter learning. FouriDown, as a generic operator, comprises four key components: 2D discrete Fourier transform, context shuffling rules, Fourier weighting-adaptively superposing rules, and 2D inverse Fourier transform. These components can be easily integrated into existing image restoration networks. To demonstrate the efficacy of FouriDown, we conduct extensive experiments on image de-blurring and low-light image enhancement. The results consistently show that FouriDown can provide significant performance improvements. We will make the code publicly available to facilitate further exploration and application of FouriDown.

----

## [0] Participatory Personalization in Classification

**Authors**: *Hailey Joren, Chirag Nagpal, Katherine A. Heller, Berk Ustun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2dbb8bfe4cd3875609b23799830ee865-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2dbb8bfe4cd3875609b23799830ee865-Abstract-Conference.html)

**Abstract**:

Machine learning models are often personalized based on information that is protected, sensitive, self-reported, or costly to acquire. These models use information about people, but do not facilitate nor inform their consent. Individuals cannot opt out of reporting information that a model needs to personalize their predictions nor tell if they benefit from personalization in the first place. We introduce a new family of prediction models, called participatory systems, that let individuals opt into personalization at prediction time. We present a model-agnostic algorithm to learn participatory systems for supervised learning tasks where models are personalized with categorical group attributes. We conduct a comprehensive empirical study of participatory systems in clinical prediction tasks, comparing them to common approaches for personalization and imputation. Our results show that participatory systems can facilitate and inform consent in a way that improves performance and privacy across all groups who report personal data.

----

## [0] A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks

**Authors**: *Vignesh Kothapalli, Tom Tirer, Joan Bruna*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2dd8a2a8685602586c1173f0b644d0e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2dd8a2a8685602586c1173f0b644d0e3-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) have become increasingly popular for classification tasks on graph-structured data. Yet, the interplay between graph topology and feature evolution in GNNs is not well understood. In this paper, we focus on node-wise classification, illustrated with community detection on stochastic block model graphs, and explore the feature evolution through the lens of the "Neural Collapse" (NC) phenomenon. When training instance-wise deep classifiers (e.g. for image classification) beyond the zero training error point, NC demonstrates a reduction in the deepest features' within-class variability and an increased alignment of their class means to certain symmetric structures. We start with an empirical study that shows that a decrease in within-class variability is also prevalent in the node-wise classification setting, however, not to the extent observed in the instance-wise case. Then, we theoretically study this distinction. Specifically, we show that even an "optimistic" mathematical model requires that the graphs obey a strict structural condition in order to possess a minimizer with exact collapse. Furthermore, by studying the gradient dynamics of this model, we provide reasoning for the partial collapse observed empirically. Finally, we present a study on the evolution of within- and between-class feature variability across layers of a well-trained GNN and contrast the behavior with spectral methods.

----

## [0] ResoNet: Noise-Trained Physics-Informed MRI Off-Resonance Correction

**Authors**: *Alfredo De Goyeneche Macaya, Shreya Ramachandran, Ke Wang, Ekin Karasan, Joseph Y. Cheng, Stella X. Yu, Michael Lustig*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e0bd92a1d3600d4288df51ac5e6be5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e0bd92a1d3600d4288df51ac5e6be5f-Abstract-Conference.html)

**Abstract**:

Magnetic Resonance Imaging (MRI) is a powerful medical imaging modality that offers diagnostic information without harmful ionizing radiation. Unlike optical imaging, MRI sequentially samples the spatial Fourier domain (k-space) of the image. Measurements are collected in multiple shots, or readouts, and in each shot, data along a smooth trajectory is sampled.Conventional MRI data acquisition relies on sampling k-space row-by-row in short intervals, which is slow and inefficient. More efficient, non-Cartesian sampling trajectories (e.g., Spirals) use longer data readout intervals, but are more susceptible to magnetic field inhomogeneities, leading to off-resonance artifacts. Spiral trajectories cause off-resonance blurring in the image, and the mathematics of this blurring resembles that of optical blurring, where magnetic field variation corresponds to depth and readout duration to aperture size. Off-resonance blurring is a system issue with a physics-based, accurate forward model. We present a physics-informed deep learning framework for off-resonance correction in MRI, which is trained exclusively on synthetic, noise-like data with representative marginal statistics. Our approach allows for fat/water separation and is compatible with parallel imaging acceleration. Through end-to-end training using synthetic randomized data (i.e., noise-like images, coil sensitivities, field maps), we train the network to reverse off-resonance effects across diverse anatomies and contrasts without retraining. We demonstrate the effectiveness of our approach through results on phantom and in-vivo data. This work has the potential to facilitate the clinical adoption of non-Cartesian sampling trajectories, enabling efficient, rapid, and motion-robust MRI scans. Code is publicly available at: https://github.com/mikgroup/ResoNet.

----

## [0] Eliminating Domain Bias for Federated Learning in Representation Space

**Authors**: *Jianqing Zhang, Yang Hua, Jian Cao, Hao Wang, Tao Song, Zhengui Xue, Ruhui Ma, Haibing Guan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e0d3c6ad1a4d85bef3cfe63af58bc76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e0d3c6ad1a4d85bef3cfe63af58bc76-Abstract-Conference.html)

**Abstract**:

Recently, federated learning (FL) is popular for its privacy-preserving and collaborative learning abilities. However, under statistically heterogeneous scenarios, we observe that biased data domains on clients cause a representation bias phenomenon and further degenerate generic representations during local training, i.e., the representation degeneration phenomenon. To address these issues, we propose a general framework Domain Bias Eliminator (DBE) for FL. Our theoretical analysis reveals that DBE can promote bi-directional knowledge transfer between server and client, as it reduces the domain discrepancy between server and client in representation space. Besides, extensive experiments on four datasets show that DBE can greatly improve existing FL methods in both generalization and personalization abilities. The DBE-equipped FL method can outperform ten state-of-the-art personalized FL methods by a large margin. Our code is public at https://github.com/TsingZ0/DBE.

----

## [0] Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression

**Authors**: *Allan Raventós, Mansheej Paul, Feng Chen, Surya Ganguli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e10b2c2e1aa4f8083c37dfe269873f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e10b2c2e1aa4f8083c37dfe269873f8-Abstract-Conference.html)

**Abstract**:

Pretrained transformers exhibit the remarkable ability of in-context learning (ICL): they can learn tasks from just a few examples provided in the prompt without updating any weights. This raises a foundational question: can ICL solve fundamentally new tasks that are very different from those seen during pretraining? To probe this question, we examine ICL’s performance on linear regression while varying the diversity of tasks in the pretraining dataset. We empirically demonstrate a task diversity threshold for the emergence of ICL. Below this threshold, the pretrained transformer cannot solve unseen regression tasks, instead behaving like a Bayesian estimator with the non-diverse pretraining task distribution as the prior. Beyond this threshold, the transformer significantly outperforms this estimator; its behavior aligns with that of ridge regression, corresponding to a Gaussian prior over all tasks, including those not seen during pretraining. Thus, when pretrained on data with task diversity greater than the threshold, transformers can optimally solve fundamentally new tasks in-context. Importantly, this capability hinges on it deviating from the Bayes optimal estimator with the pretraining distribution as the prior. This study also explores the effect of regularization, model capacity and task structure and underscores, in a concrete example, the critical role of task diversity, alongside data and model scale, in the emergence of ICL.

----

## [0] Two-Stage Predict+Optimize for MILPs with Unknown Parameters in Constraints

**Authors**: *Xinyi Hu, Jasper C. H. Lee, Jimmy Ho-Man Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e14be0332c04c76742710e417cedb2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e14be0332c04c76742710e417cedb2a-Abstract-Conference.html)

**Abstract**:

Consider the setting of constrained optimization, with some parameters unknown at solving time and requiring prediction from relevant features. Predict+Optimize is a recent framework for end-to-end training supervised learning models for such predictions, incorporating information about the optimization problem in the training process in order to yield better predictions in terms of the quality of the predicted solution under the true parameters. Almost all prior works have focused on the special case where the unknowns appear only in the optimization objective and not the constraints. Hu et al. proposed the first adaptation of Predict+Optimize to handle unknowns appearing in constraints, but the framework has somewhat ad-hoc elements, and they provided a training algorithm only for covering and packing linear programs. In this work, we give a new simpler and more powerful framework called Two-Stage Predict+Optimize, which we believe should be the canonical framework for the Predict+Optimize setting. We also give a training algorithm usable for all mixed integer linear programs, vastly generalizing the applicability of the framework. Experimental results demonstrate the superior prediction performance of our training framework over all classical and state-of-the-art methods.

----

## [0] Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective

**Authors**: *Zhiding Liu, Mingyue Cheng, Zhi Li, Zhenya Huang, Qi Liu, Yanhu Xie, Enhong Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html)

**Abstract**:

Deep learning models have progressively advanced time series forecasting due to their powerful capacity in capturing sequence dependence. Nevertheless, it is still challenging to make accurate predictions due to the existence of non-stationarity in real-world data, denoting the data distribution rapidly changes over time. To mitigate such a dilemma, several efforts have been conducted by reducing the non-stationarity with normalization operation. However, these methods typically overlook the distribution discrepancy between the input series and the horizon series, and assume that all time points within the same instance share the same statistical properties, which is too ideal and may lead to suboptimal relative improvements. To this end, we propose a novel slice-level adaptive normalization, referred to \textbf{SAN}, which is a novel scheme for empowering time series forecasting with more flexible normalization and denormalization. SAN includes two crucial designs. First, SAN tries to eliminate the non-stationarity of time series in units of a local temporal slice (i.e., sub-series) rather than a global instance. Second, SAN employs a slight network module to independently model the evolving trends of statistical properties of raw time series. Consequently, SAN could serve as a general model-agnostic plugin and better alleviate the impact of the non-stationary nature of time series data. We instantiate the proposed SAN on four widely used forecasting models and test their prediction results on benchmark datasets to evaluate its effectiveness. Also, we report some insightful findings to deeply analyze and understand our proposed SAN. We make our codes publicly available.

----

## [0] Distance-Restricted Folklore Weisfeiler-Leman GNNs with Provable Cycle Counting Power

**Authors**: *Junru Zhou, Jiarui Feng, Xiyuan Wang, Muhan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e2e7c2e3c2e70fa2e9756dce728fcca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e2e7c2e3c2e70fa2e9756dce728fcca-Abstract-Conference.html)

**Abstract**:

The ability of graph neural networks (GNNs) to count certain graph substructures, especially cycles, is important for the success of GNNs on a wide range of tasks. It has been recently used as a popular metric for evaluating the expressive power of GNNs. Many of the proposed GNN models with provable cycle counting power are based on subgraph GNNs, i.e., extracting a bag of subgraphs from the input graph, generating representations for each subgraph, and using them to augment the representation of the input graph. However, those methods require heavy preprocessing, and suffer from high time and memory costs. In this paper, we overcome the aforementioned limitations of subgraph GNNs by proposing a novel class of GNNs---$d$-Distance-Restricted FWL(2) GNNs, or $d$-DRFWL(2) GNNs, based on the well-known FWL(2) algorithm. As a heuristic method for graph isomorphism testing, FWL(2) colors all node pairs in a graph and performs message passing among those node pairs. In order to balance the expressive power and complexity, $d$-DRFWL(2) GNNs simplify FWL(2) by restricting the range of message passing to node pairs whose mutual distances are at most $d$. This way, $d$-DRFWL(2) GNNs exploit graph sparsity while avoiding the expensive subgraph extraction operations in subgraph GNNs, making both the time and space complexity lower. We theoretically investigate both the discriminative power and the cycle counting power of $d$-DRFWL(2) GNNs. Our most important finding is that $d$-DRFWL(2) GNNs have provably strong cycle counting power even with $d=2$: they can count all 3, 4, 5, 6-cycles. Since 6-cycles (e.g., benzene rings) are ubiquitous in organic molecules, being able to detect and count them is crucial for achieving robust and generalizable performance on molecular tasks. Experiments on both synthetic datasets and molecular datasets verify our theory. To the best of our knowledge, 2-DRFWL(2) GNN is the most efficient GNN model to date (both theoretically and empirically) that can count up to 6-cycles.

----

## [0] Computing a human-like reaction time metric from stable recurrent vision models

**Authors**: *Lore Goetschalckx, Lakshmi Narasimhan Govindarajan, Alekh Karkada Ashok, Aarit Ahuja, David L. Sheinberg, Thomas Serre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e351740d4ec4200df6160f34cd181c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e351740d4ec4200df6160f34cd181c3-Abstract-Conference.html)

**Abstract**:

The meteoric rise in the adoption of deep neural networks as computational models of vision has inspired efforts to ``align‚Äù these models with humans. One dimension of interest for alignment includes behavioral choices, but moving beyond characterizing choice patterns to capturing temporal aspects of visual decision-making has been challenging. Here, we sketch a general-purpose methodology to construct computational accounts of reaction times from a stimulus-computable, task-optimized model. Specifically, we introduce a novel metric leveraging insights from subjective logic theory summarizing evidence accumulation in recurrent vision models. We demonstrate that our metric aligns with patterns of human reaction times for stimulus manipulations across four disparate visual decision-making tasks spanning perceptual grouping, mental simulation, and scene categorization. This work paves the way for exploring the temporal alignment of model and human visual strategies in the context of various other cognitive tasks toward generating testable hypotheses for neuroscience. Links to the code and data can be found on the project page: https://serre-lab.github.io/rnnrtssite/.

----

## [0] ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence

**Authors**: *Ben Dai, Yixuan Qiu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e37e56599e3f49cc899f40ae4f5d1fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e37e56599e3f49cc899f40ae4f5d1fa-Abstract-Conference.html)

**Abstract**:

Empirical risk minimization (ERM) is a crucial framework that offers a general approach to handling a broad range of machine learning tasks. In this paper, we propose a novel algorithm, called ReHLine, for minimizing a set of regularized ERMs with convex piecewise linear-quadratic loss functions and optional linear constraints. The proposed algorithm can effectively handle diverse combinations of loss functions, regularization, and constraints, making it particularly well-suited for complex domain-specific problems. Examples of such problems include FairSVM, elastic net regularized quantile regression, Huber minimization, etc. In addition, ReHLine enjoys a provable linear convergence rate and exhibits a per-iteration computational complexity that scales linearly with the sample size. The algorithm is implemented with both Python and R interfaces, and its performance is benchmarked on various tasks and datasets. Our experimental results demonstrate that ReHLine significantly surpasses generic optimization solvers in terms of computational efficiency on large-scale datasets. Moreover, it also outperforms specialized solvers such as Liblinear in SVMs, hqreg in Huber minimization, and Lightning (SAGA, SAG, SDCA, SVRG) in smoothed SVMs, exhibiting exceptional flexibility and efficiency. The source code, project page, accompanying software, and the Python/R interface can be accessed through the link: https://github.com/softmin/ReHLine.

----

## [0] Improved Frequency Estimation Algorithms with and without Predictions

**Authors**: *Anders Aamand, Justin Y. Chen, Huy Lê Nguyen, Sandeep Silwal, Ali Vakilian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e49934cac6cb8604b0c67cfa0828718-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e49934cac6cb8604b0c67cfa0828718-Abstract-Conference.html)

**Abstract**:

Estimating frequencies of elements appearing in a data stream is a key task in large-scale data analysis. Popular sketching approaches to this problem (e.g., CountMin and CountSketch) come with worst-case guarantees that probabilistically bound the error of the estimated frequencies for any possible input. The work of Hsu et al.~(2019) introduced the idea of using machine learning to tailor sketching algorithms to the specific data distribution they are being run on. In particular, their learning-augmented frequency estimation algorithm uses a learned heavy-hitter oracle which predicts which elements will appear many times in the stream. We give a novel algorithm, which in some parameter regimes, already theoretically outperforms the learning based algorithm of Hsu et al. without the use of any predictions. Augmenting our algorithm with heavy-hitter predictions further reduces the error and improves upon the state of the art. Empirically, our algorithms achieve superior performance in all experiments compared to prior approaches.

----

## [0] BCDiff: Bidirectional Consistent Diffusion for Instantaneous Trajectory Prediction

**Authors**: *Rongqing Li, Changsheng Li, Dongchun Ren, Guangyi Chen, Ye Yuan, Guoren Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e57e2c14232a7b99cf76213e190822d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e57e2c14232a7b99cf76213e190822d-Abstract-Conference.html)

**Abstract**:

The objective of pedestrian trajectory prediction is to estimate the future paths of pedestrians by leveraging historical observations, which plays a vital role in ensuring the safety of self-driving vehicles and navigation robots. Previous works usually rely on a sufficient amount of observation time to accurately predict future trajectories. However, there are many real-world situations where the model lacks sufficient time to observe, such as when pedestrians abruptly emerge from blind spots, resulting in inaccurate predictions and even safety risks. Therefore, it is necessary to perform trajectory prediction based on instantaneous observations, which has rarely been studied before. In this paper, we propose a Bi-directional Consistent Diffusion framework tailored for instantaneous trajectory prediction, named BCDiff. At its heart, we develop two coupled diffusion models by designing a mutual guidance mechanism which can bidirectionally and consistently generate unobserved historical trajectories and future trajectories step-by-step,  to utilize the complementary information between them. Specifically, at each step, the predicted unobserved historical trajectories and limited observed trajectories guide one diffusion model to generate future trajectories, while the predicted future trajectories and observed trajectories guide the other diffusion model to predict unobserved historical trajectories. Given the presence of relatively high noise in the generated trajectories during the initial steps, we introduce a gating mechanism to learn the weights between the predicted trajectories and the limited observed trajectories for automatically balancing their contributions. By means of this iterative and mutually guided generation process, both the future and unobserved historical trajectories undergo continuous refinement, ultimately leading to accurate predictions. Essentially, BCDiff is an encoder-free framework that can be compatible with existing trajectory prediction models in principle. Experiments show that our proposed BCDiff significantly improves the accuracy of instantaneous trajectory prediction on the ETH/UCY and Stanford Drone datasets, compared to related approaches.

----

## [0] Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition

**Authors**: *Yuhang Zhang, Yaqi Li, Lixiong Qin, Xuannan Liu, Weihong Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e6744370a8616c90d1e3b7a41993b7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e6744370a8616c90d1e3b7a41993b7c-Abstract-Conference.html)

**Abstract**:

Facial expression data is characterized by a significant imbalance, with most collected data showing happy or neutral expressions and fewer instances of fear or disgust. This imbalance poses challenges to facial expression recognition (FER) models, hindering their ability to fully understand various human emotional states. Existing FER methods typically report overall accuracy on highly imbalanced test sets but exhibit low performance in terms of the mean accuracy across all expression classes. In this paper, our aim is to address the imbalanced FER problem. Existing methods primarily focus on learning knowledge of minor classes solely from minor-class samples. However, we propose a novel approach to extract extra knowledge related to the minor classes from both major and minor class samples. Our motivation stems from the belief that FER resembles a distribution learning task, wherein a sample may contain information about multiple classes. For instance, a sample from the major class surprise might also contain useful features of the minor class fear. Inspired by that, we propose a novel method that leverages re-balanced attention maps to regularize the model, enabling it to extract transformation invariant information about the minor classes from all training samples. Additionally, we introduce re-balanced smooth labels to regulate the cross-entropy loss, guiding the model to pay more attention to the minor classes by utilizing the extra information regarding the label distribution of the imbalanced training data. Extensive experiments on different datasets and backbones show that the two proposed modules work together to regularize the model and achieve state-of-the-art performance under the imbalanced FER task. Code is available at https://github.com/zyh-uaiaaaa.

----

## [0] ARTree: A Deep Autoregressive Model for Phylogenetic Inference

**Authors**: *Tianyu Xie, Cheng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e9e513860b1342f3a12ebecf0528a21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e9e513860b1342f3a12ebecf0528a21-Abstract-Conference.html)

**Abstract**:

Designing flexible probabilistic models over tree topologies is important for developing efficient phylogenetic inference methods. To do that, previous works often leverage the similarity of tree topologies via hand-engineered heuristic features which would require domain expertise and may suffer from limited approximation capability. In this paper, we propose a deep autoregressive model for phylogenetic inference based on graph neural networks (GNNs), called ARTree. By decomposing a tree topology into a sequence of leaf node addition operations and modeling the involved conditional distributions based on learnable topological features via GNNs, ARTree can provide a rich family of distributions over tree topologies that have simple sampling algorithms, without using heuristic features. We demonstrate the effectiveness and efficiency of our method on a benchmark of challenging real data tree topology density estimation and variational Bayesian phylogenetic inference problems.

----

## [0] A One-Size-Fits-All Approach to Improving Randomness in Paper Assignment

**Authors**: *Yixuan Xu, Steven Jecmen, Zimeng Song, Fei Fang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e9f9cde1b709281a06dd14f679e4c51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e9f9cde1b709281a06dd14f679e4c51-Abstract-Conference.html)

**Abstract**:

The assignment of papers to reviewers is a crucial part of the peer review processes of large publication venues, where organizers (e.g., conference program chairs) rely on algorithms to perform automated paper assignment. As such, a major challenge for the organizers of these processes is to specify paper assignment algorithms that find appropriate assignments with respect to various desiderata. Although the main objective when choosing a good paper assignment is to maximize the expertise of each reviewer for their assigned papers, several other considerations make introducing randomization into the paper assignment desirable: robustness to malicious behavior, the ability to evaluate alternative paper assignments, reviewer diversity, and reviewer anonymity. However, it is unclear in what way one should randomize the paper assignment in order to best satisfy all of these considerations simultaneously. In this work, we present a practical, one-size-fits-all method for randomized paper assignment intended to perform well across different motivations for randomness. We show theoretically and experimentally that our method outperforms currently-deployed methods for randomized paper assignment on several intuitive randomness metrics, demonstrating that the randomized assignments produced by our method are general-purpose.

----

## [0] Loss Dynamics of Temporal Difference Reinforcement Learning

**Authors**: *Blake Bordelon, Paul Masset, Henry Kuo, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ea04b568a2deb2d000c59f3a72829b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ea04b568a2deb2d000c59f3a72829b5-Abstract-Conference.html)

**Abstract**:

Reinforcement learning has been successful across several applications in which agents have to learn to act in environments with sparse feedback. However, despite this empirical success there is still a lack of theoretical understanding of how the parameters of reinforcement learning models and the features used to represent states interact to control the dynamics of learning. In this work, we use concepts from statistical physics, to study the typical case learning curves for temporal difference learning of a value function with linear function approximators. Our theory is derived under a Gaussian equivalence hypothesis where averages over the random trajectories are replaced with temporally correlated Gaussian feature averages and we validate our assumptions on small scale Markov Decision Processes. We find that the stochastic semi-gradient noise due to subsampling the space of possible episodes leads to significant plateaus in the value error, unlike in traditional gradient descent dynamics. We study how learning dynamics and plateaus depend on feature structure, learning rate, discount factor, and reward function. We then analyze how strategies like learning rate annealing and reward shaping can favorably alter learning dynamics and plateaus. To conclude, our work introduces new tools to open a new direction towards developing a theory of learning dynamics in reinforcement learning.

----

## [0] REASONER: An Explainable Recommendation Dataset with Comprehensive Labeling Ground Truths

**Authors**: *Xu Chen, Jingsen Zhang, Lei Wang, Quanyu Dai, Zhenhua Dong, Ruiming Tang, Rui Zhang, Li Chen, Xin Zhao, Ji-Rong Wen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ebf43d20e5933ab6d98225bbb908ade-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ebf43d20e5933ab6d98225bbb908ade-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Explainable recommendation has attracted much attention from the industry and academic communities. It has shown great potential to improve the recommendation persuasiveness, informativeness and user satisfaction. In the past few years, while a lot of promising explainable recommender models have been proposed, the datasets used to evaluate them still suffer from several limitations, for example, the explanation ground truths are not labeled by the real users, the explanations are mostly single-modal and around only one aspect. To bridge these gaps, in this paper, we build a new explainable recommendation dataset, which, to our knowledge, is the first contribution that provides a large amount of real user labeled multi-modal and multi-aspect explaination ground truths. In specific, we firstly develop a video recommendation platform, where a series of questions around the recommendation explainability are carefully designed. Then, we recruit about 3000 high-quality labelers with different backgrounds to use the system, and collect their behaviors and feedback to our questions. In this paper, we detail the construction process of our dataset and also provide extensive analysis on its characteristics. In addition, we develop a library, where ten well-known explainable recommender models are implemented in a unified framework. Based on this library, we build several benchmarks for different explainable recommendation tasks. At last, we present many new opportunities brought by our dataset, which are expected to promote the field of explainable recommendation. Our dataset, library and the related documents have been released at https://reasoner2023.github.io/.

----

## [0] Fast Model DeBias with Machine Unlearning

**Authors**: *Ruizhe Chen, Jianfei Yang, Huimin Xiong, Jianhong Bai, Tianxiang Hu, Jin Hao, Yang Feng, Joey Tianyi Zhou, Jian Wu, Zuozhu Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ecc80084c96cc25b11b0ab995c25f47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ecc80084c96cc25b11b0ab995c25f47-Abstract-Conference.html)

**Abstract**:

Recent discoveries have revealed that deep neural networks might behave in a biased manner in many real-world scenarios. For instance, deep networks trained on a large-scale face recognition dataset CelebA tend to predict blonde hair for females and black hair for males. Such biases not only jeopardize the robustness of models but also perpetuate and amplify social biases, which is especially concerning for automated decision-making processes in healthcare, recruitment, etc., as they could exacerbate unfair economic and social inequalities among different groups. Existing debiasing methods suffer from high costs in bias labeling or model re-training, while also exhibiting a deficiency in terms of elucidating the origins of biases within the model. To this respect, we propose a fast model debiasing method (FMD) which offers an efficient approach to identify, evaluate and remove biases inherent in trained models. The FMD identifies biased attributes through an explicit counterfactual concept and quantifies the influence of data samples with influence functions. Moreover, we design a machine unlearning-based strategy to efficiently and effectively remove the bias in a trained model with a small counterfactual dataset. Experiments on the Colored MNIST, CelebA, and Adult Income datasets demonstrate that our method achieves superior or competing classification accuracies compared with state-of-the-art retraining-based methods while attaining significantly fewer biases and requiring much less debiasing cost. Notably, our method requires only a small external dataset and updating a minimal amount of model parameters, without the requirement of access to training data that may be too large or unavailable in practice.

----

## [0] Coherent Soft Imitation Learning

**Authors**: *Joe Watson, Sandy H. Huang, Nicolas Heess*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f0435cffef91068ced08d7c7d8e643e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f0435cffef91068ced08d7c7d8e643e-Abstract-Conference.html)

**Abstract**:

Imitation learning methods seek to learn from an expert either through behavioral cloning (BC) for the policy or inverse reinforcement learning (IRL) for the reward.Such methods enable agents to learn complex tasks from humans that are difficult to capture with hand-designed reward functions.Choosing between BC or IRL for imitation depends on the quality and state-action coverage of the demonstrations, as well as additional access to the Markov decision process. Hybrid strategies that combine BC and IRL are rare, as initial policy optimization against inaccurate rewards diminishes the benefit of pretraining the policy with BC.Our work derives an imitation method that captures the strengths of both BC and IRL.In the entropy-regularized (`soft') reinforcement learning setting, we show that the behavioral-cloned policy can be used as both a shaped reward and a critic hypothesis space by inverting the regularized policy update. This coherency facilitates fine-tuning cloned policies using the reward estimate and additional interactions with the environment.This approach conveniently achieves imitation learning through initial behavioral cloning and subsequent refinement via RL with online or offline data sources.The simplicity of the approach enables graceful scaling to high-dimensional and vision-based tasks, with stable learning and minimal hyperparameter tuning, in contrast to adversarial approaches.For the open-source implementation and simulation results, see https://joemwatson.github.io/csil/.

----

## [0] Exact Generalization Guarantees for (Regularized) Wasserstein Distributionally Robust Models

**Authors**: *Waïss Azizian, Franck Iutzeler, Jérôme Malick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f060912eacace9ce61ef339205ec54c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f060912eacace9ce61ef339205ec54c-Abstract-Conference.html)

**Abstract**:

Wasserstein distributionally robust estimators have emerged as powerful models for prediction and decision-making under uncertainty. These estimators provide attractive generalization guarantees: the robust objective obtained from the training distribution is an exact upper bound on the true risk with high probability. However, existing guarantees either suffer from the curse of dimensionality, are restricted to specific settings, or lead to spurious error terms. In this paper, we show that these generalization guarantees actually hold on general classes of models, do not suffer from the curse of dimensionality, and can even cover distribution shifts at testing. We also prove that these results carry over to the newly-introduced regularized versions of Wasserstein distributionally robust problems.

----

## [0] Supply-Side Equilibria in Recommender Systems

**Authors**: *Meena Jagadeesan, Nikhil Garg, Jacob Steinhardt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f1486343c2c942a617e4f5bb0cc64c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f1486343c2c942a617e4f5bb0cc64c8-Abstract-Conference.html)

**Abstract**:

Algorithmic recommender systems such as Spotify and Netflix affect not only consumer behavior but also producer incentives. Producers seek to create content that will be shown by the recommendation algorithm, which can impact both the diversity and quality of their content. In this work, we investigate the resulting supply-side equilibria in personalized content recommender systems. We model the decisions of producers as choosing multi-dimensional content vectors and users as having heterogenous preferences, which contrasts with classical low-dimensional models. Multi-dimensionality and heterogeneity creates the potential for specialization, where different producers create different types of content at equilibrium. Using a duality argument, we derive necessary and sufficient conditions for whether specialization occurs. Then, we characterize the distribution of content at equilibrium in concrete settings with two populations of users. Lastly, we show that specialization can enable producers to achieve positive profit at equilibrium, which means that specialization can reduce the competitiveness of the marketplace. At a conceptual level, our analysis of supply-side competition takes a step towards elucidating how personalized recommendations shape the marketplace of digital goods.

----

## [0] Human-Aligned Calibration for AI-Assisted Decision Making

**Authors**: *Nina Corvelo Benz, Manuel Rodriguez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f1d1196426ba84f47d115cac3dcb9d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f1d1196426ba84f47d115cac3dcb9d8-Abstract-Conference.html)

**Abstract**:

Whenever a binary classifier is used to provide decision support, it typically provides both a label prediction and a confidence value. Then, the decision maker is supposed to use the confidence value to calibrate how much to trust the prediction. In this context, it has been often argued that the confidence value should correspond to a well calibrated estimate of the probability that the predicted label matches the ground truth label. However, multiple lines of empirical evidence suggest that decision makers have difficulties at developing a good sense on when to trust a prediction using these confidence values. In this paper, our goal is first to understand why and then investigate how to construct more useful confidence values. We first argue that, for a broad class of utility functions, there exists data distributions for which a rational decision maker is, in general, unlikely to discover the optimal decision policy using the above confidence values—an optimal decision maker would need to sometimes place more (less) trust on predictions with lower (higher) confidence values. However, we then show that, if the confidence values satisfy a natural alignment property with respect to the decision maker’s confidence on her own predictions, there always exists an optimal decision policy under which the level of trust the decision maker would need to place on predictions is monotone on the confidence values, facilitating its discoverability. Further, we show that multicalibration with respect to the decision maker’s confidence on her own prediction is a sufficient condition for alignment. Experiments on a real AI-assisted decision making scenario where a classifier provides decision support to human decision makers validate our theoretical results and suggest that alignment may lead to better decisions.

----

## [0] Transformer as a hippocampal memory consolidation model based on NMDAR-inspired nonlinearity

**Authors**: *Dong Kyum Kim, Jea Kwon, Meeyoung Cha, Chul Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f1eb4c897e63870eee9a0a0f7a10332-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f1eb4c897e63870eee9a0a0f7a10332-Abstract-Conference.html)

**Abstract**:

The hippocampus plays a critical role in learning, memory, and spatial representation, processes that depend on the NMDA receptor (NMDAR). Inspired by recent findings that compare deep learning models to the hippocampus, we propose a new nonlinear activation function that mimics NMDAR dynamics. NMDAR-like nonlinearity shifts short-term working memory into long-term reference memory in transformers, thus enhancing a process that is similar to memory consolidation in the mammalian brain. We design a navigation task assessing these two memory functions and show that manipulating the activation function (i.e., mimicking the Mg$^{2+}$-gating of NMDAR) disrupts long-term memory processes. Our experiments suggest that place cell-like functions and reference memory reside in the feed-forward network layer of transformers and that nonlinearity drives these processes. We discuss the role of NMDAR-like nonlinearity in establishing this striking resemblance between transformer architecture and hippocampal spatial representation.

----

## [0] Gaussian Differential Privacy on Riemannian Manifolds

**Authors**: *Yangdi Jiang, Xiaotian Chang, Yi Liu, Lei Ding, Linglong Kong, Bei Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html)

**Abstract**:

We develop an advanced approach for extending Gaussian Differential Privacy (GDP) to general Riemannian manifolds. The concept of GDP stands out as a prominent privacy definition that strongly warrants extension to manifold settings, due to its central limit properties. By harnessing the power of the renowned Bishop-Gromov theorem in geometric analysis, we propose a Riemannian Gaussian distribution that integrates the Riemannian distance, allowing us to achieve GDP in Riemannian manifolds with bounded Ricci curvature. To the best of our knowledge, this work marks the first instance of extending the GDP framework to accommodate general Riemannian manifolds, encompassing curved spaces, and circumventing the reliance on tangent space summaries. We provide a simple algorithm to evaluate the privacy budget $\mu$ on any one-dimensional manifold and introduce a versatile Markov Chain Monte Carlo (MCMC)-based algorithm to calculate $\mu$ on any Riemannian manifold with constant curvature. Through simulations on one of the most prevalent manifolds in statistics, the unit sphere $S^d$, we demonstrate the superior utility of our Riemannian Gaussian mechanism in comparison to the previously proposed Riemannian Laplace mechanism for implementing GDP.

----

## [0] Agnostically Learning Single-Index Models using Omnipredictors

**Authors**: *Aravind Gollakota, Parikshit Gopalan, Adam R. Klivans, Konstantinos Stavropoulos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f46ef5725a8eca24f7f24a17955ad1a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f46ef5725a8eca24f7f24a17955ad1a-Abstract-Conference.html)

**Abstract**:

We give the first result for agnostically learning Single-Index Models (SIMs) with arbitrary monotone and Lipschitz activations. All prior work either held only in the realizable setting or required the activation to be known. Moreover, we only require the marginal to have bounded second moments, whereas all prior work required stronger distributional assumptions (such as anticoncentration or boundedness). Our algorithm is based on recent work by Gopalan et al. [2023] on Omniprediction using predictors satisfying calibrated multiaccuracy. Our analysis is simple and relies on the relationship between Bregman divergences (or matching losses) and $\ell_p$ distances. We also provide new guarantees for standard algorithms like GLMtron and logistic regression in the agnostic setting.

----

## [0] On skip connections and normalisation layers in deep optimisation

**Authors**: *Lachlan E. MacDonald, Jack Valmadre, Hemanth Saratchandran, Simon Lucey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f4d6f8e0f4f543db12260696b2a3551-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f4d6f8e0f4f543db12260696b2a3551-Abstract-Conference.html)

**Abstract**:

We introduce a general theoretical framework, designed for the study of gradient optimisation of deep neural networks, that encompasses ubiquitous architecture choices including batch normalisation, weight normalisation and skip connections.  Our framework determines the curvature and regularity properties of multilayer loss landscapes in terms of their constituent layers, thereby elucidating the roles played by normalisation layers and skip connections in globalising these properties. We then demonstrate the utility of this framework in two respects. First, we give the only proof of which we are aware that a class of deep neural networks can be trained using gradient descent to global optima even when such optima only exist at infinity, as is the case for the cross-entropy cost.  Second, we identify a novel causal mechanism by which skip connections accelerate training, which we verify predictively with ResNets on MNIST, CIFAR10, CIFAR100 and ImageNet.

----

## [0] Efficient Low-rank Backpropagation for Vision Transformer Adaptation

**Authors**: *Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, Radu Marculescu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f75a57e9c71e8369da0150ea769d5a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f75a57e9c71e8369da0150ea769d5a2-Abstract-Conference.html)

**Abstract**:

The increasing scale of vision transformers (ViT) has made the efficient fine-tuning of these large models for specific needs a significant challenge in various applications. This issue originates from the computationally demanding matrix multiplications required during the backpropagation process through linear layers in ViT.In this paper, we tackle this problem by proposing a new Low-rank BackPropagation via Walsh-Hadamard Transformation (LBP-WHT) method. Intuitively, LBP-WHT projects the gradient into a low-rank space and carries out backpropagation. This approach substantially reduces the computation needed for adapting ViT, as matrix multiplication in the low-rank space is far less resource-intensive. We conduct extensive experiments with different models (ViT, hybrid convolution-ViT model) on multiple datasets to demonstrate the effectiveness of our method. For instance, when adapting an EfficientFormer-L1 model on CIFAR100, our LBP-WHT achieves 10.4\% higher accuracy than the state-of-the-art baseline, while requiring 9 MFLOPs less computation.As the first work to accelerate ViT adaptation with low-rank backpropagation, our LBP-WHT method is complementary to many prior efforts and can be combined with them for better performance.

----

## [0] AdaVAE: Bayesian Structural Adaptation for Variational Autoencoders

**Authors**: *Paribesh Regmi, Rui Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f76a3a0f44263b5e56fec69ee1220f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f76a3a0f44263b5e56fec69ee1220f9-Abstract-Conference.html)

**Abstract**:

The neural network structures of generative models and their corresponding inference models paired in variational autoencoders (VAEs) play a critical role in the models' generative performance. However, powerful VAE network structures are hand-crafted and fixed prior to training, resulting in a one-size-fits-all approach that requires heavy computation to tune for given data. Moreover, existing VAE regularization methods largely overlook the importance of network structures and fail to prevent overfitting in deep VAE models with cascades of hidden layers. To address these issues, we propose a Bayesian inference framework that automatically adapts VAE network structures to data and prevent overfitting as they grow deeper. We model the number of hidden layers with a beta process to infer the most plausible encoding/decoding network depths warranted by data and perform layer-wise dropout regularization with a conjugate Bernoulli process. We develop a scalable estimator that performs joint inference on both VAE network structures and latent variables. Our experiments show that the inference framework effectively prevents overfitting in both shallow and deep VAE models, yielding state-of-the-art performance. We demonstrate that our framework is compatible with different types of VAE backbone networks and can be applied to various VAE variants, further improving their performance.

----

## [0] Safety Verification of Decision-Tree Policies in Continuous Time

**Authors**: *Christian Schilling, Anna Lukina, Emir Demirovic, Kim Guldstrand Larsen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html)

**Abstract**:

Decision trees have gained popularity as interpretable surrogate models for learning-based control policies. However, providing safety guarantees for systems controlled by decision trees is an open challenge. We show that the problem is undecidable even for systems with the simplest dynamics, and PSPACE-complete for finite-horizon properties. The latter can be verified for discrete-time systems via bounded model checking. However, for continuous-time systems, such an approach requires discretization, thereby weakening the guarantees for the original system. This paper presents the first algorithm to directly verify decision-tree controlled system in continuous time. The key aspect of our method is exploiting the decision-tree structure to propagate a set-based approximation through the decision nodes. We demonstrate the effectiveness of our approach by verifying safety of several decision trees distilled to imitate neural-network policies for nonlinear systems.

----

## [0] Quasi-Monte Carlo Graph Random Features

**Authors**: *Isaac Reid, Adrian Weller, Krzysztof Marcin Choromanski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f9b3ee2bcea04b327c09d7e3145bd1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f9b3ee2bcea04b327c09d7e3145bd1e-Abstract-Conference.html)

**Abstract**:

We present a novel mechanism to improve the accuracy of the recently-introduced class of graph random features (GRFs). Our method induces negative correlations between the lengths of the algorithm's random walks by imposing antithetic termination: a procedure to sample more diverse random walks which may be of independent interest. It has a trivial drop-in implementation. We derive strong theoretical guarantees on the properties of these quasi-Monte Carlo GRFs (q-GRFs), proving that they yield lower-variance estimators of the $2$-regularised Laplacian kernel under mild conditions. Remarkably, our results hold for any graph topology. We demonstrate empirical accuracy improvements on a variety of tasks including a new practical application: time-efficient approximation of the graph diffusion process. To our knowledge, q-GRFs constitute the first rigorously studied quasi-Monte Carlo scheme for kernels defined on combinatorial objects, inviting new research on correlations between graph random walks.

----

## [0] Functional Renyi Differential Privacy for Generative Modeling

**Authors**: *Dihong Jiang, Sun Sun, Yaoliang Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f9ee101e35b890d9eae79ee27bcd69a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f9ee101e35b890d9eae79ee27bcd69a-Abstract-Conference.html)

**Abstract**:

Differential privacy (DP) has emerged as a rigorous notion to quantify data privacy.  Subsequently, Renyi differential privacy (RDP) becomes an alternative to the ordinary DP notion in both theoretical and empirical studies, for its convenient compositional rules and flexibility. However, most mechanisms with DP (RDP) guarantees are essentially based on randomizing a fixed, finite-dimensional vector output. In this work, following Hall et al. (2013) we further extend RDP to functional outputs, where the output space can be infinite-dimensional, and develop all necessary tools, *e.g.*, (subsampled) Gaussian mechanism, composition, and post-processing rules, to facilitate its practical adoption. As an illustration, we apply functional RDP (f-RDP) to functions in the reproducing kernel Hilbert space (RKHS) to develop a differentially private generative model (DPGM), where training can be interpreted as iteratively releasing loss functions (in an RKHS) with DP (RDP) guarantees. Empirically, the new training paradigm achieves a  significant improvement in privacy-utility trade-off compared to existing alternatives, especially when $\epsilon=0.2$. Our code is available at https://github.com/dihjiang/DP-kernel.

----

## [0] FedL2P: Federated Learning to Personalize

**Authors**: *Royson Lee, Minyoung Kim, Da Li, Xinchi Qiu, Timothy M. Hospedales, Ferenc Huszar, Nicholas D. Lane*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2fb57276bfbaf1b832d7bfcba36bb41c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2fb57276bfbaf1b832d7bfcba36bb41c-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) research has made progress in developing algorithms for distributed learning of global models, as well as algorithms for local personalization of those common models to the specifics of each client’s local data distribution. However, different FL problems may require different personalization strategies, and it may not even be possible to define an effective one-size-fits-all personalization strategy for all clients: Depending on how similar each client’s optimal predictor is to that of the global model, different personalization strategies may be preferred. In this paper, we consider the federated meta-learning problem of learning personalization strategies. Specifically, we consider meta-nets that induce the batch-norm and learning rate parameters for each client given local data statistics. By learning these meta-nets through FL, we allow the whole FL network to collaborate in learning a customized personalization strategy for each client. Empirical results show that this framework improves on a range of standard hand-crafted personalization baselines in both label and feature shift situations.

----

## [0] Learning Reliable Logical Rules with SATNet

**Authors**: *Zhaoyu Li, Jinpei Guo, Yuhe Jiang, Xujie Si*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ff46d83d1dcc063e075058b29d55efe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ff46d83d1dcc063e075058b29d55efe-Abstract-Conference.html)

**Abstract**:

Bridging logical reasoning and deep learning is crucial for advanced AI systems. In this work, we present a new framework that addresses this goal by generating interpretable and verifiable logical rules through differentiable learning, without relying on pre-specified logical structures. Our approach builds upon SATNet, a differentiable MaxSAT solver that learns the underlying rules from input-output examples. Despite its efficacy, the learned weights in SATNet are not straightforwardly interpretable, failing to produce human-readable rules. To address this, we propose a novel specification method called ``maximum equality'', which enables the interchangeability between the learned weights of SATNet and a set of propositional logical rules in weighted MaxSAT form. With the decoded weighted MaxSAT formula, we further introduce several effective verification techniques to validate it against the ground truth rules. Experiments on stream transformations and Sudoku problems show that our decoded rules are highly reliable: using exact solvers on them could achieve 100% accuracy, whereas the original SATNet fails to give correct solutions in many cases. Furthermore, we formally verify that our decoded logical rules are functionally equivalent to the ground truth ones.

----

## [0] Demo2Code: From Summarizing Demonstrations to Synthesizing Code via Extended Chain-of-Thought

**Authors**: *Yuki Wang, Gonzalo Gonzalez-Pumariega, Yash Sharma, Sanjiban Choudhury*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30699996ff411d48903c9752b782a5c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30699996ff411d48903c9752b782a5c1-Abstract-Conference.html)

**Abstract**:

Language instructions and demonstrations are two natural ways for users to teach robots personalized tasks. Recent progress in Large Language Models (LLMs) has shown impressive performance in translating language instructions into code for robotic tasks. However, translating demonstrations into task code continues to be a challenge due to the length and complexity of both demonstrations and code, making learning a direct mapping intractable. This paper presents Demo2Code, a novel framework that generates robot task code from demonstrations via an extended chain-of-thought and defines a common latent specification to connect the two. Our framework employs a robust two-stage process: (1) a recursive summarization technique that condenses demonstrations into concise specifications, and (2) a code synthesis approach that expands each function recursively from the generated specifications. We conduct extensive evaluation on various robot task benchmarks, including a novel game benchmark Robotouille, designed to simulate diverse cooking tasks in a kitchen environment.

----

## [0] Easy Learning from Label Proportions

**Authors**: *Róbert Busa-Fekete, Heejin Choi, Travis Dick, Claudio Gentile, Andrés Muñoz Medina*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3085fd61063840fdb2e6eafac58589f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3085fd61063840fdb2e6eafac58589f8-Abstract-Conference.html)

**Abstract**:

We consider the problem of Learning from Label Proportions (LLP), a weakly supervised classification setup where instances are grouped into i.i.d. “bags”, and only the frequency of class labels at each bag is available.  Albeit, the objective of the learner is to achieve low task loss at an individual instance level.  Here we propose EASYLLP, a flexible and simple-to-implement debiasing approach based on aggregate labels, which operates on arbitrary loss functions. Our technique allows us to accurately estimate the expected loss of an arbitrary model at an individual level.  We elucidate the differences between our method and standard methods based on label proportion matching, in terms of applicability and optimality conditions. We showcase the flexibility of our approach compared to alternatives by applying our method to popular learning frameworks, like Empirical Risk Minimization (ERM) and Stochastic Gradient Descent (SGD) with provable guarantees on instance level performance.  Finally, we validate our theoretical results on multiple datasets, empirically illustrating the conditions under which our algorithm is expected to perform better or worse than previous LLP approaches

----

## [0] Jigsaw: Learning to Assemble Multiple Fractured Objects

**Authors**: *Jiaxin Lu, Yifan Sun, Qixing Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30ae2af8612ac74357363e8ae877d80c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30ae2af8612ac74357363e8ae877d80c-Abstract-Conference.html)

**Abstract**:

Automated assembly of 3D fractures is essential in orthopedics, archaeology, and our daily life. This paper presents Jigsaw, a novel framework for assembling physically broken 3D objects from multiple pieces. Our approach leverages hierarchical features of global and local geometry to match and align the fracture surfaces. Our framework consists of four components: (1) front-end point feature extractor with attention layers, (2) surface segmentation to separate fracture and original parts, (3) multi-parts matching to find correspondences among fracture surface points, and (4) robust global alignment to recover the global poses of the pieces. We show how to jointly learn segmentation and matching and seamlessly integrate feature matching and rigidity constraints. We evaluate Jigsaw on the Breaking Bad dataset and achieve superior performance compared to state-of-the-art methods. Our method also generalizes well to diverse fracture modes, objects, and unseen instances. To the best of our knowledge, this is the first learning-based method designed specifically for 3D fracture assembly over multiple pieces. Our code is available at https://jiaxin-lu.github.io/Jigsaw/.

----

## [0] Persuading Farsighted Receivers in MDPs: the Power of Honesty

**Authors**: *Martino Bernasconi, Matteo Castiglioni, Alberto Marchesi, Mirco Mutti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30b28eb87fe7a6c4af8520293317d4c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30b28eb87fe7a6c4af8520293317d4c6-Abstract-Conference.html)

**Abstract**:

Bayesian persuasion studies the problem faced by an informed sender who strategically discloses information to influence the behavior of an uninformed receiver. Recently, a growing attention has been devoted to settings where the sender and the receiver interact sequentially, in which the receiver's decision-making problem is usually modeled as a Markov decision process (MDP). However, the literature focuses on computing optimal information-revelation policies (a.k.a. signaling schemes) under the restrictive assumption that the receiver acts myopically, selecting actions to maximize the one-step utility and disregarding future rewards. This is justified by the fact that, when the receiver is farsighted and thus considers future rewards, finding an optimal Markovian signaling scheme is NP-hard. In this paper, we show that Markovian signaling schemes do not constitute the "right" class of policies. Indeed, differently from most of the MDPs settings, we show that Markovian signaling schemes are not optimal, and general history-dependent signaling schemes should be considered. Moreover, we also show that history-dependent signaling schemes circumvent the negative complexity results affecting Markovian signaling schemes. Formally, we design an algorithm that computes an optimal and $\epsilon$-persuasive history-dependent signaling scheme in time polynomial in ${1}/{\epsilon}$ and in the instance size. The crucial challenge is that general history-dependent signaling schemes cannot be represented in polynomial space. Nevertheless, we introduce a convenient subclass of history-dependent signaling schemes, called promise-form, which are as powerful as general history-dependent ones and efficiently representable. Intuitively, promise-form signaling schemes compactly encode histories in the form of honest promises on future receiver's rewards.

----

## [0] When are ensembles really effective?

**Authors**: *Ryan Theisen, Hyunsuk Kim, Yaoqing Yang, Liam Hodgkinson, Michael W. Mahoney*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30b6fa308e62ed52180c31ae3ba6bb0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30b6fa308e62ed52180c31ae3ba6bb0a-Abstract-Conference.html)

**Abstract**:

Ensembling has a long history in statistical data analysis, with many impactful applications. However, in many modern machine learning settings, the benefits of ensembling are less ubiquitous and less obvious. We study, both theoretically and empirically, the fundamental question of when ensembling yields significant performance improvements in classification tasks. Theoretically, we prove new results relating the \emph{ensemble improvement rate} (a measure of how much ensembling decreases the error rate versus a single model, on a relative scale) to the \emph{disagreement-error ratio}. We show that ensembling improves performance significantly whenever the disagreement rate is large relative to the average error rate; and that, conversely, one classifier is often enough whenever the disagreement rate is low relative to the average error rate. On the way to proving these results, we derive, under a mild condition called \emph{competence}, improved upper and lower bounds on the average test error rate of the majority vote classifier.To complement this theory, we study ensembling empirically in a variety of settings, verifying the predictions made by our theory, and identifying practical scenarios where ensembling does and does not result in large performance improvements. Perhaps most notably, we demonstrate a distinct difference in behavior between interpolating models (popular in current practice) and non-interpolating models (such as tree-based methods, where ensembling is popular), demonstrating that ensembling helps considerably more in the latter case than in the former.

----

## [0] A Unified Approach to Domain Incremental Learning with Memory: Theory and Algorithm

**Authors**: *Haizhou Shi, Hao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30d046e94d7b8037d6ef27c4357a8dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30d046e94d7b8037d6ef27c4357a8dd4-Abstract-Conference.html)

**Abstract**:

Domain incremental learning aims to adapt to a sequence of domains with access to only a small subset of data (i.e., memory) from previous domains. Various methods have been proposed for this problem, but it is still unclear how they are related and when practitioners should choose one method over another. In response, we propose a unified framework, dubbed Unified Domain Incremental Learning (UDIL), for domain incremental learning with memory. Our UDIL unifies various existing methods, and our theoretical analysis shows that UDIL always achieves a tighter generalization error bound compared to these methods. The key insight is that different existing methods correspond to our bound with different fixed coefficients; based on insights from this unification, our UDIL allows adaptive coefficients during training, thereby always achieving the tightest bound. Empirical results show that our UDIL outperforms the state-of-the-art domain incremental learning methods on both synthetic and real-world datasets. Code will be available at https://github.com/Wang-ML-Lab/unified-continual-learning.

----

## [0] Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration

**Authors**: *Qi-Wei Wang, Da-Wei Zhou, Yi-Kai Zhang, De-Chuan Zhan, Han-Jia Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30dfe47a3ccbee68cffa0c19ccb1bc00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30dfe47a3ccbee68cffa0c19ccb1bc00-Abstract-Conference.html)

**Abstract**:

Real-world scenarios are usually accompanied by continuously appearing classes with scare labeled samples, which require the machine learning model to incrementally learn new classes and maintain the knowledge of base classes. In this Few-Shot Class-Incremental Learning (FSCIL) scenario, existing methods either introduce extra learnable components or rely on a frozen feature extractor to mitigate catastrophic forgetting and overfitting problems.  However, we find a tendency for existing methods to misclassify the samples of new classes into base classes, which leads to the poor performance of new classes. In other words, the strong discriminability of base classes distracts the classification of new classes. To figure out this intriguing phenomenon, we observe that although the feature extractor is only trained on base classes, it can surprisingly represent the semantic similarity between the base and unseen new classes. Building upon these analyses, we propose a simple yet effective Training-frEE calibratioN (TEEN) strategy to enhance the discriminability of new classes by fusing the new prototypes (i.e., mean features of a class) with weighted base prototypes. In addition to standard benchmarks in FSCIL, TEEN demonstrates remarkable performance and consistent improvements over baseline methods in the few-shot learning scenario. Code is available at: https://github.com/wangkiw/TEEN

----

## [0] RADAR: Robust AI-Text Detection via Adversarial Learning

**Authors**: *Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30e15e5941ae0cdab7ef58cc8d59a4ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30e15e5941ae0cdab7ef58cc8d59a4ca-Abstract-Conference.html)

**Abstract**:

Recent advances in large language models (LLMs) and the intensifying popularity of ChatGPT-like applications have blurred the boundary of high-quality text generation between humans and machines. However, in addition to the anticipated revolutionary changes to our technology and society, the difficulty of distinguishing LLM-generated texts (AI-text) from human-generated texts poses new challenges of misuse and fairness, such as fake content generation, plagiarism, and false accusations of innocent writers. While existing works show that current AI-text detectors are not robust to LLM-based paraphrasing, this paper aims to bridge this gap by proposing a new framework called RADAR, which jointly trains a $\underline{r}$obust  $\underline{A}$I-text  $\underline{d}$etector via  $\underline{a}$dversarial lea$\underline{r}$ning. RADAR is based on adversarial training of a paraphraser and a detector. The paraphraser's goal is to generate realistic content to evade AI-text detection.RADAR uses the feedback from the detector to update the paraphraser, and vice versa.Evaluated with 8 different LLMs (Pythia, Dolly 2.0, Palmyra, Camel, GPT-J, Dolly 1.0, LLaMA, and Vicuna) across 4 datasets, experimental results show that RADAR significantly outperforms existing AI-text detection methods, especially when paraphrasing is in place. We also identify the strong transferability of RADAR from instruction-tuned LLMs to other LLMs, and evaluate the improved capability of RADAR via GPT-3.5-Turbo.

----

## [0] Alleviating the Semantic Gap for Generalized fMRI-to-Image Reconstruction

**Authors**: *Tao Fang, Qian Zheng, Gang Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3106c718fe84b91fc301fe2f5b738448-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3106c718fe84b91fc301fe2f5b738448-Abstract-Conference.html)

**Abstract**:

Although existing fMRI-to-image reconstruction methods could predict high-quality images, they do not explicitly consider the semantic gap between training and testing data, resulting in reconstruction with unstable and uncertain semantics. This paper addresses the problem of generalized fMRI-to-image reconstruction by explicitly alleviates the semantic gap. Specifically, we leverage the pre-trained CLIP model to map the training data to a compact feature representation, which essentially extends the sparse semantics of training data to dense ones, thus alleviating the semantic gap of the instances nearby known concepts (i.e., inside the training super-classes). Inspired by the robust low-level representation in fMRI data, which could help alleviate the semantic gap for instances that far from the known concepts (i.e., outside the training super-classes), we leverage structural information as a general cue to guide image reconstruction. Further, we quantify the semantic uncertainty based on probability density estimation and achieve Generalized fMRI-to-image reconstruction by adaptively integrating Expanded Semantics and Structural information (GESS) within a diffusion process. Experimental results demonstrate that the proposed GESS model outperforms state-of-the-art methods, and we propose a generalized scenario split strategy to evaluate the advantage of GESS in closing the semantic gap.

----

## [0] Softmax Output Approximation for Activation Memory-Efficient Training of Attention-based Networks

**Authors**: *Changhyeon Lee, Seulki Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/311257424b6d80e930fc93b224f0a63e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/311257424b6d80e930fc93b224f0a63e-Abstract-Conference.html)

**Abstract**:

In this paper, we propose to approximate the softmax output, which is the key product of the attention mechanism, to reduce its activation memory usage when training attention-based networks (aka Transformers). During the forward pass of the network, the proposed softmax output approximation method stores only a small fraction of the entire softmax output required for back-propagation and evicts the rest of the softmax output from memory. Then, during the backward pass, the evicted softmax activation output is approximated to compose the gradient to perform back-propagation for model training. Considering most attention-based models heavily rely on the softmax-based attention module that usually takes one of the biggest portions of the network, approximating the softmax activation output can be a simple yet effective way to decrease the training memory requirement of many attention-based networks. The experiment with various attention-based models and relevant tasks, i.e., machine translation, text classification, and sentiment analysis, shows that it curtails the activation memory usage of the softmax-based attention module by up to 84% (6.2Ã— less memory) in model training while achieving comparable or better performance, e.g., up to 5.4% higher classification accuracy.

----

## [0] Data Portraits: Recording Foundation Model Training Data

**Authors**: *Marc Marone, Benjamin Van Durme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3112ee706d21d734c15532c1239773e1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3112ee706d21d734c15532c1239773e1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Foundation models are trained on increasingly immense and opaque datasets. Even while these models are now key in AI system building, it can be  difficult to answer the straightforward question: has the model already encountered a given example during training? We therefore propose a widespread adoption of Data Portraits: artifacts that record training data and allow for downstream inspection. First we outline the properties of such an artifact and discuss how existing solutions can be used to increase transparency. We then propose and implement a solution based on data sketching, stressing fast and space efficient querying. Using our tools, we document a popular language modeling corpus (The Pile) and a recently released code modeling dataset (The Stack). We show that our solution enables answering questions about test set leakage and model plagiarism. Our tool is lightweight and fast, costing only 3% of the dataset size in overhead. We release a live interface of our tools at https://dataportraits.org/ and call on dataset and model creators to release Data Portraits as a complement to current documentation practices.

----

## [0] Memory Efficient Optimizers with 4-bit States

**Authors**: *Bingrui Li, Jianfei Chen, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3122aaa22b2fe83f9cead1a696f65ceb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3122aaa22b2fe83f9cead1a696f65ceb-Abstract-Conference.html)

**Abstract**:

Optimizer states are a major source of memory consumption for training neural networks, limiting the maximum trainable model within given memory budget. Compressing the optimizer states from 32-bit floating points to lower bitwidth is promising to reduce the training memory footprint, while the current lowest achievable bitwidth is 8-bit. In this work, we push optimizer states bitwidth down to 4-bit through a detailed empirical analysis of first and second moments. Specifically, we find that moments have complicated outlier patterns, that current block-wise quantization cannot accurately approximate. We use a smaller block size and propose to utilize both row-wise and column-wise information for better quantization. We further identify a zero point problem of quantizing the second moment, and solve this problem with a linear quantizer that excludes the zero point. Our 4-bit optimizers are evaluated on a wide variety of benchmarks including natural language understanding, machine translation, image classification, and instruction tuning. On all the tasks our optimizers can achieve comparable accuracy with their full-precision counterparts, while enjoying better memory efficiency.

----

## [0] Replicable Reinforcement Learning

**Authors**: *Eric Eaton, Marcel Hussing, Michael Kearns, Jessica Sorrell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/313829757739365201b5adb3a1cbd9bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/313829757739365201b5adb3a1cbd9bd-Abstract-Conference.html)

**Abstract**:

The replicability crisis in the social, behavioral, and data sciences has led to the formulation of algorithm frameworks for replicability --- i.e., a requirement that an algorithm produce identical outputs (with high probability) when run on two different samples from the same underlying distribution. While still in its infancy, provably replicable algorithms have been developed for many fundamental tasks in machine learning and statistics, including statistical query learning, the heavy hitters problem, and distribution testing. In this work we initiate the study of replicable reinforcement learning, providing a provably replicable algorithm for parallel value iteration, and a provably replicable version of R-Max in the episodic setting. These are the first formal replicability results for control problems, which present different challenges for replication than batch learning settings.

----

## [0] Make Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning

**Authors**: *Baohao Liao, Shaomu Tan, Christof Monz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3151e460c41ba67dc55412861184ef35-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3151e460c41ba67dc55412861184ef35-Abstract-Conference.html)

**Abstract**:

Parameter-efficient fine-tuning (PEFT) of pre-trained language models (PLMs) has emerged as a highly successful approach, with training only a small number of parameters without sacrificing performance and becoming the de-facto learning paradigm with the increasing size of PLMs. However, existing PEFT methods are not memory-efficient, because they still require caching most of the intermediate activations for the gradient calculation, akin to fine-tuning. One effective way to reduce the activation memory is to apply a reversible model, so the intermediate activations are not necessary to be cached and can be recomputed. Nevertheless, modifying a PLM to its reversible variant is not straightforward, since the reversible model has a distinct architecture from the currently released PLMs. In this paper, we first investigate what is a key factor for the success of existing PEFT methods, and realize that it's essential to preserve the PLM's starting point when initializing a PEFT method. With this finding, we propose memory-efficient fine-tuning (MEFT) that inserts adapters into a PLM, preserving the PLM's starting point and making it reversible without additional pre-training. We evaluate MEFT on the GLUE benchmark and five question-answering tasks with various backbones, BERT, RoBERTa, BART and OPT. MEFT significantly reduces the activation memory up to 84% of full fine-tuning with a negligible amount of trainable parameters. Moreover, MEFT achieves the same score on GLUE and a comparable score on the question-answering tasks as full fine-tuning. A similar finding is also observed for the image classification task.

----

## [0] Design from Policies: Conservative Test-Time Adaptation for Offline Policy Optimization

**Authors**: *Jinxin Liu, Hongyin Zhang, Zifeng Zhuang, Yachen Kang, Donglin Wang, Bin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31610e68fe41a62e460e044216a10766-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31610e68fe41a62e460e044216a10766-Abstract-Conference.html)

**Abstract**:

In this work, we decouple the iterative bi-level offline RL (value estimation and policy extraction) from the offline training phase, forming a non-iterative bi-level paradigm and avoiding the iterative error propagation over two levels. Specifically, this non-iterative paradigm allows us to conduct inner-level optimization (value estimation) in training, while performing outer-level optimization (policy extraction) in testing. Naturally, such a paradigm raises three core questions that are not fully answered by prior non-iterative offline RL counterparts like reward-conditioned policy: (q1) What information should we transfer from the inner-level to the outer-level? (q2) What should we pay attention to when exploiting the transferred information for safe/confident outer-level optimization? (q3) What are the benefits of concurrently conducting outer-level optimization during testing? Motivated by model-based optimization (MBO), we propose DROP (design from policies), which fully answers the above questions. Specifically, in the inner-level, DROP decomposes offline data into multiple subsets, and learns an MBO score model (a1). To keep safe exploitation to the score model in the outer-level, we explicitly learn a behavior embedding and introduce a conservative regularization (a2). During testing, we show that DROP permits deployment adaptation, enabling an adaptive inference across states (a3). Empirically, we evaluate DROP on various tasks, showing that DROP gains comparable or better performance compared to prior methods.

----

## [0] Lightweight Vision Transformer with Bidirectional Interaction

**Authors**: *Qihang Fan, Huaibo Huang, Xiaoqiang Zhou, Ran He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3170de57bc1899315b97712043d8bb22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3170de57bc1899315b97712043d8bb22-Abstract-Conference.html)

**Abstract**:

Recent advancements in vision backbones have significantly improved their performance by simultaneously modeling imagesâ€™ local and global contexts. However, the bidirectional interaction between these two contexts has not been well explored and exploited, which is important in the human visual system. This paper proposes a Fully Adaptive Self-Attention (FASA) mechanism for vision transformer to model the local and global information as well as the bidirectional interaction between them in context-aware ways. Specifically, FASA employs self-modulated convolutions to adaptively extract local representation while utilizing self-attention in down-sampled space to extract global representation. Subsequently, it conducts a bidirectional adaptation process between local and global representation to model their interaction. In addition, we introduce a fine-grained downsampling strategy to enhance the down-sampled self-attention mechanism for finer-grained global perception capability. Based on FASA, we develop a family of lightweight vision backbones, Fully Adaptive Transformer (FAT) family. Extensive experiments on multiple vision tasks demonstrate that FAT achieves impressive performance. Notably, FAT accomplishes a 77.6% accuracy on ImageNet-1K using only 4.5M parameters and 0.7G FLOPs, which surpasses the most advanced ConvNets and Transformers with similar model size and computational costs. Moreover, our model exhibits faster speed on modern GPU compared to other models.

----

## [0] Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications

**Authors**: *Xinyu Ma, Xu Chu, Yasha Wang, Yang Lin, Junfeng Zhao, Liantao Ma, Wenwu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3173c427cb4ed2d5eaab029c17f221ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3173c427cb4ed2d5eaab029c17f221ae-Abstract-Conference.html)

**Abstract**:

Graph data augmentation has shown superiority in enhancing generalizability and robustness of GNNs in graph-level classifications. However, existing methods primarily focus on the augmentation in the graph signal space and the graph structure space independently, neglecting the joint interaction between them. In this paper, we address this limitation by formulating the problem as an optimal transport problem that aims to find an optimal inter-graph node matching strategy considering the interactions between graph structures and signals. To solve this problem, we propose a novel graph mixup algorithm called FGWMixup, which seeks a "midpoint" of source graphs in the Fused Gromov-Wasserstein (FGW) metric space. To enhance the scalability of our method, we introduce a relaxed FGW solver that accelerates FGWMixup by improving the convergence rate from $\mathcal{O}(t^{-1})$ to $\mathcal{O}(t^{-2})$. Extensive experiments conducted on five datasets using both classic (MPNNs) and advanced (Graphormers) GNN backbones demonstrate that \mname\xspace effectively improves the generalizability and robustness of GNNs. Codes are available at https://github.com/ArthurLeoM/FGWMixup.

----

## [0] Modeling Dynamics over Meshes with Gauge Equivariant Nonlinear Message Passing

**Authors**: *Jung Yeon Park, Lawson L. S. Wong, Robin Walters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/317470b3fde29f3bb8d6dee563afffc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/317470b3fde29f3bb8d6dee563afffc4-Abstract-Conference.html)

**Abstract**:

Data over non-Euclidean manifolds, often discretized as surface meshes, naturally arise in computer graphics and biological and physical systems. In particular, solutions to partial differential equations (PDEs) over manifolds depend critically on the underlying geometry. While graph neural networks have been successfully applied to PDEs, they do not incorporate surface geometry and do not consider local gauge symmetries of the manifold. Alternatively, recent works on gauge equivariant convolutional and attentional architectures on meshes leverage the underlying geometry but underperform in modeling surface PDEs with complex nonlinear dynamics. To address these issues, we introduce a new gauge equivariant architecture using nonlinear message passing. Our novel architecture achieves higher performance than either convolutional or attentional networks on domains with highly complex and nonlinear dynamics. However, similar to the non-mesh case, design trade-offs favor convolutional, attentional, or message passing networks for different tasks; we investigate in which circumstances our message passing method provides the most benefit.

----

## [0] ASIF: Coupled Data Turns Unimodal Models to Multimodal without Training

**Authors**: *Antonio Norelli, Marco Fumero, Valentino Maiorca, Luca Moschella, Emanuele Rodolà, Francesco Locatello*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3186591903d9db31770ad131adb5ceb4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3186591903d9db31770ad131adb5ceb4-Abstract-Conference.html)

**Abstract**:

CLIP proved that aligning visual and language spaces is key to solving many vision tasks without explicit training, but required to train image and text encoders from scratch on a huge dataset. LiT improved this by only training the text encoder and using a pre-trained vision network. In this paper, we show that a common space can be created without any training at all, using single-domain encoders (trained with or without supervision) and a much smaller amount of image-text pairs. Furthermore, our model has unique properties. Most notably, deploying a new version with updated training samples can be done in a matter of seconds. Additionally, the representations in the common space are easily interpretable as every dimension corresponds to the similarity of the input to a unique entry in the multimodal dataset. Experiments on standard zero-shot visual benchmarks demonstrate the typical transfer ability of image-text models. Overall, our method represents a simple yet surprisingly strong baseline for foundation multi-modal models, raising important questions on their data efficiency and on the role of retrieval in machine learning.

----

## [0] A Metadata-Driven Approach to Understand Graph Neural Networks

**Authors**: *Ting Wei Li, Qiaozhu Mei, Jiaqi Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31994923f58ae5b2d661b300bd439107-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31994923f58ae5b2d661b300bd439107-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have achieved remarkable success in various applications, but their performance can be sensitive to specific data properties of the graph datasets they operate on. Current literature on understanding the limitations of GNNs has primarily employed a \emph{model-driven} approach that leverage heuristics and domain knowledge from network science or graph theory to model the GNN behaviors, which is time-consuming and highly subjective. In this work, we propose a \emph{metadata-driven} approach to analyze the sensitivity of GNNs to graph data properties, motivated by the increasing availability of graph learning benchmarks. We perform a multivariate sparse regression analysis on the metadata derived from benchmarking GNN performance across diverse datasets, yielding a set of salient data properties. To validate the effectiveness of our data-driven approach, we focus on one identified data property, the degree distribution, and investigate how this property influences GNN performance through theoretical analysis and controlled experiments. Our theoretical findings reveal that datasets with more balanced degree distribution exhibit better linear separability of node representations, thus leading to better GNN performance. We also conduct controlled experiments using synthetic datasets with varying degree distributions, and the results align well with our theoretical findings. Collectively, both the theoretical analysis and controlled experiments verify that the proposed metadata-driven approach is effective in identifying critical data properties for GNNs.

----

## [0] Multimodal Deep Learning Model Unveils Behavioral Dynamics of V1 Activity in Freely Moving Mice

**Authors**: *Aiwen Xu, Yuchen Hou, Cristopher Niell, Michael Beyeler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31a19921acd38cdf7a8c86ec032cef2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31a19921acd38cdf7a8c86ec032cef2d-Abstract-Conference.html)

**Abstract**:

Despite their immense success as a model of macaque visual cortex, deep convolutional neural networks (CNNs) have struggled to predict activity in visual cortex of the mouse, which is thought to be strongly dependent on the animalâ€™s behavioral state. Furthermore, most computational models focus on predicting neural responses to static images presented under head fixation, which are dramatically different from the dynamic, continuous visual stimuli that arise during movement in the real world. Consequently, it is still unknown how natural visual input and different behavioral variables may integrate over time to generate responses in primary visual cortex (V1). To address this, we introduce a multimodal recurrent neural network that integrates gaze-contingent visual input with behavioral and temporal dynamics to explain V1 activity in freely moving mice. We show that the model achieves state-of-the-art predictions of V1 activity during free exploration and demonstrate the importance of each component in an extensive ablation study. Analyzing our model using maximally activating stimuli and saliency maps, we reveal new insights into cortical function, including the prevalence of mixed selectivity for behavioral variables in mouse V1. In summary, our model offers a comprehensive deep-learning framework for exploring the computational principles underlying V1 neurons in freely-moving animals engaged in natural behavior.

----

## [0] Goal-conditioned Offline Planning from Curious Exploration

**Authors**: *Marco Bagatella, Georg Martius*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31ceb5aed43e2ec1b132e389cc1dcb56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31ceb5aed43e2ec1b132e389cc1dcb56-Abstract-Conference.html)

**Abstract**:

Curiosity has established itself as a powerful exploration strategy in deep reinforcement learning. Notably, leveraging expected future novelty as intrinsic motivation has been shown to efficiently generate exploratory trajectories, as well as a robust dynamics model. We consider the challenge of extracting goal-conditioned behavior from the products of such unsupervised exploration techniques, without any additional environment interaction. We find that conventional goal-conditioned reinforcement learning approaches for extracting a value function and policy fall short in this difficult offline setting. By analyzing the geometry of optimal goal-conditioned value functions, we relate this issue to a specific class of estimation artifacts in learned values. In order to mitigate their occurrence, we propose to combine model-based planning over learned value landscapes with a graph-based value aggregation scheme. We show how this combination can correct both local and global artifacts, obtaining significant improvements in zero-shot goal-reaching performance across diverse simulated environments.

----

## [0] Rethinking the Role of Token Retrieval in Multi-Vector Retrieval

**Authors**: *Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, Vincent Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31d997278ee9069d6721bc194174bb4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31d997278ee9069d6721bc194174bb4c-Abstract-Conference.html)

**Abstract**:

Multi-vector retrieval models such as ColBERT [Khattab et al., 2020] allow token-level interactions between queries and documents, and hence achieve state of the art on many information retrieval benchmarks. However, their non-linear scoring function cannot be scaled to millions of documents, necessitating a three-stage process for inference: retrieving initial candidates via token retrieval, accessing all token vectors, and scoring the initial candidate documents. The non-linear scoring function is applied over all token vectors of each candidate document, making the inference process complicated and slow. In this paper, we aim to simplify the multi-vector retrieval by rethinking the role of token retrieval. We present XTR, ConteXtualized Token Retriever, which introduces a simple, yet novel, objective function that encourages the model to retrieve the most important document tokens first. The improvement to token retrieval allows XTR to rank candidates only using the retrieved tokens rather than all tokens in the document, and enables a newly designed scoring stage that is two-to-three orders of magnitude cheaper than that of ColBERT. On the popular BEIR benchmark, XTR advances the state-of-the-art by 2.8 nDCG@10 without any distillation. Detailed analysis confirms our decision to revisit the token retrieval stage, as XTR demonstrates much better recall of the token retrieval stage compared to ColBERT.

----

## [0] Optimal Exploration for Model-Based RL in Nonlinear Systems

**Authors**: *Andrew Wagenmaker, Guanya Shi, Kevin G. Jamieson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31e018f43ab9c7065c058cc2c5848128-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31e018f43ab9c7065c058cc2c5848128-Abstract-Conference.html)

**Abstract**:

Learning to control unknown nonlinear dynamical systems is a fundamental problem in reinforcement learning and control theory. A commonly applied approach is to first explore the environment (exploration), learn an accurate model of it (system identification), and then compute an optimal controller with the minimum cost on this estimated system (policy optimization). While existing work has shown that it is possible to learn a uniformly good model of the system (Mania et al., 2020), in practice, if we aim to learn a good controller with a low cost on the actual system, certain system parameters may be significantly more critical than others, and we therefore ought to focus our exploration on learning such parameters.In this work, we consider the setting of nonlinear dynamical systems and seek to formally quantify, in such settings, (a) which parameters are most relevant to learning a good controller, and (b) how we can best explore so as to minimize uncertainty in such parameters. Inspired by recent work in linear systems (Wagenmaker et al., 2021), we show that minimizing the controller loss in nonlinear systems translates to estimating the system parameters in a particular, task-dependent metric. Motivated by this, we develop an algorithm able to efficiently explore the system to reduce uncertainty in this metric, and prove a lower bound showing that our approach learns a controller at a near-instance-optimal rate. Our algorithm relies on a general reduction from policy optimization to optimal experiment design in arbitrary systems, and may be of independent interest. We conclude with experiments demonstrating the effectiveness of our method in realistic nonlinear robotic systems.

----

## [0] ELDEN: Exploration via Local Dependencies

**Authors**: *Zizhao Wang, Jiaheng Hu, Peter Stone, Roberto Martín-Martín*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31ed129feae64a7e44a15b148c15558d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31ed129feae64a7e44a15b148c15558d-Abstract-Conference.html)

**Abstract**:

Tasks with large state space and sparse rewards present a longstanding challenge to reinforcement learning. In these tasks, an agent needs to explore the state space efficiently until it finds a reward. To deal with this problem, the community has proposed to augment the reward function with intrinsic reward, a bonus signal that encourages the agent to visit interesting states. In this work, we propose a new way of defining interesting states for environments with factored state spaces and complex chained dependencies, where an agent's actions may change the value of one entity that, in order, may affect the value of another entity. Our insight is that, in these environments, interesting states for exploration are states where the agent is uncertain whether (as opposed to how) entities such as the agent or objects have some influence on each other. We present ELDEN, Exploration via Local DepENdencies, a novel intrinsic reward that encourages the discovery of new interactions between entities. ELDEN utilizes a novel scheme --- the partial derivative of the learned dynamics to model the local dependencies between entities accurately and computationally efficiently. The uncertainty of the predicted dependencies is then used as an intrinsic reward to encourage exploration toward new interactions. We evaluate the performance of ELDEN on four different domains with complex dependencies, ranging from 2D grid worlds to 3D robotic tasks. In all domains, ELDEN correctly identifies local dependencies and learns successful policies, significantly outperforming previous state-of-the-art exploration methods.

----

## [0] Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness

**Authors**: *Gang Li, Wei Tong, Tianbao Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31f04c174a6af322e9417b7a9a91097a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31f04c174a6af322e9417b7a9a91097a-Abstract-Conference.html)

**Abstract**:

This paper seeks to address a gap in optimizing Average Precision (AP) while ensuring adversarial robustness, an area that has not been extensively explored to the best of our knowledge. AP maximization for deep learning has widespread applications, particularly when there is a significant imbalance between positive and negative examples. Although numerous studies have been conducted on adversarial training, they primarily focus on robustness concerning accuracy, ensuring that the average accuracy on adversarially perturbed examples is well maintained. However, this type of adversarial robustness is insufficient for many applications, as minor perturbations on a single example can significantly impact AP  while not greatly influencing the accuracy of the prediction system. To tackle this issue, we introduce a novel formulation that combines an AP surrogate loss with a regularization term representing adversarial ranking robustness, which maintains the consistency between ranking of clean data and that of perturbed data. We then devise an efficient stochastic optimization algorithm to optimize the resulting objective. Our empirical studies, which compare our method to current leading adversarial training baselines and other robust AP maximization strategies, demonstrate the effectiveness of the proposed approach. Notably, our methods outperform a state-of-the-art method (TRADES) by more than 4\% in terms of robust AP against PGD attacks while achieving 7\% higher AP on clean data simultaneously on CIFAR10 and CIFAR100.The code is available at: https://github.com/GangLii/Adversarial-AP

----

## [0] Act As You Wish: Fine-Grained Control of Motion Diffusion Model with Hierarchical Semantic Graphs

**Authors**: *Peng Jin, Yang Wu, Yanbo Fan, Zhongqian Sun, Wei Yang, Li Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31fc85f7461ce71eadf27fb7281973bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31fc85f7461ce71eadf27fb7281973bd-Abstract-Conference.html)

**Abstract**:

Most text-driven human motion generation methods employ sequential modeling approaches, e.g., transformer, to extract sentence-level text representations automatically and implicitly for human motion synthesis. However, these compact text representations may overemphasize the action names at the expense of other important properties and lack fine-grained details to guide the synthesis of subtly distinct motion. In this paper, we propose hierarchical semantic graphs for fine-grained control over motion generation. Specifically, we disentangle motion descriptions into hierarchical semantic graphs including three levels of motions, actions, and specifics. Such global-to-local structures facilitate a comprehensive understanding of motion description and fine-grained control of motion generation. Correspondingly, to leverage the coarse-to-fine topology of hierarchical semantic graphs, we decompose the text-to-motion diffusion process into three semantic levels, which correspond to capturing the overall motion, local actions, and action specifics. Extensive experiments on two benchmark human motion datasets, including HumanML3D and KIT, with superior performances, justify the efficacy of our method. More encouragingly, by modifying the edge weights of hierarchical semantic graphs, our method can continuously refine the generated motion, which may have a far-reaching impact on the community. Code and pre-trained weights are available at https://github.com/jpthu17/GraphMotion.

----

## [0] DynaDojo: An Extensible Platform for Benchmarking Scaling in Dynamical System Identification

**Authors**: *Logan M. Bhamidipaty, Tommy Bruzzese, Caryn Tran, Rami Ratl Mrad, Maxinder S. Kanwal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32093649cbbcff773d9a991d8c30a7fe-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/32093649cbbcff773d9a991d8c30a7fe-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Modeling complex dynamical systems poses significant challenges, with traditional methods struggling to work on a variety of systems and scale to high-dimensional dynamics. In response, we present DynaDojo, a novel benchmarking platform designed for data-driven dynamical system identification. DynaDojo provides diagnostics on three ways an algorithm’s performance scales: across the number of training samples, the complexity of a dynamical system, and a target error to achieve. Furthermore, DynaDojo enables studying out-of-distribution generalization (by providing unique test conditions for each system) and active learning (by supporting closed-loop control). Through its user-friendly and easily extensible API, DynaDojo accommodates a wide range of user-defined \texttt{Algorithms}, \texttt{Systems}, and \texttt{Challenges} (evaluation metrics). The platform also prioritizes resource-efficient training with parallel processing strategies for running on a cluster. To showcase its utility, in DynaDojo 0.9, we include implementations of 7 baseline algorithms and 20 dynamical systems, along with several demos exhibiting insights researchers can glean using our platform. This work aspires to make DynaDojo a unifying benchmarking platform for system identification, paralleling the role of OpenAI’s Gym in reinforcement learning.

----

## [0] Online Constrained Meta-Learning: Provable Guarantees for Generalization

**Authors**: *Siyuan Xu, Minghui Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/320e941f53db45bddc8757d1c8c4f6aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/320e941f53db45bddc8757d1c8c4f6aa-Abstract-Conference.html)

**Abstract**:

Meta-learning has attracted attention due to its strong ability to learn experiences from known tasks, which can speed up and enhance the learning process for new tasks. However, most existing meta-learning approaches only can learn from tasks without any constraint. This paper proposes an online constrained meta-learning framework, which continuously learns meta-knowledge from sequential learning tasks, and the learning tasks are subject to hard constraints. Beyond existing meta-learning analyses, we provide the upper bounds of optimality gaps and constraint violations produced by the proposed framework, which considers the dynamic regret of online learning, as well as the generalization ability of the task-specific models. Moreover, we provide a practical algorithm for the framework, and validate its superior effectiveness through experiments conducted on meta-imitation learning and few-shot image classification.

----

## [0] Mean-field Langevin dynamics: Time-space discretization, stochastic gradient, and variance reduction

**Authors**: *Taiji Suzuki, Denny Wu, Atsushi Nitanda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32133a6a24d6554263d3584e3ac10faa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32133a6a24d6554263d3584e3ac10faa-Abstract-Conference.html)

**Abstract**:

The mean-field Langevin dynamics (MFLD) is a nonlinear generalization of the Langevin dynamics that incorporates a distribution-dependent drift, and it naturally arises from the optimization of two-layer neural networks via (noisy) gradient descent. Recent works have shown that MFLD globally minimizes an entropy-regularized convex functional in the space of measures. However, all prior analyses assumed the infinite-particle or continuous-time limit, and cannot handle stochastic gradient updates. We provide a general framework to prove a uniform-in-time propagation of chaos for MFLD that takes into account the errors due to finite-particle approximation, time-discretization, and stochastic gradient. To demonstrate the wide applicability of our framework, we establish quantitative convergence rate guarantees to the regularized global optimal solution for $(i)$ a wide range of learning problems such as mean-field neural network and MMD minimization, and $(ii)$ different gradient estimators including SGD and SVRG. Despite the generality of our results, we achieve an improved convergence rate in both the SGD and SVRG settings when specialized to the standard Langevin dynamics.

----

## [0] Public Opinion Field Effect Fusion in Representation Learning for Trending Topics Diffusion

**Authors**: *Junliang Li, Yajun Yang, Qinghua Hu, Xin Wang, Hong Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32246544c237164c365c0527b677a79a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32246544c237164c365c0527b677a79a-Abstract-Conference.html)

**Abstract**:

Trending topic diffusion and prediction analysis is an important problem and has been well studied in social networks. Representation learning is an effective way to extract node embeddings, which can help for topic propagation analysis by completing downstream tasks such as link prediction and node classification. In real world, there are often several trending topics or opinion leaders in public opinion space at the same time and they can be regarded as different centers of public opinion. A public opinion field will be formed surrounding every center. These public opinion fields compete for public's attention and it will potentially affect the development of public opinion. However, the existing methods do not consider public opinion field effect for trending topics diffusion. In this paper, we introduce three well-known observations about public opinion field effect in media and communication studies, and propose a novel and effective heterogeneous representation learning framework to incorporate public opinion field effect and social circle influence effect. To the best of our knowledge, our work is the first to consider these effects in representation learning for trending topic diffusion. Extensive experiments on real-world datasets validate the superiority of our model.

----

## [0] Distributional Pareto-Optimal Multi-Objective Reinforcement Learning

**Authors**: *Xin-Qiang Cai, Pushi Zhang, Li Zhao, Jiang Bian, Masashi Sugiyama, Ashley Llorens*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html)

**Abstract**:

Multi-objective reinforcement learning (MORL) has been proposed to learn control policies over multiple competing objectives with each possible preference over returns. However, current MORL algorithms fail to account for distributional preferences over the multi-variate returns, which are particularly important in real-world scenarios such as autonomous driving. To address this issue, we extend the concept of Pareto-optimality in MORL into distributional Pareto-optimality, which captures the optimality of return distributions, rather than the expectations. Our proposed method, called Distributional Pareto-Optimal Multi-Objective Reinforcement Learning~(DPMORL), is capable of learning distributional Pareto-optimal policies that balance multiple objectives while considering the return uncertainty. We evaluated our method on several benchmark problems and demonstrated its effectiveness in discovering distributional Pareto-optimal policies and satisfying diverse distributional preferences compared to existing MORL methods.

----

## [0] Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning

**Authors**: *Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, William Yang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3255a7554605a88800f4e120b3a929e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3255a7554605a88800f4e120b3a929e1-Abstract-Conference.html)

**Abstract**:

In recent years, pre-trained large language models (LLMs) have demonstrated remarkable efficiency in achieving an inference-time few-shot learning capability known as in-context learning. However, existing literature has highlighted the sensitivity of this capability to the selection of few-shot demonstrations. Current understandings of the underlying mechanisms by which this capability arises from regular language model pretraining objectives remain disconnected from the real-world LLMs. This study aims to examine the in-context learning phenomenon through a Bayesian lens, viewing real-world LLMs as latent variable models. On this premise, we propose an algorithm to select optimal demonstrations from a set of annotated data with a small LM, and then directly generalize the selected demonstrations to larger LMs. We demonstrate significant improvement over baselines, averaged over eight GPT models on eight real-world text classification datasets. We also demonstrate the real-world usefulness of our algorithm on GSM8K, a math word problem dataset. Our empirical findings support our hypothesis that LLMs implicitly infer a latent variable containing task information.

----

## [0] Physics-Driven ML-Based Modelling for Correcting Inverse Estimation

**Authors**: *Ruiyuan Kang, Tingting Mu, Panagiotis Liatsis, Dimitrios C. Kyritsis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3268353cd4ff87451347f242c7401773-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3268353cd4ff87451347f242c7401773-Abstract-Conference.html)

**Abstract**:

When deploying machine learning  estimators in science and engineering (SAE) domains, it is critical  to avoid failed estimations that can have disastrous consequences, e.g., in aero engine design. This work focuses on detecting and correcting  failed  state estimations before adopting them in SAE inverse problems, by  utilizing simulations and performance metrics guided by physical laws. We suggest to flag a machine learning estimation when its physical model error exceeds a feasible threshold, and propose a novel approach, GEESE, to correct it  through optimization, aiming at delivering both low error and high efficiency. The key designs of GEESE include (1) a hybrid surrogate error model to  provide fast  error estimations  to reduce simulation cost and to enable gradient based backpropagation of error feedback, and (2) two generative models to approximate the probability distributions of the candidate states for simulating the  exploitation and exploration behaviours. All three models are constructed as neural networks. GEESE is tested on three real-world SAE inverse problems and compared to a number of state-of-the-art optimization/search approaches. Results show that it fails the least number of times in terms of finding a feasible state correction, and requires physical evaluations less frequently in general.

----

## [0] One-Pass Distribution Sketch for Measuring Data Heterogeneity in Federated Learning

**Authors**: *Zichang Liu, Zhaozhuo Xu, Benjamin Coleman, Anshumali Shrivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32c2f3e0a44d55820da7fbcee0a1d95c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32c2f3e0a44d55820da7fbcee0a1d95c-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is a machine learning paradigm where multiple client devices train models collaboratively without data exchange. Data heterogeneity problem is naturally inherited in FL since data in different clients follow diverse distributions.  To mitigate the negative influence of data heterogeneity, we need to start by measuring it across clients. However, the efficient measurement between distributions is a challenging problem, especially in high dimensionality. In this paper, we propose a one-pass distribution sketch to represent the client data distribution. Our sketching algorithm only requires a single pass of the client data, which is efficient in terms of time and memory. Moreover, we show in both theory and practice that the distance between two distribution sketches represents the divergence between their corresponding distributions. Furthermore, we demonstrate with extensive experiments that our distribution sketch improves the client selection in the FL training. We also showcase that our distribution sketch is an efficient solution to the cold start problem in FL for new clients with unlabeled data.

----

## [0] Kernel-Based Tests for Likelihood-Free Hypothesis Testing

**Authors**: *Patrik Róbert Gerber, Tianze Jiang, Yury Polyanskiy, Rui Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32c6d65ec2591dfcfb3f0e345a51f585-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32c6d65ec2591dfcfb3f0e345a51f585-Abstract-Conference.html)

**Abstract**:

Given $n$ observations from two balanced classes, consider the task of labeling an additional $m$ inputs that are known to all belong to \emph{one} of the two classes. Special cases of this problem are well-known: with completeknowledge of class distributions ($n=\infty$) theproblem is solved optimally by the likelihood-ratio test; when$m=1$ it corresponds to binary classification; and when $m\approx n$ it is equivalent to two-sample testing. The intermediate settings occur in the field of likelihood-free inference, where labeled samples are obtained by running forward simulations and the unlabeled sample is collected experimentally. In recent work it was discovered that there is a fundamental trade-offbetween $m$ and $n$: increasing the data sample $m$ reduces the amount $n$ of training/simulationdata needed. In this work we (a) introduce a generalization where unlabeled samples come from a mixture of the two classes -- a case often encountered in practice; (b) study the minimax sample complexity for non-parametric classes of densities under \textit{maximum meandiscrepancy} (MMD) separation; and (c) investigate the empirical performance of kernels parameterized by neural networks on two tasks: detectionof the Higgs boson and detection of planted DDPM generated images amidstCIFAR-10 images. For both problems we confirm the existence of the theoretically predicted asymmetric $m$ vs $n$ trade-off.

----

## [0] Deep Equilibrium Based Neural Operators for Steady-State PDEs

**Authors**: *Tanya Marwah, Ashwini Pokle, J. Zico Kolter, Zachary C. Lipton, Jianfeng Lu, Andrej Risteski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32cc61322f1e2f56f989d29ccc7cfbb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32cc61322f1e2f56f989d29ccc7cfbb7-Abstract-Conference.html)

**Abstract**:

Data-driven machine learning approaches are being increasingly used to solve partial differential equations (PDEs). They have shown particularly striking successes when training an operator, which takes as input a PDE in some family, and outputs its solution. However, the architectural design space, especially given structural knowledge of the PDE family of interest, is still poorly understood. We seek to remedy this gap by studying the benefits of weight-tied neural network architectures for steady-state PDEs. To achieve this, we first demonstrate that the solution of most steady-state PDEs can be expressed as a fixed point of a non-linear operator. Motivated by this observation, we propose FNO-DEQ, a deep equilibrium variant of the FNO architecture that directly solves for the solution of a steady-state PDE as the infinite-depth fixed point of an implicit operator layer using a black-box root solver and differentiates analytically through this fixed point resulting in $\mathcal{O}(1)$ training memory. Our experiments indicate that FNO-DEQ-based architectures outperform FNO-based baselines with $4\times$ the number of parameters in predicting the solution to steady-state PDEs such as Darcy Flow and steady-state incompressible Navier-Stokes. Finally, we show FNO-DEQ is more robust when trained with datasets with more noisy observations than the FNO-based baselines, demonstrating the benefits of using appropriate inductive biases in architectural design for different neural network based PDE solvers. Further, we show a universal approximation result that demonstrates that FNO-DEQ can approximate the solution to any steady-state PDE that can be written as a fixed point equation.

----

## [0] An Efficient and Robust Framework for Approximate Nearest Neighbor Search with Attribute Constraint

**Authors**: *Mengzhao Wang, Lingwei Lv, Xiaoliang Xu, Yuxiang Wang, Qiang Yue, Jiongkang Ni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32e41d6b0a51a63a9a90697da19d235d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32e41d6b0a51a63a9a90697da19d235d-Abstract-Conference.html)

**Abstract**:

This paper introduces an efficient and robust framework for hybrid query (HQ) processing, which combines approximate nearest neighbor search (ANNS) with attribute constraint. HQ aims to find objects that are similar to a feature vector and match some structured attributes. Existing methods handle ANNS and attribute filtering separately, leading to inefficiency and inaccuracy. Our framework, called native hybrid query (NHQ), builds a composite index based on proximity graph (PG) and applies joint pruning for HQ. We can easily adapt existing PGs to this framework for efficient HQ processing. We also propose two new navigable PGs (NPGs) with optimized edge selection and routing, which improve the overall ANNS performance. We implement five HQ methods based on the proposed NPGs and existing PGs in NHQ, and show that they outperform the state-of-the-art methods on 10 real-world datasets (up to 315$\times$ faster with the same accuracy).

----

## [0] Parameter-efficient Tuning of Large-scale Multimodal Foundation Model

**Authors**: *Haixin Wang, Xinlong Yang, Jianlong Chang, Dian Jin, Jinan Sun, Shikun Zhang, Xiao Luo, Qi Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32ebb6b560ee58abbdae834e5f37cb5d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32ebb6b560ee58abbdae834e5f37cb5d-Abstract-Conference.html)

**Abstract**:

Driven by the progress of large-scale pre-training, parameter-efficient transfer learning has gained immense popularity across different subfields of Artificial Intelligence. The core is to adapt the model to downstream tasks with only a small set of parameters. Recently, researchers have leveraged such proven techniques in multimodal tasks and achieve promising results. However, two critical issues remain unresolved: how to further reduce the complexity with lightweight design and how to boost alignment between modalities under extremely low parameters. In this paper, we propose A gracefUl pRompt framewOrk for cRoss-modal trAnsfer (AURORA) to overcome these challenges. Considering the redundancy in existing architectures, we first utilize the mode approximation to generate 0.1M trainable parameters to implement the multimodal parameter-efficient tuning, which explores the low intrinsic dimension with only 0.04% parameters of the pre-trained model. Then, for better modality alignment, we propose the Informative Context Enhancement and Gated Query Transformation module under extremely few parameters scenes. A thorough evaluation on six cross-modal benchmarks shows that it not only outperforms the state-of-the-art but even outperforms the full fine-tuning approach. Our code is available at: https://github.com/WillDreamer/Aurora.

----

## [0] Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective

**Authors**: *Yifei Wang, Liangchen Li, Jiansheng Yang, Zhouchen Lin, Yisen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32f9049217da6e718a426b07242dff73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32f9049217da6e718a426b07242dff73-Abstract-Conference.html)

**Abstract**:

Adversarial Training (AT) has become arguably the state-of-the-art algorithm for extracting robust features. However, researchers recently notice that AT suffers from severe robust overfitting problems, particularly after learning rate (LR) decay. In this paper, we explain this phenomenon by viewing adversarial training as a dynamic minimax game between the model trainer and the attacker. Specifically, we analyze how LR decay breaks the balance between the minimax game by empowering the trainer with a stronger memorization ability, and show such imbalance induces robust overfitting as a result of memorizing non-robust features. We validate this understanding with extensive experiments, and provide a holistic view of robust overfitting from the dynamics of both the two game players. This understanding further inspires us to alleviate robust overfitting by rebalancing the two players by either regularizing the trainer's capacity or improving the attack strength. Experiments show that the proposed ReBalanced Adversarial Training (ReBAT) can attain good robustness and does not suffer from robust overfitting even after very long training. Code is available at https://github.com/PKU-ML/ReBAT.

----

## [0] Should I Stop or Should I Go: Early Stopping with Heterogeneous Populations

**Authors**: *Hammaad Adam, Fan Yin, Huibin Hu, Neil A. Tenenholtz, Lorin Crawford, Lester Mackey, Allison Koenecke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3322a9a72a1707de14badd5e552ff466-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3322a9a72a1707de14badd5e552ff466-Abstract-Conference.html)

**Abstract**:

Randomized experiments often need to be stopped prematurely due to the treatment having an unintended harmful effect. Existing methods that determine when to stop an experiment early are typically applied to the data in aggregate and do not account for treatment effect heterogeneity. In this paper, we study the early stopping of experiments for harm on heterogeneous populations. We first establish that current methods often fail to stop experiments when the treatment harms a minority group of participants. We then use causal machine learning to develop CLASH, the first broadly-applicable method for heterogeneous early stopping. We demonstrate CLASH's performance on simulated and real data and show that it yields effective early stopping for both clinical trials and A/B tests.

----

## [0] Adaptive Privacy Composition for Accuracy-first Mechanisms

**Authors**: *Ryan M. Rogers, Gennady Samorodnitsky, Zhiwei Steven Wu, Aaditya Ramdas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33301bb40020a56ef56b8b5081e5c4d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33301bb40020a56ef56b8b5081e5c4d5-Abstract-Conference.html)

**Abstract**:

Although there has been work to develop ex-post private mechanisms from Ligett et al. '17 and Whitehouse et al '22 that seeks to provide privacy guarantees subject to a target level of accuracy, there was not a way to use them in conjunction with differentially private mechanisms.  Furthermore, there has yet to be work in developing a theory for how these ex-post privacy mechanisms compose, so that we can track the accumulated privacy over several mechanisms.  We develop privacy filters that allow an analyst to adaptively switch between differentially private mechanisms and ex-post private mechanisms subject to an overall privacy loss guarantee.  We show that using  a particular ex-post private mechanism --- noise reduction mechanisms --- can substantially outperform baseline approaches that use existing privacy loss composition bounds.  We use the common task of returning as many counts as possible subject to a relative error guarantee and an overall privacy budget as a motivating example.

----

## [0] CaMP: Causal Multi-policy Planning for Interactive Navigation in Multi-room Scenes

**Authors**: *Xiaohan Wang, Yuehu Liu, Xinhang Song, Beibei Wang, Shuqiang Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/333581887bf483296118a97773cab0c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/333581887bf483296118a97773cab0c1-Abstract-Conference.html)

**Abstract**:

Visual navigation has been widely studied under the assumption that there may be several clear routes to reach the goal. However, in more practical scenarios such as a house with several messy rooms, there may not. Interactive Navigation (InterNav) considers agents navigating to their goals more effectively with object interactions, posing new challenges of learning interaction dynamics and extra action space. Previous works learn single vision-to-action policy with the guidance of designed representations. However, the causality between actions and outcomes is prone to be confounded when the attributes of obstacles are diverse and hard to measure. Learning policy for long-term action planning in complex scenes also leads to extensive inefficient exploration. In this paper, we introduce a causal diagram of InterNav clarifying the confounding bias caused by obstacles. To address the problem, we propose a multi-policy model that enables the exploration of counterfactual interactions as well as reduces unnecessary exploration. We develop a large-scale dataset containing 600k task episodes in 12k multi-room scenes based on the ProcTHOR simulator and showcase the effectiveness of our method with the evaluations on our dataset.

----

## [0] DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models

**Authors**: *Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, Dong Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/333e67fc4728f147d31608db3ca78e09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/333e67fc4728f147d31608db3ca78e09-Abstract-Conference.html)

**Abstract**:

Even though trained mainly on images, we discover that pretrained diffusion models show impressive power in guiding sketch synthesis. In this paper, we present DiffSketcher, an innovative algorithm that creates \textit{vectorized} free-hand sketches using natural language input. DiffSketcher is developed based on a pre-trained text-to-image diffusion model. It performs the task by directly optimizing a set of BÃ©zier curves with an extended version of the score distillation sampling (SDS) loss, which allows us to use a raster-level diffusion model as a prior for optimizing a parametric vectorized sketch generator. Furthermore, we explore attention maps embedded in the diffusion model for effective stroke initialization to speed up the generation process. The generated sketches demonstrate multiple levels of abstraction while maintaining recognizability, underlying structure, and essential visual details of the subject drawn. Our experiments show that DiffSketcher achieves greater quality than prior work. The code and demo of DiffSketcher can be found at https://ximinng.github.io/DiffSketcher-project/.

----

## [0] Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models

**Authors**: *Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, Mike Zheng Shou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3340ee1e4a8bad8d32c35721712b4d0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3340ee1e4a8bad8d32c35721712b4d0a-Abstract-Conference.html)

**Abstract**:

Public large-scale text-to-image diffusion models, such as Stable Diffusion, have gained significant attention from the community. These models can be easily customized for new concepts using low-rank adaptations (LoRAs). However,  the utilization of multiple-concept LoRAs to jointly support multiple customized concepts presents a challenge. We refer to this scenario as decentralized multi-concept customization, which involves single-client concept tuning and center-node concept fusion. In this paper, we propose a new framework called Mix-of-Show that addresses the challenges of decentralized multi-concept customization, including concept conflicts resulting from existing single-client LoRA tuning and identity loss during model fusion. Mix-of-Show adopts an embedding-decomposed LoRA (ED-LoRA) for single-client tuning and gradient fusion for the center node to preserve the in-domain essence of single concepts and support theoretically limitless concept fusion. Additionally, we introduce regionally controllable sampling, which extends spatially controllable sampling (e.g., ControlNet and T2I-Adapter) to address attribute binding and missing object problems in multi-concept sampling. Extensive experiments demonstrate that Mix-of-Show is capable of composing multiple customized concepts with high fidelity, including characters, objects, and scenes.

----

## [0] ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation

**Authors**: *Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, Yuxiao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33646ef0ed554145eab65f6250fab0c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33646ef0ed554145eab65f6250fab0c9-Abstract-Conference.html)

**Abstract**:

We present a comprehensive solution to learn and improve text-to-image models from human preference feedback.To begin with, we build ImageReward---the first general-purpose text-to-image human preference reward model---to effectively encode human preferences.Its training is based on our systematic annotation pipeline including rating and ranking, which collects 137k expert comparisons to date.In human evaluation, ImageReward outperforms existing scoring models and metrics, making it a promising automatic metric for evaluating text-to-image synthesis.On top of it, we propose Reward Feedback Learning (ReFL), a direct tuning algorithm to optimize diffusion models against a scorer.Both automatic and human evaluation support ReFL's advantages over compared methods.All code and datasets are provided at \url{https://github.com/THUDM/ImageReward}.

----



[Go to the previous page](NIPS-2023-list600.md)

[Go to the next page](NIPS-2023-list602.md)

[Go to the catalog section](README.md)