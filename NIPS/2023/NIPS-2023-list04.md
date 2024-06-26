## [600] Policy Optimization for Continuous Reinforcement Learning

**Authors**: *Hanyang Zhao, Wenpin Tang, David D. Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c53bc01e30711a08f6ac86919193022-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c53bc01e30711a08f6ac86919193022-Abstract-Conference.html)

**Abstract**:

We study reinforcement learning (RL) in the setting of continuous time and space, for an infinite horizon with a discounted objective and the underlying dynamics driven by a stochastic differential equation. Built upon recent advances in the continuous approach to RL, we develop a notion of occupation time (specifically for a discounted objective),  and show how it can be effectively used to derive performance difference and local approximation formulas. We further extend these results to illustrate their applications in the PG (policy gradient) and TRPO/PPO (trust region policy optimization/ proximal policy optimization) methods,  which have been familiar and powerful tools in the discrete RL setting but under-developed in continuous RL. Through numerical experiments, we demonstrate the effectiveness and advantages of our approach.

----

## [601] PrimDiffusion: Volumetric Primitives Diffusion for 3D Human Generation

**Authors**: *Zhaoxi Chen, Fangzhou Hong, Haiyi Mei, Guangcong Wang, Lei Yang, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c575c088de5cfef858b8837251f3027-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c575c088de5cfef858b8837251f3027-Abstract-Conference.html)

**Abstract**:

We present PrimDiffusion, the first diffusion-based framework for 3D human generation. Devising diffusion models for 3D human generation is difficult due to the intensive computational cost of 3D representations and the articulated topology of 3D humans. To tackle these challenges, our key insight is operating the denoising diffusion process directly on a set of volumetric primitives, which models the human body as a number of small volumes with radiance and kinematic information. This volumetric primitives representation marries the capacity of volumetric representations with the efficiency of primitive-based rendering. Our PrimDiffusion framework has three appealing properties: **1)** compact and expressive parameter space for the diffusion model, **2)** flexible representation that incorporates human prior, and **3)** decoder-free rendering for efficient novel-view and novel-pose synthesis. Extensive experiments validate that PrimDiffusion outperforms state-of-the-art methods in 3D human generation. Notably, compared to GAN-based methods, our PrimDiffusion supports real-time rendering of high-quality 3D humans at a resolution of $512\times512$ once the denoising process is done. We also demonstrate the flexibility of our framework on training-free conditional generation such as texture transfer and 3D inpainting.

----

## [602] A Closer Look at the Robustness of Contrastive Language-Image Pre-Training (CLIP)

**Authors**: *Weijie Tu, Weijian Deng, Tom Gedeon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c6be9f09e08ca166cdc0aa26306c61f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c6be9f09e08ca166cdc0aa26306c61f-Abstract-Conference.html)

**Abstract**:

Contrastive Language-Image Pre-training (CLIP) models have demonstrated remarkable generalization capabilities across multiple challenging distribution shifts. However, there is still much to be explored in terms of their robustness to the variations of specific visual factors. In real-world applications, reliable and safe systems must consider other safety measures beyond classification accuracy, such as predictive uncertainty. Yet, the effectiveness of CLIP models on such safety-related objectives is less-explored. Driven by the above, this work comprehensively investigates the safety measures of CLIP models, specifically focusing on three key properties: resilience to visual factor variations, calibrated uncertainty estimations, and the ability to detect anomalous inputs. To this end, we study $83$ CLIP models and $127$ ImageNet classifiers. They are diverse in architecture (pre)training distribution and training strategies. We consider $10$ visual factors (\emph{e.g.}, shape and pattern), $5$ types of out-of-distribution data, and $8$ natural and challenging test conditions with different shift types, such as texture, style, and perturbation shifts. Our study has unveiled several previously unknown insights into CLIP models. For instance, they are not consistently more calibrated than other ImageNet models, which contradicts existing findings. Additionally, our analysis underscores the significance of training source design by showcasing its profound influence on the three key properties. We believe our comprehensive study can shed light on and help guide the development of more robust and reliable CLIP models.

----

## [603] Model Spider: Learning to Rank Pre-Trained Models Efficiently

**Authors**: *Yi-Kai Zhang, Ting-Ji Huang, Yao-Xiang Ding, De-Chuan Zhan, Han-Jia Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c71b14637802ed08eaa3cf50342b2b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c71b14637802ed08eaa3cf50342b2b9-Abstract-Conference.html)

**Abstract**:

Figuring out which Pre-Trained Model (PTM) from a model zoo fits the target task is essential to take advantage of plentiful model resources. With the availability of numerous heterogeneous PTMs from diverse fields, efficiently selecting the most suitable one is challenging due to the time-consuming costs of carrying out forward or backward passes over all PTMs. In this paper, we propose Model Spider, which tokenizes both PTMs and tasks by summarizing their characteristics into vectors to enable efficient PTM selection. By leveraging the approximated performance of PTMs on a separate set of training tasks, Model Spider learns to construct representation and measure the fitness score between a model-task pair via their representation. The ability to rank relevant PTMs higher than others generalizes to new tasks. With the top-ranked PTM candidates, we further learn to enrich task repr. with their PTM-specific semantics to re-rank the PTMs for better selection. Model Spider balances efficiency and selection ability, making PTM selection like a spider preying on a web. Model Spider exhibits promising performance across diverse model zoos, including visual models and Large Language Models (LLMs). Code is available at https://github.com/zhangyikaii/Model-Spider.

----

## [604] Investigating how ReLU-networks encode symmetries

**Authors**: *Georg Bökman, Fredrik Kahl*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c74f005aabbf90a8f1747d99f387321-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c74f005aabbf90a8f1747d99f387321-Abstract-Conference.html)

**Abstract**:

Many data symmetries can be described in terms of group equivariance and the most common way of encoding group equivariances in neural networks is by building linear layers that are group equivariant.In this work we investigate whether equivariance of a network implies that all layers are equivariant.On the theoretical side we find cases where equivariance implies layerwise equivariance, but alsodemonstrate that this is not the case generally.Nevertheless, we conjecture that CNNs that are trained to be equivariant will exhibit layerwise equivariance and explain how this conjecture is a weaker version of the recent permutation conjecture by Entezari et al.\ [2022].We perform quantitative experiments with VGG-nets on CIFAR10 and qualitative experiments with ResNets on ImageNet to illustrate and support our theoretical findings. These experiments are not only of interest for understanding how group equivariance is encoded in ReLU-networks, but they also give a new perspective on Entezari et al.'s permutation conjecture as we find that itis typically easier to merge a network with a group-transformed version of itself than merging two different networks.

----

## [605] Optimal and Fair Encouragement Policy Evaluation and Learning

**Authors**: *Angela Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c7967a442300bff58e9d7b73aa26f24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c7967a442300bff58e9d7b73aa26f24-Abstract-Conference.html)

**Abstract**:

In consequential domains, it is often impossible to compel individuals to take treatment, so that optimal policy rules are merely suggestions in the presence of human non-adherence to treatment recommendations. In these same domains, there may be heterogeneity both in who responds in taking-up treatment, and heterogeneity in treatment efficacy. For example, in social services, a persistent puzzle is the gap in take-up of beneficial services among those who may benefit from them the most. When in addition the decision-maker has distributional preferences over both access and average outcomes, the optimal decision rule changes. We study identification, doubly-robust estimation, and robust estimation under potential violations of positivity. We consider fairness constraints such as demographic parity in treatment take-up, and other constraints, via constrained optimization. Our framework can be extended to handle algorithmic recommendations under an often-reasonable covariate-conditional exclusion restriction, using our robustness checks for lack of positivity in the recommendation. We develop a two-stage, online learning-based algorithm for solving over parametrized policy classes under general constraints to obtain variance-sensitive regret bounds. We assess improved recommendation rules in a stylized case study of optimizing recommendation of supervised release in the PSA-DMF pretrial risk-assessment tool while reducing surveillance disparities.

----

## [606] NICE: NoIse-modulated Consistency rEgularization for Data-Efficient GANs

**Authors**: *Yao Ni, Piotr Koniusz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c8047bf3ed8ef6905351608d641f02f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c8047bf3ed8ef6905351608d641f02f-Abstract-Conference.html)

**Abstract**:

Generative Adversarial Networks (GANs) are powerful tools for image synthesis. However, they require access to vast amounts of training data, which is often costly and prohibitive. Limited data affects GANs, leading to discriminator overfitting and training instability. In this paper, we present a novel approach called NoIse-modulated Consistency rEgularization (NICE) to overcome these challenges. To this end, we introduce an adaptive multiplicative noise into the discriminator to modulate its latent features. We demonstrate the effectiveness of such a modulation in preventing discriminator overfitting by adaptively reducing the Rademacher complexity of the discriminator. However, this modulation leads to an unintended consequence of increased gradient norm, which can undermine the stability of GAN training. To mitigate this undesirable effect, we impose a constraint on the discriminator, ensuring its consistency for the same inputs under different noise modulations. The constraint effectively penalizes the first and second-order gradients of latent features, enhancing GAN stability. Experimental evidence aligns with our theoretical analysis, demonstrating the reduction of generalization error and gradient penalization of NICE. This substantiates the efficacy of NICE in reducing discriminator overfitting and improving stability of GAN training. NICE achieves state-of-the-art results on CIFAR-10, CIFAR-100, ImageNet and FFHQ datasets when trained with limited data, as well as in low-shot generation tasks.

----

## [607] Not All Out-of-Distribution Data Are Harmful to Open-Set Active Learning

**Authors**: *Yang Yang, Yuxuan Zhang, Xin Song, Yi Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2c8d9636f74d0207ff4f65956010f450-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2c8d9636f74d0207ff4f65956010f450-Abstract-Conference.html)

**Abstract**:

Active learning (AL) methods have been proven to be an effective way to reduce the labeling effort by intelligently selecting valuable instances for annotation. Despite their great success with in-distribution (ID) scenarios, AL methods suffer from performance degradation in many real-world applications because out-of-distribution (OOD) instances are always inevitably contained in unlabeled data, which may lead to inefficient sampling. Therefore, several attempts have been explored open-set AL by strategically selecting pure ID instances while filtering OOD instances. However, concentrating solely on selecting pseudo-ID instances may cause the training constraint of the ID classifier and OOD detector. To address this issue, we propose a simple yet effective sampling scheme, Progressive Active Learning (PAL), which employs a progressive sampling mechanism to leverage the active selection of valuable OOD instances. The proposed PAL measures unlabeled instances by synergistically evaluating instances' informativeness and representativeness, and thus it can balance the pseudo-ID and pseudo-OOD instances in each round to enhance both the capacity of the ID classifier and the OOD detector. %Meanwhile, PAL measures unlabeled instances by synergistically evaluating instances' informativeness and representativeness, which can more effectively estimate the values of instances. Extensive experiments on various open-set AL scenarios demonstrate the effectiveness of the proposed PAL, compared with the state-of-the-art methods. The code is available at \url{https://github.com/njustkmg/PAL}.

----

## [608] Improving the Privacy and Practicality of Objective Perturbation for Differentially Private Linear Learners

**Authors**: *Rachel Redberg, Antti Koskela, Yu-Xiang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ceda49041816da6d5a34eb3b612607f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ceda49041816da6d5a34eb3b612607f-Abstract-Conference.html)

**Abstract**:

In the arena of privacy-preserving machine learning, differentially private stochastic gradient descent (DP-SGD) has outstripped the objective perturbation mechanism in popularity and interest. Though unrivaled in versatility, DP-SGD requires a non-trivial privacy overhead (for privately tuning the modelâ€™s hyperparameters) and a computational complexity which might be extravagant for simple models such as linear and logistic regression. This paper revamps the objective perturbation mechanism with tighter privacy analyses and new computational tools that boost it to perform competitively with DP-SGD on unconstrained convex generalized linear problems.

----

## [609] Implicit Differentiable Outlier Detection Enable Robust Deep Multimodal Analysis

**Authors**: *Zhu Wang, Sourav Medya, Sathya N. Ravi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2cf153951b5e9b39564fc4a0ef6adc1a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2cf153951b5e9b39564fc4a0ef6adc1a-Abstract-Conference.html)

**Abstract**:

Deep network models are often purely inductive during both training and inference on unseen data. When these models are used for prediction, but they may fail to capture important semantic information and implicit dependencies within datasets. Recent advancements have shown that combining multiple modalities in large-scale vision and language settings can improve understanding and generalization performance. However, as the model size increases, fine-tuning and deployment become computationally expensive, even for a small number of downstream tasks. Moreover, it is still unclear how domain or prior modal knowledge can be specified in a backpropagation friendly manner, especially in large-scale and noisy settings. To address these challenges, we propose a simplified alternative of combining features from pretrained deep networks and freely available semantic explicit knowledge. In order to remove irrelevant explicit knowledge that does not correspond well to the images, we introduce an implicit Differentiable Out-of-Distribution (OOD) detection layer. This layer addresses outlier detection by solving for fixed points of a differentiable function and using the last iterate of fixed point solver to backpropagate. In practice, we apply our model on several vision and language downstream tasks including visual question answering, visual reasoning, and image-text retrieval on different datasets. Our experiments show that it is possible to design models that perform similarly to state-of-the-art results but with significantly fewer samples and less training time. Our models and code are available here: https://github.com/ellenzhuwang/implicit_vkood

----

## [610] DSR: Dynamical Surface Representation as Implicit Neural Networks for Protein

**Authors**: *Daiwen Sun, He Huang, Yao Li, Xinqi Gong, Qiwei Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d025936bae21d2c2d4cc74779aa77c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d025936bae21d2c2d4cc74779aa77c7-Abstract-Conference.html)

**Abstract**:

We propose a novel neural network-based approach to modeling protein dynamics using an implicit representation of a proteinâ€™s surface in 3D and time. Our method utilizes the zero-level set of signed distance functions (SDFs) to represent protein surfaces, enabling temporally and spatially continuous representations of protein dynamics. Our experimental results demonstrate that our model accurately captures protein dynamic trajectories and can interpolate and extrapolate in 3D and time. Importantly, this is the first study to introduce this method and successfully model large-scale protein dynamics. This approach offers a promising alternative to current methods, overcoming the limitations of first-principles-based and deep learning methods, and provides a more scalable and efficient approach to modeling protein dynamics. Additionally, our surface representation approach simplifies calculations and allows identifying movement trends and amplitudes of protein domains, making it a useful tool for protein dynamics research. Codes are available at https://github.com/Sundw-818/DSR, and we have a project webpage that shows some video results, https://sundw-818.github.io/DSR/.

----

## [611] A Theory of Transfer-Based Black-Box Attacks: Explanation and Implications

**Authors**: *Yanbo Chen, Weiwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d0842550e6d92b0e27e7e810b1a4792-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d0842550e6d92b0e27e7e810b1a4792-Abstract-Conference.html)

**Abstract**:

Transfer-based attacks are a practical method of black-box adversarial attacks, in which the attacker aims to craft adversarial examples from a source (surrogate) model that is transferable to the target model. A wide range of empirical works has tried to explain the transferability of adversarial examples from different angles. However, these works only provide ad hoc explanations without quantitative analyses. The theory behind transfer-based attacks remains a mystery.This paper studies transfer-based attacks under a unified theoretical framework. We propose an explanatory model, called the manifold attack model, that formalizes popular beliefs and explains the existing empirical results. Our model explains why adversarial examples are transferable even when the source model is inaccurate. Moreover, our model implies that the existence of transferable adversarial examples depends on the “curvature” of the data manifold, which quantitatively explains why the success rates of transfer-based attacks are hard to improve. We also discuss the expressive power and the possible extensions of our model in general applications.

----

## [612] Explaining V1 Properties with a Biologically Constrained Deep Learning Architecture

**Authors**: *Galen Pogoncheff, Jacob Granley, Michael Beyeler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d1ef4aba0503226330661d74fdb236e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d1ef4aba0503226330661d74fdb236e-Abstract-Conference.html)

**Abstract**:

Convolutional neural networks (CNNs) have recently emerged as promising models of the ventral visual stream, despite their lack of biological specificity.While current state-of-the-art models of the primary visual cortex (V1) have surfaced from training with adversarial examples and extensively augmented data, these models are still unable to explain key neural properties observed in V1 that arise from biological circuitry.To address this gap, we systematically incorporated neuroscience-derived architectural components into CNNs to identify a set of mechanisms and architectures that more comprehensively explain V1 activity.Upon enhancing task-driven CNNs with architectural components that simulate center-surround antagonism, local receptive fields, tuned normalization, and cortical magnification, we uncover models with latent representations that yield state-of-the-art explanation of V1 neural activity and tuning properties.Moreover, analyses of the learned parameters of these components and stimuli that maximally activate neurons of the evaluated networks provide support for their role in explaining neural properties of V1.Our results highlight an important advancement in the field of NeuroAI, as we systematically establish a set of architectural components that contribute to unprecedented explanation of V1.The neuroscience insights that could be gleaned from increasingly accurate in-silico models of the brain have the potential to greatly advance the fields of both neuroscience and artificial intelligence.

----

## [613] Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models

**Authors**: *Naman Deep Singh, Francesco Croce, Matthias Hein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d3b007613940def7a5ec9d6d635937b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d3b007613940def7a5ec9d6d635937b-Abstract-Conference.html)

**Abstract**:

While adversarial training has been extensively studied for ResNet architectures and low resolution datasets like CIFAR-10, much less is known for ImageNet. Given the recent debate about whether transformers are more robust than convnets, we revisit adversarial training on ImageNet comparing ViTs and ConvNeXts. Extensive experiments show that minor changes in architecture, most notably replacing PatchStem with ConvStem, and training scheme have a significant impact on the achieved robustness. These changes not only increase robustness in the seen $\ell_\infty$-threat model, but even more so improve generalization to unseen $\ell_1/\ell_2$-attacks. Our modified ConvNeXt, ConvNeXt + ConvStem, yields the most robust $\ell_\infty$-models across different ranges of model parameters and FLOPs, while our ViT + ConvStem yields the best generalization to unseen threat models.

----

## [614] URL: A Representation Learning Benchmark for Transferable Uncertainty Estimates

**Authors**: *Michael Kirchhof, Bálint Mucsányi, Seong Joon Oh, Enkelejda Kasneci*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d421cd0e763f9f01958a30bace955bf-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d421cd0e763f9f01958a30bace955bf-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Representation learning has significantly driven the field to develop pretrained models that can act as a valuable starting point when transferring to new datasets. With the rising demand for reliable machine learning and uncertainty quantification, there is a need for pretrained models that not only provide embeddings but also transferable uncertainty estimates. To guide the development of such models, we propose the Uncertainty-aware Representation Learning (URL) benchmark. Besides the transferability of the representations, it also measures the zero-shot transferability of the uncertainty estimate using a novel metric. We apply URL to evaluate ten uncertainty quantifiers that are pretrained on ImageNet and transferred to eight downstream datasets. We find that approaches that focus on the uncertainty of the representation itself or estimate the prediction risk directly outperform those that are based on the probabilities of upstream classes. Yet, achieving transferable uncertainty quantification remains an open challenge. Our findings indicate that it is not necessarily in conflict with traditional representation learning goals. Code is available at https://github.com/mkirchhof/url.

----

## [615] FineMoGen: Fine-Grained Spatio-Temporal Motion Generation and Editing

**Authors**: *Mingyuan Zhang, Huirong Li, Zhongang Cai, Jiawei Ren, Lei Yang, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d52879ef2ba487445ca2e143b104c3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d52879ef2ba487445ca2e143b104c3b-Abstract-Conference.html)

**Abstract**:

Text-driven motion generation has achieved substantial progress with the emergence of diffusion models. However, existing methods still struggle to generate complex motion sequences that correspond to fine-grained descriptions, depicting detailed and accurate spatio-temporal actions.This lack of fine controllability limits the usage of motion generation to a larger audience. To tackle these challenges, we present FineMoGen, a diffusion-based motion generation and editing framework that can synthesize fine-grained motions, with spatial-temporal composition to the user instructions. Specifically, FineMoGen builds upon diffusion model with a novel transformer architecture dubbed Spatio-Temporal Mixture Attention SAMI. SAMI optimizes the generation of the global attention template from two perspectives: 1) explicitly modeling the constraints of spatio-temporal composition; and 2) utilizing sparsely-activated mixture-of-experts to adaptively extract fine-grained features. To facilitate a large-scale study on this new fine-grained motion generation task, we contribute the HuMMan-MoGen dataset, which consists of 2,968 videos and 102,336 fine-grained spatio-temporal descriptions. Extensive experiments validate that FineMoGen  exhibits superior motion generation quality over state-of-the-art methods. Notably, FineMoGen further enables zero-shot motion editing capabilities with the aid of modern large language models (LLM), which faithfully manipulates motion sequences with fine-grained instructions.

----

## [616] Stabilizing the Optimization of Neural Signed Distance Functions and Finer Shape Representation

**Authors**: *Huizong Yang, Yuxin Sun, Ganesh Sundaramoorthi, Anthony J. Yezzi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d6336c1c2987e9d1d9894edd593478d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d6336c1c2987e9d1d9894edd593478d-Abstract-Conference.html)

**Abstract**:

We present new insights and a novel paradigm for learning implicit neural representations (INR) of shapes. In particular, we shed light on the popular eikonal loss used for imposing a signed distance function constraint in INR. We show analytically that as the representation power of the network increases, the optimization approaches a partial differential equation (PDE) in the continuum limit that is unstable. We show that this instability can manifest in existing network optimization, leading to irregularities in the reconstructed surface and/or convergence to sub-optimal local minima, and thus fails to capture fine geometric and topological structure. We show analytically how other terms added to the loss, currently used in the literature for other purposes, can actually eliminate these instabilities. However, such terms can over-regularize the surface, preventing the representation of fine shape detail. Based on a similar PDE theory for the continuum limit, we introduce a new regularization term that still counteracts the eikonal instability but without over-regularizing. Furthermore, since stability is now guaranteed in the continuum limit, this stabilization also allows for considering new network structures that are able to represent finer shape detail. We introduce such a structure based on quadratic layers. Experiments on multiple benchmark data sets show that our new regularization and network are able to capture more precise shape details and more accurate topology than existing state-of-the-art.

----

## [617] Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale

**Authors**: *Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, Wei-Ning Hsu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d8911db9ecedf866015091b28946e15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d8911db9ecedf866015091b28946e15-Abstract-Conference.html)

**Abstract**:

Large-scale generative models such as GPT and DALL-E have revolutionized the research community. These models not only generate high fidelity outputs, but are also generalists which can solve tasks not explicitly taught. In contrast, speech generative models are still primitive in terms of scale and task generalization. In this paper, we present Voicebox, the most versatile text-guided generative model for speech at scale. Voicebox is a non-autoregressive flow-matching model trained to infill speech, given audio context and text, trained on over 50K hours of speech that are not filtered or enhanced. Similar to GPT, Voicebox can perform many different tasks through in-context learning, but is more flexible as it can also condition on future context. Voicebox can be used for mono or cross-lingual zero-shot text-to-speech synthesis, noise removal, content editing, style conversion, and diverse sample generation. In particular, Voicebox outperforms the state-of-the-art zero-shot TTS model VALL-E on both intelligibility (5.9\% vs 1.9\% word error rates) and audio similarity (0.580 vs 0.681) while being up to 20 times faster. Audio samples can be found in \url{https://voicebox.metademolab.com}.

----

## [618] Optimizing Solution-Samplers for Combinatorial Problems: The Landscape of Policy-Gradient Method

**Authors**: *Constantine Caramanis, Dimitris Fotakis, Alkis Kalavasis, Vasilis Kontonis, Christos Tzamos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2d950a2cfd8a75124c178a89545b97fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2d950a2cfd8a75124c178a89545b97fd-Abstract-Conference.html)

**Abstract**:

Deep Neural Networks and Reinforcement Learning methods have empirically shown great promise in tackling challenging combinatorial problems. In those methods a deep neural network is used as a solution generator which is then trained by gradient-based methods (e.g., policy gradient) to successively obtain better solution distributions.In this work we introduce a novel theoretical framework for analyzing the effectiveness of such methods. We ask whether there exist generative models that (i) are expressive enough to generate approximately optimal solutions; (ii) have a tractable, i.e, polynomial in the size of the input, number of parameters; (iii) their optimization landscape is benign in the sense that it does not contain sub-optimal stationary points. Our main contribution is a positive answer to this question. Our result holds for a broad class of combinatorial problems including Max- and Min-Cut, Max-$k$-CSP, Maximum-Weight-Bipartite-Matching, and the Traveling Salesman Problem. As a byproduct of our analysis we introduce a novel regularization process over vanilla gradient descent and provide theoretical and experimental evidence that it helps address vanishing-gradient issues and escape bad stationary points.

----

## [619] SoTTA: Robust Test-Time Adaptation on Noisy Data Streams

**Authors**: *Taesik Gong, Yewon Kim, Taeckyung Lee, Sorn Chottananurak, Sung-Ju Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2da53cd1abdae59150e35f4693834f32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2da53cd1abdae59150e35f4693834f32-Abstract-Conference.html)

**Abstract**:

Test-time adaptation (TTA) aims to address distributional shifts between training and testing data using only unlabeled test data streams for continual model adaptation. However, most TTA methods assume benign test streams, while test samples could be unexpectedly diverse in the wild. For instance, an unseen object or noise could appear in autonomous driving. This leads to a new threat to existing TTA algorithms; we found that prior TTA algorithms suffer from those noisy test samples as they blindly adapt to incoming samples. To address this problem, we present Screening-out Test-Time Adaptation (SoTTA), a novel TTA algorithm that is robust to noisy samples. The key enabler of SoTTA is two-fold: (i) input-wise robustness via high-confidence uniform-class sampling that effectively filters out the impact of noisy samples and (ii) parameter-wise robustness via entropy-sharpness minimization that improves the robustness of model parameters against large gradients from noisy samples. Our evaluation with standard TTA benchmarks with various noisy scenarios shows that our method outperforms state-of-the-art TTA methods under the presence of noisy samples and achieves comparable accuracy to those methods without noisy samples. The source code is available at https://github.com/taeckyung/SoTTA.

----

## [620] FouriDown: Factoring Down-Sampling into Shuffling and Superposing

**Authors**: *Qi Zhu, Man Zhou, Jie Huang, Naishan Zheng, Hongzhi Gao, Chongyi Li, Yuan Xu, Feng Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2dae7d1ccf1edf76f8ce7c282bdf4730-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2dae7d1ccf1edf76f8ce7c282bdf4730-Abstract-Conference.html)

**Abstract**:

Spatial down-sampling techniques, such as strided convolution, Gaussian, and Nearest down-sampling, are essential in deep neural networks. In this study, we revisit the working mechanism of the spatial down-sampling family and analyze the biased effects caused by the static weighting strategy employed in previous approaches. To overcome this limitation, we propose a novel  down-sampling paradigm in the Fourier domain, abbreviated as FouriDown, which unifies existing down-sampling techniques. Drawing inspiration from the signal sampling theorem, we parameterize the non-parameter static weighting down-sampling operator as a learnable and context-adaptive operator within a unified Fourier function. Specifically, we organize the corresponding frequency positions of the 2D plane in a physically-closed manner within a single channel dimension. We then perform point-wise channel shuffling based on an indicator that determines whether a channel's signal frequency bin is susceptible to aliasing, ensuring the consistency of the weighting parameter learning. FouriDown, as a generic operator, comprises four key components: 2D discrete Fourier transform, context shuffling rules, Fourier weighting-adaptively superposing rules, and 2D inverse Fourier transform. These components can be easily integrated into existing image restoration networks. To demonstrate the efficacy of FouriDown, we conduct extensive experiments on image de-blurring and low-light image enhancement. The results consistently show that FouriDown can provide significant performance improvements. We will make the code publicly available to facilitate further exploration and application of FouriDown.

----

## [621] Participatory Personalization in Classification

**Authors**: *Hailey Joren, Chirag Nagpal, Katherine A. Heller, Berk Ustun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2dbb8bfe4cd3875609b23799830ee865-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2dbb8bfe4cd3875609b23799830ee865-Abstract-Conference.html)

**Abstract**:

Machine learning models are often personalized based on information that is protected, sensitive, self-reported, or costly to acquire. These models use information about people, but do not facilitate nor inform their consent. Individuals cannot opt out of reporting information that a model needs to personalize their predictions nor tell if they benefit from personalization in the first place. We introduce a new family of prediction models, called participatory systems, that let individuals opt into personalization at prediction time. We present a model-agnostic algorithm to learn participatory systems for supervised learning tasks where models are personalized with categorical group attributes. We conduct a comprehensive empirical study of participatory systems in clinical prediction tasks, comparing them to common approaches for personalization and imputation. Our results show that participatory systems can facilitate and inform consent in a way that improves performance and privacy across all groups who report personal data.

----

## [622] A Neural Collapse Perspective on Feature Evolution in Graph Neural Networks

**Authors**: *Vignesh Kothapalli, Tom Tirer, Joan Bruna*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2dd8a2a8685602586c1173f0b644d0e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2dd8a2a8685602586c1173f0b644d0e3-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) have become increasingly popular for classification tasks on graph-structured data. Yet, the interplay between graph topology and feature evolution in GNNs is not well understood. In this paper, we focus on node-wise classification, illustrated with community detection on stochastic block model graphs, and explore the feature evolution through the lens of the "Neural Collapse" (NC) phenomenon. When training instance-wise deep classifiers (e.g. for image classification) beyond the zero training error point, NC demonstrates a reduction in the deepest features' within-class variability and an increased alignment of their class means to certain symmetric structures. We start with an empirical study that shows that a decrease in within-class variability is also prevalent in the node-wise classification setting, however, not to the extent observed in the instance-wise case. Then, we theoretically study this distinction. Specifically, we show that even an "optimistic" mathematical model requires that the graphs obey a strict structural condition in order to possess a minimizer with exact collapse. Furthermore, by studying the gradient dynamics of this model, we provide reasoning for the partial collapse observed empirically. Finally, we present a study on the evolution of within- and between-class feature variability across layers of a well-trained GNN and contrast the behavior with spectral methods.

----

## [623] ResoNet: Noise-Trained Physics-Informed MRI Off-Resonance Correction

**Authors**: *Alfredo De Goyeneche Macaya, Shreya Ramachandran, Ke Wang, Ekin Karasan, Joseph Y. Cheng, Stella X. Yu, Michael Lustig*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e0bd92a1d3600d4288df51ac5e6be5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e0bd92a1d3600d4288df51ac5e6be5f-Abstract-Conference.html)

**Abstract**:

Magnetic Resonance Imaging (MRI) is a powerful medical imaging modality that offers diagnostic information without harmful ionizing radiation. Unlike optical imaging, MRI sequentially samples the spatial Fourier domain (k-space) of the image. Measurements are collected in multiple shots, or readouts, and in each shot, data along a smooth trajectory is sampled.Conventional MRI data acquisition relies on sampling k-space row-by-row in short intervals, which is slow and inefficient. More efficient, non-Cartesian sampling trajectories (e.g., Spirals) use longer data readout intervals, but are more susceptible to magnetic field inhomogeneities, leading to off-resonance artifacts. Spiral trajectories cause off-resonance blurring in the image, and the mathematics of this blurring resembles that of optical blurring, where magnetic field variation corresponds to depth and readout duration to aperture size. Off-resonance blurring is a system issue with a physics-based, accurate forward model. We present a physics-informed deep learning framework for off-resonance correction in MRI, which is trained exclusively on synthetic, noise-like data with representative marginal statistics. Our approach allows for fat/water separation and is compatible with parallel imaging acceleration. Through end-to-end training using synthetic randomized data (i.e., noise-like images, coil sensitivities, field maps), we train the network to reverse off-resonance effects across diverse anatomies and contrasts without retraining. We demonstrate the effectiveness of our approach through results on phantom and in-vivo data. This work has the potential to facilitate the clinical adoption of non-Cartesian sampling trajectories, enabling efficient, rapid, and motion-robust MRI scans. Code is publicly available at: https://github.com/mikgroup/ResoNet.

----

## [624] Eliminating Domain Bias for Federated Learning in Representation Space

**Authors**: *Jianqing Zhang, Yang Hua, Jian Cao, Hao Wang, Tao Song, Zhengui Xue, Ruhui Ma, Haibing Guan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e0d3c6ad1a4d85bef3cfe63af58bc76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e0d3c6ad1a4d85bef3cfe63af58bc76-Abstract-Conference.html)

**Abstract**:

Recently, federated learning (FL) is popular for its privacy-preserving and collaborative learning abilities. However, under statistically heterogeneous scenarios, we observe that biased data domains on clients cause a representation bias phenomenon and further degenerate generic representations during local training, i.e., the representation degeneration phenomenon. To address these issues, we propose a general framework Domain Bias Eliminator (DBE) for FL. Our theoretical analysis reveals that DBE can promote bi-directional knowledge transfer between server and client, as it reduces the domain discrepancy between server and client in representation space. Besides, extensive experiments on four datasets show that DBE can greatly improve existing FL methods in both generalization and personalization abilities. The DBE-equipped FL method can outperform ten state-of-the-art personalized FL methods by a large margin. Our code is public at https://github.com/TsingZ0/DBE.

----

## [625] Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression

**Authors**: *Allan Raventós, Mansheej Paul, Feng Chen, Surya Ganguli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e10b2c2e1aa4f8083c37dfe269873f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e10b2c2e1aa4f8083c37dfe269873f8-Abstract-Conference.html)

**Abstract**:

Pretrained transformers exhibit the remarkable ability of in-context learning (ICL): they can learn tasks from just a few examples provided in the prompt without updating any weights. This raises a foundational question: can ICL solve fundamentally new tasks that are very different from those seen during pretraining? To probe this question, we examine ICL’s performance on linear regression while varying the diversity of tasks in the pretraining dataset. We empirically demonstrate a task diversity threshold for the emergence of ICL. Below this threshold, the pretrained transformer cannot solve unseen regression tasks, instead behaving like a Bayesian estimator with the non-diverse pretraining task distribution as the prior. Beyond this threshold, the transformer significantly outperforms this estimator; its behavior aligns with that of ridge regression, corresponding to a Gaussian prior over all tasks, including those not seen during pretraining. Thus, when pretrained on data with task diversity greater than the threshold, transformers can optimally solve fundamentally new tasks in-context. Importantly, this capability hinges on it deviating from the Bayes optimal estimator with the pretraining distribution as the prior. This study also explores the effect of regularization, model capacity and task structure and underscores, in a concrete example, the critical role of task diversity, alongside data and model scale, in the emergence of ICL.

----

## [626] Two-Stage Predict+Optimize for MILPs with Unknown Parameters in Constraints

**Authors**: *Xinyi Hu, Jasper C. H. Lee, Jimmy Ho-Man Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e14be0332c04c76742710e417cedb2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e14be0332c04c76742710e417cedb2a-Abstract-Conference.html)

**Abstract**:

Consider the setting of constrained optimization, with some parameters unknown at solving time and requiring prediction from relevant features. Predict+Optimize is a recent framework for end-to-end training supervised learning models for such predictions, incorporating information about the optimization problem in the training process in order to yield better predictions in terms of the quality of the predicted solution under the true parameters. Almost all prior works have focused on the special case where the unknowns appear only in the optimization objective and not the constraints. Hu et al. proposed the first adaptation of Predict+Optimize to handle unknowns appearing in constraints, but the framework has somewhat ad-hoc elements, and they provided a training algorithm only for covering and packing linear programs. In this work, we give a new simpler and more powerful framework called Two-Stage Predict+Optimize, which we believe should be the canonical framework for the Predict+Optimize setting. We also give a training algorithm usable for all mixed integer linear programs, vastly generalizing the applicability of the framework. Experimental results demonstrate the superior prediction performance of our training framework over all classical and state-of-the-art methods.

----

## [627] Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective

**Authors**: *Zhiding Liu, Mingyue Cheng, Zhi Li, Zhenya Huang, Qi Liu, Yanhu Xie, Enhong Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html)

**Abstract**:

Deep learning models have progressively advanced time series forecasting due to their powerful capacity in capturing sequence dependence. Nevertheless, it is still challenging to make accurate predictions due to the existence of non-stationarity in real-world data, denoting the data distribution rapidly changes over time. To mitigate such a dilemma, several efforts have been conducted by reducing the non-stationarity with normalization operation. However, these methods typically overlook the distribution discrepancy between the input series and the horizon series, and assume that all time points within the same instance share the same statistical properties, which is too ideal and may lead to suboptimal relative improvements. To this end, we propose a novel slice-level adaptive normalization, referred to \textbf{SAN}, which is a novel scheme for empowering time series forecasting with more flexible normalization and denormalization. SAN includes two crucial designs. First, SAN tries to eliminate the non-stationarity of time series in units of a local temporal slice (i.e., sub-series) rather than a global instance. Second, SAN employs a slight network module to independently model the evolving trends of statistical properties of raw time series. Consequently, SAN could serve as a general model-agnostic plugin and better alleviate the impact of the non-stationary nature of time series data. We instantiate the proposed SAN on four widely used forecasting models and test their prediction results on benchmark datasets to evaluate its effectiveness. Also, we report some insightful findings to deeply analyze and understand our proposed SAN. We make our codes publicly available.

----

## [628] Distance-Restricted Folklore Weisfeiler-Leman GNNs with Provable Cycle Counting Power

**Authors**: *Junru Zhou, Jiarui Feng, Xiyuan Wang, Muhan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e2e7c2e3c2e70fa2e9756dce728fcca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e2e7c2e3c2e70fa2e9756dce728fcca-Abstract-Conference.html)

**Abstract**:

The ability of graph neural networks (GNNs) to count certain graph substructures, especially cycles, is important for the success of GNNs on a wide range of tasks. It has been recently used as a popular metric for evaluating the expressive power of GNNs. Many of the proposed GNN models with provable cycle counting power are based on subgraph GNNs, i.e., extracting a bag of subgraphs from the input graph, generating representations for each subgraph, and using them to augment the representation of the input graph. However, those methods require heavy preprocessing, and suffer from high time and memory costs. In this paper, we overcome the aforementioned limitations of subgraph GNNs by proposing a novel class of GNNs---$d$-Distance-Restricted FWL(2) GNNs, or $d$-DRFWL(2) GNNs, based on the well-known FWL(2) algorithm. As a heuristic method for graph isomorphism testing, FWL(2) colors all node pairs in a graph and performs message passing among those node pairs. In order to balance the expressive power and complexity, $d$-DRFWL(2) GNNs simplify FWL(2) by restricting the range of message passing to node pairs whose mutual distances are at most $d$. This way, $d$-DRFWL(2) GNNs exploit graph sparsity while avoiding the expensive subgraph extraction operations in subgraph GNNs, making both the time and space complexity lower. We theoretically investigate both the discriminative power and the cycle counting power of $d$-DRFWL(2) GNNs. Our most important finding is that $d$-DRFWL(2) GNNs have provably strong cycle counting power even with $d=2$: they can count all 3, 4, 5, 6-cycles. Since 6-cycles (e.g., benzene rings) are ubiquitous in organic molecules, being able to detect and count them is crucial for achieving robust and generalizable performance on molecular tasks. Experiments on both synthetic datasets and molecular datasets verify our theory. To the best of our knowledge, 2-DRFWL(2) GNN is the most efficient GNN model to date (both theoretically and empirically) that can count up to 6-cycles.

----

## [629] Computing a human-like reaction time metric from stable recurrent vision models

**Authors**: *Lore Goetschalckx, Lakshmi Narasimhan Govindarajan, Alekh Karkada Ashok, Aarit Ahuja, David L. Sheinberg, Thomas Serre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e351740d4ec4200df6160f34cd181c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e351740d4ec4200df6160f34cd181c3-Abstract-Conference.html)

**Abstract**:

The meteoric rise in the adoption of deep neural networks as computational models of vision has inspired efforts to ``align‚Äù these models with humans. One dimension of interest for alignment includes behavioral choices, but moving beyond characterizing choice patterns to capturing temporal aspects of visual decision-making has been challenging. Here, we sketch a general-purpose methodology to construct computational accounts of reaction times from a stimulus-computable, task-optimized model. Specifically, we introduce a novel metric leveraging insights from subjective logic theory summarizing evidence accumulation in recurrent vision models. We demonstrate that our metric aligns with patterns of human reaction times for stimulus manipulations across four disparate visual decision-making tasks spanning perceptual grouping, mental simulation, and scene categorization. This work paves the way for exploring the temporal alignment of model and human visual strategies in the context of various other cognitive tasks toward generating testable hypotheses for neuroscience. Links to the code and data can be found on the project page: https://serre-lab.github.io/rnnrtssite/.

----

## [630] ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence

**Authors**: *Ben Dai, Yixuan Qiu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e37e56599e3f49cc899f40ae4f5d1fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e37e56599e3f49cc899f40ae4f5d1fa-Abstract-Conference.html)

**Abstract**:

Empirical risk minimization (ERM) is a crucial framework that offers a general approach to handling a broad range of machine learning tasks. In this paper, we propose a novel algorithm, called ReHLine, for minimizing a set of regularized ERMs with convex piecewise linear-quadratic loss functions and optional linear constraints. The proposed algorithm can effectively handle diverse combinations of loss functions, regularization, and constraints, making it particularly well-suited for complex domain-specific problems. Examples of such problems include FairSVM, elastic net regularized quantile regression, Huber minimization, etc. In addition, ReHLine enjoys a provable linear convergence rate and exhibits a per-iteration computational complexity that scales linearly with the sample size. The algorithm is implemented with both Python and R interfaces, and its performance is benchmarked on various tasks and datasets. Our experimental results demonstrate that ReHLine significantly surpasses generic optimization solvers in terms of computational efficiency on large-scale datasets. Moreover, it also outperforms specialized solvers such as Liblinear in SVMs, hqreg in Huber minimization, and Lightning (SAGA, SAG, SDCA, SVRG) in smoothed SVMs, exhibiting exceptional flexibility and efficiency. The source code, project page, accompanying software, and the Python/R interface can be accessed through the link: https://github.com/softmin/ReHLine.

----

## [631] Improved Frequency Estimation Algorithms with and without Predictions

**Authors**: *Anders Aamand, Justin Y. Chen, Huy Lê Nguyen, Sandeep Silwal, Ali Vakilian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e49934cac6cb8604b0c67cfa0828718-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e49934cac6cb8604b0c67cfa0828718-Abstract-Conference.html)

**Abstract**:

Estimating frequencies of elements appearing in a data stream is a key task in large-scale data analysis. Popular sketching approaches to this problem (e.g., CountMin and CountSketch) come with worst-case guarantees that probabilistically bound the error of the estimated frequencies for any possible input. The work of Hsu et al.~(2019) introduced the idea of using machine learning to tailor sketching algorithms to the specific data distribution they are being run on. In particular, their learning-augmented frequency estimation algorithm uses a learned heavy-hitter oracle which predicts which elements will appear many times in the stream. We give a novel algorithm, which in some parameter regimes, already theoretically outperforms the learning based algorithm of Hsu et al. without the use of any predictions. Augmenting our algorithm with heavy-hitter predictions further reduces the error and improves upon the state of the art. Empirically, our algorithms achieve superior performance in all experiments compared to prior approaches.

----

## [632] BCDiff: Bidirectional Consistent Diffusion for Instantaneous Trajectory Prediction

**Authors**: *Rongqing Li, Changsheng Li, Dongchun Ren, Guangyi Chen, Ye Yuan, Guoren Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e57e2c14232a7b99cf76213e190822d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e57e2c14232a7b99cf76213e190822d-Abstract-Conference.html)

**Abstract**:

The objective of pedestrian trajectory prediction is to estimate the future paths of pedestrians by leveraging historical observations, which plays a vital role in ensuring the safety of self-driving vehicles and navigation robots. Previous works usually rely on a sufficient amount of observation time to accurately predict future trajectories. However, there are many real-world situations where the model lacks sufficient time to observe, such as when pedestrians abruptly emerge from blind spots, resulting in inaccurate predictions and even safety risks. Therefore, it is necessary to perform trajectory prediction based on instantaneous observations, which has rarely been studied before. In this paper, we propose a Bi-directional Consistent Diffusion framework tailored for instantaneous trajectory prediction, named BCDiff. At its heart, we develop two coupled diffusion models by designing a mutual guidance mechanism which can bidirectionally and consistently generate unobserved historical trajectories and future trajectories step-by-step,  to utilize the complementary information between them. Specifically, at each step, the predicted unobserved historical trajectories and limited observed trajectories guide one diffusion model to generate future trajectories, while the predicted future trajectories and observed trajectories guide the other diffusion model to predict unobserved historical trajectories. Given the presence of relatively high noise in the generated trajectories during the initial steps, we introduce a gating mechanism to learn the weights between the predicted trajectories and the limited observed trajectories for automatically balancing their contributions. By means of this iterative and mutually guided generation process, both the future and unobserved historical trajectories undergo continuous refinement, ultimately leading to accurate predictions. Essentially, BCDiff is an encoder-free framework that can be compatible with existing trajectory prediction models in principle. Experiments show that our proposed BCDiff significantly improves the accuracy of instantaneous trajectory prediction on the ETH/UCY and Stanford Drone datasets, compared to related approaches.

----

## [633] Leave No Stone Unturned: Mine Extra Knowledge for Imbalanced Facial Expression Recognition

**Authors**: *Yuhang Zhang, Yaqi Li, Lixiong Qin, Xuannan Liu, Weihong Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e6744370a8616c90d1e3b7a41993b7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e6744370a8616c90d1e3b7a41993b7c-Abstract-Conference.html)

**Abstract**:

Facial expression data is characterized by a significant imbalance, with most collected data showing happy or neutral expressions and fewer instances of fear or disgust. This imbalance poses challenges to facial expression recognition (FER) models, hindering their ability to fully understand various human emotional states. Existing FER methods typically report overall accuracy on highly imbalanced test sets but exhibit low performance in terms of the mean accuracy across all expression classes. In this paper, our aim is to address the imbalanced FER problem. Existing methods primarily focus on learning knowledge of minor classes solely from minor-class samples. However, we propose a novel approach to extract extra knowledge related to the minor classes from both major and minor class samples. Our motivation stems from the belief that FER resembles a distribution learning task, wherein a sample may contain information about multiple classes. For instance, a sample from the major class surprise might also contain useful features of the minor class fear. Inspired by that, we propose a novel method that leverages re-balanced attention maps to regularize the model, enabling it to extract transformation invariant information about the minor classes from all training samples. Additionally, we introduce re-balanced smooth labels to regulate the cross-entropy loss, guiding the model to pay more attention to the minor classes by utilizing the extra information regarding the label distribution of the imbalanced training data. Extensive experiments on different datasets and backbones show that the two proposed modules work together to regularize the model and achieve state-of-the-art performance under the imbalanced FER task. Code is available at https://github.com/zyh-uaiaaaa.

----

## [634] ARTree: A Deep Autoregressive Model for Phylogenetic Inference

**Authors**: *Tianyu Xie, Cheng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e9e513860b1342f3a12ebecf0528a21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e9e513860b1342f3a12ebecf0528a21-Abstract-Conference.html)

**Abstract**:

Designing flexible probabilistic models over tree topologies is important for developing efficient phylogenetic inference methods. To do that, previous works often leverage the similarity of tree topologies via hand-engineered heuristic features which would require domain expertise and may suffer from limited approximation capability. In this paper, we propose a deep autoregressive model for phylogenetic inference based on graph neural networks (GNNs), called ARTree. By decomposing a tree topology into a sequence of leaf node addition operations and modeling the involved conditional distributions based on learnable topological features via GNNs, ARTree can provide a rich family of distributions over tree topologies that have simple sampling algorithms, without using heuristic features. We demonstrate the effectiveness and efficiency of our method on a benchmark of challenging real data tree topology density estimation and variational Bayesian phylogenetic inference problems.

----

## [635] A One-Size-Fits-All Approach to Improving Randomness in Paper Assignment

**Authors**: *Yixuan Xu, Steven Jecmen, Zimeng Song, Fei Fang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2e9f9cde1b709281a06dd14f679e4c51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2e9f9cde1b709281a06dd14f679e4c51-Abstract-Conference.html)

**Abstract**:

The assignment of papers to reviewers is a crucial part of the peer review processes of large publication venues, where organizers (e.g., conference program chairs) rely on algorithms to perform automated paper assignment. As such, a major challenge for the organizers of these processes is to specify paper assignment algorithms that find appropriate assignments with respect to various desiderata. Although the main objective when choosing a good paper assignment is to maximize the expertise of each reviewer for their assigned papers, several other considerations make introducing randomization into the paper assignment desirable: robustness to malicious behavior, the ability to evaluate alternative paper assignments, reviewer diversity, and reviewer anonymity. However, it is unclear in what way one should randomize the paper assignment in order to best satisfy all of these considerations simultaneously. In this work, we present a practical, one-size-fits-all method for randomized paper assignment intended to perform well across different motivations for randomness. We show theoretically and experimentally that our method outperforms currently-deployed methods for randomized paper assignment on several intuitive randomness metrics, demonstrating that the randomized assignments produced by our method are general-purpose.

----

## [636] Loss Dynamics of Temporal Difference Reinforcement Learning

**Authors**: *Blake Bordelon, Paul Masset, Henry Kuo, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ea04b568a2deb2d000c59f3a72829b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ea04b568a2deb2d000c59f3a72829b5-Abstract-Conference.html)

**Abstract**:

Reinforcement learning has been successful across several applications in which agents have to learn to act in environments with sparse feedback. However, despite this empirical success there is still a lack of theoretical understanding of how the parameters of reinforcement learning models and the features used to represent states interact to control the dynamics of learning. In this work, we use concepts from statistical physics, to study the typical case learning curves for temporal difference learning of a value function with linear function approximators. Our theory is derived under a Gaussian equivalence hypothesis where averages over the random trajectories are replaced with temporally correlated Gaussian feature averages and we validate our assumptions on small scale Markov Decision Processes. We find that the stochastic semi-gradient noise due to subsampling the space of possible episodes leads to significant plateaus in the value error, unlike in traditional gradient descent dynamics. We study how learning dynamics and plateaus depend on feature structure, learning rate, discount factor, and reward function. We then analyze how strategies like learning rate annealing and reward shaping can favorably alter learning dynamics and plateaus. To conclude, our work introduces new tools to open a new direction towards developing a theory of learning dynamics in reinforcement learning.

----

## [637] REASONER: An Explainable Recommendation Dataset with Comprehensive Labeling Ground Truths

**Authors**: *Xu Chen, Jingsen Zhang, Lei Wang, Quanyu Dai, Zhenhua Dong, Ruiming Tang, Rui Zhang, Li Chen, Xin Zhao, Ji-Rong Wen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ebf43d20e5933ab6d98225bbb908ade-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ebf43d20e5933ab6d98225bbb908ade-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Explainable recommendation has attracted much attention from the industry and academic communities. It has shown great potential to improve the recommendation persuasiveness, informativeness and user satisfaction. In the past few years, while a lot of promising explainable recommender models have been proposed, the datasets used to evaluate them still suffer from several limitations, for example, the explanation ground truths are not labeled by the real users, the explanations are mostly single-modal and around only one aspect. To bridge these gaps, in this paper, we build a new explainable recommendation dataset, which, to our knowledge, is the first contribution that provides a large amount of real user labeled multi-modal and multi-aspect explaination ground truths. In specific, we firstly develop a video recommendation platform, where a series of questions around the recommendation explainability are carefully designed. Then, we recruit about 3000 high-quality labelers with different backgrounds to use the system, and collect their behaviors and feedback to our questions. In this paper, we detail the construction process of our dataset and also provide extensive analysis on its characteristics. In addition, we develop a library, where ten well-known explainable recommender models are implemented in a unified framework. Based on this library, we build several benchmarks for different explainable recommendation tasks. At last, we present many new opportunities brought by our dataset, which are expected to promote the field of explainable recommendation. Our dataset, library and the related documents have been released at https://reasoner2023.github.io/.

----

## [638] Fast Model DeBias with Machine Unlearning

**Authors**: *Ruizhe Chen, Jianfei Yang, Huimin Xiong, Jianhong Bai, Tianxiang Hu, Jin Hao, Yang Feng, Joey Tianyi Zhou, Jian Wu, Zuozhu Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ecc80084c96cc25b11b0ab995c25f47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ecc80084c96cc25b11b0ab995c25f47-Abstract-Conference.html)

**Abstract**:

Recent discoveries have revealed that deep neural networks might behave in a biased manner in many real-world scenarios. For instance, deep networks trained on a large-scale face recognition dataset CelebA tend to predict blonde hair for females and black hair for males. Such biases not only jeopardize the robustness of models but also perpetuate and amplify social biases, which is especially concerning for automated decision-making processes in healthcare, recruitment, etc., as they could exacerbate unfair economic and social inequalities among different groups. Existing debiasing methods suffer from high costs in bias labeling or model re-training, while also exhibiting a deficiency in terms of elucidating the origins of biases within the model. To this respect, we propose a fast model debiasing method (FMD) which offers an efficient approach to identify, evaluate and remove biases inherent in trained models. The FMD identifies biased attributes through an explicit counterfactual concept and quantifies the influence of data samples with influence functions. Moreover, we design a machine unlearning-based strategy to efficiently and effectively remove the bias in a trained model with a small counterfactual dataset. Experiments on the Colored MNIST, CelebA, and Adult Income datasets demonstrate that our method achieves superior or competing classification accuracies compared with state-of-the-art retraining-based methods while attaining significantly fewer biases and requiring much less debiasing cost. Notably, our method requires only a small external dataset and updating a minimal amount of model parameters, without the requirement of access to training data that may be too large or unavailable in practice.

----

## [639] Coherent Soft Imitation Learning

**Authors**: *Joe Watson, Sandy H. Huang, Nicolas Heess*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f0435cffef91068ced08d7c7d8e643e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f0435cffef91068ced08d7c7d8e643e-Abstract-Conference.html)

**Abstract**:

Imitation learning methods seek to learn from an expert either through behavioral cloning (BC) for the policy or inverse reinforcement learning (IRL) for the reward.Such methods enable agents to learn complex tasks from humans that are difficult to capture with hand-designed reward functions.Choosing between BC or IRL for imitation depends on the quality and state-action coverage of the demonstrations, as well as additional access to the Markov decision process. Hybrid strategies that combine BC and IRL are rare, as initial policy optimization against inaccurate rewards diminishes the benefit of pretraining the policy with BC.Our work derives an imitation method that captures the strengths of both BC and IRL.In the entropy-regularized (`soft') reinforcement learning setting, we show that the behavioral-cloned policy can be used as both a shaped reward and a critic hypothesis space by inverting the regularized policy update. This coherency facilitates fine-tuning cloned policies using the reward estimate and additional interactions with the environment.This approach conveniently achieves imitation learning through initial behavioral cloning and subsequent refinement via RL with online or offline data sources.The simplicity of the approach enables graceful scaling to high-dimensional and vision-based tasks, with stable learning and minimal hyperparameter tuning, in contrast to adversarial approaches.For the open-source implementation and simulation results, see https://joemwatson.github.io/csil/.

----

## [640] Exact Generalization Guarantees for (Regularized) Wasserstein Distributionally Robust Models

**Authors**: *Waïss Azizian, Franck Iutzeler, Jérôme Malick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f060912eacace9ce61ef339205ec54c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f060912eacace9ce61ef339205ec54c-Abstract-Conference.html)

**Abstract**:

Wasserstein distributionally robust estimators have emerged as powerful models for prediction and decision-making under uncertainty. These estimators provide attractive generalization guarantees: the robust objective obtained from the training distribution is an exact upper bound on the true risk with high probability. However, existing guarantees either suffer from the curse of dimensionality, are restricted to specific settings, or lead to spurious error terms. In this paper, we show that these generalization guarantees actually hold on general classes of models, do not suffer from the curse of dimensionality, and can even cover distribution shifts at testing. We also prove that these results carry over to the newly-introduced regularized versions of Wasserstein distributionally robust problems.

----

## [641] Supply-Side Equilibria in Recommender Systems

**Authors**: *Meena Jagadeesan, Nikhil Garg, Jacob Steinhardt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f1486343c2c942a617e4f5bb0cc64c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f1486343c2c942a617e4f5bb0cc64c8-Abstract-Conference.html)

**Abstract**:

Algorithmic recommender systems such as Spotify and Netflix affect not only consumer behavior but also producer incentives. Producers seek to create content that will be shown by the recommendation algorithm, which can impact both the diversity and quality of their content. In this work, we investigate the resulting supply-side equilibria in personalized content recommender systems. We model the decisions of producers as choosing multi-dimensional content vectors and users as having heterogenous preferences, which contrasts with classical low-dimensional models. Multi-dimensionality and heterogeneity creates the potential for specialization, where different producers create different types of content at equilibrium. Using a duality argument, we derive necessary and sufficient conditions for whether specialization occurs. Then, we characterize the distribution of content at equilibrium in concrete settings with two populations of users. Lastly, we show that specialization can enable producers to achieve positive profit at equilibrium, which means that specialization can reduce the competitiveness of the marketplace. At a conceptual level, our analysis of supply-side competition takes a step towards elucidating how personalized recommendations shape the marketplace of digital goods.

----

## [642] Human-Aligned Calibration for AI-Assisted Decision Making

**Authors**: *Nina Corvelo Benz, Manuel Rodriguez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f1d1196426ba84f47d115cac3dcb9d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f1d1196426ba84f47d115cac3dcb9d8-Abstract-Conference.html)

**Abstract**:

Whenever a binary classifier is used to provide decision support, it typically provides both a label prediction and a confidence value. Then, the decision maker is supposed to use the confidence value to calibrate how much to trust the prediction. In this context, it has been often argued that the confidence value should correspond to a well calibrated estimate of the probability that the predicted label matches the ground truth label. However, multiple lines of empirical evidence suggest that decision makers have difficulties at developing a good sense on when to trust a prediction using these confidence values. In this paper, our goal is first to understand why and then investigate how to construct more useful confidence values. We first argue that, for a broad class of utility functions, there exists data distributions for which a rational decision maker is, in general, unlikely to discover the optimal decision policy using the above confidence values—an optimal decision maker would need to sometimes place more (less) trust on predictions with lower (higher) confidence values. However, we then show that, if the confidence values satisfy a natural alignment property with respect to the decision maker’s confidence on her own predictions, there always exists an optimal decision policy under which the level of trust the decision maker would need to place on predictions is monotone on the confidence values, facilitating its discoverability. Further, we show that multicalibration with respect to the decision maker’s confidence on her own prediction is a sufficient condition for alignment. Experiments on a real AI-assisted decision making scenario where a classifier provides decision support to human decision makers validate our theoretical results and suggest that alignment may lead to better decisions.

----

## [643] Transformer as a hippocampal memory consolidation model based on NMDAR-inspired nonlinearity

**Authors**: *Dong Kyum Kim, Jea Kwon, Meeyoung Cha, Chul Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f1eb4c897e63870eee9a0a0f7a10332-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f1eb4c897e63870eee9a0a0f7a10332-Abstract-Conference.html)

**Abstract**:

The hippocampus plays a critical role in learning, memory, and spatial representation, processes that depend on the NMDA receptor (NMDAR). Inspired by recent findings that compare deep learning models to the hippocampus, we propose a new nonlinear activation function that mimics NMDAR dynamics. NMDAR-like nonlinearity shifts short-term working memory into long-term reference memory in transformers, thus enhancing a process that is similar to memory consolidation in the mammalian brain. We design a navigation task assessing these two memory functions and show that manipulating the activation function (i.e., mimicking the Mg$^{2+}$-gating of NMDAR) disrupts long-term memory processes. Our experiments suggest that place cell-like functions and reference memory reside in the feed-forward network layer of transformers and that nonlinearity drives these processes. We discuss the role of NMDAR-like nonlinearity in establishing this striking resemblance between transformer architecture and hippocampal spatial representation.

----

## [644] Gaussian Differential Privacy on Riemannian Manifolds

**Authors**: *Yangdi Jiang, Xiaotian Chang, Yi Liu, Lei Ding, Linglong Kong, Bei Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html)

**Abstract**:

We develop an advanced approach for extending Gaussian Differential Privacy (GDP) to general Riemannian manifolds. The concept of GDP stands out as a prominent privacy definition that strongly warrants extension to manifold settings, due to its central limit properties. By harnessing the power of the renowned Bishop-Gromov theorem in geometric analysis, we propose a Riemannian Gaussian distribution that integrates the Riemannian distance, allowing us to achieve GDP in Riemannian manifolds with bounded Ricci curvature. To the best of our knowledge, this work marks the first instance of extending the GDP framework to accommodate general Riemannian manifolds, encompassing curved spaces, and circumventing the reliance on tangent space summaries. We provide a simple algorithm to evaluate the privacy budget $\mu$ on any one-dimensional manifold and introduce a versatile Markov Chain Monte Carlo (MCMC)-based algorithm to calculate $\mu$ on any Riemannian manifold with constant curvature. Through simulations on one of the most prevalent manifolds in statistics, the unit sphere $S^d$, we demonstrate the superior utility of our Riemannian Gaussian mechanism in comparison to the previously proposed Riemannian Laplace mechanism for implementing GDP.

----

## [645] Agnostically Learning Single-Index Models using Omnipredictors

**Authors**: *Aravind Gollakota, Parikshit Gopalan, Adam R. Klivans, Konstantinos Stavropoulos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f46ef5725a8eca24f7f24a17955ad1a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f46ef5725a8eca24f7f24a17955ad1a-Abstract-Conference.html)

**Abstract**:

We give the first result for agnostically learning Single-Index Models (SIMs) with arbitrary monotone and Lipschitz activations. All prior work either held only in the realizable setting or required the activation to be known. Moreover, we only require the marginal to have bounded second moments, whereas all prior work required stronger distributional assumptions (such as anticoncentration or boundedness). Our algorithm is based on recent work by Gopalan et al. [2023] on Omniprediction using predictors satisfying calibrated multiaccuracy. Our analysis is simple and relies on the relationship between Bregman divergences (or matching losses) and $\ell_p$ distances. We also provide new guarantees for standard algorithms like GLMtron and logistic regression in the agnostic setting.

----

## [646] On skip connections and normalisation layers in deep optimisation

**Authors**: *Lachlan E. MacDonald, Jack Valmadre, Hemanth Saratchandran, Simon Lucey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f4d6f8e0f4f543db12260696b2a3551-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f4d6f8e0f4f543db12260696b2a3551-Abstract-Conference.html)

**Abstract**:

We introduce a general theoretical framework, designed for the study of gradient optimisation of deep neural networks, that encompasses ubiquitous architecture choices including batch normalisation, weight normalisation and skip connections.  Our framework determines the curvature and regularity properties of multilayer loss landscapes in terms of their constituent layers, thereby elucidating the roles played by normalisation layers and skip connections in globalising these properties. We then demonstrate the utility of this framework in two respects. First, we give the only proof of which we are aware that a class of deep neural networks can be trained using gradient descent to global optima even when such optima only exist at infinity, as is the case for the cross-entropy cost.  Second, we identify a novel causal mechanism by which skip connections accelerate training, which we verify predictively with ResNets on MNIST, CIFAR10, CIFAR100 and ImageNet.

----

## [647] Efficient Low-rank Backpropagation for Vision Transformer Adaptation

**Authors**: *Yuedong Yang, Hung-Yueh Chiang, Guihong Li, Diana Marculescu, Radu Marculescu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f75a57e9c71e8369da0150ea769d5a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f75a57e9c71e8369da0150ea769d5a2-Abstract-Conference.html)

**Abstract**:

The increasing scale of vision transformers (ViT) has made the efficient fine-tuning of these large models for specific needs a significant challenge in various applications. This issue originates from the computationally demanding matrix multiplications required during the backpropagation process through linear layers in ViT.In this paper, we tackle this problem by proposing a new Low-rank BackPropagation via Walsh-Hadamard Transformation (LBP-WHT) method. Intuitively, LBP-WHT projects the gradient into a low-rank space and carries out backpropagation. This approach substantially reduces the computation needed for adapting ViT, as matrix multiplication in the low-rank space is far less resource-intensive. We conduct extensive experiments with different models (ViT, hybrid convolution-ViT model) on multiple datasets to demonstrate the effectiveness of our method. For instance, when adapting an EfficientFormer-L1 model on CIFAR100, our LBP-WHT achieves 10.4\% higher accuracy than the state-of-the-art baseline, while requiring 9 MFLOPs less computation.As the first work to accelerate ViT adaptation with low-rank backpropagation, our LBP-WHT method is complementary to many prior efforts and can be combined with them for better performance.

----

## [648] AdaVAE: Bayesian Structural Adaptation for Variational Autoencoders

**Authors**: *Paribesh Regmi, Rui Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f76a3a0f44263b5e56fec69ee1220f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f76a3a0f44263b5e56fec69ee1220f9-Abstract-Conference.html)

**Abstract**:

The neural network structures of generative models and their corresponding inference models paired in variational autoencoders (VAEs) play a critical role in the models' generative performance. However, powerful VAE network structures are hand-crafted and fixed prior to training, resulting in a one-size-fits-all approach that requires heavy computation to tune for given data. Moreover, existing VAE regularization methods largely overlook the importance of network structures and fail to prevent overfitting in deep VAE models with cascades of hidden layers. To address these issues, we propose a Bayesian inference framework that automatically adapts VAE network structures to data and prevent overfitting as they grow deeper. We model the number of hidden layers with a beta process to infer the most plausible encoding/decoding network depths warranted by data and perform layer-wise dropout regularization with a conjugate Bernoulli process. We develop a scalable estimator that performs joint inference on both VAE network structures and latent variables. Our experiments show that the inference framework effectively prevents overfitting in both shallow and deep VAE models, yielding state-of-the-art performance. We demonstrate that our framework is compatible with different types of VAE backbone networks and can be applied to various VAE variants, further improving their performance.

----

## [649] Safety Verification of Decision-Tree Policies in Continuous Time

**Authors**: *Christian Schilling, Anna Lukina, Emir Demirovic, Kim Guldstrand Larsen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html)

**Abstract**:

Decision trees have gained popularity as interpretable surrogate models for learning-based control policies. However, providing safety guarantees for systems controlled by decision trees is an open challenge. We show that the problem is undecidable even for systems with the simplest dynamics, and PSPACE-complete for finite-horizon properties. The latter can be verified for discrete-time systems via bounded model checking. However, for continuous-time systems, such an approach requires discretization, thereby weakening the guarantees for the original system. This paper presents the first algorithm to directly verify decision-tree controlled system in continuous time. The key aspect of our method is exploiting the decision-tree structure to propagate a set-based approximation through the decision nodes. We demonstrate the effectiveness of our approach by verifying safety of several decision trees distilled to imitate neural-network policies for nonlinear systems.

----

## [650] Quasi-Monte Carlo Graph Random Features

**Authors**: *Isaac Reid, Adrian Weller, Krzysztof Marcin Choromanski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f9b3ee2bcea04b327c09d7e3145bd1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f9b3ee2bcea04b327c09d7e3145bd1e-Abstract-Conference.html)

**Abstract**:

We present a novel mechanism to improve the accuracy of the recently-introduced class of graph random features (GRFs). Our method induces negative correlations between the lengths of the algorithm's random walks by imposing antithetic termination: a procedure to sample more diverse random walks which may be of independent interest. It has a trivial drop-in implementation. We derive strong theoretical guarantees on the properties of these quasi-Monte Carlo GRFs (q-GRFs), proving that they yield lower-variance estimators of the $2$-regularised Laplacian kernel under mild conditions. Remarkably, our results hold for any graph topology. We demonstrate empirical accuracy improvements on a variety of tasks including a new practical application: time-efficient approximation of the graph diffusion process. To our knowledge, q-GRFs constitute the first rigorously studied quasi-Monte Carlo scheme for kernels defined on combinatorial objects, inviting new research on correlations between graph random walks.

----

## [651] Functional Renyi Differential Privacy for Generative Modeling

**Authors**: *Dihong Jiang, Sun Sun, Yaoliang Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2f9ee101e35b890d9eae79ee27bcd69a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2f9ee101e35b890d9eae79ee27bcd69a-Abstract-Conference.html)

**Abstract**:

Differential privacy (DP) has emerged as a rigorous notion to quantify data privacy.  Subsequently, Renyi differential privacy (RDP) becomes an alternative to the ordinary DP notion in both theoretical and empirical studies, for its convenient compositional rules and flexibility. However, most mechanisms with DP (RDP) guarantees are essentially based on randomizing a fixed, finite-dimensional vector output. In this work, following Hall et al. (2013) we further extend RDP to functional outputs, where the output space can be infinite-dimensional, and develop all necessary tools, *e.g.*, (subsampled) Gaussian mechanism, composition, and post-processing rules, to facilitate its practical adoption. As an illustration, we apply functional RDP (f-RDP) to functions in the reproducing kernel Hilbert space (RKHS) to develop a differentially private generative model (DPGM), where training can be interpreted as iteratively releasing loss functions (in an RKHS) with DP (RDP) guarantees. Empirically, the new training paradigm achieves a  significant improvement in privacy-utility trade-off compared to existing alternatives, especially when $\epsilon=0.2$. Our code is available at https://github.com/dihjiang/DP-kernel.

----

## [652] FedL2P: Federated Learning to Personalize

**Authors**: *Royson Lee, Minyoung Kim, Da Li, Xinchi Qiu, Timothy M. Hospedales, Ferenc Huszar, Nicholas D. Lane*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2fb57276bfbaf1b832d7bfcba36bb41c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2fb57276bfbaf1b832d7bfcba36bb41c-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) research has made progress in developing algorithms for distributed learning of global models, as well as algorithms for local personalization of those common models to the specifics of each client’s local data distribution. However, different FL problems may require different personalization strategies, and it may not even be possible to define an effective one-size-fits-all personalization strategy for all clients: Depending on how similar each client’s optimal predictor is to that of the global model, different personalization strategies may be preferred. In this paper, we consider the federated meta-learning problem of learning personalization strategies. Specifically, we consider meta-nets that induce the batch-norm and learning rate parameters for each client given local data statistics. By learning these meta-nets through FL, we allow the whole FL network to collaborate in learning a customized personalization strategy for each client. Empirical results show that this framework improves on a range of standard hand-crafted personalization baselines in both label and feature shift situations.

----

## [653] Learning Reliable Logical Rules with SATNet

**Authors**: *Zhaoyu Li, Jinpei Guo, Yuhe Jiang, Xujie Si*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/2ff46d83d1dcc063e075058b29d55efe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/2ff46d83d1dcc063e075058b29d55efe-Abstract-Conference.html)

**Abstract**:

Bridging logical reasoning and deep learning is crucial for advanced AI systems. In this work, we present a new framework that addresses this goal by generating interpretable and verifiable logical rules through differentiable learning, without relying on pre-specified logical structures. Our approach builds upon SATNet, a differentiable MaxSAT solver that learns the underlying rules from input-output examples. Despite its efficacy, the learned weights in SATNet are not straightforwardly interpretable, failing to produce human-readable rules. To address this, we propose a novel specification method called ``maximum equality'', which enables the interchangeability between the learned weights of SATNet and a set of propositional logical rules in weighted MaxSAT form. With the decoded weighted MaxSAT formula, we further introduce several effective verification techniques to validate it against the ground truth rules. Experiments on stream transformations and Sudoku problems show that our decoded rules are highly reliable: using exact solvers on them could achieve 100% accuracy, whereas the original SATNet fails to give correct solutions in many cases. Furthermore, we formally verify that our decoded logical rules are functionally equivalent to the ground truth ones.

----

## [654] Demo2Code: From Summarizing Demonstrations to Synthesizing Code via Extended Chain-of-Thought

**Authors**: *Yuki Wang, Gonzalo Gonzalez-Pumariega, Yash Sharma, Sanjiban Choudhury*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30699996ff411d48903c9752b782a5c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30699996ff411d48903c9752b782a5c1-Abstract-Conference.html)

**Abstract**:

Language instructions and demonstrations are two natural ways for users to teach robots personalized tasks. Recent progress in Large Language Models (LLMs) has shown impressive performance in translating language instructions into code for robotic tasks. However, translating demonstrations into task code continues to be a challenge due to the length and complexity of both demonstrations and code, making learning a direct mapping intractable. This paper presents Demo2Code, a novel framework that generates robot task code from demonstrations via an extended chain-of-thought and defines a common latent specification to connect the two. Our framework employs a robust two-stage process: (1) a recursive summarization technique that condenses demonstrations into concise specifications, and (2) a code synthesis approach that expands each function recursively from the generated specifications. We conduct extensive evaluation on various robot task benchmarks, including a novel game benchmark Robotouille, designed to simulate diverse cooking tasks in a kitchen environment.

----

## [655] Easy Learning from Label Proportions

**Authors**: *Róbert Busa-Fekete, Heejin Choi, Travis Dick, Claudio Gentile, Andrés Muñoz Medina*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3085fd61063840fdb2e6eafac58589f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3085fd61063840fdb2e6eafac58589f8-Abstract-Conference.html)

**Abstract**:

We consider the problem of Learning from Label Proportions (LLP), a weakly supervised classification setup where instances are grouped into i.i.d. “bags”, and only the frequency of class labels at each bag is available.  Albeit, the objective of the learner is to achieve low task loss at an individual instance level.  Here we propose EASYLLP, a flexible and simple-to-implement debiasing approach based on aggregate labels, which operates on arbitrary loss functions. Our technique allows us to accurately estimate the expected loss of an arbitrary model at an individual level.  We elucidate the differences between our method and standard methods based on label proportion matching, in terms of applicability and optimality conditions. We showcase the flexibility of our approach compared to alternatives by applying our method to popular learning frameworks, like Empirical Risk Minimization (ERM) and Stochastic Gradient Descent (SGD) with provable guarantees on instance level performance.  Finally, we validate our theoretical results on multiple datasets, empirically illustrating the conditions under which our algorithm is expected to perform better or worse than previous LLP approaches

----

## [656] Jigsaw: Learning to Assemble Multiple Fractured Objects

**Authors**: *Jiaxin Lu, Yifan Sun, Qixing Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30ae2af8612ac74357363e8ae877d80c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30ae2af8612ac74357363e8ae877d80c-Abstract-Conference.html)

**Abstract**:

Automated assembly of 3D fractures is essential in orthopedics, archaeology, and our daily life. This paper presents Jigsaw, a novel framework for assembling physically broken 3D objects from multiple pieces. Our approach leverages hierarchical features of global and local geometry to match and align the fracture surfaces. Our framework consists of four components: (1) front-end point feature extractor with attention layers, (2) surface segmentation to separate fracture and original parts, (3) multi-parts matching to find correspondences among fracture surface points, and (4) robust global alignment to recover the global poses of the pieces. We show how to jointly learn segmentation and matching and seamlessly integrate feature matching and rigidity constraints. We evaluate Jigsaw on the Breaking Bad dataset and achieve superior performance compared to state-of-the-art methods. Our method also generalizes well to diverse fracture modes, objects, and unseen instances. To the best of our knowledge, this is the first learning-based method designed specifically for 3D fracture assembly over multiple pieces. Our code is available at https://jiaxin-lu.github.io/Jigsaw/.

----

## [657] Persuading Farsighted Receivers in MDPs: the Power of Honesty

**Authors**: *Martino Bernasconi, Matteo Castiglioni, Alberto Marchesi, Mirco Mutti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30b28eb87fe7a6c4af8520293317d4c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30b28eb87fe7a6c4af8520293317d4c6-Abstract-Conference.html)

**Abstract**:

Bayesian persuasion studies the problem faced by an informed sender who strategically discloses information to influence the behavior of an uninformed receiver. Recently, a growing attention has been devoted to settings where the sender and the receiver interact sequentially, in which the receiver's decision-making problem is usually modeled as a Markov decision process (MDP). However, the literature focuses on computing optimal information-revelation policies (a.k.a. signaling schemes) under the restrictive assumption that the receiver acts myopically, selecting actions to maximize the one-step utility and disregarding future rewards. This is justified by the fact that, when the receiver is farsighted and thus considers future rewards, finding an optimal Markovian signaling scheme is NP-hard. In this paper, we show that Markovian signaling schemes do not constitute the "right" class of policies. Indeed, differently from most of the MDPs settings, we show that Markovian signaling schemes are not optimal, and general history-dependent signaling schemes should be considered. Moreover, we also show that history-dependent signaling schemes circumvent the negative complexity results affecting Markovian signaling schemes. Formally, we design an algorithm that computes an optimal and $\epsilon$-persuasive history-dependent signaling scheme in time polynomial in ${1}/{\epsilon}$ and in the instance size. The crucial challenge is that general history-dependent signaling schemes cannot be represented in polynomial space. Nevertheless, we introduce a convenient subclass of history-dependent signaling schemes, called promise-form, which are as powerful as general history-dependent ones and efficiently representable. Intuitively, promise-form signaling schemes compactly encode histories in the form of honest promises on future receiver's rewards.

----

## [658] When are ensembles really effective?

**Authors**: *Ryan Theisen, Hyunsuk Kim, Yaoqing Yang, Liam Hodgkinson, Michael W. Mahoney*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30b6fa308e62ed52180c31ae3ba6bb0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30b6fa308e62ed52180c31ae3ba6bb0a-Abstract-Conference.html)

**Abstract**:

Ensembling has a long history in statistical data analysis, with many impactful applications. However, in many modern machine learning settings, the benefits of ensembling are less ubiquitous and less obvious. We study, both theoretically and empirically, the fundamental question of when ensembling yields significant performance improvements in classification tasks. Theoretically, we prove new results relating the \emph{ensemble improvement rate} (a measure of how much ensembling decreases the error rate versus a single model, on a relative scale) to the \emph{disagreement-error ratio}. We show that ensembling improves performance significantly whenever the disagreement rate is large relative to the average error rate; and that, conversely, one classifier is often enough whenever the disagreement rate is low relative to the average error rate. On the way to proving these results, we derive, under a mild condition called \emph{competence}, improved upper and lower bounds on the average test error rate of the majority vote classifier.To complement this theory, we study ensembling empirically in a variety of settings, verifying the predictions made by our theory, and identifying practical scenarios where ensembling does and does not result in large performance improvements. Perhaps most notably, we demonstrate a distinct difference in behavior between interpolating models (popular in current practice) and non-interpolating models (such as tree-based methods, where ensembling is popular), demonstrating that ensembling helps considerably more in the latter case than in the former.

----

## [659] A Unified Approach to Domain Incremental Learning with Memory: Theory and Algorithm

**Authors**: *Haizhou Shi, Hao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30d046e94d7b8037d6ef27c4357a8dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30d046e94d7b8037d6ef27c4357a8dd4-Abstract-Conference.html)

**Abstract**:

Domain incremental learning aims to adapt to a sequence of domains with access to only a small subset of data (i.e., memory) from previous domains. Various methods have been proposed for this problem, but it is still unclear how they are related and when practitioners should choose one method over another. In response, we propose a unified framework, dubbed Unified Domain Incremental Learning (UDIL), for domain incremental learning with memory. Our UDIL unifies various existing methods, and our theoretical analysis shows that UDIL always achieves a tighter generalization error bound compared to these methods. The key insight is that different existing methods correspond to our bound with different fixed coefficients; based on insights from this unification, our UDIL allows adaptive coefficients during training, thereby always achieving the tightest bound. Empirical results show that our UDIL outperforms the state-of-the-art domain incremental learning methods on both synthetic and real-world datasets. Code will be available at https://github.com/Wang-ML-Lab/unified-continual-learning.

----

## [660] Few-Shot Class-Incremental Learning via Training-Free Prototype Calibration

**Authors**: *Qi-Wei Wang, Da-Wei Zhou, Yi-Kai Zhang, De-Chuan Zhan, Han-Jia Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30dfe47a3ccbee68cffa0c19ccb1bc00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30dfe47a3ccbee68cffa0c19ccb1bc00-Abstract-Conference.html)

**Abstract**:

Real-world scenarios are usually accompanied by continuously appearing classes with scare labeled samples, which require the machine learning model to incrementally learn new classes and maintain the knowledge of base classes. In this Few-Shot Class-Incremental Learning (FSCIL) scenario, existing methods either introduce extra learnable components or rely on a frozen feature extractor to mitigate catastrophic forgetting and overfitting problems.  However, we find a tendency for existing methods to misclassify the samples of new classes into base classes, which leads to the poor performance of new classes. In other words, the strong discriminability of base classes distracts the classification of new classes. To figure out this intriguing phenomenon, we observe that although the feature extractor is only trained on base classes, it can surprisingly represent the semantic similarity between the base and unseen new classes. Building upon these analyses, we propose a simple yet effective Training-frEE calibratioN (TEEN) strategy to enhance the discriminability of new classes by fusing the new prototypes (i.e., mean features of a class) with weighted base prototypes. In addition to standard benchmarks in FSCIL, TEEN demonstrates remarkable performance and consistent improvements over baseline methods in the few-shot learning scenario. Code is available at: https://github.com/wangkiw/TEEN

----

## [661] RADAR: Robust AI-Text Detection via Adversarial Learning

**Authors**: *Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/30e15e5941ae0cdab7ef58cc8d59a4ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/30e15e5941ae0cdab7ef58cc8d59a4ca-Abstract-Conference.html)

**Abstract**:

Recent advances in large language models (LLMs) and the intensifying popularity of ChatGPT-like applications have blurred the boundary of high-quality text generation between humans and machines. However, in addition to the anticipated revolutionary changes to our technology and society, the difficulty of distinguishing LLM-generated texts (AI-text) from human-generated texts poses new challenges of misuse and fairness, such as fake content generation, plagiarism, and false accusations of innocent writers. While existing works show that current AI-text detectors are not robust to LLM-based paraphrasing, this paper aims to bridge this gap by proposing a new framework called RADAR, which jointly trains a $\underline{r}$obust  $\underline{A}$I-text  $\underline{d}$etector via  $\underline{a}$dversarial lea$\underline{r}$ning. RADAR is based on adversarial training of a paraphraser and a detector. The paraphraser's goal is to generate realistic content to evade AI-text detection.RADAR uses the feedback from the detector to update the paraphraser, and vice versa.Evaluated with 8 different LLMs (Pythia, Dolly 2.0, Palmyra, Camel, GPT-J, Dolly 1.0, LLaMA, and Vicuna) across 4 datasets, experimental results show that RADAR significantly outperforms existing AI-text detection methods, especially when paraphrasing is in place. We also identify the strong transferability of RADAR from instruction-tuned LLMs to other LLMs, and evaluate the improved capability of RADAR via GPT-3.5-Turbo.

----

## [662] Alleviating the Semantic Gap for Generalized fMRI-to-Image Reconstruction

**Authors**: *Tao Fang, Qian Zheng, Gang Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3106c718fe84b91fc301fe2f5b738448-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3106c718fe84b91fc301fe2f5b738448-Abstract-Conference.html)

**Abstract**:

Although existing fMRI-to-image reconstruction methods could predict high-quality images, they do not explicitly consider the semantic gap between training and testing data, resulting in reconstruction with unstable and uncertain semantics. This paper addresses the problem of generalized fMRI-to-image reconstruction by explicitly alleviates the semantic gap. Specifically, we leverage the pre-trained CLIP model to map the training data to a compact feature representation, which essentially extends the sparse semantics of training data to dense ones, thus alleviating the semantic gap of the instances nearby known concepts (i.e., inside the training super-classes). Inspired by the robust low-level representation in fMRI data, which could help alleviate the semantic gap for instances that far from the known concepts (i.e., outside the training super-classes), we leverage structural information as a general cue to guide image reconstruction. Further, we quantify the semantic uncertainty based on probability density estimation and achieve Generalized fMRI-to-image reconstruction by adaptively integrating Expanded Semantics and Structural information (GESS) within a diffusion process. Experimental results demonstrate that the proposed GESS model outperforms state-of-the-art methods, and we propose a generalized scenario split strategy to evaluate the advantage of GESS in closing the semantic gap.

----

## [663] Softmax Output Approximation for Activation Memory-Efficient Training of Attention-based Networks

**Authors**: *Changhyeon Lee, Seulki Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/311257424b6d80e930fc93b224f0a63e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/311257424b6d80e930fc93b224f0a63e-Abstract-Conference.html)

**Abstract**:

In this paper, we propose to approximate the softmax output, which is the key product of the attention mechanism, to reduce its activation memory usage when training attention-based networks (aka Transformers). During the forward pass of the network, the proposed softmax output approximation method stores only a small fraction of the entire softmax output required for back-propagation and evicts the rest of the softmax output from memory. Then, during the backward pass, the evicted softmax activation output is approximated to compose the gradient to perform back-propagation for model training. Considering most attention-based models heavily rely on the softmax-based attention module that usually takes one of the biggest portions of the network, approximating the softmax activation output can be a simple yet effective way to decrease the training memory requirement of many attention-based networks. The experiment with various attention-based models and relevant tasks, i.e., machine translation, text classification, and sentiment analysis, shows that it curtails the activation memory usage of the softmax-based attention module by up to 84% (6.2Ã— less memory) in model training while achieving comparable or better performance, e.g., up to 5.4% higher classification accuracy.

----

## [664] Data Portraits: Recording Foundation Model Training Data

**Authors**: *Marc Marone, Benjamin Van Durme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3112ee706d21d734c15532c1239773e1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3112ee706d21d734c15532c1239773e1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Foundation models are trained on increasingly immense and opaque datasets. Even while these models are now key in AI system building, it can be  difficult to answer the straightforward question: has the model already encountered a given example during training? We therefore propose a widespread adoption of Data Portraits: artifacts that record training data and allow for downstream inspection. First we outline the properties of such an artifact and discuss how existing solutions can be used to increase transparency. We then propose and implement a solution based on data sketching, stressing fast and space efficient querying. Using our tools, we document a popular language modeling corpus (The Pile) and a recently released code modeling dataset (The Stack). We show that our solution enables answering questions about test set leakage and model plagiarism. Our tool is lightweight and fast, costing only 3% of the dataset size in overhead. We release a live interface of our tools at https://dataportraits.org/ and call on dataset and model creators to release Data Portraits as a complement to current documentation practices.

----

## [665] Memory Efficient Optimizers with 4-bit States

**Authors**: *Bingrui Li, Jianfei Chen, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3122aaa22b2fe83f9cead1a696f65ceb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3122aaa22b2fe83f9cead1a696f65ceb-Abstract-Conference.html)

**Abstract**:

Optimizer states are a major source of memory consumption for training neural networks, limiting the maximum trainable model within given memory budget. Compressing the optimizer states from 32-bit floating points to lower bitwidth is promising to reduce the training memory footprint, while the current lowest achievable bitwidth is 8-bit. In this work, we push optimizer states bitwidth down to 4-bit through a detailed empirical analysis of first and second moments. Specifically, we find that moments have complicated outlier patterns, that current block-wise quantization cannot accurately approximate. We use a smaller block size and propose to utilize both row-wise and column-wise information for better quantization. We further identify a zero point problem of quantizing the second moment, and solve this problem with a linear quantizer that excludes the zero point. Our 4-bit optimizers are evaluated on a wide variety of benchmarks including natural language understanding, machine translation, image classification, and instruction tuning. On all the tasks our optimizers can achieve comparable accuracy with their full-precision counterparts, while enjoying better memory efficiency.

----

## [666] Replicable Reinforcement Learning

**Authors**: *Eric Eaton, Marcel Hussing, Michael Kearns, Jessica Sorrell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/313829757739365201b5adb3a1cbd9bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/313829757739365201b5adb3a1cbd9bd-Abstract-Conference.html)

**Abstract**:

The replicability crisis in the social, behavioral, and data sciences has led to the formulation of algorithm frameworks for replicability --- i.e., a requirement that an algorithm produce identical outputs (with high probability) when run on two different samples from the same underlying distribution. While still in its infancy, provably replicable algorithms have been developed for many fundamental tasks in machine learning and statistics, including statistical query learning, the heavy hitters problem, and distribution testing. In this work we initiate the study of replicable reinforcement learning, providing a provably replicable algorithm for parallel value iteration, and a provably replicable version of R-Max in the episodic setting. These are the first formal replicability results for control problems, which present different challenges for replication than batch learning settings.

----

## [667] Make Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning

**Authors**: *Baohao Liao, Shaomu Tan, Christof Monz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3151e460c41ba67dc55412861184ef35-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3151e460c41ba67dc55412861184ef35-Abstract-Conference.html)

**Abstract**:

Parameter-efficient fine-tuning (PEFT) of pre-trained language models (PLMs) has emerged as a highly successful approach, with training only a small number of parameters without sacrificing performance and becoming the de-facto learning paradigm with the increasing size of PLMs. However, existing PEFT methods are not memory-efficient, because they still require caching most of the intermediate activations for the gradient calculation, akin to fine-tuning. One effective way to reduce the activation memory is to apply a reversible model, so the intermediate activations are not necessary to be cached and can be recomputed. Nevertheless, modifying a PLM to its reversible variant is not straightforward, since the reversible model has a distinct architecture from the currently released PLMs. In this paper, we first investigate what is a key factor for the success of existing PEFT methods, and realize that it's essential to preserve the PLM's starting point when initializing a PEFT method. With this finding, we propose memory-efficient fine-tuning (MEFT) that inserts adapters into a PLM, preserving the PLM's starting point and making it reversible without additional pre-training. We evaluate MEFT on the GLUE benchmark and five question-answering tasks with various backbones, BERT, RoBERTa, BART and OPT. MEFT significantly reduces the activation memory up to 84% of full fine-tuning with a negligible amount of trainable parameters. Moreover, MEFT achieves the same score on GLUE and a comparable score on the question-answering tasks as full fine-tuning. A similar finding is also observed for the image classification task.

----

## [668] Design from Policies: Conservative Test-Time Adaptation for Offline Policy Optimization

**Authors**: *Jinxin Liu, Hongyin Zhang, Zifeng Zhuang, Yachen Kang, Donglin Wang, Bin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31610e68fe41a62e460e044216a10766-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31610e68fe41a62e460e044216a10766-Abstract-Conference.html)

**Abstract**:

In this work, we decouple the iterative bi-level offline RL (value estimation and policy extraction) from the offline training phase, forming a non-iterative bi-level paradigm and avoiding the iterative error propagation over two levels. Specifically, this non-iterative paradigm allows us to conduct inner-level optimization (value estimation) in training, while performing outer-level optimization (policy extraction) in testing. Naturally, such a paradigm raises three core questions that are not fully answered by prior non-iterative offline RL counterparts like reward-conditioned policy: (q1) What information should we transfer from the inner-level to the outer-level? (q2) What should we pay attention to when exploiting the transferred information for safe/confident outer-level optimization? (q3) What are the benefits of concurrently conducting outer-level optimization during testing? Motivated by model-based optimization (MBO), we propose DROP (design from policies), which fully answers the above questions. Specifically, in the inner-level, DROP decomposes offline data into multiple subsets, and learns an MBO score model (a1). To keep safe exploitation to the score model in the outer-level, we explicitly learn a behavior embedding and introduce a conservative regularization (a2). During testing, we show that DROP permits deployment adaptation, enabling an adaptive inference across states (a3). Empirically, we evaluate DROP on various tasks, showing that DROP gains comparable or better performance compared to prior methods.

----

## [669] Lightweight Vision Transformer with Bidirectional Interaction

**Authors**: *Qihang Fan, Huaibo Huang, Xiaoqiang Zhou, Ran He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3170de57bc1899315b97712043d8bb22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3170de57bc1899315b97712043d8bb22-Abstract-Conference.html)

**Abstract**:

Recent advancements in vision backbones have significantly improved their performance by simultaneously modeling imagesâ€™ local and global contexts. However, the bidirectional interaction between these two contexts has not been well explored and exploited, which is important in the human visual system. This paper proposes a Fully Adaptive Self-Attention (FASA) mechanism for vision transformer to model the local and global information as well as the bidirectional interaction between them in context-aware ways. Specifically, FASA employs self-modulated convolutions to adaptively extract local representation while utilizing self-attention in down-sampled space to extract global representation. Subsequently, it conducts a bidirectional adaptation process between local and global representation to model their interaction. In addition, we introduce a fine-grained downsampling strategy to enhance the down-sampled self-attention mechanism for finer-grained global perception capability. Based on FASA, we develop a family of lightweight vision backbones, Fully Adaptive Transformer (FAT) family. Extensive experiments on multiple vision tasks demonstrate that FAT achieves impressive performance. Notably, FAT accomplishes a 77.6% accuracy on ImageNet-1K using only 4.5M parameters and 0.7G FLOPs, which surpasses the most advanced ConvNets and Transformers with similar model size and computational costs. Moreover, our model exhibits faster speed on modern GPU compared to other models.

----

## [670] Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications

**Authors**: *Xinyu Ma, Xu Chu, Yasha Wang, Yang Lin, Junfeng Zhao, Liantao Ma, Wenwu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3173c427cb4ed2d5eaab029c17f221ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3173c427cb4ed2d5eaab029c17f221ae-Abstract-Conference.html)

**Abstract**:

Graph data augmentation has shown superiority in enhancing generalizability and robustness of GNNs in graph-level classifications. However, existing methods primarily focus on the augmentation in the graph signal space and the graph structure space independently, neglecting the joint interaction between them. In this paper, we address this limitation by formulating the problem as an optimal transport problem that aims to find an optimal inter-graph node matching strategy considering the interactions between graph structures and signals. To solve this problem, we propose a novel graph mixup algorithm called FGWMixup, which seeks a "midpoint" of source graphs in the Fused Gromov-Wasserstein (FGW) metric space. To enhance the scalability of our method, we introduce a relaxed FGW solver that accelerates FGWMixup by improving the convergence rate from $\mathcal{O}(t^{-1})$ to $\mathcal{O}(t^{-2})$. Extensive experiments conducted on five datasets using both classic (MPNNs) and advanced (Graphormers) GNN backbones demonstrate that \mname\xspace effectively improves the generalizability and robustness of GNNs. Codes are available at https://github.com/ArthurLeoM/FGWMixup.

----

## [671] Modeling Dynamics over Meshes with Gauge Equivariant Nonlinear Message Passing

**Authors**: *Jung Yeon Park, Lawson L. S. Wong, Robin Walters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/317470b3fde29f3bb8d6dee563afffc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/317470b3fde29f3bb8d6dee563afffc4-Abstract-Conference.html)

**Abstract**:

Data over non-Euclidean manifolds, often discretized as surface meshes, naturally arise in computer graphics and biological and physical systems. In particular, solutions to partial differential equations (PDEs) over manifolds depend critically on the underlying geometry. While graph neural networks have been successfully applied to PDEs, they do not incorporate surface geometry and do not consider local gauge symmetries of the manifold. Alternatively, recent works on gauge equivariant convolutional and attentional architectures on meshes leverage the underlying geometry but underperform in modeling surface PDEs with complex nonlinear dynamics. To address these issues, we introduce a new gauge equivariant architecture using nonlinear message passing. Our novel architecture achieves higher performance than either convolutional or attentional networks on domains with highly complex and nonlinear dynamics. However, similar to the non-mesh case, design trade-offs favor convolutional, attentional, or message passing networks for different tasks; we investigate in which circumstances our message passing method provides the most benefit.

----

## [672] ASIF: Coupled Data Turns Unimodal Models to Multimodal without Training

**Authors**: *Antonio Norelli, Marco Fumero, Valentino Maiorca, Luca Moschella, Emanuele Rodolà, Francesco Locatello*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3186591903d9db31770ad131adb5ceb4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3186591903d9db31770ad131adb5ceb4-Abstract-Conference.html)

**Abstract**:

CLIP proved that aligning visual and language spaces is key to solving many vision tasks without explicit training, but required to train image and text encoders from scratch on a huge dataset. LiT improved this by only training the text encoder and using a pre-trained vision network. In this paper, we show that a common space can be created without any training at all, using single-domain encoders (trained with or without supervision) and a much smaller amount of image-text pairs. Furthermore, our model has unique properties. Most notably, deploying a new version with updated training samples can be done in a matter of seconds. Additionally, the representations in the common space are easily interpretable as every dimension corresponds to the similarity of the input to a unique entry in the multimodal dataset. Experiments on standard zero-shot visual benchmarks demonstrate the typical transfer ability of image-text models. Overall, our method represents a simple yet surprisingly strong baseline for foundation multi-modal models, raising important questions on their data efficiency and on the role of retrieval in machine learning.

----

## [673] A Metadata-Driven Approach to Understand Graph Neural Networks

**Authors**: *Ting Wei Li, Qiaozhu Mei, Jiaqi Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31994923f58ae5b2d661b300bd439107-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31994923f58ae5b2d661b300bd439107-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have achieved remarkable success in various applications, but their performance can be sensitive to specific data properties of the graph datasets they operate on. Current literature on understanding the limitations of GNNs has primarily employed a \emph{model-driven} approach that leverage heuristics and domain knowledge from network science or graph theory to model the GNN behaviors, which is time-consuming and highly subjective. In this work, we propose a \emph{metadata-driven} approach to analyze the sensitivity of GNNs to graph data properties, motivated by the increasing availability of graph learning benchmarks. We perform a multivariate sparse regression analysis on the metadata derived from benchmarking GNN performance across diverse datasets, yielding a set of salient data properties. To validate the effectiveness of our data-driven approach, we focus on one identified data property, the degree distribution, and investigate how this property influences GNN performance through theoretical analysis and controlled experiments. Our theoretical findings reveal that datasets with more balanced degree distribution exhibit better linear separability of node representations, thus leading to better GNN performance. We also conduct controlled experiments using synthetic datasets with varying degree distributions, and the results align well with our theoretical findings. Collectively, both the theoretical analysis and controlled experiments verify that the proposed metadata-driven approach is effective in identifying critical data properties for GNNs.

----

## [674] Multimodal Deep Learning Model Unveils Behavioral Dynamics of V1 Activity in Freely Moving Mice

**Authors**: *Aiwen Xu, Yuchen Hou, Cristopher Niell, Michael Beyeler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31a19921acd38cdf7a8c86ec032cef2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31a19921acd38cdf7a8c86ec032cef2d-Abstract-Conference.html)

**Abstract**:

Despite their immense success as a model of macaque visual cortex, deep convolutional neural networks (CNNs) have struggled to predict activity in visual cortex of the mouse, which is thought to be strongly dependent on the animalâ€™s behavioral state. Furthermore, most computational models focus on predicting neural responses to static images presented under head fixation, which are dramatically different from the dynamic, continuous visual stimuli that arise during movement in the real world. Consequently, it is still unknown how natural visual input and different behavioral variables may integrate over time to generate responses in primary visual cortex (V1). To address this, we introduce a multimodal recurrent neural network that integrates gaze-contingent visual input with behavioral and temporal dynamics to explain V1 activity in freely moving mice. We show that the model achieves state-of-the-art predictions of V1 activity during free exploration and demonstrate the importance of each component in an extensive ablation study. Analyzing our model using maximally activating stimuli and saliency maps, we reveal new insights into cortical function, including the prevalence of mixed selectivity for behavioral variables in mouse V1. In summary, our model offers a comprehensive deep-learning framework for exploring the computational principles underlying V1 neurons in freely-moving animals engaged in natural behavior.

----

## [675] Goal-conditioned Offline Planning from Curious Exploration

**Authors**: *Marco Bagatella, Georg Martius*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31ceb5aed43e2ec1b132e389cc1dcb56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31ceb5aed43e2ec1b132e389cc1dcb56-Abstract-Conference.html)

**Abstract**:

Curiosity has established itself as a powerful exploration strategy in deep reinforcement learning. Notably, leveraging expected future novelty as intrinsic motivation has been shown to efficiently generate exploratory trajectories, as well as a robust dynamics model. We consider the challenge of extracting goal-conditioned behavior from the products of such unsupervised exploration techniques, without any additional environment interaction. We find that conventional goal-conditioned reinforcement learning approaches for extracting a value function and policy fall short in this difficult offline setting. By analyzing the geometry of optimal goal-conditioned value functions, we relate this issue to a specific class of estimation artifacts in learned values. In order to mitigate their occurrence, we propose to combine model-based planning over learned value landscapes with a graph-based value aggregation scheme. We show how this combination can correct both local and global artifacts, obtaining significant improvements in zero-shot goal-reaching performance across diverse simulated environments.

----

## [676] Rethinking the Role of Token Retrieval in Multi-Vector Retrieval

**Authors**: *Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, Vincent Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31d997278ee9069d6721bc194174bb4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31d997278ee9069d6721bc194174bb4c-Abstract-Conference.html)

**Abstract**:

Multi-vector retrieval models such as ColBERT [Khattab et al., 2020] allow token-level interactions between queries and documents, and hence achieve state of the art on many information retrieval benchmarks. However, their non-linear scoring function cannot be scaled to millions of documents, necessitating a three-stage process for inference: retrieving initial candidates via token retrieval, accessing all token vectors, and scoring the initial candidate documents. The non-linear scoring function is applied over all token vectors of each candidate document, making the inference process complicated and slow. In this paper, we aim to simplify the multi-vector retrieval by rethinking the role of token retrieval. We present XTR, ConteXtualized Token Retriever, which introduces a simple, yet novel, objective function that encourages the model to retrieve the most important document tokens first. The improvement to token retrieval allows XTR to rank candidates only using the retrieved tokens rather than all tokens in the document, and enables a newly designed scoring stage that is two-to-three orders of magnitude cheaper than that of ColBERT. On the popular BEIR benchmark, XTR advances the state-of-the-art by 2.8 nDCG@10 without any distillation. Detailed analysis confirms our decision to revisit the token retrieval stage, as XTR demonstrates much better recall of the token retrieval stage compared to ColBERT.

----

## [677] Optimal Exploration for Model-Based RL in Nonlinear Systems

**Authors**: *Andrew Wagenmaker, Guanya Shi, Kevin G. Jamieson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31e018f43ab9c7065c058cc2c5848128-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31e018f43ab9c7065c058cc2c5848128-Abstract-Conference.html)

**Abstract**:

Learning to control unknown nonlinear dynamical systems is a fundamental problem in reinforcement learning and control theory. A commonly applied approach is to first explore the environment (exploration), learn an accurate model of it (system identification), and then compute an optimal controller with the minimum cost on this estimated system (policy optimization). While existing work has shown that it is possible to learn a uniformly good model of the system (Mania et al., 2020), in practice, if we aim to learn a good controller with a low cost on the actual system, certain system parameters may be significantly more critical than others, and we therefore ought to focus our exploration on learning such parameters.In this work, we consider the setting of nonlinear dynamical systems and seek to formally quantify, in such settings, (a) which parameters are most relevant to learning a good controller, and (b) how we can best explore so as to minimize uncertainty in such parameters. Inspired by recent work in linear systems (Wagenmaker et al., 2021), we show that minimizing the controller loss in nonlinear systems translates to estimating the system parameters in a particular, task-dependent metric. Motivated by this, we develop an algorithm able to efficiently explore the system to reduce uncertainty in this metric, and prove a lower bound showing that our approach learns a controller at a near-instance-optimal rate. Our algorithm relies on a general reduction from policy optimization to optimal experiment design in arbitrary systems, and may be of independent interest. We conclude with experiments demonstrating the effectiveness of our method in realistic nonlinear robotic systems.

----

## [678] ELDEN: Exploration via Local Dependencies

**Authors**: *Zizhao Wang, Jiaheng Hu, Peter Stone, Roberto Martín-Martín*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31ed129feae64a7e44a15b148c15558d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31ed129feae64a7e44a15b148c15558d-Abstract-Conference.html)

**Abstract**:

Tasks with large state space and sparse rewards present a longstanding challenge to reinforcement learning. In these tasks, an agent needs to explore the state space efficiently until it finds a reward. To deal with this problem, the community has proposed to augment the reward function with intrinsic reward, a bonus signal that encourages the agent to visit interesting states. In this work, we propose a new way of defining interesting states for environments with factored state spaces and complex chained dependencies, where an agent's actions may change the value of one entity that, in order, may affect the value of another entity. Our insight is that, in these environments, interesting states for exploration are states where the agent is uncertain whether (as opposed to how) entities such as the agent or objects have some influence on each other. We present ELDEN, Exploration via Local DepENdencies, a novel intrinsic reward that encourages the discovery of new interactions between entities. ELDEN utilizes a novel scheme --- the partial derivative of the learned dynamics to model the local dependencies between entities accurately and computationally efficiently. The uncertainty of the predicted dependencies is then used as an intrinsic reward to encourage exploration toward new interactions. We evaluate the performance of ELDEN on four different domains with complex dependencies, ranging from 2D grid worlds to 3D robotic tasks. In all domains, ELDEN correctly identifies local dependencies and learns successful policies, significantly outperforming previous state-of-the-art exploration methods.

----

## [679] Maximization of Average Precision for Deep Learning with Adversarial Ranking Robustness

**Authors**: *Gang Li, Wei Tong, Tianbao Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31f04c174a6af322e9417b7a9a91097a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31f04c174a6af322e9417b7a9a91097a-Abstract-Conference.html)

**Abstract**:

This paper seeks to address a gap in optimizing Average Precision (AP) while ensuring adversarial robustness, an area that has not been extensively explored to the best of our knowledge. AP maximization for deep learning has widespread applications, particularly when there is a significant imbalance between positive and negative examples. Although numerous studies have been conducted on adversarial training, they primarily focus on robustness concerning accuracy, ensuring that the average accuracy on adversarially perturbed examples is well maintained. However, this type of adversarial robustness is insufficient for many applications, as minor perturbations on a single example can significantly impact AP  while not greatly influencing the accuracy of the prediction system. To tackle this issue, we introduce a novel formulation that combines an AP surrogate loss with a regularization term representing adversarial ranking robustness, which maintains the consistency between ranking of clean data and that of perturbed data. We then devise an efficient stochastic optimization algorithm to optimize the resulting objective. Our empirical studies, which compare our method to current leading adversarial training baselines and other robust AP maximization strategies, demonstrate the effectiveness of the proposed approach. Notably, our methods outperform a state-of-the-art method (TRADES) by more than 4\% in terms of robust AP against PGD attacks while achieving 7\% higher AP on clean data simultaneously on CIFAR10 and CIFAR100.The code is available at: https://github.com/GangLii/Adversarial-AP

----

## [680] Act As You Wish: Fine-Grained Control of Motion Diffusion Model with Hierarchical Semantic Graphs

**Authors**: *Peng Jin, Yang Wu, Yanbo Fan, Zhongqian Sun, Wei Yang, Li Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/31fc85f7461ce71eadf27fb7281973bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/31fc85f7461ce71eadf27fb7281973bd-Abstract-Conference.html)

**Abstract**:

Most text-driven human motion generation methods employ sequential modeling approaches, e.g., transformer, to extract sentence-level text representations automatically and implicitly for human motion synthesis. However, these compact text representations may overemphasize the action names at the expense of other important properties and lack fine-grained details to guide the synthesis of subtly distinct motion. In this paper, we propose hierarchical semantic graphs for fine-grained control over motion generation. Specifically, we disentangle motion descriptions into hierarchical semantic graphs including three levels of motions, actions, and specifics. Such global-to-local structures facilitate a comprehensive understanding of motion description and fine-grained control of motion generation. Correspondingly, to leverage the coarse-to-fine topology of hierarchical semantic graphs, we decompose the text-to-motion diffusion process into three semantic levels, which correspond to capturing the overall motion, local actions, and action specifics. Extensive experiments on two benchmark human motion datasets, including HumanML3D and KIT, with superior performances, justify the efficacy of our method. More encouragingly, by modifying the edge weights of hierarchical semantic graphs, our method can continuously refine the generated motion, which may have a far-reaching impact on the community. Code and pre-trained weights are available at https://github.com/jpthu17/GraphMotion.

----

## [681] DynaDojo: An Extensible Platform for Benchmarking Scaling in Dynamical System Identification

**Authors**: *Logan M. Bhamidipaty, Tommy Bruzzese, Caryn Tran, Rami Ratl Mrad, Maxinder S. Kanwal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32093649cbbcff773d9a991d8c30a7fe-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/32093649cbbcff773d9a991d8c30a7fe-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Modeling complex dynamical systems poses significant challenges, with traditional methods struggling to work on a variety of systems and scale to high-dimensional dynamics. In response, we present DynaDojo, a novel benchmarking platform designed for data-driven dynamical system identification. DynaDojo provides diagnostics on three ways an algorithm’s performance scales: across the number of training samples, the complexity of a dynamical system, and a target error to achieve. Furthermore, DynaDojo enables studying out-of-distribution generalization (by providing unique test conditions for each system) and active learning (by supporting closed-loop control). Through its user-friendly and easily extensible API, DynaDojo accommodates a wide range of user-defined \texttt{Algorithms}, \texttt{Systems}, and \texttt{Challenges} (evaluation metrics). The platform also prioritizes resource-efficient training with parallel processing strategies for running on a cluster. To showcase its utility, in DynaDojo 0.9, we include implementations of 7 baseline algorithms and 20 dynamical systems, along with several demos exhibiting insights researchers can glean using our platform. This work aspires to make DynaDojo a unifying benchmarking platform for system identification, paralleling the role of OpenAI’s Gym in reinforcement learning.

----

## [682] Online Constrained Meta-Learning: Provable Guarantees for Generalization

**Authors**: *Siyuan Xu, Minghui Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/320e941f53db45bddc8757d1c8c4f6aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/320e941f53db45bddc8757d1c8c4f6aa-Abstract-Conference.html)

**Abstract**:

Meta-learning has attracted attention due to its strong ability to learn experiences from known tasks, which can speed up and enhance the learning process for new tasks. However, most existing meta-learning approaches only can learn from tasks without any constraint. This paper proposes an online constrained meta-learning framework, which continuously learns meta-knowledge from sequential learning tasks, and the learning tasks are subject to hard constraints. Beyond existing meta-learning analyses, we provide the upper bounds of optimality gaps and constraint violations produced by the proposed framework, which considers the dynamic regret of online learning, as well as the generalization ability of the task-specific models. Moreover, we provide a practical algorithm for the framework, and validate its superior effectiveness through experiments conducted on meta-imitation learning and few-shot image classification.

----

## [683] Mean-field Langevin dynamics: Time-space discretization, stochastic gradient, and variance reduction

**Authors**: *Taiji Suzuki, Denny Wu, Atsushi Nitanda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32133a6a24d6554263d3584e3ac10faa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32133a6a24d6554263d3584e3ac10faa-Abstract-Conference.html)

**Abstract**:

The mean-field Langevin dynamics (MFLD) is a nonlinear generalization of the Langevin dynamics that incorporates a distribution-dependent drift, and it naturally arises from the optimization of two-layer neural networks via (noisy) gradient descent. Recent works have shown that MFLD globally minimizes an entropy-regularized convex functional in the space of measures. However, all prior analyses assumed the infinite-particle or continuous-time limit, and cannot handle stochastic gradient updates. We provide a general framework to prove a uniform-in-time propagation of chaos for MFLD that takes into account the errors due to finite-particle approximation, time-discretization, and stochastic gradient. To demonstrate the wide applicability of our framework, we establish quantitative convergence rate guarantees to the regularized global optimal solution for $(i)$ a wide range of learning problems such as mean-field neural network and MMD minimization, and $(ii)$ different gradient estimators including SGD and SVRG. Despite the generality of our results, we achieve an improved convergence rate in both the SGD and SVRG settings when specialized to the standard Langevin dynamics.

----

## [684] Public Opinion Field Effect Fusion in Representation Learning for Trending Topics Diffusion

**Authors**: *Junliang Li, Yajun Yang, Qinghua Hu, Xin Wang, Hong Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32246544c237164c365c0527b677a79a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32246544c237164c365c0527b677a79a-Abstract-Conference.html)

**Abstract**:

Trending topic diffusion and prediction analysis is an important problem and has been well studied in social networks. Representation learning is an effective way to extract node embeddings, which can help for topic propagation analysis by completing downstream tasks such as link prediction and node classification. In real world, there are often several trending topics or opinion leaders in public opinion space at the same time and they can be regarded as different centers of public opinion. A public opinion field will be formed surrounding every center. These public opinion fields compete for public's attention and it will potentially affect the development of public opinion. However, the existing methods do not consider public opinion field effect for trending topics diffusion. In this paper, we introduce three well-known observations about public opinion field effect in media and communication studies, and propose a novel and effective heterogeneous representation learning framework to incorporate public opinion field effect and social circle influence effect. To the best of our knowledge, our work is the first to consider these effects in representation learning for trending topic diffusion. Extensive experiments on real-world datasets validate the superiority of our model.

----

## [685] Distributional Pareto-Optimal Multi-Objective Reinforcement Learning

**Authors**: *Xin-Qiang Cai, Pushi Zhang, Li Zhao, Jiang Bian, Masashi Sugiyama, Ashley Llorens*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html)

**Abstract**:

Multi-objective reinforcement learning (MORL) has been proposed to learn control policies over multiple competing objectives with each possible preference over returns. However, current MORL algorithms fail to account for distributional preferences over the multi-variate returns, which are particularly important in real-world scenarios such as autonomous driving. To address this issue, we extend the concept of Pareto-optimality in MORL into distributional Pareto-optimality, which captures the optimality of return distributions, rather than the expectations. Our proposed method, called Distributional Pareto-Optimal Multi-Objective Reinforcement Learning~(DPMORL), is capable of learning distributional Pareto-optimal policies that balance multiple objectives while considering the return uncertainty. We evaluated our method on several benchmark problems and demonstrated its effectiveness in discovering distributional Pareto-optimal policies and satisfying diverse distributional preferences compared to existing MORL methods.

----

## [686] Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning

**Authors**: *Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, William Yang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3255a7554605a88800f4e120b3a929e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3255a7554605a88800f4e120b3a929e1-Abstract-Conference.html)

**Abstract**:

In recent years, pre-trained large language models (LLMs) have demonstrated remarkable efficiency in achieving an inference-time few-shot learning capability known as in-context learning. However, existing literature has highlighted the sensitivity of this capability to the selection of few-shot demonstrations. Current understandings of the underlying mechanisms by which this capability arises from regular language model pretraining objectives remain disconnected from the real-world LLMs. This study aims to examine the in-context learning phenomenon through a Bayesian lens, viewing real-world LLMs as latent variable models. On this premise, we propose an algorithm to select optimal demonstrations from a set of annotated data with a small LM, and then directly generalize the selected demonstrations to larger LMs. We demonstrate significant improvement over baselines, averaged over eight GPT models on eight real-world text classification datasets. We also demonstrate the real-world usefulness of our algorithm on GSM8K, a math word problem dataset. Our empirical findings support our hypothesis that LLMs implicitly infer a latent variable containing task information.

----

## [687] Physics-Driven ML-Based Modelling for Correcting Inverse Estimation

**Authors**: *Ruiyuan Kang, Tingting Mu, Panagiotis Liatsis, Dimitrios C. Kyritsis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3268353cd4ff87451347f242c7401773-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3268353cd4ff87451347f242c7401773-Abstract-Conference.html)

**Abstract**:

When deploying machine learning  estimators in science and engineering (SAE) domains, it is critical  to avoid failed estimations that can have disastrous consequences, e.g., in aero engine design. This work focuses on detecting and correcting  failed  state estimations before adopting them in SAE inverse problems, by  utilizing simulations and performance metrics guided by physical laws. We suggest to flag a machine learning estimation when its physical model error exceeds a feasible threshold, and propose a novel approach, GEESE, to correct it  through optimization, aiming at delivering both low error and high efficiency. The key designs of GEESE include (1) a hybrid surrogate error model to  provide fast  error estimations  to reduce simulation cost and to enable gradient based backpropagation of error feedback, and (2) two generative models to approximate the probability distributions of the candidate states for simulating the  exploitation and exploration behaviours. All three models are constructed as neural networks. GEESE is tested on three real-world SAE inverse problems and compared to a number of state-of-the-art optimization/search approaches. Results show that it fails the least number of times in terms of finding a feasible state correction, and requires physical evaluations less frequently in general.

----

## [688] One-Pass Distribution Sketch for Measuring Data Heterogeneity in Federated Learning

**Authors**: *Zichang Liu, Zhaozhuo Xu, Benjamin Coleman, Anshumali Shrivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32c2f3e0a44d55820da7fbcee0a1d95c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32c2f3e0a44d55820da7fbcee0a1d95c-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is a machine learning paradigm where multiple client devices train models collaboratively without data exchange. Data heterogeneity problem is naturally inherited in FL since data in different clients follow diverse distributions.  To mitigate the negative influence of data heterogeneity, we need to start by measuring it across clients. However, the efficient measurement between distributions is a challenging problem, especially in high dimensionality. In this paper, we propose a one-pass distribution sketch to represent the client data distribution. Our sketching algorithm only requires a single pass of the client data, which is efficient in terms of time and memory. Moreover, we show in both theory and practice that the distance between two distribution sketches represents the divergence between their corresponding distributions. Furthermore, we demonstrate with extensive experiments that our distribution sketch improves the client selection in the FL training. We also showcase that our distribution sketch is an efficient solution to the cold start problem in FL for new clients with unlabeled data.

----

## [689] Kernel-Based Tests for Likelihood-Free Hypothesis Testing

**Authors**: *Patrik Róbert Gerber, Tianze Jiang, Yury Polyanskiy, Rui Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32c6d65ec2591dfcfb3f0e345a51f585-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32c6d65ec2591dfcfb3f0e345a51f585-Abstract-Conference.html)

**Abstract**:

Given $n$ observations from two balanced classes, consider the task of labeling an additional $m$ inputs that are known to all belong to \emph{one} of the two classes. Special cases of this problem are well-known: with completeknowledge of class distributions ($n=\infty$) theproblem is solved optimally by the likelihood-ratio test; when$m=1$ it corresponds to binary classification; and when $m\approx n$ it is equivalent to two-sample testing. The intermediate settings occur in the field of likelihood-free inference, where labeled samples are obtained by running forward simulations and the unlabeled sample is collected experimentally. In recent work it was discovered that there is a fundamental trade-offbetween $m$ and $n$: increasing the data sample $m$ reduces the amount $n$ of training/simulationdata needed. In this work we (a) introduce a generalization where unlabeled samples come from a mixture of the two classes -- a case often encountered in practice; (b) study the minimax sample complexity for non-parametric classes of densities under \textit{maximum meandiscrepancy} (MMD) separation; and (c) investigate the empirical performance of kernels parameterized by neural networks on two tasks: detectionof the Higgs boson and detection of planted DDPM generated images amidstCIFAR-10 images. For both problems we confirm the existence of the theoretically predicted asymmetric $m$ vs $n$ trade-off.

----

## [690] Deep Equilibrium Based Neural Operators for Steady-State PDEs

**Authors**: *Tanya Marwah, Ashwini Pokle, J. Zico Kolter, Zachary C. Lipton, Jianfeng Lu, Andrej Risteski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32cc61322f1e2f56f989d29ccc7cfbb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32cc61322f1e2f56f989d29ccc7cfbb7-Abstract-Conference.html)

**Abstract**:

Data-driven machine learning approaches are being increasingly used to solve partial differential equations (PDEs). They have shown particularly striking successes when training an operator, which takes as input a PDE in some family, and outputs its solution. However, the architectural design space, especially given structural knowledge of the PDE family of interest, is still poorly understood. We seek to remedy this gap by studying the benefits of weight-tied neural network architectures for steady-state PDEs. To achieve this, we first demonstrate that the solution of most steady-state PDEs can be expressed as a fixed point of a non-linear operator. Motivated by this observation, we propose FNO-DEQ, a deep equilibrium variant of the FNO architecture that directly solves for the solution of a steady-state PDE as the infinite-depth fixed point of an implicit operator layer using a black-box root solver and differentiates analytically through this fixed point resulting in $\mathcal{O}(1)$ training memory. Our experiments indicate that FNO-DEQ-based architectures outperform FNO-based baselines with $4\times$ the number of parameters in predicting the solution to steady-state PDEs such as Darcy Flow and steady-state incompressible Navier-Stokes. Finally, we show FNO-DEQ is more robust when trained with datasets with more noisy observations than the FNO-based baselines, demonstrating the benefits of using appropriate inductive biases in architectural design for different neural network based PDE solvers. Further, we show a universal approximation result that demonstrates that FNO-DEQ can approximate the solution to any steady-state PDE that can be written as a fixed point equation.

----

## [691] An Efficient and Robust Framework for Approximate Nearest Neighbor Search with Attribute Constraint

**Authors**: *Mengzhao Wang, Lingwei Lv, Xiaoliang Xu, Yuxiang Wang, Qiang Yue, Jiongkang Ni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32e41d6b0a51a63a9a90697da19d235d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32e41d6b0a51a63a9a90697da19d235d-Abstract-Conference.html)

**Abstract**:

This paper introduces an efficient and robust framework for hybrid query (HQ) processing, which combines approximate nearest neighbor search (ANNS) with attribute constraint. HQ aims to find objects that are similar to a feature vector and match some structured attributes. Existing methods handle ANNS and attribute filtering separately, leading to inefficiency and inaccuracy. Our framework, called native hybrid query (NHQ), builds a composite index based on proximity graph (PG) and applies joint pruning for HQ. We can easily adapt existing PGs to this framework for efficient HQ processing. We also propose two new navigable PGs (NPGs) with optimized edge selection and routing, which improve the overall ANNS performance. We implement five HQ methods based on the proposed NPGs and existing PGs in NHQ, and show that they outperform the state-of-the-art methods on 10 real-world datasets (up to 315$\times$ faster with the same accuracy).

----

## [692] Parameter-efficient Tuning of Large-scale Multimodal Foundation Model

**Authors**: *Haixin Wang, Xinlong Yang, Jianlong Chang, Dian Jin, Jinan Sun, Shikun Zhang, Xiao Luo, Qi Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32ebb6b560ee58abbdae834e5f37cb5d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32ebb6b560ee58abbdae834e5f37cb5d-Abstract-Conference.html)

**Abstract**:

Driven by the progress of large-scale pre-training, parameter-efficient transfer learning has gained immense popularity across different subfields of Artificial Intelligence. The core is to adapt the model to downstream tasks with only a small set of parameters. Recently, researchers have leveraged such proven techniques in multimodal tasks and achieve promising results. However, two critical issues remain unresolved: how to further reduce the complexity with lightweight design and how to boost alignment between modalities under extremely low parameters. In this paper, we propose A gracefUl pRompt framewOrk for cRoss-modal trAnsfer (AURORA) to overcome these challenges. Considering the redundancy in existing architectures, we first utilize the mode approximation to generate 0.1M trainable parameters to implement the multimodal parameter-efficient tuning, which explores the low intrinsic dimension with only 0.04% parameters of the pre-trained model. Then, for better modality alignment, we propose the Informative Context Enhancement and Gated Query Transformation module under extremely few parameters scenes. A thorough evaluation on six cross-modal benchmarks shows that it not only outperforms the state-of-the-art but even outperforms the full fine-tuning approach. Our code is available at: https://github.com/WillDreamer/Aurora.

----

## [693] Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective

**Authors**: *Yifei Wang, Liangchen Li, Jiansheng Yang, Zhouchen Lin, Yisen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/32f9049217da6e718a426b07242dff73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/32f9049217da6e718a426b07242dff73-Abstract-Conference.html)

**Abstract**:

Adversarial Training (AT) has become arguably the state-of-the-art algorithm for extracting robust features. However, researchers recently notice that AT suffers from severe robust overfitting problems, particularly after learning rate (LR) decay. In this paper, we explain this phenomenon by viewing adversarial training as a dynamic minimax game between the model trainer and the attacker. Specifically, we analyze how LR decay breaks the balance between the minimax game by empowering the trainer with a stronger memorization ability, and show such imbalance induces robust overfitting as a result of memorizing non-robust features. We validate this understanding with extensive experiments, and provide a holistic view of robust overfitting from the dynamics of both the two game players. This understanding further inspires us to alleviate robust overfitting by rebalancing the two players by either regularizing the trainer's capacity or improving the attack strength. Experiments show that the proposed ReBalanced Adversarial Training (ReBAT) can attain good robustness and does not suffer from robust overfitting even after very long training. Code is available at https://github.com/PKU-ML/ReBAT.

----

## [694] Should I Stop or Should I Go: Early Stopping with Heterogeneous Populations

**Authors**: *Hammaad Adam, Fan Yin, Huibin Hu, Neil A. Tenenholtz, Lorin Crawford, Lester Mackey, Allison Koenecke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3322a9a72a1707de14badd5e552ff466-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3322a9a72a1707de14badd5e552ff466-Abstract-Conference.html)

**Abstract**:

Randomized experiments often need to be stopped prematurely due to the treatment having an unintended harmful effect. Existing methods that determine when to stop an experiment early are typically applied to the data in aggregate and do not account for treatment effect heterogeneity. In this paper, we study the early stopping of experiments for harm on heterogeneous populations. We first establish that current methods often fail to stop experiments when the treatment harms a minority group of participants. We then use causal machine learning to develop CLASH, the first broadly-applicable method for heterogeneous early stopping. We demonstrate CLASH's performance on simulated and real data and show that it yields effective early stopping for both clinical trials and A/B tests.

----

## [695] Adaptive Privacy Composition for Accuracy-first Mechanisms

**Authors**: *Ryan M. Rogers, Gennady Samorodnitsky, Zhiwei Steven Wu, Aaditya Ramdas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33301bb40020a56ef56b8b5081e5c4d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33301bb40020a56ef56b8b5081e5c4d5-Abstract-Conference.html)

**Abstract**:

Although there has been work to develop ex-post private mechanisms from Ligett et al. '17 and Whitehouse et al '22 that seeks to provide privacy guarantees subject to a target level of accuracy, there was not a way to use them in conjunction with differentially private mechanisms.  Furthermore, there has yet to be work in developing a theory for how these ex-post privacy mechanisms compose, so that we can track the accumulated privacy over several mechanisms.  We develop privacy filters that allow an analyst to adaptively switch between differentially private mechanisms and ex-post private mechanisms subject to an overall privacy loss guarantee.  We show that using  a particular ex-post private mechanism --- noise reduction mechanisms --- can substantially outperform baseline approaches that use existing privacy loss composition bounds.  We use the common task of returning as many counts as possible subject to a relative error guarantee and an overall privacy budget as a motivating example.

----

## [696] CaMP: Causal Multi-policy Planning for Interactive Navigation in Multi-room Scenes

**Authors**: *Xiaohan Wang, Yuehu Liu, Xinhang Song, Beibei Wang, Shuqiang Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/333581887bf483296118a97773cab0c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/333581887bf483296118a97773cab0c1-Abstract-Conference.html)

**Abstract**:

Visual navigation has been widely studied under the assumption that there may be several clear routes to reach the goal. However, in more practical scenarios such as a house with several messy rooms, there may not. Interactive Navigation (InterNav) considers agents navigating to their goals more effectively with object interactions, posing new challenges of learning interaction dynamics and extra action space. Previous works learn single vision-to-action policy with the guidance of designed representations. However, the causality between actions and outcomes is prone to be confounded when the attributes of obstacles are diverse and hard to measure. Learning policy for long-term action planning in complex scenes also leads to extensive inefficient exploration. In this paper, we introduce a causal diagram of InterNav clarifying the confounding bias caused by obstacles. To address the problem, we propose a multi-policy model that enables the exploration of counterfactual interactions as well as reduces unnecessary exploration. We develop a large-scale dataset containing 600k task episodes in 12k multi-room scenes based on the ProcTHOR simulator and showcase the effectiveness of our method with the evaluations on our dataset.

----

## [697] DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models

**Authors**: *Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, Dong Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/333e67fc4728f147d31608db3ca78e09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/333e67fc4728f147d31608db3ca78e09-Abstract-Conference.html)

**Abstract**:

Even though trained mainly on images, we discover that pretrained diffusion models show impressive power in guiding sketch synthesis. In this paper, we present DiffSketcher, an innovative algorithm that creates \textit{vectorized} free-hand sketches using natural language input. DiffSketcher is developed based on a pre-trained text-to-image diffusion model. It performs the task by directly optimizing a set of BÃ©zier curves with an extended version of the score distillation sampling (SDS) loss, which allows us to use a raster-level diffusion model as a prior for optimizing a parametric vectorized sketch generator. Furthermore, we explore attention maps embedded in the diffusion model for effective stroke initialization to speed up the generation process. The generated sketches demonstrate multiple levels of abstraction while maintaining recognizability, underlying structure, and essential visual details of the subject drawn. Our experiments show that DiffSketcher achieves greater quality than prior work. The code and demo of DiffSketcher can be found at https://ximinng.github.io/DiffSketcher-project/.

----

## [698] Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models

**Authors**: *Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, Yixiao Ge, Ying Shan, Mike Zheng Shou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3340ee1e4a8bad8d32c35721712b4d0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3340ee1e4a8bad8d32c35721712b4d0a-Abstract-Conference.html)

**Abstract**:

Public large-scale text-to-image diffusion models, such as Stable Diffusion, have gained significant attention from the community. These models can be easily customized for new concepts using low-rank adaptations (LoRAs). However,  the utilization of multiple-concept LoRAs to jointly support multiple customized concepts presents a challenge. We refer to this scenario as decentralized multi-concept customization, which involves single-client concept tuning and center-node concept fusion. In this paper, we propose a new framework called Mix-of-Show that addresses the challenges of decentralized multi-concept customization, including concept conflicts resulting from existing single-client LoRA tuning and identity loss during model fusion. Mix-of-Show adopts an embedding-decomposed LoRA (ED-LoRA) for single-client tuning and gradient fusion for the center node to preserve the in-domain essence of single concepts and support theoretically limitless concept fusion. Additionally, we introduce regionally controllable sampling, which extends spatially controllable sampling (e.g., ControlNet and T2I-Adapter) to address attribute binding and missing object problems in multi-concept sampling. Extensive experiments demonstrate that Mix-of-Show is capable of composing multiple customized concepts with high fidelity, including characters, objects, and scenes.

----

## [699] ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation

**Authors**: *Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, Yuxiao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33646ef0ed554145eab65f6250fab0c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33646ef0ed554145eab65f6250fab0c9-Abstract-Conference.html)

**Abstract**:

We present a comprehensive solution to learn and improve text-to-image models from human preference feedback.To begin with, we build ImageReward---the first general-purpose text-to-image human preference reward model---to effectively encode human preferences.Its training is based on our systematic annotation pipeline including rating and ranking, which collects 137k expert comparisons to date.In human evaluation, ImageReward outperforms existing scoring models and metrics, making it a promising automatic metric for evaluating text-to-image synthesis.On top of it, we propose Reward Feedback Learning (ReFL), a direct tuning algorithm to optimize diffusion models against a scorer.Both automatic and human evaluation support ReFL's advantages over compared methods.All code and datasets are provided at \url{https://github.com/THUDM/ImageReward}.

----

## [700] To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning

**Authors**: *Ildus Sadrtdinov, Dmitrii Pozdeev, Dmitry P. Vetrov, Ekaterina Lobacheva*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/336572db3e99930814d6b328d4220cb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/336572db3e99930814d6b328d4220cb6-Abstract-Conference.html)

**Abstract**:

Transfer learning and ensembling are two popular techniques for improving the performance and robustness of neural networks. Due to the high cost of pre-training, ensembles of models fine-tuned from a single pre-trained checkpoint are often used in practice. Such models end up in the same basin of the loss landscape, which we call the pre-train basin, and thus have limited diversity. In this work, we show that ensembles trained from a single pre-trained checkpoint may be improved by better exploring the pre-train basin, however, leaving the basin results in losing the benefits of transfer learning and in degradation of the ensemble quality. Based on the analysis of existing exploration methods, we propose a more effective modification of the Snapshot Ensembles (SSE) for transfer learning setup, StarSSE, which results in stronger ensembles and uniform model soups.

----

## [701] Statistical Limits of Adaptive Linear Models: Low-Dimensional Estimation and Inference

**Authors**: *Licong Lin, Mufang Ying, Suvrojit Ghosh, Koulik Khamaru, Cun-Hui Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3368e8f592b0d46ed85def795fd5168f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3368e8f592b0d46ed85def795fd5168f-Abstract-Conference.html)

**Abstract**:

Estimation and inference in statistics pose significant challenges when data are collected adaptively. Even in linear models, the Ordinary Least Squares (OLS) estimator may fail to exhibit asymptotic normality for single coordinate estimation and have inflated error. This issue is highlighted by a recent minimax lower bound, which shows that the error of estimating a  single coordinate can be enlarged by a multiple of $\sqrt{d}$ when data are allowed to be arbitrarily adaptive, compared with the case when they are i.i.d. Our work explores this striking difference in estimation performance between utilizing i.i.d. and adaptive data. We investigate how the degree of adaptivity in data collection impacts the performance of estimating a low-dimensional parameter component in high-dimensional linear models. We identify conditions on the data collection mechanism under which the estimation error for a low-dimensional parameter component matches its counterpart in the i.i.d. setting, up to a factor that depends on the degree of adaptivity. We show that OLS or OLS on centered data can achieve this matching error. In addition, we propose a novel estimator for single coordinate inference via solving a Two-stage Adaptive Linear Estimating equation (TALE). Under a weaker form of adaptivity in data collection, we establish an asymptotic normality property of the proposed estimator.

----

## [702] Future-Dependent Value-Based Off-Policy Evaluation in POMDPs

**Authors**: *Masatoshi Uehara, Haruka Kiyohara, Andrew Bennett, Victor Chernozhukov, Nan Jiang, Nathan Kallus, Chengchun Shi, Wen Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3380e8116452e0efbf36f35d95e88c94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3380e8116452e0efbf36f35d95e88c94-Abstract-Conference.html)

**Abstract**:

We study off-policy evaluation (OPE) for partially observable MDPs (POMDPs) with general function approximation. Existing methods such as sequential importance sampling estimators and fitted-Q evaluation suffer from the curse of horizon in POMDPs. To circumvent this problem, we develop a novel model-free OPE method by introducing future-dependent value functions that take future proxies as inputs. Future-dependent value functions play similar roles as classical value functions in fully-observable MDPs. We derive a new off-policy Bellman equation for future-dependent value functions as conditional moment equations that use history proxies as instrumental variables. We further propose a minimax learning method to learn future-dependent value functions using the new Bellman equation. We obtain the PAC result, which implies our OPE estimator is close to the true policy value as long as futures and histories contain sufficient information about latent states, and the Bellman completeness. Our code is available at https://github.com/aiueola/neurips2023-future-dependent-ope

----

## [703] Visual Explanations of Image-Text Representations via Multi-Modal Information Bottleneck Attribution

**Authors**: *Ying Wang, Tim G. J. Rudner, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/339caf45a6fa281cae8adc6465343464-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/339caf45a6fa281cae8adc6465343464-Abstract-Conference.html)

**Abstract**:

Vision-language pretrained models have seen remarkable success, but their application to safety-critical settings is limited by their lack of interpretability. To improve the interpretability of vision-language models such as CLIP, we propose a multi-modal information bottleneck (M2IB) approach that learns latent representations that compress irrelevant information while preserving relevant visual and textual features. We demonstrate how M2IB can be applied to attribution analysis of vision-language pretrained models, increasing attribution accuracy and improving the interpretability of such models when applied to safety-critical domains such as healthcare. Crucially, unlike commonly used unimodal attribution methods, M2IB does not require ground truth labels, making it possible to audit representations of vision-language pretrained models when multiple modalities but no ground-truth data is available. Using CLIP as an example, we demonstrate the effectiveness of M2IB attribution and show that it outperforms gradient-based, perturbation-based, and attention-based attribution methods both qualitatively and quantitatively.

----

## [704] PlanE: Representation Learning over Planar Graphs

**Authors**: *Radoslav Dimitrov, Zeyang Zhao, Ralph Abboud, Ismail Ilkan Ceylan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33b47b3d2441a17b95344cd635f3dd01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33b47b3d2441a17b95344cd635f3dd01-Abstract-Conference.html)

**Abstract**:

Graph neural networks are prominent models for representation learning over graphs, where the idea is to iteratively compute representations of nodes of an input graph through a series of transformations in such a way that the learned graph function is isomorphism-invariant on graphs, which makes the learned representations graph invariants. On the other hand, it is well-known that graph invariants learned by these class of models are incomplete: there are pairs of non-isomorphic graphs which cannot be distinguished by standard graph neural networks. This is unsurprising given the computational difficulty of graph isomorphism testing on general graphs, but the situation begs to differ for special graph classes, for which efficient graph isomorphism testing algorithms are known, such as planar graphs. The goal of this work is to design architectures for efficiently learning complete invariants of planar graphs. Inspired by the classical planar graph isomorphism algorithm of Hopcroft and Tarjan, we propose PlanE as a framework for planar representation learning. PlanE includes architectures which can learn complete invariants over planar graphs while remaining practically scalable. We empirically validate the strong performance of the resulting model architectures on well-known planar graph benchmarks, achieving multiple state-of-the-art results.

----

## [705] Optimal Regret Is Achievable with Bounded Approximate Inference Error: An Enhanced Bayesian Upper Confidence Bound Framework

**Authors**: *Ziyi Huang, Henry Lam, Amirhossein Meisami, Haofeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33bb58be3f0e903c75afa73d75b5c67e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33bb58be3f0e903c75afa73d75b5c67e-Abstract-Conference.html)

**Abstract**:

Bayesian bandit algorithms with approximate Bayesian inference have been widely used in real-world applications. However, there is a large discrepancy between the superior practical performance of these approaches and their theoretical justification. Previous research only indicates a negative theoretical result: Thompson sampling could have a worst-case linear regret $\Omega(T)$ with a constant threshold on the inference error measured by one $\alpha$-divergence. To bridge this gap, we propose an Enhanced Bayesian Upper Confidence Bound (EBUCB) framework that can efficiently accommodate bandit problems in the presence of approximate inference. Our theoretical analysis demonstrates that for Bernoulli multi-armed bandits, EBUCB can achieve the optimal regret order $O(\log T)$ if the inference error measured by two different $\alpha$-divergences is less than a constant, regardless of how large this constant is. To our best knowledge, our study provides the first theoretical regret bound that is better than $o(T)$ in the setting of constant approximate inference error. Furthermore, in concordance with the negative results in previous studies, we show that only one bounded $\alpha$-divergence is insufficient to guarantee a sub-linear regret.

----

## [706] Any-to-Any Generation via Composable Diffusion

**Authors**: *Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33edf072fe44f19079d66713a1831550-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33edf072fe44f19079d66713a1831550-Abstract-Conference.html)

**Abstract**:

We present Composable Diffusion (CoDi), a novel generative model capable of generating any combination of output modalities, such as language, image, video, or audio, from any combination of input modalities. Unlike existing generative AI systems, CoDi can generate multiple modalities in parallel and its input is not limited to a subset of modalities like text or image. Despite the absence of training datasets for many combinations of modalities, we propose to align modalities in both the input and output space. This allows CoDi to freely condition on any input combination and generate any group of modalities, even if they are not present in the training data. CoDi employs a novel composable generation strategy which involves building a shared multimodal space by bridging alignment in the diffusion process, enabling the synchronized generation of intertwined modalities, such as temporally aligned video and audio. Highly customizable and flexible, CoDi achieves strong joint-modality generation quality, and outperforms or is on par with the unimodal state-of-the-art for single-modality synthesis.

----

## [707] Rank-DETR for High Quality Object Detection

**Authors**: *Yifan Pu, Weicong Liang, Yiduo Hao, Yuhui Yuan, Yukang Yang, Chao Zhang, Han Hu, Gao Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html)

**Abstract**:

Modern detection transformers (DETRs) use a set of object queries to predict a list of bounding boxes, sort them by their classification confidence scores, and select the top-ranked predictions as the final detection results for the given input image. A highly performant object detector requires accurate ranking for the bounding box predictions. For DETR-based detectors, the top-ranked bounding boxes suffer from less accurate localization quality due to the misalignment between classification scores and localization accuracy, thus impeding the construction of high-quality detectors. In this work, we introduce a simple and highly performant DETR-based object detector by proposing a series of rank-oriented designs, combinedly called Rank-DETR. Our key contributions include: (i) a rank-oriented architecture design that can prompt positive predictions and suppress the negative ones to ensure lower false positive rates, as well as (ii) a rank-oriented loss function and matching cost design that prioritizes predictions of more accurate localization accuracy during ranking to boost the AP under high IoU thresholds. We apply our method to improve the recent SOTA methods (e.g., H-DETR and DINO-DETR) and report strong COCO object detection results when using different backbones such as ResNet-$50$, Swin-T, and Swin-L, demonstrating the effectiveness of our approach. Code is available at \url{https://github.com/LeapLabTHU/Rank-DETR}.

----

## [708] Towards Efficient Pre-Trained Language Model via Feature Correlation Distillation

**Authors**: *Kun Huang, Xin Guo, Meng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34260a400e39a802961470b3d3de99cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34260a400e39a802961470b3d3de99cc-Abstract-Conference.html)

**Abstract**:

Knowledge Distillation (KD) has emerged as a promising approach for compressing large Pre-trained Language Models (PLMs). The performance of KD relies on how to effectively formulate and transfer the knowledge from the teacher model to the student model. Prior arts mainly focus on directly aligning output features from the transformer block, which may impose overly strict constraints on the student model's learning process and complicate the training process by introducing extra parameters and computational cost. Moreover, our analysis indicates that the different relations within self-attention, as adopted in other works, involves more computation complexities and can easily be constrained by the number of heads, potentially leading to suboptimal solutions.  To address these issues, we propose a novel approach that builds relationships directly from output features. Specifically, we introduce token-level and sequence-level relations concurrently  to fully exploit the knowledge from the teacher model. Furthermore, we propose a correlation-based distillation loss to alleviate the exact match properties inherent in traditional KL divergence or MSE loss functions. Our method, dubbed FCD, presents a simple yet effective method to compress various architectures (BERT, RoBERTa, and GPT) and model sizes (base-size and large-size).  Extensive experimental results demonstrate that our distilled, smaller language models significantly surpass existing KD methods across various NLP tasks.

----

## [709] Simple and Asymmetric Graph Contrastive Learning without Augmentations

**Authors**: *Teng Xiao, Huaisheng Zhu, Zhengyu Chen, Suhang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3430bcc30cdaabd0bf6c5d0c31bda67c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3430bcc30cdaabd0bf6c5d0c31bda67c-Abstract-Conference.html)

**Abstract**:

Graph Contrastive Learning (GCL) has shown superior performance in representation learning in graph-structured data. Despite their success, most existing GCL methods rely on prefabricated graph augmentation and homophily assumptions. Thus, they fail to generalize well to heterophilic graphs where connected nodes may have different class labels and dissimilar features. In this paper, we study the problem of conducting contrastive learning on homophilic and heterophilic graphs. We find that we can achieve promising performance simply by considering an asymmetric view of the neighboring nodes. The resulting simple algorithm, Asymmetric Contrastive Learning for Graphs (GraphACL), is easy to implement and does not rely on graph augmentations and homophily assumptions. We provide theoretical and empirical evidence that GraphACL can capture one-hop local neighborhood information and two-hop monophily similarity, which are both important for modeling heterophilic graphs. Experimental results show that the simple GraphACL significantly outperforms state-of-the-art graph contrastive learning and self-supervised learning methods on  homophilic and heterophilic graphs. The code of GraphACL is available at https://github.com/tengxiao1/GraphACL.

----

## [710] Large-Scale Distributed Learning via Private On-Device LSH

**Authors**: *Tahseen Rabbani, Marco Bornstein, Furong Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html)

**Abstract**:

Locality-sensitive hashing (LSH) based frameworks have been used efficiently to select weight vectors in a dense hidden layer with high cosine similarity to an input, enabling dynamic pruning.  While this type of scheme has been shown to improve computational training efficiency, existing algorithms require repeated randomized projection of the full layer weight, which is impractical for computational- and memory-constrained devices.  In a distributed setting, deferring LSH analysis to a centralized host is (i) slow if the device cluster is large and (ii) requires access to input data which is forbidden in a federated context.  Using a new family of hash functions, we develop the first private, personalized, and memory-efficient on-device LSH framework.Our framework enables privacy and personalization by allowing each device to generate hash tables, without the help of a central host, using device-specific hashing hyper-parameters (e.g., number of hash tables or hash length).Hash tables are generated with a compressed set of the full weights, and can be serially generated and discarded if the process is memory-intensive.This allows devices to avoid maintaining (i) the fully-sized model and (ii) large amounts of hash tables in local memory for LSH analysis. We prove several statistical and sensitivity properties of our hash functions, and experimentally demonstrate that our framework is competitive in training large scale recommender networks compared to other LSH frameworks which assume unrestricted on-device capacity.

----

## [711] Facilitating Graph Neural Networks with Random Walk on Simplicial Complexes

**Authors**: *Cai Zhou, Xiyuan Wang, Muhan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/345208bdbbb6104616311dfc1d093fe7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/345208bdbbb6104616311dfc1d093fe7-Abstract-Conference.html)

**Abstract**:

Node-level random walk has been widely used to improve Graph Neural Networks. However, there is limited attention to random walk on edge and, more generally, on $k$-simplices. This paper systematically analyzes how random walk on different orders of simplicial complexes (SC) facilitates GNNs in their theoretical expressivity. First, on $0$-simplices or node level, we establish a connection between existing positional encoding (PE) and structure encoding (SE) methods through the bridge of random walk. Second, on $1$-simplices or edge level, we bridge edge-level random walk and Hodge $1$-Laplacians and design corresponding edge PE respectively. In spatial domain, we directly make use of edge level random walk to construct EdgeRWSE. Based on spectral analysis of Hodge $1$-Laplcians, we propose Hodge1Lap, a permutation equivariant and expressive edge-level positional encoding. Third, we generalize our theory to random walk on higher-order simplices and propose the general principle to design PE on simplices based on random walk and Hodge Laplacians. Inter-level random walk is also introduced to unify a wide range of simplicial networks. Extensive experiments verify the effectiveness of our random walk-based methods.

----

## [712] Koopman Kernel Regression

**Authors**: *Petar Bevanda, Max Beier, Armin Lederer, Stefan Sosnowski, Eyke Hüllermeier, Sandra Hirche*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34678d08b36076de986df95c5bbba92f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34678d08b36076de986df95c5bbba92f-Abstract-Conference.html)

**Abstract**:

Many machine learning approaches for decision making, such as reinforcement learning, rely on simulators or predictive models to forecast the time-evolution of quantities of interest, e.g., the state of an agent or the reward of a policy. Forecasts of such complex phenomena are commonly described by highly nonlinear dynamical systems, making their use in optimization-based decision-making challenging.Koopman operator theory offers a beneficial paradigm for addressing this problem by characterizing forecasts via linear time-invariant (LTI) ODEs, turning multi-step forecasts into sparse matrix multiplication.Though there exists a variety of learning approaches, they usually lack crucial learning-theoretic guarantees, making the behavior of the obtained models with increasing data and dimensionality unclear.We address the aforementioned by deriving a universal Koopman-invariant reproducing kernel Hilbert space (RKHS) that solely spans transformations into LTI dynamical systems. The resulting Koopman Kernel Regression (KKR) framework enables the use of statistical learning tools from function approximation for novel convergence results and generalization error bounds under weaker assumptions than existing work. Our experiments demonstrate superior forecasting performance compared to Koopman operator and sequential data predictors in RKHS.

----

## [713] Diffusion Self-Guidance for Controllable Image Generation

**Authors**: *Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, Aleksander Holynski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3469b211b829b39d2b0cfd3b880a869c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3469b211b829b39d2b0cfd3b880a869c-Abstract-Conference.html)

**Abstract**:

Large-scale generative models are capable of producing high-quality images from detailed prompts. However, many aspects of an image are difficult or impossible to convey through text. We introduce self-guidance, a method that provides precise control over properties of the generated image by guiding the internal representations of diffusion models. We demonstrate that the size, location, and appearance of objects can be extracted from these representations, and show how to use them to steer the sampling process. Self-guidance operates similarly to standard classifier guidance, but uses signals present in the pretrained model itself, requiring no additional models or training. We demonstrate the flexibility and effectiveness of self-guided generation through a wide range of challenging image manipulations, such as modifying the position or size of a single object (keeping the rest of the image unchanged), merging the appearance of objects in one image with the layout of another, composing objects from multiple images into one, and more. We also propose a new method for reconstruction using self-guidance, which allows extending our approach to editing real images.

----

## [714] Neural (Tangent Kernel) Collapse

**Authors**: *Mariia Seleznova, Dana Weitzner, Raja Giryes, Gitta Kutyniok, Hung-Hsu Chou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3477ca0ce484aa2fa42c1361ab601c25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3477ca0ce484aa2fa42c1361ab601c25-Abstract-Conference.html)

**Abstract**:

This work bridges two important concepts: the Neural Tangent Kernel (NTK), which captures the evolution of deep neural networks (DNNs) during training, and the Neural Collapse (NC) phenomenon, which refers to the emergence of symmetry and structure in the last-layer features of well-trained classification DNNs. We adopt the natural assumption that the empirical NTK develops a block structure aligned with the class labels, i.e., samples within the same class have stronger correlations than samples from different classes. Under this assumption, we derive the dynamics of DNNs trained with mean squared (MSE) loss and break them into interpretable phases. Moreover, we identify an invariant that captures the essence of the dynamics, and use it to prove the emergence of NC in DNNs with block-structured NTK. We provide large-scale numerical experiments on three common DNN architectures and three benchmark datasets to support our theory.

----

## [715] Fast Optimal Locally Private Mean Estimation via Random Projections

**Authors**: *Hilal Asi, Vitaly Feldman, Jelani Nelson, Huy L. Nguyen, Kunal Talwar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34822dab66c13f0100017b8ea373038a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34822dab66c13f0100017b8ea373038a-Abstract-Conference.html)

**Abstract**:

We study the problem of locally private mean estimation of high-dimensional vectors in the Euclidean ball. Existing algorithms for this problem either incur sub-optimal error or have high communication and/or run-time complexity. We propose a new algorithmic framework, namely ProjUnit, for private mean estimation that yields algorithms that are computationally efficient, have low communication complexity, and incur optimal error up to a $1+o(1)$-factor. Our framework is deceptively simple: each randomizer projects its input to a random low-dimensional subspace and then runs an optimal algorithm such a PrivUnitG in the lower dimensional space. We analyze the error of the algorithm in terms of properties of the random projection ensemble, and study two instantiations. We conduct several experiments for private mean estimation and private federated learning which demonstrate that our algorithms obtain nearly the same utility as optimal algorithms while having significantly lower communication and computational cost.

----

## [716] Expert load matters: operating networks at high accuracy and low manual effort

**Authors**: *Sara Sangalli, Ertunc Erdil, Ender Konukoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/348346383eb58ed19def02e233c408d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/348346383eb58ed19def02e233c408d6-Abstract-Conference.html)

**Abstract**:

In human-AI collaboration systems for critical applications, in order to ensure minimal error, users should set an operating point based on model confidence to determine when the decision should be delegated to human experts. Samples for which model confidence is lower than the operating point would be manually analysed by experts to avoid mistakes.Such systems can become truly useful only if they consider two aspects: models should be confident only for samples for which they are accurate, and the number of samples delegated to experts should be minimized.The latter aspect is especially crucial for applications where available expert time is limited and expensive, such as healthcare. The trade-off between the model accuracy and the number of samples delegated to experts can be represented by a curve that is similar to an ROC curve, which we refer to as confidence operating characteristic (COC) curve. In this paper, we argue that deep neural networks should be trained by taking into account both accuracy and expert load and, to that end, propose a new complementary loss function for classification that maximizes the area under this COC curve.This promotes simultaneously the increase in network accuracy and the reduction in number of samples delegated to humans.We perform experiments on multiple computer vision and medical image datasets for classification.Our results demonstrate that the proposed loss improves classification accuracy and delegates less number of decisions to experts, achieves better out-of-distribution samples detection and on par calibration performance compared to existing loss functions.

----

## [717] PRODIGY: Enabling In-context Learning Over Graphs

**Authors**: *Qian Huang, Hongyu Ren, Peng Chen, Gregor Krzmanc, Daniel Zeng, Percy Liang, Jure Leskovec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34dce0dc3121951dd0399ba02c0f0d06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34dce0dc3121951dd0399ba02c0f0d06-Abstract-Conference.html)

**Abstract**:

In-context learning is the ability of a pretrained model to  adapt to novel and diverse downstream tasks by conditioning on prompt examples, without optimizing any parameters. While large language models have demonstrated this ability, how in-context learning could be performed over graphs is unexplored. In this paper, we develop \textbf{Pr}etraining \textbf{O}ver \textbf{D}iverse \textbf{I}n-Context  \textbf{G}raph S\textbf{y}stems (PRODIGY), the first pretraining framework that enables in-context learning over graphs. The key idea of our framework is to formulate in-context learning over graphs with a novel \emph{prompt graph} representation, which connects prompt examples and queries. We then propose a graph neural network architecture over the prompt graph and a corresponding family of in-context pretraining objectives. With PRODIGY, the pretrained model can directly perform novel downstream classification tasks on unseen graphs via in-context learning. We provide empirical evidence of the effectiveness of our framework by showcasing its strong in-context learning performance on tasks involving citation networks and knowledge graphs. Our approach outperforms the in-context learning accuracy of contrastive pretraining baselines with hard-coded adaptation by 18\% on average across all setups. Moreover, it also outperforms standard finetuning with limited data by 33\% on average with in-context learning.

----

## [718] Towards Automated Circuit Discovery for Mechanistic Interpretability

**Authors**: *Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html)

**Abstract**:

Through considerable effort and intuition, several recent works have reverse-engineered nontrivial behaviors oftransformer models. This paper systematizes the mechanistic interpretability process they followed. First, researcherschoose a metric and dataset that elicit the desired model behavior. Then, they apply activation patching to find whichabstract neural network units are involved in the behavior. By varying the dataset, metric, and units underinvestigation, researchers can understand the functionality of each component.We automate one of the process' steps: finding the connections between the abstract neural network units that form a circuit. We propose several algorithms and reproduce previous interpretability results to validate them. Forexample, the ACDC algorithm rediscovered 5/5 of the component types in a circuit in GPT-2 Small that computes theGreater-Than operation. ACDC selected 68 of the 32,000 edges in GPT-2 Small, all of which were manually found byprevious work. Our code is available at https://github.com/ArthurConmy/Automatic-Circuit-Discovery

----

## [719] Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation

**Authors**: *Hongcheng Wang, Andy Guan Hong Chen, Xiaoqi Li, Mingdong Wu, Hao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34e278fbbd7d6d7d788c98065988e1a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34e278fbbd7d6d7d788c98065988e1a9-Abstract-Conference.html)

**Abstract**:

The task of Visual Object Navigation (VON) involves an agent's ability to locate a particular object within a given scene. To successfully accomplish the VON task, two essential conditions must be fulfiled:  1) the user knows the name of the desired object; and 2) the user-specified object actually is present within the scene. To meet these conditions, a simulator can incorporate predefined object names and positions into the metadata of the scene. However, in real-world scenarios, it is often challenging to ensure that these conditions are always met. Humans in an unfamiliar environment may not know which objects are present in the scene, or they may mistakenly specify an object that is not actually present.  Nevertheless, despite these challenges, humans may still have a demand for an object, which could potentially be fulfilled by other objects present within the scene in an equivalent manner. Hence, this paper proposes Demand-driven Navigation (DDN), which leverages the user's demand as the task instruction and prompts the agent to find an object which matches the specified demand. DDN aims to relax the stringent conditions of VON by focusing on fulfilling the user's demand rather than relying solely on specified object names. This paper proposes a method of acquiring textual attribute features of objects by extracting common sense knowledge  from a large language model (LLM). These textual attribute features are subsequently aligned with visual attribute features using Contrastive Language-Image Pre-training (CLIP). Incorporating the visual attribute features as prior knowledge, enhances the navigation process. Experiments on AI2Thor with the ProcThor dataset demonstrate that the visual attribute features improve the agent's navigation performance and outperform the baseline methods commonly used in the VON and VLN task and methods with LLMs. The codes and demonstrations can be viewed at https://sites.google.com/view/demand-driven-navigation.

----

## [720] On the Convergence of No-Regret Learning Dynamics in Time-Varying Games

**Authors**: *Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34f1c2e7ab91b6fa481ad0286a08ad02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34f1c2e7ab91b6fa481ad0286a08ad02-Abstract-Conference.html)

**Abstract**:

Most of the literature on learning in games has focused on the restrictive setting where the underlying repeated game does not change over time. Much less is known about the convergence of no-regret learning algorithms in dynamic multiagent settings. In this paper, we characterize the convergence of optimistic gradient descent (OGD) in time-varying games. Our framework yields sharp convergence bounds for the equilibrium gap of OGD in zero-sum games parameterized on natural variation measures of the sequence of games, subsuming known results for static games. Furthermore, we establish improved second-order variation bounds under strong convexity-concavity, as long as each game is repeated multiple times. Our results also apply to time-varying general-sum multi-player games via a bilinear formulation of correlated equilibria, which has novel implications for meta-learning and for obtaining refined variation-dependent regret bounds, addressing questions left open in prior papers. Finally, we leverage our framework to also provide new insights on dynamic regret guarantees in static games.

----

## [721] Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design

**Authors**: *Ibrahim M. Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, Lucas Beyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3504a4fa45685d668ce92797fbbf1895-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3504a4fa45685d668ce92797fbbf1895-Abstract-Conference.html)

**Abstract**:

Scaling laws have been recently employed to derive compute-optimal model size (number of parameters) for a given compute duration. We advance and refine such methods to infer compute-optimal model shapes, such as width and depth, and successfully implement this in vision transformers. Our shape-optimized vision transformer, SoViT, achieves results competitive with models that exceed twice its size, despite being pre-trained with an equivalent amount of  compute. For example, SoViT-400m/14 achieves 90.3% fine-tuning accuracy on ILSRCV2012, surpassing the much larger ViT-g/14 and approaching ViT-G/14 under identical settings, with also less than half the inference cost. We conduct a thorough evaluation across multiple tasks, such as image classification, captioning, VQA and zero-shot transfer, demonstrating the effectiveness of our model across a broad range of domains and identifying limitations. Overall, our findings challenge the prevailing approach of blindly scaling up vision models and pave a path for a more informed scaling.

----

## [722] Expressive Sign Equivariant Networks for Spectral Geometric Learning

**Authors**: *Derek Lim, Joshua Robinson, Stefanie Jegelka, Haggai Maron*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3516aa3393f0279e04c099f724664f99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3516aa3393f0279e04c099f724664f99-Abstract-Conference.html)

**Abstract**:

Recent work has shown the utility of developing machine learning models that respect the structure and symmetries of eigenvectors. These works promote sign invariance, since for any eigenvector v the negation -v is also an eigenvector. However, we show that sign invariance is theoretically limited for tasks such as building orthogonally equivariant models and learning node positional encodings for link prediction in graphs. In this work, we demonstrate the benefits of sign equivariance for these tasks. To obtain these benefits, we develop novel sign equivariant neural network architectures. Our models are based on a new analytic characterization of sign equivariant polynomials and thus inherit provable expressiveness properties. Controlled synthetic experiments show that our networks can achieve the theoretically predicted benefits of sign equivariant models.

----

## [723] FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery

**Authors**: *Anatol Garioud, Nicolas Gonthier, Loïc Landrieu, Apolline De Wit, Marion Valette, Marc Poupée, Sébastien Giordano, Boris Wattrelos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/353ca88f722cdd0c481b999428ae113a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/353ca88f722cdd0c481b999428ae113a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce the French Land cover from Aerospace ImageRy (FLAIR), an extensive dataset from the French National Institute of Geographical and Forest Information (IGN) that provides a unique and rich resource for large-scale geospatial analysis. FLAIR contains high-resolution aerial imagery with a ground sample distance of 20 cm and over 20 billion individually labeled pixels for precise land-cover classification. The dataset also integrates temporal and spectral data from optical satellite time series. FLAIR thus combines data with varying spatial, spectral, and temporal resolutions across over 817 km² of acquisitions representing the full landscape diversity of France. This diversity makes FLAIR a valuable resource for the development and evaluation of novel methods for large-scale land-cover semantic segmentation and raises significant challenges in terms of computer vision, data fusion, and geospatial analysis. We also provide powerful uni- and multi-sensor baseline models that can be employed to assess algorithm's performance and for downstream applications.

----

## [724] Strategic Data Sharing between Competitors

**Authors**: *Nikita Tsoy, Nikola Konstantinov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/355091f86e3e2296fbeefa10676ddb17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/355091f86e3e2296fbeefa10676ddb17-Abstract-Conference.html)

**Abstract**:

Collaborative learning techniques have significantly advanced in recent years, enabling private model training across multiple organizations. Despite this opportunity, firms face a dilemma when considering data sharing with competitors—while collaboration can improve a company’s machine learning model, it may also benefit competitors and hence reduce profits. In this work, we introduce a general framework for analyzing this data-sharing trade-off. The framework consists of three components, representing the firms’ production decisions, the effect of additional data on model quality, and the data-sharing negotiation process, respectively. We then study an instantiation of the framework, based on a conventional market model from economic theory, to identify key factors that affect collaboration incentives. Our findings indicate a profound impact of market conditions on the data-sharing incentives. In particular, we find that reduced competition, in terms of the similarities between the firms’ products, and harder learning tasks foster collaboration.

----

## [725] Optimal Time Complexities of Parallel Stochastic Optimization Methods Under a Fixed Computation Model

**Authors**: *Alexander Tyurin, Peter Richtárik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3563abb1040f4e150f4242a7282cd1ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3563abb1040f4e150f4242a7282cd1ec-Abstract-Conference.html)

**Abstract**:

Parallelization is a popular strategy for improving the performance of methods. Optimization methods are no exception: design of efficient parallel optimization methods and tight analysis of their theoretical properties are important research endeavors. While the minimax complexities are well known for sequential optimization methods, the theory of parallel optimization methods is less explored. In this paper, we propose a new protocol that generalizes the classical oracle framework approach. Using this protocol, we establish minimax complexities for parallel optimization methods that have access to an unbiased stochastic gradient oracle with bounded variance. We consider a fixed computation model characterized by each worker requiring a fixed but worker-dependent time to calculate stochastic gradient. We prove lower bounds and develop optimal algorithms that attain them. Our results have surprising consequences for the literature of asynchronous optimization methods.

----

## [726] An ε-Best-Arm Identification Algorithm for Fixed-Confidence and Beyond

**Authors**: *Marc Jourdan, Rémy Degenne, Emilie Kaufmann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/358e4a39b8ace4744fbad77e84a7e757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/358e4a39b8ace4744fbad77e84a7e757-Abstract-Conference.html)

**Abstract**:

We propose EB-TC$\varepsilon$, a novel sampling rule for $\varepsilon$-best arm identification in stochastic bandits.It is the first instance of Top Two algorithm analyzed for approximate best arm identification. EB-TC$\varepsilon$ is an *anytime* sampling rule that can therefore be employed without modification for fixed confidence or fixed budget identification (without prior knowledge of the budget).We provide three types of theoretical guarantees for EB-TC$\varepsilon$.First, we prove bounds on its expected sample complexity in the fixed confidence setting, notably showing its asymptotic optimality in combination with an adaptive tuning of its exploration parameter.We complement these findings with upper bounds on its probability of error at any time and for any slack parameter, which further yield upper bounds on its simple regret at any time.Finally, we show through numerical simulations that EB-TC$\varepsilon$ performs favorably compared to existing algorithms for different approximate best arm identification tasks.

----

## [727] Debiased and Denoised Entity Recognition from Distant Supervision

**Authors**: *Haobo Wang, Yiwen Dong, Ruixuan Xiao, Fei Huang, Gang Chen, Junbo Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/359ddb9caccb4c54cc915dceeacf4892-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/359ddb9caccb4c54cc915dceeacf4892-Abstract-Conference.html)

**Abstract**:

While distant supervision has been extensively explored and exploited in NLP tasks like named entity recognition, a major obstacle stems from the inevitable noisy distant labels tagged unsupervisedly. A few past works approach this problem by adopting a self-training framework with a sample-selection mechanism. In this work, we innovatively identify two types of biases that were omitted by prior work, and these biases lead to inferior performance of the distant-supervised NER setup. First, we characterize the noise concealed in the distant labels as highly structural rather than fully randomized. Second, the self-training framework would ubiquitously introduce an inherent bias that causes erroneous behavior in both sample selection and eventually prediction. To cope with these problems, we propose a novel self-training framework, dubbed DesERT. This framework augments the conventional NER predicative pathway to a dual form that effectively adapts the sample-selection process to conform to its innate distributional-bias structure. The other crucial component of DesERT composes a debiased module aiming to enhance the token representations, hence the quality of the pseudo-labels. Extensive experiments are conducted to validate the DesERT. The results show that our framework establishes a new state-of-art performance, it achieves a +2.22% average F1 score improvement on five standardized benchmarking datasets. Lastly, DesERT demonstrates its effectiveness under a new DSNER benchmark where additional distant supervision comes from the ChatGPT model.

----

## [728] Accelerated Training via Incrementally Growing Neural Networks using Variance Transfer and Learning Rate Adaptation

**Authors**: *Xin Yuan, Pedro Savarese, Michael Maire*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/359ffa88712bd688963a0ca641d8330b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/359ffa88712bd688963a0ca641d8330b-Abstract-Conference.html)

**Abstract**:

We develop an approach to efficiently grow neural networks, within which parameterization and optimization strategies are designed by considering their effects on the training dynamics.  Unlike existing growing methods, which follow simple replication heuristics or utilize auxiliary gradient-based local optimization, we craft a parameterization scheme which dynamically stabilizes weight, activation, and gradient scaling as the architecture evolves, and maintains the inference functionality of the network.  To address the optimization difficulty resulting from imbalanced training effort distributed to subnetworks fading in at different growth phases, we propose a learning rate adaption mechanism that rebalances the gradient contribution of these separate subcomponents.  Experiments show that our method achieves comparable or better accuracy than training large fixed-size models, while saving a substantial portion of the original training computation budget.  We demonstrate that these gains translate into real wall-clock training speedups.

----

## [729] Likelihood-Based Diffusion Language Models

**Authors**: *Ishaan Gulrajani, Tatsunori B. Hashimoto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html)

**Abstract**:

Despite a growing interest in diffusion-based language models, existing work has not shown that these models can attain nontrivial likelihoods on standard language modeling benchmarks. In this work, we take the first steps towards closing the likelihood gap between autoregressive and diffusion-based language models, with the goal of building and releasing a diffusion model which outperforms a small but widely-known autoregressive model. We pursue this goal through algorithmic improvements, scaling laws, and increased compute. On the algorithmic front, we introduce several methodological improvements for the maximum-likelihood training of diffusion language models. We then study scaling laws for our diffusion models and find compute-optimal training regimes which differ substantially from autoregressive models. Using our methods and scaling analysis, we train and release Plaid 1B, a large diffusion language model which outperforms GPT-2 124M in likelihood on benchmark datasets and generates fluent samples in unconditional and zero-shot control settings.

----

## [730] Structural Pruning for Diffusion Models

**Authors**: *Gongfan Fang, Xinyin Ma, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/35c1d69d23bb5dd6b9abcd68be005d5c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/35c1d69d23bb5dd6b9abcd68be005d5c-Abstract-Conference.html)

**Abstract**:

Generative modeling has recently undergone remarkable advancements, primarily propelled by the transformative implications of Diffusion Probabilistic Models (DPMs). The impressive capability of these models, however, often entails significant computational overhead during both training and inference. To tackle this challenge, we present Diff-Pruning, an efficient compression method tailored for learning lightweight diffusion models from pre-existing ones, without the need for extensive re-training. The essence of Diff-Pruning is encapsulated in a Taylor expansion over pruned timesteps, a process that disregards non-contributory diffusion steps and ensembles informative gradients to identify important weights. Our empirical assessment, undertaken across several datasets highlights two primary benefits of our proposed method: 1) Efficiency: it enables approximately a 50\% reduction in FLOPs at a mere 10% to 20% of the original training expenditure; 2) Consistency: the pruned diffusion models inherently preserve generative behavior congruent with their pre-trained models.

----

## [731] Fast and Regret Optimal Best Arm Identification: Fundamental Limits and Low-Complexity Algorithms

**Authors**: *Qining Zhang, Lei Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/35fdecdf8861bc15110d48fbec3193cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/35fdecdf8861bc15110d48fbec3193cf-Abstract-Conference.html)

**Abstract**:

This paper considers a stochastic Multi-Armed Bandit (MAB) problem with dual objectives: (i) quick identification and commitment to the optimal arm, and (ii) reward maximization throughout a sequence of $T$ consecutive rounds. Though each objective has been individually well-studied, i.e., best arm identification for (i) and regret minimization for (ii), the simultaneous realization of both objectives remains an open problem, despite its practical importance. This paper introduces \emph{Regret Optimal Best Arm Identification} (ROBAI) which aims to achieve these dual objectives. To solve ROBAI with both pre-determined stopping time and adaptive stopping time requirements, we present an algorithm called EOCP and its variants respectively, which not only achieve asymptotic optimal regret in both Gaussian and general bandits, but also commit to the optimal arm in $\mathcal{O}(\log T)$ rounds with pre-determined stopping time and $\mathcal{O}(\log^2 T)$ rounds with adaptive stopping time. We further characterize lower bounds on the commitment time (equivalent to the sample complexity) of ROBAI, showing that EOCP and its variants are sample optimal with pre-determined stopping time, and almost sample optimal with adaptive stopping time. Numerical results confirm our theoretical analysis and reveal an interesting ``over-exploration'' phenomenon carried by classic UCB algorithms, such that EOCP has smaller regret even though it stops exploration much earlier than UCB, i.e., $\mathcal{O}(\log T)$ versus $\mathcal{O}(T)$, which suggests over-exploration is unnecessary and potentially harmful to system performance.

----

## [732] Simultaneous embedding of multiple attractor manifolds in a recurrent neural network using constrained gradient optimization

**Authors**: *Haggai Agmon, Yoram Burak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/361e5112d2eca09513bbd266e4b2d2be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/361e5112d2eca09513bbd266e4b2d2be-Abstract-Conference.html)

**Abstract**:

The storage of continuous variables in working memory is hypothesized to be sustained in the brain by the dynamics of recurrent neural networks (RNNs) whose steady states form continuous manifolds. In some cases, it is thought that the synaptic connectivity supports multiple attractor manifolds, each mapped to a different context or task. For example, in hippocampal area CA3, positions in distinct environments are represented by distinct sets of population activity patterns, each forming a continuum. It has been argued that the embedding of multiple continuous attractors in a single RNN inevitably causes detrimental interference: quenched noise in the synaptic connectivity disrupts the continuity of each attractor, replacing it by a discrete set of steady states that can be conceptualized as lying on local minima of an abstract energy landscape. Consequently, population activity patterns exhibit systematic drifts towards one of these discrete minima, thereby degrading the stored memory over time. Here we show that it is possible to dramatically attenuate these detrimental interference effects by adjusting the synaptic weights. Synaptic weight adjustment are derived from a loss function that quantifies the roughness of the energy landscape along each of the embedded attractor manifolds. By minimizing this loss function, the stability of states can be dramatically improved, without compromising the capacity.

----

## [733] Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization

**Authors**: *Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan S. Kankanhalli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/364071531ff2398e0fb8bae31f615b69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/364071531ff2398e0fb8bae31f615b69-Abstract-Conference.html)

**Abstract**:

Adversarial contrastive learning (ACL) is a technique that enhances standard contrastive learning (SCL) by incorporating adversarial data to learn a robust representation that can withstand adversarial attacks and common corruptions without requiring costly annotations. To improve transferability, the existing work introduced the standard invariant regularization (SIR) to impose style-independence property to SCL, which can exempt the impact of nuisance style factors in the standard representation. However, it is unclear how the style-independence property benefits ACL-learned robust representations. In this paper, we leverage the technique of causal reasoning to interpret the ACL and propose adversarial invariant regularization (AIR) to enforce independence from style factors. We regulate the ACL using both SIR and AIR to output the robust representation. Theoretically, we show that AIR implicitly encourages the representational distance between different views of natural data and their adversarial variants to be independent of style factors. Empirically, our experimental results show that invariant regularization significantly improves the performance of state-of-the-art ACL methods in terms of both standard generalization and robustness on downstream tasks. To the best of our knowledge, we are the first to apply causal reasoning to interpret ACL and develop AIR for enhancing ACL-learned robust representations. Our source code is at https://github.com/GodXuxilie/EnhancingACLvia_AIR.

----

## [734] Best Arm Identification with Fixed Budget: A Large Deviation Perspective

**Authors**: *Po-An Wang, Ruo-Chun Tzeng, Alexandre Proutière*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/364d565b4b726c607aa40e1632045873-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/364d565b4b726c607aa40e1632045873-Abstract-Conference.html)

**Abstract**:

We consider the problem of identifying the best arm in stochastic Multi-Armed Bandits (MABs) using a fixed sampling budget. Characterizing the minimal instance-specific error probability for this problem constitutes one of the important remaining open problems in MABs. When arms are selected using a static sampling strategy, the error probability decays exponentially with the number of samples at a rate that can be explicitly derived via Large Deviation techniques. Analyzing the performance of algorithms with adaptive sampling strategies is however much more challenging. In this paper, we establish a connection between the Large Deviation Principle (LDP) satisfied by the empirical proportions of arm draws and that satisfied by the empirical arm rewards. This connection holds for any adaptive algorithm, and is leveraged (i) to improve error probability upper bounds of some existing algorithms, such as the celebrated SR (Successive Rejects) algorithm \cite{audibert2010best}, and (ii) to devise and analyze new algorithms. In particular, we present CR (Continuous Rejects), a truly adaptive algorithm that can reject arms in {\it any} round based on the observed empirical gaps between the rewards of various arms. Applying our Large Deviation results, we prove that CR enjoys better performance guarantees than existing algorithms, including SR. Extensive numerical experiments confirm this observation.

----

## [735] Full-Atom Protein Pocket Design via Iterative Refinement

**Authors**: *Zaixi Zhang, Zepu Lu, Zhongkai Hao, Marinka Zitnik, Qi Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/365a6f71486ecdfa7eb8d61cbe168782-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/365a6f71486ecdfa7eb8d61cbe168782-Abstract-Conference.html)

**Abstract**:

The design of \emph{de novo} functional proteins that bind with specific ligand molecules is crucial in various domains like therapeutics and bio-engineering. One vital yet challenging step is to design the protein pocket, the cavity region of protein where the ligand binds with. Existing methods suffer from inefficient generation, insufficient context modeling (ligand molecule), and incapability of generating sidechain atoms. To overcome the limitations, we propose a \textbf{F}ull-\textbf{A}tom \textbf{I}terative \textbf{R}efinement framework (\textbf{FAIR}) for protein pocket sequence (i.e., residue types) and 3D structure co-design. Generally, FAIR consists of two steps that follow a coarse-to-fine pipeline (backbone atoms to full atoms including sidechain) for full-atom generation. For efficiency, all residue types and structures are updated together in each round (i.e., full-shot refinement). In the first step, the residue types and backbone coordinates are updated with a hierarchical context encoder and two structure refinement modules capturing inter-residue and pocket-ligand interactions. The second step further models the sidechain atoms of pockets and updates residue types to achieve sequence-structure consistency. The structure of the binding ligand is also updated along with the above refinement iterations accounting for its flexibility. Finally, extensive evaluations showthat FAIR outperforms baselines in efficiently designing high-quality pocket sequences and structures. Specifically, the average improvements on AAR and RMSD are over 10$\%$.

----

## [736] Flow Matching for Scalable Simulation-Based Inference

**Authors**: *Jonas Wildberger, Maximilian Dax, Simon Buchholz, Stephen R. Green, Jakob H. Macke, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3663ae53ec078860bb0b9c6606e092a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3663ae53ec078860bb0b9c6606e092a0-Abstract-Conference.html)

**Abstract**:

Neural posterior estimation methods based on discrete normalizing flows have become established tools for simulation-based inference (SBI), but scaling them to high-dimensional problems can be challenging. Building on recent advances in generative modeling, we here present flow matching posterior estimation (FMPE), a technique for SBI using continuous normalizing flows. Like diffusion models, and in contrast to discrete flows, flow matching allows for unconstrained architectures, providing enhanced flexibility for complex data modalities. Flow matching, therefore, enables exact density evaluation, fast training, and seamless scalability to large architectures---making it ideal for SBI. We show that FMPE achieves competitive performance on an established SBI benchmark, and then demonstrate its improved scalability on a challenging scientific problem: for gravitational-wave inference, FMPE outperforms methods based on comparable discrete flows, reducing training time by 30\% with substantially improved accuracy. Our work underscores the potential of FMPE to enhance performance in challenging inference scenarios, thereby paving the way for more advanced applications to scientific problems.

----

## [737] Learning DAGs from Data with Few Root Causes

**Authors**: *Panagiotis Misiakos, Chris Wendler, Markus Püschel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/367ab3106d990825d5b47ce91db75a73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/367ab3106d990825d5b47ce91db75a73-Abstract-Conference.html)

**Abstract**:

We present a novel perspective and algorithm for learning directed acyclic graphs (DAGs) from data generated by a linear structural equation model (SEM). First, we show that a linear SEM can be viewed as a linear transform that, in prior work, computes the data from a dense input vector of random valued root causes (as we will call them) associated with the nodes. Instead, we consider the case of (approximately) few root causes and also introduce noise in the measurement of the data. Intuitively, this means that the DAG data is produced by few data generating events whose effect percolates through the DAG. We prove identifiability in this new setting and show that the true DAG is the global minimizer of the $L^0$-norm of the vector of root causes. For data satisfying the few root causes assumption, we show superior performance compared to prior DAG learning methods.

----

## [738] Robust Learning for Smoothed Online Convex Optimization with Feedback Delay

**Authors**: *Pengfei Li, Jianyi Yang, Adam Wierman, Shaolei Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/36848567d39a5128e671ad04a6075374-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/36848567d39a5128e671ad04a6075374-Abstract-Conference.html)

**Abstract**:

We study a general form of Smoothed Online Convex Optimization, a.k.a. SOCO, including multi-step switching costs and feedback delay. We propose a novel machine learning (ML) augmented online algorithm, Robustness-Constrained Learning (RCL), which combines untrusted ML predictions with a trusted expert online algorithm via constrained projection to robustify the ML prediction. Specifically, we prove that RCL is able to guarantee $(1+\lambda)$-competitiveness against any given expert for any $\lambda>0$, while also explicitly training the ML model in a robustification-aware manner to improve the average-case performance. Importantly, RCL is the first ML-augmented algorithm with a provable robustness guarantee in the case of multi-step switching cost and feedback delay. We demonstrate the improvement of RCL in both robustness and average performance using battery management as a case study.

----

## [739] Scalarization for Multi-Task and Multi-Domain Learning at Scale

**Authors**: *Amelie Royer, Tijmen Blankevoort, Babak Ehteshami Bejnordi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/368559ed8ede03b21f624feaeb3a5867-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/368559ed8ede03b21f624feaeb3a5867-Abstract-Conference.html)

**Abstract**:

Training a single model on multiple input domains and/or output tasks allows for compressing information from multiple sources into a unified backbone hence improves model efficiency. It also enables potential positive knowledge transfer across tasks/domains, leading to improved accuracy and data-efficient training. However, optimizing such networks is a challenge, in particular due to discrepancies between the different tasks or domains: Despite several hypotheses and solutions proposed over the years, recent work has shown that uniform scalarization training, i.e., simply minimizing the average of the task losses,  yields on-par performance with more costly SotA optimization methods. This raises the issue of how well we understand the training dynamics of multi-task and multi-domain networks. In this work, we first devise a large-scale unified analysis of multi-domain and multi-task learning to better understand the dynamics of scalarization across varied task/domain combinations and model sizes. Following these insights, we then propose to leverage population-based training to efficiently search for the optimal scalarization weights when dealing with a large number of tasks or domains.

----

## [740] Causal discovery from observational and interventional data across multiple environments

**Authors**: *Adam Li, Amin Jaber, Elias Bareinboim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/368cba57d00902c752eaa9e4770bbbbe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/368cba57d00902c752eaa9e4770bbbbe-Abstract-Conference.html)

**Abstract**:

A fundamental problem in many sciences is the learning of causal structure underlying a system, typically through observation and experimentation. Commonly, one even collects data across multiple domains, such as gene sequencing from different labs, or neural recordings from different species. Although there exist methods for learning the equivalence class of causal diagrams from observational and experimental data, they are meant to operate in a single domain. In this paper, we develop a fundamental approach to structure learning in non-Markovian systems (i.e. when there exist latent confounders) leveraging observational and interventional data collected from multiple domains. Specifically, we start by showing that learning from observational data in multiple domains is equivalent to learning from interventional data with unknown targets in a single domain. But there are also subtleties when considering observational and experimental data. Using causal invariances derived from do-calculus, we define a property called S-Markov that connects interventional distributions from multiple-domains to graphical criteria on a selection diagram. Leveraging the S-Markov property, we introduce a new constraint-based causal discovery algorithm, S-FCI, that can learn from observational and interventional data from different domains. We prove that the algorithm is sound and subsumes existing constraint-based causal discovery algorithms.

----

## [741] Beyond Normal: On the Evaluation of Mutual Information Estimators

**Authors**: *Pawel Czyz, Frederic Grabowski, Julia E. Vogt, Niko Beerenwinkel, Alexander Marx*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/36b80eae70ff629d667f210e13497edf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/36b80eae70ff629d667f210e13497edf-Abstract-Conference.html)

**Abstract**:

Mutual information is a general statistical dependency measure which has found applications in representation learning, causality, domain generalization and computational biology. However, mutual information estimators are typically evaluated on simple families of probability distributions, namely multivariate normal distribution and selected distributions with one-dimensional random variables. In this paper, we show how to construct a diverse family of distributions with known ground-truth mutual information and propose a language-independent benchmarking platform for mutual information estimators. We discuss the general applicability and limitations of classical and neural estimators in settings involving high dimensions, sparse interactions, long-tailed distributions, and high mutual information. Finally, we provide guidelines for practitioners on how to select appropriate estimator adapted to the difficulty of problem considered and issues one needs to consider when applying an estimator to a new data set.

----

## [742] Structured Semidefinite Programming for Recovering Structured Preconditioners

**Authors**: *Arun Jambulapati, Jerry Li, Christopher Musco, Kirankumar Shiragur, Aaron Sidford, Kevin Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/36ce475705c1dc6c50a5956cedff3d01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/36ce475705c1dc6c50a5956cedff3d01-Abstract-Conference.html)

**Abstract**:

We develop a general framework for finding approximately-optimal preconditioners for solving linear systems. Leveraging this framework we obtain improved runtimes for fundamental preconditioning and linear system solving problems including:Diagonal preconditioning. We give an algorithm which, given positive definite $\mathbf{K} \in \mathbb{R}^{d \times d}$ with $\mathrm{nnz}(\mathbf{K})$ nonzero entries, computes an $\epsilon$-optimal diagonal preconditioner in time $\widetilde{O}(\mathrm{nnz}(\mathbf{K}) \cdot \mathrm{poly}(\kappa^\star,\epsilon^{-1}))$, where $\kappa^\star$ is the optimal condition number of the rescaled matrix.Structured linear systems. We give an algorithm which, given $\mathbf{M} \in \mathbb{R}^{d \times d}$ that is either the pseudoinverse of a graph Laplacian matrix or a constant spectral approximation of one, solves linear systems in $\mathbf{M}$ in $\widetilde{O}(d^2)$ time. Our diagonal preconditioning results improve state-of-the-art runtimes of $\Omega(d^{3.5})$ attained by general-purpose semidefinite programming, and our solvers improve state-of-the-art runtimes of $\Omega(d^{\omega})$ where $\omega > 2.3$ is the current matrix multiplication constant. We attain our results via new algorithms for a class of semidefinite programs (SDPs) we call matrix-dictionary approximation SDPs, which we leverage to solve an associated problem we call matrix-dictionary recovery.

----

## [743] Certifiably Robust Graph Contrastive Learning

**Authors**: *Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/37050ebbbd7096719ab96cec19a4c69f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/37050ebbbd7096719ab96cec19a4c69f-Abstract-Conference.html)

**Abstract**:

Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model. The source code of RES is available at https://github.com/ventr1c/RES-GCL.

----

## [744] FACE: Evaluating Natural Language Generation with Fourier Analysis of Cross-Entropy

**Authors**: *Zuhao Yang, Yingfang Yuan, Yang Xu, Shuo Zhan, Huajun Bai, Kefan Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/37094fdc81632915a5738293cf9b7ad4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/37094fdc81632915a5738293cf9b7ad4-Abstract-Conference.html)

**Abstract**:

Measuring the distance between machine-produced and human language is a critical open problem. Inspired by empirical findings from psycholinguistics on the periodicity of entropy in language, we propose FACE, a set of metrics based on Fourier Analysis of the estimated Cross-Entropy of language, for measuring the similarity between model-generated and human-written languages. Based on an open-ended generation task and the experimental data from previous studies, we find that FACE can effectively identify the human-model gap, scales with model size, reflects the outcomes of different sampling methods for decoding, correlates well with other evaluation metrics and with human judgment scores.

----

## [745] 3D Copy-Paste: Physically Plausible Object Insertion for Monocular 3D Detection

**Authors**: *Yunhao Ge, Hong-Xing Yu, Cheng Zhao, Yuliang Guo, Xinyu Huang, Liu Ren, Laurent Itti, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/370fa2e691f57eb319bc263a07dad4a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/370fa2e691f57eb319bc263a07dad4a5-Abstract-Conference.html)

**Abstract**:

A major challenge in monocular 3D object detection is the limited diversity and quantity of objects in real datasets. While augmenting real scenes with virtual objects holds promise to improve both the diversity and quantity of the objects, it remains elusive due to the lack of an effective 3D object insertion method in complex real captured scenes. In this work, we study augmenting complex real indoor scenes with virtual objects for monocular 3D object detection. The main challenge is to automatically identify plausible physical properties for virtual assets (e.g., locations, appearances, sizes, etc.) in cluttered real scenes. To address this challenge, we propose a physically plausible indoor 3D object insertion approach to automatically copy virtual objects and paste them into real scenes. The resulting objects in scenes have 3D bounding boxes with plausible physical locations and appearances. In particular, our method first identifies physically feasible locations and poses for the inserted objects to prevent collisions with the existing room layout. Subsequently, it estimates spatially-varying illumination for the insertion location, enabling the immersive blending of the virtual objects into the original scene with plausible appearances and cast shadows. We show that our augmentation method significantly improves existing monocular 3D object models and achieves state-of-the-art performance. For the first time, we demonstrate that a physically plausible 3D object insertion, serving as a generative data augmentation technique, can lead to significant improvements for discriminative downstream tasks such as monocular 3D object detection. Project website: https://gyhandy.github.io/3D-Copy-Paste/.

----

## [746] Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions

**Authors**: *Stefano Massaroli, Michael Poli, Daniel Y. Fu, Hermann Kumbong, Rom N. Parnichkun, David W. Romero, Aman Timalsina, Quinn McIntyre, Beidi Chen, Atri Rudra, Ce Zhang, Christopher Ré, Stefano Ermon, Yoshua Bengio*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/371355cd42caaf83412c3fbef4688979-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/371355cd42caaf83412c3fbef4688979-Abstract-Conference.html)

**Abstract**:

Recent advances in attention-free sequence models rely on convolutions as alternatives to the attention operator at the core of Transformers. In particular, long convolution sequence models have achieved state-of-the-art performance in many domains, but incur a significant cost during auto-regressive inference workloads -- naively requiring a full pass (or caching of activations) over the input sequence for each generated token -- similarly to attention-based models. In this paper, we seek to enable $\mathcal O(1)$ compute and memory cost per token in any pre-trained long convolution architecture to reduce memory footprint and increase throughput during generation. Concretely, our methods consist in extracting low-dimensional linear state-space models from each convolution layer, building upon rational interpolation and model-order reduction techniques. We further introduce architectural improvements to convolution-based layers such as Hyena: by weight-tying the filters across channels into heads, we achieve higher pre-training quality and reduce the number of filters to be distilled. The resulting model achieves 10x higher throughput than Transformers and 1.5x higher than Hyena at 1.3B parameters, without any loss in quality after distillation.

----

## [747] Incomplete Multimodality-Diffused Emotion Recognition

**Authors**: *Yuanzhi Wang, Yong Li, Zhen Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/372cb7805eaccb2b7eed641271a30eec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/372cb7805eaccb2b7eed641271a30eec-Abstract-Conference.html)

**Abstract**:

Human multimodal emotion recognition (MER) aims to perceive and understand human emotions via various heterogeneous modalities, such as language, vision, and acoustic. Compared with unimodality, the complementary information in the multimodalities facilitates robust emotion understanding. Nevertheless,  in real-world scenarios, the missing modalities hinder multimodal understanding and result in degraded MER performance. In this paper, we propose an Incomplete Multimodality-Diffused emotion recognition (IMDer) method to mitigate the challenge of MER under incomplete multimodalities. To recover the missing modalities, IMDer exploits the score-based diffusion model that maps the input Gaussian noise into the desired distribution space of the missing modalities and recovers missing data abided by their original distributions. Specially, to reduce semantic ambiguity between the missing and the recovered modalities, the available modalities are embedded as the condition to guide and refine the diffusion-based recovering process. In contrast to previous work,  the diffusion-based modality recovery mechanism in IMDer allows to simultaneously reach both distribution consistency and semantic disambiguation. Feature visualization of the recovered modalities illustrates the consistent modality-specific distribution and semantic alignment. Besides, quantitative experimental results verify that IMDer obtains state-of-the-art MER accuracy under various missing modality patterns.

----

## [748] Diffusion-Based Probabilistic Uncertainty Estimation for Active Domain Adaptation

**Authors**: *Zhekai Du, Jingjing Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html)

**Abstract**:

Active Domain Adaptation (ADA) has emerged as an attractive technique for assisting domain adaptation by actively annotating a small subset of target samples. Most ADA methods focus on measuring the target representativeness beyond traditional active learning criteria to handle the domain shift problem, while leaving the uncertainty estimation to be performed by an uncalibrated deterministic model. In this work, we introduce a probabilistic framework that captures both data-level and prediction-level uncertainties beyond a point estimate. Specifically, we use variational inference to approximate the joint posterior distribution of latent representation and model prediction. The variational objective of labeled data can be formulated by a variational autoencoder and a latent diffusion classifier, and the objective of unlabeled data can be implemented in a knowledge distillation framework. We utilize adversarial learning to ensure an invariant latent space. The resulting diffusion classifier enables efficient sampling of all possible predictions for each individual to recover the predictive distribution. We then leverage a t-test-based criterion upon the sampling and select informative unlabeled target samples based on the p-value, which encodes both prediction variability and cross-category ambiguity. Experiments on both ADA and Source-Free ADA settings show that our method provides more calibrated predictions than previous ADA methods and achieves favorable performance on three domain adaptation datasets.

----

## [749] Augmented Memory Replay-based Continual Learning Approaches for Network Intrusion Detection

**Authors**: *Suresh Kumar Amalapuram, Sumohana S. Channappayya, Bheemarjuna Reddy Tamma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3755a02b1035fbadd5f93a022170e46f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3755a02b1035fbadd5f93a022170e46f-Abstract-Conference.html)

**Abstract**:

Intrusion detection is a form of anomalous activity detection in communication network traffic. Continual learning (CL) approaches to the intrusion detection task accumulate old knowledge while adapting to the latest threat knowledge. Previous works have shown the effectiveness of memory replay-based CL approaches for this task. In this work, we present two novel contributions to improve the performance of CL-based network intrusion detection in the context of class imbalance and scalability. First, we extend class balancing reservoir sampling (CBRS), a memory-based CL method, to address the problems of severe class imbalance for large datasets. Second, we propose a novel approach titled perturbation assistance for parameter approximation (PAPA) based on the Gaussian mixture model to reduce the number of \textit{virtual stochastic gradient descent (SGD) parameter} computations needed to discover maximally interfering samples for CL. We demonstrate that the proposed approaches perform remarkably better than the baselines on standard intrusion detection benchmarks created over shorter periods (KDDCUP'99, NSL-KDD, CICIDS-2017/2018, UNSW-NB15, and CTU-13) and a longer period with distribution shift (AnoShift). We also validated proposed approaches on standard continual learning benchmarks (SVHN, CIFAR-10/100, and CLEAR-10/100) and anomaly detection benchmarks (SMAP, SMD, and MSL). Further, the proposed PAPA approach significantly lowers the number of virtual SGD update operations, thus resulting in training time savings in the range of 12 to 40\% compared to the maximally interfered samples retrieval algorithm.

----

## [750] Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models

**Authors**: *Alvin Heng, Harold Soh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/376276a95781fa17c177b1ccdd0a03ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/376276a95781fa17c177b1ccdd0a03ac-Abstract-Conference.html)

**Abstract**:

The recent proliferation of large-scale text-to-image models has led to growing concerns that such models may be misused to generate harmful, misleading, and inappropriate content. Motivated by this issue, we derive a technique inspired by continual learning to selectively forget concepts in pretrained deep generative models. Our method, dubbed Selective Amnesia, enables controllable forgetting  where a user can specify how a concept should be forgotten. Selective Amnesia can be applied to conditional variational likelihood models, which encompass a variety of popular deep generative frameworks, including variational autoencoders and large-scale text-to-image diffusion models. Experiments across different models demonstrate that our approach induces forgetting on a variety of concepts, from entire classes in standard datasets to celebrity and nudity prompts in text-to-image models.

----

## [751] Structure from Duplicates: Neural Inverse Graphics from a Pile of Objects

**Authors**: *Tianhang Cheng, Wei-Chiu Ma, Kaiyu Guan, Antonio Torralba, Shenlong Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3764a9c8abc84c7482f778fefc24f10b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3764a9c8abc84c7482f778fefc24f10b-Abstract-Conference.html)

**Abstract**:

Abstract Our world is full of identical objects (\emph{e.g.}, cans of coke, cars of same model). These duplicates, when seen together, provide additional and strong cues for us to effectively reason about 3D. Inspired by this observation, we introduce Structure from Duplicates (SfD), a novel inverse graphics framework that reconstructs geometry, material, and illumination from a single image containing multiple identical objects. SfD begins by identifying multiple instances of an object within an image, and then jointly estimates the 6DoF pose for all instances. An inverse graphics pipeline is subsequently employed to jointly reason about the shape, material of the object, and the environment light, while adhering to the shared geometry and material constraint across instances.Our primary contributions involve utilizing object duplicates as a robust prior for single-image inverse graphics and proposing an in-plane rotation-robust Structure from Motion (SfM) formulation for joint 6-DoF object pose estimation. By leveraging multi-view cues from a single image, SfD generates more realistic and detailed 3D reconstructions, significantly outperforming existing single image reconstruction models and multi-view reconstruction approaches with a similar or greater number of observations.

----

## [752] Weakly-Supervised Audio-Visual Segmentation

**Authors**: *Shentong Mo, Bhiksha Raj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/377b2e39e97e917b9e625b35241e33df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/377b2e39e97e917b9e625b35241e33df-Abstract-Conference.html)

**Abstract**:

Audio-visual segmentation is a challenging task that aims to predict pixel-level masks for sound sources in a video.  Previous work applied a comprehensive manually designed architecture with countless pixel-wise accurate masks as supervision.  However, these pixel-level masks are expensive and not available in all cases.  In this work, we aim to simplify the supervision as the instance-level annotation, $\textit{i.e.}$, weakly-supervised audio-visual segmentation.  We present a novel Weakly-Supervised Audio-Visual Segmentation framework, namely WS-AVS, that can learn multi-scale audio-visual alignment with multi-scale multiple-instance contrastive learning for audio-visual segmentation.  Extensive experiments on AVSBench demonstrate the effectiveness of our WS-AVS in the weakly-supervised audio-visual segmentation of single-source and multi-source scenarios.

----

## [753] Adversarial Examples Are Not Real Features

**Authors**: *Ang Li, Yifei Wang, Yiwen Guo, Yisen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/378b284f7f03274d1bf5322bb15c5c16-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/378b284f7f03274d1bf5322bb15c5c16-Abstract-Conference.html)

**Abstract**:

The existence of adversarial examples has been a mystery for years and attracted much interest. A well-known theory by \citet{ilyas2019adversarial} explains adversarial vulnerability from a data perspective by showing that one can extract non-robust features from adversarial examples and these features alone are useful for classification. However, the explanation remains quite counter-intuitive since non-robust features are mostly noise features to humans. In this paper, we re-examine the theory from a larger context by incorporating multiple learning paradigms. Notably, we find that contrary to their good usefulness under supervised learning, non-robust features attain poor usefulness when transferred to other self-supervised learning paradigms, such as contrastive learning, masked image modeling, and diffusion models. It reveals that non-robust features are not really as useful as robust or natural features that enjoy good transferability between these paradigms. Meanwhile, for robustness, we also show that naturally trained encoders from robust features are largely non-robust under AutoAttack. Our cross-paradigm examination suggests that the non-robust features are not really useful but more like paradigm-wise shortcuts, and robust features alone might be insufficient to attain reliable model robustness. Code is available at \url{https://github.com/PKU-ML/AdvNotRealFeatures}.

----

## [754] A Comprehensive Study on Text-attributed Graphs: Benchmarking and Rethinking

**Authors**: *Hao Yan, Chaozhuo Li, Ruosong Long, Chao Yan, Jianan Zhao, Wenwen Zhuang, Jun Yin, Peiyan Zhang, Weihao Han, Hao Sun, Weiwei Deng, Qi Zhang, Lichao Sun, Xing Xie, Senzhang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/37d00f567a18b478065f1a91b95622a0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/37d00f567a18b478065f1a91b95622a0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Text-attributed graphs (TAGs) are prevalent in various real-world scenarios, where each node is associated with a text description. The cornerstone of representation learning on TAGs lies in the seamless integration of textual semantics within individual nodes and the topological connections across nodes. Recent advancements in pre-trained language models (PLMs) and graph neural networks (GNNs) have facilitated effective learning on TAGs, garnering increased research interest. However, the absence of meaningful benchmark datasets and standardized evaluation procedures for TAGs has impeded progress in this field. In this paper, we propose CS-TAG, a comprehensive and diverse collection of challenging benchmark datasets for TAGs. The CS-TAG datasets are notably large in scale and encompass a wide range of domains, spanning from citation networks to purchase graphs. In addition to building the datasets,  we conduct extensive benchmark experiments over CS-TAG with various learning paradigms, including PLMs, GNNs, PLM-GNN co-training methods, and the proposed novel topological pre-training of language models. In a nutshell, we provide an overview of the CS-TAG datasets, standardized evaluation procedures, and present baseline experiments. The entire CS-TAG project is publicly accessible at \url{https://github.com/sktsherlock/TAG-Benchmark}.

----

## [755] Online Ad Allocation with Predictions

**Authors**: *Fabian Spaeh, Alina Ene*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3815d62554efad0878fad6c1c30ffda0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3815d62554efad0878fad6c1c30ffda0-Abstract-Conference.html)

**Abstract**:

Display Ads and the generalized assignment problem are two well-studied online packing problems with important applications in ad allocation and other areas. In both problems, ad impressions arrive online and have to be allocated immediately to budget-constrained advertisers. Worst-case algorithms that achieve the ideal competitive ratio are known for both problems, but might act overly conservative given the predictable and usually tame nature of real-world input. Given this discrepancy, we develop an algorithm for both problems that incorporate machine-learned predictions and can thus improve the performance beyond the worst-case. Our algorithm is based on the work of Feldman et al. (2009) and similar in nature to Mahdian et al. (2007) who were the first to develop a learning-augmented algorithm for the related, but more structured Ad Words problem. We use a novel analysis to show that our algorithm is able to capitalize on a good prediction, while being robust against poor predictions. We experimentally evaluate our algorithm on synthetic and real-world data on a wide range of predictions. Our algorithm is consistently outperforming the worst-case algorithm without predictions.

----

## [756] Transfer Learning with Affine Model Transformation

**Authors**: *Shunya Minami, Kenji Fukumizu, Yoshihiro Hayashi, Ryo Yoshida*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3819a070922cc0d19f3d66ce108f28e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3819a070922cc0d19f3d66ce108f28e0-Abstract-Conference.html)

**Abstract**:

Supervised transfer learning has received considerable attention due to its potential to boost the predictive power of machine learning in scenarios where data are scarce. Generally, a given set of source models and a dataset from a target domain are used to adapt the pre-trained models to a target domain by statistically learning domain shift and domain-specific factors. While such procedurally and intuitively plausible methods have achieved great success in a wide range of real-world applications, the lack of a theoretical basis hinders further methodological development. This paper presents a general class of transfer learning regression called affine model transfer, following the principle of expected-square loss minimization. It is shown that the affine model transfer broadly encompasses various existing methods, including the most common procedure based on neural feature extractors. Furthermore, the current paper clarifies theoretical properties of the affine model transfer such as generalization error and excess risk. Through several case studies, we demonstrate the practical benefits of modeling and estimating inter-domain commonality and domain-specific factors separately with the affine-type transfer models.

----

## [757] Towards Robust and Expressive Whole-body Human Pose and Shape Estimation

**Authors**: *Hui En Pang, Zhongang Cai, Lei Yang, Qingyi Tao, Zhonghua Wu, Tianwei Zhang, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/381d36bf8e115cdeda48763c9cb77616-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/381d36bf8e115cdeda48763c9cb77616-Abstract-Conference.html)

**Abstract**:

Whole-body pose and shape estimation aims to jointly predict different behaviors (e.g., pose, hand gesture, facial expression) of the entire human body from a monocular image. Existing methods often exhibit suboptimal performance due to the complexity of in-the-wild scenarios. We argue that the prediction accuracy of these models is significantly affected by the quality of the bounding box, e.g., scale, alignment. The natural discrepancy between the ideal bounding box annotations and model detection results is particularly detrimental to the performance of whole-body pose and shape estimation.In this paper, we propose a novel framework to enhance the robustness of whole-body pose and shape estimation. Our framework incorporates three new modules to address the above challenges from three perspectives: (1) a Localization Module enhances the model's awareness of the subject's location and semantics within the image space; (2) a Contrastive Feature Extraction Module encourages the model to be invariant to robust augmentations by incorporating a contrastive loss and positive samples; (3) a Pixel Alignment Module ensures the reprojected mesh from the predicted camera and body model parameters are more accurate and pixel-aligned. We perform comprehensive experiments to demonstrate the effectiveness of our proposed framework on body, hands, face and whole-body benchmarks.

----

## [758] Neural Lad: A Neural Latent Dynamics Framework for Times Series Modeling

**Authors**: *Ting Li, Jianguo Li, Zhanxing Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/382a8606a85ca6ec7c06185a1a95ce8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/382a8606a85ca6ec7c06185a1a95ce8b-Abstract-Conference.html)

**Abstract**:

Neural ordinary differential equation (Neural ODE) is an elegant yet powerful framework to learn the temporal dynamics for time series modeling.However, we observe that existing Neural ODE forecasting models suffer from two disadvantages:i) controlling the latent states only through the linear transformation over the local change of the observed signals may be inadequate;ii) lacking the ability to capture the inherent periodical property in time series forecasting tasks;To overcome the two issues, we introduce a new neural ODE framework called \textbf{Neural Lad}, a \textbf{Neural} \textbf{La}tent \textbf{d}ynamics model in which the latent representations evolve with an ODE enhanced by the change of observed signal and seasonality-trend characterization. We incorporate the local change of input signal into the latent dynamics in an attention-based manner and design a residual architecture over basis expansion to depict the periodicity in the underlying dynamics. To accommodate the multivariate time series forecasting, we extend the  Neural Lad through learning an adaptive relationship between multiple time series.        Experiments demonstrate that our model can achieve better or comparable performance against existing neural ODE families and transformer variants in various datasets. Remarkably, the empirical superiority of Neural Lad is consistent across  short and long-horizon forecasting for both univariate, multivariate and even irregular sampled time series.

----

## [759] Weighted ROC Curve in Cost Space: Extending AUC to Cost-Sensitive Learning

**Authors**: *Huiyang Shao, Qianqian Xu, Zhiyong Yang, Peisong Wen, Peifeng Gao, Qingming Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38687a6a01f127d6db92561508a225b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38687a6a01f127d6db92561508a225b7-Abstract-Conference.html)

**Abstract**:

In this paper, we aim to tackle flexible cost requirements for long-tail datasets, where we need to construct a (a) cost-sensitive and (b) class-distribution robust learning framework. The misclassification cost and the area under the ROC curve (AUC) are popular metrics for (a) and (b), respectively. However, limited by their formulations, models trained with AUC cannot be applied to cost-sensitive decision problems, and models trained with fixed costs are sensitive to the class distribution shift. To address this issue, we present a new setting where costs are treated like a dataset to deal with arbitrarily unknown cost distributions. Moreover, we propose a novel weighted version of AUC where the cost distribution can be integrated into its calculation through decision thresholds.  To formulate this setting, we propose a novel bilevel paradigm to bridge weighted AUC (WAUC) and cost. The inner-level problem approximates the optimal threshold from sampling costs, and the outer-level problem minimizes the WAUC loss over the optimal threshold distribution. To optimize this bilevel paradigm, we employ a stochastic optimization algorithm (SACCL) to optimize it. Finally, experiment results show that our algorithm performs better than existing cost-sensitive learning methods and two-stage AUC decisions approach.

----

## [760] Dynamic Non-monotone Submodular Maximization

**Authors**: *Kiarash Banihashem, Leyla Biabani, Samira Goudarzi, MohammadTaghi Hajiaghayi, Peyman Jabbarzade, Morteza Monemizadeh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/387982dbf23d9975c7fc45813dd3dabc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/387982dbf23d9975c7fc45813dd3dabc-Abstract-Conference.html)

**Abstract**:

Maximizing submodular functions has been increasingly used in many applications of machine learning, such as data summarization, recommendation systems, and feature selection. Moreover, there has been a growing interest in both submodular maximization and dynamic algorithms. In 2020, Monemizadeh and Lattanzi, Mitrovic, Norouzi-Fard, Tarnawski, and Zadimoghaddam initiated developing dynamic algorithms for the monotone submodular maximization problem under the cardinality constraint $k$. In 2022, Chen and Peng studied the complexity of this problem and raised an important open question: "\emph{Can we extend [fully dynamic] results (algorithm or hardness) to non-monotone submodular maximization?}". We affirmatively answer their question by demonstrating a reduction from maximizing a non-monotone submodular function under the cardinality constraint $k$ to maximizing a monotone submodular function under the same constraint. Through this reduction, we obtain the first dynamic algorithms to solve the non-monotone submodular maximization problem under the cardinality constraint $k$. Our algorithms maintain an $(8+\epsilon)$-approximate of the solution and use expected amortized $O(\epsilon^{-3}k^3\log^3(n)\log(k))$ or $O(\epsilon^{-1}k^2\log^3(k))$ oracle queries per update, respectively. Furthermore, we showcase the benefits of our dynamic algorithm for video summarization and max-cut problems on several real-world data sets.

----

## [761] Semi-Implicit Denoising Diffusion Models (SIDDMs)

**Authors**: *Yanwu Xu, Mingming Gong, Shaoan Xie, Wei Wei, Matthias Grundmann, Kayhan Batmanghelich, Tingbo Hou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3882ca2c952276247fe9a993193b00e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3882ca2c952276247fe9a993193b00e4-Abstract-Conference.html)

**Abstract**:

Despite the proliferation of generative models, achieving fast sampling during inference without compromising sample diversity and quality remains challenging. Existing models such as Denoising Diffusion Probabilistic Models (DDPM) deliver high-quality, diverse samples but are slowed by an inherently high number of iterative steps. The Denoising Diffusion Generative Adversarial Networks (DDGAN) attempted to circumvent this limitation by integrating a GAN model for larger jumps in the diffusion process. However, DDGAN encountered scalability limitations when applied to large datasets. To address these limitations, we introduce a novel approach that tackles the problem by matching implicit and explicit factors. More specifically, our approach involves utilizing an implicit model to match the marginal distributions of noisy data and the explicit conditional distribution of the forward diffusion. This combination allows us to effectively match the joint denoising distributions. Unlike DDPM but similar to DDGAN, we do not enforce a parametric distribution for the reverse step, enabling us to take large steps during inference. Similar to the DDPM but unlike DDGAN, we take advantage of the exact form of the diffusion process. We demonstrate that our proposed method obtains comparable generative performance to diffusion-based models and vastly superior results to models with a small number of sampling steps.

----

## [762] Implicit Convolutional Kernels for Steerable CNNs

**Authors**: *Maksim Zhdanov, Nico Hoffmann, Gabriele Cesa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/389a55c90f839d58188060a42bb9138a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/389a55c90f839d58188060a42bb9138a-Abstract-Conference.html)

**Abstract**:

Steerable convolutional neural networks (CNNs) provide a general framework for building neural networks equivariant to translations and transformations of an origin-preserving group $G$, such as reflections and rotations. They rely on standard convolutions with $G$-steerable kernels obtained by analytically solving the group-specific equivariance constraint imposed onto the kernel space. As the solution is tailored to a particular group $G$, implementing a kernel basis does not generalize to other symmetry transformations, complicating the development of general group equivariant models. We propose using implicit neural representation via multi-layer perceptrons (MLPs) to parameterize $G$-steerable kernels. The resulting framework offers a simple and flexible way to implement Steerable CNNs and generalizes to any group $G$ for which a $G$-equivariant MLP can be built. We prove the effectiveness of our method on multiple tasks, including N-body simulations, point cloud classification and molecular property prediction.

----

## [763] Block Low-Rank Preconditioner with Shared Basis for Stochastic Optimization

**Authors**: *Jui-Nan Yen, Sai Surya Duvvuri, Inderjit S. Dhillon, Cho-Jui Hsieh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/389cfad711d2b1e2128e931feee80230-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/389cfad711d2b1e2128e931feee80230-Abstract-Conference.html)

**Abstract**:

Adaptive methods with non-diagonal preconditioning have shown state-of-the-art results on various tasks. However, their computational complexity and memory requirement makes it challenging to scale these methods to modern neural network architectures. To address this challenge, some previous works have adopted block-diagonal preconditioners. However, the memory cost of storing the block-diagonal matrix remains substantial, leading to the use of smaller block sizes and ultimately resulting in suboptimal performance. To reduce the time and memory complexity without sacrificing performance, we propose approximating each diagonal block of the second moment matrix by low-rank matrices and enforcing the same basis for the blocks within each layer.  We provide theoretical justification for such sharing and design an algorithm to efficiently maintain this shared-basis block low-rank approximation during training. Our results on a deep autoencoder and a transformer benchmark demonstrate that the proposed method outperforms first-order methods with slightly more time and memory usage, while also achieving competitive or superior performance compared to other second-order methods with less time and memory usage.

----

## [764] Learning in the Presence of Low-dimensional Structure: A Spiked Random Matrix Perspective

**Authors**: *Jimmy Ba, Murat A. Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38a1671ab0747b6ffe4d1c6ef117a3a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38a1671ab0747b6ffe4d1c6ef117a3a9-Abstract-Conference.html)

**Abstract**:

We consider the learning of a single-index target function $f_*: \mathbb{R}^d\to\mathbb{R}$ under spiked covariance data: $$f_*(\boldsymbol{x}) = \textstyle\sigma_*(\frac{1}{\sqrt{1+\theta}}\langle\boldsymbol{x},\boldsymbol{\mu}\rangle), ~~ \boldsymbol{x}\overset{\small\mathrm{i.i.d.}}{\sim}\mathcal{N}(0,\boldsymbol{I_d} + \theta\boldsymbol{\mu}\boldsymbol{\mu}^\top), ~~ \theta\asymp d^{\beta} \text{ for } \beta\in[0,1), $$ where the link function $\sigma_*:\mathbb{R}\to\mathbb{R}$ is a degree-$p$ polynomial with information exponent $k$ (defined as the lowest degree in the Hermite expansion of $\sigma_*$), and it depends on the projection of input $\boldsymbol{x}$ onto the spike (signal) direction $\boldsymbol{\mu}\in\mathbb{R}^d$. In the proportional asymptotic limit where the number of training examples $n$ and the dimensionality $d$ jointly diverge: $n,d\to\infty, n/d\to\psi\in(0,\infty)$, we ask the following question: how large should the spike magnitude $\theta$ (i.e., the strength of the low-dimensional component) be, in order for $(i)$ kernel methods, $(ii)$ neural networks optimized by gradient descent, to learn $f_*$? We show that for kernel ridge regression, $\beta\ge 1-\frac{1}{p}$ is both sufficient and necessary. Whereas for two-layer neural networks trained with gradient descent, $\beta>1-\frac{1}{k}$ suffices. Our results demonstrate that both kernel methods and neural networks benefit from low-dimensional structures in the data. Further, since $k\le p$ by definition, neural networks can adapt to such structures more effectively.

----

## [765] Efficient Neural Music Generation

**Authors**: *Max W. Y. Lam, Qiao Tian, Tang Li, Zongyu Yin, Siyuan Feng, Ming Tu, Yuliang Ji, Rui Xia, Mingbo Ma, Xuchen Song, Jitong Chen, Yuping Wang, Yuxuan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38b23e2328096520e9c889ae03e372c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38b23e2328096520e9c889ae03e372c9-Abstract-Conference.html)

**Abstract**:

Recent progress in music generation has been remarkably advanced by the state-of-the-art MusicLM, which comprises a hierarchy of three LMs, respectively, for semantic, coarse acoustic, and fine acoustic modelings. Yet, sampling with the MusicLM requires processing through these LMs one by one to obtain the fine-grained acoustic tokens, making it computationally expensive and prohibitive for a real-time generation. Efficient music generation with a quality on par with MusicLM remains a significant challenge.In this paper, we present MeLoDy (M for music; L for LM; D for diffusion), an LM-guided diffusion model that generates music audios of state-of-the-art quality meanwhile reducing 95.7\% to 99.6\% forward passes in MusicLM, respectively, for sampling 10s to 30s music. MeLoDy inherits the highest-level LM from MusicLM for semantic modeling, and applies a novel dual-path diffusion (DPD) model and an audio VAE-GAN to efficiently decode the conditioning semantic tokens into waveform. DPD is proposed to simultaneously model the coarse and fine acoustics by incorporating the semantic information into segments of latents effectively via cross-attention at each denoising step. Our experimental results suggest the superiority of MeLoDy, not only in its practical advantages on sampling speed and infinitely continuable generation, but also in its state-of-the-art musicality, audio quality, and text correlation.Our samples are available at https://Efficient-MeLoDy.github.io/.

----

## [766] Crystal Structure Prediction by Joint Equivariant Diffusion

**Authors**: *Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, Yang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html)

**Abstract**:

Crystal Structure Prediction (CSP) is crucial in various scientific disciplines. While CSP can be addressed by employing currently-prevailing generative models (e.g. diffusion models), this task encounters unique challenges owing to the symmetric geometry of crystal structures---the invariance of translation, rotation, and periodicity. To incorporate the above symmetries, this paper proposes DiffCSP, a novel diffusion model to learn the structure distribution from stable crystals. To be specific, DiffCSP jointly generates the lattice and atom coordinates for each crystal by employing a periodic-E(3)-equivariant denoising model, to better model the crystal geometry. Notably, different from related equivariant generative approaches, DiffCSP leverages fractional coordinates other than Cartesian coordinates to represent crystals, remarkably promoting the diffusion and the generation process of atom positions. Extensive experiments verify that our DiffCSP remarkably outperforms existing CSP methods, with a much lower computation cost in contrast to DFT-based methods. Moreover, the superiority of DiffCSP is still observed when it is extended for ab initio crystal generation.

----

## [767] Understanding the detrimental class-level effects of data augmentation

**Authors**: *Polina Kirichenko, Mark Ibrahim, Randall Balestriero, Diane Bouchacourt, Shanmukha Ramakrishna Vedantam, Hamed Firooz, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38c05a5410a6ab7eeeb26c9dbebbc41b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38c05a5410a6ab7eeeb26c9dbebbc41b-Abstract-Conference.html)

**Abstract**:

Data augmentation (DA) encodes invariance and provides implicit regularization critical to a model's performance in image classification tasks. However, while DA improves average accuracy, recent studies have shown that its impact can be highly class dependent: achieving optimal average accuracy comes at the cost of significantly hurting individual class accuracy by as much as 20% on ImageNet. There has been little progress in resolving class-level accuracy drops due to a limited understanding of these effects. In this work, we present a framework for understanding how DA interacts with class-level learning dynamics. Using higher-quality multi-label annotations on ImageNet, we systematically categorize the affected classes and find that the majority are inherently ambiguous, co-occur, or involve fine-grained distinctions, while DA controls the model's bias towards one of the closely related classes. While many of the previously reported performance drops are explained by multi-label annotations, we identify other sources of accuracy degradations by analyzing class confusions. We show that simple class-conditional augmentation strategies informed by our framework improve performance on the negatively affected classes.

----

## [768] Optimal Guarantees for Algorithmic Reproducibility and Gradient Complexity in Convex Optimization

**Authors**: *Liang Zhang, Junchi Yang, Amin Karbasi, Niao He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38c5feed4b72c96f6cf925ccc9832ecf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38c5feed4b72c96f6cf925ccc9832ecf-Abstract-Conference.html)

**Abstract**:

Algorithmic reproducibility measures the deviation in outputs of machine learning algorithms upon minor changes in the training process. Previous work suggests that first-order methods would need to trade-off convergence rate (gradient complexity) for better reproducibility. In this work, we challenge this perception and demonstrate that both optimal reproducibility and near-optimal convergence guarantees can be achieved for smooth convex minimization and smooth convex-concave minimax problems under various error-prone oracle settings. Particularly, given the inexact initialization oracle, our regularization-based algorithms achieve the best of both worlds -- optimal reproducibility and near-optimal gradient complexity -- for minimization and minimax optimization. With the inexact gradient oracle, the near-optimal guarantees also hold for minimax optimization. Additionally, with the stochastic gradient oracle, we show that stochastic gradient descent ascent is optimal in terms of both reproducibility and gradient complexity. We believe our results contribute to an enhanced understanding of the reproducibility-convergence trade-off in the context of convex optimization.

----

## [769] Test-time Adaptation of Discriminative Models via Diffusion Generative Feedback

**Authors**: *Mihir Prabhudesai, Tsung-Wei Ke, Alexander C. Li, Deepak Pathak, Katerina Fragkiadaki*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38e511a690709603d4cc3a1c52b4a9fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38e511a690709603d4cc3a1c52b4a9fd-Abstract-Conference.html)

**Abstract**:

The advancements in generative modeling, particularly the advent of diffusion models, have sparked a fundamental question: how can these models be effectively used for discriminative tasks? In this work, we find that generative models can be great test-time adapters for discriminative models. Our method, Diffusion-TTA, adapts pre-trained discriminative models such as image classifiers, segmenters and depth predictors, to each unlabelled example in the test set using generative feedback from a diffusion model. We achieve this by modulating the conditioning of the diffusion model using the output of the discriminative model. We then maximize the image likelihood objective by backpropagating the gradients to discriminative modelâ€™s parameters. We show Diffusion-TTA significantly enhances the accuracy of various large-scale pre-trained discriminative models, such as, ImageNet classifiers, CLIP models, image pixel labellers and image depth predictors. Diffusion-TTA outperforms existing test-time adaptation methods, including TTT-MAE and TENT, and particularly shines in online adaptation setups, where the discriminative model is continually adapted to each example in the test set. We provide access to code, results, and visualizations on our website: diffusion-tta.github.io/

----

## [770] Fragment-based Pretraining and Finetuning on Molecular Graphs

**Authors**: *Kha-Dinh Luong, Ambuj K. Singh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38ec60a949c3538e5cbb337b1b386dcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38ec60a949c3538e5cbb337b1b386dcf-Abstract-Conference.html)

**Abstract**:

Property prediction on molecular graphs is an important application of Graph Neural Networks (GNNs). Recently, unlabeled molecular data has become abundant, which facilitates the rapid development of self-supervised learning for GNNs in the chemical domain. In this work, we propose pretraining GNNs at the fragment level, a promising middle ground to overcome the limitations of node-level and graph-level pretraining. Borrowing techniques from recent work on principal subgraph mining, we obtain a compact vocabulary of prevalent fragments from a large pretraining dataset. From the extracted vocabulary, we introduce several fragment-based contrastive and predictive pretraining tasks. The contrastive learning task jointly pretrains two different GNNs: one on molecular graphs and the other on fragment graphs, which represents higher-order connectivity within molecules. By enforcing consistency between the fragment embedding and the aggregated embedding of the corresponding atoms from the molecular graphs, we ensure that the embeddings capture structural information at multiple resolutions. The structural information of fragment graphs is further exploited to extract auxiliary labels for graph-level predictive pretraining. We employ both the pretrained molecular-based and fragment-based GNNs for downstream prediction, thus utilizing the fragment information during finetuning. Our graph fragment-based pretraining (GraphFP) advances the performances on 5 out of 8 common molecular benchmarks and improves the performances on long-range biological benchmarks by at least 11.5%. Code is available at: https://github.com/lvkd84/GraphFP.

----

## [771] Characterizing Out-of-Distribution Error via Optimal Transport

**Authors**: *Yuzhe Lu, Yilong Qin, Runtian Zhai, Andrew Shen, Ketong Chen, Zhenlin Wang, Soheil Kolouri, Simon Stepputtis, Joseph Campbell, Katia P. Sycara*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38fd51cf36f28566230a93a5fbeaabbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38fd51cf36f28566230a93a5fbeaabbf-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) data poses serious challenges in deployed machine learning models,so methods of predicting a model's performance on OOD data without labels are important for machine learning safety.While a number of methods have been proposed by prior work, they often underestimate the actual error, sometimes by a large margin, which greatly impacts their applicability to real tasks. In this work, we identify pseudo-label shift, or the difference between the predicted and true OOD label distributions, as a key indicator of this underestimation. Based on this observation, we introduce a novel method for estimating model performance by leveraging optimal transport theory, Confidence Optimal Transport (COT), and show that it provably provides more robust error estimates in the presence of pseudo-label shift. Additionally, we introduce an empirically-motivated variant of COT, Confidence Optimal Transport with Thresholding (COTT), which applies thresholding to the individual transport costs and further improves the accuracy of COT's error estimates. We evaluate COT and COTT on a variety of standard benchmarks that induce various types of distribution shift -- synthetic, novel subpopulation, and natural -- and show that our approaches significantly outperform existing state-of-the-art methods with up to 3x lower prediction errors.

----

## [772] Unsupervised Video Domain Adaptation for Action Recognition: A Disentanglement Perspective

**Authors**: *Pengfei Wei, Lingdong Kong, Xinghua Qu, Yi Ren, Zhiqiang Xu, Jing Jiang, Xiang Yin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html)

**Abstract**:

Unsupervised video domain adaptation is a practical yet challenging task. In this work, for the first time, we tackle it from a disentanglement view. Our key idea is to handle the spatial and temporal domain divergence separately through disentanglement. Specifically, we consider the generation of cross-domain videos from two sets of latent factors, one encoding the static information and another encoding the dynamic information. A Transfer Sequential VAE (TranSVAE) framework is then developed to model such generation. To better serve for adaptation, we propose several objectives to constrain the latent factors. With these constraints, the spatial divergence can be readily removed by disentangling the static domain-specific information out, and the temporal divergence is further reduced from both frame- and video-levels through adversarial learning. Extensive experiments on the UCF-HMDB, Jester, and Epic-Kitchens datasets verify the effectiveness and superiority of TranSVAE compared with several state-of-the-art approaches.

----

## [773] Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs Knowledge Editing in Language Models

**Authors**: *Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html)

**Abstract**:

Language models learn a great quantity of factual information during pretraining, and recent work localizes this information to specific model weights like mid-layer MLP weights. In this paper, we find that we can change how a fact is stored in a model by editing weights that are in a different location than where existing methods suggest that the fact is stored. This is surprising because we would expect that localizing facts to specific model parameters would tell us where to manipulate knowledge in models, and this assumption has motivated past work on model editing methods. Specifically, we show that localization conclusions from representation denoising (also known as Causal Tracing) do not provide any insight into which model MLP layer would be best to edit in order to override an existing stored fact with a new one. This finding raises questions about how past work relies on Causal Tracing to select which model layers to edit. Next, we consider several variants of the editing problem, including erasing and amplifying facts. For one of our editing problems, editing performance does relate to localization results from representation denoising, but we find that which layer we edit is a far better predictor of performance. Our results suggest, counterintuitively, that better mechanistic understanding of how pretrained language models work may not always translate to insights about how to best change their behavior.

----

## [774] The Geometry of Neural Nets' Parameter Spaces Under Reparametrization

**Authors**: *Agustinus Kristiadi, Felix Dangel, Philipp Hennig*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html)

**Abstract**:

Model reparametrization, which follows the change-of-variable rule of calculus, is a popular way to improve the training of neural nets. But it can also be problematic since it can induce inconsistencies in, e.g., Hessian-based flatness measures, optimization trajectories, and modes of probability densities. This complicates downstream analyses: e.g. one cannot definitively relate flatness with generalization since arbitrary reparametrization changes their relationship. In this work, we study the invariance of neural nets under reparametrization from the perspective of Riemannian geometry. From this point of view, invariance is an inherent property of any neural net if one explicitly represents the metric and uses the correct associated transformation rules. This is important since although the metric is always present, it is often implicitly assumed as identity, and thus dropped from the notation, then lost under reparametrization. We discuss implications for measuring the flatness of minima, optimization, and for probability-density maximization. Finally, we explore some interesting directions where invariance is useful.

----

## [775] A Dataset of Relighted 3D Interacting Hands

**Authors**: *Gyeongsik Moon, Shunsuke Saito, Weipeng Xu, Rohan Joshi, Julia Buffalini, Harley Bellan, Nicholas Rosen, Jesse Richardson, Mallorie Mize, Philippe de Bree, Tomas Simon, Bo Peng, Shubham Garg, Kevyn McPhail, Takaaki Shiratori*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/396beafa6feba781a7114780e6837253-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/396beafa6feba781a7114780e6837253-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The two-hand interaction is one of the most challenging signals to analyze due to the self-similarity, complicated articulations, and occlusions of hands. Although several datasets have been proposed for the two-hand interaction analysis, all of them do not achieve 1) diverse and realistic image appearances and 2) diverse and large-scale groundtruth (GT) 3D poses at the same time. In this work, we propose Re:InterHand, a dataset of relighted 3D interacting hands that achieve the two goals. To this end, we employ a state-of-the-art hand relighting network with our accurately tracked two-hand 3D poses. We compare our Re:InterHand with existing 3D interacting hands datasets and show the benefit of it. Our Re:InterHand is available in https://mks0601.github.io/ReInterHand/

----

## [776] Multi-Objective Intrinsic Reward Learning for Conversational Recommender Systems

**Authors**: *Zhendong Chu, Nan Wang, Hongning Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/396ea38391e8b96a3add6126006f1a53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/396ea38391e8b96a3add6126006f1a53-Abstract-Conference.html)

**Abstract**:

Conversational Recommender Systems (CRS) actively elicit user preferences to generate adaptive recommendations. Mainstream reinforcement learning-based CRS solutions heavily rely on handcrafted reward functions, which may not be aligned with user intent in CRS tasks. Therefore, the design of task-specific rewards is critical to facilitate CRS policy learning, which remains largely under-explored in the literature. In this work, we propose a novel approach to address this challenge by learning intrinsic rewards from interactions with users. Specifically, we formulate intrinsic reward learning as a multi-objective bi-level optimization problem. The inner level optimizes the CRS policy augmented by the learned intrinsic rewards, while the outer level drives the intrinsic rewards to optimize two CRS-specific objectives: maximizing the success rate and minimizing the number of turns to reach a successful recommendation}in conversations. To evaluate the effectiveness of our approach, we conduct extensive experiments on three public CRS benchmarks. The results show that our algorithm significantly improves CRS performance by exploiting informative learned intrinsic rewards.

----

## [777] Predict-then-Calibrate: A New Perspective of Robust Contextual LP

**Authors**: *Chunlin Sun, Linyu Liu, Xiaocheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/397271e11322fae8ba7f827c50ca8d9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/397271e11322fae8ba7f827c50ca8d9b-Abstract-Conference.html)

**Abstract**:

Contextual optimization, also known as predict-then-optimize or prescriptive analytics, considers an optimization problem with the presence of covariates (context or side information). The goal is to learn a prediction model (from the training data) that predicts the objective function from the covariates, and then in the test phase, solve the optimization problem with the covariates but without the observation of the objective function. In this paper, we consider a risk-sensitive version of the problem and propose a generic algorithm design paradigm called predict-then-calibrate. The idea is to first develop a prediction model without concern for the downstream risk profile or robustness guarantee, and then utilize calibration (or recalibration) methods to quantify the uncertainty of the prediction. While the existing methods suffer from either a restricted choice of the prediction model or strong assumptions on the underlying data, we show the disentangling of the prediction model and the calibration/uncertainty quantification has several advantages. First, it imposes no restriction on the prediction model and thus fully unleashes the potential of off-the-shelf machine learning methods. Second, the derivation of the risk and robustness guarantee can be made independent of the choice of the prediction model through a data-splitting idea. Third, our paradigm of predict-then-calibrate applies to both (risk-sensitive) robust and (risk-neutral) distributionally robust optimization (DRO) formulations. Theoretically, it gives new generalization bounds for the contextual LP problem and sheds light on the existing results of DRO for contextual LP. Numerical experiments further reinforce the advantage of the predict-then-calibrate paradigm in that an improvement on either the prediction model or the calibration model will lead to a better final performance.

----

## [778] INSPECT: A Multimodal Dataset for Patient Outcome Prediction of Pulmonary Embolisms

**Authors**: *Shih-Cheng Huang, Zepeng Huo, Ethan Steinberg, Chia-Chun Chiang, Curtis P. Langlotz, Matthew P. Lungren, Serena Yeung, Nigam Shah, Jason Alan Fries*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39736af1b9d87a1fddad9f84a6bcf64c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/39736af1b9d87a1fddad9f84a6bcf64c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Synthesizing information from various data sources plays a crucial role in the practice of modern medicine. Current applications of artificial intelligence in medicine often focus on single-modality data due to a lack of publicly available, multimodal medical datasets. To address this limitation, we introduce INSPECT, which contains de-identified longitudinal records from a large cohort of pulmonary embolism (PE) patients, along with ground truth labels for multiple outcomes. INSPECT contains data from 19,402 patients, including CT images, sections of radiology reports, and structured electronic health record (EHR) data (including demographics, diagnoses, procedures, and vitals). Using our provided dataset, we develop and release a benchmark for evaluating several baseline modeling approaches on a variety of important PE related tasks. We evaluate image-only, EHR-only, and fused models. Trained models and the de-identified dataset are made available for non-commercial use under a data use agreement. To the best our knowledge, INSPECT is the largest multimodal dataset for enabling reproducible research on strategies for integrating 3D medical imaging and EHR data.

----

## [779] What Makes Good Examples for Visual In-Context Learning?

**Authors**: *Yuanhan Zhang, Kaiyang Zhou, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/398ae57ed4fda79d0781c65c926d667b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/398ae57ed4fda79d0781c65c926d667b-Abstract-Conference.html)

**Abstract**:

Large vision models with billions of parameters and trained on broad data have great potential in numerous downstream applications. However, these models are typically difficult to adapt due to their large parameter size and sometimes lack of accesss to their weights---entities able to develop large vision models often provide APIs only. In this paper, we study how to better utilize large vision models through the lens of in-context learning, a concept that has been well-known in natural language processing but has only been studied very recently in computer vision. In-context learning refers to the ability to perform inference on tasks never seen during training by simply conditioning on in-context examples (i.e., input-output pairs) without updating any internal model parameters. To demystify in-context learning in computer vision, we conduct an extensive research and identify a critical problem: downstream performance is highly sensitivie to the choice of visual in-context examples. To address this problem, we propose a prompt retrieval framework specifically for large vision models, allowing the selection of in-context examples to be fully automated. Concretely, we provide two implementations: (i) an unsupervised prompt retrieval method based on nearest example search using an off-the-shelf model, and (ii) a supervised prompt retrieval method, which trains a neural network to choose examples that directly maximize in-context learning performance. Both methods do not require access to the internal weights of large vision models. Our results demonstrate that our methods can bring non-trivial improvements to visual in-context learning in comparison to the commonly-used random selection. Code and models will be released.

----

## [780] Parameterizing Context: Unleashing the Power of Parameter-Efficient Fine-Tuning and In-Context Tuning for Continual Table Semantic Parsing

**Authors**: *Yongrui Chen, Shenyu Zhang, Guilin Qi, Xinnan Guo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/398b00a05b847ac65eb98c8e5e865fe8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/398b00a05b847ac65eb98c8e5e865fe8-Abstract-Conference.html)

**Abstract**:

Continual table semantic parsing aims to train a parser on a sequence of tasks, where each task requires the parser to translate natural language into SQL based on task-specific tables but only offers limited training examples. Conventional methods tend to suffer from overfitting with limited supervision, as well as catastrophic forgetting due to parameter updates.Despite recent advancements that partially alleviate these issues through semi-supervised data augmentation and retention of a few past examples, the performance is still limited by the volume of unsupervised data and stored examples.To overcome these challenges, this paper introduces a novel method integrating parameter-efficient fine-tuning (PEFT) and in-context tuning (ICT) for training a continual table semantic parser. Initially, we present a task-adaptive PEFT framework capable of fully circumventing catastrophic forgetting, which is achieved by freezing the pre-trained model backbone and fine-tuning small-scale prompts. Building on this, we propose a teacher-student framework-based solution. The teacher addresses the few-shot problem using ICT, which procures contextual information by demonstrating a few training examples. In turn, the student leverages the proposed PEFT framework to learn from the teacher's output distribution, and subsequently compresses and saves the contextual information to the prompts, eliminating the need to store any training examples.Experimental evaluations on two benchmarks affirm the superiority of our method over prevalent few-shot and continual learning baselines across various metrics.

----

## [781] Incentives in Federated Learning: Equilibria, Dynamics, and Mechanisms for Welfare Maximization

**Authors**: *Aniket Murhekar, Zhuowen Yuan, Bhaskar Ray Chaudhury, Bo Li, Ruta Mehta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39b77b5e422b4e070e2811b73ea9bcf7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39b77b5e422b4e070e2811b73ea9bcf7-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) has emerged as a powerful scheme to facilitate the collaborative learning of models amongst a set of agents holding their own private data.  Although the agents benefit from the global model trained on shared data, by participating in federated learning, they may also incur costs (related to privacy and communication) due to data sharing. In this paper, we model a collaborative FL framework, where every agent attempts to achieve an optimal trade-off between her learning payoff and data sharing cost. We show the existence of Nash equilibrium (NE) under mild assumptions on agents' payoff and costs. Furthermore, we show that agents can discover the NE via best response dynamics. However, some of the NE may be bad in terms of overall welfare for the agents, implying little incentive for some fraction of the agents to participate in the learning. To remedy this, we design a budget-balanced mechanism involving payments to the agents, that ensures that any $p$-mean welfare function of the agents' utilities is maximized at NE. In addition, we introduce a FL protocol FedBR-BG that incorporates our budget-balanced mechanism, utilizing best response dynamics. Our empirical validation on MNIST and CIFAR-10 substantiates our theoretical analysis. We show that FedBR-BG outperforms the basic best-response-based protocol without additional incentivization, the standard federated learning protocol FedAvg, as well as a recent baseline MWFed in terms of achieving superior $p$-mean welfare.

----

## [782] MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates

**Authors**: *Mohammad Mozaffari, Sikan Li, Zhao Zhang, Maryam Mehri Dehnavi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39bc6e3cbf5a1991d33dc10ebff9a9cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39bc6e3cbf5a1991d33dc10ebff9a9cf-Abstract-Conference.html)

**Abstract**:

This work proposes a Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 updates, called MKOR, that improves the training time and convergence properties of deep neural networks (DNNs). Second-order techniques, while enjoying higher convergence rates vs first-order counterparts, have cubic complexity with respect to either the model size and/or the training batch size. Hence they exhibit poor scalability and performance in transformer models, e.g. large language models (LLMs), because the batch sizes in these models  scale by the attention mechanism sequence length, leading to large model size and batch sizes. MKOR's complexity is quadratic with respect to the model size, alleviating the computation bottlenecks in  second-order methods. Because of their high computation complexity, state-of-the-art implementations of second-order methods can only afford to update the second order information infrequently, and thus do not fully exploit the promise of better convergence from these updates. By reducing the communication complexity of the second-order updates as well as achieving a linear communication complexity, MKOR increases the frequency of second order updates. We also propose a hybrid version of MKOR (called MKOR-H) that mid-training falls backs to a first order optimizer if the second order updates  no longer accelerate convergence.  Our experiments show that MKOR outperforms state -of-the-art first order methods, e.g. the LAMB optimizer, and best implementations of second-order methods, i.e. KAISA/KFAC, up to 2.57x and 1.85x respectively on BERT-Large-Uncased on 64 GPUs.

----

## [783] DFRD: Data-Free Robustness Distillation for Heterogeneous Federated Learning

**Authors**: *Kangyang Luo, Shuai Wang, Yexuan Fu, Xiang Li, Yunshi Lan, Ming Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39ca8893ea38905a9d2ffe786e85af0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39ca8893ea38905a9d2ffe786e85af0f-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) is a privacy-constrained decentralized machine learning paradigm in which clients enable collaborative training without compromising private data. However, how to learn a robust global model in the data-heterogeneous and model-heterogeneous FL scenarios is challenging. To address it, we resort to data-free knowledge distillation to propose a new FL method (namely DFRD).DFRD equips a conditional generator on the server to approximate the training space of the local models uploaded by clients, and systematically investigates its training in terms of fidelity, transferability and diversity. To overcome the catastrophic forgetting of the global model caused by the distribution shifts of the generator across communication rounds, we maintain an exponential moving average copy of the generator on the server.  Additionally, we propose dynamic weighting and label sampling to accurately extract knowledge from local models. Finally, our extensive experiments on various image classification tasks illustrate that DFRD achieves significant performance gains compared to SOTA baselines.

----

## [784] SG×P : A Sorghum Genotype × Phenotype Prediction Dataset and Benchmark

**Authors**: *Zeyu Zhang, Robert Pless, Nadia Shakoor, Austin Carnahan, Abby Stylianou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39d02e8e23bafadd7cd405f2281bc05c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/39d02e8e23bafadd7cd405f2281bc05c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large scale field-phenotyping approaches have the potential to solve important questions about the relationship of plant genotype to plant phenotype.  Computational approaches to measuring the phenotype (the observable plant features) are required to address the problem at a large scale, but machine learning approaches to extract phenotypes from sensor data have been hampered by limited access to (a) sufficiently large, organized multi-sensor datasets, (b) field trials that have a large scale and significant number of genotypes, (c) full genetic sequencing of those phenotypes, and (d) datasets sufficiently organized so that algorithm centered researchers can directly address the real biological problems.  To address this, we present SGxP, a novel benchmark dataset from a large-scale field trial consisting of the complete genotype of over 300 sorghum varieties, and time sequences of imagery from several field plots growing each variety, taken with RGB and laser 3D scanner imaging.  To lower the barrier to entry and facilitate further developments, we provide a set of well organized, multi-sensor imagery and corresponding genomic data.  We implement baseline deep learning based phenotyping approaches to create baseline results for individual sensors and multi-sensor fusion for detecting genetic mutations with known impacts.  We also provide and support an open-ended challenge by identifying thousands of genetic mutations whose phenotypic impacts are currently unknown.  A web interface for machine learning researchers and practitioners to share approaches, visualizations and hypotheses supports engagement with plant biologists to further the understanding of the sorghum genotype x phenotype relationship. The full dataset, leaderboard (including baseline results) and discussion forums can be found at http://sorghumsnpbenchmark.com.

----

## [785] Rank-N-Contrast: Learning Continuous Representations for Regression

**Authors**: *Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, Dina Katabi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)

**Abstract**:

Deep regression models typically learn in an end-to-end fashion without explicitly emphasizing a regression-aware representation. Consequently, the learned representations exhibit fragmentation and fail to capture the continuous nature of sample orders, inducing suboptimal results across a wide range of regression tasks. To fill the gap, we propose Rank-N-Contrast (RNC), a framework that learns continuous representations for regression by contrasting samples against each other based on their rankings in the target space. We demonstrate, theoretically and empirically, that RNC guarantees the desired order of learned representations in accordance with the target orders, enjoying not only better performance but also significantly improved robustness, efficiency, and generalization. Extensive experiments using five real-world regression datasets that span computer vision, human-computer interaction, and healthcare verify that RNC achieves state-of-the-art performance, highlighting its intriguing properties including better data efficiency, robustness to spurious targets and data corruptions, and generalization to distribution shifts.

----

## [786] OpenGSL: A Comprehensive Benchmark for Graph Structure Learning

**Authors**: *Zhiyao Zhou, Sheng Zhou, Bochao Mao, Xuanyi Zhou, Jiawei Chen, Qiaoyu Tan, Daochen Zha, Yan Feng, Chun Chen, Can Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39f8ef62e061042cca8c8f46d7e0e31b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/39f8ef62e061042cca8c8f46d7e0e31b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Graph Neural Networks (GNNs) have emerged as the de facto standard for representation learning on graphs, owing to their ability to effectively integrate graph topology and node attributes. However, the inherent suboptimal nature of node connections, resulting from the complex and contingent formation process of graphs, presents significant challenges in modeling them effectively. To tackle this issue, Graph Structure Learning (GSL), a family of data-centric learning approaches, has garnered substantial attention in recent years. The core concept behind GSL is to jointly optimize the graph structure and the corresponding GNN models. Despite the proposal of numerous GSL methods, the progress in this field remains unclear due to inconsistent experimental protocols, including variations in datasets, data processing techniques, and splitting strategies. In this paper, we introduce OpenGSL, the first comprehensive benchmark for GSL, aimed at addressing this gap. OpenGSL enables a fair comparison among state-of-the-art GSL methods by evaluating them across various popular datasets using uniform data processing and splitting strategies. Through extensive experiments, we observe that existing GSL methods do not consistently outperform vanilla GNN counterparts. We also find that there is no significant correlation between the homophily of the learned structure and task performance, challenging the common belief. Moreover, we observe that the learned graph structure demonstrates a strong generalization ability across different GNN models, despite the high computational and space consumption. We hope that our open-sourced library will facilitate rapid and equitable evaluation and inspire further innovative research in this field. The code of the benchmark can be found in https://github.com/OpenGSL/OpenGSL.

----

## [787] Global Optimality in Bivariate Gradient-based DAG Learning

**Authors**: *Chang Deng, Kevin Bello, Pradeep Ravikumar, Bryon Aragam*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a02b6df276223b68c69ca572cb3c4a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a02b6df276223b68c69ca572cb3c4a8-Abstract-Conference.html)

**Abstract**:

Recently, a new class of non-convex optimization problems motivated by the statistical problem of learning an acyclic directed graphical model from data has attracted significant interest. While existing work uses standard first-order optimization schemes to solve this problem, proving the global optimality of such approaches has proven elusive. The difficulty lies in the fact that unlike other non-convex problems in the literature, this problem is not "benign", and possesses multiple spurious solutions that standard approaches can easily get trapped in. In this paper, we prove that a simple path-following optimization scheme globally converges to the global minimum of the population loss in the bivariate setting.

----

## [788] Perceptual adjustment queries and an inverted measurement paradigm for low-rank metric learning

**Authors**: *Austin Xu, Andrew D. McRae, Jingyan Wang, Mark A. Davenport, Ashwin Pananjady*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a07c3a67cfe50d3236b71fb674c7f30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a07c3a67cfe50d3236b71fb674c7f30-Abstract-Conference.html)

**Abstract**:

We introduce a new type of query mechanism for collecting human feedback, called the perceptual adjustment query (PAQ). Being both informative and cognitively lightweight, the PAQ adopts an inverted measurement scheme, and combines advantages from both cardinal and ordinal queries. We showcase the PAQ in the metric learning problem, where we collect PAQ measurements to learn an unknown Mahalanobis distance. This gives rise to a high-dimensional, low-rank matrix estimation problem to which standard matrix estimators cannot be applied. Consequently, we develop a two-stage estimator for metric learning from PAQs, and provide sample complexity guarantees for this estimator. We present numerical simulations demonstrating the performance of the estimator and its notable properties.

----

## [789] Joint processing of linguistic properties in brains and language models

**Authors**: *Subba Reddy Oota, Manish Gupta, Mariya Toneva*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a0e2de215bd17c39ad08ba1d16c1b12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a0e2de215bd17c39ad08ba1d16c1b12-Abstract-Conference.html)

**Abstract**:

Language models have been shown to be very effective in predicting brain recordings of subjects experiencing complex language stimuli. For a deeper understanding of this alignment, it is important to understand the correspondence between the detailed processing of linguistic information by the human brain versus language models. We investigate this correspondence via a direct approach, in which we eliminate information related to specific linguistic properties in the language model representations and observe how this intervention affects the alignment with fMRI brain recordings obtained while participants listened to a story. We investigate a range of linguistic properties (surface, syntactic, and semantic) and find that the elimination of each one results in a significant decrease in brain alignment. Specifically, we find that syntactic properties (i.e. Top Constituents and Tree Depth) have the largest effect on the trend of brain alignment across model layers. These findings provide clear evidence for the role of specific linguistic information in the alignment between brain and language models, and open new avenues for mapping the joint information processing in both systems. We make the code publicly available https://github.com/subbareddy248/lingprop-brain-alignment.

----

## [790] S3: Increasing GPU Utilization during Generative Inference for Higher Throughput

**Authors**: *Yunho Jin, Chun-Feng Wu, David Brooks, Gu-Yeon Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a13be0c5dae69e0f08065f113fb10b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a13be0c5dae69e0f08065f113fb10b8-Abstract-Conference.html)

**Abstract**:

Generating texts with a large language model (LLM) consumes massive amounts of memory. Apart from the already-large model parameters, the key/value (KV) cache that holds information about previous tokens in a sequence can grow to be even larger than the model itself. This problem is exacerbated in one of the current LLM serving frameworks which reserves the maximum sequence length of memory for the KV cache to guarantee generating a complete sequence as they do not know the output sequence length. This restricts us to use a smaller batch size leading to lower GPU utilization and above all, lower throughput. We argue that designing a system with a priori knowledge of the output sequence can mitigate this problem. To this end, we propose $S^3$, which predicts the output sequence length, schedules generation queries based on the prediction to increase device resource utilization and throughput, and handle mispredictions. Our proposed method achieves 6.49Ã— throughput over those systems that assume the worst case for the output sequence length.

----

## [791] Disentangling Cognitive Diagnosis with Limited Exercise Labels

**Authors**: *Xiangzhi Chen, Le Wu, Fei Liu, Lei Chen, Kun Zhang, Richang Hong, Meng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a14ae9951e8153a8fc814b5f506b5b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a14ae9951e8153a8fc814b5f506b5b7-Abstract-Conference.html)

**Abstract**:

Cognitive diagnosis is an important task in intelligence education, which aims at measuring studentsâ€™ proficiency in specific knowledge concepts. Given a fully labeled exercise-concept matrix, most existing models focused on mining students' response records for cognitive diagnosis. Despite their success, due to the huge cost of labeling exercises, a more practical scenario is that limited exercises are labeled with concepts. Performing cognitive diagnosis with limited exercise labels is under-explored and remains pretty much open. In this paper, we propose Disentanglement based Cognitive Diagnosis (DCD) to address the challenges of limited exercise labels. Specifically, we utilize students' response records to model student proficiency, exercise difficulty and exercise label distribution.   Then, we introduce two novel modules - group-based disentanglement and limited-labeled alignment modules - to disentangle the factors relevant to concepts and align them with real limited labels.  Particularly, we introduce the tree-like structure of concepts with negligible cost for group-based disentangling, as concepts of different levels exhibit different independence relationships.Extensive experiments on widely used benchmarks demonstrate the superiority of our proposed model.

----

## [792] Energy-Based Sliced Wasserstein Distance

**Authors**: *Khai Nguyen, Nhat Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a23caeb904c822575fa56fb114ca499-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a23caeb904c822575fa56fb114ca499-Abstract-Conference.html)

**Abstract**:

The sliced Wasserstein (SW) distance has been widely recognized as a statistically effective and computationally efficient metric between two probability measures. A key component of the SW distance is the slicing distribution. There are two existing approaches for choosing this distribution. The first approach is using a fixed prior distribution. The second approach is optimizing for the best distribution which belongs to a parametric family of distributions and can maximize the expected distance. However, both approaches have their limitations. A fixed prior distribution is non-informative in terms of highlighting projecting directions that can discriminate two general probability measures. Doing optimization for the best distribution is often expensive and unstable. Moreover, designing the parametric family of the candidate distribution could be easily misspecified. To address the issues, we propose to design the slicing distribution as an energy-based distribution that is parameter-free and has the density proportional to an energy function of the projected one-dimensional Wasserstein distance. We then derive a novel sliced Wasserstein variant, energy-based sliced Waserstein (EBSW) distance, and investigate its topological, statistical, and computational properties via importance sampling, sampling importance resampling, and Markov Chain methods. Finally, we conduct experiments on point-cloud gradient flow, color transfer, and point-cloud reconstruction to show the favorable performance of the EBSW.

----

## [793] E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning

**Authors**: *Xiuhong Lin, Changjie Qiu, Zhipeng Cai, Siqi Shen, Yu Zang, Weiquan Liu, Xuesheng Bian, Matthias Müller, Cheng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a2d1bf9bc0a9794cf82c1341a7a75e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a2d1bf9bc0a9794cf82c1341a7a75e6-Abstract-Conference.html)

**Abstract**:

Event cameras have emerged as a promising vision sensor in recent years due to their unparalleled temporal resolution and dynamic range. While registration of 2D RGB images to 3D point clouds is a long-standing problem in computer vision, no prior work studies 2D-3D registration for event cameras. To this end, we propose E2PNet, the first learning-based method for event-to-point cloud registration.The core of E2PNet is a novel feature representation network called Event-Points-to-Tensor (EP2T), which encodes event data into a 2D grid-shaped feature tensor. This grid-shaped feature enables matured RGB-based frameworks to be easily used for event-to-point cloud registration, without changing hyper-parameters and the training procedure. EP2T treats the event input as spatio-temporal point clouds. Unlike standard 3D learning architectures that treat all dimensions of point clouds equally, the novel sampling and information aggregation modules in EP2T are designed to handle the inhomogeneity of the spatial and temporal dimensions. Experiments on the MVSEC and VECtor datasets demonstrate the superiority of E2PNet over hand-crafted and other learning-based methods. Compared to RGB-based registration, E2PNet is more robust to extreme illumination or fast motion due to the use of event data. Beyond 2D-3D registration, we also show the potential of EP2T for other vision tasks such as flow estimation, event-to-image reconstruction and object recognition. The source code can be found at: https://github.com/Xmu-qcj/E2PNet.

----

## [794] Pengi: An Audio Language Model for Audio Tasks

**Authors**: *Soham Deshmukh, Benjamin Elizalde, Rita Singh, Huaming Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a2e5889b4bbef997ddb13b55d5acf77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a2e5889b4bbef997ddb13b55d5acf77-Abstract-Conference.html)

**Abstract**:

In the domain of audio processing, Transfer Learning has facilitated the rise of Self-Supervised Learning and Zero-Shot Learning techniques. These approaches have led to the development of versatile models capable of tackling a wide array of tasks, while delivering state-of-the-art performance. However, current models inherently lack the capacity to produce the requisite language for open-ended tasks, such as Audio Captioning or Audio Question Answering. We introduce Pengi, a novel Audio Language Model that leverages Transfer Learning by framing all audio tasks as text-generation tasks. It takes as input, an audio recording, and text, and generates free-form text as output. The input audio is represented as a sequence of continuous embeddings by an audio encoder. A text encoder does the same for the corresponding text input. Both sequences are combined as a prefix to prompt a pre-trained frozen language model. The unified architecture of Pengi enables open-ended tasks and close-ended tasks without any additional fine-tuning or task-specific extensions. When evaluated on 21 downstream tasks, our approach yields state-of-the-art performance in several of them. Our results show that connecting language models with audio models is a major step towards general-purpose audio understanding.

----

## [795] Unleashing the Power of Graph Data Augmentation on Covariate Distribution Shift

**Authors**: *Yongduo Sui, Qitian Wu, Jiancan Wu, Qing Cui, Longfei Li, Jun Zhou, Xiang Wang, Xiangnan He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a33ddacb2798fc7d83b8334d552e05a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a33ddacb2798fc7d83b8334d552e05a-Abstract-Conference.html)

**Abstract**:

The issue of distribution shifts is emerging as a critical concern in graph representation learning. From the perspective of invariant learning and stable learning, a recently well-established paradigm for out-of-distribution generalization, stable features of the graph are assumed to causally determine labels, while environmental features tend to be unstable and can lead to the two primary types of distribution shifts. The correlation shift is often caused by the spurious correlation between environmental features and labels that differs between the training and test data; the covariate shift often stems from the presence of new environmental features in test data. However, most strategies, such as invariant learning or graph augmentation, typically struggle with limited training environments or perturbed stable features, thus exposing limitations in handling the problem of covariate shift. To address this challenge, we propose a simple-yet-effective data augmentation strategy, Adversarial Invariant Augmentation (AIA), to handle the covariate shift on graphs. Specifically, given the training data, AIA aims to extrapolate and generate new environments, while concurrently preserving the original stable features during the augmentation process. Such a design equips the graph classification model with an enhanced capability to identify stable features in new environments, thereby effectively tackling the covariate shift in data. Extensive experiments with in-depth empirical analysis demonstrate the superiority of our approach. The implementation codes are publicly available at https://github.com/yongduosui/AIA.

----

## [796] Adaptive recurrent vision performs zero-shot computation scaling to unseen difficulty levels

**Authors**: *Vijay Veerabadran, Srinivas Ravishankar, Yuan Tang, Ritik Raina, Virginia de Sa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a40e042c66e84659249f3254460c123-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a40e042c66e84659249f3254460c123-Abstract-Conference.html)

**Abstract**:

Humans solving algorithmic (or) reasoning problems typically exhibit solution times that grow as a function of problem difficulty. Adaptive recurrent neural networks have been shown to exhibit this property for various language-processing tasks. However, little work has been performed to assess whether such adaptive computation can also enable vision models to extrapolate solutions beyond their training distribution's difficulty level, with prior work focusing on very simple tasks. In this study, we investigate a critical functional role of such adaptive processing using recurrent neural networks: to dynamically scale computational resources conditional on input requirements that allow for zero-shot generalization to novel difficulty levels not seen during training using two challenging visual reasoning tasks: PathFinder and Mazes. We combine convolutional recurrent neural networks (ConvRNNs) with a learnable halting mechanism based on Graves (2016). We explore various implementations of such adaptive ConvRNNs (AdRNNs) ranging from tying weights across layers to more sophisticated biologically inspired recurrent networks that possess lateral connections and gating. We show that 1) AdRNNs learn to dynamically halt processing early (or late) to solve easier (or harder) problems, 2) these RNNs zero-shot generalize to more difficult problem settings not shown during training by dynamically increasing the number of recurrent iterations at test time. Our study provides modeling evidence supporting the hypothesis that recurrent processing enables the functional advantage of adaptively allocating compute resources conditional on input requirements and hence allowing generalization to harder difficulty levels of a visual reasoning problem without training.

----

## [797] NurViD: A Large Expert-Level Video Database for Nursing Procedure Activity Understanding

**Authors**: *Ming Hu, Lin Wang, Siyuan Yan, Don Ma, Qingli Ren, Peng Xia, Wei Feng, Peibo Duan, Lie Ju, Zongyuan Ge*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a48b0eaba26ba862220a307a9edb0bb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a48b0eaba26ba862220a307a9edb0bb-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The application of deep learning to nursing procedure activity understanding has the potential to greatly enhance the quality and safety of nurse-patient interactions. By utilizing the technique, we can facilitate training and education, improve quality control, and enable operational compliance monitoring. However, the development of automatic recognition systems in this field is currently hindered by the scarcity of appropriately labeled datasets. The existing video datasets pose several limitations: 1) these datasets are small-scale in size to support comprehensive investigations of nursing activity; 2) they primarily focus on single procedures, lacking expert-level annotations for various nursing procedures and action steps; and 3) they lack temporally localized annotations, which prevents the effective localization of targeted actions within longer video sequences. To mitigate these limitations, we propose NurViD, a large video dataset with expert-level annotation for nursing procedure activity understanding. NurViD consists of over 1.5k videos totaling 144 hours, making it approximately four times longer than the existing largest nursing activity datasets. Notably, it encompasses 51 distinct nursing procedures and 177 action steps, providing a much more comprehensive coverage compared to existing datasets that primarily focus on limited procedures. To evaluate the efficacy of current deep learning methods on nursing activity understanding, we establish three benchmarks on NurViD: procedure recognition on untrimmed videos, procedure and action recognition on trimmed videos, and action detection. Our benchmark and code will be available at https://github.com/minghu0830/NurViD-benchmark.

----

## [798] The Pick-to-Learn Algorithm: Empowering Compression for Tight Generalization Bounds and Improved Post-training Performance

**Authors**: *Dario Paccagnan, Marco C. Campi, Simone Garatti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a4f287883609241031e6818bd01133e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a4f287883609241031e6818bd01133e-Abstract-Conference.html)

**Abstract**:

Generalization bounds are valuable both for theory and applications. On the one hand, they shed light on the mechanisms that underpin the learning processes; on the other, they certify how well a learned model performs against unseen inputs.  In this work we build upon a recent breakthrough in compression theory to develop a new framework yielding tight generalization bounds of wide practical applicability.  The core idea is to embed any given learning algorithm into a suitably-constructed meta-algorithm (here called Pick-to-Learn, P2L) in order to instill desirable compression properties. When applied to the MNIST classification dataset and to a synthetic regression problem, P2L not only attains generalization bounds that compare favorably with the state of the art (test-set and PAC-Bayes bounds), but it also learns models with better post-training performance.

----

## [799] Orthogonal Non-negative Tensor Factorization based Multi-view Clustering

**Authors**: *Jing Li, Quanxue Gao, Qianqian Wang, Ming Yang, Wei Xia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a5b75ce6cbd3aaaa32d6e935ffc4cff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a5b75ce6cbd3aaaa32d6e935ffc4cff-Abstract-Conference.html)

**Abstract**:

Multi-view clustering (MVC) based on non-negative matrix factorization (NMF) and its variants have attracted much attention due to their advantages in clustering interpretability. However, existing NMF-based multi-view clustering methods perform NMF on each view respectively and ignore the impact of between-view. Thus, they can't well exploit the within-view spatial structure and between-view complementary information. To resolve this issue, we present orthogonal non-negative tensor factorization (Orth-NTF) and develop a novel multi-view clustering based on Orth-NTF with one-side orthogonal constraint. Our model directly performs Orth-NTF on the 3rd-order tensor which is composed of anchor graphs of views. Thus, our model directly considers the between-view relationship. Moreover, we use the tensor Schatten $p$-norm regularization as a rank approximation of the 3rd-order tensor which characterizes the cluster structure of multi-view data and exploits the between-view complementary information. In addition, we provide an optimization algorithm for the proposed method and prove mathematically that the algorithm always converges to the stationary KKT point. Extensive experiments on various benchmark datasets indicate that our proposed method is able to achieve satisfactory clustering performance.

----



[Go to the previous page](NIPS-2023-list03.md)

[Go to the next page](NIPS-2023-list05.md)

[Go to the catalog section](README.md)