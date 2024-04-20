### Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder.

**Authors**: Michael Bereket, Theofanis Karaletsos

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0001ca33ba34ce0351e4612b744b3936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0001ca33ba34ce0351e4612b744b3936-Abstract-Conference.html)

**Abstract**:

Generative models of observations under interventions have been a vibrant topic of interest across machine learning and the sciences in recent years. For example, in drug discovery, there is a need to model the effects of diverse interventions on cells in order to characterize unknown biological mechanisms of action. We propose the Sparse Additive Mechanism Shift Variational Autoencoder, SAMS-VAE, to combine compositionality, disentanglement, and interpretability for perturbation models. SAMS-VAE models the latent state of a perturbed sample as the sum of a local latent variable capturing sample-specific variation and sparse global variables of latent intervention effects. Crucially, SAMS-VAE sparsifies these global latent variables for individual perturbations to identify disentangled, perturbation-specific latent subspaces that are flexibly composable. We evaluate SAMS-VAE both quantitatively and qualitatively on a range of tasks using two popular single cell sequencing datasets.In order to measure perturbation-specific model-properties, we also introduce a framework for evaluation of perturbation models based on average treatment effects with links to posterior predictive checks. SAMS-VAE outperforms comparable models in terms of generalization across in-distribution and out-of-distribution tasks, including a combinatorial reasoning task under resource paucity, and yields interpretable latent structures which correlate strongly to known biological mechanisms. Our results suggest SAMS-VAE is an interesting addition to the modeling toolkit for machine learning-driven scientific discovery.

----

### Cross-Episodic Curriculum for Transformer Agents.

**Authors**: Lucy Xiaoyang Shi, Yunfan Jiang, Jake Grigsby, Linxi Fan, Yuke Zhu

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/001608167bb652337af5df0129aeaabd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/001608167bb652337af5df0129aeaabd-Abstract-Conference.html)

**Abstract**:

We present a new algorithm, Cross-Episodic Curriculum (CEC), to boost the learning efficiency and generalization of Transformer agents. Central to CEC is the placement of cross-episodic experiences into a Transformerâ€™s context, which forms the basis of a curriculum. By sequentially structuring online learning trials and mixed-quality demonstrations, CEC constructs curricula that encapsulate learning progression and proficiency increase across episodes. Such synergy combined with the potent pattern recognition capabilities of Transformer models delivers a powerful cross-episodic attention mechanism. The effectiveness of CEC is demonstrated under two representative scenarios: one involving multi-task reinforcement learning with discrete control, such as in DeepMind Lab, where the curriculum captures the learning progression in both individual and progressively complex settings; and the other involving imitation learning with mixed-quality data for continuous control, as seen in RoboMimic, where the curriculum captures the improvement in demonstrators' expertise. In all instances, policies resulting from CEC exhibit superior performance and strong generalization. Code is open-sourced on the project website https://cec-agent.github.io/ to facilitate research on Transformer agent learning.

----

### PaintSeg: Painting Pixels for Training-free Segmentation.

**Authors**: Xiang Li, Chung-Ching Lin, Yinpeng Chen, Zicheng Liu, Jinglu Wang, Rita Singh, Bhiksha Raj

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0021c2cb1b9b6a71ac478ea52a93b25a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0021c2cb1b9b6a71ac478ea52a93b25a-Abstract-Conference.html)

**Abstract**:

The paper introduces PaintSeg, a new unsupervised method for segmenting objects without any training. We propose an adversarial masked contrastive painting (AMCP) process, which creates a contrast between the original image and a painted image in which a masked area is painted using off-the-shelf generative models. During the painting process, inpainting and outpainting are alternated, with the former masking the foreground and filling in the background, and the latter masking the background while recovering the missing part of the foreground object. Inpainting and outpainting, also referred to as I-step and O-step, allow our method to gradually advance the target segmentation mask toward the ground truth without supervision or training. PaintSeg can be configured to work with a variety of prompts, e.g. coarse masks, boxes, scribbles, and points. Our experimental results demonstrate that PaintSeg outperforms existing approaches in coarse mask-prompt, box-prompt, and point-prompt segmentation tasks, providing a training-free solution suitable for unsupervised segmentation. Code: https://github.com/lxa9867/PaintSeg.

----

### Bootstrapping Vision-Language Learning with Decoupled Language Pre-training.

**Authors**: Yiren Jian, Chongyang Gao, Soroush Vosoughi

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/002262941c9edfd472a79298b2ac5e17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/002262941c9edfd472a79298b2ac5e17-Abstract-Conference.html)

**Abstract**:

We present a novel methodology aimed at optimizing the application of frozen large language models (LLMs) for resource-intensive vision-language (VL) pre-training. The current paradigm uses visual features as prompts to guide language models, with a focus on determining the most relevant visual features for corresponding text. Our approach diverges by concentrating on the language component, specifically identifying the optimal prompts to align with visual features. We introduce the Prompt-Transformer (P-Former), a model that predicts these ideal prompts, which is trained exclusively on linguistic data, bypassing the need for image-text pairings. This strategy subtly bifurcates the end-to-end VL training process into an additional, separate stage. Our experiments reveal that our framework significantly enhances the performance of a robust image-to-text baseline (BLIP-2), and effectively narrows the performance gap between models trained with either 4M or 129M image-text pairs. Importantly, our framework is modality-agnostic and flexible in terms of architectural design, as validated by its successful application in a video learning task using varied base modules. The code will be made available at https://github.com/yiren-jian/BLIText.

----

### Path following algorithms for 𝓁

**Authors**: Yunzhang Zhu, Renxiong Liu

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00296c0e10cd24d415c2db63ea2a2c68-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00296c0e10cd24d415c2db63ea2a2c68-Abstract-Conference.html)

**Abstract**:

Many modern machine learning algorithms are formulated as regularized M-estimation problems, in which a regularization (tuning) parameter controls a trade-off between model fit to the training data and model complexity. To select the ``best'' tuning parameter value that achieves a good trade-off, an approximated solution path needs to be computed. In practice, this is often done through selecting a grid of tuning parameter values and solving the regularized problem at the selected grid points. However, given any desired level of accuracy, it is often not clear how to choose the grid points and also how accurately one should solve the regularized problems at the selected gird points, both of which can greatly impact the overall amount of computation. In the context of  $\ell_2$-regularized $M$-estimation problem, we propose a novel grid point selection scheme and an adaptive stopping criterion for any given optimization algorithm that produces an approximated solution path with approximation error guarantee. Theoretically, we prove that the proposed solution path can approximate the exact solution path to arbitrary level of accuracy, while saving the overall computation as much as possible. Numerical results also corroborate with our theoretical analysis.

----

### PDF: Point Diffusion Implicit Function for Large-scale Scene Neural Representation.

**Authors**: Yuhan Ding, Fukun Yin, Jiayuan Fan, Hui Li, Xin Chen, Wen Liu, Chongshan Lu, Gang Yu, Tao Chen

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0073cc73e1873b35345209b50a3dab66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0073cc73e1873b35345209b50a3dab66-Abstract-Conference.html)

**Abstract**:

Recent advances in implicit neural representations have achieved impressive results by sampling and fusing individual points along sampling rays in the sampling space. However, due to the explosively growing sampling space, finely representing and synthesizing detailed textures remains a challenge for unbounded large-scale outdoor scenes. To alleviate the dilemma of using individual points to perceive the entire colossal space, we explore learning the surface distribution of the scene to provide structural priors and reduce the samplable space and propose a Point Diffusion implicit Function, PDF, for large-scale scene neural representation. The core of our method is a large-scale point cloud super-resolution diffusion module that enhances the sparse point cloud reconstructed from several training images into a dense point cloud as an explicit prior. Then in the rendering stage, only sampling points with prior points within the sampling radius are retained. That is, the sampling space is reduced from the unbounded space to the scene surface. Meanwhile, to fill in the background of the scene that cannot be provided by point clouds, the region sampling based on Mip-NeRF 360 is employed to model the background representation. Expensive experiments have demonstrated the effectiveness of our method for large-scale scene novel view synthesis, which outperforms relevant state-of-the-art baselines.

----

### Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation.

**Authors**: Ruida Zhou, Tao Liu, Min Cheng, Dileep Kalathil, P. R. Kumar, Chao Tian

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/007f4927e60699392425f267d43f0940-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/007f4927e60699392425f267d43f0940-Abstract-Conference.html)

**Abstract**:

We study robust reinforcement learning (RL) with the goal of determining a well-performing policy that is robust against model mismatch between the training simulator and the testing environment. Previous policy-based robust RL algorithms mainly focus on the tabular setting under uncertainty sets that facilitate robust policy evaluation, but are no longer tractable when the number of states scales up. To this end, we propose two novel uncertainty set formulations, one based on double sampling and the other on an integral probability metric. Both make large-scale robust RL tractable even when one only has access to a simulator. We propose a robust natural actor-critic (RNAC) approach that incorporates the new uncertainty sets and employs function approximation. We provide finite-time convergence guarantees for the proposed RNAC algorithm to the optimal robust policy within the function approximation error. Finally, we demonstrate the robust performance of the policy learned by our proposed RNAC approach in multiple  MuJoCo environments and a real-world TurtleBot navigation task.

----

### Adaptive Selective Sampling for Online Prediction with Experts.

**Authors**: Rui M. Castro, Fredrik Hellström, Tim van Erven

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html)

**Abstract**:

We consider online prediction of a binary sequence with expert advice. For this setting, we devise label-efficient forecasting algorithms, which use a selective sampling scheme that enables collecting much fewer labels than standard procedures. For the general case without a perfect expert, we prove best-of-both-worlds guarantees, demonstrating that the proposed forecasting algorithm always queries sufficiently many labels in the worst case to obtain optimal regret guarantees, while simultaneously querying much fewer labels in more benign settings. Specifically, for a scenario where one expert is strictly better than the others in expectation, we show that the label complexity of the label-efficient forecaster is roughly upper-bounded by the square root of the number of rounds. Finally, we present numerical experiments empirically showing that the normalized regret of the label-efficient forecaster can asymptotically match known minimax rates for pool-based active learning, suggesting it can optimally adapt to benign settings.

----

### Gigastep - One Billion Steps per Second Multi-agent Reinforcement Learning.

**Authors**: Mathias Lechner, Lianhao Yin, Tim Seyde, Tsun-Hsuan Johnson Wang, Wei Xiao, Ramin M. Hasani, Joshua Rountree, Daniela Rus

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00ba06ba5c324efdfb068865ca44cf0b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/00ba06ba5c324efdfb068865ca44cf0b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Multi-agent reinforcement learning (MARL) research is faced with a trade-off: it either uses complex environments requiring large compute resources, which makes it inaccessible to researchers with limited resources, or relies on simpler dynamics for faster execution, which makes the transferability of the results to more realistic tasks challenging. Motivated by these challenges, we present Gigastep, a fully vectorizable, MARL environment implemented in JAX, capable of executing up to one billion environment steps per second on consumer-grade hardware. Its design allows for comprehensive MARL experimentation, including a complex, high-dimensional space defined by 3D dynamics, stochasticity, and partial observations. Gigastep supports both collaborative and adversarial tasks, continuous and discrete action spaces, and provides RGB image and feature vector observations, allowing the evaluation of a wide range of MARL algorithms. We validate Gigastep's usability through an extensive set of experiments, underscoring its role in widening participation and promoting inclusivity in the MARL research community.

----

### Attentive Transfer Entropy to Exploit Transient Emergence of Coupling Effect.

**Authors**: Xiaolei Ru, Xinya Zhang, Zijia Liu, Jack Murdoch Moore, Gang Yan

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html)

**Abstract**:

We consider the problem of reconstructing coupled networks (e.g., biological neural networks) connecting large numbers of variables (e.g.,nerve cells), of which state evolution is governed by dissipative dynamics consisting of strong self-drive (dominants the evolution) and weak coupling-drive. The core difficulty is sparseness of coupling effect that emerges (the coupling force is significant) only momentarily and otherwise remains quiescent in time series (e.g., neuronal activity sequence). Here we learn the idea from attention mechanism to guide the classifier to make inference focusing on the critical regions of time series data where coupling effect may manifest. Specifically, attention coefficients are assigned autonomously by artificial neural networks trained to maximise the Attentive Transfer Entropy (ATEn), which is a novel generalization of the iconic transfer entropy metric. Our results show that, without any prior knowledge of dynamics, ATEn explicitly identifies areas where the strength of coupling-drive is distinctly greater than zero. This innovation substantially improves reconstruction performance for both synthetic and real directed coupling networks using data generated by neuronal models widely used in neuroscience.

----

### PopSign ASL v1.0: An Isolated American Sign Language Dataset Collected via Smartphones.

**Authors**: Thad Starner, Sean Forbes, Matthew So, David Martin, Rohit Sridhar, Gururaj Deshpande, Sam S. Sepah, Sahir Shahryar, Khushi Bhardwaj, Tyler Kwok, Daksh Sehgal, Saad Hassan, Bill Neubauer, Sofia Anandi Vempala, Alec Tan, Jocelyn Heath, Unnathi Kumar, Priyanka Mosur, Tavenner Hall, Rajandeep Singh, Christopher Cui, Glenn Cameron, Sohier Dane, Garrett Tanzer

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

PopSign is a smartphone-based bubble-shooter game that helps hearing parentsof deaf infants learn sign language. To help parents practice their ability to sign,PopSign is integrating sign language recognition as part of its gameplay. Fortraining the recognizer, we introduce the PopSign ASL v1.0 dataset that collectsexamples of 250 isolated American Sign Language (ASL) signs using Pixel 4Asmartphone selfie cameras in a variety of environments. It is the largest publiclyavailable, isolated sign dataset by number of examples and is the first dataset tofocus on one-handed, smartphone signs. We collected over 210,000 examplesat 1944x2592 resolution made by 47 consenting Deaf adult signers for whomAmerican Sign Language is their primary language. We manually reviewed 217,866of these examples, of which 175,023 (approximately 700 per sign) were the signintended for the educational game. 39,304 examples were recognizable as a signbut were not the desired variant or were a different sign. We provide a training setof 31 signers, a validation set of eight signers, and a test set of eight signers. Abaseline LSTM model for the 250-sign vocabulary achieves 82.1% accuracy (81.9%class-weighted F1 score) on the validation set and 84.2% (83.9% class-weightedF1 score) on the test set. Gameplay suggests that accuracy will be sufficient forcreating educational games involving sign language recognition.

----

### (Provable) Adversarial Robustness for Group Equivariant Tasks: Graphs, Point Clouds, Molecules, and More.

**Authors**: Jan Schuchardt, Yan Scholten, Stephan Günnemann

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00db17c36b5435195760520efa96d99c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00db17c36b5435195760520efa96d99c-Abstract-Conference.html)

**Abstract**:

A machine learning model is traditionally considered robust if its prediction remains (almost) constant under input perturbations with small norm. However, real-world tasks like molecular property prediction or point cloud segmentation have inherent equivariances, such as rotation or permutation equivariance. In such tasks, even perturbations with large norm do not necessarily change an input's semantic content. Furthermore, there are perturbations for which a model's prediction explicitly needs to change. For the first time, we propose a sound notion of adversarial robustness that accounts for task equivariance. We then demonstrate that provable robustness can be achieved by (1) choosing a model that matches the task's equivariances (2) certifying traditional adversarial robustness. Certification methods are, however, unavailable for many models, such as those with continuous equivariances. We close this gap by developing the framework of equivariance-preserving randomized smoothing, which enables architecture-agnostic certification. We additionally derive the first architecture-specific graph edit distance certificates, i.e. sound robustness guarantees for isomorphism equivariant tasks like node classification. Overall, a sound notion of robustness is an important prerequisite for future work at the intersection of robust and geometric machine learning.

----

### Self-Supervised Motion Magnification by Backpropagating Through Optical Flow.

**Authors**: Zhaoying Pan, Daniel Geng, Andrew Owens

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00ed9ab006311be67879ecef8f80d7c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00ed9ab006311be67879ecef8f80d7c5-Abstract-Conference.html)

**Abstract**:

This paper presents a simple, self-supervised method for magnifying subtle motions in video: given an input video and a magnification factor, we manipulate the video such that its new optical flow is scaled by the desired amount. To train our model, we propose a loss function that estimates the optical flow of the generated video and penalizes how far if deviates from the given magnification factor. Thus, training involves differentiating through a pretrained optical flow network. Since our model is self-supervised, we can further improve its performance through test-time adaptation, by finetuning it on the input video. It can also be easily extended to magnify the motions of only user-selected objects. Our approach avoids the need for synthetic magnification datasets that have been used to train prior learning-based approaches. Instead, it leverages the existing capabilities of off-the-shelf motion estimators. We demonstrate the effectiveness of our method through evaluations of both visual quality and quantitative metrics on a range of real-world and synthetic videos, and we show our method works for both supervised and unsupervised optical flow methods.

----

### TexQ: Zero-shot Network Quantization with Texture Feature Distribution Calibration.

**Authors**: Xinrui Chen, Yizhi Wang, Renao Yan, Yiqing Liu, Tian Guan, Yonghong He

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html)

**Abstract**:

Quantization is an effective way to compress neural networks. By reducing the bit width of the parameters, the processing efficiency of neural network models at edge devices can be notably improved. Most conventional quantization methods utilize real datasets to optimize quantization parameters and fine-tune. Due to the inevitable privacy and security issues of real samples, the existing real-data-driven methods are no longer applicable. Thus, a natural method is to introduce synthetic samples for zero-shot quantization (ZSQ). However, the conventional synthetic samples fail to retain the detailed texture feature distributions, which severely limits the knowledge transfer and performance of the quantized model. In this paper, a novel ZSQ method, TexQ is proposed to address this issue. We first synthesize a calibration image and extract its calibration center for each class with a texture feature energy distribution calibration method. Then, the calibration centers are used to guide the generator to synthesize samples. Finally, we introduce the mixup knowledge distillation module to diversify synthetic samples for fine-tuning. Extensive experiments on CIFAR10/100 and ImageNet show that TexQ is observed to perform state-of-the-art in ultra-low bit width quantization. For example, when ResNet-18 is quantized to 3-bit, TexQ achieves a 12.18% top-1 accuracy increase on ImageNet compared to state-of-the-art methods. Code at https://github.com/dangsingrue/TexQ.

----

### Ambient Diffusion: Learning Clean Distributions from Corrupted Data.

**Authors**: Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alex Dimakis, Adam R. Klivans

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/012af729c5d14d279581fc8a5db975a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/012af729c5d14d279581fc8a5db975a1-Abstract-Conference.html)

**Abstract**:

We present the first diffusion-based framework that can learn an unknown distribution using only highly-corrupted samples. This problem arises in scientific applications where access to uncorrupted samples is impossible or expensive to acquire. Another benefit of our approach is the ability to train generative models that are less likely to memorize any individual training sample, since they never observe clean training data. Our main idea is to introduce additional measurement distortion during the diffusion process and require the model to predict the original corrupted image from the further corrupted image.  We prove that our method leads to models that learn the conditional expectation of the full uncorrupted image given this additional measurement corruption.  This holds for any corruption process that satisfies some technical conditions (and in particular includes inpainting and compressed sensing).  We train models on standard benchmarks (CelebA, CIFAR-10 and AFHQ) and show that we can learn the distribution even when all the training samples have 90\% of their pixels missing. We also show that we can finetune foundation models on small corrupted datasets (e.g. MRI scans with block corruptions) and learn the clean distribution without memorizing the training set.

----

### Scalable Membership Inference Attacks via Quantile Regression.

**Authors**: Martin Bertran, Shuai Tang, Aaron Roth, Michael Kearns, Jamie Morgenstern, Steven Wu

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01328d0767830e73a612f9073e9ff15f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01328d0767830e73a612f9073e9ff15f-Abstract-Conference.html)

**Abstract**:

Membership inference attacks are designed to determine, using black box access to trained models, whether a particular example was used in training or not. Membership inference can be formalized as a hypothesis testing problem. The most effective existing attacks estimate the distribution of some test statistic (usually the model's confidence on the true label) on points that were (and were not) used in training by training many \emph{shadow models}---i.e. models of the same architecture as the model being attacked, trained on a random subsample of data. While effective, these attacks are extremely computationally expensive, especially when the model under attack is large. \footnotetext[0]{Martin and Shuai are the lead authors, and other authors are ordered alphabetically. {maberlop,shuat}@amazon.com}We introduce a new class of attacks based on performing quantile regression on the distribution of confidence scores induced by the model under attack on points that are not used in training. We show that our method is competitive with state-of-the-art shadow model attacks, while requiring substantially less compute because our attack requires training only a single model. Moreover, unlike shadow model attacks, our proposed attack does not require any knowledge of the architecture of the model under attack and is therefore truly ``black-box". We show the efficacy of this approach in an extensive series of experiments on various datasets and model architectures. Our code is available at \href{https://github.com/amazon-science/quantile-mia}{github.com/amazon-science/quantile-mia.}

----

### ESSEN: Improving Evolution State Estimation for Temporal Networks using Von Neumann Entropy.

**Authors**: Qiyao Huang, Yingyue Zhang, Zhihong Zhang, Edwin R. Hancock

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0147d967a5db3b8dde08d2a327b24568-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0147d967a5db3b8dde08d2a327b24568-Abstract-Conference.html)

**Abstract**:

Temporal networks are widely used as abstract graph representations for real-world dynamic systems. Indeed, recognizing the network evolution states is crucial in understanding and analyzing temporal networks. For instance, social networks will generate the clustering and formation of tightly-knit groups or communities over time, relying on the triadic closure theory. However, the existing methods often struggle to account for the time-varying nature of these network structures, hindering their performance when applied to networks with complex evolution states. To mitigate this problem, we propose a novel framework called ESSEN, an Evolution StateS awarE Network, to measure temporal network evolution using von Neumann entropy and thermodynamic temperature. The developed framework utilizes a von Neumann entropy aware attention mechanism and network evolution state contrastive learning in the graph encoding. In addition, it employs a unique decoder the so-called Mixture of Thermodynamic Experts (MoTE) for decoding. ESSEN extracts local and global network evolution information using thermodynamic features and adaptively recognizes the network evolution states. Moreover, the proposed method is evaluated on link prediction tasks under both transductive and inductive settings, with the corresponding results demonstrating its effectiveness compared to various state-of-the-art baselines.

----

### Label Correction of Crowdsourced Noisy Annotations with an Instance-Dependent Noise Transition Model.

**Authors**: Hui Guo, Boyu Wang, Grace Yi

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/015a8c69bedcb0a7b2ed2e1678f34399-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/015a8c69bedcb0a7b2ed2e1678f34399-Abstract-Conference.html)

**Abstract**:

The predictive ability of supervised learning algorithms hinges on the quality of annotated examples, whose labels often come from multiple crowdsourced annotators with diverse expertise. To aggregate noisy crowdsourced annotations, many existing methods employ an annotator-specific instance-independent noise transition matrix  to characterize the labeling skills of each annotator. Learning an instance-dependent noise transition model, however, is challenging and remains relatively less explored. To address this problem, in this paper, we formulate the noise transition model in a Bayesian framework and subsequently design a new label correction algorithm. Specifically, we approximate the instance-dependent noise transition matrices using a Bayesian network with a hierarchical spike and slab prior. To theoretically characterize the distance between the noise transition model and the true instance-dependent noise transition matrix, we provide a posterior-concentration theorem that ensures the posterior consistency in terms of the Hellinger distance. We further formulate the label correction process as a hypothesis testing problem and propose a novel algorithm to infer the true label from the noisy annotations based on the pairwise likelihood ratio test. Moreover, we establish an information-theoretic bound on the Bayes error for the proposed method. We validate the effectiveness of our approach through experiments on benchmark and real-world datasets.

----

### Diffused Task-Agnostic Milestone Planner.

**Authors**: Mineui Hong, Minjae Kang, Songhwai Oh

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0163ca1c69f848e766cfb0b7bb7e17f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0163ca1c69f848e766cfb0b7bb7e17f4-Abstract-Conference.html)

**Abstract**:

Addressing decision-making problems using sequence modeling to predict future trajectories shows promising results in recent years.In this paper, we take a step further to leverage the sequence predictive method in wider areas such as long-term planning, vision-based control, and multi-task decision-making.To this end, we propose a method to utilize a diffusion-based generative sequence model to plan a series of milestones in a latent space and to have an agent to follow the milestones to accomplish a given task.The proposed method can learn control-relevant, low-dimensional latent representations of milestones, which makes it possible to efficiently perform long-term planning and vision-based control.Furthermore, our approach exploits generation flexibility of the diffusion model, which makes it possible to plan diverse trajectories for multi-task decision-making.We demonstrate the proposed method across offline reinforcement learning (RL) benchmarks and an visual manipulation environment.The results show that our approach outperforms offline RL methods in solving long-horizon, sparse-reward tasks and multi-task problems,while also achieving the state-of-the-art performance on the most challenging vision-based manipulation benchmark.

----

### Task-aware Distributed Source Coding under Dynamic Bandwidth.

**Authors**: Po-han Li, Sravan Kumar Ankireddy, Ruihan Philip Zhao, Hossein Nourkhiz Mahjoub, Ehsan Moradi-Pari, Ufuk Topcu, Sandeep Chinchali, Hyeji Kim

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/016c63403370d81c24c1ca0123de6cfa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/016c63403370d81c24c1ca0123de6cfa-Abstract-Conference.html)

**Abstract**:

Efficient compression of correlated data is essential to minimize communication overload in multi-sensor networks. In such networks, each sensor independently compresses the data and transmits them to a central node. A decoder at the central node decompresses and passes the data to a pre-trained machine learning-based task model to generate the final output. Due to limited communication bandwidth, it is important for the compressor to learn only the features that are relevant to the task. Additionally, the final performance depends heavily on the total available bandwidth. In practice, it is common to encounter varying availability in bandwidth. Since higher bandwidth results in better performance, it is essential for the compressor to dynamically take advantage of the maximum available bandwidth at any instant. In this work, we propose a novel distributed compression framework composed of independent encoders and a joint decoder, which we call neural distributed principal component analysis (NDPCA). NDPCA flexibly compresses data from multiple sources to any available bandwidth with a single model, reducing compute and storage overhead. NDPCA achieves this by learning low-rank task representations and efficiently distributing bandwidth among sensors, thus providing a graceful trade-off between performance and bandwidth. Experiments show that NDPCA improves the success rate of multi-view robotic arm manipulation by 9% and the accuracy of object detection tasks on satellite imagery by 14% compared to an autoencoder with uniform bandwidth allocation.

----

### BubbleML: A Multiphase Multiphysics Dataset and Benchmarks for Machine Learning.

**Authors**: Sheikh Md Shakeel Hassan, Arthur Feeney, Akash Dhruv, Jihoon Kim, Youngjoon Suh, Jaiyoung Ryu, Yoonjin Won, Aparna Chandramowlishwaran

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01726ae05d72ddba3ac784a5944fa1ef-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/01726ae05d72ddba3ac784a5944fa1ef-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In the field of phase change phenomena, the lack of accessible and diverse datasets suitable for machine learning (ML) training poses a significant challenge. Existing experimental datasets are often restricted, with limited availability and sparse ground truth, impeding our understanding of this complex multiphysics phenomena. To bridge this gap, we present the BubbleML dataset which leverages physics-driven simulations to provide accurate ground truth information for various boiling scenarios, encompassing nucleate pool boiling, flow boiling, and sub-cooled boiling. This extensive dataset covers a wide range of parameters, including varying gravity conditions, flow rates, sub-cooling levels, and wall superheat, comprising 79 simulations.  BubbleML is validated against experimental observations and trends, establishing it as an invaluable resource for ML research. Furthermore, we showcase its potential to facilitate the exploration of diverse downstream tasks by introducing two benchmarks: (a) optical flow analysis to capture bubble dynamics, and (b) neural PDE solvers for learning temperature and flow dynamics. The BubbleML dataset and its benchmarks aim to catalyze progress in ML-driven research on multiphysics phase change phenomena, providing robust baselines for the development and comparison of state-of-the-art techniques and models.

----

### ANTN: Bridging Autoregressive Neural Networks and Tensor Networks for Quantum Many-Body Simulation.

**Authors**: Zhuo Chen, Laker Newhouse, Eddie Chen, Di Luo, Marin Soljacic

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01772a8b0420baec00c4d59fe2fbace6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01772a8b0420baec00c4d59fe2fbace6-Abstract-Conference.html)

**Abstract**:

Quantum many-body physics simulation has important impacts on understanding fundamental science and has applications to quantum materials design and quantum technology. However, due to the exponentially growing size of the Hilbert space with respect to the particle number, a direct simulation is intractable. While representing quantum states with tensor networks and neural networks are the two state-of-the-art methods for approximate simulations, each has its own limitations in terms of expressivity and inductive bias. To address these challenges, we develop a novel architecture, Autoregressive Neural TensorNet (ANTN), which bridges tensor networks and autoregressive neural networks. We show that Autoregressive Neural TensorNet parameterizes normalized wavefunctions, allows for exact sampling, generalizes the expressivity of tensor networks and autoregressive neural networks, and inherits a variety of symmetries from autoregressive neural networks. We demonstrate our approach on quantum state learning as well as finding the ground state of the challenging 2D $J_1$-$J_2$ Heisenberg model with different systems sizes and coupling parameters, outperforming both tensor networks and autoregressive neural networks. Our work opens up new opportunities for quantum many-body physics simulation, quantum technology design, and generative modeling in artificial intelligence.

----

### Causal Effect Identification in Uncertain Causal Networks.

**Authors**: Sina Akbari, Fateme Jamshidi, Ehsan Mokhtarian, Matthew J. Vowels, Jalal Etesami, Negar Kiyavash

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/017c897b4d85a744f345ccbf9d71e501-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/017c897b4d85a744f345ccbf9d71e501-Abstract-Conference.html)

**Abstract**:

Causal identification is at the core of the causal inference literature, where complete algorithms have been proposed to identify causal queries of interest. The validity of these algorithms hinges on the restrictive assumption of having access to a correctly specified causal structure. In this work, we study the setting where a probabilistic model of the causal structure is available. Specifically, the edges in a causal graph exist with uncertainties which may, for example, represent degree of belief from domain experts. Alternatively, the uncertainty about an edge may reflect the confidence of a particular statistical test. The question that naturally arises in this setting is: Given such a probabilistic graph and a specific causal effect of interest, what is the subgraph which has the highest plausibility and for which the causal effect is identifiable? We show that answering this question reduces to solving an NP-hard combinatorial optimization problem which we call the edge ID problem. We propose efficient algorithms to approximate this problem and evaluate them against both real-world networks and randomly generated graphs.

----

### FAST: a Fused and Accurate Shrinkage Tree for Heterogeneous Treatment Effects Estimation.

**Authors**: Jia Gu, Caizhi Tang, Han Yan, Qing Cui, Longfei Li, Jun Zhou

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01830c92c6558179fa6d7fb1edff692c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01830c92c6558179fa6d7fb1edff692c-Abstract-Conference.html)

**Abstract**:

This paper proposes a novel strategy for estimating the heterogeneous treatment effect  called the Fused and Accurate Shrinkage Tree ($\mathrm{FAST}$). Our approach utilizes both trial and observational data to improve the accuracy and robustness of the estimator. Inspired by the concept of shrinkage estimation in statistics, we develop an optimal weighting scheme and a corresponding estimator that balances the unbiased estimator based on the trial data with the potentially biased estimator based on the observational data. Specifically, combined with tree-based techniques, we introduce a new split criterion that utilizes both trial data and observational data to more accurately estimate the treatment effect. Furthermore, we confirm the consistency of our proposed tree-based estimator and demonstrate the effectiveness of our criterion in reducing prediction error through theoretical analysis.  The advantageous  finite sample performance of the $\mathrm{FAST}$ and its ensemble version over existing methods is demonstrated via  simulations and real data analysis.

----

### Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond.

**Authors**: Oleg Platonov, Denis Kuznedelev, Artem Babenko, Liudmila Prokhorenkova

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01b681025fdbda8e935a66cc5bb6e9de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01b681025fdbda8e935a66cc5bb6e9de-Abstract-Conference.html)

**Abstract**:

Homophily is a graph property describing the tendency of edges to connect similar nodes; the opposite is called heterophily. It is often believed that heterophilous graphs are challenging for standard message-passing graph neural networks (GNNs), and much effort has been put into developing efficient methods for this setting. However, there is no universally agreed-upon measure of homophily in the literature. In this work, we show that commonly used homophily measures have critical drawbacks preventing the comparison of homophily levels across different datasets. For this, we formalize desirable properties for a proper homophily measure and verify which measures satisfy which properties. In particular, we show that a measure that we call adjusted homophily satisfies more desirable properties than other popular homophily measures while being rarely used in graph machine learning literature. Then, we go beyond the homophily-heterophily dichotomy and propose a new characteristic that allows one to further distinguish different sorts of heterophily. The proposed label informativeness (LI) characterizes how much information a neighbor's label provides about a node's label. We prove that this measure satisfies important desirable properties. We also observe empirically that LI better agrees with GNN performance compared to homophily measures, which confirms that it is a useful characteristic of the graph structure.

----

### Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation.

**Authors**: Yuxuan Song, Jingjing Gong, Minkai Xu, Ziyao Cao, Yanyan Lan, Stefano Ermon, Hao Zhou, Wei-Ying Ma

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01d64478381c33e29ed611f1719f5a37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01d64478381c33e29ed611f1719f5a37-Abstract-Conference.html)

**Abstract**:

The generation of 3D molecules requires simultaneously deciding the categorical features (atom types) and continuous features (atom coordinates). Deep generative models, especially Diffusion Models (DMs), have demonstrated effectiveness in generating feature-rich geometries. However, existing DMs typically suffer from unstable probability dynamics with inefficient sampling speed. In this paper, we introduce geometric flow matching, which enjoys the advantages of both equivariant modeling and stabilized probability dynamics. More specifically, we propose a hybrid probability path where the coordinates probability path is regularized by an equivariant optimal transport, and the information between different modalities is aligned. Experimentally, the proposed method could consistently achieve better performance on multiple molecule generation benchmarks with 4.75$\times$ speed up of sampling on average.

----

### Hyperbolic VAE via Latent Gaussian Distributions.

**Authors**: Seunghyuk Cho, Juyong Lee, Dongwoo Kim

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html)

**Abstract**:

We propose a Gaussian manifold variational auto-encoder (GM-VAE) whose latent space consists of a set of Gaussian distributions. It is known that the set of the univariate Gaussian distributions with the Fisher information metric form a hyperbolic space, which we call a Gaussian manifold. To learn the VAE endowed with the Gaussian manifolds, we propose a pseudo-Gaussian manifold normal distribution based on the Kullback-Leibler divergence, a local approximation of the squared Fisher-Rao distance, to define a density over the latent space. We demonstrate the efficacy of GM-VAE on two different tasks: density estimation of image datasets and state representation learning for model-based reinforcement learning. GM-VAE outperforms the other variants of hyperbolic- and Euclidean-VAEs on density estimation tasks and shows competitive performance in model-based reinforcement learning. We observe that our model provides strong numerical stability, addressing a common limitation reported in previous hyperbolic-VAEs. The implementation is available at https://github.com/ml-postech/GM-VAE.

----

### A Simple Solution for Offline Imitation from Observations and Examples with Possibly Incomplete Trajectories.

**Authors**: Kai Yan, Alexander G. Schwing, Yu-Xiong Wang

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0203f489345567b4a048c38f507cdbfa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0203f489345567b4a048c38f507cdbfa-Abstract-Conference.html)

**Abstract**:

Offline imitation from observations aims to solve MDPs where only task-specific expert states and task-agnostic non-expert state-action pairs are available. Offline imitation is useful in real-world scenarios where arbitrary interactions are costly and expert actions are unavailable. The state-of-the-art ‘DIstribution Correction Estimation’ (DICE) methods minimize divergence of state occupancy between expert and learner policies and retrieve a policy with weighted behavior cloning; however, their results are unstable when learning from incomplete trajectories, due to a non-robust optimization in the dual domain. To address the issue, in this paper, we propose Trajectory-Aware Imitation Learning from Observations (TAILO). TAILO uses a discounted sum along the future trajectory as the weight for weighted behavior cloning. The terms for the sum are scaled by the output of a discriminator, which aims to identify expert states. Despite simplicity, TAILO works well if there exist trajectories or segments of expert behavior in the task-agnostic data, a common assumption in prior work. In experiments across multiple testbeds, we find TAILO to be more robust and effective, particularly with incomplete trajectories.

----

### Defending against Data-Free Model Extraction by Distributionally Robust Defensive Training.

**Authors**: Zhenyi Wang, Li Shen, Tongliang Liu, Tiehang Duan, Yanjun Zhu, Donglin Zhan, David S. Doermann, Mingchen Gao

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0207c9ea9faf66c6e892c3fa3c167b75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0207c9ea9faf66c6e892c3fa3c167b75-Abstract-Conference.html)

**Abstract**:

Data-Free Model Extraction (DFME) aims to clone a black-box model without knowing its original training data distribution, making it much easier for attackers to steal commercial models. Defense against DFME faces several challenges: (i) effectiveness; (ii) efficiency; (iii) no prior on the attacker's query data distribution and strategy. However, existing defense methods: (1) are highly computation and memory inefficient; or (2) need strong assumptions about attack data distribution; or (3) can only delay the attack or prove a model theft after the model stealing has happened. In this work, we propose a Memory and Computation efficient defense approach, named MeCo, to prevent DFME from happening while maintaining the model utility simultaneously by distributionally robust defensive training on the target victim model. Specifically, we randomize the input so that it: (1) causes a mismatch of the knowledge distillation loss for attackers; (2) disturbs the zeroth-order gradient estimation; (3) changes the label prediction for the attack query data. Therefore, the attacker can only extract misleading information from the black-box model. Extensive experiments on defending against both decision-based and score-based DFME demonstrate that MeCo can significantly reduce the effectiveness of existing DFME methods and substantially improve running efficiency.

----

### Large language models transition from integrating across position-yoked, exponential windows to structure-yoked, power-law windows.

**Authors**: David Skrill, Samuel Norman-Haignere

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/020ad0ac6a1974e6748e4a5a48110a07-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/020ad0ac6a1974e6748e4a5a48110a07-Abstract-Conference.html)

**Abstract**:

Modern language models excel at integrating across long temporal scales needed to encode linguistic meaning and show non-trivial similarities to biological neural systems. Prior work suggests that human brain responses to language exhibit hierarchically organized "integration windows" that substantially constrain the overall influence of an input token (e.g., a word) on the neural response. However, little prior work has attempted to use integration windows to characterize computations in large language models (LLMs). We developed a simple word-swap procedure for estimating integration windows from black-box language models that does not depend on access to gradients or knowledge of the model architecture (e.g., attention weights). Using this method, we show that trained LLMs exhibit stereotyped integration windows that are well-fit by a convex combination of an exponential and a power-law function, with a partial transition from exponential to power-law dynamics across network layers. We then introduce a metric for quantifying the extent to which these integration windows vary with structural boundaries (e.g., sentence boundaries), and using this metric, we show that integration windows become increasingly yoked to structure at later network layers. None of these findings were observed in an untrained model, which as expected integrated uniformly across its input. These results suggest that LLMs learn to integrate information in natural language using a stereotyped pattern: integrating across position-yoked, exponential windows at early layers, followed by structure-yoked, power-law windows at later layers. The methods we describe in this paper provide a general-purpose toolkit for understanding temporal integration in language models, facilitating cross-disciplinary research at the intersection of biological and artificial intelligence.

----

### Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?

**Authors**: Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Yecheng Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, Franziska Meier

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/022ca1bed6b574b962c48a2856eb207b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/022ca1bed6b574b962c48a2856eb207b-Abstract-Conference.html)

**Abstract**:

We present the largest and most comprehensive empirical study of pre-trained visual representations (PVRs) or visual ‘foundation models’ for Embodied AI. First, we curate CortexBench, consisting of 17 different tasks spanning locomotion, navigation, dexterous, and mobile manipulation. Next, we systematically evaluate existing PVRs and find that none are universally dominant. To study the effect of pre-training data size and diversity, we combine over 4,000 hours of egocentric videos from 7 different sources (over 4.3M images) and ImageNet to train different-sized vision transformers using Masked Auto-Encoding (MAE) on slices of this data. Contrary to inferences from prior work, we find that scaling dataset size and diversity does not improve performance universally (but does so on average). Our largest model, named VC-1, outperforms all prior PVRs on average but does not universally dominate either. Next, we show that task- or domain-specific adaptation of VC-1 leads to substantial gains, with VC-1 (adapted) achieving competitive or superior performance than the best known results on all of the benchmarks in CortexBench. Finally, we present real-world hardware experiments, in which VC-1 and VC-1 (adapted) outperform the strongest pre-existing PVR. Overall, this paper presents no new techniques but a rigorous systematic evaluation, a broad set of findings about PVRs (that in some cases, refute those made in narrow domains in prior work), and open-sourced code and models (that required over 10,000 GPU-hours to train) for the benefit of the research community.

----

### Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback.

**Authors**: Jangwon Kim, Hangyeol Kim, Jiwook Kang, Jongchan Baek, Soohee Han

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0252a434b18962c94910c07cd9a7fecc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0252a434b18962c94910c07cd9a7fecc-Abstract-Conference.html)

**Abstract**:

We present a novel actor-critic algorithm for an environment with delayed feedback, which addresses the state-space explosion problem of conventional approaches. Conventional approaches use an augmented state constructed from the last observed state and actions executed since visiting the last observed state. Using the augmented state space, the correct Markov decision process for delayed environments can be constructed; however, this causes the state space to explode as the number of delayed timesteps increases, leading to slow convergence. Our proposed algorithm, called Belief-Projection-Based Q-learning (BPQL), addresses the state-space explosion problem by evaluating the values of the critic for which the input state size is equal to the original state-space size rather than that of the augmented one. We compare BPQL to traditional approaches in continuous control tasks and demonstrate that it significantly outperforms other algorithms in terms of asymptotic performance and sample efficiency. We also show that BPQL solves long-delayed environments, which conventional approaches are unable to do.

----

### Batchnorm Allows Unsupervised Radial Attacks.

**Authors**: Amur Ghose, Apurv Gupta, Yaoliang Yu, Pascal Poupart

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0266d95023740481d22d437aa8aba0e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0266d95023740481d22d437aa8aba0e9-Abstract-Conference.html)

**Abstract**:

The construction of adversarial examples usually requires the existence of soft or hard labels for each instance, with respect to which a loss gradient provides the signal for construction of the example. We show that for batch normalized deep image recognition architectures, intermediate latents that are produced after a batch normalization step by themselves suffice to produce adversarial examples using an intermediate loss solely utilizing angular deviations, without relying on any label. We motivate our loss through the geometry of batch normed representations and their concentration of norm on a hypersphere and distributional proximity to Gaussians. Our losses expand intermediate latent based attacks that usually require labels. The success of our method implies that leakage of intermediate representations may create a security breach for deployed models, which persists even when the model is transferred to downstream usage. Removal of batch norm weakens our attack, indicating it contributes to this vulnerability. Our attacks also succeed against LayerNorm empirically, thus being relevant for transformer architectures, most notably vision transformers which we analyze.

----

### Detecting Any Human-Object Interaction Relationship: Universal HOI Detector with Spatial Prompt Learning on Foundation Models.

**Authors**: Yichao Cao, Qingfei Tang, Xiu Su, Song Chen, Shan You, Xiaobo Lu, Chang Xu

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html)

**Abstract**:

Human-object interaction (HOI) detection aims to comprehend the intricate relationships between humans and objects, predicting  triplets, and serving as the foundation for numerous computer vision tasks. The complexity and diversity of human-object interactions in the real world, however, pose significant challenges for both annotation and recognition, particularly in recognizing interactions within an open world context. This study explores the universal interaction recognition in an open-world setting through the use of Vision-Language (VL) foundation models and large language models (LLMs). The proposed method is dubbed as UniHOI. We conduct a deep analysis of the three hierarchical features inherent in visual HOI detectors and propose a method for high-level relation extraction aimed at VL foundation models, which we call HO prompt-based learning. Our design includes an HO Prompt-guided Decoder (HOPD), facilitates the association of high-level relation representations in the foundation model with various HO pairs within the image. Furthermore, we utilize a LLM (i.e. GPT) for interaction interpretation, generating a richer linguistic understanding for complex HOIs. For open-category interaction recognition, our method supports either of two input types: interaction phrase or interpretive sentence.  Our efficient architecture design and learning methods effectively unleash the potential of the VL foundation models and LLMs, allowing UniHOI to surpass all existing methods with a substantial margin, under both supervised and zero-shot settings. The code and pre-trained weights will be made publicly available.

----

### Smoothing the Landscape Boosts the Signal for SGD: Optimal Sample Complexity for Learning Single Index Models.

**Authors**: Alex Damian, Eshaan Nichani, Rong Ge, Jason D. Lee

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02763667a5761ff92bb15d8751bcd223-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02763667a5761ff92bb15d8751bcd223-Abstract-Conference.html)

**Abstract**:

We focus on the task of learning a single index model $\sigma(w^\star \cdot x)$ with respect to the isotropic Gaussian distribution in $d$ dimensions. Prior work has shown that the sample complexity of learning $w^\star$ is governed by the information exponent $k^\star$ of the link function $\sigma$, which is defined as the index of the first nonzero Hermite coefficient of $\sigma$. Ben Arous et al. (2021) showed that $n \gtrsim d^{k^\star-1}$ samples suffice for learning $w^\star$ and that this is tight for online SGD. However, the CSQ lower bound for gradient based methods only shows that $n \gtrsim d^{k^\star/2}$ samples are necessary. In this work, we close the gap between the upper and lower bounds by showing that online SGD on a smoothed loss learns $w^\star$ with $n \gtrsim d^{k^\star/2}$ samples. We also draw connections to statistical analyses of tensor PCA and to the implicit regularization effects of minibatch SGD on empirical losses.

----

### A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models.

**Authors**: Alexander G. Reisach, Myriam Tami, Christof Seiler, Antoine Chambaz, Sebastian Weichwald

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/027e86facfe7c1ea52ca1fca7bc1402b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/027e86facfe7c1ea52ca1fca7bc1402b-Abstract-Conference.html)

**Abstract**:

Additive Noise Models (ANMs) are a common model class for causal discovery from observational data. Due to a lack of real-world data for which an underlying ANM is known, ANMs with randomly sampled parameters are commonly used to simulate data for the evaluation of causal discovery algorithms. While some parameters may be fixed by explicit assumptions, fully specifying an ANM requires choosing all parameters. Reisach et al. (2021) show that, for many ANM parameter choices, sorting the variables by increasing variance yields an ordering close to a causal order and introduce ‘var-sortability’ to quantify this alignment. Since increasing variances may be unrealistic and cannot be exploited when data scales are arbitrary, ANM data are often rescaled to unit variance in causal discovery benchmarking.We show that synthetic ANM data are characterized by another pattern that is scale-invariant and thus persists even after standardization: the explainable fraction of a variable’s variance, as captured by the coefficient of determination $R^2$, tends to increase along the causal order. The result is high ‘$R^2$-sortability’, meaning that sorting the variables by increasing $R^2$ yields an ordering close to a causal order. We propose a computationally efficient baseline algorithm termed ‘$R^2$-SortnRegress’ that exploits high $R^2$-sortability and that can match and exceed the performance of established causal discovery algorithms. We show analytically that sufficiently high edge weights lead to a relative decrease of the noise contributions along causal chains, resulting in increasingly deterministic relationships and high $R^2$. We characterize $R^2$-sortability on synthetic data with different simulation parameters and find high values in common settings. Our findings reveal high $R^2$-sortability as an assumption about the data generating process relevant to causal discovery and implicit in many ANM sampling schemes. It should be made explicit, as its prevalence in real-world data is an open question. For causal discovery benchmarking, we provide implementations of $R^2$-sortability, the $R^2$-SortnRegress algorithm, and ANM simulation procedures in our library CausalDisco at https://causaldisco.github.io/CausalDisco/.

----

### PROTES: Probabilistic Optimization with Tensor Sampling.

**Authors**: Anastasia Batsheva, Andrei Chertkov, Gleb V. Ryzhakov, Ivan V. Oseledets

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/028957869e560af14243ac37663a471e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/028957869e560af14243ac37663a471e-Abstract-Conference.html)

**Abstract**:

We developed a new method PROTES for black-box optimization, which is based on the probabilistic sampling from a probability density function given in the low-parametric tensor train format. We tested it on complex multidimensional arrays and discretized multivariable functions taken, among others, from real-world applications, including unconstrained binary optimization and optimal control problems, for which the possible number of elements is up to $2^{1000}$. In numerical experiments, both on analytic model functions and on complex problems, PROTES outperforms popular discrete optimization methods (Particle Swarm Optimization, Covariance Matrix Adaptation, Differential Evolution, and others).

----

### Perturbation Towards Easy Samples Improves Targeted Adversarial Transferability.

**Authors**: Junqi Gao, Biqing Qi, Yao Li, Zhichang Guo, Dong Li, Yuming Xing, Dazhi Zhang

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html)

**Abstract**:

The transferability of adversarial perturbations provides an effective shortcut for black-box attacks. Targeted perturbations have greater practicality but are more difficult to transfer between models. In this paper, we experimentally and theoretically demonstrated that neural networks trained on the same dataset have more consistent performance in High-Sample-Density-Regions (HSDR) of each class instead of low sample density regions. Therefore, in the target setting, adding perturbations towards HSDR of the target class is more effective in improving transferability. However, density estimation is challenging in high-dimensional scenarios. Further theoretical and experimental verification demonstrates that easy samples with low loss are more likely to be located in HSDR. Perturbations towards such easy samples in the target class can avoid density estimation for HSDR location. Based on the above facts, we verified that adding perturbations to easy samples in the target class improves targeted adversarial transferability of existing attack methods. A generative targeted attack strategy named Easy Sample Matching Attack (ESMA) is proposed, which has a higher success rate for targeted attacks and outperforms the SOTA generative method. Moreover, ESMA requires only $5\%$ of the storage space and much less computation time comparing to the current SOTA, as ESMA attacks all classes with only one model instead of seperate models for each class. Our code is available at https://github.com/gjq100/ESMA

----

### AllSim: Simulating and Benchmarking Resource Allocation Policies in Multi-User Systems.

**Authors**: Jeroen Berrevoets, Daniel Jarrett, Alex J. Chan, Mihaela van der Schaar

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0296e17ec30fc36007edaaa2f96b5f17-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0296e17ec30fc36007edaaa2f96b5f17-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Numerous real-world systems, ranging from healthcare to energy grids, involve users competing for finite and potentially scarce resources. Designing policies for resource allocation in such real-world systems is challenging for many reasons, including the changing nature of user types and their (possibly urgent) need for resources. Researchers have developed numerous machine learning solutions for determining resource allocation policies in these challenging settings. However, a key limitation has been the absence of good methods and test-beds for benchmarking these policies; almost all resource allocation policies are benchmarked in environments which are either completely synthetic or do not allow any deviation from historical data. In this paper we introduce AllSim, which is a benchmarking environment for realistically simulating the impact and utility of policies for resource allocation in systems in which users compete for such scarce resources. Building such a benchmarking environment is challenging because it needs to successfully take into account the entire collective of potential users and the impact a resource allocation policy has on all the other users in the system. AllSim's benchmarking environment is modular (each component being parameterized individually), learnable (informed by historical data), and customizable (adaptable to changing conditions). These, when interacting with an allocation policy, produce a dataset of simulated outcomes for evaluation and comparison of such policies. We believe AllSim is an essential step towards a more systematic evaluation of policies for scarce resource allocation compared to current approaches for benchmarking such methods.

----

### AVIS: Autonomous Visual Information Seeking with Large Language Model Agent.

**Authors**: Ziniu Hu, Ahmet Iscen, Chen Sun, Kai-Wei Chang, Yizhou Sun, David Ross, Cordelia Schmid, Alireza Fathi

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/029df12a9363313c3e41047844ecad94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/029df12a9363313c3e41047844ecad94-Abstract-Conference.html)

**Abstract**:

In this paper, we propose an autonomous information seeking visual question answering framework, AVIS. Our method leverages a Large Language Model (LLM) to dynamically strategize the utilization of external tools and to investigate their outputs via tree search, thereby acquiring the indispensable knowledge needed to provide answers to the posed questions. Responding to visual questions that necessitate external knowledge, such as "What event is commemorated by the building depicted in this image?", is a complex task. This task presents a combinatorial search space that demands a sequence of actions, including invoking APIs, analyzing their responses, and making informed decisions. We conduct a user study to collect a variety of instances of human decision-making when faced with this task. This data is then used to design a system comprised of three components: an LLM-powered planner that dynamically determines which tool to use next, an LLM-powered reasoner that analyzes and extracts key information from the tool outputs, and a working memory component that retains the acquired information throughout the process. The collected user behavior serves as a guide for our system in two key ways. First, we create a transition graph by analyzing the sequence of decisions made by users. This graph delineates distinct states and confines the set of actions available at each state. Second, we use examples of user decision-making to provide our LLM-powered planner and reasoner with relevant contextual instances, enhancing their capacity to make informed decisions. We show that AVIS achieves state-of-the-art results on knowledge-based visual question answering benchmarks such as Infoseek and OK-VQA.

----

### Conformal Prediction Sets for Ordinal Classification.

**Authors**: Prasenjit Dey, Srujana Merugu, Sivaramakrishnan R. Kaveri

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/029f699912bf3db747fe110948cc6169-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/029f699912bf3db747fe110948cc6169-Abstract-Conference.html)

**Abstract**:

Ordinal classification (OC), i.e., labeling instances along classes with a natural ordering, is common in multiple  applications such as size or budget based recommendations and disease severity labeling.  Often in practical scenarios, it is desirable to obtain a small set of likely classes with a guaranteed high chance of including the true class. Recent works on conformal prediction (CP) address this problem for the classification setting with non-ordered labels but the resulting prediction sets (PS) are often non-contiguous and unsuitable for ordinal classification. In this work, we propose a framework to adapt existing CP methods to generate contiguous sets with guaranteed coverage and minimal cardinality. Our framework employs a novel non-parametric approach for modeling unimodal distributions. Empirical results on both synthetic and real-world datasets demonstrate our method outperforms SOTA baselines by 4% on Accuracy@K and 8% on PS size.

----

### Minimax-Optimal Location Estimation.

**Authors**: Shivam Gupta, Jasper C. H. Lee, Eric Price, Paul Valiant

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02a589ef9a4f6f1e2dcc1cfb3b978a51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02a589ef9a4f6f1e2dcc1cfb3b978a51-Abstract-Conference.html)

**Abstract**:

Location estimation is one of the most basic questions in parametric statistics. Suppose we have a known distribution density $f$, and we get $n$ i.i.d. samples from $f(x-\mu)$ for some unknown shift $\mu$.The task is to estimate $\mu$ to high accuracy with high probability.The maximum likelihood estimator (MLE) is known to be asymptotically optimal as $n \to \infty$, but what is possible for finite $n$?In this paper, we give two location estimators that are optimal under different criteria: 1) an estimator that has minimax-optimal estimation error subject to succeeding with probability $1-\delta$ and 2) a confidence interval estimator which, subject to its output interval containing $\mu$ with probability at least $1-\delta$, has the minimum expected squared interval width among all shift-invariant estimators.The latter construction can be generalized to minimizing the expectation of any loss function on the interval width.

----

### Tight Bounds for Volumetric Spanners and Applications.

**Authors**: Aditya Bhaskara, Sepideh Mahabadi, Ali Vakilian

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02a92b52670752daf17b53f04f1ab405-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02a92b52670752daf17b53f04f1ab405-Abstract-Conference.html)

**Abstract**:

Given a set of points of interest, a volumetric spanner is a subset of the points using which all the points can be expressed using "small" coefficients (measured in an appropriate norm). Formally, given a set of vectors $X = [v_1, v_2, \dots, v_n]$, the goal is to find $T \subseteq [n]$ such that every $v \in X$ can be expressed as $\sum_{i\in T} \alpha_i v_i$, with $\Vert \alpha \Vert$ being small.  This notion, which has also been referred to as a well-conditioned basis, has found several applications, including bandit linear optimization, determinant maximization, and matrix low rank approximation. In this paper, we give almost optimal bounds on the size of volumetric spanners for all $\ell_p$ norms, and show that they can be constructed using a simple local search procedure. We then show the applications of our result to other tasks and in particular the problem of finding coresets for the Minimum Volume Enclosing Ellipsoid (MVEE) problem.

----

### Wyze Rule: Federated Rule Dataset for Rule Recommendation Benchmarking.

**Authors**: Mohammad Mahdi Kamani, Yuhang Yao, Hanjia Lyu, Zhongwei Cheng, Lin Chen, Liangju Li, Carlee Joe-Wong, Jiebo Luo

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02b9d1e6d1b5295a6f883969ddc1bbbd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/02b9d1e6d1b5295a6f883969ddc1bbbd-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In the rapidly evolving landscape of smart home automation, the potential of IoT devices is vast. In this realm, rules are the main tool utilized for this automation, which are predefined conditions or triggers that establish connections between devices, enabling seamless automation of specific processes. However, one significant challenge researchers face is the lack of comprehensive datasets to explore and advance the field of smart home rule recommendations. These datasets are essential for developing and evaluating intelligent algorithms that can effectively recommend rules for automating processes while preserving the privacy of the users, as it involves personal information about users' daily lives. To bridge this gap, we present the Wyze Rule Dataset, a large-scale dataset designed specifically for smart home rule recommendation research. Wyze Rule encompasses over 1 million rules gathered from a diverse user base of 300,000 individuals from Wyze Labs, offering an extensive and varied collection of real-world data.   With a focus on federated learning, our dataset is tailored to address the unique challenges of a cross-device federated learning setting in the recommendation domain, featuring a large-scale number of clients with widely heterogeneous data. To establish a benchmark for comparison and evaluation, we have meticulously implemented multiple baselines in both centralized and federated settings. Researchers can leverage these baselines to gauge the performance and effectiveness of their rule recommendation systems, driving advancements in the domain. The Wyze Rule Dataset is publicly accessible through HuggingFace's dataset API.

----

### Learning better with Dale's Law: A Spectral Perspective.

**Authors**: Pingsheng Li, Jonathan Cornford, Arna Ghosh, Blake A. Richards

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02dd0db10c40092de3d9ec2508d12f60-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02dd0db10c40092de3d9ec2508d12f60-Abstract-Conference.html)

**Abstract**:

Most recurrent neural networks (RNNs) do not include a fundamental constraint of real neural circuits: Dale's Law, which implies that neurons must be excitatory (E) or inhibitory (I). Dale's Law is generally absent from RNNs because simply partitioning a standard network's units into E and I populations impairs learning. However, here we extend a recent feedforward bio-inspired EI network architecture, named Dale's ANNs, to recurrent networks, and demonstrate that good performance is possible while respecting Dale's Law. This begs the question: What makes some forms of EI network learn poorly and others learn well? And, why does the simple approach of incorporating Dale's Law impair learning?  Historically the answer was thought to be the sign constraints on EI network parameters, and this was a motivation behind Dale's ANNs. However, here we show the spectral properties of the recurrent weight matrix at initialisation are more impactful on network performance than sign constraints. We find that simple EI partitioning results in a singular value distribution that is multimodal and dispersed, whereas standard RNNs have an unimodal, more clustered singular value distribution, as do recurrent Dale's ANNs. We also show that the spectral properties and performance of partitioned EI networks are worse for small networks with fewer I units, and we present normalised SVD entropy as a measure of spectrum pathology that correlates with performance. Overall, this work sheds light on a long-standing mystery in neuroscience-inspired AI and computational neuroscience, paving the way for greater alignment between neural networks and biology.

----

### Dense-Exponential Random Features: Sharp Positive Estimators of the Gaussian Kernel.

**Authors**: Valerii Likhosherstov, Krzysztof Marcin Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamás Sarlós, Adrian Weller

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02dec8877fb7c6aa9a79f81661baca7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02dec8877fb7c6aa9a79f81661baca7c-Abstract-Conference.html)

**Abstract**:

The problem of efficient approximation of a linear operator induced by the Gaussian or softmax kernel is often addressed using random features (RFs) which yield an unbiased approximation of the operator's result. Such operators emerge in important applications ranging from kernel methods to efficient Transformers. We propose parameterized, positive, non-trigonometric RFs which approximate Gaussian and softmax-kernels. In contrast to traditional RF approximations, parameters of these new methods can be optimized to reduce the variance of the approximation, and the optimum can be expressed in closed form. We show that our methods lead to variance reduction in practice (e^{10}-times smaller variance and beyond) and outperform previous methods in a kernel regression task. Using our proposed mechanism, we also present FAVOR#, a method for self-attention approximation in Transformers. We show that FAVOR# outperforms other random feature methods in speech modelling and natural language processing.

----

### Projection-Free Online Convex Optimization via Efficient Newton Iterations.

**Authors**: Khashayar Gatmiry, Zakaria Mhammedi

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03261886741f1f21f52f2a2d570616a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03261886741f1f21f52f2a2d570616a2-Abstract-Conference.html)

**Abstract**:

This paper presents new projection-free algorithms for Online Convex Optimization (OCO) over a convex domain $\mathcal{K} \subset \mathbb{R}^d$. Classical OCO algorithms (such as Online Gradient Descent) typically need to perform Euclidean projections onto the convex set $\mathcal{K}$ to ensure feasibility of their iterates. Alternative algorithms, such as those based on the Frank-Wolfe method, swap potentially-expensive Euclidean projections onto $\mathcal{K}$ for linear optimization over $\mathcal{K}$. However, such algorithms have a sub-optimal regret in OCO compared to projection-based algorithms. In this paper, we look at a third type of algorithms that output approximate Newton iterates using a self-concordant barrier for the set of interest. The use of a self-concordant barrier automatically ensures feasibility without the need of projections. However, the computation of the Newton iterates requires a matrix inverse, which can still be expensive. As our main contribution, we show how the stability of the Newton iterates can be leveraged to only compute the inverse Hessian a vanishing fractions of the rounds, leading to a new efficient projection-free OCO algorithm with a state-of-the-art regret bound.

----

### Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals.

**Authors**: Yue Wu, Yewen Fan, Paul Pu Liang, Amos Azaria, Yuanzhi Li, Tom M. Mitchell

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/034d7bfeace2a9a258648b16fc626298-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/034d7bfeace2a9a258648b16fc626298-Abstract-Conference.html)

**Abstract**:

High sample complexity has long been a challenge for RL. On the other hand, humans learn to perform tasks not only from interaction or demonstrations, but also by reading unstructured text documents, e.g., instruction manuals. Instruction manuals and wiki pages are among the most abundant data that could inform agents of valuable features and policies or task-specific environmental dynamics and reward structures. Therefore, we hypothesize that the ability to utilize human-written instruction manuals to assist learning policies for specific tasks should lead to a more efficient and better-performing agent. We propose the Read and Reward framework. Read and Reward speeds up RL algorithms on Atari games by reading manuals released by the Atari game developers. Our framework consists of a QA Extraction module that extracts and summarizes relevant information from the manual and a Reasoning module that evaluates object-agent interactions based on information from the manual. An auxiliary reward is then provided to a standard A2C RL agent, when interaction is detected. Experimentally, various RL algorithms obtain significant improvement in performance and training speed when assisted by our design. Code at github.com/Holmeswww/RnR

----

### Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization.

**Authors**: Kaiyue Wen, Zhiyuan Li, Tengyu Ma

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0354767c6386386be17cabe4fc59711b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0354767c6386386be17cabe4fc59711b-Abstract-Conference.html)

**Abstract**:

Despite extensive studies, the underlying reason as to why overparameterizedneural networks can generalize remains elusive. Existing theory shows that common stochastic optimizers prefer flatter minimizers of the training loss, and thusa natural potential explanation is that flatness implies generalization. This workcritically examines this explanation. Through theoretical and empirical investigation, we identify the following three scenarios for two-layer ReLU networks: (1)flatness provably implies generalization; (2) there exist non-generalizing flattestmodels and sharpness minimization algorithms fail to generalize poorly, and (3)perhaps most strikingly, there exist non-generalizing flattest models, but sharpnessminimization algorithms still generalize. Our results suggest that the relationshipbetween sharpness and generalization subtly depends on the data distributionsand the model architectures and sharpness minimization algorithms do not onlyminimize sharpness to achieve better generalization. This calls for the search forother explanations for the generalization of over-parameterized neural networks

----

### Feature-Learning Networks Are Consistent Across Widths At Realistic Scales.

**Authors**: Nikhil Vyas, Alexander Atanasov, Blake Bordelon, Depen Morwani, Sabarish Sainathan, Cengiz Pehlevan

**Conference**: nips 2023

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03600ae6c3392fd65ad7c3a90c6f7ce8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03600ae6c3392fd65ad7c3a90c6f7ce8-Abstract-Conference.html)

**Abstract**:

We study the effect of width on the dynamics of feature-learning neural networks across a variety of architectures and datasets. Early in training, wide neural networks trained on online data have not only identical loss curves but also agree in their point-wise test predictions throughout training. For simple tasks such as CIFAR-5m this holds throughout training for networks of realistic widths. We also show that structural properties of the models, including internal representations, preactivation distributions, edge of stability phenomena, and large learning rate effects are consistent across large widths. This motivates the hypothesis that phenomena seen in realistic models can be captured by infinite-width, feature-learning limits. For harder tasks (such as ImageNet and language modeling), and later training times, finite-width deviations grow systematically. Two distinct effects cause these deviations across widths. First, the network output has an initialization-dependent variance scaling inversely with width, which can be removed by ensembling networks. We observe, however, that ensembles of narrower networks perform worse than a single wide network. We call this the bias of narrower width. We conclude with a spectral perspective on the origin of this finite-width bias.

----

