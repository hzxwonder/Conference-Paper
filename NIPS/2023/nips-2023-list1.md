## [0] Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder

**Authors**: *Michael Bereket, Theofanis Karaletsos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0001ca33ba34ce0351e4612b744b3936-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0001ca33ba34ce0351e4612b744b3936-Abstract-Conference.html)

**Abstract**:

Generative models of observations under interventions have been a vibrant topic of interest across machine learning and the sciences in recent years. For example, in drug discovery, there is a need to model the effects of diverse interventions on cells in order to characterize unknown biological mechanisms of action. We propose the Sparse Additive Mechanism Shift Variational Autoencoder, SAMS-VAE, to combine compositionality, disentanglement, and interpretability for perturbation models. SAMS-VAE models the latent state of a perturbed sample as the sum of a local latent variable capturing sample-specific variation and sparse global variables of latent intervention effects. Crucially, SAMS-VAE sparsifies these global latent variables for individual perturbations to identify disentangled, perturbation-specific latent subspaces that are flexibly composable. We evaluate SAMS-VAE both quantitatively and qualitatively on a range of tasks using two popular single cell sequencing datasets.In order to measure perturbation-specific model-properties, we also introduce a framework for evaluation of perturbation models based on average treatment effects with links to posterior predictive checks. SAMS-VAE outperforms comparable models in terms of generalization across in-distribution and out-of-distribution tasks, including a combinatorial reasoning task under resource paucity, and yields interpretable latent structures which correlate strongly to known biological mechanisms. Our results suggest SAMS-VAE is an interesting addition to the modeling toolkit for machine learning-driven scientific discovery.

----

## [1] Cross-Episodic Curriculum for Transformer Agents

**Authors**: *Lucy Xiaoyang Shi, Yunfan Jiang, Jake Grigsby, Linxi Fan, Yuke Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/001608167bb652337af5df0129aeaabd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/001608167bb652337af5df0129aeaabd-Abstract-Conference.html)

**Abstract**:

We present a new algorithm, Cross-Episodic Curriculum (CEC), to boost the learning efficiency and generalization of Transformer agents. Central to CEC is the placement of cross-episodic experiences into a Transformerâ€™s context, which forms the basis of a curriculum. By sequentially structuring online learning trials and mixed-quality demonstrations, CEC constructs curricula that encapsulate learning progression and proficiency increase across episodes. Such synergy combined with the potent pattern recognition capabilities of Transformer models delivers a powerful cross-episodic attention mechanism. The effectiveness of CEC is demonstrated under two representative scenarios: one involving multi-task reinforcement learning with discrete control, such as in DeepMind Lab, where the curriculum captures the learning progression in both individual and progressively complex settings; and the other involving imitation learning with mixed-quality data for continuous control, as seen in RoboMimic, where the curriculum captures the improvement in demonstrators' expertise. In all instances, policies resulting from CEC exhibit superior performance and strong generalization. Code is open-sourced on the project website https://cec-agent.github.io/ to facilitate research on Transformer agent learning.

----

## [2] PaintSeg: Painting Pixels for Training-free Segmentation

**Authors**: *Xiang Li, Chung-Ching Lin, Yinpeng Chen, Zicheng Liu, Jinglu Wang, Rita Singh, Bhiksha Raj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0021c2cb1b9b6a71ac478ea52a93b25a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0021c2cb1b9b6a71ac478ea52a93b25a-Abstract-Conference.html)

**Abstract**:

The paper introduces PaintSeg, a new unsupervised method for segmenting objects without any training. We propose an adversarial masked contrastive painting (AMCP) process, which creates a contrast between the original image and a painted image in which a masked area is painted using off-the-shelf generative models. During the painting process, inpainting and outpainting are alternated, with the former masking the foreground and filling in the background, and the latter masking the background while recovering the missing part of the foreground object. Inpainting and outpainting, also referred to as I-step and O-step, allow our method to gradually advance the target segmentation mask toward the ground truth without supervision or training. PaintSeg can be configured to work with a variety of prompts, e.g. coarse masks, boxes, scribbles, and points. Our experimental results demonstrate that PaintSeg outperforms existing approaches in coarse mask-prompt, box-prompt, and point-prompt segmentation tasks, providing a training-free solution suitable for unsupervised segmentation. Code: https://github.com/lxa9867/PaintSeg.

----

## [3] Bootstrapping Vision-Language Learning with Decoupled Language Pre-training

**Authors**: *Yiren Jian, Chongyang Gao, Soroush Vosoughi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/002262941c9edfd472a79298b2ac5e17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/002262941c9edfd472a79298b2ac5e17-Abstract-Conference.html)

**Abstract**:

We present a novel methodology aimed at optimizing the application of frozen large language models (LLMs) for resource-intensive vision-language (VL) pre-training. The current paradigm uses visual features as prompts to guide language models, with a focus on determining the most relevant visual features for corresponding text. Our approach diverges by concentrating on the language component, specifically identifying the optimal prompts to align with visual features. We introduce the Prompt-Transformer (P-Former), a model that predicts these ideal prompts, which is trained exclusively on linguistic data, bypassing the need for image-text pairings. This strategy subtly bifurcates the end-to-end VL training process into an additional, separate stage. Our experiments reveal that our framework significantly enhances the performance of a robust image-to-text baseline (BLIP-2), and effectively narrows the performance gap between models trained with either 4M or 129M image-text pairs. Importantly, our framework is modality-agnostic and flexible in terms of architectural design, as validated by its successful application in a video learning task using varied base modules. The code will be made available at https://github.com/yiren-jian/BLIText.

----

## [4] Path following algorithms for 𝓁2-regularized M-estimation with approximation guarantee

**Authors**: *Yunzhang Zhu, Renxiong Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00296c0e10cd24d415c2db63ea2a2c68-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00296c0e10cd24d415c2db63ea2a2c68-Abstract-Conference.html)

**Abstract**:

Many modern machine learning algorithms are formulated as regularized M-estimation problems, in which a regularization (tuning) parameter controls a trade-off between model fit to the training data and model complexity. To select the ``best'' tuning parameter value that achieves a good trade-off, an approximated solution path needs to be computed. In practice, this is often done through selecting a grid of tuning parameter values and solving the regularized problem at the selected grid points. However, given any desired level of accuracy, it is often not clear how to choose the grid points and also how accurately one should solve the regularized problems at the selected gird points, both of which can greatly impact the overall amount of computation. In the context of  $\ell_2$-regularized $M$-estimation problem, we propose a novel grid point selection scheme and an adaptive stopping criterion for any given optimization algorithm that produces an approximated solution path with approximation error guarantee. Theoretically, we prove that the proposed solution path can approximate the exact solution path to arbitrary level of accuracy, while saving the overall computation as much as possible. Numerical results also corroborate with our theoretical analysis.

----

## [5] PDF: Point Diffusion Implicit Function for Large-scale Scene Neural Representation

**Authors**: *Yuhan Ding, Fukun Yin, Jiayuan Fan, Hui Li, Xin Chen, Wen Liu, Chongshan Lu, Gang Yu, Tao Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0073cc73e1873b35345209b50a3dab66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0073cc73e1873b35345209b50a3dab66-Abstract-Conference.html)

**Abstract**:

Recent advances in implicit neural representations have achieved impressive results by sampling and fusing individual points along sampling rays in the sampling space. However, due to the explosively growing sampling space, finely representing and synthesizing detailed textures remains a challenge for unbounded large-scale outdoor scenes. To alleviate the dilemma of using individual points to perceive the entire colossal space, we explore learning the surface distribution of the scene to provide structural priors and reduce the samplable space and propose a Point Diffusion implicit Function, PDF, for large-scale scene neural representation. The core of our method is a large-scale point cloud super-resolution diffusion module that enhances the sparse point cloud reconstructed from several training images into a dense point cloud as an explicit prior. Then in the rendering stage, only sampling points with prior points within the sampling radius are retained. That is, the sampling space is reduced from the unbounded space to the scene surface. Meanwhile, to fill in the background of the scene that cannot be provided by point clouds, the region sampling based on Mip-NeRF 360 is employed to model the background representation. Expensive experiments have demonstrated the effectiveness of our method for large-scale scene novel view synthesis, which outperforms relevant state-of-the-art baselines.

----

## [6] Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation

**Authors**: *Ruida Zhou, Tao Liu, Min Cheng, Dileep Kalathil, P. R. Kumar, Chao Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/007f4927e60699392425f267d43f0940-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/007f4927e60699392425f267d43f0940-Abstract-Conference.html)

**Abstract**:

We study robust reinforcement learning (RL) with the goal of determining a well-performing policy that is robust against model mismatch between the training simulator and the testing environment. Previous policy-based robust RL algorithms mainly focus on the tabular setting under uncertainty sets that facilitate robust policy evaluation, but are no longer tractable when the number of states scales up. To this end, we propose two novel uncertainty set formulations, one based on double sampling and the other on an integral probability metric. Both make large-scale robust RL tractable even when one only has access to a simulator. We propose a robust natural actor-critic (RNAC) approach that incorporates the new uncertainty sets and employs function approximation. We provide finite-time convergence guarantees for the proposed RNAC algorithm to the optimal robust policy within the function approximation error. Finally, we demonstrate the robust performance of the policy learned by our proposed RNAC approach in multiple  MuJoCo environments and a real-world TurtleBot navigation task.

----

## [7] Adaptive Selective Sampling for Online Prediction with Experts

**Authors**: *Rui M. Castro, Fredrik Hellström, Tim van Erven*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html)

**Abstract**:

We consider online prediction of a binary sequence with expert advice. For this setting, we devise label-efficient forecasting algorithms, which use a selective sampling scheme that enables collecting much fewer labels than standard procedures. For the general case without a perfect expert, we prove best-of-both-worlds guarantees, demonstrating that the proposed forecasting algorithm always queries sufficiently many labels in the worst case to obtain optimal regret guarantees, while simultaneously querying much fewer labels in more benign settings. Specifically, for a scenario where one expert is strictly better than the others in expectation, we show that the label complexity of the label-efficient forecaster is roughly upper-bounded by the square root of the number of rounds. Finally, we present numerical experiments empirically showing that the normalized regret of the label-efficient forecaster can asymptotically match known minimax rates for pool-based active learning, suggesting it can optimally adapt to benign settings.

----

## [8] Gigastep - One Billion Steps per Second Multi-agent Reinforcement Learning

**Authors**: *Mathias Lechner, Lianhao Yin, Tim Seyde, Tsun-Hsuan Johnson Wang, Wei Xiao, Ramin M. Hasani, Joshua Rountree, Daniela Rus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00ba06ba5c324efdfb068865ca44cf0b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/00ba06ba5c324efdfb068865ca44cf0b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Multi-agent reinforcement learning (MARL) research is faced with a trade-off: it either uses complex environments requiring large compute resources, which makes it inaccessible to researchers with limited resources, or relies on simpler dynamics for faster execution, which makes the transferability of the results to more realistic tasks challenging. Motivated by these challenges, we present Gigastep, a fully vectorizable, MARL environment implemented in JAX, capable of executing up to one billion environment steps per second on consumer-grade hardware. Its design allows for comprehensive MARL experimentation, including a complex, high-dimensional space defined by 3D dynamics, stochasticity, and partial observations. Gigastep supports both collaborative and adversarial tasks, continuous and discrete action spaces, and provides RGB image and feature vector observations, allowing the evaluation of a wide range of MARL algorithms. We validate Gigastep's usability through an extensive set of experiments, underscoring its role in widening participation and promoting inclusivity in the MARL research community.

----

## [9] Attentive Transfer Entropy to Exploit Transient Emergence of Coupling Effect

**Authors**: *Xiaolei Ru, Xinya Zhang, Zijia Liu, Jack Murdoch Moore, Gang Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html)

**Abstract**:

We consider the problem of reconstructing coupled networks (e.g., biological neural networks) connecting large numbers of variables (e.g.,nerve cells), of which state evolution is governed by dissipative dynamics consisting of strong self-drive (dominants the evolution) and weak coupling-drive. The core difficulty is sparseness of coupling effect that emerges (the coupling force is significant) only momentarily and otherwise remains quiescent in time series (e.g., neuronal activity sequence). Here we learn the idea from attention mechanism to guide the classifier to make inference focusing on the critical regions of time series data where coupling effect may manifest. Specifically, attention coefficients are assigned autonomously by artificial neural networks trained to maximise the Attentive Transfer Entropy (ATEn), which is a novel generalization of the iconic transfer entropy metric. Our results show that, without any prior knowledge of dynamics, ATEn explicitly identifies areas where the strength of coupling-drive is distinctly greater than zero. This innovation substantially improves reconstruction performance for both synthetic and real directed coupling networks using data generated by neuronal models widely used in neuroscience.

----

## [10] PopSign ASL v10: An Isolated American Sign Language Dataset Collected via Smartphones

**Authors**: *Thad Starner, Sean Forbes, Matthew So, David Martin, Rohit Sridhar, Gururaj Deshpande, Sam S. Sepah, Sahir Shahryar, Khushi Bhardwaj, Tyler Kwok, Daksh Sehgal, Saad Hassan, Bill Neubauer, Sofia Anandi Vempala, Alec Tan, Jocelyn Heath, Unnathi Kumar, Priyanka Mosur, Tavenner Hall, Rajandeep Singh, Christopher Cui, Glenn Cameron, Sohier Dane, Garrett Tanzer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

PopSign is a smartphone-based bubble-shooter game that helps hearing parentsof deaf infants learn sign language. To help parents practice their ability to sign,PopSign is integrating sign language recognition as part of its gameplay. Fortraining the recognizer, we introduce the PopSign ASL v1.0 dataset that collectsexamples of 250 isolated American Sign Language (ASL) signs using Pixel 4Asmartphone selfie cameras in a variety of environments. It is the largest publiclyavailable, isolated sign dataset by number of examples and is the first dataset tofocus on one-handed, smartphone signs. We collected over 210,000 examplesat 1944x2592 resolution made by 47 consenting Deaf adult signers for whomAmerican Sign Language is their primary language. We manually reviewed 217,866of these examples, of which 175,023 (approximately 700 per sign) were the signintended for the educational game. 39,304 examples were recognizable as a signbut were not the desired variant or were a different sign. We provide a training setof 31 signers, a validation set of eight signers, and a test set of eight signers. Abaseline LSTM model for the 250-sign vocabulary achieves 82.1% accuracy (81.9%class-weighted F1 score) on the validation set and 84.2% (83.9% class-weightedF1 score) on the test set. Gameplay suggests that accuracy will be sufficient forcreating educational games involving sign language recognition.

----

## [11] (Provable) Adversarial Robustness for Group Equivariant Tasks: Graphs, Point Clouds, Molecules, and More

**Authors**: *Jan Schuchardt, Yan Scholten, Stephan Günnemann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00db17c36b5435195760520efa96d99c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00db17c36b5435195760520efa96d99c-Abstract-Conference.html)

**Abstract**:

A machine learning model is traditionally considered robust if its prediction remains (almost) constant under input perturbations with small norm. However, real-world tasks like molecular property prediction or point cloud segmentation have inherent equivariances, such as rotation or permutation equivariance. In such tasks, even perturbations with large norm do not necessarily change an input's semantic content. Furthermore, there are perturbations for which a model's prediction explicitly needs to change. For the first time, we propose a sound notion of adversarial robustness that accounts for task equivariance. We then demonstrate that provable robustness can be achieved by (1) choosing a model that matches the task's equivariances (2) certifying traditional adversarial robustness. Certification methods are, however, unavailable for many models, such as those with continuous equivariances. We close this gap by developing the framework of equivariance-preserving randomized smoothing, which enables architecture-agnostic certification. We additionally derive the first architecture-specific graph edit distance certificates, i.e. sound robustness guarantees for isomorphism equivariant tasks like node classification. Overall, a sound notion of robustness is an important prerequisite for future work at the intersection of robust and geometric machine learning.

----

## [12] Self-Supervised Motion Magnification by Backpropagating Through Optical Flow

**Authors**: *Zhaoying Pan, Daniel Geng, Andrew Owens*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/00ed9ab006311be67879ecef8f80d7c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/00ed9ab006311be67879ecef8f80d7c5-Abstract-Conference.html)

**Abstract**:

This paper presents a simple, self-supervised method for magnifying subtle motions in video: given an input video and a magnification factor, we manipulate the video such that its new optical flow is scaled by the desired amount. To train our model, we propose a loss function that estimates the optical flow of the generated video and penalizes how far if deviates from the given magnification factor. Thus, training involves differentiating through a pretrained optical flow network. Since our model is self-supervised, we can further improve its performance through test-time adaptation, by finetuning it on the input video. It can also be easily extended to magnify the motions of only user-selected objects. Our approach avoids the need for synthetic magnification datasets that have been used to train prior learning-based approaches. Instead, it leverages the existing capabilities of off-the-shelf motion estimators. We demonstrate the effectiveness of our method through evaluations of both visual quality and quantitative metrics on a range of real-world and synthetic videos, and we show our method works for both supervised and unsupervised optical flow methods.

----

## [13] TexQ: Zero-shot Network Quantization with Texture Feature Distribution Calibration

**Authors**: *Xinrui Chen, Yizhi Wang, Renao Yan, Yiqing Liu, Tian Guan, Yonghong He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html)

**Abstract**:

Quantization is an effective way to compress neural networks. By reducing the bit width of the parameters, the processing efficiency of neural network models at edge devices can be notably improved. Most conventional quantization methods utilize real datasets to optimize quantization parameters and fine-tune. Due to the inevitable privacy and security issues of real samples, the existing real-data-driven methods are no longer applicable. Thus, a natural method is to introduce synthetic samples for zero-shot quantization (ZSQ). However, the conventional synthetic samples fail to retain the detailed texture feature distributions, which severely limits the knowledge transfer and performance of the quantized model. In this paper, a novel ZSQ method, TexQ is proposed to address this issue. We first synthesize a calibration image and extract its calibration center for each class with a texture feature energy distribution calibration method. Then, the calibration centers are used to guide the generator to synthesize samples. Finally, we introduce the mixup knowledge distillation module to diversify synthetic samples for fine-tuning. Extensive experiments on CIFAR10/100 and ImageNet show that TexQ is observed to perform state-of-the-art in ultra-low bit width quantization. For example, when ResNet-18 is quantized to 3-bit, TexQ achieves a 12.18% top-1 accuracy increase on ImageNet compared to state-of-the-art methods. Code at https://github.com/dangsingrue/TexQ.

----

## [14] Ambient Diffusion: Learning Clean Distributions from Corrupted Data

**Authors**: *Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alex Dimakis, Adam R. Klivans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/012af729c5d14d279581fc8a5db975a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/012af729c5d14d279581fc8a5db975a1-Abstract-Conference.html)

**Abstract**:

We present the first diffusion-based framework that can learn an unknown distribution using only highly-corrupted samples. This problem arises in scientific applications where access to uncorrupted samples is impossible or expensive to acquire. Another benefit of our approach is the ability to train generative models that are less likely to memorize any individual training sample, since they never observe clean training data. Our main idea is to introduce additional measurement distortion during the diffusion process and require the model to predict the original corrupted image from the further corrupted image.  We prove that our method leads to models that learn the conditional expectation of the full uncorrupted image given this additional measurement corruption.  This holds for any corruption process that satisfies some technical conditions (and in particular includes inpainting and compressed sensing).  We train models on standard benchmarks (CelebA, CIFAR-10 and AFHQ) and show that we can learn the distribution even when all the training samples have 90\% of their pixels missing. We also show that we can finetune foundation models on small corrupted datasets (e.g. MRI scans with block corruptions) and learn the clean distribution without memorizing the training set.

----

## [15] Scalable Membership Inference Attacks via Quantile Regression

**Authors**: *Martin Bertran, Shuai Tang, Aaron Roth, Michael Kearns, Jamie Morgenstern, Steven Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01328d0767830e73a612f9073e9ff15f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01328d0767830e73a612f9073e9ff15f-Abstract-Conference.html)

**Abstract**:

Membership inference attacks are designed to determine, using black box access to trained models, whether a particular example was used in training or not. Membership inference can be formalized as a hypothesis testing problem. The most effective existing attacks estimate the distribution of some test statistic (usually the model's confidence on the true label) on points that were (and were not) used in training by training many \emph{shadow models}---i.e. models of the same architecture as the model being attacked, trained on a random subsample of data. While effective, these attacks are extremely computationally expensive, especially when the model under attack is large. \footnotetext[0]{Martin and Shuai are the lead authors, and other authors are ordered alphabetically. {maberlop,shuat}@amazon.com}We introduce a new class of attacks based on performing quantile regression on the distribution of confidence scores induced by the model under attack on points that are not used in training. We show that our method is competitive with state-of-the-art shadow model attacks, while requiring substantially less compute because our attack requires training only a single model. Moreover, unlike shadow model attacks, our proposed attack does not require any knowledge of the architecture of the model under attack and is therefore truly ``black-box". We show the efficacy of this approach in an extensive series of experiments on various datasets and model architectures. Our code is available at \href{https://github.com/amazon-science/quantile-mia}{github.com/amazon-science/quantile-mia.}

----

## [16] ESSEN: Improving Evolution State Estimation for Temporal Networks using Von Neumann Entropy

**Authors**: *Qiyao Huang, Yingyue Zhang, Zhihong Zhang, Edwin R. Hancock*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0147d967a5db3b8dde08d2a327b24568-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0147d967a5db3b8dde08d2a327b24568-Abstract-Conference.html)

**Abstract**:

Temporal networks are widely used as abstract graph representations for real-world dynamic systems. Indeed, recognizing the network evolution states is crucial in understanding and analyzing temporal networks. For instance, social networks will generate the clustering and formation of tightly-knit groups or communities over time, relying on the triadic closure theory. However, the existing methods often struggle to account for the time-varying nature of these network structures, hindering their performance when applied to networks with complex evolution states. To mitigate this problem, we propose a novel framework called ESSEN, an Evolution StateS awarE Network, to measure temporal network evolution using von Neumann entropy and thermodynamic temperature. The developed framework utilizes a von Neumann entropy aware attention mechanism and network evolution state contrastive learning in the graph encoding. In addition, it employs a unique decoder the so-called Mixture of Thermodynamic Experts (MoTE) for decoding. ESSEN extracts local and global network evolution information using thermodynamic features and adaptively recognizes the network evolution states. Moreover, the proposed method is evaluated on link prediction tasks under both transductive and inductive settings, with the corresponding results demonstrating its effectiveness compared to various state-of-the-art baselines.

----

## [17] Label Correction of Crowdsourced Noisy Annotations with an Instance-Dependent Noise Transition Model

**Authors**: *Hui Guo, Boyu Wang, Grace Yi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/015a8c69bedcb0a7b2ed2e1678f34399-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/015a8c69bedcb0a7b2ed2e1678f34399-Abstract-Conference.html)

**Abstract**:

The predictive ability of supervised learning algorithms hinges on the quality of annotated examples, whose labels often come from multiple crowdsourced annotators with diverse expertise. To aggregate noisy crowdsourced annotations, many existing methods employ an annotator-specific instance-independent noise transition matrix  to characterize the labeling skills of each annotator. Learning an instance-dependent noise transition model, however, is challenging and remains relatively less explored. To address this problem, in this paper, we formulate the noise transition model in a Bayesian framework and subsequently design a new label correction algorithm. Specifically, we approximate the instance-dependent noise transition matrices using a Bayesian network with a hierarchical spike and slab prior. To theoretically characterize the distance between the noise transition model and the true instance-dependent noise transition matrix, we provide a posterior-concentration theorem that ensures the posterior consistency in terms of the Hellinger distance. We further formulate the label correction process as a hypothesis testing problem and propose a novel algorithm to infer the true label from the noisy annotations based on the pairwise likelihood ratio test. Moreover, we establish an information-theoretic bound on the Bayes error for the proposed method. We validate the effectiveness of our approach through experiments on benchmark and real-world datasets.

----

## [18] Diffused Task-Agnostic Milestone Planner

**Authors**: *Mineui Hong, Minjae Kang, Songhwai Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0163ca1c69f848e766cfb0b7bb7e17f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0163ca1c69f848e766cfb0b7bb7e17f4-Abstract-Conference.html)

**Abstract**:

Addressing decision-making problems using sequence modeling to predict future trajectories shows promising results in recent years.In this paper, we take a step further to leverage the sequence predictive method in wider areas such as long-term planning, vision-based control, and multi-task decision-making.To this end, we propose a method to utilize a diffusion-based generative sequence model to plan a series of milestones in a latent space and to have an agent to follow the milestones to accomplish a given task.The proposed method can learn control-relevant, low-dimensional latent representations of milestones, which makes it possible to efficiently perform long-term planning and vision-based control.Furthermore, our approach exploits generation flexibility of the diffusion model, which makes it possible to plan diverse trajectories for multi-task decision-making.We demonstrate the proposed method across offline reinforcement learning (RL) benchmarks and an visual manipulation environment.The results show that our approach outperforms offline RL methods in solving long-horizon, sparse-reward tasks and multi-task problems,while also achieving the state-of-the-art performance on the most challenging vision-based manipulation benchmark.

----

## [19] Task-aware Distributed Source Coding under Dynamic Bandwidth

**Authors**: *Po-han Li, Sravan Kumar Ankireddy, Ruihan Philip Zhao, Hossein Nourkhiz Mahjoub, Ehsan Moradi-Pari, Ufuk Topcu, Sandeep Chinchali, Hyeji Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/016c63403370d81c24c1ca0123de6cfa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/016c63403370d81c24c1ca0123de6cfa-Abstract-Conference.html)

**Abstract**:

Efficient compression of correlated data is essential to minimize communication overload in multi-sensor networks. In such networks, each sensor independently compresses the data and transmits them to a central node. A decoder at the central node decompresses and passes the data to a pre-trained machine learning-based task model to generate the final output. Due to limited communication bandwidth, it is important for the compressor to learn only the features that are relevant to the task. Additionally, the final performance depends heavily on the total available bandwidth. In practice, it is common to encounter varying availability in bandwidth. Since higher bandwidth results in better performance, it is essential for the compressor to dynamically take advantage of the maximum available bandwidth at any instant. In this work, we propose a novel distributed compression framework composed of independent encoders and a joint decoder, which we call neural distributed principal component analysis (NDPCA). NDPCA flexibly compresses data from multiple sources to any available bandwidth with a single model, reducing compute and storage overhead. NDPCA achieves this by learning low-rank task representations and efficiently distributing bandwidth among sensors, thus providing a graceful trade-off between performance and bandwidth. Experiments show that NDPCA improves the success rate of multi-view robotic arm manipulation by 9% and the accuracy of object detection tasks on satellite imagery by 14% compared to an autoencoder with uniform bandwidth allocation.

----

## [20] BubbleML: A Multiphase Multiphysics Dataset and Benchmarks for Machine Learning

**Authors**: *Sheikh Md Shakeel Hassan, Arthur Feeney, Akash Dhruv, Jihoon Kim, Youngjoon Suh, Jaiyoung Ryu, Yoonjin Won, Aparna Chandramowlishwaran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01726ae05d72ddba3ac784a5944fa1ef-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/01726ae05d72ddba3ac784a5944fa1ef-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In the field of phase change phenomena, the lack of accessible and diverse datasets suitable for machine learning (ML) training poses a significant challenge. Existing experimental datasets are often restricted, with limited availability and sparse ground truth, impeding our understanding of this complex multiphysics phenomena. To bridge this gap, we present the BubbleML dataset which leverages physics-driven simulations to provide accurate ground truth information for various boiling scenarios, encompassing nucleate pool boiling, flow boiling, and sub-cooled boiling. This extensive dataset covers a wide range of parameters, including varying gravity conditions, flow rates, sub-cooling levels, and wall superheat, comprising 79 simulations.  BubbleML is validated against experimental observations and trends, establishing it as an invaluable resource for ML research. Furthermore, we showcase its potential to facilitate the exploration of diverse downstream tasks by introducing two benchmarks: (a) optical flow analysis to capture bubble dynamics, and (b) neural PDE solvers for learning temperature and flow dynamics. The BubbleML dataset and its benchmarks aim to catalyze progress in ML-driven research on multiphysics phase change phenomena, providing robust baselines for the development and comparison of state-of-the-art techniques and models.

----

## [21] ANTN: Bridging Autoregressive Neural Networks and Tensor Networks for Quantum Many-Body Simulation

**Authors**: *Zhuo Chen, Laker Newhouse, Eddie Chen, Di Luo, Marin Soljacic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01772a8b0420baec00c4d59fe2fbace6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01772a8b0420baec00c4d59fe2fbace6-Abstract-Conference.html)

**Abstract**:

Quantum many-body physics simulation has important impacts on understanding fundamental science and has applications to quantum materials design and quantum technology. However, due to the exponentially growing size of the Hilbert space with respect to the particle number, a direct simulation is intractable. While representing quantum states with tensor networks and neural networks are the two state-of-the-art methods for approximate simulations, each has its own limitations in terms of expressivity and inductive bias. To address these challenges, we develop a novel architecture, Autoregressive Neural TensorNet (ANTN), which bridges tensor networks and autoregressive neural networks. We show that Autoregressive Neural TensorNet parameterizes normalized wavefunctions, allows for exact sampling, generalizes the expressivity of tensor networks and autoregressive neural networks, and inherits a variety of symmetries from autoregressive neural networks. We demonstrate our approach on quantum state learning as well as finding the ground state of the challenging 2D $J_1$-$J_2$ Heisenberg model with different systems sizes and coupling parameters, outperforming both tensor networks and autoregressive neural networks. Our work opens up new opportunities for quantum many-body physics simulation, quantum technology design, and generative modeling in artificial intelligence.

----

## [22] Causal Effect Identification in Uncertain Causal Networks

**Authors**: *Sina Akbari, Fateme Jamshidi, Ehsan Mokhtarian, Matthew J. Vowels, Jalal Etesami, Negar Kiyavash*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/017c897b4d85a744f345ccbf9d71e501-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/017c897b4d85a744f345ccbf9d71e501-Abstract-Conference.html)

**Abstract**:

Causal identification is at the core of the causal inference literature, where complete algorithms have been proposed to identify causal queries of interest. The validity of these algorithms hinges on the restrictive assumption of having access to a correctly specified causal structure. In this work, we study the setting where a probabilistic model of the causal structure is available. Specifically, the edges in a causal graph exist with uncertainties which may, for example, represent degree of belief from domain experts. Alternatively, the uncertainty about an edge may reflect the confidence of a particular statistical test. The question that naturally arises in this setting is: Given such a probabilistic graph and a specific causal effect of interest, what is the subgraph which has the highest plausibility and for which the causal effect is identifiable? We show that answering this question reduces to solving an NP-hard combinatorial optimization problem which we call the edge ID problem. We propose efficient algorithms to approximate this problem and evaluate them against both real-world networks and randomly generated graphs.

----

## [23] FAST: a Fused and Accurate Shrinkage Tree for Heterogeneous Treatment Effects Estimation

**Authors**: *Jia Gu, Caizhi Tang, Han Yan, Qing Cui, Longfei Li, Jun Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01830c92c6558179fa6d7fb1edff692c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01830c92c6558179fa6d7fb1edff692c-Abstract-Conference.html)

**Abstract**:

This paper proposes a novel strategy for estimating the heterogeneous treatment effect  called the Fused and Accurate Shrinkage Tree ($\mathrm{FAST}$). Our approach utilizes both trial and observational data to improve the accuracy and robustness of the estimator. Inspired by the concept of shrinkage estimation in statistics, we develop an optimal weighting scheme and a corresponding estimator that balances the unbiased estimator based on the trial data with the potentially biased estimator based on the observational data. Specifically, combined with tree-based techniques, we introduce a new split criterion that utilizes both trial data and observational data to more accurately estimate the treatment effect. Furthermore, we confirm the consistency of our proposed tree-based estimator and demonstrate the effectiveness of our criterion in reducing prediction error through theoretical analysis.  The advantageous  finite sample performance of the $\mathrm{FAST}$ and its ensemble version over existing methods is demonstrated via  simulations and real data analysis.

----

## [24] Characterizing Graph Datasets for Node Classification: Homophily-Heterophily Dichotomy and Beyond

**Authors**: *Oleg Platonov, Denis Kuznedelev, Artem Babenko, Liudmila Prokhorenkova*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01b681025fdbda8e935a66cc5bb6e9de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01b681025fdbda8e935a66cc5bb6e9de-Abstract-Conference.html)

**Abstract**:

Homophily is a graph property describing the tendency of edges to connect similar nodes; the opposite is called heterophily. It is often believed that heterophilous graphs are challenging for standard message-passing graph neural networks (GNNs), and much effort has been put into developing efficient methods for this setting. However, there is no universally agreed-upon measure of homophily in the literature. In this work, we show that commonly used homophily measures have critical drawbacks preventing the comparison of homophily levels across different datasets. For this, we formalize desirable properties for a proper homophily measure and verify which measures satisfy which properties. In particular, we show that a measure that we call adjusted homophily satisfies more desirable properties than other popular homophily measures while being rarely used in graph machine learning literature. Then, we go beyond the homophily-heterophily dichotomy and propose a new characteristic that allows one to further distinguish different sorts of heterophily. The proposed label informativeness (LI) characterizes how much information a neighbor's label provides about a node's label. We prove that this measure satisfies important desirable properties. We also observe empirically that LI better agrees with GNN performance compared to homophily measures, which confirms that it is a useful characteristic of the graph structure.

----

## [25] Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation

**Authors**: *Yuxuan Song, Jingjing Gong, Minkai Xu, Ziyao Cao, Yanyan Lan, Stefano Ermon, Hao Zhou, Wei-Ying Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01d64478381c33e29ed611f1719f5a37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01d64478381c33e29ed611f1719f5a37-Abstract-Conference.html)

**Abstract**:

The generation of 3D molecules requires simultaneously deciding the categorical features (atom types) and continuous features (atom coordinates). Deep generative models, especially Diffusion Models (DMs), have demonstrated effectiveness in generating feature-rich geometries. However, existing DMs typically suffer from unstable probability dynamics with inefficient sampling speed. In this paper, we introduce geometric flow matching, which enjoys the advantages of both equivariant modeling and stabilized probability dynamics. More specifically, we propose a hybrid probability path where the coordinates probability path is regularized by an equivariant optimal transport, and the information between different modalities is aligned. Experimentally, the proposed method could consistently achieve better performance on multiple molecule generation benchmarks with 4.75$\times$ speed up of sampling on average.

----

## [26] Hyperbolic VAE via Latent Gaussian Distributions

**Authors**: *Seunghyuk Cho, Juyong Lee, Dongwoo Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html)

**Abstract**:

We propose a Gaussian manifold variational auto-encoder (GM-VAE) whose latent space consists of a set of Gaussian distributions. It is known that the set of the univariate Gaussian distributions with the Fisher information metric form a hyperbolic space, which we call a Gaussian manifold. To learn the VAE endowed with the Gaussian manifolds, we propose a pseudo-Gaussian manifold normal distribution based on the Kullback-Leibler divergence, a local approximation of the squared Fisher-Rao distance, to define a density over the latent space. We demonstrate the efficacy of GM-VAE on two different tasks: density estimation of image datasets and state representation learning for model-based reinforcement learning. GM-VAE outperforms the other variants of hyperbolic- and Euclidean-VAEs on density estimation tasks and shows competitive performance in model-based reinforcement learning. We observe that our model provides strong numerical stability, addressing a common limitation reported in previous hyperbolic-VAEs. The implementation is available at https://github.com/ml-postech/GM-VAE.

----

## [27] A Simple Solution for Offline Imitation from Observations and Examples with Possibly Incomplete Trajectories

**Authors**: *Kai Yan, Alexander G. Schwing, Yu-Xiong Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0203f489345567b4a048c38f507cdbfa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0203f489345567b4a048c38f507cdbfa-Abstract-Conference.html)

**Abstract**:

Offline imitation from observations aims to solve MDPs where only task-specific expert states and task-agnostic non-expert state-action pairs are available. Offline imitation is useful in real-world scenarios where arbitrary interactions are costly and expert actions are unavailable. The state-of-the-art ‘DIstribution Correction Estimation’ (DICE) methods minimize divergence of state occupancy between expert and learner policies and retrieve a policy with weighted behavior cloning; however, their results are unstable when learning from incomplete trajectories, due to a non-robust optimization in the dual domain. To address the issue, in this paper, we propose Trajectory-Aware Imitation Learning from Observations (TAILO). TAILO uses a discounted sum along the future trajectory as the weight for weighted behavior cloning. The terms for the sum are scaled by the output of a discriminator, which aims to identify expert states. Despite simplicity, TAILO works well if there exist trajectories or segments of expert behavior in the task-agnostic data, a common assumption in prior work. In experiments across multiple testbeds, we find TAILO to be more robust and effective, particularly with incomplete trajectories.

----

## [28] Defending against Data-Free Model Extraction by Distributionally Robust Defensive Training

**Authors**: *Zhenyi Wang, Li Shen, Tongliang Liu, Tiehang Duan, Yanjun Zhu, Donglin Zhan, David S. Doermann, Mingchen Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0207c9ea9faf66c6e892c3fa3c167b75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0207c9ea9faf66c6e892c3fa3c167b75-Abstract-Conference.html)

**Abstract**:

Data-Free Model Extraction (DFME) aims to clone a black-box model without knowing its original training data distribution, making it much easier for attackers to steal commercial models. Defense against DFME faces several challenges: (i) effectiveness; (ii) efficiency; (iii) no prior on the attacker's query data distribution and strategy. However, existing defense methods: (1) are highly computation and memory inefficient; or (2) need strong assumptions about attack data distribution; or (3) can only delay the attack or prove a model theft after the model stealing has happened. In this work, we propose a Memory and Computation efficient defense approach, named MeCo, to prevent DFME from happening while maintaining the model utility simultaneously by distributionally robust defensive training on the target victim model. Specifically, we randomize the input so that it: (1) causes a mismatch of the knowledge distillation loss for attackers; (2) disturbs the zeroth-order gradient estimation; (3) changes the label prediction for the attack query data. Therefore, the attacker can only extract misleading information from the black-box model. Extensive experiments on defending against both decision-based and score-based DFME demonstrate that MeCo can significantly reduce the effectiveness of existing DFME methods and substantially improve running efficiency.

----

## [29] Large language models transition from integrating across position-yoked, exponential windows to structure-yoked, power-law windows

**Authors**: *David Skrill, Samuel Norman-Haignere*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/020ad0ac6a1974e6748e4a5a48110a07-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/020ad0ac6a1974e6748e4a5a48110a07-Abstract-Conference.html)

**Abstract**:

Modern language models excel at integrating across long temporal scales needed to encode linguistic meaning and show non-trivial similarities to biological neural systems. Prior work suggests that human brain responses to language exhibit hierarchically organized "integration windows" that substantially constrain the overall influence of an input token (e.g., a word) on the neural response. However, little prior work has attempted to use integration windows to characterize computations in large language models (LLMs). We developed a simple word-swap procedure for estimating integration windows from black-box language models that does not depend on access to gradients or knowledge of the model architecture (e.g., attention weights). Using this method, we show that trained LLMs exhibit stereotyped integration windows that are well-fit by a convex combination of an exponential and a power-law function, with a partial transition from exponential to power-law dynamics across network layers. We then introduce a metric for quantifying the extent to which these integration windows vary with structural boundaries (e.g., sentence boundaries), and using this metric, we show that integration windows become increasingly yoked to structure at later network layers. None of these findings were observed in an untrained model, which as expected integrated uniformly across its input. These results suggest that LLMs learn to integrate information in natural language using a stereotyped pattern: integrating across position-yoked, exponential windows at early layers, followed by structure-yoked, power-law windows at later layers. The methods we describe in this paper provide a general-purpose toolkit for understanding temporal integration in language models, facilitating cross-disciplinary research at the intersection of biological and artificial intelligence.

----

## [30] Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?

**Authors**: *Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Yecheng Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, Franziska Meier*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/022ca1bed6b574b962c48a2856eb207b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/022ca1bed6b574b962c48a2856eb207b-Abstract-Conference.html)

**Abstract**:

We present the largest and most comprehensive empirical study of pre-trained visual representations (PVRs) or visual ‘foundation models’ for Embodied AI. First, we curate CortexBench, consisting of 17 different tasks spanning locomotion, navigation, dexterous, and mobile manipulation. Next, we systematically evaluate existing PVRs and find that none are universally dominant. To study the effect of pre-training data size and diversity, we combine over 4,000 hours of egocentric videos from 7 different sources (over 4.3M images) and ImageNet to train different-sized vision transformers using Masked Auto-Encoding (MAE) on slices of this data. Contrary to inferences from prior work, we find that scaling dataset size and diversity does not improve performance universally (but does so on average). Our largest model, named VC-1, outperforms all prior PVRs on average but does not universally dominate either. Next, we show that task- or domain-specific adaptation of VC-1 leads to substantial gains, with VC-1 (adapted) achieving competitive or superior performance than the best known results on all of the benchmarks in CortexBench. Finally, we present real-world hardware experiments, in which VC-1 and VC-1 (adapted) outperform the strongest pre-existing PVR. Overall, this paper presents no new techniques but a rigorous systematic evaluation, a broad set of findings about PVRs (that in some cases, refute those made in narrow domains in prior work), and open-sourced code and models (that required over 10,000 GPU-hours to train) for the benefit of the research community.

----

## [31] Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback

**Authors**: *Jangwon Kim, Hangyeol Kim, Jiwook Kang, Jongchan Baek, Soohee Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0252a434b18962c94910c07cd9a7fecc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0252a434b18962c94910c07cd9a7fecc-Abstract-Conference.html)

**Abstract**:

We present a novel actor-critic algorithm for an environment with delayed feedback, which addresses the state-space explosion problem of conventional approaches. Conventional approaches use an augmented state constructed from the last observed state and actions executed since visiting the last observed state. Using the augmented state space, the correct Markov decision process for delayed environments can be constructed; however, this causes the state space to explode as the number of delayed timesteps increases, leading to slow convergence. Our proposed algorithm, called Belief-Projection-Based Q-learning (BPQL), addresses the state-space explosion problem by evaluating the values of the critic for which the input state size is equal to the original state-space size rather than that of the augmented one. We compare BPQL to traditional approaches in continuous control tasks and demonstrate that it significantly outperforms other algorithms in terms of asymptotic performance and sample efficiency. We also show that BPQL solves long-delayed environments, which conventional approaches are unable to do.

----

## [32] Batchnorm Allows Unsupervised Radial Attacks

**Authors**: *Amur Ghose, Apurv Gupta, Yaoliang Yu, Pascal Poupart*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0266d95023740481d22d437aa8aba0e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0266d95023740481d22d437aa8aba0e9-Abstract-Conference.html)

**Abstract**:

The construction of adversarial examples usually requires the existence of soft or hard labels for each instance, with respect to which a loss gradient provides the signal for construction of the example. We show that for batch normalized deep image recognition architectures, intermediate latents that are produced after a batch normalization step by themselves suffice to produce adversarial examples using an intermediate loss solely utilizing angular deviations, without relying on any label. We motivate our loss through the geometry of batch normed representations and their concentration of norm on a hypersphere and distributional proximity to Gaussians. Our losses expand intermediate latent based attacks that usually require labels. The success of our method implies that leakage of intermediate representations may create a security breach for deployed models, which persists even when the model is transferred to downstream usage. Removal of batch norm weakens our attack, indicating it contributes to this vulnerability. Our attacks also succeed against LayerNorm empirically, thus being relevant for transformer architectures, most notably vision transformers which we analyze.

----

## [33] Detecting Any Human-Object Interaction Relationship: Universal HOI Detector with Spatial Prompt Learning on Foundation Models

**Authors**: *Yichao Cao, Qingfei Tang, Xiu Su, Song Chen, Shan You, Xiaobo Lu, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html)

**Abstract**:

Human-object interaction (HOI) detection aims to comprehend the intricate relationships between humans and objects, predicting  triplets, and serving as the foundation for numerous computer vision tasks. The complexity and diversity of human-object interactions in the real world, however, pose significant challenges for both annotation and recognition, particularly in recognizing interactions within an open world context. This study explores the universal interaction recognition in an open-world setting through the use of Vision-Language (VL) foundation models and large language models (LLMs). The proposed method is dubbed as UniHOI. We conduct a deep analysis of the three hierarchical features inherent in visual HOI detectors and propose a method for high-level relation extraction aimed at VL foundation models, which we call HO prompt-based learning. Our design includes an HO Prompt-guided Decoder (HOPD), facilitates the association of high-level relation representations in the foundation model with various HO pairs within the image. Furthermore, we utilize a LLM (i.e. GPT) for interaction interpretation, generating a richer linguistic understanding for complex HOIs. For open-category interaction recognition, our method supports either of two input types: interaction phrase or interpretive sentence.  Our efficient architecture design and learning methods effectively unleash the potential of the VL foundation models and LLMs, allowing UniHOI to surpass all existing methods with a substantial margin, under both supervised and zero-shot settings. The code and pre-trained weights will be made publicly available.

----

## [34] Smoothing the Landscape Boosts the Signal for SGD: Optimal Sample Complexity for Learning Single Index Models

**Authors**: *Alex Damian, Eshaan Nichani, Rong Ge, Jason D. Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02763667a5761ff92bb15d8751bcd223-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02763667a5761ff92bb15d8751bcd223-Abstract-Conference.html)

**Abstract**:

We focus on the task of learning a single index model $\sigma(w^\star \cdot x)$ with respect to the isotropic Gaussian distribution in $d$ dimensions. Prior work has shown that the sample complexity of learning $w^\star$ is governed by the information exponent $k^\star$ of the link function $\sigma$, which is defined as the index of the first nonzero Hermite coefficient of $\sigma$. Ben Arous et al. (2021) showed that $n \gtrsim d^{k^\star-1}$ samples suffice for learning $w^\star$ and that this is tight for online SGD. However, the CSQ lower bound for gradient based methods only shows that $n \gtrsim d^{k^\star/2}$ samples are necessary. In this work, we close the gap between the upper and lower bounds by showing that online SGD on a smoothed loss learns $w^\star$ with $n \gtrsim d^{k^\star/2}$ samples. We also draw connections to statistical analyses of tensor PCA and to the implicit regularization effects of minibatch SGD on empirical losses.

----

## [35] A Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models

**Authors**: *Alexander G. Reisach, Myriam Tami, Christof Seiler, Antoine Chambaz, Sebastian Weichwald*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/027e86facfe7c1ea52ca1fca7bc1402b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/027e86facfe7c1ea52ca1fca7bc1402b-Abstract-Conference.html)

**Abstract**:

Additive Noise Models (ANMs) are a common model class for causal discovery from observational data. Due to a lack of real-world data for which an underlying ANM is known, ANMs with randomly sampled parameters are commonly used to simulate data for the evaluation of causal discovery algorithms. While some parameters may be fixed by explicit assumptions, fully specifying an ANM requires choosing all parameters. Reisach et al. (2021) show that, for many ANM parameter choices, sorting the variables by increasing variance yields an ordering close to a causal order and introduce ‘var-sortability’ to quantify this alignment. Since increasing variances may be unrealistic and cannot be exploited when data scales are arbitrary, ANM data are often rescaled to unit variance in causal discovery benchmarking.We show that synthetic ANM data are characterized by another pattern that is scale-invariant and thus persists even after standardization: the explainable fraction of a variable’s variance, as captured by the coefficient of determination $R^2$, tends to increase along the causal order. The result is high ‘$R^2$-sortability’, meaning that sorting the variables by increasing $R^2$ yields an ordering close to a causal order. We propose a computationally efficient baseline algorithm termed ‘$R^2$-SortnRegress’ that exploits high $R^2$-sortability and that can match and exceed the performance of established causal discovery algorithms. We show analytically that sufficiently high edge weights lead to a relative decrease of the noise contributions along causal chains, resulting in increasingly deterministic relationships and high $R^2$. We characterize $R^2$-sortability on synthetic data with different simulation parameters and find high values in common settings. Our findings reveal high $R^2$-sortability as an assumption about the data generating process relevant to causal discovery and implicit in many ANM sampling schemes. It should be made explicit, as its prevalence in real-world data is an open question. For causal discovery benchmarking, we provide implementations of $R^2$-sortability, the $R^2$-SortnRegress algorithm, and ANM simulation procedures in our library CausalDisco at https://causaldisco.github.io/CausalDisco/.

----

## [36] PROTES: Probabilistic Optimization with Tensor Sampling

**Authors**: *Anastasia Batsheva, Andrei Chertkov, Gleb V. Ryzhakov, Ivan V. Oseledets*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/028957869e560af14243ac37663a471e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/028957869e560af14243ac37663a471e-Abstract-Conference.html)

**Abstract**:

We developed a new method PROTES for black-box optimization, which is based on the probabilistic sampling from a probability density function given in the low-parametric tensor train format. We tested it on complex multidimensional arrays and discretized multivariable functions taken, among others, from real-world applications, including unconstrained binary optimization and optimal control problems, for which the possible number of elements is up to $2^{1000}$. In numerical experiments, both on analytic model functions and on complex problems, PROTES outperforms popular discrete optimization methods (Particle Swarm Optimization, Covariance Matrix Adaptation, Differential Evolution, and others).

----

## [37] Perturbation Towards Easy Samples Improves Targeted Adversarial Transferability

**Authors**: *Junqi Gao, Biqing Qi, Yao Li, Zhichang Guo, Dong Li, Yuming Xing, Dazhi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html)

**Abstract**:

The transferability of adversarial perturbations provides an effective shortcut for black-box attacks. Targeted perturbations have greater practicality but are more difficult to transfer between models. In this paper, we experimentally and theoretically demonstrated that neural networks trained on the same dataset have more consistent performance in High-Sample-Density-Regions (HSDR) of each class instead of low sample density regions. Therefore, in the target setting, adding perturbations towards HSDR of the target class is more effective in improving transferability. However, density estimation is challenging in high-dimensional scenarios. Further theoretical and experimental verification demonstrates that easy samples with low loss are more likely to be located in HSDR. Perturbations towards such easy samples in the target class can avoid density estimation for HSDR location. Based on the above facts, we verified that adding perturbations to easy samples in the target class improves targeted adversarial transferability of existing attack methods. A generative targeted attack strategy named Easy Sample Matching Attack (ESMA) is proposed, which has a higher success rate for targeted attacks and outperforms the SOTA generative method. Moreover, ESMA requires only $5\%$ of the storage space and much less computation time comparing to the current SOTA, as ESMA attacks all classes with only one model instead of seperate models for each class. Our code is available at https://github.com/gjq100/ESMA

----

## [38] AllSim: Simulating and Benchmarking Resource Allocation Policies in Multi-User Systems

**Authors**: *Jeroen Berrevoets, Daniel Jarrett, Alex J. Chan, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0296e17ec30fc36007edaaa2f96b5f17-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0296e17ec30fc36007edaaa2f96b5f17-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Numerous real-world systems, ranging from healthcare to energy grids, involve users competing for finite and potentially scarce resources. Designing policies for resource allocation in such real-world systems is challenging for many reasons, including the changing nature of user types and their (possibly urgent) need for resources. Researchers have developed numerous machine learning solutions for determining resource allocation policies in these challenging settings. However, a key limitation has been the absence of good methods and test-beds for benchmarking these policies; almost all resource allocation policies are benchmarked in environments which are either completely synthetic or do not allow any deviation from historical data. In this paper we introduce AllSim, which is a benchmarking environment for realistically simulating the impact and utility of policies for resource allocation in systems in which users compete for such scarce resources. Building such a benchmarking environment is challenging because it needs to successfully take into account the entire collective of potential users and the impact a resource allocation policy has on all the other users in the system. AllSim's benchmarking environment is modular (each component being parameterized individually), learnable (informed by historical data), and customizable (adaptable to changing conditions). These, when interacting with an allocation policy, produce a dataset of simulated outcomes for evaluation and comparison of such policies. We believe AllSim is an essential step towards a more systematic evaluation of policies for scarce resource allocation compared to current approaches for benchmarking such methods.

----

## [39] AVIS: Autonomous Visual Information Seeking with Large Language Model Agent

**Authors**: *Ziniu Hu, Ahmet Iscen, Chen Sun, Kai-Wei Chang, Yizhou Sun, David Ross, Cordelia Schmid, Alireza Fathi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/029df12a9363313c3e41047844ecad94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/029df12a9363313c3e41047844ecad94-Abstract-Conference.html)

**Abstract**:

In this paper, we propose an autonomous information seeking visual question answering framework, AVIS. Our method leverages a Large Language Model (LLM) to dynamically strategize the utilization of external tools and to investigate their outputs via tree search, thereby acquiring the indispensable knowledge needed to provide answers to the posed questions. Responding to visual questions that necessitate external knowledge, such as "What event is commemorated by the building depicted in this image?", is a complex task. This task presents a combinatorial search space that demands a sequence of actions, including invoking APIs, analyzing their responses, and making informed decisions. We conduct a user study to collect a variety of instances of human decision-making when faced with this task. This data is then used to design a system comprised of three components: an LLM-powered planner that dynamically determines which tool to use next, an LLM-powered reasoner that analyzes and extracts key information from the tool outputs, and a working memory component that retains the acquired information throughout the process. The collected user behavior serves as a guide for our system in two key ways. First, we create a transition graph by analyzing the sequence of decisions made by users. This graph delineates distinct states and confines the set of actions available at each state. Second, we use examples of user decision-making to provide our LLM-powered planner and reasoner with relevant contextual instances, enhancing their capacity to make informed decisions. We show that AVIS achieves state-of-the-art results on knowledge-based visual question answering benchmarks such as Infoseek and OK-VQA.

----

## [40] Conformal Prediction Sets for Ordinal Classification

**Authors**: *Prasenjit Dey, Srujana Merugu, Sivaramakrishnan R. Kaveri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/029f699912bf3db747fe110948cc6169-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/029f699912bf3db747fe110948cc6169-Abstract-Conference.html)

**Abstract**:

Ordinal classification (OC), i.e., labeling instances along classes with a natural ordering, is common in multiple  applications such as size or budget based recommendations and disease severity labeling.  Often in practical scenarios, it is desirable to obtain a small set of likely classes with a guaranteed high chance of including the true class. Recent works on conformal prediction (CP) address this problem for the classification setting with non-ordered labels but the resulting prediction sets (PS) are often non-contiguous and unsuitable for ordinal classification. In this work, we propose a framework to adapt existing CP methods to generate contiguous sets with guaranteed coverage and minimal cardinality. Our framework employs a novel non-parametric approach for modeling unimodal distributions. Empirical results on both synthetic and real-world datasets demonstrate our method outperforms SOTA baselines by 4% on Accuracy@K and 8% on PS size.

----

## [41] Minimax-Optimal Location Estimation

**Authors**: *Shivam Gupta, Jasper C. H. Lee, Eric Price, Paul Valiant*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02a589ef9a4f6f1e2dcc1cfb3b978a51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02a589ef9a4f6f1e2dcc1cfb3b978a51-Abstract-Conference.html)

**Abstract**:

Location estimation is one of the most basic questions in parametric statistics. Suppose we have a known distribution density $f$, and we get $n$ i.i.d. samples from $f(x-\mu)$ for some unknown shift $\mu$.The task is to estimate $\mu$ to high accuracy with high probability.The maximum likelihood estimator (MLE) is known to be asymptotically optimal as $n \to \infty$, but what is possible for finite $n$?In this paper, we give two location estimators that are optimal under different criteria: 1) an estimator that has minimax-optimal estimation error subject to succeeding with probability $1-\delta$ and 2) a confidence interval estimator which, subject to its output interval containing $\mu$ with probability at least $1-\delta$, has the minimum expected squared interval width among all shift-invariant estimators.The latter construction can be generalized to minimizing the expectation of any loss function on the interval width.

----

## [42] Tight Bounds for Volumetric Spanners and Applications

**Authors**: *Aditya Bhaskara, Sepideh Mahabadi, Ali Vakilian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02a92b52670752daf17b53f04f1ab405-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02a92b52670752daf17b53f04f1ab405-Abstract-Conference.html)

**Abstract**:

Given a set of points of interest, a volumetric spanner is a subset of the points using which all the points can be expressed using "small" coefficients (measured in an appropriate norm). Formally, given a set of vectors $X = [v_1, v_2, \dots, v_n]$, the goal is to find $T \subseteq [n]$ such that every $v \in X$ can be expressed as $\sum_{i\in T} \alpha_i v_i$, with $\Vert \alpha \Vert$ being small.  This notion, which has also been referred to as a well-conditioned basis, has found several applications, including bandit linear optimization, determinant maximization, and matrix low rank approximation. In this paper, we give almost optimal bounds on the size of volumetric spanners for all $\ell_p$ norms, and show that they can be constructed using a simple local search procedure. We then show the applications of our result to other tasks and in particular the problem of finding coresets for the Minimum Volume Enclosing Ellipsoid (MVEE) problem.

----

## [43] Wyze Rule: Federated Rule Dataset for Rule Recommendation Benchmarking

**Authors**: *Mohammad Mahdi Kamani, Yuhang Yao, Hanjia Lyu, Zhongwei Cheng, Lin Chen, Liangju Li, Carlee Joe-Wong, Jiebo Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02b9d1e6d1b5295a6f883969ddc1bbbd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/02b9d1e6d1b5295a6f883969ddc1bbbd-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In the rapidly evolving landscape of smart home automation, the potential of IoT devices is vast. In this realm, rules are the main tool utilized for this automation, which are predefined conditions or triggers that establish connections between devices, enabling seamless automation of specific processes. However, one significant challenge researchers face is the lack of comprehensive datasets to explore and advance the field of smart home rule recommendations. These datasets are essential for developing and evaluating intelligent algorithms that can effectively recommend rules for automating processes while preserving the privacy of the users, as it involves personal information about users' daily lives. To bridge this gap, we present the Wyze Rule Dataset, a large-scale dataset designed specifically for smart home rule recommendation research. Wyze Rule encompasses over 1 million rules gathered from a diverse user base of 300,000 individuals from Wyze Labs, offering an extensive and varied collection of real-world data.   With a focus on federated learning, our dataset is tailored to address the unique challenges of a cross-device federated learning setting in the recommendation domain, featuring a large-scale number of clients with widely heterogeneous data. To establish a benchmark for comparison and evaluation, we have meticulously implemented multiple baselines in both centralized and federated settings. Researchers can leverage these baselines to gauge the performance and effectiveness of their rule recommendation systems, driving advancements in the domain. The Wyze Rule Dataset is publicly accessible through HuggingFace's dataset API.

----

## [44] Learning better with Dale's Law: A Spectral Perspective

**Authors**: *Pingsheng Li, Jonathan Cornford, Arna Ghosh, Blake A. Richards*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02dd0db10c40092de3d9ec2508d12f60-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02dd0db10c40092de3d9ec2508d12f60-Abstract-Conference.html)

**Abstract**:

Most recurrent neural networks (RNNs) do not include a fundamental constraint of real neural circuits: Dale's Law, which implies that neurons must be excitatory (E) or inhibitory (I). Dale's Law is generally absent from RNNs because simply partitioning a standard network's units into E and I populations impairs learning. However, here we extend a recent feedforward bio-inspired EI network architecture, named Dale's ANNs, to recurrent networks, and demonstrate that good performance is possible while respecting Dale's Law. This begs the question: What makes some forms of EI network learn poorly and others learn well? And, why does the simple approach of incorporating Dale's Law impair learning?  Historically the answer was thought to be the sign constraints on EI network parameters, and this was a motivation behind Dale's ANNs. However, here we show the spectral properties of the recurrent weight matrix at initialisation are more impactful on network performance than sign constraints. We find that simple EI partitioning results in a singular value distribution that is multimodal and dispersed, whereas standard RNNs have an unimodal, more clustered singular value distribution, as do recurrent Dale's ANNs. We also show that the spectral properties and performance of partitioned EI networks are worse for small networks with fewer I units, and we present normalised SVD entropy as a measure of spectrum pathology that correlates with performance. Overall, this work sheds light on a long-standing mystery in neuroscience-inspired AI and computational neuroscience, paving the way for greater alignment between neural networks and biology.

----

## [45] Dense-Exponential Random Features: Sharp Positive Estimators of the Gaussian Kernel

**Authors**: *Valerii Likhosherstov, Krzysztof Marcin Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamás Sarlós, Adrian Weller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/02dec8877fb7c6aa9a79f81661baca7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/02dec8877fb7c6aa9a79f81661baca7c-Abstract-Conference.html)

**Abstract**:

The problem of efficient approximation of a linear operator induced by the Gaussian or softmax kernel is often addressed using random features (RFs) which yield an unbiased approximation of the operator's result. Such operators emerge in important applications ranging from kernel methods to efficient Transformers. We propose parameterized, positive, non-trigonometric RFs which approximate Gaussian and softmax-kernels. In contrast to traditional RF approximations, parameters of these new methods can be optimized to reduce the variance of the approximation, and the optimum can be expressed in closed form. We show that our methods lead to variance reduction in practice (e^{10}-times smaller variance and beyond) and outperform previous methods in a kernel regression task. Using our proposed mechanism, we also present FAVOR#, a method for self-attention approximation in Transformers. We show that FAVOR# outperforms other random feature methods in speech modelling and natural language processing.

----

## [46] Projection-Free Online Convex Optimization via Efficient Newton Iterations

**Authors**: *Khashayar Gatmiry, Zakaria Mhammedi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03261886741f1f21f52f2a2d570616a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03261886741f1f21f52f2a2d570616a2-Abstract-Conference.html)

**Abstract**:

This paper presents new projection-free algorithms for Online Convex Optimization (OCO) over a convex domain $\mathcal{K} \subset \mathbb{R}^d$. Classical OCO algorithms (such as Online Gradient Descent) typically need to perform Euclidean projections onto the convex set $\mathcal{K}$ to ensure feasibility of their iterates. Alternative algorithms, such as those based on the Frank-Wolfe method, swap potentially-expensive Euclidean projections onto $\mathcal{K}$ for linear optimization over $\mathcal{K}$. However, such algorithms have a sub-optimal regret in OCO compared to projection-based algorithms. In this paper, we look at a third type of algorithms that output approximate Newton iterates using a self-concordant barrier for the set of interest. The use of a self-concordant barrier automatically ensures feasibility without the need of projections. However, the computation of the Newton iterates requires a matrix inverse, which can still be expensive. As our main contribution, we show how the stability of the Newton iterates can be leveraged to only compute the inverse Hessian a vanishing fractions of the rounds, leading to a new efficient projection-free OCO algorithm with a state-of-the-art regret bound.

----

## [47] Read and Reap the Rewards: Learning to Play Atari with the Help of Instruction Manuals

**Authors**: *Yue Wu, Yewen Fan, Paul Pu Liang, Amos Azaria, Yuanzhi Li, Tom M. Mitchell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/034d7bfeace2a9a258648b16fc626298-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/034d7bfeace2a9a258648b16fc626298-Abstract-Conference.html)

**Abstract**:

High sample complexity has long been a challenge for RL. On the other hand, humans learn to perform tasks not only from interaction or demonstrations, but also by reading unstructured text documents, e.g., instruction manuals. Instruction manuals and wiki pages are among the most abundant data that could inform agents of valuable features and policies or task-specific environmental dynamics and reward structures. Therefore, we hypothesize that the ability to utilize human-written instruction manuals to assist learning policies for specific tasks should lead to a more efficient and better-performing agent. We propose the Read and Reward framework. Read and Reward speeds up RL algorithms on Atari games by reading manuals released by the Atari game developers. Our framework consists of a QA Extraction module that extracts and summarizes relevant information from the manual and a Reasoning module that evaluates object-agent interactions based on information from the manual. An auxiliary reward is then provided to a standard A2C RL agent, when interaction is detected. Experimentally, various RL algorithms obtain significant improvement in performance and training speed when assisted by our design. Code at github.com/Holmeswww/RnR

----

## [48] Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization

**Authors**: *Kaiyue Wen, Zhiyuan Li, Tengyu Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0354767c6386386be17cabe4fc59711b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0354767c6386386be17cabe4fc59711b-Abstract-Conference.html)

**Abstract**:

Despite extensive studies, the underlying reason as to why overparameterizedneural networks can generalize remains elusive. Existing theory shows that common stochastic optimizers prefer flatter minimizers of the training loss, and thusa natural potential explanation is that flatness implies generalization. This workcritically examines this explanation. Through theoretical and empirical investigation, we identify the following three scenarios for two-layer ReLU networks: (1)flatness provably implies generalization; (2) there exist non-generalizing flattestmodels and sharpness minimization algorithms fail to generalize poorly, and (3)perhaps most strikingly, there exist non-generalizing flattest models, but sharpnessminimization algorithms still generalize. Our results suggest that the relationshipbetween sharpness and generalization subtly depends on the data distributionsand the model architectures and sharpness minimization algorithms do not onlyminimize sharpness to achieve better generalization. This calls for the search forother explanations for the generalization of over-parameterized neural networks

----

## [49] Feature-Learning Networks Are Consistent Across Widths At Realistic Scales

**Authors**: *Nikhil Vyas, Alexander Atanasov, Blake Bordelon, Depen Morwani, Sabarish Sainathan, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03600ae6c3392fd65ad7c3a90c6f7ce8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03600ae6c3392fd65ad7c3a90c6f7ce8-Abstract-Conference.html)

**Abstract**:

We study the effect of width on the dynamics of feature-learning neural networks across a variety of architectures and datasets. Early in training, wide neural networks trained on online data have not only identical loss curves but also agree in their point-wise test predictions throughout training. For simple tasks such as CIFAR-5m this holds throughout training for networks of realistic widths. We also show that structural properties of the models, including internal representations, preactivation distributions, edge of stability phenomena, and large learning rate effects are consistent across large widths. This motivates the hypothesis that phenomena seen in realistic models can be captured by infinite-width, feature-learning limits. For harder tasks (such as ImageNet and language modeling), and later training times, finite-width deviations grow systematically. Two distinct effects cause these deviations across widths. First, the network output has an initialization-dependent variance scaling inversely with width, which can be removed by ensembling networks. We observe, however, that ensembles of narrower networks perform worse than a single wide network. We call this the bias of narrower width. We conclude with a spectral perspective on the origin of this finite-width bias.

----

## [50] Taylor TD-learning

**Authors**: *Michele Garibbo, Maxime Robeyns, Laurence Aitchison*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/036912a83bdbb1fd792baf6532f102d8-Abstract-Conference.html)

**Abstract**:

Many reinforcement learning approaches rely on temporal-difference (TD) learning to learn a critic.However, TD-learning updates can be high variance.Here, we introduce a model-based RL framework, Taylor TD, which reduces this variance in continuous state-action settings. Taylor TD uses a first-order Taylor series expansion of TD updates.This expansion allows Taylor TD to analytically integrate over stochasticity in the action-choice, and some stochasticity in the state distribution for the initial state and action of each TD update.We include theoretical and empirical evidence that Taylor TD updates are indeed lower variance than standard TD updates. Additionally, we show Taylor TD has the same stable learning guarantees as standard TD-learning with linear function approximation under a reasonable assumption.Next, we combine Taylor TD with the TD3 algorithm, forming TaTD3.We show TaTD3 performs as well, if not better, than several state-of-the art model-free and model-based baseline algorithms on a set of standard benchmark tasks.

----

## [51] Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability

**Authors**: *Maciej Falkiewicz, Naoya Takeishi, Imahn Shekhzadeh, Antoine Wehenkel, Arnaud Delaunoy, Gilles Louppe, Alexandros Kalousis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03a9a9c1e15850439653bb971a4ad4b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03a9a9c1e15850439653bb971a4ad4b3-Abstract-Conference.html)

**Abstract**:

Bayesian inference allows expressing the uncertainty of posterior belief under a probabilistic model given prior information and the likelihood of the evidence. Predominantly, the likelihood function is only implicitly established by a simulator posing the need for simulation-based inference (SBI). However, the existing algorithms can yield overconfident posteriors (Hermans et al., 2022) defeating the whole purpose of credibility if the uncertainty quantification is inaccurate. We propose to include a calibration term directly into the training objective of the neural model in selected amortized SBI techniques. By introducing a relaxation of the classical formulation of calibration error we enable end-to-end backpropagation. The proposed method is not tied to any particular neural model and brings moderate computational overhead compared to the profits it introduces. It is directly applicable to existing computational pipelines allowing reliable black-box posterior inference. We empirically show on six benchmark problems that the proposed method achieves competitive or better results in terms of coverage and expected posterior density than the previously existing approaches.

----

## [52] Agnostic Multi-Group Active Learning

**Authors**: *Nicholas Rittler, Kamalika Chaudhuri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03b1043052700b1a471996b0baf309d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03b1043052700b1a471996b0baf309d4-Abstract-Conference.html)

**Abstract**:

Inspired by the problem of improving classification accuracy on rare or hard subsets of a population, there has been recent interest in models of learning where the goal is to generalize to a collection of distributions, each representing a ``group''. We consider a variant of this problem from the perspective of active learning, where the learner is endowed with the power to decide which examples are labeled from each distribution in the collection, and the goal is to minimize the number of label queries while maintaining PAC-learning guarantees. Our main challenge is that standard active learning techniques such as disagreement-based active learning do not directly apply to the multi-group learning objective. We modify existing algorithms to provide a consistent active learning algorithm for an agnostic formulation of multi-group learning, which given a collection of $G$ distributions and a hypothesis class $\mathcal{H}$ with VC-dimension $d$, outputs an $\epsilon$-optimal hypothesis using $\tilde{O}\left( (\nu^2/\epsilon^2) G d  \theta_{\mathcal{G}}^2 \log^2(1/\epsilon) + G\log(1/\epsilon)/\epsilon^2 \right)$ label queries, where $\theta_{\mathcal{G}}$ is the worst-case disagreement coefficient over the collection. Roughly speaking, this guarantee improves upon the label complexity of standard multi-group learning in regimes where disagreement-based active learning algorithms may be expected to succeed, and the number of groups is not too large. We also consider the special case where each distribution in the collection is individually realizable with respect to $\mathcal{H}$, and demonstrate $\tilde{O}\left( G d \theta_{\mathcal{G}} \log(1/\epsilon) \right)$ label queries are sufficient for learning in this case. We further give an approximation result for the full agnostic case inspired by the group realizable strategy.

----

## [53] Self-Weighted Contrastive Learning among Multiple Views for Mitigating Representation Degeneration

**Authors**: *Jie Xu, Shuo Chen, Yazhou Ren, Xiaoshuang Shi, Hengtao Shen, Gang Niu, Xiaofeng Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03b13b0db740b95cb741e007178ef5e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03b13b0db740b95cb741e007178ef5e5-Abstract-Conference.html)

**Abstract**:

Recently, numerous studies have demonstrated the effectiveness of contrastive learning (CL), which learns feature representations by pulling in positive samples while pushing away negative samples. Many successes of CL lie in that there exists semantic consistency between data augmentations of the same instance. In multi-view scenarios, however, CL might cause representation degeneration when the collected multiple views inherently have inconsistent semantic information or their representations subsequently do not capture sufficient discriminative information. To address this issue, we propose a novel framework called SEM: SElf-weighted Multi-view contrastive learning with reconstruction regularization. Specifically, SEM is a general framework where we propose to first measure the discrepancy between pairwise representations and then minimize the corresponding self-weighted contrastive loss, and thus making SEM adaptively strengthen the useful pairwise views and also weaken the unreliable pairwise views. Meanwhile, we impose a self-supervised reconstruction term to regularize the hidden features of encoders, to assist CL in accessing sufficient discriminative information of data. Experiments on public multi-view datasets verified that SEM can mitigate representation degeneration in existing CL methods and help them achieve significant performance improvements. Ablation studies also demonstrated the effectiveness of SEM with different options of weighting strategies and reconstruction terms.

----

## [54] Neural Polarizer: A Lightweight and Effective Backdoor Defense via Purifying Poisoned Features

**Authors**: *Mingli Zhu, Shaokui Wei, Hongyuan Zha, Baoyuan Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03df5246cc78af497940338dd3eacbaa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03df5246cc78af497940338dd3eacbaa-Abstract-Conference.html)

**Abstract**:

Recent studies have demonstrated the susceptibility of deep neural networks to backdoor attacks. Given a backdoored model, its prediction of a poisoned sample with trigger will be dominated by the trigger information, though trigger information and benign information coexist. Inspired by the mechanism of the optical polarizer that a polarizer could pass light waves with particular polarizations while filtering light waves with other polarizations, we propose a novel backdoor defense method by inserting a learnable neural polarizer into the backdoored model as an intermediate layer, in order to purify the poisoned sample via filtering trigger information while maintaining benign information. The neural polarizer is instantiated as one lightweight linear transformation layer, which is learned through solving a well designed bi-level optimization problem, based on a limited clean dataset. Compared to other fine-tuning-based defense methods which often adjust all parameters of the backdoored model, the proposed method only needs to learn one additional layer, such that it is more efficient and requires less clean data. Extensive experiments demonstrate the effectiveness and efficiency of our method in removing backdoors across various neural network architectures and datasets, especially in the case of very limited clean data. Codes are available at \href{https://github.com/SCLBD/BackdoorBench}{https://github.com/SCLBD/BackdoorBench} (PyTorch) and \href{https://github.com/JulieCarlon/NPD-MindSpore}{https://github.com/JulieCarlon/NPD-MindSpore} (MindSpore).

----

## [55] Tools for Verifying Neural Models' Training Data

**Authors**: *Dami Choi, Yonadav Shavit, David Kristjanson Duvenaud*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/03e33e1f62e3302b47fe1d38a235921e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/03e33e1f62e3302b47fe1d38a235921e-Abstract-Conference.html)

**Abstract**:

It is important that consumers and regulators can verify the provenance of large neural models to evaluate their capabilities and risks. We introduce the concept of a "Proof-of-Training-Data": any protocol that allows a model trainer to convince a Verifier of the training data that produced a set of model weights. Such protocols could verify the amount and kind of data and compute used to train the model, including whether it was trained on specific harmful or beneficial data sources. We explore efficient verification strategies for Proof-of-Training-Data that are compatible with most current large-model training procedures. These include a method for the model-trainer to verifiably pre-commit to a random seed used in training, and a method that exploits models' tendency to temporarily overfit to training data in order to detect whether a given data-point was included in training. We show experimentally that our verification procedures can catch a wide variety of attacks, including all known attacks from the Proof-of-Learning literature.

----

## [56] Towards Higher Ranks via Adversarial Weight Pruning

**Authors**: *Yuchuan Tian, Hanting Chen, Tianyu Guo, Chao Xu, Yunhe Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/040ace837dd270a87055bb10dd7c0392-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/040ace837dd270a87055bb10dd7c0392-Abstract-Conference.html)

**Abstract**:

Convolutional Neural Networks (CNNs) are hard to deploy on edge devices due to its high computation and storage complexities. As a common practice for model compression, network pruning consists of two major categories: unstructured and structured pruning, where unstructured pruning constantly performs better. However, unstructured pruning presents a structured pattern at high pruning rates, which limits its performance. To this end, we propose a Rank-based PruninG (RPG) method to maintain the ranks of sparse weights in an adversarial manner. In each step, we minimize the low-rank approximation error for the weight matrices using singular value decomposition, and maximize their distance by pushing the weight matrices away from its low rank approximation. This rank-based optimization objective guides sparse weights towards a high-rank topology. The proposed method is conducted in a gradual pruning fashion to stabilize the change of rank during training. Experimental results on various datasets and different tasks demonstrate the effectiveness of our algorithm in high sparsity. The proposed RPG outperforms the state-of-the-art performance by 1.13\% top-1 accuracy on ImageNet in ResNet-50 with 98\% sparsity. The codes are available at https://github.com/huawei-noah/Efficient-Computing/tree/master/Pruning/RPG and https://gitee.com/mindspore/models/tree/master/research/cv/RPG.

----

## [57] On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective

**Authors**: *Zeke Xie, Zhiqiang Xu, Jingzhao Zhang, Issei Sato, Masashi Sugiyama*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/040d3b6af368bf71f952c18da5713b48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/040d3b6af368bf71f952c18da5713b48-Abstract-Conference.html)

**Abstract**:

Weight decay is a simple yet powerful regularization technique that has been very widely used in training of deep neural networks (DNNs). While weight decay has attracted much attention, previous studies fail to discover some overlooked pitfalls on large gradient norms resulted by weight decay. In this paper, we discover that, weight decay can unfortunately lead to large gradient norms at the final phase (or the terminated solution) of training, which often indicates bad convergence and poor generalization. To mitigate the gradient-norm-centered pitfalls, we present the first practical scheduler for weight decay, called the Scheduled Weight Decay (SWD) method that can dynamically adjust the weight decay strength according to the gradient norm and significantly penalize large gradient norms during training. Our experiments also support that SWD indeed mitigates large gradient norms and often significantly outperforms the conventional constant weight decay strategy for Adaptive Moment Estimation (Adam).

----

## [58] Leveraging Early-Stage Robustness in Diffusion Models for Efficient and High-Quality Image Synthesis

**Authors**: *Yulhwa Kim, Dongwon Jo, Hyesung Jeon, Taesu Kim, Daehyun Ahn, Hyungjun Kim, Jae-Joon Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/04261fce1705c4f02f062866717d592a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/04261fce1705c4f02f062866717d592a-Abstract-Conference.html)

**Abstract**:

While diffusion models have demonstrated exceptional image generation capabilities, the iterative noise estimation process required for these models is compute-intensive and their practical implementation is limited by slow sampling speeds. In this paper, we propose a novel approach to speed up the noise estimation network by leveraging the robustness of early-stage diffusion models. Our findings indicate that inaccurate computation during the early-stage of the reverse diffusion process has minimal impact on the quality of generated images, as this stage primarily outlines the image while later stages handle the finer details that require more sensitive information. To improve computational efficiency, we combine our findings with post-training quantization (PTQ) to introduce a method that utilizes low-bit activation for the early reverse diffusion process while maintaining high-bit activation for the later stages. Experimental results show that the proposed method can accelerate the early-stage computation without sacrificing the quality of the generated images.

----

## [59] Adversarial Model for Offline Reinforcement Learning

**Authors**: *Mohak Bhardwaj, Tengyang Xie, Byron Boots, Nan Jiang, Ching-An Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0429ececfb199efc93182990169e73bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0429ececfb199efc93182990169e73bb-Abstract-Conference.html)

**Abstract**:

We propose a novel model-based offline Reinforcement Learning (RL) framework, called Adversarial Model for Offline Reinforcement Learning (ARMOR), which can robustly learn policies to improve upon an arbitrary reference policy regardless of data coverage. ARMOR is designed to optimize policies for the worst-case performance relative to the reference policy through adversarially training a Markov decision process model. In theory, we prove that ARMOR, with a well-tuned hyperparameter, can compete with the best policy within data coverage when the reference policy is supported by the data. At the same time, ARMOR is robust to hyperparameter choices: the policy learned by ARMOR, with any admissible hyperparameter, would never degrade the performance of the reference policy, even when the reference policy is not covered by the dataset. To validate these properties in practice, we design a scalable implementation of ARMOR, which by adversarial training, can optimize policies without using model ensembles in contrast to typical model-based methods. We show that ARMOR achieves competent performance with both state-of-the-art offline model-free and model-based RL algorithms and can robustly improve the reference policy over various hyperparameter choices.

----

## [60] Training Your Image Restoration Network Better with Random Weight Network as Optimization Function

**Authors**: *Man Zhou, Naishan Zheng, Yuan Xu, Chun-Le Guo, Chongyi Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/043f0503c4f652c737add3690aa5d12c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/043f0503c4f652c737add3690aa5d12c-Abstract-Conference.html)

**Abstract**:

The blooming progress made in deep learning-based image restoration has been largely attributed to the availability of high-quality, large-scale datasets and advanced network structures. However, optimization functions such as L1 and L2 are still de facto. In this study, we propose to  investigate new optimization functions to improve image restoration performance. Our key insight is  that ``random weight network can be acted as a constraint for training better image restoration networks''. However, not all random weight networks are suitable as constraints. We draw inspiration from Functional theory and show that alternative random weight networks should be represented in the form of a strict mathematical manifold. We explore the potential of our random weight network prototypes that satisfy this requirement:  Taylor's unfolding network, invertible neural network, central difference convolution, and zero-order filtering. We investigate these prototypes from four aspects: 1)   random weight strategies, 2)  network architectures, 3)   network depths, and 4) combinations of random weight networks. Furthermore, we devise the random weight in two variants:  the weights are randomly initialized only once during the entire training procedure, and  the weights are randomly initialized in each training epoch. Our approach can be directly integrated into existing networks without incurring additional training and testing computational costs. We perform extensive experiments across multiple image restoration tasks, including image denoising, low-light image enhancement, and guided image super-resolution to demonstrate the consistent performance gains achieved by our method.  Upon acceptance of this paper, we will release the code.

----

## [61] Passive learning of active causal strategies in agents and language models

**Authors**: *Andrew K. Lampinen, Stephanie C. Y. Chan, Ishita Dasgupta, Andrew J. Nam, Jane X. Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/045c87def0c02e3ad0d3d849766d7f1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/045c87def0c02e3ad0d3d849766d7f1e-Abstract-Conference.html)

**Abstract**:

What can be learned about causality and experimentation from passive data? This question is salient given recent successes of passively-trained language models in interactive domains such as tool use. Passive learning is inherently limited. However, we show that purely passive learning can in fact allow an agent to learn generalizable strategies for determining and using causal structures, as long as the agent can intervene at test time. We formally illustrate that learning a strategy of first experimenting, then seeking goals, can allow generalization from passive learning in principle. We then show empirically that agents trained via imitation on expert data can indeed generalize at test time to infer and use causal links which are never present in the training data; these agents can also generalize experimentation strategies to novel variable sets never observed in training.We then show that strategies for causal intervention and exploitation can be generalized from passive data even in a more complex environment with high-dimensional observations, with the support of natural language explanations. Explanations can even allow passive learners to generalize out-of-distribution from perfectly-confounded training data. Finally, we show that language models, trained only on passive next-word prediction, can generalize causal intervention strategies from a few-shot prompt containing explanations and reasoning. These results highlight the surprising power of passive learning of active causal strategies, and have implications for understanding the behaviors and capabilities of language models.

----

## [62] Zero-Regret Performative Prediction Under Inequality Constraints

**Authors**: *Wenjing Yan, Xuanyu Cao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/047397849f63b4fcfced4ff720159f3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/047397849f63b4fcfced4ff720159f3d-Abstract-Conference.html)

**Abstract**:

Performative prediction is a recently proposed framework where predictions guide decision-making and hence influence future data distributions. Such performative phenomena are ubiquitous in various areas, such as transportation, finance, public policy, and recommendation systems. To date, work on performative prediction has only focused on unconstrained problems, neglecting the fact that many real-world learning problems are subject to constraints. This paper bridges this gap by studying performative prediction under inequality constraints. Unlike most existing work that provides only performative stable points, we aim to find the optimal solutions. Anticipating performative gradient is a challenging task, due to the agnostic performative effect on data distributions. To address this issue, we first develop a robust primal-dual framework that requires only approximate gradients up to a certain accuracy, yet delivers the same order of performance as the stationary stochastic primal-dual algorithm without performativity. Based on this framework, we then propose an adaptive primal-dual algorithm for location families. Our analysis demonstrates that the proposed adaptive primal-dual algorithm attains $\mathcal{O}(\sqrt{T})$ regret and constraint violations, using only $\sqrt{T} + 2T$ samples, where $T$ is the time horizon. To our best knowledge, this is the first study and analysis on the optimality of the performative prediction problem under inequality constraints. Finally, we validate the effectiveness of our algorithm and theoretical results through numerical simulations.

----

## [63] Towards Free Data Selection with General-Purpose Models

**Authors**: *Yichen Xie, Mingyu Ding, Masayoshi Tomizuka, Wei Zhan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/047682108c3b053c61ad2da5a6057b4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/047682108c3b053c61ad2da5a6057b4e-Abstract-Conference.html)

**Abstract**:

A desirable data selection algorithm can efficiently choose the most informative samples to maximize the utility of limited annotation budgets. However, current approaches, represented by active learning methods, typically follow a cumbersome pipeline that iterates the time-consuming model training and batch data selection repeatedly. In this paper, we challenge this status quo by designing a distinct data selection pipeline that utilizes existing general-purpose models to select data from various datasets with a single-pass inference without the need for additional training or supervision. A novel free data selection (FreeSel) method is proposed following this new pipeline. Specifically, we define semantic patterns extracted from inter-mediate features of the general-purpose model to capture subtle local information in each image. We then enable the selection of all data samples in a single pass through distance-based sampling at the fine-grained semantic pattern level. FreeSel bypasses the heavy batch selection process, achieving a significant improvement in efficiency and being 530x faster than existing active learning methods. Extensive experiments verify the effectiveness of FreeSel on various computer vision tasks.

----

## [64] Communication-Efficient Federated Bilevel Optimization with Global and Local Lower Level Problems

**Authors**: *Junyi Li, Feihu Huang, Heng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/04bd683d5428d91c5fbb5a7d2c27064d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/04bd683d5428d91c5fbb5a7d2c27064d-Abstract-Conference.html)

**Abstract**:

Bilevel Optimization has witnessed notable progress recently with new emerging efficient algorithms. However, its application in the Federated Learning setting remains relatively underexplored, and the impact of Federated Learning's inherent challenges on the convergence of bilevel algorithms remain obscure.In this work, we investigate Federated Bilevel Optimization problems and propose a communication-efficient algorithm, named FedBiOAcc. The algorithm leverages an efficient estimation of the hyper-gradient in the distributed setting and utilizes the momentum-based variance-reduction acceleration. Remarkably, FedBiOAcc achieves a communication complexity $O(\epsilon^{-1})$, a sample complexity $O(\epsilon^{-1.5})$ and the linear speed up with respect to the number of clients. We also analyze a special case of the Federated Bilevel Optimization problems, where lower level problems are locally managed by clients. We prove that FedBiOAcc-Local, a modified version of FedBiOAcc, converges at the same rate for this type of problems. Finally, we validate the proposed algorithms through two real-world tasks: Federated Data-cleaning and Federated Hyper-representation Learning. Empirical results show superior performance of our algorithms.

----

## [65] Partial Multi-Label Learning with Probabilistic Graphical Disambiguation

**Authors**: *Jun-Yi Hang, Min-Ling Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/04e05ba5cbc36044f6499d1edf15247e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/04e05ba5cbc36044f6499d1edf15247e-Abstract-Conference.html)

**Abstract**:

In partial multi-label learning (PML), each training example is associated with a set of candidate labels, among which only some labels are valid. As a common strategy to tackle PML problem, disambiguation aims to recover the ground-truth labeling information from such inaccurate annotations. However, existing approaches mainly rely on heuristics or ad-hoc rules to disambiguate candidate labels, which may not be universal enough in complicated real-world scenarios. To provide a principled way for disambiguation, we make a first attempt to explore the probabilistic graphical model for PML problem, where a directed graph is tailored to infer latent ground-truth labeling information from the generative process of partial multi-label data. Under the framework of stochastic gradient variational Bayes, a unified variational lower bound is derived for this graphical model, which is further relaxed probabilistically so that the desired prediction model can be induced with simultaneously identified ground-truth labeling information. Comprehensive experiments on multiple synthetic and real-world data sets show that our approach outperforms the state-of-the-art counterparts.

----

## [66] Reward Scale Robustness for Proximal Policy Optimization via DreamerV3 Tricks

**Authors**: *Ryan Sullivan, Akarsh Kumar, Shengyi Huang, John P. Dickerson, Joseph Suarez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/04f61ec02d1b3a025a59d978269ce437-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/04f61ec02d1b3a025a59d978269ce437-Abstract-Conference.html)

**Abstract**:

Most reinforcement learning methods rely heavily on dense, well-normalized environment rewards. DreamerV3 recently introduced a model-based method with a number of tricks that mitigate these limitations, achieving state-of-the-art on a wide range of benchmarks with a single set of hyperparameters. This result sparked discussion about the generality of the tricks, since they appear to be applicable to other reinforcement learning algorithms. Our work applies DreamerV3's tricks to PPO and is the first such empirical study outside of the original work. Surprisingly, we find that the tricks presented do not transfer as general improvements to PPO. We use a high quality PPO reference implementation and present extensive ablation studies totaling over 10,000 A100 hours on the Arcade Learning Environment and the DeepMind Control Suite. Though our experiments demonstrate that these tricks do not generally outperform PPO, we identify cases where they succeed and offer insight into the relationship between the implementation tricks. In particular, PPO with these tricks performs comparably to PPO on Atari games with reward clipping and significantly outperforms PPO without reward clipping.

----

## [67] Emergent Correspondence from Image Diffusion

**Authors**: *Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, Bharath Hariharan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0503f5dce343a1d06d16ba103dd52db1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0503f5dce343a1d06d16ba103dd52db1-Abstract-Conference.html)

**Abstract**:

Finding correspondences between images is a fundamental problem in computer vision. In this paper, we show that correspondence emerges in image diffusion models without any explicit supervision. We propose a simple strategy to extract this implicit knowledge out of diffusion networks as image features, namely DIffusion FeaTures (DIFT), and use them to establish correspondences between real images. Without any additional fine-tuning or supervision on the task-specific data or annotations, DIFT is able to outperform both weakly-supervised methods and competitive off-the-shelf features in identifying semantic, geometric, and temporal correspondences. Particularly for semantic correspondence, DIFT from Stable Diffusion is able to outperform DINO and OpenCLIP by 19 and 14 accuracy points respectively on the challenging SPair-71k benchmark. It even outperforms the state-of-the-art supervised methods on 9 out of 18 categories while remaining on par for the overall performance. Project page: https://diffusionfeatures.github.io.

----

## [68] Robust Learning with Progressive Data Expansion Against Spurious Correlation

**Authors**: *Yihe Deng, Yu Yang, Baharan Mirzasoleiman, Quanquan Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0506ad3d1bcc8398a920db9340f27fe4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0506ad3d1bcc8398a920db9340f27fe4-Abstract-Conference.html)

**Abstract**:

While deep learning models have shown remarkable performance in various tasks, they are susceptible to learning non-generalizable _spurious features_ rather than the core features that are genuinely correlated to the true label. In this paper, beyond existing analyses of linear models, we theoretically examine the learning process of a two-layer nonlinear convolutional neural network in the presence of spurious features. Our analysis suggests that imbalanced data groups and easily learnable spurious features can lead to the dominance of spurious features during the learning process. In light of this, we propose a new training algorithm called **PDE** that efficiently enhances the model's robustness for a better worst-group performance. PDE begins with a group-balanced subset of training data and progressively expands it to facilitate the learning of the core features. Experiments on synthetic and real-world benchmark datasets confirm the superior performance of our method on models such as ResNets and Transformers. On average, our method achieves a $2.8$ \% improvement in worst-group accuracy compared with the state-of-the-art method, while enjoying up to $10\times$ faster training efficiency.

----

## [69] Multiclass Boosting: Simple and Intuitive Weak Learning Criteria

**Authors**: *Nataly Brukhim, Amit Daniely, Yishay Mansour, Shay Moran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/050f8591be3874b52fdac4e1060eeb29-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/050f8591be3874b52fdac4e1060eeb29-Abstract-Conference.html)

**Abstract**:

We study a generalization of boosting to the multiclass setting.We introduce a weak learning condition for multiclass classification that captures the original notion of weak learnability as being “slightly better than random guessing”. We give a simple and efficient boosting algorithm, that does not require realizability assumptions and its sample and oracle complexity bounds are independent of the number of classes. In addition, we utilize our new boosting technique in several theoretical applications within the context of List PAC Learning. First, we establish an equivalence to weak PAC learning. Furthermore, we present a new result on boosting for list learners, as well as provide a novel proof for the characterization of multiclass PAC learning and List PAC learning. Notably, our technique gives rise to simplified algorithms and analysis compared to previous works.

----

## [70] Approximate Heavy Tails in Offline (Multi-Pass) Stochastic Gradient Descent

**Authors**: *Kruno Lehman, Alain Durmus, Umut Simsekli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0525a72df7fb2cd943c780d059b94774-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0525a72df7fb2cd943c780d059b94774-Abstract-Conference.html)

**Abstract**:

A recent line of empirical studies has demonstrated that SGD might exhibit a heavy-tailed behavior in practical settings, and the heaviness of the tails might correlate with the overall performance. In this paper, we investigate the emergence of such heavy tails. Previous works on this problem only considered, up to our knowledge, online (also called single-pass) SGD, in which the emergence of heavy tails in theoretical findings is contingent upon access to an infinite amount of data. Hence, the underlying mechanism generating the reported heavy-tailed behavior in practical settings, where the amount of training data is finite, is still not well-understood. Our contribution aims to fill this gap. In particular, we show that the stationary distribution of offline (also called multi-pass) SGD exhibits ‘approximate’ power-law tails and the approximation error is controlled by how fast the empirical distribution of the training data converges to the true underlying data distribution in the Wasserstein metric. Our main takeaway is that, as the number of data points increases, offline SGD will behave increasingly ‘power-law-like’. To achieve this result, we first prove nonasymptotic Wasserstein convergence bounds for offline SGD to online SGD as the number of data points increases, which can be interesting on their own. Finally, we illustrate our theory on various experiments conducted on synthetic data and neural networks.

----

## [71] Uncovering Neural Scaling Laws in Molecular Representation Learning

**Authors**: *Dingshuo Chen, Yanqiao Zhu, Jieyu Zhang, Yuanqi Du, Zhixun Li, Qiang Liu, Shu Wu, Liang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/052e22cfdd344c79634f7ec76fa03e22-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/052e22cfdd344c79634f7ec76fa03e22-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Molecular Representation Learning (MRL) has emerged as a powerful tool for drug and materials discovery in a variety of tasks such as virtual screening and inverse design. While there has been a surge of interest in advancing model-centric techniques, the influence of both data quantity and quality on molecular representations is not yet clearly understood within this field. In this paper, we delve into the neural scaling behaviors of MRL from a data-centric viewpoint, examining four key dimensions: (1) data modalities, (2) dataset splitting, (3) the role of pre-training, and (4) model capacity.Our empirical studies confirm a consistent power-law relationship between data volume and MRL performance across these dimensions. Additionally, through detailed analysis, we identify potential avenues for improving learning efficiency.To challenge these scaling laws, we adapt seven popular data pruning strategies to molecular data and benchmark their performance. Our findings underline the importance of data-centric MRL and highlight possible directions for future research.

----

## [72] FlowCam: Training Generalizable 3D Radiance Fields without Camera Poses via Pixel-Aligned Scene Flow

**Authors**: *Cameron Smith, Yilun Du, Ayush Tewari, Vincent Sitzmann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0534abc9e6db91683d82186ef0d68202-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0534abc9e6db91683d82186ef0d68202-Abstract-Conference.html)

**Abstract**:

Reconstruction of 3D neural fields from posed images has emerged as a promising method for self-supervised representation learning. The key challenge preventing the deployment of these 3D scene learners on large-scale video data is their dependence on precise camera poses from structure-from-motion, which is prohibitively expensive to run at scale. We propose a method that jointly reconstructs camera poses and 3D neural scene representations online and in a single forward pass. We estimate poses by first lifting frame-to-frame optical flow to 3D scene flow via differentiable rendering, preserving locality and shift-equivariance of the image processing backbone. SE(3) camera pose estimation is then performed via a weighted least-squares fit to the scene flow field. This formulation enables us to jointly supervise pose estimation and a generalizable neural scene representation via re-rendering the input video, and thus, train end-to-end and fully self-supervised on real-world video datasets. We demonstrate that our method performs robustly on diverse, real-world video, notably on sequences traditionally challenging to optimization-based pose estimation techniques.

----

## [73] Minimum Description Length and Generalization Guarantees for Representation Learning

**Authors**: *Milad Sefidgaran, Abdellatif Zaidi, Piotr Krasnowski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/054e9f9a286671ababa3213d6e59c1c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/054e9f9a286671ababa3213d6e59c1c2-Abstract-Conference.html)

**Abstract**:

A major challenge in designing efficient statistical supervised learning algorithms is finding representations that perform well not only on available training samples but also on unseen data. While the study of representation learning has spurred much interest, most existing such approaches are heuristic; and very little is known about theoretical generalization guarantees. For example, the information bottleneck method seeks a good generalization by finding a minimal description of the input that is maximally informative about the label variable, where minimality and informativeness are both measured by Shannon’s mutual information. In this paper, we establish a compressibility framework that allows us to derive upper bounds on the generalization error of a representation learning algorithm in terms of the ``Minimum Description Length'' (MDL) of the labels or the latent variables (representations). Rather than the mutual information between the encoder’s input and the representation, which is often believed to reflect the algorithm’s generalization capability in the related literature but in fact, falls short of doing so, our new bounds involve the "multi-letter" relative entropy between the distribution of the representations (or labels) of the training and test sets and a fixed prior. In particular, these new bounds reflect the structure of the encoder and are not vacuous for deterministic algorithms. Our compressibility approach, which is information-theoretic in nature, builds upon that of Blum-Langford for PAC-MDL bounds and introduces two essential ingredients: block-coding and lossy-compression. The latter allows our approach to subsume the so-called geometrical compressibility as a special case. To the best knowledge of the authors, the established generalization bounds are the first of their kind for Information Bottleneck type encoders and representation learning. Finally, we partly exploit the theoretical results by introducing a new data-dependent prior. Numerical simulations illustrate the advantages of well-chosen such priors over classical priors used in IB.

----

## [74] From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion

**Authors**: *Robin San Roman, Yossi Adi, Antoine Deleforge, Romain Serizel, Gabriel Synnaeve, Alexandre Défossez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/054f771d614df12fe8def8ecdbe4e8e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/054f771d614df12fe8def8ecdbe4e8e1-Abstract-Conference.html)

**Abstract**:

Deep generative models can generate high-fidelity audio conditioned on varioustypes of representations (e.g., mel-spectrograms, Mel-frequency Cepstral Coefficients(MFCC)). Recently, such models have been used to synthesize audiowaveforms conditioned on highly compressed representations. Although suchmethods produce impressive results, they are prone to generate audible artifactswhen the conditioning is flawed or imperfect. An alternative modeling approach isto use diffusion models. However, these have mainly been used as speech vocoders(i.e., conditioned on mel-spectrograms) or generating relatively low samplingrate signals. In this work, we propose a high-fidelity multi-band diffusion-basedframework that generates any type of audio modality (e.g., speech, music, environmentalsounds) from low-bitrate discrete representations. At equal bit rate,the proposed approach outperforms state-of-the-art generative techniques in termsof perceptual quality. Training and evaluation code are available on the facebookresearch/audiocraft github project. Samples are available on the followinglink (https://ai.honu.io/papers/mbd/).

----

## [75] Fixing the NTK: From Neural Network Linearizations to Exact Convex Programs

**Authors**: *Rajat Vadiraj Dwaraknath, Tolga Ergen, Mert Pilanci*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/055fc19a3ce780b96cff15ffe738c1f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/055fc19a3ce780b96cff15ffe738c1f1-Abstract-Conference.html)

**Abstract**:

Recently, theoretical analyses of deep neural networks have broadly focused on two directions: 1) Providing insight into neural network training by SGD in the limit of infinite hidden-layer width and infinitesimally small learning rate (also known as gradient flow) via the Neural Tangent Kernel (NTK), and 2) Globally optimizing the regularized training objective via cone-constrained convex reformulations of ReLU networks. The latter research direction also yielded an alternative formulation of the ReLU network, called a gated ReLU network, that is globally optimizable via efficient unconstrained convex programs. In this work, we interpret the convex program for this gated ReLU network as a Multiple Kernel Learning (MKL) model with a weighted data masking feature map and establish a connection to the NTK. Specifically, we show that for a particular choice of mask weights that do not depend on the learning targets, this kernel is equivalent to the NTK of the gated ReLU network on the training data. A consequence of this lack of dependence on the targets is that the NTK cannot perform better than the optimal MKL kernel on the training set. By using iterative reweighting, we improve the weights induced by the NTK to obtain the optimal MKL kernel which is equivalent to the solution of the exact convex reformulation of the gated ReLU network. We also provide several numerical simulations corroborating our theory. Additionally, we provide an analysis of the prediction error of the resulting optimal kernel via consistency results for the group lasso.

----

## [76] Birth of a Transformer: A Memory Viewpoint

**Authors**: *Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Hervé Jégou, Léon Bottou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0561738a239a995c8cd2ef0e50cfa4fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0561738a239a995c8cd2ef0e50cfa4fd-Abstract-Conference.html)

**Abstract**:

Large language models based on transformers have achieved great empirical successes. However, as they are deployed more widely, there is a growing need to better understand their internal mechanisms in order to make them more reliable. These models appear to store vast amounts of knowledge from their training data, and to adapt quickly to new information provided in their context or prompt. We study how transformers balance these two types of knowledge by considering a synthetic setup where tokens are generated from either global or context-specific bigram distributions. By a careful empirical analysis of the training process on a simplified two-layer transformer, we illustrate the fast learning of global bigrams and the slower development of an "induction head" mechanism for the in-context bigrams. We highlight the role of weight matrices as associative memories, provide theoretical insights on how gradients enable their learning during training, and study the role of data-distributional properties.

----

## [77] A Variational Perspective on High-Resolution ODEs

**Authors**: *Hoomaan Maskan, Konstantinos Zygalakis, Alp Yurtsever*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0569458210c88d8db2985799da830d27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0569458210c88d8db2985799da830d27-Abstract-Conference.html)

**Abstract**:

We consider unconstrained minimization of smooth convex functions. We propose a novel variational perspective using forced Euler-Lagrange equation that allows for studying high-resolution ODEs. Through this, we obtain a faster convergence rate for gradient norm minimization using Nesterov's accelerated gradient method. Additionally, we show that Nesterov's method can be interpreted as a rate-matching discretization of an appropriately chosen high-resolution ODE. Finally, using the results from the new variational perspective, we propose a stochastic method for noisy gradients. Several numerical experiments compare and illustrate our stochastic algorithm with state of the art methods.

----

## [78] What You See is What You Read? Improving Text-Image Alignment Evaluation

**Authors**: *Michal Yarom, Yonatan Bitton, Soravit Changpinyo, Roee Aharoni, Jonathan Herzig, Oran Lang, Eran Ofek, Idan Szpektor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/056e8e9c8ca9929cb6cf198952bf1dbb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/056e8e9c8ca9929cb6cf198952bf1dbb-Abstract-Conference.html)

**Abstract**:

Automatically determining whether a text and a corresponding image are semantically aligned is a significant challenge for vision-language models, with applications in generative text-to-image and image-to-text tasks. In this work, we study methods for automatic text-image alignment evaluation. We first introduce SeeTRUE: a comprehensive evaluation set, spanning multiple datasets from both text-to-image and image-to-text generation tasks, with human judgements for whether a given text-image pair is semantically aligned. We then describe two automatic methods to determine alignment: the first involving a pipeline based on question generation and visual question answering models, and the second employing an end-to-end classification approach by finetuning multimodal pretrained models. Both methods surpass prior approaches in various text-image alignment tasks, with significant improvements in challenging cases that involve complex composition or unnatural images. Finally, we demonstrate how our approaches can localize specific misalignments between an image and a given text, and how they can be used to automatically re-rank candidates in text-to-image generation.

----

## [79] On the Robustness of Mechanism Design under Total Variation Distance

**Authors**: *Anuran Makur, Marios Mertzanidis, Alexandros Psomas, Athina Terzoglou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/058983528186511a74968e88a6d0ad63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/058983528186511a74968e88a6d0ad63-Abstract-Conference.html)

**Abstract**:

We study the problem of designing mechanisms when agents' valuation functions are drawn from unknown and correlated prior distributions. In particular, we are given a prior distribution $D$, and we are interested in designing a (truthful) mechanism that has good performance for all "true distributions" that are close to $D$ in Total Variation (TV) distance. We show that DSIC and BIC mechanisms in this setting are strongly robust with respect to TV distance, for any bounded objective function $\mathcal{O}$, extending a recent result of Brustle et al. ([BCD20], EC 2020). At the heart of our result is a fundamental duality property of total variation distance. As direct applications of our result, we (i) demonstrate how to find approximately revenue-optimal and approximately BIC mechanisms for weakly dependent prior distributions; (ii) show how to find correlation-robust mechanisms when only ``noisy'' versions of marginals are accessible, extending recent results of Bei et. al. ([BGLT19], SODA 2019); (iii) prove that prophet-inequality type guarantees are preserved for correlated priors, recovering a variant of a result of D{\"u}tting and Kesselheim ([DK19], EC 2019) as a special case; (iv) give a new necessary condition for a correlated distribution to witness an infinite separation in revenue between simple and optimal mechanisms, complementing recent results of Psomas et al. ([PSCW22], NeurIPS 2022); (v) give a new condition for simple mechanisms to approximate revenue-optimal mechanisms for the case of a single agent whose type is drawn from a correlated distribution that can be captured by a Markov Random Field, complementing recent results of Cai and Oikonomou ([CO21], EC 2021).

----

## [80] M4: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics, Modalities and Models

**Authors**: *Xuhong Li, Mengnan Du, Jiamin Chen, Yekun Chai, Himabindu Lakkaraju, Haoyi Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/05957c194f4c77ac9d91e1374d2def6b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While Explainable Artificial Intelligence (XAI) techniques have been widely studied to explain predictions made by deep neural networks, the way to evaluate the faithfulness of explanation results remains challenging, due to the heterogeneity of explanations for various models and the lack of ground-truth explanations. This paper introduces an XAI benchmark named $\mathcal{M}^4$, which allows evaluating various input feature attribution methods using the same set of faithfulness metrics across multiple data modalities (images and texts) and network structures (ResNets, MobileNets, Transformers). A taxonomy for the metrics has been proposed as well. We first categorize commonly used XAI evaluation metrics into three groups based on the ground truth they require. We then implement classic and state-of-the-art feature attribution methods using InterpretDL and conduct extensive experiments to compare methods and gain insights. Extensive experiments have been conducted to provide holistic evaluations as benchmark baselines. Several interesting observations are noticed for designing attribution algorithms. The implementation of state-of-the-art explanation methods and evaluation metrics of $\mathcal{M}^4$ is publicly available at \url{https://github.com/PaddlePaddle/InterpretDL}.

----

## [81] A generative model of the hippocampal formation trained with theta driven local learning rules

**Authors**: *Tom M. George, Kimberly L. Stachenfeld, Caswell Barry, Claudia Clopath, Tomoki Fukai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05ab457c7b769f01c2973e2a5ab66ad9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05ab457c7b769f01c2973e2a5ab66ad9-Abstract-Conference.html)

**Abstract**:

Advances in generative models have recently revolutionised machine learning. Meanwhile, in neuroscience, generative models have long been thought fundamental to animal intelligence. Understanding the biological mechanisms that support these processes promises to shed light on the relationship between biological and artificial intelligence. In animals, the hippocampal formation is thought to learn and use a generative model to support its role in spatial and non-spatial memory. Here we introduce a biologically plausible model of the hippocampal formation tantamount to a Helmholtz machine that we apply to a temporal stream of inputs. A novel component of our model is that fast theta-band oscillations (5-10 Hz) gate the direction of information flow throughout the network, training it akin to a high-frequency wake-sleep algorithm. Our model accurately infers the latent state of high-dimensional sensory environments and generates realistic sensory predictions. Furthermore, it can learn to path integrate by developing a ring attractor connectivity structure matching previous theoretical proposals and flexibly transfer this structure between environments. Whereas many models trade-off biological plausibility with generality, our model captures a variety of hippocampal cognitive functions under one biologically plausible local learning rule.

----

## [82] Risk-Averse Model Uncertainty for Distributionally Robust Safe Reinforcement Learning

**Authors**: *James Queeney, Mouhacine Benosman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05b63fa06784b71aab3939004e0f0a0d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05b63fa06784b71aab3939004e0f0a0d-Abstract-Conference.html)

**Abstract**:

Many real-world domains require safe decision making in uncertain environments. In this work, we introduce a deep reinforcement learning framework for approaching this important problem. We consider a distribution over transition models, and apply a risk-averse perspective towards model uncertainty through the use of coherent distortion risk measures. We provide robustness guarantees for this framework by showing it is equivalent to a specific class of distributionally robust safe reinforcement learning problems. Unlike existing approaches to robustness in deep reinforcement learning, however, our formulation does not involve minimax optimization. This leads to an efficient, model-free implementation of our approach that only requires standard data collection from a single training environment. In experiments on continuous control tasks with safety constraints, we demonstrate that our framework produces robust performance and safety at deployment time across a range of perturbed test environments.

----

## [83] Optimal approximation using complex-valued neural networks

**Authors**: *Paul Geuchen, Felix Voigtländer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05b69cc4c8ff6e24c5de1ecd27223d37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05b69cc4c8ff6e24c5de1ecd27223d37-Abstract-Conference.html)

**Abstract**:

Complex-valued neural networks (CVNNs) have recently shown promising empirical success, for instance for increasing the stability of recurrent neural networks and for improving the performance in tasks with complex-valued inputs, such as MRI fingerprinting. While the overwhelming success of Deep Learning in the real-valued case is supported by a growing mathematical foundation, such a foundation is still largely lacking in the complex-valued case. We thus analyze the expressivity of CVNNs by studying their approximation properties. Our results yield the first quantitative approximation bounds for CVNNs that apply to a wide class of activation functions including the popular modReLU and complex cardioid activation functions. Precisely, our results apply to any activation function that is smooth but not polyharmonic on some non-empty open set; this is the natural generalization of the class of smooth and non-polynomial activation functions to the complex setting. Our main result shows that the approximation error scales as $m^{-k/(2n)}$ for $m \to \infty$ where $m$ is the number of neurons, $k$ the smoothness of the target function and $n$ is the (complex) input dimension. Under a natural continuity assumption, we show that this rate is optimal; we further discuss the optimality when dropping this assumption. Moreover, we prove that the problem of approximating $C^k$-functions using continuous approximation methods unavoidably suffers from the curse of dimensionality.

----

## [84] BayesDAG: Gradient-Based Posterior Inference for Causal Discovery

**Authors**: *Yashas Annadani, Nick Pawlowski, Joel Jennings, Stefan Bauer, Cheng Zhang, Wenbo Gong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05cf28e3d3c9a179d789c55270fe6f72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05cf28e3d3c9a179d789c55270fe6f72-Abstract-Conference.html)

**Abstract**:

Bayesian causal discovery aims to infer the posterior distribution over causal models from observed data, quantifying epistemic uncertainty and benefiting downstream tasks. However, computational challenges arise due to joint inference over combinatorial space of Directed Acyclic Graphs (DAGs) and nonlinear functions. Despite recent progress towards efficient posterior inference over DAGs,  existing methods are either limited to variational inference on node permutation matrices for linear causal models, leading to compromised inference accuracy, or continuous relaxation of adjacency matrices constrained by a DAG regularizer, which cannot ensure resulting graphs are DAGs. In this work, we introduce a scalable Bayesian causal discovery framework based on a combination of stochastic gradient Markov Chain Monte Carlo (SG-MCMC) and Variational Inference (VI) that overcomes these limitations. Our approach directly samples DAGs from the posterior without requiring any DAG regularization, simultaneously draws function parameter samples and is applicable to both linear and nonlinear causal models. To enable our approach, we derive a novel equivalence to the permutation-based DAG learning, which opens up possibilities of using any relaxed gradient estimator defined over permutations. To our knowledge, this is the first framework applying gradient-based MCMC sampling for causal discovery. Empirical evaluation on synthetic and real-world datasets demonstrate our approach's effectiveness compared to state-of-the-art baselines.

----

## [85] Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces

**Authors**: *Leonard Papenmeier, Luigi Nardi, Matthias Poloczek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05d2175de7ee637588d1b5ced8b15b32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05d2175de7ee637588d1b5ced8b15b32-Abstract-Conference.html)

**Abstract**:

Impactful applications such as materials discovery, hardware design, neural architecture search, or portfolio optimization require optimizing high-dimensional black-box functions with mixed and combinatorial input spaces.While Bayesian optimization has recently made significant progress in solving such problems, an in-depth analysis reveals that the current state-of-the-art methods are not reliable. Their performances degrade substantially when the unknown optima of the function do not have a certain structure. To fill the need for a reliable algorithm for combinatorial and mixed spaces, this paper proposes Bounce that relies on a novel map of various variable types into nested embeddings of increasing dimensionality.Comprehensive experiments show that Bounce reliably achieves and often even improves upon state-of-the-art performance on a variety of high-dimensional problems.

----

## [86] Uniform-in-Time Wasserstein Stability Bounds for (Noisy) Stochastic Gradient Descent

**Authors**: *Lingjiong Zhu, Mert Gürbüzbalaban, Anant Raj, Umut Simsekli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05d6b5b6901fb57d2c287e1d3ce6d63c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05d6b5b6901fb57d2c287e1d3ce6d63c-Abstract-Conference.html)

**Abstract**:

Algorithmic stability is an important notion that has proven powerful for deriving generalization bounds for practical algorithms. The last decade has witnessed an increasing number of stability bounds for different algorithms applied on different classes of loss functions. While these bounds have illuminated various properties of optimization algorithms, the analysis of each case typically required a different proof technique with significantly different mathematical tools. In this study, we make a novel connection between learning theory and applied probability and introduce a unified guideline for proving Wasserstein stability bounds for stochastic optimization algorithms. We illustrate our approach on stochastic gradient descent (SGD) and we obtain time-uniform  stability bounds (i.e., the bound does not increase with the number of iterations) for strongly convex losses and non-convex losses with additive noise, where we recover similar results to the prior art or extend them to more general cases by using a single proof technique. Our approach is flexible and can be generalizable to other popular optimizers, as it mainly requires developing Lyapunov functions, which are often readily available in the literature. It also illustrates that ergodicity is an important component for obtaining time-uniform bounds --  which might not be achieved for convex or non-convex losses unless additional noise is injected to the iterates. Finally, we slightly stretch our analysis technique and prove time-uniform bounds for SGD under convex and non-convex losses (without additional additive noise), which, to our knowledge, is novel.

----

## [87] Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation

**Authors**: *Haonan Wang, Xiaomeng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05dc08730e32441edff52b0fa6caab5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05dc08730e32441edff52b0fa6caab5f-Abstract-Conference.html)

**Abstract**:

Volume-wise labeling in 3D medical images is a time-consuming task that requires expertise. As a result, there is growing interest in using semi-supervised learning (SSL) techniques to train models with limited labeled data. However, the challenges and practical applications extend beyond SSL to settings such as unsupervised domain adaptation (UDA) and semi-supervised domain generalization (SemiDG). This work aims to develop a generic SSL framework that can handle all three settings. We identify two main obstacles to achieving this goal in the existing SSL framework: 1) the weakness of capturing distribution-invariant features; and 2) the tendency for unlabeled data to be overwhelmed by labeled data, leading to over-fitting to the labeled data during training. To address these issues, we propose an Aggregating & Decoupling framework. The aggregating part consists of a Diffusion encoder that constructs a "common knowledge set" by extracting distribution-invariant features from aggregated information from multiple distributions/domains. The decoupling part consists of three decoders that decouple the training process with labeled and unlabeled data, thus avoiding over-fitting to labeled data, specific domains and classes. We evaluate our proposed framework on four benchmark datasets for SSL, Class-imbalanced SSL, UDA and SemiDG. The results showcase notable improvements compared to state-of-the-art methods across all four settings, indicating the potential of our framework to tackle more challenging SSL scenarios. Code and models are available at: https://github.com/xmed-lab/GenericSSL.

----

## [88] Stochastic Distributed Optimization under Average Second-order Similarity: Algorithms and Analysis

**Authors**: *Dachao Lin, Yuze Han, Haishan Ye, Zhihua Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05e552739c2629f3324c1063a382b4bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05e552739c2629f3324c1063a382b4bd-Abstract-Conference.html)

**Abstract**:

We study finite-sum distributed optimization problems involving a master node and $n-1$ local nodes under the popular $\delta$-similarity and $\mu$-strong convexity conditions. We propose two new algorithms, SVRS and AccSVRS, motivated by previous works. The non-accelerated SVRS method combines the techniques of gradient sliding and variance reduction and achieves a better communication complexity of $\tilde{\mathcal{O}}(n {+} \sqrt{n}\delta/\mu)$ compared to existing non-accelerated algorithms. Applying the framework proposed in Katyusha X, we also develop a directly accelerated version named AccSVRS with the $\tilde{\mathcal{O}}(n {+} n^{3/4}\sqrt{\delta/\mu})$ communication complexity. In contrast to existing results, our complexity bounds are entirely smoothness-free and exhibit superiority in ill-conditioned cases. Furthermore, we establish a nearly matched lower bound to verify the tightness of our AccSVRS method.

----

## [89] PolyDiffuse: Polygonal Shape Reconstruction via Guided Set Diffusion Models

**Authors**: *Jiacheng Chen, Ruizhi Deng, Yasutaka Furukawa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05f0e2fa003602db2d98ca72b79dec51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05f0e2fa003602db2d98ca72b79dec51-Abstract-Conference.html)

**Abstract**:

This paper presents \textit{PolyDiffuse}, a novel structured reconstruction algorithm that transforms visual sensor data into polygonal shapes with Diffusion Models (DM), an emerging machinery amid exploding generative AI, while formulating reconstruction as a generation process conditioned on sensor data. The task of structured reconstruction poses two fundamental challenges to DM: 1) A structured geometry is a ''set'' (e.g., a set of polygons for a floorplan geometry), where a sample of $N$ elements has $N!$ different but equivalent representations, making the denoising highly ambiguous; and 2) A ''reconstruction'' task has a single solution, where an initial noise needs to be chosen carefully, while any initial noise works for a generation task.Our technical contribution is the introduction of a Guided Set Diffusion Model where 1) the forward diffusion process learns \textit{guidance networks} to control noise injection so that one representation of a sample remains distinct from its other permutation variants, thus resolving denoising ambiguity; and 2) the reverse denoising process reconstructs polygonal shapes, initialized and directed by the guidance networks, as a conditional generation process subject to the sensor data.We have evaluated our approach for reconstructing two types of polygonal shapes: floorplan as a set of polygons and HD map for autonomous cars as a set of polylines.Through extensive experiments on standard benchmarks, we demonstrate that PolyDiffuse significantly advances the current state of the art and enables broader practical applications. The code and data are available on our project page: https://poly-diffuse.github.io.

----

## [90] Can You Rely on Your Model Evaluation? Improving Model Evaluation with Synthetic Test Data

**Authors**: *Boris van Breugel, Nabeel Seedat, Fergus Imrie, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05fb0f4e645cad23e0ab59d6b9901428-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05fb0f4e645cad23e0ab59d6b9901428-Abstract-Conference.html)

**Abstract**:

Evaluating the performance of machine learning models on diverse and underrepresented subgroups is essential for ensuring fairness and reliability in real-world applications. However, accurately assessing model performance becomes challenging due to two main issues: (1) a scarcity of test data, especially for small subgroups, and (2) possible distributional shifts in the model's deployment setting, which may not align with the available test data.  In this work, we introduce 3S Testing, a deep generative modeling framework to facilitate model evaluation by generating synthetic test sets for small subgroups and simulating distributional shifts. Our experiments demonstrate that 3S-Testing outperforms traditional baselines---including real test data alone---in estimating model performance on minority subgroups and under plausible distributional shifts. In addition, 3S offers intervals around its performance estimates, exhibiting superior coverage of the ground truth compared to existing approaches.  Overall, these results raise the question of whether we need a paradigm shift away from limited real test data towards synthetic test data.

----

## [91] Rethinking the Backward Propagation for Adversarial Transferability

**Authors**: *Xiaosen Wang, Kangheng Tong, Kun He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05fe0c633ae41756540dba2a99a36306-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/05fe0c633ae41756540dba2a99a36306-Abstract-Conference.html)

**Abstract**:

Transfer-based attacks generate adversarial examples on the surrogate model, which can mislead other black-box models without access, making it promising to attack real-world applications. Recently, several works have been proposed to boost adversarial transferability, in which the surrogate model is usually overlooked. In this work, we identify that non-linear layers (e.g., ReLU, max-pooling, etc.) truncate the gradient during backward propagation, making the gradient w.r.t. input image imprecise to the loss function. We hypothesize and empirically validate that such truncation undermines the transferability of adversarial examples. Based on these findings, we propose a novel method called Backward Propagation Attack (BPA) to increase the relevance between the gradient w.r.t. input image and loss function so as to generate adversarial examples with higher transferability. Specifically, BPA adopts a non-monotonic function as the derivative of ReLU and incorporates softmax with temperature to smooth the derivative of max-pooling, thereby mitigating the information loss during the backward propagation of gradients. Empirical results on the ImageNet dataset demonstrate that not only does our method substantially boost the adversarial transferability, but it is also general to existing transfer-based attacks. Code is available at https://github.com/Trustworthy-AI-Group/RPA.

----

## [92] Bullying10K: A Large-Scale Neuromorphic Dataset towards Privacy-Preserving Bullying Recognition

**Authors**: *Yiting Dong, Yang Li, Dongcheng Zhao, Guobin Shen, Yi Zeng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/05ffe69463062b7f9fb506c8351ffdd7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/05ffe69463062b7f9fb506c8351ffdd7-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The prevalence of violence in daily life poses significant threats to individuals' physical and mental well-being. Using surveillance cameras in public spaces has proven effective in proactively deterring and preventing such incidents. However, concerns regarding privacy invasion have emerged due to their widespread deployment.To address the problem, we leverage Dynamic Vision Sensors (DVS) cameras to detect violent incidents and preserve privacy since it captures pixel brightness variations instead of static imagery. We introduce the Bullying10K dataset, encompassing various actions, complex movements, and occlusions from real-life scenarios. It provides three benchmarks for evaluating different tasks: action recognition, temporal action localization, and pose estimation. With 10,000 event segments, totaling 12 billion events and 255 GB of data, Bullying10K contributes significantly by balancing violence detection and personal privacy persevering. And it also poses a challenge to the neuromorphic dataset. It will serve as a valuable resource for training and developing privacy-protecting video systems. The Bullying10K opens new possibilities for innovative approaches in these domains.

----

## [93] Compression with Bayesian Implicit Neural Representations

**Authors**: *Zongyu Guo, Gergely Flamich, Jiajun He, Zhibo Chen, José Miguel Hernández-Lobato*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/060b2af0081a460f7f466f7f174d9052-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/060b2af0081a460f7f466f7f174d9052-Abstract-Conference.html)

**Abstract**:

Many common types of data can be represented as functions that map coordinates to signal values, such as pixel locations to RGB values in the case of an image. Based on this view, data can be compressed by overfitting a compact neural network to its functional representation and then encoding the network weights. However, most current solutions for this are inefficient, as quantization to low-bit precision substantially degrades the reconstruction quality. To address this issue, we propose overfitting variational Bayesian neural networks to the data and compressing an approximate posterior weight sample using relative entropy coding instead of quantizing and entropy coding it. This strategy enables direct optimization of the rate-distortion performance by minimizing the $\beta$-ELBO, and target different rate-distortion trade-offs for a given network architecture by adjusting $\beta$. Moreover, we introduce an iterative algorithm for learning prior weight distributions and employ a progressive refinement process for the variational posterior that significantly enhances performance. Experiments show that our method achieves strong performance on image and audio compression while retaining simplicity.

----

## [94] Towards Unbounded Machine Unlearning

**Authors**: *Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, Eleni Triantafillou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/062d711fb777322e2152435459e6e9d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/062d711fb777322e2152435459e6e9d9-Abstract-Conference.html)

**Abstract**:

Deep machine unlearning is the problem of 'removing' from a trained neural network a subset of its training set. This problem is very timely and has many applications, including the key tasks of removing biases (RB), resolving confusion (RC) (caused by mislabelled data in trained models), as well as allowing users to exercise their 'right to be forgotten' to protect User Privacy (UP). This paper is the first, to our knowledge, to study unlearning for different applications (RB, RC, UP), with the view that each has its own desiderata, definitions for 'forgetting' and associated metrics for forget quality. For UP, we propose a novel adaptation of a strong Membership Inference Attack for unlearning. We also propose SCRUB, a novel unlearning algorithm, which is the only method that is consistently a top performer for forget quality across the different application-dependent metrics for RB, RC, and UP. At the same time, SCRUB is also consistently a top performer on metrics that measure model utility (i.e. accuracy on retained data and generalization), and is more efficient than previous work. The above are substantiated through a comprehensive empirical evaluation against previous state-of-the-art.

----

## [95] Collaborative Learning via Prediction Consensus

**Authors**: *Dongyang Fan, Celestine Mendler-Dünner, Martin Jaggi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/065e259a1d2d955e63b99aac6a3a3081-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/065e259a1d2d955e63b99aac6a3a3081-Abstract-Conference.html)

**Abstract**:

We consider a collaborative learning setting where the goal of each agent is to improve their own model by leveraging the expertise of collaborators, in addition to their own training data. To facilitate the exchange of expertise among agents, we propose a distillation-based method leveraging shared unlabeled auxiliary data, which is pseudo-labeled by the collective. Central to our method is a trust weighting scheme that serves to adaptively weigh the influence of each collaborator on the pseudo-labels until a consensus on how to label the auxiliary data is reached. We demonstrate empirically that our collaboration scheme is able to significantly boost individual models’ performance in the target domain from which the auxiliary data is sampled. At the same time, it can provably mitigate the negative impact of bad models on the collective. By design, our method adeptly accommodates heterogeneity in model architectures and substantially reduces communication overhead compared to typical collaborative learning methods.

----

## [96] Identification of Nonlinear Latent Hierarchical Models

**Authors**: *Lingjing Kong, Biwei Huang, Feng Xie, Eric P. Xing, Yuejie Chi, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/065ef23a944b3995de7dd4a3e203d133-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/065ef23a944b3995de7dd4a3e203d133-Abstract-Conference.html)

**Abstract**:

Identifying latent variables and causal structures from observational data is essential to many real-world applications involving biological data, medical data, and unstructured data such as images and languages. However, this task can be highly challenging, especially when observed variables are generated by causally related latent variables and the relationships are nonlinear. In this work, we investigate the identification problem for nonlinear latent hierarchical causal models in which observed variables are generated by a set of causally related latent variables, and some latent variables may not have observed children. We show that the identifiability of causal structures and latent variables (up to invertible transformations) can be achieved under mild assumptions: on causal structures, we allow for multiple paths between any pair of variables in the graph, which relaxes latent tree assumptions in prior work; on structural functions, we permit general nonlinearity and multi-dimensional continuous variables, alleviating existing work's parametric assumptions. Specifically, we first develop an identification criterion in the form of novel identifiability guarantees for an elementary latent variable model. Leveraging this criterion, we show that both causal structures and latent variables of the hierarchical model can be identified asymptotically by explicitly constructing an estimation procedure. To the best of our knowledge, our work is the first to establish identifiability guarantees for both causal structures and latent variables in nonlinear latent hierarchical models.

----

## [97] Sample Efficient Reinforcement Learning in Mixed Systems through Augmented Samples and Its Applications to Queueing Networks

**Authors**: *Honghao Wei, Xin Liu, Weina Wang, Lei Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0663a39baab211328fc865f91abc75ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0663a39baab211328fc865f91abc75ab-Abstract-Conference.html)

**Abstract**:

This paper considers a class of reinforcement learning problems, which involve systems with two types of states: stochastic and pseudo-stochastic. In such systems, stochastic states follow a stochastic transition kernel while the transitions of pseudo-stochastic states are deterministic {\em given} the stochastic states/transitions. We refer to such systems as mixed systems, which are widely used in various applications, including Manufacturing systems, communication networks, and queueing networks. We propose a sample-efficient RL method that accelerates learning by generating augmented data samples. The proposed algorithm is data-driven (model-free), but it learns the policy from data samples from both real and augmented samples. This method significantly improves learning by reducing the sample complexity such that the dataset only needs to have sufficient coverage of the stochastic states. We analyze the sample complexity of the proposed method under Fitted Q Iteration (FQI) and demonstrate that the optimality gap decreases as  $O\left(\sqrt{\frac{1}{n}}+\sqrt{\frac{1}{m}}\right),$ where $n$ represents the number of real samples, and $m$ is the number of augmented samples per real sample. It is important to note that without augmented samples, the optimality gap is $O(1)$ due to the insufficient data coverage of the pseudo-stochastic states. Our experimental results on multiple queueing network applications confirm that the proposed method indeed significantly accelerates both deep Q-learning and deep policy gradient.

----

## [98] Temporal Graph Benchmark for Machine Learning on Temporal Graphs

**Authors**: *Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael M. Bronstein, Guillaume Rabusseau, Reihaneh Rabbany*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/066b98e63313162f6562b35962671288-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/066b98e63313162f6562b35962671288-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present the Temporal Graph Benchmark (TGB), a collection of challenging and diverse benchmark datasets for realistic, reproducible, and robust evaluation of machine learning models on temporal graphs. TGB datasets are of large scale, spanning years in duration, incorporate both node and edge-level prediction tasks and cover a diverse set of domains including social, trade, transaction, and transportation networks. For both tasks, we design evaluation protocols based on realistic use-cases. We extensively benchmark each dataset and find that the performance of common models can vary drastically across datasets. In addition, on dynamic node property prediction tasks, we show that simple methods often achieve superior performance compared to existing temporal graph models. We believe that these findings open up opportunities for future research on temporal graphs. Finally, TGB provides an automated machine learning pipeline for reproducible and accessible temporal graph research, including data loading, experiment setup and performance evaluation. TGB will be maintained and updated on a regular basis and welcomes community feedback. TGB datasets, data loaders, example codes, evaluation setup, and leaderboards are publicly available at https://tgb.complexdatalab.com/.

----

## [99] Navigating Data Heterogeneity in Federated Learning: A Semi-Supervised Approach for Object Detection

**Authors**: *Taehyeon Kim, Eric Lin, Junu Lee, Christian Lau, Vaikkunth Mugunthan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/066e4dbfeccb5dc2851acd5eca584937-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/066e4dbfeccb5dc2851acd5eca584937-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) has emerged as a potent framework for training models across distributed data sources while maintaining data privacy. Nevertheless, it faces challenges with limited high-quality labels and non-IID client data, particularly in applications like autonomous driving. To address these hurdles, we navigate the uncharted waters of Semi-Supervised Federated Object Detection (SSFOD). We present a pioneering SSFOD framework, designed for scenarios where labeled data reside only at the server while clients possess unlabeled data. Notably, our method represents the inaugural implementation of SSFOD for clients with 0% labeled non-IID data, a stark contrast to previous studies that maintain some subset of labels at each client. We propose FedSTO, a two-stage strategy encompassing Selective Training followed by Orthogonally enhanced full-parameter training, to effectively address data shift (e.g. weather conditions) between server and clients. Our contributions include selectively refining the backbone of the detector to avert overfitting, orthogonality regularization to boost representation divergence, and local EMA-driven pseudo label assignment to yield high-quality pseudo labels. Extensive validation on prominent autonomous driving datasets (BDD100K, Cityscapes, and SODA10M) attests to the efficacy of our approach, demonstrating state-of-the-art results. Remarkably, FedSTO, using just 20-30% of labels, performs nearly as well as fully-supervised centralized training methods.

----

## [100] On the Generalization Properties of Diffusion Models

**Authors**: *Puheng Li, Zhong Li, Huishuai Zhang, Jiang Bian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06abed94583030dd50abe6767bd643b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06abed94583030dd50abe6767bd643b1-Abstract-Conference.html)

**Abstract**:

Diffusion models are a class of generative models that serve to establish a stochastic transport map between an empirically observed, yet unknown, target distribution and a known prior. Despite their remarkable success in real-world applications, a theoretical understanding of their generalization capabilities remains underdeveloped. This work embarks on a comprehensive theoretical exploration of the generalization attributes of diffusion models. We establish the theoretical estimates of the generalization gap that evolves in tandem with the training dynamics of score-based diffusion models, suggesting a polynomially small generalization error ($O(n^{-2/5}+m^{-4/5})$) on both the sample size $n$ and the model capacity $m$, evading the curse of dimensionality (i.e., independent of the data dimension) when *early-stopped*. Furthermore, we extend our quantitative analysis to a *data-dependent* scenario, wherein target distributions are portrayed as a succession of densities with progressively increasing distances between modes. This precisely elucidates the *adverse* effect of "*modes shift*'' in ground truths on the model generalization. Furthermore, these estimates are not solely theoretical constructs but have also been confirmed through numerical simulations. Our findings contribute to the rigorous understanding of diffusion models' generalization properties and provide insights that may guide practical applications.

----

## [101] Regularized Behavior Cloning for Blocking the Leakage of Past Action Information

**Authors**: *Seokin Seo, HyeongJoo Hwang, Hongseok Yang, Kee-Eung Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06b71ad997f7e3e4b2e2f2ea12e5a759-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06b71ad997f7e3e4b2e2f2ea12e5a759-Abstract-Conference.html)

**Abstract**:

For partially observable environments, imitation learning with observation histories (ILOH) assumes that control-relevant information is sufficiently captured in the observation histories for imitating the expert actions. In the offline setting wherethe agent is required to learn to imitate without interaction with the environment, behavior cloning (BC) has been shown to be a simple yet effective method for imitation learning. However, when the information about the actions executed in the past timesteps leaks into the observation histories, ILOH via BC often ends up imitating its own past actions. In this paper, we address this catastrophic failure by proposing a principled regularization for BC, which we name Past Action Leakage Regularization (PALR). The main idea behind our approach is to leverage the classical notion of conditional independence to mitigate the leakage. We compare different instances of our framework with natural choices of conditional independence metric and its estimator. The result of our comparison advocates the use of a particular kernel-based estimator for the conditional independence metric. We conduct an extensive set of experiments on benchmark datasets in order to assess the effectiveness of our regularization method. The experimental results show that our method significantly outperforms prior related approaches, highlighting its potential to successfully imitate expert actions when the past action information leaks into the observation histories.

----

## [102] The Distortion of Binomial Voting Defies Expectation

**Authors**: *Yannai A. Gonczarowski, Gregory Kehne, Ariel D. Procaccia, Ben Schiffer, Shirley Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06cb881ec90a657a8f949a62f1b4ee5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06cb881ec90a657a8f949a62f1b4ee5f-Abstract-Conference.html)

**Abstract**:

In computational social choice, the distortion of a voting rule quantifies the degree to which the rule overcomes limited preference information to select a socially desirable outcome. This concept has been investigated extensively, but only through a worst-case lens. Instead, we study the expected distortion of voting rules with respect to an underlying distribution over voter utilities. Our main contribution is the design and analysis of a novel and intuitive rule, binomial voting, which provides strong distribution-independent guarantees for both expected distortion and expected welfare.

----

## [103] UP-DP: Unsupervised Prompt Learning for Data Pre-Selection with Vision-Language Models

**Authors**: *Xin Li, Sima Behpour, Thang Long Doan, Wenbin He, Liang Gou, Liu Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06d5f1fe6509b001e6d4e0ec1afd83dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06d5f1fe6509b001e6d4e0ec1afd83dd-Abstract-Conference.html)

**Abstract**:

In this study, we investigate the task of data pre-selection, which aims to select instances for labeling from an unlabeled dataset through a single pass, thereby optimizing performance for undefined downstream tasks with a limited annotation budget. Previous approaches to data pre-selection relied solely on visual features extracted from foundation models, such as CLIP and BLIP-2, but largely ignored the powerfulness of text features. In this work, we argue that, with proper design, the joint feature space of both vision and text can yield a better representation for data pre-selection. To this end, we introduce UP-DP, a simple yet effective unsupervised prompt learning approach that adapts vision-language models, like BLIP-2, for data pre-selection. Specifically, with the BLIP-2 parameters frozen, we train text prompts to extract the joint features with improved representation, ensuring a diverse cluster structure that covers the entire dataset. We extensively compare our method with the state-of-the-art using seven benchmark datasets in different settings, achieving up to a performance gain of 20\%. Interestingly, the prompts learned from one dataset demonstrate significant generalizability and can be applied directly to enhance the feature extraction of BLIP-2 from other datasets. To the best of our knowledge, UP-DP is the first work to incorporate unsupervised prompt learning in a vision-language model for data pre-selection.

----

## [104] Optimistic Rates for Multi-Task Representation Learning

**Authors**: *Austin Watkins, Enayat Ullah, Thanh Nguyen-Tang, Raman Arora*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06e3c330d140f3a25671acf2dc2d6357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06e3c330d140f3a25671acf2dc2d6357-Abstract-Conference.html)

**Abstract**:

We study the problem of transfer learning via Multi-Task Representation Learning (MTRL), wherein multiple source tasks are used to learn a good common representation, and a predictor is trained on top of it for the target task. Under standard regularity assumptions on the loss function and task diversity, we provide new statistical rates on the excess risk of the target task, which demonstrate the benefit of representation learning. Importantly, our rates are optimistic, i.e., they interpolate between the standard $O(m^{-1/2})$ rate and the fast $O(m^{-1})$ rate, depending on the difficulty of the learning task, where $m$ is the number of samples for the target task. Besides the main result, we make several new contributions, including giving optimistic rates for excess risk of source tasks (multi-task learning (MTL)), a local Rademacher complexity theorem for MTRL and MTL, as well as a chain rule for local Rademacher complexity for composite predictor classes.

----

## [105] Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

**Authors**: *Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M. Alabdulmohsin, Avital Oliver, Piotr Padlewski, Alexey A. Gritsenko, Mario Lucic, Neil Houlsby*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html)

**Abstract**:

The ubiquitous and demonstrably suboptimal choice of resizing images to a fixed resolution before processing them with computer vision models has not yet been successfully challenged. However, models such as the Vision Transformer (ViT) offer flexible sequence-based modeling, and hence varying input sequence lengths.  We take advantage of this with NaViT (Native Resolution ViT) which uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios. Alongside flexible model usage, we demonstrate improved training efficiency for large-scale supervised and contrastive image-text pretraining.NaViT can be efficiently transferred to standard tasks such as image and video classification, object detection, and semantic segmentation and leads to improved results on robustness and fairness benchmarks. At inference time, the input resolution flexibility can be used to smoothly navigate the test-time cost-performance trade-off. We believe that NaViTmarks a departure from the standard, CNN-designed, input and modelling pipeline used by most computer vision models, and represents a promising direction for ViTs.

----

## [106] The Benefits of Being Distributional: Small-Loss Bounds for Reinforcement Learning

**Authors**: *Kaiwen Wang, Kevin Zhou, Runzhe Wu, Nathan Kallus, Wen Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06fc38f5c21ae66ef955e28b7a78ece5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06fc38f5c21ae66ef955e28b7a78ece5-Abstract-Conference.html)

**Abstract**:

While distributional reinforcement learning (DistRL) has been empirically effective, the question of when and why it is better than vanilla, non-distributional RL has remained unanswered.This paper explains the benefits of DistRL through the lens of small-loss bounds, which are instance-dependent bounds that scale with optimal achievable cost.Particularly, our bounds converge much faster than those from non-distributional approaches if the optimal cost is small.As warmup, we propose a distributional contextual bandit (DistCB) algorithm, which we show enjoys small-loss regret bounds and empirically outperforms the state-of-the-art on three real-world tasks.In online RL, we propose a DistRL algorithm that constructs confidence sets using maximum likelihood estimation. We prove that our algorithm enjoys novel small-loss PAC bounds in low-rank MDPs.As part of our analysis, we introduce the $\ell_1$ distributional eluder dimension which may be of independent interest. Then, in offline RL, we show that pessimistic DistRL enjoys small-loss PAC bounds that are novel to the offline setting and are more robust to bad single-policy coverage.

----

## [107] Honesty Is the Best Policy: Defining and Mitigating AI Deception

**Authors**: *Francis Ward, Francesca Toni, Francesco Belardinelli, Tom Everitt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/06fc7ae4a11a7eb5e20fe018db6c036f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/06fc7ae4a11a7eb5e20fe018db6c036f-Abstract-Conference.html)

**Abstract**:

Deceptive agents are a challenge for the safety, trustworthiness, and cooperation of AI systems. We focus on the problem that agents might deceive in order to achieve their goals (for instance, in our experiments with language models, the goal of being evaluated as truthful).There are a number of existing definitions of deception in the literature on game theory and symbolic AI, but there is no overarching theory of deception for learning agents in games. We introduce a formaldefinition of deception in structural causal games, grounded in the philosophyliterature, and applicable to real-world machine learning systems.Several examples and results illustrate that our formal definition aligns with the philosophical and commonsense meaning of deception.Our main technical result is to provide graphical criteria for deception. We show, experimentally, that these results can be used to mitigate deception in reinforcement learning agents and language models.

----

## [108] Improving *day-ahead* Solar Irradiance Time Series Forecasting by Leveraging Spatio-Temporal Context

**Authors**: *Oussama Boussif, Ghait Boukachab, Dan Assouline, Stefano Massaroli, Tianle Yuan, Loubna Benabbou, Yoshua Bengio*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/070a57c5ef1e58cc90201b11d369b3c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/070a57c5ef1e58cc90201b11d369b3c2-Abstract-Conference.html)

**Abstract**:

Solar power harbors immense potential in mitigating climate change by substantially reducing CO$_{2}$ emissions. Nonetheless, the inherent variability of solar irradiance poses a significant challenge for seamlessly integrating solar power into the electrical grid. While the majority of prior research has centered on employing purely time series-based methodologies for solar forecasting, only a limited number of studies have taken into account factors such as cloud cover or the surrounding physical context.In this paper, we put forth a deep learning architecture designed to harness spatio-temporal context using satellite data, to attain highly accurate day-ahead time-series forecasting for any given station, with a particular emphasis on forecasting Global Horizontal Irradiance (GHI). We also suggest a methodology to extract a distribution for each time step prediction, which can serve as a very valuable measure of uncertainty attached to the forecast. When evaluating models, we propose a testing scheme in which we separate particularly difficult examples from easy ones, in order to capture the model performances in crucial situations, which in the case of this study are the days suffering from varying cloudy conditions. Furthermore, we present a new multi-modal dataset gathering satellite imagery over a large zone and time series for solar irradiance and other related physical variables from multiple geographically diverse solar stations. Our approach exhibits robust performance in solar irradiance forecasting, including zero-shot generalization tests at unobserved solar stations, and holds great promise in promoting the effective integration of solar power into the grid.

----

## [109] Uncovering and Quantifying Social Biases in Code Generation

**Authors**: *Yan Liu, Xiaokang Chen, Yan Gao, Zhe Su, Fengji Zhang, Daoguang Zan, Jian-Guang Lou, Pin-Yu Chen, Tsung-Yi Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/071a637d41ea290ac4360818a8323f33-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/071a637d41ea290ac4360818a8323f33-Abstract-Conference.html)

**Abstract**:

With the popularity of automatic code generation tools, such as Copilot, the study of the potential hazards of these tools is gaining importance. In this work, we explore the social bias problem in pre-trained code generation models. We propose a new paradigm to construct code prompts and successfully uncover social biases in code generation models. To quantify the severity of social biases in generated code, we develop a dataset along with three metrics to evaluate the overall social bias and fine-grained unfairness across different demographics. Experimental results on three pre-trained code generation models (Codex, InCoder, and CodeGen) with varying sizes, reveal severe social biases. Moreover, we conduct analysis to provide useful insights for further choice of code generation models with low social bias.

----

## [110] A Bounded Ability Estimation for Computerized Adaptive Testing

**Authors**: *Yan Zhuang, Qi Liu, Guanhao Zhao, Zhenya Huang, Weizhe Huang, Zachary A. Pardos, Enhong Chen, Jinze Wu, Xin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0730b81dbc16cce7e85b519cb7fe5a8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0730b81dbc16cce7e85b519cb7fe5a8d-Abstract-Conference.html)

**Abstract**:

Computerized adaptive testing (CAT), as a tool that can efficiently measure student's ability, has been widely used in various standardized tests (e.g., GMAT and GRE). The adaptivity of CAT refers to the selection of the most informative questions for each student, reducing test length. Existing CAT methods do not explicitly target ability estimation accuracy since there is no student's true ability as ground truth; therefore, these methods cannot be guaranteed to make the estimate converge to the true with such limited responses. In this paper, we analyze the statistical properties of estimation and find a theoretical approximation of the true ability: the ability estimated by full responses to question bank. Based on this, a Bounded Ability Estimation framework for CAT (BECAT) is proposed in a data-summary manner, which selects a question subset that closely matches the gradient of the full responses. Thus, we develop an expected gradient difference approximation to design a simple greedy selection algorithm, and show the rigorous theoretical and error upper-bound guarantees of its ability estimate. Experiments on both real-world and synthetic datasets, show that it can reach the same estimation accuracy using 15\% less questions on average, significantly reducing test length.

----

## [111] ForecastPFN: Synthetically-Trained Zero-Shot Forecasting

**Authors**: *Samuel Dooley, Gurnoor Singh Khurana, Chirag Mohapatra, Siddartha V. Naidu, Colin White*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0731f0e65559059eb9cd9d6f44ce2dd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0731f0e65559059eb9cd9d6f44ce2dd8-Abstract-Conference.html)

**Abstract**:

The vast majority of time-series forecasting approaches require a substantial training dataset. However, many real-life forecasting applications have very little initial observations, sometimes just 40 or fewer. Thus, the applicability of most forecasting methods is restricted in data-sparse commercial applications. While there is recent work in the setting of very limited initial data (so-called `zero-shot' forecasting), its performance is inconsistent depending on the data used for pretraining. In this work, we take a different approach and devise ForecastPFN, the first zero-shot forecasting model trained purely on a novel synthetic data distribution. ForecastPFN is a prior-data fitted network, trained to approximate Bayesian inference, which can make predictions on a new time series dataset in a single forward pass. Through extensive experiments, we show that zero-shot predictions made by ForecastPFN are more accurate and faster compared to state-of-the-art forecasting methods, even when the other methods are allowed to train on hundreds of additional in-distribution data points.

----

## [112] Exact Bayesian Inference on Discrete Models via Probability Generating Functions: A Probabilistic Programming Approach

**Authors**: *Fabian Zaiser, Andrzej S. Murawski, Chih-Hao Luke Ong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0747af6f877c0cb555fea595f01b0e83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0747af6f877c0cb555fea595f01b0e83-Abstract-Conference.html)

**Abstract**:

We present an exact Bayesian inference method for discrete statistical models, which can find exact solutions to a large class of discrete inference problems, even with infinite support and continuous priors.To express such models, we introduce a probabilistic programming language that supports discrete and continuous sampling, discrete observations, affine functions, (stochastic) branching, and conditioning on discrete events.Our key tool is probability generating functions:they provide a compact closed-form representation of distributions that are definable by programs, thus enabling the exact computation of posterior probabilities, expectation, variance, and higher moments.Our inference method is provably correct and fully automated in a tool called Genfer, which uses automatic differentiation (specifically, Taylor polynomials), but does not require computer algebra.Our experiments show that Genfer is often faster than the existing exact inference tools PSI, Dice, and Prodigy.On a range of real-world inference problems that none of these exact tools can solve, Genfer's performance is competitive with approximate Monte Carlo methods, while avoiding approximation errors.

----

## [113] SE(3) Equivariant Convolution and Transformer in Ray Space

**Authors**: *Yinshuang Xu, Jiahui Lei, Kostas Daniilidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html)

**Abstract**:

3D reconstruction and novel view rendering can greatly benefit from geometric priors when the input views are not sufficient in terms of coverage and inter-view baselines. Deep learning of geometric priors from 2D images requires each image to be represented in a $2D$ canonical frame and the prior to be learned in a given or learned $3D$ canonical frame. In this paper, given only the relative poses of the cameras, we show how to learn priors from multiple views equivariant to coordinate frame transformations by proposing an $SE(3)$-equivariant convolution and transformer in the space of rays in 3D. We model the ray space as a homogeneous space of $SE(3)$ and introduce the $SE(3)$-equivariant convolution in ray space. Depending on the output domain of the convolution, we present convolution-based $SE(3)$-equivariant maps from ray space to ray space and to $\mathbb{R}^3$. Our mathematical framework allows us to go beyond convolution to $SE(3)$-equivariant attention in the ray space. We showcase how to tailor and adapt the equivariant convolution and transformer in the tasks of equivariant $3D$ reconstruction and equivariant neural rendering from multiple views. We demonstrate $SE(3)$-equivariance by obtaining robust results in roto-translated datasets without performing transformation augmentation.

----

## [114] Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision

**Authors**: *Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David D. Cox, Yiming Yang, Chuang Gan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0764db1151b936aca59249e2c1386101-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0764db1151b936aca59249e2c1386101-Abstract-Conference.html)

**Abstract**:

Recent AI-assistant agents, such as ChatGPT, predominantly rely on supervised fine-tuning (SFT) with human annotations and reinforcement learning from human feedback (RLHF) to align the output of large language models (LLMs) with human intentions, ensuring they are helpful, ethical, and reliable. However, this dependence can significantly constrain the true potential of AI-assistant agents due to the high cost of obtaining human supervision and the related issues on quality, reliability, diversity, self-consistency, and undesirable biases. To address these challenges, we propose a novel approach called SELF-ALIGN, which combines principle-driven reasoning and the generative power of LLMs for the self-alignment of AI agents with minimal human supervision. Our approach encompasses four stages: first, we use an LLM to generate synthetic prompts, and a topic-guided method to augment the prompt diversity; second, we use a small set of human-written principles for AI models to follow, and guide the LLM through in-context learning from demonstrations (of principles application) to produce helpful, ethical, and reliable responses to user's queries; third, we fine-tune the original LLM with the high-quality self-aligned responses so that the resulting model can generate desirable responses for each query directly without the principle set and the demonstrations anymore; and finally, we offer a refinement step to address the issues of overly-brief or indirect responses. Applying SELF-ALIGN to the LLaMA-65b base language model, we develop an AI assistant named Dromedary. With fewer than 300 lines of human annotations (including < 200 seed prompts, 16 generic principles, and 5 exemplars for in-context learning). Dromedary significantly surpasses the performance of several state-of-the-art AI systems, including Text-Davinci-003 and Alpaca, on benchmark datasets with various settings.

----

## [115] Prototypical Variational Autoencoder for 3D Few-shot Object Detection

**Authors**: *Weiliang Tang, Biqi Yang, Xianzhi Li, Yun-Hui Liu, Pheng-Ann Heng, Chi-Wing Fu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/076a93fd42aa85f5ccee921a01d77dd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/076a93fd42aa85f5ccee921a01d77dd5-Abstract-Conference.html)

**Abstract**:

Few-Shot 3D Point Cloud Object Detection (FS3D) is a challenging task, aiming to detect 3D objects of novel classes using only limited annotated samples for training. Considering that the detection performance highly relies on the quality of the latent features, we design a VAE-based prototype learning scheme, named prototypical VAE (P-VAE), to learn a probabilistic latent space for enhancing the diversity and distinctiveness of the sampled features. The network encodes a multi-center GMM-like posterior, in which each distribution centers at a prototype. For regularization, P-VAE incorporates a reconstruction task to preserve geometric information. To adopt P-VAE for the detection framework, we formulate Geometric-informative Prototypical VAE (GP-VAE) to handle varying geometric components and Class-specific Prototypical VAE (CP-VAE) to handle varying object categories. In the first stage, we harness GP-VAE to aid feature extraction from the input scene. In the second stage, we cluster the geometric-informative features into per-instance features and use CP-VAE to refine each instance feature with category-level guidance. Experimental results show the top performance of our approach over the state of the arts on two FS3D benchmarks. Quantitative ablations and qualitative prototype analysis further demonstrate that our probabilistic modeling can significantly boost prototype learning for FS3D.

----

## [116] Double Gumbel Q-Learning

**Authors**: *David Yu-Tung Hui, Aaron C. Courville, Pierre-Luc Bacon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07956d40074d6523bad11112b3225c6e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07956d40074d6523bad11112b3225c6e-Abstract-Conference.html)

**Abstract**:

We show that Deep Neural Networks introduce two heteroscedastic Gumbel noise sources into Q-Learning.  To account for these noise sources, we propose Double Gumbel Q-Learning, a Deep Q-Learning algorithm applicable for both discrete and continuous control.  In discrete control, we derive a closed-form expression for the loss function of our algorithm.  In continuous control, this loss function is intractable and we therefore derive an approximation with a hyperparameter whose value regulates pessimism in Q-Learning.  We present a default value for our pessimism hyperparameter that enables DoubleGum to outperform DDPG, TD3, SAC, XQL, quantile regression, and Mixture-of-Gaussian Critics in aggregate over 33 tasks from DeepMind Control, MuJoCo, MetaWorld, and Box2D and show that tuning this hyperparameter may further improve sample efficiency.

----

## [117] Mutual-Information Regularized Multi-Agent Policy Iteration

**Authors**: *Jiangxing Wang, Deheng Ye, Zongqing Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0799492e7be38b66d10ead5e8809616d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0799492e7be38b66d10ead5e8809616d-Abstract-Conference.html)

**Abstract**:

Despite the success of cooperative multi-agent reinforcement learning algorithms, most of them focus on a single team composition, which prevents them from being used in more realistic scenarios where dynamic team composition is possible. While some studies attempt to solve this problem via multi-task learning in a fixed set of team compositions, there is still a risk of overfitting to the training set, which may lead to catastrophic performance when facing dramatically varying team compositions during execution. To address this problem, we propose to use mutual information (MI) as an augmented reward to prevent individual policies from relying too much on team-related information and encourage agents to learn policies that are robust in different team compositions. Optimizing this MI-augmented objective in an off-policy manner can be intractable due to the existence of dynamic marginal distribution. To alleviate this problem, we first propose a multi-agent policy iteration algorithm with a fixed marginal distribution and prove its convergence and optimality. Then, we propose to employ the Blahutâ€“Arimoto algorithm and an imaginary team composition distribution for optimization with approximate marginal distribution as the practical implementation. Empirically, our method demonstrates strong zero-shot generalization to dynamic team compositions in complex cooperative tasks.

----

## [118] An Efficient End-to-End Training Approach for Zero-Shot Human-AI Coordination

**Authors**: *Xue Yan, Jiaxian Guo, Xingzhou Lou, Jun Wang, Haifeng Zhang, Yali Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07a363fd2263091c2063998e0034999c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07a363fd2263091c2063998e0034999c-Abstract-Conference.html)

**Abstract**:

The goal of zero-shot human-AI coordination is to develop an agent that can collaborate with humans without relying on human data. Prevailing two-stage population-based methods require a diverse population of mutually distinct policies to simulate diverse human behaviors. The necessity of such populations severely limits their computational efficiency. To address this issue, we propose E3T, an Efficient End-to-End Training approach for zero-shot human-AI coordination. E3T employs a mixture of ego policy and random policy to construct the partner policy, making it both coordination-skilled and diverse. In this way, the ego agent is end-to-end trained with this mixture policy without the need of a pre-trained population, thus significantly improving the training efficiency.  In addition, a partner modeling module is proposed to predict the partner's action from historical information. With the predicted partner's action, the ego policy is able to adapt its policy and take actions accordingly when collaborating with humans of different behavior patterns. Empirical results on the Overcooked environment show that our method significantly improves the training efficiency while preserving comparable or superior performance than the population-based baselines. Demo videos are available at https://sites.google.com/view/e3t-overcooked.

----

## [119] Computing Optimal Equilibria and Mechanisms via Learning in Zero-Sum Extensive-Form Games

**Authors**: *Brian Hu Zhang, Gabriele Farina, Ioannis Anagnostides, Federico Cacciamani, Stephen McAleer, Andreas A. Haupt, Andrea Celli, Nicola Gatti, Vincent Conitzer, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07be1a0850e58ca29e2b6ce31fc0c791-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07be1a0850e58ca29e2b6ce31fc0c791-Abstract-Conference.html)

**Abstract**:

We introduce a new approach for computing optimal equilibria via learning in games. It applies to extensive-form settings with any number of players, including mechanism design, information design, and solution concepts such as correlated, communication, and certification equilibria. We observe that optimal equilibria are minimax equilibrium strategies of a player in an extensive-form zero-sum game. This reformulation allows to apply techniques for learning in zero-sum games, yielding the first learning dynamics that converge to optimal equilibria, not only in empirical averages, but also in iterates. We demonstrate the practical scalability and flexibility of our approach by attaining state-of-the-art performance in benchmark tabular games, and by computing an optimal mechanism for a sequential auction design problem using deep reinforcement learning.

----

## [120] Parts of Speech-Grounded Subspaces in Vision-Language Models

**Authors**: *James Oldfield, Christos Tzelepis, Yannis Panagakis, Mihalis Nicolaou, Ioannis Patras*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07cf32cf61224da628157b7ed0ce994a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07cf32cf61224da628157b7ed0ce994a-Abstract-Conference.html)

**Abstract**:

Latent image representations arising from vision-language models have proved immensely useful for a variety of downstream tasks. However, their utility is limited by their entanglement with respect to different visual attributes. For instance, recent work has shown that CLIP image representations are often biased toward specific visual properties (such as objects or actions) in an unpredictable manner. In this paper, we propose to separate representations of the different visual modalities in CLIP’s joint vision-language space by leveraging the association between parts of speech and specific visual modes of variation (e.g. nouns relate to objects, adjectives describe appearance). This is achieved by formulating an appropriate component analysis model that learns subspaces capturing variability corresponding to a specific part of speech, while jointly minimising variability to the rest. Such a subspace yields disentangled representations of the different visual properties of an image or text in closed form while respecting the underlying geometry of the manifold on which the representations lie. What’s more, we show the proposed model additionally facilitates learning subspaces corresponding to specific visual appearances (e.g. artists’ painting styles), which enables the selective removal of entire visual themes from CLIP-based text-to-image synthesis. We validate the model both qualitatively, by visualising the subspace projections with a text-to-image model and by preventing the imitation of artists’ styles, and quantitatively, through class invariance metrics and improvements to baseline zero-shot classification.

----

## [121] Searching for Optimal Per-Coordinate Step-sizes with Multidimensional Backtracking

**Authors**: *Frederik Kunstner, Victor Sanches Portella, Mark Schmidt, Nicholas J. A. Harvey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07e436cdeb48e2a67618274f5d5eff85-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07e436cdeb48e2a67618274f5d5eff85-Abstract-Conference.html)

**Abstract**:

The backtracking line-search is an effective technique to automatically tune the step-size in smooth optimization. It guarantees similar performance to using the theoretically optimal step-size. Many approaches have been developed to instead tune per-coordinate step-sizes, also known as diagonal preconditioners, but none of the existing methods are provably competitive with the optimal per-coordinate step-sizes. We propose multidimensional backtracking, an extension of the backtracking line-search to find good diagonal preconditioners for smooth convex problems. Our key insight is that the gradient with respect to the step-sizes, also known as hyper-gradients, yields separating hyperplanes that let us search for good preconditioners using cutting-plane methods. As black-box cutting-plane approaches like the ellipsoid method are computationally prohibitive, we develop an efficient algorithm tailored to our setting. Multidimensional backtracking is provably competitive with the best diagonal preconditioner and requires no manual tuning.

----

## [122] Estimating the Rate-Distortion Function by Wasserstein Gradient Descent

**Authors**: *Yibo Yang, Stephan Eckstein, Marcel Nutz, Stephan Mandt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07eea3fb833c905c5edf46f914231f15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07eea3fb833c905c5edf46f914231f15-Abstract-Conference.html)

**Abstract**:

In the theory of lossy compression, the rate-distortion (R-D) function $R(D)$ describes how much a data source can be compressed (in bit-rate) at any given level of fidelity (distortion). Obtaining $R(D)$ for a given data source establishes the fundamental performance limit for all compression algorithms.  We propose a new method to estimate $R(D)$ from the perspective of optimal transport. Unlike the classic Blahut--Arimoto algorithm which fixes the support of the reproduction distribution in advance, our Wasserstein gradient descent algorithm learns the support of the optimal reproduction distribution by moving particles. We prove its local convergence and analyze the sample complexity of our R-D estimator based on a connection to entropic optimal transport.  Experimentally, we obtain comparable or tighter bounds than state-of-the-art neural network methods on low-rate sources while requiring considerably less tuning and computation effort.  We also highlight a connection to maximum-likelihood deconvolution and introduce a new class of sources that can be used as test cases with known solutions to the R-D problem.

----

## [123] Epistemic Neural Networks

**Authors**: *Ian Osband, Zheng Wen, Seyed Mohammad Asghari, Vikranth Dwaracherla, Morteza Ibrahimi, Xiuyuan Lu, Benjamin Van Roy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/07fbde96bee50f4e09303fd4f877c2f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/07fbde96bee50f4e09303fd4f877c2f3-Abstract-Conference.html)

**Abstract**:

Intelligence relies on an agent's knowledge of what it does not know.This capability can be assessed based on the quality of joint predictions of labels across multiple inputs.In principle, ensemble-based approaches can produce effective joint predictions, but the computational costs of large ensembles become prohibitive.We introduce the epinet: an architecture that can supplement any conventional neural network, including large pretrained models, and can be trained with modest incremental computation to estimate uncertainty.With an epinet, conventional neural networks outperform very large ensembles, consisting of hundreds or more particles, with orders of magnitude less computation.The epinet does not fit the traditional framework of Bayesian neural networks.To accommodate development of approaches beyond BNNs, such as the epinet, we introduce the epistemic neural network (ENN) as a general interface for models that produce joint predictions.

----

## [124] HotBEV: Hardware-oriented Transformer-based Multi-View 3D Detector for BEV Perception

**Authors**: *Peiyan Dong, Zhenglun Kong, Xin Meng, Pinrui Yu, Yifan Gong, Geng Yuan, Hao Tang, Yanzhi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/081b08068e4733ae3e7ad019fe8d172f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/081b08068e4733ae3e7ad019fe8d172f-Abstract-Conference.html)

**Abstract**:

The bird's-eye-view (BEV) perception plays a critical role in autonomous driving systems, involving the accurate and efficient detection and tracking of objects from a top-down perspective. To achieve real-time decision-making in self-driving scenarios, low-latency computation is essential. While recent approaches to BEV detection have focused on improving detection precision using Lift-Splat-Shoot (LSS)-based or transformer-based schemas, the substantial computational and memory burden of these approaches increases the risk of system crashes when multiple on-vehicle tasks run simultaneously. Unfortunately, there is a dearth of literature on efficient BEV detector paradigms, let alone achieving realistic speedups.Unlike existing works that focus on reducing computation costs, this paper focuses on developing an efficient model design that prioritizes actual on-device latency.To achieve this goal, we propose a latency-aware design methodology that considers key hardware properties, such as memory access cost and degree of parallelism.Given the prevalence of GPUs as the main computation platform for autonomous driving systems, we develop a theoretical latency prediction model and introduce efficient building operators.By leveraging these operators and following an effective local-to-global visual modeling process, we propose a hardware-oriented backbone that is also optimized for strong feature capturing and fusing.Using these insights, we present a new hardware-oriented framework for efficient yet accurate camera-view BEV detectors.Experiments show that HotBEV achieves a 2\%$\sim$23\% NDS gain, and 2\%$\sim$7.8\% mAP gain with a 1.1$\times$$\sim$3.4$\times$ speedups compared to existing works on V100;On multiple GPU devices such as GPU GTX 2080 and the low-end GTX 1080, HotBEV achieves 1.1$\times$$\sim$6.3$\times$ faster than others.

----

## [125] Mip-Grid: Anti-aliased Grid Representations for Neural Radiance Fields

**Authors**: *Seungtae Nam, Daniel Rho, Jong Hwan Ko, Eunbyung Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/082d3d795520c43214da5123e56a3a34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/082d3d795520c43214da5123e56a3a34-Abstract-Conference.html)

**Abstract**:

Despite the remarkable achievements of neural radiance fields (NeRF) in representing 3D scenes and generating novel view images, the aliasing issue, rendering 'jaggies' or 'blurry' images at varying camera distances, remains unresolved in most existing approaches. The recently proposed mip-NeRF has effectively addressed this challenge by introducing integrated positional encodings (IPE). However, it relies on MLP architecture to represent the radiance fields, missing out on the fast training speed offered by the latest grid-based methods. In this work, we present mip-Grid, a novel approach that integrates anti-aliasing techniques into grid-based representations for radiance fields, mitigating the aliasing artifacts while enjoying fast training time. Notably, the proposed method uses a single-scale shared grid representation and a single-sampling approach, which only introduces minimal additions to the model parameters and computational costs. To handle scale ambiguity, mip-Grid generates multiple grids by applying simple convolution operations over the shared grid and uses the scale-aware coordinate to retrieve the appropriate features from the generated multiple grids. To test the effectiveness, we incorporated the proposed approach into the two recent representative grid-based methods, TensoRF and K-Planes. The experimental results demonstrated that mip-Grid greatly improved the rendering performance of both methods and showed comparable performance to mip-NeRF on multi-scale datasets while achieving significantly faster training time.

----

## [126] Theoretically Guaranteed Bidirectional Data Rectification for Robust Sequential Recommendation

**Authors**: *Yatong Sun, Bin Wang, Zhu Sun, Xiaochun Yang, Yan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08309150af77fc7c79ade0bf8bb6a562-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08309150af77fc7c79ade0bf8bb6a562-Abstract-Conference.html)

**Abstract**:

Sequential recommender systems (SRSs) are typically trained to predict the next item as the target given its preceding (and succeeding) items as the input. Such a paradigm assumes that every input-target pair is reliable for training. However, users can be induced to click on items that are inconsistent with their true preferences, resulting in unreliable instances, i.e., mismatched input-target pairs. Current studies on mitigating this issue suffer from two limitations: (i) they discriminate instance reliability according to models trained with unreliable data, yet without theoretical guarantees that such a seemingly contradictory solution can be effective; and (ii) most methods can only tackle either unreliable input or targets but fail to handle both simultaneously. To fill the gap, we theoretically unveil the relationship between SRS predictions and instance reliability, whereby two error-bounded strategies are proposed to rectify unreliable targets and input, respectively. On this basis, we devise a model-agnostic Bidirectional Data Rectification (BirDRec) framework, which can be flexibly implemented with most existing SRSs for robust training against unreliable data. Additionally, a rectification sampling strategy is devised and a self-ensemble mechanism is adopted to reduce the (time and space) complexity of BirDRec. Extensive experiments on four real-world datasets verify the generality, effectiveness, and efficiency of our proposed BirDRec.

----

## [127] Consistent Aggregation of Objectives with Diverse Time Preferences Requires Non-Markovian Rewards

**Authors**: *Silviu Pitis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08342dc6ab69f23167b4123086ad4d38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08342dc6ab69f23167b4123086ad4d38-Abstract-Conference.html)

**Abstract**:

As the capabilities of artificial agents improve, they are being increasingly deployed to service multiple diverse objectives and stakeholders. However, the composition of these objectives is often performed ad hoc, with no clear justification. This paper takes a normative approach to multi-objective agency: from a set of intuitively appealing axioms, it is shown that Markovian aggregation of Markovian reward functions is not possible when the time preference (discount factor) for each objective may vary. It follows that optimal multi-objective agents must admit rewards that are non-Markovian with respect to the individual objectives. To this end, a practical non-Markovian aggregation scheme is proposed, which overcomes the impossibility with only one additional parameter for each objective. This work offers new insights into sequential, multi-objective agency and intertemporal choice, and has practical implications for the design of AI systems deployed to serve multiple generations of principals with varying time preference.

----

## [128] Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability

**Authors**: *Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/088463cd3126aef2002ffc69da42ec59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/088463cd3126aef2002ffc69da42ec59-Abstract-Conference.html)

**Abstract**:

Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberatelymislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g. content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods.

----

## [129] InstanT: Semi-supervised Learning with Instance-dependent Thresholds

**Authors**: *Muyang Li, Runze Wu, Haoyu Liu, Jun Yu, Xun Yang, Bo Han, Tongliang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/088d99765bc121c6df215da7d45bc4e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/088d99765bc121c6df215da7d45bc4e9-Abstract-Conference.html)

**Abstract**:

Semi-supervised learning (SSL) has been a fundamental challenge in machine learning for decades. The primary family of SSL algorithms, known as pseudo-labeling, involves assigning pseudo-labels to confident unlabeled instances and incorporating them into the training set. Therefore, the selection criteria of confident instances are crucial to the success of SSL. Recently, there has been growing interest in the development of SSL methods that use dynamic or adaptive thresholds. Yet, these methods typically apply the same threshold to all samples, or use class-dependent thresholds for instances belonging to a certain class, while neglecting instance-level information. In this paper, we propose the study of instance-dependent thresholds, which has the highest degree of freedom compared with existing methods. Specifically, we devise a novel instance-dependent threshold function for all unlabeled instances by utilizing their instance-level ambiguity and the instance-dependent error rates of pseudo-labels, so instances that are more likely to have incorrect pseudo-labels will have higher thresholds. Furthermore, we demonstrate that our instance-dependent threshold function provides a bounded probabilistic guarantee for the correctness of the pseudo-labels it assigns.

----

## [130] Neural Lyapunov Control for Discrete-Time Systems

**Authors**: *Junlin Wu, Andrew Clark, Yiannis Kantaros, Yevgeniy Vorobeychik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08bf1773e94763b6cc366ee7c6582f27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08bf1773e94763b6cc366ee7c6582f27-Abstract-Conference.html)

**Abstract**:

While ensuring stability for linear systems is well understood, it remains a major challenge for nonlinear systems. A general approach in such cases is to compute a combination of a Lyapunov function and an associated control policy. However, finding Lyapunov functions for general nonlinear systems is a challenging task. To address this challenge, several methods have been proposed that represent Lyapunov functions using neural networks. However, such approaches either focus on continuous-time systems, or highly restricted classes of nonlinear dynamics. We propose the first approach for learning neural Lyapunov control in a broad class of discrete-time systems. Three key ingredients enable us to effectively learn provably stable control policies. The first is a novel mixed-integer linear programming approach for verifying the discrete-time Lyapunov stability conditions, leveraging the particular structure of these conditions. The second is a novel approach for computing verified sublevel sets. The third is a heuristic gradient-based method for quickly finding counterexamples to significantly speed up Lyapunov function learning. Our experiments on four standard benchmarks demonstrate that our approach significantly outperforms state-of-the-art baselines. For example, on the path tracking benchmark, we outperform recent neural Lyapunov control baselines by an order of magnitude in both running time and the size of the region of attraction, and on two of the four benchmarks (cartpole and PVTOL), ours is the first automated approach to return a provably stable controller. Our code is available at: https://github.com/jlwu002/nlc_discrete.

----

## [131] Information Maximization Perspective of Orthogonal Matching Pursuit with Applications to Explainable AI

**Authors**: *Aditya Chattopadhyay, Ryan Pilgrim, René Vidal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08eac13583b310ec55d755f99c549be3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08eac13583b310ec55d755f99c549be3-Abstract-Conference.html)

**Abstract**:

Information Pursuit (IP) is a classical active testing algorithm for predicting an output by sequentially and greedily querying the input in order of information gain. However, IP is computationally intensive since it involves estimating mutual information in high-dimensional spaces. This paper explores Orthogonal Matching Pursuit (OMP) as an alternative to IP for greedily selecting the queries. OMP is a classical signal processing algorithm for sequentially encoding a signal in terms of dictionary atoms chosen in order of correlation gain. In each iteration, OMP selects the atom that is most correlated with the signal residual (the signal minus its reconstruction thus far). Our first contribution is to establish a fundamental connection between IP and OMP, where we prove that IP with random projections of dictionary atoms as queries ``almost'' reduces to OMP, with the difference being that IP selects atoms in order of normalized correlation gain. We call this version IP-OMP and present simulations indicating that this difference does not have any appreciable effect on the sparse code recovery rate of IP-OMP compared to that of OMP for random Gaussian dictionaries. Inspired by this connection, our second contribution is to explore the utility of IP-OMP for generating explainable predictions, an area in which IP has recently gained traction. More specifically, we propose a simple explainable AI algorithm which encodes an image as a sparse combination of semantically meaningful dictionary atoms that are defined as text embeddings of interpretable concepts. The final prediction is made using the weights of this sparse combination, which serve as an explanation. Empirically, our proposed algorithm is not only competitive with existing explainability methods but also computationally less expensive.

----

## [132] Evolving Connectivity for Recurrent Spiking Neural Networks

**Authors**: *Guan Wang, Yuhao Sun, Sijie Cheng, Sen Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/08f9de0232c0b485110237f6e6cf88f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/08f9de0232c0b485110237f6e6cf88f1-Abstract-Conference.html)

**Abstract**:

Recurrent spiking neural networks (RSNNs) hold great potential for advancing artificial general intelligence, as they draw inspiration from the biological nervous system and show promise in modeling complex dynamics.However, the widely-used surrogate gradient-based training methods for RSNNs are inherently inaccurate and unfriendly to neuromorphic hardware.To address these limitations, we propose the evolving connectivity (EC) framework, an inference-only method for training RSNNs.The EC framework reformulates weight-tuning as a search into parameterized connection probability distributions, and employs Natural Evolution Strategies (NES) for optimizing these distributions.Our EC framework circumvents the need for gradients and features hardware-friendly characteristics, including sparse boolean connections and high scalability.We evaluate EC on a series of standard robotic locomotion tasks, where it achieves comparable performance with deep neural networks and outperforms gradient-trained RSNNs, even solving the complex 17-DoF humanoid task.Additionally, the EC framework demonstrates a two to three fold speedup in efficiency compared to directly evolving parameters.By providing a performant and hardware-friendly alternative, the EC framework lays the groundwork for further energy-efficient applications of RSNNs and advances the development of neuromorphic devices.Our code is publicly available at https://github.com/imoneoi/EvolvingConnectivity.

----

## [133] Bayesian Optimization with Cost-varying Variable Subsets

**Authors**: *Sebastian Tay, Chuan Sheng Foo, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/090b23d52bc2722eef2fbf79c5ebf9ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/090b23d52bc2722eef2fbf79c5ebf9ec-Abstract-Conference.html)

**Abstract**:

We introduce the problem of Bayesian optimization with cost-varying variable subsets (BOCVS) where in each iteration, the learner chooses a subset of query variables and specifies their values while the rest are randomly sampled. Each chosen subset has an associated cost. This presents the learner with the novel challenge of balancing between choosing more informative subsets for more directed learning versus leaving some variables to be randomly sampled to reduce incurred costs. This paper presents a novel Gaussian process upper confidence bound-based algorithm for solving the BOCVS problem that is provably no-regret. We analyze how the availability of cheaper control sets helps in exploration and reduces overall regret. We empirically show that our proposed algorithm can find significantly better solutions than comparable baselines with the same budget.

----

## [134] Transformed Low-Rank Parameterization Can Help Robust Generalization for Tensor Neural Networks

**Authors**: *Andong Wang, Chao Li, Mingyuan Bai, Zhong Jin, Guoxu Zhou, Qibin Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html)

**Abstract**:

Multi-channel learning has gained significant attention in recent applications, where neural networks with t-product layers (t-NNs) have shown promising performance through novel feature mapping in the transformed domain. However, despite the practical success of t-NNs, the theoretical analysis of their generalization remains unexplored. We address this gap by deriving upper bounds on the generalization error of t-NNs in both standard and adversarial settings. Notably, it reveals that t-NNs compressed with exact transformed low-rank parameterization can achieve tighter adversarial generalization bounds compared to non-compressed models. While exact transformed low-rank weights are rare in practice, the analysis demonstrates that through adversarial training with gradient flow, highly over-parameterized t-NNs with the ReLU activation can be implicitly regularized towards a transformed low-rank parameterization under certain conditions. Moreover, this paper establishes sharp adversarial generalization bounds for t-NNs with approximately transformed low-rank weights. Our analysis highlights the potential of transformed low-rank parameterization in enhancing the robust generalization of t-NNs, offering valuable insights for further research and development.

----

## [135] Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples

**Authors**: *Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Mehran Kazemi, Najoung Kim, He He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09425891e393e64b0535194a81ba15b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/09425891e393e64b0535194a81ba15b7-Abstract-Conference.html)

**Abstract**:

Given the intractably large size of the space of proofs, any model that is capable of general deductive reasoning must generalize to proofs of greater complexity. Recent studies have shown that large language models (LLMs) possess some abstract deductive reasoning ability given chain-of-thought prompts. However, they have primarily been tested on proofs using modus ponens or of a specific size, and from the same distribution as the in-context examples. To measure the general deductive reasoning ability of LLMs, we test on a broad set of deduction rules and measure their ability to generalize to more complex proofs from simpler demonstrations from multiple angles: depth-, width-, and compositional generalization. To facilitate systematic exploration, we construct a new synthetic and programmable reasoning dataset that enables control over deduction rules and proof complexity. Our experiments on four LLMs of various sizes and training objectives show that they are able to generalize to compositional proofs. However, they have difficulty generalizing to longer proofs, and they require explicit demonstrations to produce hypothetical subproofs, specifically in proof by cases and proof by contradiction.

----

## [136] MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining

**Authors**: *Jacob Portes, Alexander Trott, Sam Havens, Daniel King, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, Jonathan Frankle*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/095a6917768712b7ccc61acbeecad1d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/095a6917768712b7ccc61acbeecad1d8-Abstract-Conference.html)

**Abstract**:

Although BERT-style encoder models are heavily used in NLP research, many researchers do not pretrain their own BERTs from scratch due to the high cost of training. In the past half-decade since BERT first rose to prominence, many advances have been made with other transformer architectures and training configurations that have yet to be systematically incorporated into BERT. Here, we introduce MosaicBERT, a BERT-style encoder architecture and training recipe that is empirically optimized for fast pretraining. This efficient architecture incorporates FlashAttention, Attention with Linear Biases (ALiBi), Gated Linear Units (GLU), a module to dynamically remove padded tokens, and low precision LayerNorm into the classic transformer encoder block. The training recipe includes a 30% masking ratio for the Masked Language Modeling (MLM) objective, bfloat16 precision, and vocabulary size optimized for GPU throughput, in addition to best-practices from RoBERTa and other encoder models. When pretrained from scratch on the C4 dataset, this base model achieves a downstream average GLUE (dev) score of 79.6 in 1.13 hours on 8 A100 80 GB GPUs at a cost of roughly $20. We plot extensive accuracy vs. pretraining speed Pareto curves and show that MosaicBERT base and large are consistently Pareto optimal when compared to a competitive BERT base and large. This empirical speed up in pretraining enables researchers and engineers to pretrain custom BERT-style models at low cost instead of finetune on existing generic models. We open source our model weights and code.

----

## [137] GraphMP: Graph Neural Network-based Motion Planning with Efficient Graph Search

**Authors**: *Xiao Zang, Miao Yin, Jinqi Xiao, Saman A. Zonouz, Bo Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/096961cae3c3423c44ea045aeb584e05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/096961cae3c3423c44ea045aeb584e05-Abstract-Conference.html)

**Abstract**:

Motion planning, which aims to find a high-quality collision-free path in the configuration space, is a fundamental task in robotic systems. Recently, learning-based motion planners, especially the graph neural network-powered, have shown promising planning performance. However, though the state-of-the-art GNN planner can efficiently extract and learn graph information, its inherent mechanism is not well suited for graph search process, hindering its further performance improvement. To address this challenge and fully unleash the potential of GNN in motion planning, this paper proposes GraphMP, a neural motion planner for both low and high-dimensional planning tasks. With the customized model architecture and training mechanism design, GraphMP can simultaneously perform efficient graph pattern extraction and graph search processing, leading to strong planning performance. Experiments on a variety of environments, ranging from 2D Maze to 14D dual KUKA robotic arm, show that our proposed GraphMP achieves significant improvement on path quality and planning speed over the state-of-the-art learning-based and classical planners; while preserving the competitive success rate.

----

## [138] Accountability in Offline Reinforcement Learning: Explaining Decisions with a Corpus of Examples

**Authors**: *Hao Sun, Alihan Hüyük, Daniel Jarrett, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/096b1019463f34eb241e87cfce8dfe16-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/096b1019463f34eb241e87cfce8dfe16-Abstract-Conference.html)

**Abstract**:

Learning controllers with offline data in decision-making systems is an essential area of research due to its potential to reduce the risk of applications in real-world systems. However, in responsibility-sensitive settings such as healthcare, decision accountability is of paramount importance, yet has not been adequately addressed by the literature.This paper introduces the Accountable Offline Controller (AOC) that employs the offline dataset as the Decision Corpus and performs accountable control based on a tailored selection of examples, referred to as the Corpus Subset. AOC operates effectively in low-data scenarios, can be extended to the strictly offline imitation setting, and displays qualities of both conservation and adaptability.We assess AOC's performance in both simulated and real-world healthcare scenarios, emphasizing its capability to manage offline control tasks with high levels of performance while maintaining accountability.

----

## [139] Synthcity: a benchmark framework for diverse use cases of tabular synthetic data

**Authors**: *Zhaozhi Qian, Robert Davis, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09723c9f291f6056fd1885081859c186-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/09723c9f291f6056fd1885081859c186-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Accessible high-quality data is the bread and butter of machine learning research, and the demand for data has exploded as larger and more advanced ML models are built across different domains. Yet, real data often contain sensitive information, are subject to various biases, and are costly to acquire, which compromise their quality and accessibility. Synthetic data have thus emerged as a complement to, sometimes even a replacement for, real data for ML training. However, the landscape of synthetic data research has been fragmented due to the diverse range of data modalities, such as tabular, time series, and images, and the wide array of use cases, including privacy preservation, fairness considerations, and data augmentation. This fragmentation poses practical challenges when comparing and selecting synthetic data generators in for different problem settings. To this end, we develop Synthcity, an open-source Python library that allows researchers and practitioners to perform one-click benchmarking of synthetic data generators across data modalities and use cases. Beyond benchmarking, Synthcity serves as a centralized toolkit for accessing cutting-edge data generators. In addition, Synthcityâ€™s flexible plug-in style API makes it easy to incorporate additional data generators into the framework. Using examples of tabular data generation and data augmentation, we illustrate the general applicability of Synthcity, and the insight one can obtain.

----

## [140] SOAR: Improved Indexing for Approximate Nearest Neighbor Search

**Authors**: *Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, Sanjiv Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0973524e02a712af33325d0688ae6f49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0973524e02a712af33325d0688ae6f49-Abstract-Conference.html)

**Abstract**:

This paper introduces SOAR: Spilling with Orthogonality-Amplified Residuals, a novel data indexing technique for approximate nearest neighbor (ANN) search. SOAR extends upon previous approaches to ANN search, such as spill trees, that utilize multiple redundant representations while partitioning the data to reduce the probability of missing a nearest neighbor during search. Rather than training and computing these redundant representations independently, however, SOAR uses an orthogonality-amplified residual loss, which optimizes each representation to compensate for cases where other representations perform poorly. This drastically improves the overall index quality, resulting in state-of-the-art ANN benchmark performance while maintaining fast indexing times and low memory consumption.

----

## [141] Type-to-Track: Retrieve Any Object via Prompt-based Tracking

**Authors**: *Pha Nguyen, Kha Gia Quach, Kris Kitani, Khoa Luu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html)

**Abstract**:

One of the recent trends in vision problems is to use natural language captions to describe the objects of interest. This approach can overcome some limitations of traditional methods that rely on bounding boxes or category annotations. This paper introduces a novel paradigm for Multiple Object Tracking called Type-to-Track, which allows users to track objects in videos by typing natural language descriptions. We present a new dataset for that Grounded Multiple Object Tracking task, called GroOT, that contains videos with various types of objects and their corresponding textual captions describing their appearance and action in detail. Additionally, we introduce two new evaluation protocols and formulate evaluation metrics specifically for this task. We develop a new efficient method that models a transformer-based eMbed-ENcoDE-extRact framework (MENDER) using the third-order tensor decomposition. The experiments in five scenarios show that our MENDER approach outperforms another two-stage design in terms of accuracy and efficiency, up to 14.7\% accuracy and $4\times$ speed faster.

----

## [142] Finding Counterfactually Optimal Action Sequences in Continuous State Spaces

**Authors**: *Stratis Tsirtsis, Manuel Rodriguez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09ae6beae5f1ff38f05c05979097ea0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/09ae6beae5f1ff38f05c05979097ea0f-Abstract-Conference.html)

**Abstract**:

Whenever a clinician reflects on the efficacy of a sequence of treatment decisions for a patient, they may try to identify critical time steps where, had they made different decisions, the patient's health would have improved. While recent methods at the intersection of causal inference and reinforcement learning promise to aid human experts, as the clinician above, to retrospectively analyze sequential decision making processes, they have focused on environments with finitely many discrete states. However, in many practical applications, the state of the environment is inherently continuous in nature. In this paper, we aim to fill this gap. We start by formally characterizing a sequence of discrete actions and continuous states using finite horizon Markov decision processes and a broad class of bijective structural causal models. Building upon this characterization, we formalize the problem of finding counterfactually optimal action sequences and show that, in general, we cannot expect to solve it in polynomial time. Then, we develop a search method based on the A* algorithm that, under a natural form of Lipschitz continuity of the environmentâ€™s dynamics, is guaranteed to return the optimal solution to the problem. Experiments on real clinical data show that our method is very efficient in practice, and it has the potential to offer interesting insights for sequential decision making tasks.

----

## [143] Reusing Pretrained Models by Multi-linear Operators for Efficient Training

**Authors**: *Yu Pan, Ye Yuan, Yichun Yin, Zenglin Xu, Lifeng Shang, Xin Jiang, Qun Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09d9a13f7018110cfb439c06b07940a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/09d9a13f7018110cfb439c06b07940a2-Abstract-Conference.html)

**Abstract**:

Training large models from scratch usually costs a substantial amount of resources. Towards this problem, recent studies such as bert2BERT and LiGO have reused small pretrained models to initialize a large model (termed the ``target model''), leading to a considerable acceleration in training. Despite the successes of these previous studies, they  grew pretrained models by mapping partial weights only, ignoring potential correlations across the entire model. As we show in this paper, there are inter- and intra-interactions among the weights of both the pretrained and the target models. As a result, the partial mapping may not capture the complete information and lead to inadequate growth. In this paper, we propose a method that linearly correlates each weight of the target model to all the weights of the pretrained model to further enhance acceleration ability. We utilize multi-linear operators to reduce computational and spacial complexity, enabling acceptable resource requirements. Experiments demonstrate that our method can save 76\% computational costs on DeiT-base transferred from DeiT-small, which outperforms bert2BERT by +12\% and LiGO by +21\%, respectively.

----

## [144] Tartarus: A Benchmarking Platform for Realistic And Practical Inverse Molecular Design

**Authors**: *AkshatKumar Nigam, Robert Pollice, Gary Tom, Kjell Jorner, John Willes, Luca A. Thiede, Anshul Kundaje, Alán Aspuru-Guzik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/09f8b2469a3d1089a7c60d9ef1983271-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/09f8b2469a3d1089a7c60d9ef1983271-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The efficient exploration of chemical space to design molecules with intended properties enables the accelerated discovery of drugs, materials, and catalysts, and is one of the most important outstanding challenges in chemistry. Encouraged by the recent surge in computer power and artificial intelligence development, many algorithms have been developed to tackle this problem. However, despite the emergence of many new approaches in recent years, comparatively little progress has been made in developing realistic benchmarks that reflect the complexity of molecular design for real-world applications. In this work, we develop a set of practical benchmark tasks relying on physical simulation of molecular systems mimicking real-life molecular design problems for materials, drugs, and chemical reactions. Additionally, we demonstrate the utility and ease of use of our new benchmark set by demonstrating how to compare the performance of several well-established families of algorithms. Overall, we believe that our benchmark suite will help move the field towards more realistic molecular design benchmarks, and move the development of inverse molecular design algorithms closer to the practice of designing molecules that solve existing problems in both academia and industry alike.

----

## [145] DreamSparse: Escaping from Plato's Cave with 2D Diffusion Model Given Sparse Views

**Authors**: *Paul Yoo, Jiaxian Guo, Yutaka Matsuo, Shixiang Shane Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a003511b09274348b8117f5f3b94c93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a003511b09274348b8117f5f3b94c93-Abstract-Conference.html)

**Abstract**:

Synthesizing novel view images from a few views is a challenging but practical problem. Existing methods often struggle with producing high-quality results or necessitate per-object optimization in such few-view settings due to the insufficient information provided.  In this work, we explore leveraging the strong 2D priors in pre-trained diffusion models for synthesizing novel view images. 2D diffusion models, nevertheless, lack 3D awareness, leading to distorted image synthesis and compromising the identity. To address these problems, we propose $\textit{DreamSparse}$, a framework that enables the frozen pre-trained diffusion model to generate geometry and identity-consistent novel view images. Specifically, DreamSparse incorporates a geometry module designed to capture features about spatial information from sparse views as a 3D prior. Subsequently, a spatial guidance model is introduced to convert rendered feature maps as spatial information for the generative process. This information is then used to guide the pre-trained diffusion model toencourage the synthesis of geometrically consistent images without further tuning. Leveraging the strong image priors in the pre-trained diffusion models, DreamSparse is capable of synthesizing high-quality novel views for both object and object-centric scene-level images and generalising to open-set images.Experimental results demonstrate that our framework can effectively synthesize novel view images from sparse views and outperforms baselines in both trained and open-set category images. More results can be found on our project page: https://sites.google.com/view/dreamsparse-webpage.

----

## [146] Sample Complexity Bounds for Score-Matching: Causal Discovery and Generative Modeling

**Authors**: *Zhenyu Zhu, Francesco Locatello, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a3dc35a2391cabcb59a6b123544e3db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a3dc35a2391cabcb59a6b123544e3db-Abstract-Conference.html)

**Abstract**:

This paper provides statistical sample complexity bounds for score-matching and its applications in causal discovery. We demonstrate that accurate estimation of the score function is achievable by training a standard deep ReLU neural network using stochastic gradient descent. We establish bounds on the error rate of recovering causal relationships using the score-matching-based causal discovery method of Rolland et al. [2022], assuming a sufficiently good estimation of the score function. Finally, we analyze the upper bound of score-matching estimation within the score-based generative modeling, which has been applied for causal discovery but is also of independent interest within the domain of generative models.

----

## [147] Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach

**Authors**: *Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a443a000e1cb2281480b3bac395b3b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a443a000e1cb2281480b3bac395b3b8-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) are vulnerable to adversarial perturbations, including those that affect both node features and graph topology. This paper investigates GNNs derived from diverse neural flows, concentrating on their connection to various stability notions such as BIBO stability, Lyapunov stability, structural stability, and conservative stability. We argue that Lyapunov stability, despite its common use, does not necessarily ensure adversarial robustness. Inspired by physics principles, we advocate for the use of conservative Hamiltonian neural flows to construct GNNs that are robust to adversarial attacks. The adversarial robustness of different neural flow GNNs is empirically compared on several benchmark datasets under a variety of adversarial attacks. Extensive numerical experiments demonstrate that GNNs leveraging conservative Hamiltonian flows with Lyapunov stability substantially improve robustness against adversarial perturbations. The implementation code of experiments  is available at \url{https://github.com/zknus/NeurIPS-2023-HANG-Robustness}.

----

## [148] A Path to Simpler Models Starts With Noise

**Authors**: *Lesia Semenova, Harry Chen, Ronald Parr, Cynthia Rudin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a49935d2b3d3342ca08d6db0adcfa34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a49935d2b3d3342ca08d6db0adcfa34-Abstract-Conference.html)

**Abstract**:

The Rashomon set is the set of models that perform approximately equally well on a given dataset, and the Rashomon ratio is the fraction of all models in a given hypothesis space that are in the Rashomon set. Rashomon ratios are often large for tabular datasets in criminal justice, healthcare, lending, education, and in other areas, which has practical implications about whether simpler models can attain the same level of accuracy as more complex models. An open question is why Rashomon ratios often tend to be large. In this work, we propose and study a mechanism of the data generation process, coupled with choices usually made by the analyst during the learning process, that determines the size of the Rashomon ratio. Specifically, we demonstrate that noisier datasets lead to larger Rashomon ratios through the way that practitioners train models. Additionally, we introduce a measure called pattern diversity, which captures the average difference in predictions between distinct classification patterns in the Rashomon set, and motivate why it tends to increase with label noise. Our results explain a key aspect of why simpler models often tend to perform as well as black box models on complex, noisier datasets.

----

## [149] Winner-Take-All Column Row Sampling for Memory Efficient Adaptation of Language Model

**Authors**: *Zirui Liu, Guanchu Wang, Shaochen (Henry) Zhong, Zhaozhuo Xu, Daochen Zha, Ruixiang (Ryan) Tang, Zhimeng Stephen Jiang, Kaixiong Zhou, Vipin Chaudhary, Shuai Xu, Xia Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html)

**Abstract**:

As the model size grows rapidly, fine-tuning the large pre-trained language model has become increasingly difficult due to its extensive memory usage. Previous works usually focus on reducing the number of trainable parameters in the network. While the model parameters do contribute to memory usage, the primary memory bottleneck during training arises from storing feature maps, also known as activations, as they are crucial for gradient calculation. Notably, machine learning models are typically trained using stochastic gradient descent.We argue that in stochastic optimization, models can handle noisy gradients as long as the gradient estimator is unbiased with reasonable variance.Following this motivation, we propose a new family of unbiased estimators called \sas, for matrix production with reduced variance, which only requires storing the sub-sampled activations for calculating the gradient.Our work provides both theoretical and experimental evidence that, in the context of tuning transformers, our proposed estimators exhibit lower variance compared to existing ones.By replacing the linear operation with our approximated one in transformers, we can achieve up to 2.7X peak memory reduction with almost no accuracy drop and enables up to $6.4\times$ larger batch size.Under the same hardware, \sas enables better down-streaming task performance by applying larger models and/or faster training speed with larger batch sizes.The code is available at https://anonymous.4open.science/r/WTACRS-A5C5/.

----

## [150] Zeroth-Order Methods for Nondifferentiable, Nonconvex, and Hierarchical Federated Optimization

**Authors**: *Yuyang Qiu, Uday V. Shanbhag, Farzad Yousefian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a70c9cd8179fe6f8f6135fafa2a8798-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a70c9cd8179fe6f8f6135fafa2a8798-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) has emerged as an enabling framework for communication-efficient decentralized training. We study three broadly applicable problem classes in FL: (i) Nondifferentiable nonconvex federated optimization; (ii) Federated bilevel optimization; (iii) Federated minimax problems. Notably, in an implicit sense, both (ii) and (iii) are instances of (i). However, the hierarchical problems in (ii) and (iii) are often complicated by the absence of a closed-form expression for the implicit objective function. Unfortunately, research on these problems has been limited and afflicted by reliance on strong assumptions, including the need for differentiability and L-smoothness of the implicit function. We address this shortcoming by making the following contributions. In (i), by leveraging convolution-based smoothing and Clarkeâ€™s subdifferential calculus, we devise a randomized smoothing-enabled zeroth-order FL method and derive communication and iteration complexity guarantees for computing an approximate Clarke stationary point.  To contend with (ii) and (iii), we devise a unified randomized implicit zeroth-order FL framework, equipped with explicit communication and iteration complexities. Importantly, our method utilizes delays during local steps to skip making calls to the inexact lower-level FL oracle. This results in significant reduction in communication overhead when addressing hierarchical problems. We empirically validate the theory on nonsmooth and hierarchical ML problems.

----

## [151] Language Model Alignment with Elastic Reset

**Authors**: *Michael Noukhovitch, Samuel Lavoie, Florian Strub, Aaron C. Courville*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0a980183c520446f6b8afb6fa2a2c70e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0a980183c520446f6b8afb6fa2a2c70e-Abstract-Conference.html)

**Abstract**:

Finetuning language models with reinforcement learning (RL), e.g. from human feedback (HF), is a prominent method for alignment. But optimizing against a reward model can improve on reward while degrading performance in other areas, a phenomenon known as reward hacking, alignment tax, or language drift. First, we argue that commonly-used test metrics are insufficient and instead measure how different algorithms tradeoff between reward and drift. The standard method modified the reward with a Kullback-Lieber (KL) penalty between the online and initial model. We propose Elastic Reset, a new algorithm that achieves higher reward with less drift without explicitly modifying the training objective. We periodically reset the online model to an exponentially moving average (EMA) of itself, then reset the EMA model to the initial model. Through the use of an EMA, our model recovers quickly after resets and achieves higher reward with less drift in the same number of steps. We demonstrate that fine-tuning language models with Elastic Reset leads to state-of-the-art performance on a small scale pivot-translation benchmark, outperforms all baselines in a medium-scale RLHF-like IMDB mock sentiment task and leads to a more performant and more aligned technical QA chatbot with LLaMA-7B. Code available https://github.com/mnoukhov/elastic-reset

----

## [152] Resolving the Tug-of-War: A Separation of Communication and Learning in Federated Learning

**Authors**: *Junyi Li, Heng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0aa800df4298539770b57824afc77a89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0aa800df4298539770b57824afc77a89-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is a promising privacy-preserving machine learning paradigm over distributed data. In this paradigm, each client trains the parameter of a model locally and the server aggregates the parameter from clients periodically. Therefore, we perform the learning and communication over the same set of parameters. However, we find that learning and communication have fundamentally divergent requirements for parameter selection, akin to two opposite teams in a tug-of-war game. To mitigate this discrepancy, we introduce FedSep, a novel two-layer federated learning framework. FedSep consists of separated communication and learning layers for each client and the two layers are connected through decode/encode operations. In particular, the decoding operation is formulated as a minimization problem. We view FedSep as a federated bilevel optimization problem and propose an efficient algorithm to solve it. Theoretically, we demonstrate that its convergence matches that of the standard FL algorithms. The separation of communication and learning in FedSep offers innovative solutions to various challenging problems in FL, such as Communication-Efficient FL and Heterogeneous-Model FL. Empirical validation shows the superior performance of FedSep over various baselines in these tasks.

----

## [153] GlucoSynth: Generating Differentially-Private Synthetic Glucose Traces

**Authors**: *Josephine Lamp, Mark Derdzinski, Christopher Hannemann, Joost van der Linden, Lu Feng, Tianhao Wang, David E. Evans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ab51646ca369140c3c3ece011b66587-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ab51646ca369140c3c3ece011b66587-Abstract-Conference.html)

**Abstract**:

We focus on the problem of generating high-quality, private synthetic glucose traces, a task generalizable to many other time series sources. Existing methods for time series data synthesis, such as those using Generative Adversarial Networks (GANs), are not able to capture the innate characteristics of glucose data and cannot provide any formal privacy guarantees without severely degrading the utility of the synthetic data. In this paper we present GlucoSynth, a novel privacy-preserving GAN framework to generate synthetic glucose traces. The core intuition behind our approach is to conserve relationships amongst motifs (glucose events) within the traces, in addition to temporal dynamics. Our framework incorporates differential privacy mechanisms to provide strong formal privacy guarantees. We provide a comprehensive evaluation on the real-world utility of the data using 1.2 million glucose traces; GlucoSynth outperforms all previous methods in its ability to generate high-quality synthetic glucose traces with strong privacy guarantees.

----

## [154] OBJECT 3DIT: Language-guided 3D-aware Image Editing

**Authors**: *Oscar Michel, Anand Bhattad, Eli VanderBilt, Ranjay Krishna, Aniruddha Kembhavi, Tanmay Gupta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b0153a91f827b14e8bfea4e211362f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b0153a91f827b14e8bfea4e211362f3-Abstract-Conference.html)

**Abstract**:

Existing image editing tools, while powerful, typically disregard the underlying 3D geometry from which the image is projected. As a result, edits made using these tools may become detached from the geometry and lighting conditions that are at the foundation of the image formation process; such edits break the portrayal of a coherent 3D world. 3D-aware generative models are a promising solution, but currently only succeed on small datasets or at the level of a single object. In this work, we formulate the new task of language-guided 3D-aware editing, where objects in an image should be edited according to a language instruction while remaining consistent with the underlying 3D scene. To promote progress towards this goal, we release OBJect: a benchmark dataset of 400K editing examples created from procedurally generated 3D scenes. Each example consists of an input image, editing instruction in language, and the edited image. We also introduce 3DIT: single and multi-task models for four editing tasks. Our models show impressive abilities to understand the 3D composition of entire scenes, factoring in surrounding objects, surfaces, lighting conditions, shadows, and physically-plausible object configurations. Surprisingly, training on only synthetic scenes from \dataset, editing capabilities of 3DIT generalize to real-world images.

----

## [155] Learning Rule-Induced Subgraph Representations for Inductive Relation Prediction

**Authors**: *Tianyu Liu, Qitan Lv, Jie Wang, Shuling Yang, Hanzhu Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b06c8673ebb453e5e468f7743d8f54e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b06c8673ebb453e5e468f7743d8f54e-Abstract-Conference.html)

**Abstract**:

Inductive relation prediction (IRP)---where entities can be different during training and inference---has shown great power for completing evolving knowledge graphs. Existing works mainly focus on using graph neural networks (GNNs) to learn the representation of the subgraph induced from the target link, which can be seen as an implicit rule-mining process to measure the plausibility of the target link. However, these methods are not able to differentiate the target link and other links during message passing, hence the final subgraph representation will contain irrelevant rule information to the target link, which reduces the reasoning performance and severely hinders the applications for real-world scenarios. To tackle this problem, we propose a novel $\textit{single-source edge-wise}$ GNN model to learn the $\textbf{R}$ule-induc$\textbf{E}$d $\textbf{S}$ubgraph represen$\textbf{T}$ations $(\textbf{REST}$), which encodes relevant rules and eliminates irrelevant rules within the subgraph. Specifically, we propose a $\textit{single-source}$ initialization approach to initialize edge features only for the target link, which guarantees the relevance of mined rules and target link. Then we propose several RNN-based functions for $\textit{edge-wise}$ message passing to model the sequential property of mined rules. REST is a simple and effective approach with theoretical support to learn the $\textit{rule-induced subgraph representation}$. Moreover, REST does not need node labeling, which significantly accelerates the subgraph preprocessing time by up to $\textbf{11.66}\times$. Experiments on inductive relation prediction benchmarks demonstrate the effectiveness of our REST.

----

## [156] Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment

**Authors**: *Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, Gal Chechik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b08d733a5d45a547344c4e9d88bb8bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b08d733a5d45a547344c4e9d88bb8bc-Abstract-Conference.html)

**Abstract**:

Text-conditioned image generation models often generate incorrect associations between entities and their visual attributes. This reflects an impaired mapping between linguistic binding of entities and modifiers in the prompt and visual binding of the corresponding elements in the generated image. As one example, a query like ``a pink sunflower and a yellow flamingo'' may incorrectly produce an image of a yellow sunflower and a pink flamingo. To remedy this issue, we propose SynGen, an approach which first syntactically analyses the prompt to identify entities and their modifiers, and then uses a novel loss function that encourages the cross-attention maps to agree with the linguistic binding reflected by the syntax. Specifically, we encourage large overlap between attention maps of entities and their modifiers, and small overlap with other entities and modifier words. The loss is optimized during inference, without retraining or fine-tuning the model. Human evaluation on three datasets, including one new and challenging set, demonstrate significant improvements of SynGen compared with current state of the art methods. This work highlights how making use of sentence structure during inference can efficiently and substantially improve the faithfulness of text-to-image generation.

----

## [157] Optimistic Natural Policy Gradient: a Simple Efficient Policy Optimization Framework for Online RL

**Authors**: *Qinghua Liu, Gellért Weisz, András György, Chi Jin, Csaba Szepesvári*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b13c22ca208bc08f3fd13793292f25f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b13c22ca208bc08f3fd13793292f25f-Abstract-Conference.html)

**Abstract**:

While policy optimization algorithms have played an important role in recent empirical success of Reinforcement Learning (RL), the existing theoretical understanding of policy optimization remains rather limited---they are either restricted to tabular MDPs or suffer from highly suboptimal sample complexity, especial in online RL where exploration is necessary. This paper proposes a simple efficient policy optimization framework---Optimistic NPG for online RL. Optimistic NPG can be viewed as simply combining of the classic natural policy gradient (NPG) algorithm [Kakade, 2001]  with optimistic policy evaluation subroutines to encourage exploration. For $d$-dimensional linear MDPs, Optimistic NPG is computationally efficient, and learns an $\epsilon$-optimal policy within  $\tilde{\mathcal{O}}(d^2/\epsilon^3)$ samples, which is the first computationally efficient algorithm whose sample complexity has the optimal dimension dependence $\tilde{\Theta}(d^2)$. It also improves over state-of-the-art results of policy optimization algorithms [Zanette et al., 2021] by a factor of $d$. For general function approximation that subsumes linear MDPs, Optimistic NPG, to our best knowledge, is also the first policy optimization algorithm that achieves the polynomial sample complexity for learning near-optimal policies.

----

## [158] Two-Stage Learning to Defer with Multiple Experts

**Authors**: *Anqi Mao, Christopher Mohri, Mehryar Mohri, Yutao Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b17d256cf1fe1cc084922a8c6b565b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b17d256cf1fe1cc084922a8c6b565b7-Abstract-Conference.html)

**Abstract**:

We study a two-stage scenario for learning to defer with multiple experts, which is crucial in practice for many applications.  In this scenario, a predictor is derived in a first stage by training with a common loss function such as cross-entropy.  In the second stage, a deferral function is learned to assign the most suitable expert to each input.  We design a new family of surrogate loss functions for this scenario both in the score-based and the predictor-rejector settings and prove that they are supported by $H$-consistency bounds, which implies their Bayes-consistency.  Moreover, we show that, for a constant cost function, our two-stage surrogate losses are realizable $H$-consistent. While the main focus of this work is a theoretical analysis, we also report the results of several experiments on CIFAR-10 and SVHN datasets.

----

## [159] A Computationally Efficient Sparsified Online Newton Method

**Authors**: *Devvrit, Sai Surya Duvvuri, Rohan Anil, Vineet Gupta, Cho-Jui Hsieh, Inderjit S. Dhillon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b43289db08ed60edc6451cb2132e203-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b43289db08ed60edc6451cb2132e203-Abstract-Conference.html)

**Abstract**:

Second-order methods hold significant promise for enhancing the convergence of deep neural network training; however, their large memory and computational demands have limited their practicality. Thus there is a need for scalable second-order methods that can efficiently train large models. In this paper, we introduce the Sparsified Online Newton~(SONew) method, a memory-efficient second-order algorithm that yields a sparsified yet effective preconditioner. The algorithm emerges from a novel use of the LogDet matrix divergence measure; we combine it with sparsity constraints to minimize regret in the online convex optimization framework. Empirically, we test our method on large scale benchmarks of up to 1B parameters. We achieve up to $30\\%$ faster convergence, $3.4\\%$ relative improvement in validation performance, and $80\\%$ relative improvement in training loss, in comparison to memory efficient optimizers including first order methods. Powering the method is a surprising fact -- imposing structured sparsity patterns, like tridiagonal and banded structure, requires little to no overhead, making it as efficient and parallelizable as first-order methods. In wall-clock time, tridiagonal SONew is only about $3\\%$ slower per step than first-order methods but gives overall gains due to much faster convergence. In contrast, one of the state-of-the-art (SOTA) memory-intensive second-order methods, Shampoo, is unable to scale to large benchmarks. Additionally, while Shampoo necessitates significant engineering efforts to scale to large benchmarks, SONew offers a more straightforward implementation, increasing its practical appeal. SONew code is available at: https://github.com/devvrit/SONew

----

## [160] SparseProp: Efficient Event-Based Simulation and Training of Sparse Recurrent Spiking Neural Networks

**Authors**: *Rainer Engelken*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b443d358a391166d1fbf551fb53de02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b443d358a391166d1fbf551fb53de02-Abstract-Conference.html)

**Abstract**:

Spiking Neural Networks (SNNs) are biologically-inspired models that are capable of processing information in streams of action potentials. However, simulating and training SNNs is computationally expensive due to the need to solve large systems of coupled differential equations. In this paper, we propose a novel event-based algorithm called SparseProp for simulating and training sparse SNNs. Our algorithm reduces the computational cost of both forward pass and backward pass operations from O(N) to O(log(N)) per network spike, enabling numerically exact simulations of large spiking networks and their efficient training using backpropagation through time. By exploiting the sparsity of the network, SparseProp avoids iterating through all neurons at every spike and uses efficient state updates. We demonstrate the effectiveness of SparseProp for several classical integrate-and-fire neuron models, including simulating a sparse SNN with one million LIF neurons, which is sped up by more than four orders of magnitude compared to previous implementations. Our work provides an efficient and exact solution for training large-scale spiking neural networks and opens up new possibilities for building more sophisticated brain-inspired models.

----

## [161] ConRad: Image Constrained Radiance Fields for 3D Generation from a Single Image

**Authors**: *Senthil Purushwalkam, Nikhil Naik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b68d474baf8dff30f3280c199a32089-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b68d474baf8dff30f3280c199a32089-Abstract-Conference.html)

**Abstract**:

We present a novel method for reconstructing 3D objects from a single RGB image. Our method leverages the latest image generation models to infer the hidden 3D structure while remaining faithful to the input image. While existing methods obtain impressive results in generating 3D models from text prompts, they do not provide an easy approach for conditioning on input RGB data. Naive extensions of these methods often lead to improper alignment in appearance between the input image and the 3D reconstructions. We address these challenges by introducing Image Constrained Radiance Fields (ConRad), a novel variant of neural radiance fields. ConRad is an efficient 3D representation that explicitly captures the appearance of an input image in one viewpoint. We propose a training algorithm that leverages the single RGB image in conjunction with pretrained Diffusion Models to optimize the parameters of a ConRad representation. Extensive experiments show that ConRad representations can simplify preservation of image details while producing a realistic 3D reconstruction. Compared to existing state-of-the-art baselines, we show that our 3D reconstructions remain more faithful to the input and produce more consistent 3D models while demonstrating significantly improved quantitative performance on a ShapeNet object benchmark.

----

## [162] Fair Canonical Correlation Analysis

**Authors**: *Zhuoping Zhou, Davoud Ataee Tarzanagh, Bojian Hou, Boning Tong, Jia Xu, Yanbo Feng, Qi Long, Li Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0b8e4c8468273ee3bafb288229c0acbc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0b8e4c8468273ee3bafb288229c0acbc-Abstract-Conference.html)

**Abstract**:

This paper investigates fairness and bias in Canonical Correlation Analysis (CCA), a widely used statistical technique for examining the relationship between two sets of variables. We present a framework that alleviates unfairness by minimizing the correlation disparity error associated with protected attributes. Our approach enables CCA to learn global projection matrices from all data points while ensuring that these matrices yield comparable correlation levels to group-specific projection matrices. Experimental evaluation on both synthetic and real-world datasets demonstrates the efficacy of our method in reducing correlation disparity error without compromising CCA accuracy.

----

## [163] DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization

**Authors**: *Zhiqing Sun, Yiming Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html)

**Abstract**:

Neural network-based Combinatorial Optimization (CO) methods have shown promising results in solving various NP-complete (NPC) problems without relying on hand-crafted domain knowledge. This paper broadens the current scope of neural solvers for NPC problems by introducing a new graph-based diffusion framework, namely DIFUSCO. It formulates NPC problems into a discrete {0, 1}-vector space and uses graph-based denoising diffusion models to generate high-quality solutions. Specifically, we explore diffusion models with Gaussian and Bernoulli noise, respectively, and also introduce an effective inference schedule to improve the generation quality. We evaluate our methods on two well-studied combinatorial optimization problems: Traveling Salesman Problem (TSP) and Maximal Independent Set (MIS). Experimental results show that DIFUSCO strongly outperforms the previous state-of-the-art neural solvers, improving the performance gap between ground-truth and neural solvers from 1.76% to 0.46% on TSP-500, from 2.46% to 1.17% on TSP-1000, and from 3.19% to 2.58% on TSP-10000. For the MIS problem, DIFUSCO outperforms the previous state-of-the-art neural solver on the challenging SATLIB benchmark. Our code is available at this url.

----

## [164] Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models

**Authors**: *George Stein, Jesse C. Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Leigh Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L. Caterini, J. Eric T. Taylor, Gabriel Loaiza-Ganem*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0bc795afae289ed465a65a3b4b1f4eb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0bc795afae289ed465a65a3b4b1f4eb7-Abstract-Conference.html)

**Abstract**:

We systematically study a wide variety of generative models spanning semantically-diverse image datasets to understand and improve the feature extractors and metrics used to evaluate them.Using best practices in psychophysics, we measure human perception of image realism for generated samples by conducting the largest experiment evaluating generative models to date, and find that no existing metric strongly correlates with human evaluations.Comparing to 17 modern metrics for evaluating the overall performance, fidelity, diversity, rarity, and memorization of generative models, we find that the state-of-the-art perceptual realism of diffusion models as judged by humans is not reflected in commonly reported metrics such as FID. This discrepancy is not explained by diversity in generated samples, though one cause is over-reliance on Inception-V3.We address these flaws through a study of alternative self-supervised feature extractors, find that the semantic information encoded by individual networks strongly depends on their training procedure, and show that DINOv2-ViT-L/14 allows for much richer evaluation of generative models. Next, we investigate data memorization, and find that generative models do memorize training examples on simple, smaller datasets like CIFAR10, but not necessarily on more complex datasets like ImageNet. However, our experiments show that current metrics do not properly detect memorization: none in the literature is able to separate memorization from other phenomena such as underfitting or mode shrinkage. To facilitate further development of generative models and their evaluation we release all generated image datasets, human evaluation data, and a modular library to compute 17 common metrics for 9 different encoders at https://github.com/layer6ai-labs/dgm-eval.

----

## [165] Online Clustering of Bandits with Misspecified User Models

**Authors**: *Zhiyong Wang, Jize Xie, Xutong Liu, Shuai Li, John C. S. Lui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0bcd8d153b8c548629eca53f4ebdeb42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0bcd8d153b8c548629eca53f4ebdeb42-Abstract-Conference.html)

**Abstract**:

The contextual linear bandit is an important online learning problem where given arm features, a learning agent selects an arm at each round to maximize the cumulative rewards in the long run. A line of works, called the clustering of bandits (CB), utilize the collaborative effect over user preferences and have shown significant improvements over classic linear bandit algorithms. However, existing CB algorithms require well-specified linear user models and can fail when this critical assumption does not hold. Whether robust CB algorithms can be designed for more practical scenarios with misspecified user models remains an open problem. In this paper, we are the first to present the important problem of clustering of bandits with misspecified user models (CBMUM), where the expected rewards in user models can be perturbed away from perfect linear models. We devise two robust CB algorithms, RCLUMB and RSCLUMB (representing the learned clustering structure with dynamic graph and sets, respectively), that can accommodate the inaccurate user preference estimations and erroneous clustering caused by model misspecifications. We prove regret upper bounds of $O(\epsilon_*T\sqrt{md\log T}  + d\sqrt{mT}\log T)$ for our algorithms under milder assumptions than previous CB works, which match the lower bound asymptotically in $T$ up to logarithmic factors, and also match the state-of-the-art results in several degenerate cases. Our regret analysis is novel and different from the typical proof flow of previous CB works. The techniques in proving the regret caused by misclustering users are quite general and may be of independent interest. Experiments on both synthetic and real-world data show our outperformance over previous algorithms.

----

## [166] Temporal Conditioning Spiking Latent Variable Models of the Neural Response to Natural Visual Scenes

**Authors**: *Gehua Ma, Runhao Jiang, Rui Yan, Huajin Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0bcf9cf6ffe26bba3af99e18be0e1d8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0bcf9cf6ffe26bba3af99e18be0e1d8d-Abstract-Conference.html)

**Abstract**:

Developing computational models of neural response is crucial for understanding sensory processing and neural computations. Current state-of-the-art neural network methods use temporal filters to handle temporal dependencies, resulting in an unrealistic and inflexible processing paradigm. Meanwhile, these methods target trial-averaged firing rates and fail to capture important features in spike trains. This work presents the temporal conditioning spiking latent variable models (TeCoS-LVM) to simulate the neural response to natural visual stimuli. We use spiking neurons to produce spike outputs that directly match the recorded trains. This approach helps to avoid losing information embedded in the original spike trains. We exclude the temporal dimension from the model parameter space and introduce a temporal conditioning operation to allow the model to adaptively explore and exploit temporal dependencies in stimuli sequences in a natural paradigm. We show that TeCoS-LVM models can produce more realistic spike activities and accurately fit spike statistics than powerful alternatives. Additionally, learned TeCoS-LVM models can generalize well to longer time scales. Overall, while remaining computationally tractable, our model effectively captures key features of neural coding systems. It thus provides a useful tool for building accurate predictive computational accounts for various sensory perception circuits.

----

## [167] Double Auctions with Two-sided Bandit Feedback

**Authors**: *Soumya Basu, Abishek Sankararaman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0bcfb525c8f8f07ae10a93d0b2a40e00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0bcfb525c8f8f07ae10a93d0b2a40e00-Abstract-Conference.html)

**Abstract**:

Double Auction enables decentralized transfer of goods between multiple buyers and sellers, thus underpinning functioning of many online marketplaces. Buyers and sellers compete in these markets through bidding, but do not often know their own valuation a-priori. As the allocation and pricing happens through bids, the profitability of participants, hence sustainability of such markets, depends crucially on learning respective valuations through repeated interactions. We initiate the study of Double Auction markets under bandit feedback on both buyers' and sellers' side. We show with confidence bound based bidding, and `Average Pricing' there is an efficient price discovery among the participants.  In particular, the regret on combined valuation of the buyers and the sellers -- a.k.a. the social regret -- is $O(\log(T)/\Delta)$ in $T$ rounds, where $\Delta$ is the minimum price gap. Moreover, the buyers and sellers exchanging goods attain $O(\sqrt{T})$ regret, individually. The buyers and sellers who do not benefit from exchange in turn only experience $O(\log{T}/ \Delta)$  regret individually in $T$ rounds.  We augment our upper bound by showing that $\omega(\sqrt{T})$ individual regret, and $\omega(\log{T})$ social regret is unattainable in certain Double Auction markets. Our paper is the first to provide decentralized learning algorithms in a two-sided market where \emph{both sides have uncertain preference} that need to be learned.

----

## [168] Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking

**Authors**: *Juanhui Li, Harry Shomer, Haitao Mao, Shenglai Zeng, Yao Ma, Neil Shah, Jiliang Tang, Dawei Yin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0be50b4590f1c5fdf4c8feddd63c4f67-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0be50b4590f1c5fdf4c8feddd63c4f67-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Link prediction attempts to predict whether an unseen edge exists based on only a portion of the graph. A flurry of methods has been created in recent years that attempt to make use of graph neural networks (GNNs) for this task. Furthermore, new and diverse datasets have also been created to better evaluate the effectiveness of these new models. However, multiple limitations currently exist that hinders our ability to properly evaluate these new methods. This includes, but is not limited to: (1) The underreporting of performance on multiple baselines, (2) A lack of a unified data split and evaluation metric on some datasets, (3) An unrealistic evaluation setting that produces negative samples that are easy to classify. To overcome these challenges we first conduct a fair comparison across prominent methods and datasets, utilizing the same dataset settings and hyperparameter settings. We then create a new real-world evaluation setting that samples difficult negative samples via multiple heuristics. The new evaluation setting helps promote new challenges and opportunities in link prediction by aligning the evaluation with real-world situations.

----

## [169] EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images

**Authors**: *Seongsu Bae, Daeun Kyung, Jaehee Ryu, Eunbyeol Cho, Gyubok Lee, Sunjun Kweon, Jungwoo Oh, Lei Ji, Eric I-Chao Chang, Tackeun Kim, Edward Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c007ebef1d11fd48da6ce4f54687db6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c007ebef1d11fd48da6ce4f54687db6-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Electronic Health Records (EHRs), which contain patients' medical histories in various multi-modal formats, often overlook the potential for joint reasoning across imaging and table modalities underexplored in current EHR Question Answering (QA) systems. In this paper, we introduce EHRXQA, a novel multi-modal question answering dataset combining structured EHRs and chest X-ray images. To develop our dataset, we first construct two uni-modal resources: 1) The MIMIC- CXR-VQA dataset, our newly created medical visual question answering (VQA) benchmark, specifically designed to augment the imaging modality in EHR QA, and 2) EHRSQL (MIMIC-IV), a refashioned version of a previously established table-based EHR QA dataset. By integrating these two uni-modal resources, we successfully construct a multi-modal EHR QA dataset that necessitates both uni-modal and cross-modal reasoning. To address the unique challenges of multi-modal questions within EHRs, we propose a NeuralSQL-based strategy equipped with an external VQA API. This pioneering endeavor enhances engagement with multi-modal EHR sources and we believe that our dataset can catalyze advances in real-world medical scenarios such as clinical decision-making and research. EHRXQA is available at https://github.com/baeseongsu/ehrxqa.

----

## [170] Enhancing Robot Program Synthesis Through Environmental Context

**Authors**: *Tianyi Chen, Qidi Wang, Zhen Dong, Liwei Shen, Xin Peng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c1e94af650f5c74b1f3da467c2308c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c1e94af650f5c74b1f3da467c2308c2-Abstract-Conference.html)

**Abstract**:

Program synthesis aims to automatically generate an executable program that conforms to the given specification. Recent advancements have demonstrated that deep neural methodologies and large-scale pretrained language models are highly proficient in capturing program semantics.For robot programming, prior works have facilitated program synthesis by incorporating global environments. However, the assumption of acquiring a comprehensive understanding of the entire environment is often excessively challenging to achieve.In this work, we present a framework that learns to synthesize a program by rectifying potentially erroneous code segments, with the aid of partially observed environments. To tackle the issue of inadequate attention to partial observations, we propose to first learn an environment embedding space that can implicitly evaluate the impacts of each program token based on the precondition. Furthermore, by employing a graph structure, the model can aggregate both environmental and syntactic information flow and furnish smooth program rectification guidance.Extensive experimental evaluations and ablation studies on the partially observed VizDoom domain authenticate that our method offers superior generalization capability across various tasks and greater robustness when encountering noises.

----

## [171] ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling

**Authors**: *Quanyi Li, Zhenghao Mark Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo, Bolei Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c26a501df8fb919a0350e2df06b5d39-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c26a501df8fb919a0350e2df06b5d39-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large-scale driving datasets such as Waymo Open Dataset and nuScenes substantially accelerate autonomous driving research, especially for perception tasks such as 3D detection and trajectory forecasting. Since the driving logs in these datasets contain HD maps and detailed object annotations which accurately reflect the real-world complexity of traffic behaviors, we can harvest a massive number of complex traffic scenarios and recreate their digital twins in simulation. Compared to the hand-crafted scenarios often used in existing simulators, data-driven scenarios collected from the real world can facilitate many research opportunities in machine learning and autonomous driving. In this work, we present ScenarioNet, an open-source platform for large-scale traffic scenario modeling and simulation. ScenarioNet defines a unified scenario description format and collects a large-scale repository of real-world traffic scenarios from the heterogeneous data in various driving datasets including Waymo, nuScenes, Lyft L5, and nuPlan datasets. These scenarios can be further replayed and interacted with in multiple views from Bird-Eye-View layout to realistic 3D rendering in MetaDrive simulator. This provides a benchmark for evaluating the safety of autonomous driving stacks in simulation before their real-world deployment. We further demonstrate the strengths of ScenarioNet on large-scale scenario generation, imitation learning, and reinforcement learning in both single-agent and multi-agent settings. Code, demo videos, and website are available at https://github.com/metadriverse/scenarionet

----

## [172] Understanding Deep Gradient Leakage via Inversion Influence Functions

**Authors**: *Haobo Zhang, Junyuan Hong, Yuyang Deng, Mehrdad Mahdavi, Jiayu Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c4dd7e3d9f528f0b4f2aca9fbcdca8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c4dd7e3d9f528f0b4f2aca9fbcdca8d-Abstract-Conference.html)

**Abstract**:

Deep Gradient Leakage (DGL) is a highly effective attack that recovers private training images from gradient vectors.This attack casts significant privacy challenges on distributed learning from clients with sensitive data, where clients are required to share gradients.  Defending against such attacks requires but lacks an understanding of when and how privacy leakage happens, mostly because of the black-box nature of deep networks.  In this paper, we propose a novel Inversion Influence Function (I$^2$F) that establishes a closed-form connection between the recovered images and the private gradients by implicitly solving the DGL problem.  Compared to directly solving DGL, I$^2$F is scalable for analyzing deep networks, requiring only oracle access to gradients and Jacobian-vector products.  We empirically demonstrate that I$^2$F effectively approximated the DGL generally on different model architectures, datasets, modalities, attack implementations, and perturbation-based defenses.  With this novel tool, we provide insights into effective gradient perturbation directions, the unfairness of privacy protection, and privacy-preferred model initialization.  Our codes are provided in https://github.com/illidanlab/inversion-influence-function.

----

## [173] Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization

**Authors**: *Shurui Gui, Meng Liu, Xiner Li, Youzhi Luo, Shuiwang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c6c92a0c5237761168eafd4549f1584-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c6c92a0c5237761168eafd4549f1584-Abstract-Conference.html)

**Abstract**:

We tackle the problem of graph out-of-distribution (OOD) generalization. Existing graph OOD algorithms either rely on restricted assumptions or fail to exploit environment information in training data. In this work, we propose to simultaneously incorporate label and environment causal independence (LECI) to fully make use of label and environment information, thereby addressing the challenges faced by prior methods on identifying causal and invariant subgraphs. We further develop an adversarial training strategy to jointly optimize these two properties for casual subgraph discovery with theoretical guarantees. Extensive experiments and analysis show that LECI significantly outperforms prior methods on both synthetic and real-world datasets, establishing LECI as a practical and effective solution for graph OOD generalization.

----

## [174] Bayesian Learning of Optimal Policies in Markov Decision Processes with Countably Infinite State-Space

**Authors**: *Saghar Adler, Vijay G. Subramanian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html)

**Abstract**:

Models of many real-life applications, such as queueing models of communication networks or computing systems, have a countably infinite state-space. Algorithmic and learning procedures that have been developed to produce optimal policies mainly focus on finite state settings, and do not directly apply to these models. To overcome this lacuna, in this work we study the problem of optimal control of a family of discrete-time countable state-space Markov Decision Processes (MDPs) governed by an unknown parameter $\theta\in\Theta$,  and defined on a countably-infinite state-space $\mathcal X=\mathbb{Z}_+^d$, with finite action space $\mathcal A$, and an unbounded cost function. We take a Bayesian perspective with the random unknown parameter $\boldsymbol{\theta}^*$ generated via a given fixed prior distribution on $\Theta$. To optimally control the unknown MDP, we propose an algorithm based on Thompson sampling with dynamically-sized episodes: at the beginning of each episode, the posterior distribution formed via Bayes' rule is used to produce a parameter estimate, which then decides the policy applied during the episode. To ensure the stability of the Markov chain obtained by following the policy chosen for each parameter, we impose ergodicity assumptions. From this condition and using the solution of the average cost Bellman equation, we establish an $\tilde O(dh^d\sqrt{|\mathcal A|T})$ upper bound on the Bayesian regret of our algorithm, where $T$ is the time-horizon. Finally, to elucidate the applicability of our algorithm, we consider two different queueing models with unknown dynamics, and show that our algorithm can be applied to develop approximately optimal control algorithms.

----

## [175] CARE: Modeling Interacting Dynamics Under Temporal Environmental Variation

**Authors**: *Xiao Luo, Haixin Wang, Zijie Huang, Huiyu Jiang, Abhijeet Gangan, Song Jiang, Yizhou Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c7ca207a051228f978971447a56464a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c7ca207a051228f978971447a56464a-Abstract-Conference.html)

**Abstract**:

Modeling interacting dynamical systems, such as fluid dynamics and intermolecular interactions, is a fundamental research problem for understanding and simulating complex real-world systems. Many of these systems can be naturally represented by dynamic graphs, and graph neural network-based approaches have been proposed and shown promising performance. However, most of these approaches assume the underlying dynamics does not change over time, which is unfortunately untrue. For example, a molecular dynamics can be affected by the environment temperature over the time. In this paper, we take an attempt to provide a probabilistic view for time-varying dynamics and propose a model Context-attended Graph ODE (CARE) for modeling time-varying interacting dynamical systems. In our CARE, we explicitly use a context variable to model time-varying environment and construct an encoder to initialize the context variable from historical trajectories. Furthermore, we employ a neural ODE model to depict the dynamic evolution of the context variable inferred from system states. This context variable is incorporated into a coupled ODE to simultaneously drive the evolution of systems. Comprehensive experiments on four datasets demonstrate the effectiveness of our proposed CARE compared with several state-of-the-art approaches.

----

## [176] Diffused Redundancy in Pre-trained Representations

**Authors**: *Vedant Nanda, Till Speicher, John P. Dickerson, Krishna P. Gummadi, Soheil Feizi, Adrian Weller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0c86142265c5e2c900613dd1d031cb90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0c86142265c5e2c900613dd1d031cb90-Abstract-Conference.html)

**Abstract**:

Representations learned by pre-training a neural network on a large dataset are increasingly used successfully to perform a variety of downstream tasks. In this work, we take a closer look at how features are encoded in such pre-trained representations. We find that learned representations in a given layer exhibit a degree of diffuse redundancy, ie, any randomly chosen subset of neurons in the layer that is larger than a threshold size shares a large degree of similarity with the full layer and is able to perform similarly as the whole layer on a variety of downstream tasks. For example, a linear probe trained on $20\%$ of randomly picked neurons from the penultimate layer of a ResNet50 pre-trained on ImageNet1k achieves an accuracy within $5\%$ of a linear probe trained on the full layer of neurons for downstream CIFAR10 classification. We conduct experiments on different neural architectures (including CNNs and Transformers) pre-trained on both ImageNet1k and ImageNet21k and evaluate a variety of downstream tasks taken from the VTAB benchmark. We find that the loss \& dataset used during pre-training largely govern the degree of diffuse redundancy and the "critical mass" of neurons needed often depends on the downstream task, suggesting that there is a task-inherent redundancy-performance Pareto frontier. Our findings shed light on the nature of representations learned by pre-trained deep neural networks and suggest that entire layers might not be necessary to perform many downstream tasks. We investigate the potential for exploiting this redundancy to achieve efficient generalization for downstream tasks and also draw caution to certain possible unintended consequences. Our code is available at \url{https://github.com/nvedant07/diffused-redundancy}.

----

## [177] AI for Interpretable Chemistry: Predicting Radical Mechanistic Pathways via Contrastive Learning

**Authors**: *Mohammadamin Tavakoli, Pierre Baldi, Ann Marie Carlton, Yin Ting T. Chiu, Alexander Shmakov, David Van Vranken*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ca70969597da7166128f7755c64ffd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ca70969597da7166128f7755c64ffd5-Abstract-Conference.html)

**Abstract**:

Deep learning-based reaction predictors have undergone significant architectural evolution. However, their reliance on reactions from the US Patent Office results in a lack of interpretable predictions and limited generalizability to other chemistry domains, such as radical and atmospheric chemistry. To address these challenges, we introduce a new reaction predictor system, RMechRP, that leverages contrastive learning in conjunction with mechanistic pathways, the most interpretable representation of chemical reactions. Specifically designed for radical reactions, RMechRP provides different levels of interpretation of chemical reactions. We develop and train multiple deep-learning models using RMechDB, a public database of radical reactions, to establish the first benchmark for predicting radical reactions. Our results demonstrate the effectiveness of RMechRP in providing accurate and interpretable predictions of radical reactions, and its potential for various applications in atmospheric chemistry.

----

## [178] Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks

**Authors**: *Jules Berman, Benjamin Peherstorfer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0cb310ed8121549488fea8e8c2056096-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0cb310ed8121549488fea8e8c2056096-Abstract-Conference.html)

**Abstract**:

Training neural networks sequentially in time to approximate solution fields of time-dependent partial differential equations can be beneficial for preserving causality and other physics properties; however, the sequential-in-time training is numerically challenging because training errors quickly accumulate and amplify over time. This work introduces Neural Galerkin schemes that update randomized sparse subsets of network parameters at each time step. The randomization avoids overfitting locally in time and so helps prevent the error from accumulating quickly over the sequential-in-time training, which is motivated by dropout that addresses a similar issue of overfitting due to neuron co-adaptation. The sparsity of the update reduces the computational costs of training without losing expressiveness because many of the network parameters are redundant locally at each time step. In numerical experiments with a wide range of evolution equations, the proposed scheme with randomized sparse updates is up to two orders of magnitude more accurate at a fixed computational budget and up to two orders of magnitude faster at a fixed accuracy than schemes with dense updates.

----

## [179] Handling Data Heterogeneity via Architectural Design for Federated Visual Recognition

**Authors**: *Sara Pieri, Jose Renato Restom, Samuel Horváth, Hisham Cholakkal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ccd06ff26fd6a7829293ce90e0e7f7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ccd06ff26fd6a7829293ce90e0e7f7d-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) is a promising research paradigm that enables the collaborative training of machine learning models among various parties without the need for sensitive information exchange. Nonetheless, retaining data in individual clients introduces fundamental challenges to achieving performance on par with centrally trained models. Our study provides an extensive review of federated learning applied to visual recognition. It underscores the critical role of thoughtful architectural design choices in achieving optimal performance, a factor often neglected in the FL literature. Many existing FL solutions are tested on shallow or simple networks, which may not accurately reflect real-world applications. This practice restricts the transferability of research findings to large-scale visual recognition models. Through an in-depth analysis of diverse cutting-edge architectures such as convolutional neural networks, transformers, and MLP-mixers,  we experimentally demonstrate that architectural choices can substantially enhance FL systems' performance, particularly when handling heterogeneous data.  We study visual recognition models from five different architectural families on four challenging FL datasets. We also re-investigate the inferior performance convolution-based architectures in the FL setting and analyze the influence of normalization layers on the FL performance. Our findings emphasize the importance of architectural design for computer vision tasks in practical scenarios, effectively narrowing the performance gap between federated and centralized learning.

----

## [180] Spatial-frequency channels, shape bias, and adversarial robustness

**Authors**: *Ajay Subramanian, Elena Sizikova, Najib J. Majaj, Denis G. Pelli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0cdc1e85736d9c01d366cbf9b4b81672-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0cdc1e85736d9c01d366cbf9b4b81672-Abstract-Conference.html)

**Abstract**:

What spatial frequency information do humans and neural networks use to recognize objects? In neuroscience, critical band masking is an established tool that can reveal the frequency-selective filters used for object recognition. Critical band masking measures the sensitivity of recognition performance to noise added at each spatial frequency. Existing critical band masking studies show that humans recognize periodic patterns (gratings) and letters by means of a spatial-frequency filter (or "channel") that has a frequency bandwidth of one octave (doubling of frequency). Here, we introduce critical band masking as a task for network-human comparison and test 14 humans and 76 neural networks on 16-way ImageNet categorization in the presence of narrowband noise. We find that humans recognize objects in natural images using the same one-octave-wide channel that they use for letters and gratings, making it a canonical feature of human object recognition. Unlike humans, the neural network channel is very broad, 2-4 times wider than the human channel. This means that the network channel extends to frequencies higher and lower than those that humans are sensitive to. Thus, noise at those frequencies will impair network performance and spare human performance. Adversarial and augmented-image training are commonly used to increase network robustness and shape bias. Does this training align network and human object recognition channels? Three network channel properties (bandwidth, center frequency, peak noise sensitivity) correlate strongly with shape bias (51% variance explained) and robustness of adversarially-trained networks (66% variance explained). Adversarial training increases robustness but expands the channel bandwidth even further beyond the human bandwidth. Thus, critical band masking reveals that the network channel is more than twice as wide as the human channel, and that adversarial training only makes it worse. Networks with narrower channels might be more robust.

----

## [181] Optimality in Mean Estimation: Beyond Worst-Case, Beyond Sub-Gaussian, and Beyond 1+α Moments

**Authors**: *Trung Dang, Jasper C. H. Lee, Maoyuan Raymond Song, Paul Valiant*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0cddb777d3441326544e21b67f41bdc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0cddb777d3441326544e21b67f41bdc8-Abstract-Conference.html)

**Abstract**:

There is growing interest in improving our algorithmic understanding of fundamental statistical problems such as mean estimation, driven by the goal of understanding the fundamental limits of what we can extract from limited and valuable data.The state of the art results for mean estimation in $\mathbb{R}$ are 1) the optimal sub-Gaussian mean estimator by [Lee and Valiant, 2022], attaining the optimal sub-Gaussian error constant for all distributions with finite but unknown variance, and 2) the analysis of the median-of-means algorithm by [Bubeck, Cesa-Bianchi and Lugosi, 2013] and a matching lower bound by [Devroye, Lerasle, Lugosi, and Oliveira, 2016], characterizing the big-O optimal errors for distributions that have tails heavy enough that only a $1+\alpha$ moment exists for some $\alpha \in (0,1)$.Both of these results, however, are optimal only in the worst case.Motivated by the recent effort in the community to go "beyond the worst-case analysis" of algorithms, we initiate the fine-grained study of the mean estimation problem:Is it possible for algorithms to leverage *beneficial* features/quirks of their input distribution to *beat* the sub-Gaussian rate, without explicit knowledge of these features?We resolve this question, finding an unexpectedly nuanced answer: "Yes in limited regimes, but in general no".Given a distribution $p$, assuming *only* that it has a finite mean and absent any additional assumptions,we show how to construct a distribution $q_{n,\delta}$ such that the means of $p$ and $q$ are well-separated, yet $p$ and $q$ are impossible to distinguish with $n$ samples with probability $1-\delta$, and $q$ further preserves the finiteness of moments of $p$.Moreover, the variance of $q$ is at most twice the variance of $p$ if it exists.The main consequence of our result is that, no reasonable estimator can asymptotically achieve better than the sub-Gaussian error rate for any distribution, up to constant factors, which matches the worst-case result of [Lee and Valiant, 2022].More generally, we introduce a new definitional framework to analyze the fine-grained optimality of algorithms, which we call "neighborhood optimality", interpolating between the unattainably strong "instance optimality" and the trivially weak admissibility/Pareto optimality definitions.As an application of the new framework, we show that the median-of-means algorithm is neighborhood optimal, up to constant factors.It is an open question to find a neighborhood-optimal estimator *without* constant factor slackness.

----

## [182] Provably Efficient Offline Goal-Conditioned Reinforcement Learning with General Function Approximation and Single-Policy Concentrability

**Authors**: *Hanlin Zhu, Amy Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0cfc9404f89400c5ed897035e0d3748c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0cfc9404f89400c5ed897035e0d3748c-Abstract-Conference.html)

**Abstract**:

Goal-conditioned reinforcement learning (GCRL) refers to learning general-purpose skills that aim to reach diverse goals. In particular, offline GCRL only requires purely pre-collected datasets to perform training tasks without additional interactions with the environment. Although offline GCRL has become increasingly prevalent and many previous works have demonstrated its empirical success, the theoretical understanding of efficient offline GCRL algorithms is not well established, especially when the state space is huge and the offline dataset only covers the policy we aim to learn. In this paper, we provide a rigorous theoretical analysis of an existing empirically successful offline GCRL algorithm. We prove that under slight modification, this algorithm enjoys an $\tilde{O}(\text{poly}(1/\epsilon))$ sample complexity (where $\epsilon$ is the desired suboptimality of the learned policy) with general function approximation thanks to the property of (semi-)strong convexity of the objective functions. We only require nearly minimal assumptions on the dataset (single-policy concentrability) and the function class (realizability). Moreover, this algorithm consists of two uninterleaved optimization steps, which we refer to as $V$-learning and policy learning, and is computationally stable since it does not involve minimax optimization. We also empirically validate our theory by showing that the modified algorithm outperforms the previous algorithm in various real-world environments.To the best of our knowledge, this is the first algorithm that is both provably efficient with general function approximation and single-policy concentrability, and empirically successful without requiring solving minimax optimization problems.

----

## [183] SQ Lower Bounds for Non-Gaussian Component Analysis with Weaker Assumptions

**Authors**: *Ilias Diakonikolas, Daniel Kane, Lisheng Ren, Yuxin Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d00a699f60e642b310eb04b76cc7731-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d00a699f60e642b310eb04b76cc7731-Abstract-Conference.html)

**Abstract**:

We study the complexity of Non-Gaussian Component Analysis (NGCA) in the Statistical Query (SQ) model.Prior work developed a methodology to prove SQ lower bounds for NGCA that have been applicable to a wide range of contexts.In particular, it was known that for any univariate distribution $A$ satisfying certain conditions,distinguishing between a standard multivariate Gaussian and a distribution that behaves like $A$ in a random hidden direction and like a standard Gaussian in the orthogonal complement, is SQ-hard.The required conditions were that (1) $A$ matches many low-order moments with a standard Gaussian,and (2) the chi-squared norm of $A$ with respect to the standard Gaussian is finite.While the moment-matching condition is clearly necessary for hardness, the chi-squared condition was only required for technical reasons.In this work, we establish that the latter condition is indeed not necessary.In particular, we prove near-optimal SQ lower bounds for NGCA under the moment-matching condition only.

----

## [184] Efficient Equivariant Transfer Learning from Pretrained Models

**Authors**: *Sourya Basu, Pulkit Katdare, Prasanna Sattigeri, Vijil Chenthamarakshan, Katherine Driggs Campbell, Payel Das, Lav R. Varshney*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d02892a0055c94584f6394f8d069c8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d02892a0055c94584f6394f8d069c8e-Abstract-Conference.html)

**Abstract**:

Efficient transfer learning algorithms are key to the success of foundation models on diverse downstream tasks even with limited data. Recent works of Basu et al. (2023) and Kaba et al. (2022) propose group averaging (equitune) and optimization-based methods, respectively, over features from group-transformed inputs to obtain equivariant outputs from non-equivariant neural networks. While Kaba et al. (2022) are only concerned with training from scratch, we find that equitune performs poorly on equivariant zero-shot tasks despite good finetuning results. We hypothesize that this is because pretrained models provide better quality features for certain transformations than others and simply averaging them is deleterious. Hence, we propose 位-equitune that averages the features using importance weights, 位s. These weights are learned directly from the data using a small neural network, leading to excellent zero-shot and finetuned results that outperform equitune. Further, we prove that 位-equitune is equivariant and a universal approximator of equivariant functions. Additionally, we show that the method of Kaba et al. (2022) used with appropriate loss functions, which we call equizero, also gives excellent zero-shot and finetuned performance. Both equitune and equizero are special cases of 位- equitune. To show the simplicity and generality of our method, we validate on a wide range of diverse applications and models such as 1) image classification using CLIP, 2) deep Q-learning, 3) fairness in natural language generation (NLG), 4) compositional generalization in languages, and 5) image classification using pretrained CNNs such as Resnet and Alexnet.

----

## [185] Kernelized Reinforcement Learning with Order Optimal Regret Bounds

**Authors**: *Sattar Vakili, Julia Olkhovskaya*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d17d033059bacd127f25ab28784f829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d17d033059bacd127f25ab28784f829-Abstract-Conference.html)

**Abstract**:

Modern reinforcement learning (RL) has shown empirical success in various real world settings with complex models and large state-action spaces. The existing analytical results, however, typically focus on settings with a small number of state-actions or simple models such as linearly modeled state-action value functions. To derive RL policies that efficiently handle large state-action spaces with more general value functions, some recent works have considered nonlinear function approximation using kernel ridge regression. We propose $\pi$-KRVI, an optimistic modification of least-squares value iteration, when the action-value function is represented by an RKHS. We prove the first order-optimal regret guarantees under a general setting. Our results show a significant polynomial in the number of episodes improvement over the state of the art. In particular, with highly non-smooth kernels (such as Neural Tangent kernel or some Mat√©rn kernels) the existing results lead to trivial (superlinear in the number of episodes) regret bounds. We show a sublinear regret bound that is order optimal in the cases where a lower bound on regret is known (which includes the kernels mentioned above).

----

## [186] Learning Domain-Aware Detection Head with Prompt Tuning

**Authors**: *Haochen Li, Rui Zhang, Hantao Yao, Xinkai Song, Yifan Hao, Yongwei Zhao, Ling Li, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d18ab3b5fabfa6fe47c62e711af02f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d18ab3b5fabfa6fe47c62e711af02f0-Abstract-Conference.html)

**Abstract**:

Domain adaptive object detection (DAOD) aims to generalize detectors trained on an annotated source domain to an unlabelled target domain.  However, existing methods focus on reducing the domain bias of the detection backbone by inferring a discriminative visual encoder, while ignoring the domain bias in the detection head.  Inspired by the high generalization of vision-language models (VLMs), applying a VLM as the robust detection backbone following a domain-aware detection head is a reasonable way to learn the discriminative detector for each domain, rather than reducing the domain bias in traditional methods.  To achieve the above issue, we thus propose a novel DAOD framework named Domain-Aware detection head with Prompt tuning (DA-Pro), which applies the learnable domain-adaptive prompt to generate the dynamic detection head for each domain.   Formally, the domain-adaptive prompt consists of the domain-invariant tokens, domain-specific tokens, and the domain-related textual description along with the class label.   Furthermore, two constraints between the source and target domains are applied to ensure that the domain-adaptive prompt can capture the domains-shared and domain-specific knowledge.  A prompt ensemble strategy is also proposed to reduce the effect of prompt disturbance.   Comprehensive experiments over multiple cross-domain adaptation tasks demonstrate that using the domain-adaptive prompt can produce an effectively domain-related detection head for boosting domain-adaptive object detection.  Our code is available at https://github.com/Therock90421/DA-Pro.

----

## [187] Parallel Sampling of Diffusion Models

**Authors**: *Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh, Nima Anari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html)

**Abstract**:

Diffusion models are powerful generative models but suffer from slow sampling, often taking 1000 sequential denoising steps for one sample. As a result, considerable efforts have been directed toward reducing the number of denoising steps, but these methods hurt sample quality. Instead of reducing the number of denoising steps (trading quality for speed), in this paper we explore an orthogonal approach: can we run the denoising steps in parallel (trading compute for speed)? In spite of the sequential nature of the denoising steps, we show that surprisingly it is possible to parallelize sampling via Picard iterations, by guessing the solution of future denoising steps and iteratively refining until convergence. With this insight, we present ParaDiGMS, a novel method to accelerate the sampling of pretrained diffusion models by denoising multiple steps in parallel. ParaDiGMS is the first diffusion sampling method that enables trading compute for speed and is even compatible with existing fast sampling techniques such as DDIM and DPMSolver. Using ParaDiGMS, we improve sampling speed by 2-4x across a range of robotics and image generation models, giving state-of-the-art sampling speeds of 0.2s on 100-step DiffusionPolicy and 14.6s on 1000-step StableDiffusion-v2 with no measurable degradation of task reward, FID score, or CLIP score.

----

## [188] Fractal Landscapes in Policy Optimization

**Authors**: *Tao Wang, Sylvia L. Herbert, Sicun Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d21f257b5288385cb6cb8e0ff2ce82e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d21f257b5288385cb6cb8e0ff2ce82e-Abstract-Conference.html)

**Abstract**:

Policy gradient lies at the core of deep reinforcement learning (RL) in continuous domains. Despite much success, it is often observed in practice that RL training with policy gradient can fail for many reasons, even on standard control problems with known solutions. We propose a framework for understanding one inherent limitation of the policy gradient approach: the optimization landscape in the policy space can be extremely non-smooth or fractal for certain classes of MDPs, such that there does not exist gradient to be estimated in the first place. We draw on techniques from chaos theory and non-smooth analysis, and analyze the maximal Lyapunov exponents and H\"older exponents of the policy optimization objectives. Moreover, we develop a practical method that can estimate the local smoothness of objective function from samples to identify when the training process has encountered fractal landscapes. We show experiments to illustrate how some failure cases of policy optimization can be explained by such fractal landscapes.

----

## [189] Moral Responsibility for AI Systems

**Authors**: *Sander Beckers*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d5b7fd8c669fac58d6702188ed63afa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d5b7fd8c669fac58d6702188ed63afa-Abstract-Conference.html)

**Abstract**:

As more and more decisions that have a significant ethical dimension are being outsourced to AI systems, it is important to have a definition of moral responsibility that can be applied to AI systems. Moral responsibility for an outcome of an agent who performs some action is commonly taken to involve both a causal condition and an epistemic condition: the action should cause the outcome, and the agent should have been aware - in some form or other - of the possible moral consequences of their action. This paper presents a formal definition of both conditions within the framework of causal models. I compare my approach to the existing approaches of Braham and van Hees (BvH) and of Halpern and Kleiman-Weiner (HK). I then generalize my definition into a degree of responsibility.

----

## [190] Characterizing the Impacts of Semi-supervised Learning for Weak Supervision

**Authors**: *Jeffrey Li, Jieyu Zhang, Ludwig Schmidt, Alexander J. Ratner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d6270381e018b3d83eb9be7d0b06036-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d6270381e018b3d83eb9be7d0b06036-Abstract-Conference.html)

**Abstract**:

Labeling training data is a critical and expensive step in producing high accuracy ML models, whether training from scratch or fine-tuning. To make labeling more efficient, two major approaches are programmatic weak supervision (WS) and semi-supervised learning (SSL). More recent works have either explicitly or implicitly used techniques at their intersection, but in various complex and ad hoc ways. In this work, we define a simple, modular design space to study the use of SSL techniques for WS more systematically. Surprisingly, we find that fairly simple methods from our design space match the performance of more complex state-of-the-art methods, averaging a 3 p.p. increase in accuracy/F1-score across 8 standard WS benchmarks. Further, we provide practical guidance on when different components are worth their added complexity and training costs. Contrary to current understanding, we find using SSL is not necessary to obtain the best performance on most WS benchmarks but is more effective when: (1) end models are smaller, and (2) WS provides labels for only a small portion of training examples.

----

## [191] Logarithmic Bayes Regret Bounds

**Authors**: *Alexia Atsidakou, Branislav Kveton, Sumeet Katariya, Constantine Caramanis, Sujay Sanghavi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d9057d84a9fc37523bf826232ea6820-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d9057d84a9fc37523bf826232ea6820-Abstract-Conference.html)

**Abstract**:

We derive the first finite-time logarithmic Bayes regret upper bounds for Bayesian bandits. In a multi-armed bandit, we obtain $O(c_\Delta \log n)$ and $O(c_h \log^2 n)$ upper bounds for an upper confidence bound algorithm, where $c_h$ and $c_\Delta$ are constants depending on the prior distribution and the gaps of bandit instances sampled from it, respectively. The latter bound asymptotically matches the lower bound of Lai (1987). Our proofs are a major technical departure from prior works, while being simple and general. To show the generality of our techniques, we apply them to linear bandits. Our results provide insights on the value of prior in the Bayesian setting, both in the objective and as a side information given to the learner. They significantly improve upon existing $\tilde{O}(\sqrt{n})$ bounds, which have become standard in the literature despite the logarithmic lower bound of Lai (1987).

----

## [192] Frequency-Enhanced Data Augmentation for Vision-and-Language Navigation

**Authors**: *Keji He, Chenyang Si, Zhihe Lu, Yan Huang, Liang Wang, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0d9e08f247ca7fbbfd5e50b7ff9cf357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0d9e08f247ca7fbbfd5e50b7ff9cf357-Abstract-Conference.html)

**Abstract**:

Vision-and-Language Navigation (VLN) is a challenging task that requires an agent to navigate through complex environments based on natural language instructions. In contrast to conventional approaches, which primarily focus on the spatial domain exploration, we propose a paradigm shift toward the Fourier domain. This alternative perspective aims to enhance visual-textual matching, ultimately improving the agent's ability to understand and execute navigation tasks based on the given instructions. In this study, we first explore the significance of high-frequency information in VLN and provide evidence that it is instrumental in bolstering visual-textual matching processes. Building upon this insight, we further propose a sophisticated and versatile Frequency-enhanced Data Augmentation (FDA) technique to improve the VLN model's capability of capturing critical high-frequency information. Specifically, this approach requires the agent to navigate in environments where only a subset of high-frequency visual information corresponds with the provided textual instructions, ultimately fostering the agent's ability to selectively discern and capture pertinent high-frequency features according to the given instructions. Promising results on R2R, RxR, CVDN and REVERIE demonstrate that our FDA can be readily integrated with existing VLN approaches, improving performance without adding extra parameters, and keeping models simple and efficient. The code is available at https://github.com/hekj/FDA.

----

## [193] Building Socio-culturally Inclusive Stereotype Resources with Community Engagement

**Authors**: *Sunipa Dev, Jaya Goyal, Dinesh Tewari, Shachi Dave, Vinodkumar Prabhakaran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

With rapid development and deployment of generative language models in global settings, there is an urgent need to also scale our measurements of harm, not just in the number and types of harms covered, but also how well they account for local cultural contexts, including marginalized identities and the social biases experienced by them.Current evaluation paradigms are limited in their abilities to address this, as they are not representative of diverse, locally situated but global, socio-cultural perspectives. It is imperative that our evaluation resources are enhanced and calibrated by including people and experiences from different cultures and societies worldwide, in order to prevent gross underestimations or skews in measurements of harm. In this work, we demonstrate a socio-culturally aware expansion of evaluation resources in the Indian societal context, specifically for the harm of stereotyping. We devise a community engaged effort to build a resource which contains stereotypes for axes of disparity that are uniquely present in India. The resultant resource increases the number of stereotypes known for and in the Indian context by over 1000 stereotypes across many unique identities. We also demonstrate the utility and effectiveness of such expanded resources for evaluations of language models.CONTENT WARNING: This paper contains examples of stereotypes that may be offensive.

----

## [194] Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment

**Authors**: *Hao Liu, Wilson Yan, Pieter Abbeel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0df1738319f8c6e15b58cb16ea3cfa57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0df1738319f8c6e15b58cb16ea3cfa57-Abstract-Conference.html)

**Abstract**:

Recent progress in scaling up large language models has shown impressive capabilities in performing few-shot learning across a wide range of natural language tasks. However, a key limitation is that these language models fundamentally lack grounding to visual perception - a crucial attribute needed to extend to real world tasks such as in visual-question answering and robotics. While prior works have largely connected image to text through pretraining or fine-tuning, learning such alignments are generally costly due to a combination of curating massive datasets and large computational burdens. In order to resolve these limitations, we propose a simple yet effective approach called Language-Quantized AutoEncoder (LQAE), a modification of VQ-VAE that learns to align text-image data in an unsupervised manner by leveraging pretrained language model denoisers (e.g., BERT). Our main idea is to encode images as sequences of text tokens by directly quantizing image embeddings using a pretrained language codebook. We then feed a masked version of the quantized embeddings into a BERT to reconstruct the original input. By doing so, LQAE learns to represent similar images with similar clusters of text tokens, thereby aligning these two modalities without the use of aligned text-image pairs. We show LQAE learns text-aligned image tokens that enable few-shot multi-modal learning with large language models, outperforming baseline methods in tasks such as image classification and VQA while requiring as few as 1-10 image-text pairs.

----

## [195] QuIP: 2-Bit Quantization of Large Language Models With Guarantees

**Authors**: *Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0df38cd13520747e1e64e5b123a78ef8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0df38cd13520747e1e64e5b123a78ef8-Abstract-Conference.html)

**Abstract**:

This work studies post-training parameter quantization in large language models (LLMs). We introduce quantization with incoherence processing (QuIP), a new method based on the insight that quantization benefits from incoherent weight and Hessian matrices, i.e., from the weights being even in magnitude and the directions in which it is important to round them accurately being unaligned with the coordinate axes. QuIP consists of two steps: (1) an adaptive rounding procedure minimizing a quadratic proxy objective; (2) efficient pre- and post-processing that ensures weight and Hessian incoherence via multiplication by random orthogonal matrices. We complement QuIP with the first theoretical analysis for an LLM-scale quantization algorithm, and show that our theory also applies to an existing method, OPTQ. Empirically, we find that our incoherence preprocessing improves several existing quantization algorithms and yields the first LLM quantization methods that produce viable results using only two bits per weight. Our code can be found at https://github.com/Cornell-RelaxML/QuIP.

----

## [196] Exploiting Correlated Auxiliary Feedback in Parameterized Bandits

**Authors**: *Arun Verma, Zhongxiang Dai, Yao Shu, Bryan Kian Hsiang Low*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e0157ce5ea15831072be4744cbd5334-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e0157ce5ea15831072be4744cbd5334-Abstract-Conference.html)

**Abstract**:

We study a novel variant of the parameterized bandits problem in which the learner can observe additional auxiliary feedback that is correlated with the observed reward. The auxiliary feedback is readily available in many real-life applications, e.g., an online platform that wants to recommend the best-rated services to its users can observe the user's rating of service (rewards) and collect additional information like service delivery time (auxiliary feedback). In this paper, we first develop a method that exploits auxiliary feedback to build a reward estimator with tight confidence bounds, leading to a smaller regret. We then characterize the regret reduction in terms of the correlation coefficient between reward and its auxiliary feedback. Experimental results in different settings also verify the performance gain achieved by our proposed method.

----

## [197] Multi-modal Queried Object Detection in the Wild

**Authors**: *Yifan Xu, Mengdan Zhang, Chaoyou Fu, Peixian Chen, Xiaoshan Yang, Ke Li, Changsheng Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e3af444e7d82d29871804de476d1fbe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e3af444e7d82d29871804de476d1fbe-Abstract-Conference.html)

**Abstract**:

We introduce MQ-Det, an efficient architecture and pre-training strategy design to utilize both textual description with open-set generalization and visual exemplars with rich description granularity as category queries, namely, Multi-modal Queried object Detection, for real-world detection with both open-vocabulary categories and various granularity. MQ-Det incorporates vision queries into existing well-established language-queried-only detectors. A plug-and-play gated class-scalable perceiver module upon the frozen detector is proposed to augment category text with class-wise visual information. To address the learning inertia problem brought by the frozen detector, a vision conditioned masked language prediction strategy is proposed. MQ-Det's simple yet effective architecture and training strategy design is compatible with most language-queried object detectors, thus yielding versatile applications. Experimental results demonstrate that multi-modal queries largely boost open-world detection. For instance, MQ-Det significantly improves the state-of-the-art open-set detector GLIP by +7.8% AP on the LVIS benchmark via multi-modal queries without any downstream finetuning, and averagely +6.3% AP on 13 few-shot downstream tasks, with merely additional 3% modulating time required by GLIP. Code is available at https://github.com/YifanXu74/MQ-Det.

----

## [198] H-Consistency Bounds: Characterization and Extensions

**Authors**: *Anqi Mao, Mehryar Mohri, Yutao Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e441913d4fa486c3eec967d79750b13-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e441913d4fa486c3eec967d79750b13-Abstract-Conference.html)

**Abstract**:

A series of recent publications by Awasthi et al. have introduced the key notion of *$H$-consistency bounds* for surrogate loss functions.  These are upper bounds on the zero-one estimation error of any predictor in a hypothesis set, expressed in terms of its surrogate loss estimation error. They are both non-asymptotic and hypothesis set-specific and thus stronger and more informative than Bayes-consistency. However, determining if they hold and deriving these bounds have required a specific proof and analysis for each surrogate loss. Can we derive more general tools and characterizations? This paper provides both a general characterization and an extension of $H$-consistency bounds for multi-class classification. We present new and tight $H$-consistency bounds for both the family of constrained losses and that of comp-sum losses, which covers the familiar cross-entropy, or logistic loss applied to the outputs of a neural network. We further extend our analysis beyond the completeness assumptions adopted in previous studies and cover more realistic bounded hypothesis sets.  Our characterizations are based on error transformations, which are explicitly defined for each formulation. We illustrate the application of our general results through several special examples. A by-product of our analysis is the observation that a recently derived multi-class $H$-consistency bound for cross-entropy reduces to an excess bound and is not significant. Instead, we prove a much stronger and more significant guarantee.

----

## [199] Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms

**Authors**: *Peiyao Xiao, Hao Ban, Kaiyi Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e5b96f97c1813bb75f6c28532c2ecc7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e5b96f97c1813bb75f6c28532c2ecc7-Abstract-Conference.html)

**Abstract**:

Multi-objective optimization (MOO) has become an influential framework in many machine learning problems with multiple objectives such as learning with multiple criteria and multi-task learning (MTL). In this paper, we propose a new direction-oriented multi-objective formulation by regularizing the common descent direction within a neighborhood of a direction that optimizes a linear combination of objectives such as the average loss in MTL or a weighted loss that places higher emphasis on some tasks than the others. This formulation includes GD and MGDA as special cases, enjoys the direction-oriented benefit as in CAGrad, and facilitates the design of stochastic algorithms. To solve this problem, we propose Stochastic Direction-oriented Multi-objective Gradient descent (SDMGrad) with simple SGD type of updates, and its variant SDMGrad-OS with an efficient objective sampling. We develop a comprehensive convergence analysis for the proposed methods with different loop sizes and regularization coefficients. We show that both SDMGrad and SDMGrad-OS achieve improved sample complexities to find an $\epsilon$-accurate Pareto stationary point while achieving a small $\epsilon$-level distance toward a conflict-avoidant (CA) direction. For a constant-level CA distance, their sample complexities match the best known $\mathcal{O}(\epsilon^{-2})$ without bounded function value assumption. Extensive experiments show that our methods achieve competitive or improved performance compared to existing gradient manipulation approaches in a series of tasks on multi-task supervised learning and reinforcement learning. Code is available at https://github.com/ml-opt-lab/sdmgrad.

----



[Go to the next page](NIPS-2023-list2.md)

[Go to the catalog section](README.md)