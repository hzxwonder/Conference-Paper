## [0] DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection

**Authors**: *Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, Baoyuan Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e735e4b4f07de483cbe250130992726-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e735e4b4f07de483cbe250130992726-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

A critical yet frequently overlooked challenge in the field of deepfake detection is the lack of a standardized, unified, comprehensive benchmark. This issue leads to unfair performance comparisons and potentially misleading results. Specifically, there is a lack of uniformity in data processing pipelines, resulting in inconsistent data inputs for detection models. Additionally, there are noticeable differences in experimental settings, and evaluation strategies and metrics lack standardization. To fill this gap, we present the first comprehensive benchmark for deepfake detection, called \textit{DeepfakeBench}, which offers three key contributions: 1) a unified data management system to ensure consistent input across all detectors, 2) an integrated framework for state-of-the-art methods implementation, and 3) standardized evaluation metrics and protocols to promote transparency and reproducibility.  Featuring an extensible, modular-based codebase, \textit{DeepfakeBench} contains 15 state-of-the-art detection methods, 9 deepfake datasets, a series of deepfake detection evaluation protocols and analysis tools, as well as comprehensive evaluations.  Moreover, we provide new insights based on extensive analysis of these evaluations from various perspectives (\eg, data augmentations, backbones). We hope that our efforts could facilitate future research and foster innovation in this increasingly critical domain. All codes, evaluations, and analyses of our benchmark are publicly available at \url{https://github.com/SCLBD/DeepfakeBench}.

----

## [0] DreamWaltz: Make a Scene with Complex 3D Animatable Avatars

**Authors**: *Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, Lei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e769ec2c2cd99b6ad69c9d75113e386-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e769ec2c2cd99b6ad69c9d75113e386-Abstract-Conference.html)

**Abstract**:

We present DreamWaltz, a novel framework for generating and animating complex 3D avatars given text guidance and parametric human body prior. While recent methods have shown encouraging results for text-to-3D generation of common objects, creating high-quality and animatable 3D avatars remains challenging. To create high-quality 3D avatars, DreamWaltz proposes 3D-consistent occlusion-aware Score Distillation Sampling (SDS) to optimize implicit neural representations with canonical poses. It provides view-aligned supervision via 3D-aware skeleton conditioning which enables complex avatar generation without artifacts and multiple faces. For animation, our method learns an animatable 3D avatar representation from abundant image priors of diffusion model conditioned on various poses, which could animate complex non-rigged avatars given arbitrary poses without retraining. Extensive evaluations demonstrate that DreamWaltz is an effective and robust approach for creating 3D avatars that can take on complex shapes and appearances as well as novel poses for animation. The proposed framework further enables the creation of complex scenes with diverse compositions, including avatar-avatar, avatar-object and avatar-scene interactions. See https://dreamwaltz3d.github.io/ for more vivid 3D avatar and animation results.

----

## [0] Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects

**Authors**: *Chuanruo Ning, Ruihai Wu, Haoran Lu, Kaichun Mo, Hao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e7e2af2e5ba822c9ad35a37b31b5dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e7e2af2e5ba822c9ad35a37b31b5dd4-Abstract-Conference.html)

**Abstract**:

Articulated object manipulation is a fundamental yet challenging task in robotics. Due to significant geometric and semantic variations across object categories, previous manipulation models struggle to generalize to novel categories. Few-shot learning is a promising solution for alleviating this issue by allowing robots to perform a few interactions with unseen objects. However, extant approaches often necessitate costly and inefficient test-time interactions with each unseen instance. Recognizing this limitation, we observe that despite their distinct shapes, different categories often share similar local geometries essential for manipulation, such as pullable handles and graspable edges - a factor typically underutilized in previous few-shot learning works. To harness this commonality, we introduce 'Where2Explore', an affordance learning framework that effectively explores novel categories with minimal interactions on a limited number of instances. Our framework explicitly estimates the geometric similarity across different categories, identifying local areas that differ from shapes in the training categories for efficient exploration while concurrently transferring affordance knowledge to similar parts of the objects. Extensive experiments in simulated and real-world environments demonstrate our framework's capacity for efficient few-shot exploration and generalization.

----

## [0] OpenProteinSet: Training data for structural biology at scale

**Authors**: *Gustaf Ahdritz, Nazim Bouatta, Sachin Kadyan, Lukas Jarosch, Daniel Berenberg, Ian Fisk, Andrew M. Watkins, Stephen Ra, Richard Bonneau, Mohammed AlQuraishi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0eb82171240776fe19da498bef3b1abe-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0eb82171240776fe19da498bef3b1abe-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Multiple sequence alignments (MSAs) of proteins encode rich biological information and have been workhorses in bioinformatic methods for tasks like protein design and protein structure prediction for decades. Recent breakthroughs like AlphaFold2 that use transformers to attend directly over large quantities of raw MSAs have reaffirmed their importance. Generation of MSAs is highly computationally intensive, however, and no datasets comparable to those used to train AlphaFold2 have been made available to the research community, hindering progress in machine learning for proteins. To remedy this problem, we introduce OpenProteinSet, an open-source corpus of more than 16 million MSAs, associated structural homologs from the Protein Data Bank, and AlphaFold2 protein structure predictions. We have previously demonstrated the utility of OpenProteinSet by successfully retraining AlphaFold2 on it. We expect OpenProteinSet to be broadly useful as training and validation data for 1) diverse tasks focused on protein structure, function, and design and 2) large-scale multimodal machine learning research.

----

## [0] Counting Distinct Elements in the Turnstile Model with Differential Privacy under Continual Observation

**Authors**: *Palak Jain, Iden Kalemaj, Sofya Raskhodnikova, Satchit Sivakumar, Adam D. Smith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ef1afa0daa888d695dcd5e9513bafa3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ef1afa0daa888d695dcd5e9513bafa3-Abstract-Conference.html)

**Abstract**:

Privacy is a central challenge for systems that learn from sensitive data sets, especially when a system's outputs must be continuously updated to reflect changing data. We consider the achievable error for differentially private continual release of a basic  statistic---the number of distinct items---in a stream where items may be both inserted and deleted (the turnstile model). With only insertions, existing algorithms have additive error just polylogarithmic in the length of the stream $T$. We uncover a much richer landscape in the turnstile model, even without considering memory restrictions. We show that every differentially private mechanism that handles insertions and deletions has worst-case additive error  at least $T^{1/4}$ even under  a relatively weak, event-level privacy definition. Then, we identify a parameter of the input stream, its maximum flippancy, that is low for natural data streams and for which we give tight parameterized error guarantees. Specifically, the maximum flippancy is the largest number of times that the contribution of a single item to the distinct elements count changes over the course of the stream. We present an item-level differentially private mechanism that, for all turnstile streams with  maximum flippancy  $w$, continually outputs the number of distinct elements with an $O(\sqrt{w} \cdot \mathsf{poly}\log T)$ additive error,  without requiring prior knowledge of $w$. We prove that this is the best achievable error bound  that depends only on $w$, for a large range of values of $w$. When $w$ is small, the error of our mechanism is similar to the polylogarithmic in $T$ error in the insertion-only setting, bypassing the hardness in the turnstile model.

----

## [0] Demystifying Softmax Gating Function in Gaussian Mixture of Experts

**Authors**: *Huy Nguyen, TrungTin Nguyen, Nhat Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ef6ffcb85a2d238fc4761860c31ded4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ef6ffcb85a2d238fc4761860c31ded4-Abstract-Conference.html)

**Abstract**:

Understanding the parameter estimation of softmax gating Gaussian mixture of experts has remained a long-standing open problem in the literature. It is mainly due to three fundamental theoretical challenges associated with the softmax gating function: (i) the identifiability only up to the translation of parameters; (ii) the intrinsic interaction via partial differential equations between the softmax gating and the expert functions in the Gaussian density; (iii) the complex dependence between the numerator and denominator of the conditional density of softmax gating Gaussian mixture of experts. We resolve these challenges by proposing novel Voronoi loss functions among parameters and establishing the convergence rates of maximum likelihood estimator (MLE) for solving parameter estimation in these models. When the true number of experts is unknown and over-specified, our findings show a connection between the convergence rate of the MLE and a solvability problem of a system of polynomial equations.

----

## [0] Hybrid Policy Optimization from Imperfect Demonstrations

**Authors**: *Hanlin Yang, Chao Yu, Peng Sun, Siji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f0a30c7b46be23a83317c5cb721fc43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f0a30c7b46be23a83317c5cb721fc43-Abstract-Conference.html)

**Abstract**:

Exploration is one of the main challenges in Reinforcement Learning (RL), especially in environments with sparse rewards. Learning from Demonstrations (LfD) is a promising approach to solving this problem by leveraging expert demonstrations. However, expert demonstrations of high quality  are usually costly or even impossible to collect in real-world applications. In this work, we propose a novel RL algorithm called HYbrid Policy Optimization (HYPO), which uses a small number of imperfect demonstrations to accelerate an agent's online learning process. The key idea is to train an offline guider policy using imitation learning in order to instruct an online agent policy to explore efficiently. Through mutual update of the guider policy and the agent policy, the agent can leverage suboptimal demonstrations for efficient exploration while avoiding the conservative policy caused by imperfect demonstrations. Empirical results show that HYPO significantly outperforms several baselines in various challenging tasks, such as MuJoCo with sparse rewards, Google Research Football, and the AirSim drone simulation.

----

## [0] What is Flagged in Uncertainty Quantification? Latent Density Models for Uncertainty Categorization

**Authors**: *Hao Sun, Boris van Breugel, Jonathan Crabbé, Nabeel Seedat, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f0c4f3d83c58df58380af3b0729354c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f0c4f3d83c58df58380af3b0729354c-Abstract-Conference.html)

**Abstract**:

Uncertainty quantification (UQ) is essential for creating trustworthy machine learning models. Recent years have seen a steep rise in UQ methods that can flag suspicious examples, however, it is often unclear what exactly these methods identify. In this work, we propose a framework for categorizing uncertain examples flagged by UQ methods. We introduce the confusion density matrix---a kernel-based approximation of the misclassification density---and use this to categorize suspicious examples identified by a given uncertainty method into three classes: out-of-distribution (OOD) examples, boundary (Bnd) examples, and examples in regions of high in-distribution misclassification (IDM). Through extensive experiments, we show that our framework provides a new and distinct perspective for assessing differences between uncertainty quantification methods, thereby forming a valuable assessment benchmark.

----

## [0] Datasets and Benchmarks for Nanophotonic Structure and Parametric Design Simulations

**Authors**: *Jungtaek Kim, Mingxuan Li, Oliver Hinder, Paul W. Leu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f12c9975ff4f2e44a5a26ef01b0b249-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f12c9975ff4f2e44a5a26ef01b0b249-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Nanophotonic structures have versatile applications including solar cells, anti-reflective coatings, electromagnetic interference shielding, optical filters, and light emitting diodes. To design and understand these nanophotonic structures, electrodynamic simulations are essential. These simulations enable us to model electromagnetic fields over time and calculate optical properties. In this work, we introduce frameworks and benchmarks to evaluate nanophotonic structures in the context of parametric structure design problems. The benchmarks are instrumental in assessing the performance of optimization algorithms and identifying an optimal structure based on target optical properties. Moreover, we explore the impact of varying grid sizes in electrodynamic simulations, shedding light on how evaluation fidelity can be strategically leveraged in enhancing structure designs.

----

## [0] Efficient Data Subset Selection to Generalize Training Across Models: Transductive and Inductive Networks

**Authors**: *Eeshaan Jain, Tushar Nandy, Gaurav Aggarwal, Ashish Tendulkar, Rishabh K. Iyer, Abir De*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f25eb6e9dc26c933a5d7516abf1eb8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f25eb6e9dc26c933a5d7516abf1eb8c-Abstract-Conference.html)

**Abstract**:

Existing subset selection methods for efficient learning predominantly employ discrete combinatorial and model-specific approaches, which lack generalizability--- for each new model, the algorithm has to be executed from the beginning. Therefore, for an unseen architecture, one cannot use the subset chosen for a different model. In this work, we propose $\texttt{SubSelNet}$, a non-adaptive subset selection framework, which tackles these problems. Here, we first introduce an attention-based neural gadget that leverages the graph structure of architectures and acts as a surrogate to trained deep neural networks for quick model prediction. Then, we use these predictions to build subset samplers. This naturally provides us two variants of $\texttt{SubSelNet}$. The first variant is transductive (called Transductive-$\texttt{SubSelNet}$), which computes the subset separately for each model by solving a small optimization problem. Such an optimization is still super fast, thanks to the replacement of explicit model training by the model approximator. The second variant is inductive (called Inductive-$\texttt{SubSelNet}$), which computes the subset using a trained subset selector, without any optimization.  Our experiments show that our model outperforms several methods across several real datasets.

----

## [0] NIS3D: A Completely Annotated Benchmark for Dense 3D Nuclei Image Segmentation

**Authors**: *Wei Zheng, Cheng Peng, Zeyuan Hou, Boyu Lyu, Mengfan Wang, Xuelong Mi, Shuoxuan Qiao, Yinan Wan, Guoqiang Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f2cd3d09a132757555b602e2dd43784-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f2cd3d09a132757555b602e2dd43784-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

3D segmentation of nuclei images is a fundamental task for many biological studies. Despite the rapid advances of large-volume 3D imaging acquisition methods and the emergence of sophisticated algorithms to segment the nuclei in recent years, a benchmark with all cells completely annotated is still missing, making it hard to accurately assess and further improve the performance of the algorithms. The existing nuclei segmentation benchmarks either worked on 2D only or annotated a small number of 3D cells, perhaps due to the high cost of 3D annotation for large-scale data. To fulfill the critical need, we constructed NIS3D, a 3D, high cell density, large-volume, and completely annotated Nuclei Image Segmentation benchmark, assisted by our newly designed semi-automatic annotation software. NIS3D provides more than 22,000 cells across multiple most-used species in this area. Each cell is labeled by three independent annotators, so we can measure the variability of each annotation. A confidence score is computed for each cell, allowing more nuanced testing and performance comparison. A comprehensive review on the methods of segmenting 3D dense nuclei was conducted. The benchmark was used to evaluate the performance of several selected state-of-the-art segmentation algorithms. The best of current methods is still far away from human-level accuracy, corroborating the necessity of generating such a benchmark. The testing results also demonstrated the strength and weakness of each method and pointed out the directions of further methodological development. The dataset can be downloaded here: https://github.com/yu-lab-vt/NIS3D.

----

## [0] HiBug: On Human-Interpretable Model Debug

**Authors**: *Muxi Chen, Yu Li, Qiang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f53ecc0d36a5d5d3d3e94d42c4b23ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f53ecc0d36a5d5d3d3e94d42c4b23ca-Abstract-Conference.html)

**Abstract**:

Machine learning models can frequently produce systematic errors on critical subsets (or slices) of data that share common attributes. Discovering and explaining such model bugs is crucial for reliable model deployment. However, existing bug discovery and interpretation methods usually involve heavy human intervention and annotation, which can be cumbersome and have low bug coverage.In this paper, we propose HiBug, an automated framework for interpretable model debugging. Our approach utilizes large pre-trained models, such as chatGPT, to suggest human-understandable attributes that are related to the targeted computer vision tasks. By leveraging pre-trained vision-language models, we can efficiently identify common visual attributes of underperforming data slices using human-understandable terms. This enables us to uncover rare cases in the training data, identify spurious correlations in the model, and use the interpretable debug results to select or generate new training data for model improvement. Experimental results demonstrate the efficacy of the HiBug framework.

----

## [0] A Theoretical Analysis of the Test Error of Finite-Rank Kernel Ridge Regression

**Authors**: *Tin Sum Cheng, Aurélien Lucchi, Anastasis Kratsios, Ivan Dokmanic, David Belius*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f580c1ace3b857a390575ca42de7938-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f580c1ace3b857a390575ca42de7938-Abstract-Conference.html)

**Abstract**:

Existing statistical learning guarantees for general kernel regressors often yield loose bounds when used with finite-rank kernels. Yet, finite-rank kernels naturally appear in a number of machine learning problems, e.g. when fine-tuning a pre-trained deep neural network's last layer to adapt it to a novel task when performing transfer learning.  We address this gap for finite-rank kernel ridge regression (KRR) by deriving sharp non-asymptotic upper and lower bounds for the KRR test error of any finite-rank KRR. Our bounds are tighter than previously derived bounds on finite-rank KRR and, unlike comparable results, they also remain valid for any regularization parameters.

----

## [0] Learning Invariant Representations with a Nonparametric Nadaraya-Watson Head

**Authors**: *Alan Wang, Minh Nguyen, Mert R. Sabuncu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f6931a9e339a012a9909306d7c758b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f6931a9e339a012a9909306d7c758b4-Abstract-Conference.html)

**Abstract**:

Machine learning models will often fail when deployed in an environment with a data distribution that is different than the training distribution. When multiple environments are available during training, many methods exist that learn representations which are invariant across the different distributions, with the hope that these representations will be transportable to unseen domains. In this work, we present a nonparametric strategy for learning invariant representations based on the recently-proposed Nadaraya-Watson (NW) head. The NW head makes a prediction by comparing the learned representations of the query to the elements of a support set that consists of labeled data. We demonstrate that by manipulating the support set, one can encode different causal assumptions. In particular, restricting the support set to a single environment encourages the model to learn invariant features that do not depend on the environment. We present a causally-motivated setup for our modeling and training strategy and validate on three challenging real-world domain generalization tasks in computer vision.

----

## [0] Conformalized matrix completion

**Authors**: *Yu Gui, Rina Barber, Cong Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f7e4bb7a35dd4cb426203c91a4bfa10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f7e4bb7a35dd4cb426203c91a4bfa10-Abstract-Conference.html)

**Abstract**:

Matrix completion aims to estimate missing entries in a data matrix, using the assumption of a low-complexity structure (e.g., low-rankness) so that imputation is possible. While many effective estimation algorithms exist in the literature, uncertainty quantification for this problem has proved to be challenging, and existing methods are extremely sensitive to model misspecification. In this work, we propose a distribution-free method for predictive inference in the matrix completion problem. Our method adapts the framework of conformal prediction, which provides prediction intervals with guaranteed distribution-free validity in the setting of regression, to the problem of matrix completion. Our resulting method, conformalized matrix completion (cmc), offers provable predictive coverage regardless of the accuracy of the low-rank model. Empirical results on simulated and real data demonstrate that cmc is robust to model misspecification while matching the performance of existing model-based methods when the model is correct.

----

## [0] Mixture Weight Estimation and Model Prediction in Multi-source Multi-target Domain Adaptation

**Authors**: *Yuyang Deng, Ilja Kuzborskij, Mehrdad Mahdavi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0fa81c3f0d57f95b8776de3a248ef0ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0fa81c3f0d57f95b8776de3a248ef0ed-Abstract-Conference.html)

**Abstract**:

We consider a problem of learning a model from multiple sources with the goal to performwell on a new target distribution.  Such problem arises inlearning with data collected from multiple sources (e.g. crowdsourcing) orlearning in distributed systems, where the data can be highly heterogeneous. Thegoal of learner is to mix these data sources in a target-distribution aware way andsimultaneously minimize the empirical risk on the mixed source.  The literature has made some tangible advancements in establishingtheory of learning on mixture domain.  However, there are still two unsolved problems. Firstly, how to estimate the optimal mixture of sources, given a target domain; Secondly, when there are numerous target domains, we have to solve empirical risk minimization for each target on possibly unique mixed source data , which is computationally expensive. In this paper we address both problems efficiently and with guarantees.We cast the first problem, mixture weight estimation as convex-nonconcave compositional minimax, and propose an efficient stochasticalgorithm with provable stationarity guarantees.Next, for the second problem, we identify that for certain regime,solving ERM for each target domain individually can be avoided, and instead parameters for a target optimalmodel can be viewed as a non-linear function ona space of the mixture coefficients.To this end, we show that in offline setting, a GD-trained overparameterized neural network can provably learn such function.Finally, we also consider an online setting and propose an label efficient online algorithm, which predicts parameters for new models given arbitrary sequence of mixing coefficients, while enjoying optimal regret.

----

## [0] CELLE-2: Translating Proteins to Pictures and Back with a Bidirectional Text-to-Image Transformer

**Authors**: *Emaad Khwaja, Yun Song, Aaron Agarunov, Bo Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0fb7c02d420c993385c7de44c2b5bf01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0fb7c02d420c993385c7de44c2b5bf01-Abstract-Conference.html)

**Abstract**:

We present CELL-E 2, a novel bidirectional transformer that can generate images depicting protein subcellular localization from the amino acid sequences (and vice versa). Protein localization is a challenging problem that requires integrating sequence and image information, which most existing methods ignore. CELL-E 2 extends the work of CELL-E, not only capturing the spatial complexity of protein localization and produce probability estimates of localization atop a nucleus image, but also being able to generate sequences from images, enabling de novo protein design. We train and finetune CELL-E 2 on two large-scale datasets of human proteins. We also demonstrate how to use CELL-E 2 to create hundreds of novel nuclear localization signals (NLS). Results and interactive demos are featured at https://bohuanglab.github.io/CELL-E_2/.

----

## [0] HeadSculpt: Crafting 3D Head Avatars with Text

**Authors**: *Xiao Han, Yukang Cao, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, Kwan-Yee K. Wong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0fb98d483fa580e0354bcdd3a003a3f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0fb98d483fa580e0354bcdd3a003a3f3-Abstract-Conference.html)

**Abstract**:

Recently, text-guided 3D generative methods have made remarkable advancements in producing high-quality textures and geometry, capitalizing on the proliferation of large vision-language and image diffusion models. However, existing methods still struggle to create high-fidelity 3D head avatars in two aspects: (1) They rely mostly on a pre-trained text-to-image diffusion model whilst missing the necessary 3D awareness and head priors. This makes them prone to inconsistency and geometric distortions in the generated avatars. (2) They fall short in fine-grained editing. This is primarily due to the inherited limitations from the pre-trained 2D image diffusion models, which become more pronounced when it comes to 3D head avatars. In this work, we address these challenges by introducing a versatile coarse-to-fine pipeline dubbed HeadSculpt for crafting (i.e., generating and editing) 3D head avatars from textual prompts. Specifically, we first equip the diffusion model with 3D awareness by leveraging landmark-based control and a learned textual embedding representing the back view appearance of heads, enabling 3D-consistent head avatar generations. We further propose a novel identity-aware editing score distillation strategy to optimize a textured mesh with a high-resolution differentiable rendering technique. This enables identity preservation while following the editing instruction.We showcase HeadSculpt's superior fidelity and editing capabilities through comprehensive experiments and comparisons with existing methods.

----

## [0] CBD: A Certified Backdoor Detector Based on Local Dominant Probability

**Authors**: *Zhen Xiang, Zidi Xiong, Bo Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0fbf046448d7eea18b982001320b9a10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0fbf046448d7eea18b982001320b9a10-Abstract-Conference.html)

**Abstract**:

Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.

----

## [0] SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models

**Authors**: *Hongxin Li, Jingran Su, Yuntao Chen, Qing Li, Zhaoxiang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ff30c4bf31db0119a6219e0d250e037-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ff30c4bf31db0119a6219e0d250e037-Abstract-Conference.html)

**Abstract**:

Computer end users have spent billions of hours completing daily tasks like tabular data processing and project timeline scheduling. Most of these tasks are repetitive and error-prone, yet most end users lack the skill to automate these burdensome works. With the advent of large language models (LLMs), directing software with natural language user requests become a reachable goal. In this work, we propose a SheetCopilot agent that takes natural language task and control spreadsheet to fulfill the requirements. We propose a set of atomic actions as an abstraction of spreadsheet software functionalities. We further design a state machine-based task planning framework for LLMs to robustly interact with spreadsheets. We curate a representative dataset containing 221 spreadsheet control tasks and establish a fully automated evaluation pipeline for rigorously benchmarking the ability of LLMs in software control tasks. Our SheetCopilot correctly completes 44.3\% of tasks for a single generation, outperforming the strong code generation baseline by a wide margin. Our project page: https://sheetcopilot.github.io/.

----

## [0] Beyond Uniform Sampling: Offline Reinforcement Learning with Imbalanced Datasets

**Authors**: *Zhang-Wei Hong, Aviral Kumar, Sathwik Karnik, Abhishek Bhandwaldar, Akash Srivastava, Joni Pajarinen, Romain Laroche, Abhishek Gupta, Pulkit Agrawal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ff3502bb29570b219967278db150a50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ff3502bb29570b219967278db150a50-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) enables learning a decision-making policy without interaction with the environment. This makes it particularly beneficial in situations where such interactions are costly. However, a known challenge for offline RL algorithms is the distributional mismatch between the state-action distributions of the learned policy and the dataset, which can significantly impact performance. State-of-the-art algorithms address it by constraining the policy to align with the state-action pairs in the dataset. However, this strategy struggles on datasets that predominantly consist of trajectories collected by low-performing policies and only a few trajectories from high-performing ones. Indeed, the constraint to align with the data leads the policy to imitate low-performing behaviors predominating the dataset. Our key insight to address this issue is to constrain the policy to the policy that collected the good parts of the dataset rather than all data. To this end, we optimize the importance sampling weights to emulate sampling data from a data distribution generated by a nearly optimal policy. Our method exhibits considerable performance gains (up to five times better) over the existing approaches in state-of-the-art offline RL algorithms over 72 imbalanced datasets with varying types of imbalance.

----

## [0] Variational Weighting for Kernel Density Ratios

**Authors**: *Sangwoong Yoon, Frank C. Park, Gunsu S. Yun, Iljung Kim, Yung-Kyun Noh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ff54b4ec4f70b3ae12c8621ca8a49f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ff54b4ec4f70b3ae12c8621ca8a49f4-Abstract-Conference.html)

**Abstract**:

Kernel density estimation (KDE) is integral to a range of generative and discriminative tasks in machine learning. Drawing upon tools from the multidimensional calculus of variations, we derive an optimal weight function that reduces bias in standard kernel density estimates for density ratios, leading to improved estimates of prediction posteriors and information-theoretic measures. In the process, we shed light on some fundamental aspects of density estimation, particularly from the perspective of algorithms that employ KDEs as their main building blocks.

----

## [0] Adversarial Examples Exist in Two-Layer ReLU Networks for Low Dimensional Linear Subspaces

**Authors**: *Odelia Melamed, Gilad Yehudai, Gal Vardi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ffd11b5bce666816802b86c77b54cf7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ffd11b5bce666816802b86c77b54cf7-Abstract-Conference.html)

**Abstract**:

Despite a great deal of research, it is still not well-understood why trained neural networks are highly vulnerable to adversarial examples.In this work we focus on two-layer neural networks trained using data which lie on a low dimensional linear subspace.We show that standard gradient methods lead to non-robust neural networks, namely, networks which have large gradients in directions orthogonal to the data subspace, and are susceptible to small adversarial $L_2$-perturbations in these directions.Moreover, we show that decreasing the initialization scale of the training algorithm, or adding $L_2$ regularization, can make the trained network more robust to adversarial perturbations orthogonal to the data.

----

## [0] Complexity of Derivative-Free Policy Optimization for Structured H∞ Control

**Authors**: *Xingang Guo, Darioush Keivan, Geir E. Dullerud, Peter J. Seiler, Bin Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1052b823a161aa2c808dd51c0f58dc37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1052b823a161aa2c808dd51c0f58dc37-Abstract-Conference.html)

**Abstract**:

The applications of direct policy search in reinforcement learning and continuous control have received increasing attention.In this work, we present novel theoretical results on the complexity of derivative-free policy optimization on an important class of robust control tasks, namely the structured $H_\infty$ synthesis with static output feedback. Optimal $H_\infty$ synthesis under structural constraints leads to a constrained nonconvex nonsmooth problem and is typicallyaddressed using subgradient-based policy search techniques that are built upon the concept of Goldstein subdifferential or other notions of enlarged subdifferential.  In this paper, we study the complexity of finding $(\delta,\epsilon)$-stationary points for such nonsmooth robust control design tasks using policy optimization methods which can only access the zeroth-order oracle (i.e. the $H_\infty$ norm of the closed-loop system). First, we study the exact oracle setting and identify the coerciveness of the cost function to prove high-probability feasibility/complexity bounds for derivative-free policy optimization on this problem. Next, we derive a sample complexity result for the multi-input multi-output (MIMO)  $H_\infty$-norm estimation. We combine this with our analysis to obtain the first sample complexity of model-free, trajectory-based, zeroth-order policy optimization on finding $(\delta,\epsilon)$-stationary points for structured $H_\infty$ control. Numerical results are also provided to demonstrate our theory.

----

## [0] Meet in the Middle: A New Pre-training Paradigm

**Authors**: *Anh Nguyen, Nikos Karampatziakis, Weizhu Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/105fdc31cc9eb927cc5a0110f4031287-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/105fdc31cc9eb927cc5a0110f4031287-Abstract-Conference.html)

**Abstract**:

Most language models (LMs) are trained and applied in an autoregressive left-to-right fashion, predicting the next token from the preceding ones. However, this ignores that the full sequence is available during training. In this paper, we introduce ``Meet in the Middle'' (MIM) a new pre-training paradigm that improves data efficiency by training in two directions, left-to-right and right-to-left, and encouraging the respective modelsto agree on their token distribution for each position. While the primary outcome is an improved left-to-right LM,we also obtain secondary benefits in the infilling task. There, we leverage the two pre-trained directions to propose an infilling procedure that builds the completion simultaneously from both sides. We conduct extensive experiments on both programming and natural languages and show that MIM significantly surpasses existing pre-training paradigms, in both left-to-right generation as well as infilling.Code and models available at https://github.com/microsoft/Meet-in-the-Middle

----

## [0] Score-based Source Separation with Applications to Digital Communication Signals

**Authors**: *Tejas Jayashankar, Gary C. F. Lee, Alejandro Lancho, Amir Weiss, Yury Polyanskiy, Gregory W. Wornell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/106b2434b8d496c6aed9235d478678af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/106b2434b8d496c6aed9235d478678af-Abstract-Conference.html)

**Abstract**:

We propose a new method for separating superimposed sources using diffusion-based generative models.  Our method relies only on separately trained statistical priors of independent sources to establish a new objective function guided by $\textit{maximum a posteriori}$ estimation with an $\textit{$\alpha$-posterior}$, across multiple levels of Gaussian smoothing.  Motivated by applications in radio-frequency (RF) systems, we are interested in sources with underlying discrete nature and the recovery of encoded bits from a signal of interest, as measured by the bit error rate (BER). Experimental results with RF mixtures demonstrate that our method results in a BER reduction of 95\%  over classical and existing learning-based methods.  Our analysis demonstrates that our proposed method yields solutions that asymptotically approach the modes of an underlying discrete distribution. Furthermore, our method can be viewed as a multi-source extension to the recently proposed score distillation sampling scheme, shedding additional light on its use beyond conditional sampling. The project webpage is available at https://alpha-rgs.github.io.

----

## [0] Fair Streaming Principal Component Analysis: Statistical and Algorithmic Viewpoint

**Authors**: *Junghyun Lee, Hanseul Cho, Se-Young Yun, Chulhee Yun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1074541383db5ef12d6ac66d2f8e8d34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1074541383db5ef12d6ac66d2f8e8d34-Abstract-Conference.html)

**Abstract**:

Fair Principal Component Analysis (PCA) is a problem setting where we aim to perform PCA while making the resulting representation fair in that the projected distributions, conditional on the sensitive attributes, match one another. However, existing approaches to fair PCA have two main problems: theoretically, there has been no statistical foundation of fair PCA in terms of learnability; practically, limited memory prevents us from using existing approaches, as they explicitly rely on full access to the entire data. On the theoretical side, we rigorously formulate fair PCA using a new notion called probably approximately fair and optimal (PAFO) learnability. On the practical side, motivated by recent advances in streaming algorithms for addressing memory limitation, we propose a new setting called fair streaming PCA along with a memory-efficient algorithm, fair noisy power method (FNPM). We then provide its statistical guarantee in terms of PAFO-learnability, which is the first of its kind in fair PCA literature. We verify our algorithm in the CelebA dataset without any pre-processing; while the existing approaches are inapplicable due to memory limitations, by turning it into a streaming setting, we show that our algorithm performs fair PCA efficiently and effectively.

----

## [0] DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models

**Authors**: *Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, Sibei Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/108030643e640ac050e0ed5e6aace48f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/108030643e640ac050e0ed5e6aace48f-Abstract-Conference.html)

**Abstract**:

A long-standing goal of AI systems is to perform complex multimodal reasoning like humans. Recently, large language models (LLMs) have made remarkable strides in such multi-step reasoning on the language modality solely by leveraging the chain of thought (CoT) to mimic human thinking. However, the transfer of these advancements to multimodal contexts introduces heightened challenges, including but not limited to the impractical need for labor-intensive annotation and the limitations in terms of flexibility, generalizability, and explainability. To evoke CoT reasoning in multimodality, this work first conducts an in-depth analysis of these challenges posed by multimodality and presents two key insights: “keeping critical thinking” and “letting everyone do their jobs” in multimodal CoT reasoning. Furthermore, this study proposes a novel DDCoT prompting that maintains a critical attitude through negative-space prompting and incorporates multimodality into reasoning by first dividing the reasoning responsibility of LLMs into reasoning and recognition and then integrating the visual recognition capability of visual models into the joint reasoning process. The rationales generated by DDCoT not only improve the reasoning abilities of both large and small language models in zero-shot prompting and fine-tuning learning, significantly outperforming state-of-the-art methods but also exhibit impressive generalizability and explainability.

----

## [0] Adversarially Robust Learning with Uncertain Perturbation Sets

**Authors**: *Tosca Lechner, Vinayak Pathak, Ruth Urner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1097a0aeaf00cacfa8f6aced24f3a8bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1097a0aeaf00cacfa8f6aced24f3a8bd-Abstract-Conference.html)

**Abstract**:

In many real-world settings exact perturbation sets to be used by an adversary are not plausibly available to a learner. While prior literature has studied both scenarios with completely known and completely unknown perturbation sets, we propose an in-between setting of learning with respect to a class of perturbation sets. We show that in this setting we can improve on previous results with completely unknown perturbation sets, while still addressing the concerns of not having perfect knowledge of these sets in real life. In particular, we give the first positive results for the learnability of infinite Littlestone classes when having access to a perfect-attack oracle. We also consider a setting of learning with abstention, where predictions are considered robustness violations, only when the wrong prediction is made within the perturbation set. We show there are classes for which perturbation-set unaware learning without query access is possible, but abstention is required.

----

## [0] Common Ground in Cooperative Communication

**Authors**: *Xiaoran Hao, Yash Jhaveri, Patrick Shafto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/10b7e27c8eb9571fbbd2ae6a9f8c3855-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/10b7e27c8eb9571fbbd2ae6a9f8c3855-Abstract-Conference.html)

**Abstract**:

Cooperative communication plays a fundamental role in theories of human-human interaction--cognition, culture, development, language, etc.--as well as human-robot interaction. The core challenge in cooperative communication is the problem of common ground: having enough shared knowledge and understanding to successfully communicate. Prior models of cooperative communication, however, uniformly assume the strongest form of common ground, perfect and complete knowledge sharing, and, therefore, fail to capture the core challenge of cooperative communication. We propose a general theory of cooperative communication that is mathematically principled and explicitly defines a spectrum of common ground possibilities, going well beyond that of perfect and complete knowledge sharing, on spaces that permit arbitrary representations of data and hypotheses. Our framework is a strict generalization of prior models of cooperative communication. After considering a parametric form of common ground and viewing the data selection and hypothesis inference processes of communication as encoding and decoding, we establish a connection to variational autoencoding, a powerful model in modern machine learning. Finally, we carry out a series of empirical simulations to support and elaborate on our theoretical results.

----

## [0] Keep Various Trajectories: Promoting Exploration of Ensemble Policies in Continuous Control

**Authors**: *Chao Li, Chen Gong, Qiang He, Xinwen Hou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/10cb15f4559b3d578b7f24966d48a137-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/10cb15f4559b3d578b7f24966d48a137-Abstract-Conference.html)

**Abstract**:

The combination of deep reinforcement learning (DRL) with ensemble methods has been proved to be highly effective in addressing complex sequential decision-making problems. This success can be primarily attributed to the utilization of multiple models, which enhances both the robustness of the policy and the accuracy of value function estimation. However, there has been limited analysis of the empirical success of current ensemble RL methods thus far. Our new analysis reveals that the sample efficiency of previous ensemble DRL algorithms may be limited by sub-policies that are not as diverse as they could be. Motivated by these findings, our study introduces a new ensemble RL algorithm, termed \textbf{T}rajectories-awar\textbf{E} \textbf{E}nsemble exploratio\textbf{N} (TEEN). The primary goal of TEEN is to  maximize the expected return while promoting more diverse trajectories. Through extensive experiments, we demonstrate that TEEN not only enhances the sample diversity of the ensemble policy compared to using sub-policies alone but also improves the performance over ensemble RL algorithms. On average, TEEN outperforms the baseline ensemble DRL algorithms by 41\% in performance on the tested representative environments.

----

## [0] ReSync: Riemannian Subgradient-based Robust Rotation Synchronization

**Authors**: *Huikang Liu, Xiao Li, Anthony Man-Cho So*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/10e9204f14c4daa08041343455435308-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/10e9204f14c4daa08041343455435308-Abstract-Conference.html)

**Abstract**:

This work presents ReSync, a Riemannian subgradient-based algorithm for solving the robust rotation synchronization problem, which arises in various engineering applications. ReSync solves a least-unsquared minimization formulation over the rotation group, which is nonsmooth and nonconvex, and aims at recovering the underlying rotations directly. We provide strong theoretical guarantees for ReSync under the random corruption setting. Specifically, we first show that the initialization procedure of ReSync yields a proper initial point that lies in a local region around the ground-truth rotations. We next establish the weak sharpness property of the aforementioned formulation and then utilize this property to derive the local linear convergence of ReSync to the ground-truth rotations. By combining these guarantees, we conclude that ReSync converges linearly to the ground-truth rotations under appropriate conditions. Experiment results demonstrate the effectiveness of ReSync.

----

## [0] On the Exploration of Local Significant Differences For Two-Sample Test

**Authors**: *Zhijian Zhou, Jie Ni, Jia-He Yao, Wei Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed increasing attentions on two-sample test with diverse real applications, while this work takes one more step on the exploration of local significant differences for two-sample test. We propose the ME$_\text{MaBiD}$, an effective test for two-sample testing, and the basic idea is to exploit local information by multiple Mahalanobis kernels and introduce bi-directional hypothesis for testing. On the exploration of local significant differences, we first partition the embedding space into several rectangle regions via a new splitting criterion, which is relevant to test power and data correlation. We then explore local significant differences based on our bi-directional masked $p$-value together with the ME$_\text{MaBiD}$ test. Theoretically, we present the asymptotic distribution and lower bounds of test power for our ME$_\text{MaBiD}$ test, and control the familywise error rate on the exploration of local significant differences. We finally conduct extensive experiments  to validate the effectiveness of our proposed methods on two-sample test and the exploration of local significant differences.

----

## [0] Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator

**Authors**: *Xiaolong Wang, Runsen Xu, Zhuofan Cui, Zeyu Wan, Yu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/112d8e0c7563de6e3408b49a09b4d8a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/112d8e0c7563de6e3408b49a09b4d8a3-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce a novel approach to fine-grained cross-view geo-localization. Our method aligns a warped ground image with a corresponding GPS-tagged satellite image covering the same area using homography estimation. We first employ a differentiable spherical transform, adhering to geometric principles, to accurately align the perspective of the ground image with the satellite map. This transformation effectively places ground and aerial images in the same view and on the same plane, reducing the task to an image alignment problem. To address challenges such as occlusion, small overlapping range, and seasonal variations, we propose a robust correlation-aware homography estimator to align similar parts of the transformed ground image with the satellite image. Our method achieves sub-pixel resolution and meter-level GPS accuracy by mapping the center point of the transformed ground image to the satellite image using a homography matrix and determining the orientation of the ground camera using a point above the central axis. Operating at a speed of 30 FPS, our method outperforms state-of-the-art techniques, reducing the mean metric localization error by 21.3\% and 32.4\% in same-area and cross-area generalization tasks on the VIGOR benchmark, respectively, and by 34.4\% on the KITTI benchmark in same-area evaluation.

----

## [0] DataPerf: Benchmarks for Data-Centric AI Development

**Authors**: *Mark Mazumder, Colby R. Banbury, Xiaozhe Yao, Bojan Karlas, William Gaviria Rojas, Sudnya Frederick Diamos, Greg Diamos, Lynn He, Alicia Parrish, Hannah Rose Kirk, Jessica Quaye, Charvi Rastogi, Douwe Kiela, David Jurado, David Kanter, Rafael Mosquera, Will Cukierski, Juan Ciro, Lora Aroyo, Bilge Acun, Lingjiao Chen, Mehul Raje, Max Bartolo, Evan Sabri Eyuboglu, Amirata Ghorbani, Emmett D. Goodman, Addison Howard, Oana Inel, Tariq Kane, Christine R. Kirkpatrick, D. Sculley, Tzu-Sheng Kuo, Jonas W. Mueller, Tristan Thrush, Joaquin Vanschoren, Margaret Warren, Adina Williams, Serena Yeung, Newsha Ardalani, Praveen K. Paritosh, Ce Zhang, James Y. Zou, Carole-Jean Wu, Cody Coleman, Andrew Y. Ng, Peter Mattson, Vijay Janapa Reddi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/112db88215e25b3ae2750e9eefcded94-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/112db88215e25b3ae2750e9eefcded94-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning research has long focused on models rather than datasets, and prominent datasets are used for common ML tasks without regard to the breadth, difficulty, and faithfulness of the underlying problems. Neglecting the fundamental importance of data has given rise to inaccuracy, bias, and fragility in real-world applications, and research is hindered by saturation across existing dataset benchmarks. In response, we present DataPerf, a community-led benchmark suite for evaluating ML datasets and data-centric algorithms. We aim to foster innovation in data-centric AI through competition, comparability, and reproducibility. We enable the ML community to iterate on datasets, instead of just architectures, and we provide an open, online platform with multiple rounds of challenges to support this iterative development. The first iteration of DataPerf contains five benchmarks covering a wide spectrum of data-centric techniques, tasks, and modalities in vision, speech, acquisition, debugging, and diffusion prompting, and we support hosting new contributed benchmarks from the community. The benchmarks, online evaluation platform, and baseline implementations are open source, and the MLCommons Association will maintain DataPerf to ensure long-term benefits to academia and industry.

----

## [0] Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization

**Authors**: *Quanqi Hu, Dixian Zhu, Tianbao Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1160792eab11de2bbaf9e71fce191e8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1160792eab11de2bbaf9e71fce191e8c-Abstract-Conference.html)

**Abstract**:

This paper investigates new families of compositional optimization problems, called non-smooth weakly-convex finite-sum coupled compositional optimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau envelop of the objective function. Additionally, we also extend the algorithm for solving novel non-smooth weakly-convex tri-level finite-sum coupled compositional optimization problems,  which feature a nested arrangement of three functions. Lastly, we explore the applications of our algorithms in deep learning for two-way partial AUC maximization and multi-instance two-way partial AUC maximization, using empirical studies to showcase the effectiveness of the proposed algorithms.

----

## [0] Optimal Transport for Treatment Effect Estimation

**Authors**: *Hao Wang, Jiajun Fan, Zhichao Chen, Haoxuan Li, Weiming Liu, Tianqiao Liu, Quanyu Dai, Yichao Wang, Zhenhua Dong, Ruiming Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1160e7f31d0a74abbbe1bbf7924b949c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1160e7f31d0a74abbbe1bbf7924b949c-Abstract-Conference.html)

**Abstract**:

Estimating individual treatment effects from observational data is challenging due to treatment selection bias. Prevalent methods mainly mitigate this issue by aligning different treatment groups in the latent space, the core of which is the calculation of distribution discrepancy. However, two issues that are often overlooked can render these methods invalid:(1) mini-batch sampling effects (MSE), where the calculated discrepancy is erroneous in non-ideal mini-batches with outcome imbalance and outliers;(2) unobserved confounder effects (UCE), where the unobserved confounders are not considered in the discrepancy calculation.Both of these issues invalidate the calculated discrepancy, mislead the training of estimators, and thus impede the handling of treatment selection bias.To tackle these issues, we propose Entire Space CounterFactual Regression (ESCFR), which is a new take on optimal transport technology in the context of causality.Specifically, based on the canonical optimal transport framework, we propose a relaxed mass-preserving regularizer to address the MSE issue and design a proximal factual outcome regularizer to handle the UCE issue.Extensive experiments demonstrate that ESCFR estimates distribution discrepancy accurately, handles the treatment selection bias effectively, and outperforms prevalent competitors significantly.

----

## [0] Initialization Matters: Privacy-Utility Analysis of Overparameterized Neural Networks

**Authors**: *Jiayuan Ye, Zhenyu Zhu, Fanghui Liu, Reza Shokri, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1165af8b913fb836c6280b42d6e0084f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1165af8b913fb836c6280b42d6e0084f-Abstract-Conference.html)

**Abstract**:

We analytically investigate how over-parameterization of models in randomized machine learning algorithms impacts the information leakage about their training data. Specifically, we prove a privacy bound for the KL divergence between model distributions on worst-case neighboring datasets, and explore its dependence on the initialization, width, and depth of fully connected neural networks. We find that this KL privacy bound is largely determined by the expected squared gradient norm relative to model parameters during training. Notably, for the special setting of linearized network, our analysis indicates that the squared gradient norm (and therefore the escalation of privacy loss) is tied directly to the per-layer variance of the initialization distribution. By using this analysis, we demonstrate that privacy bound improves with increasing depth under certain initializations (LeCun and Xavier), while degrades with increasing depth under other initializations (He and NTK). Our work reveals a complex interplay between privacy and depth that depends on the chosen initialization distribution. We further prove excess empirical risk bounds under a fixed KL privacy budget, and show that the interplay between privacy utility trade-off and depth is similarly affected by the initialization.

----

## [0] Cause-Effect Inference in Location-Scale Noise Models: Maximum Likelihood vs Independence Testing

**Authors**: *Xiangyu Sun, Oliver Schulte*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11715d433f6f8b9106baae0df023deb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11715d433f6f8b9106baae0df023deb3-Abstract-Conference.html)

**Abstract**:

A fundamental problem of causal discovery is cause-effect inference, to learn the correct causal direction between two random variables. Significant progress has been made through modelling the effect as a function of its cause and a noise term, which allows us to leverage assumptions about the generating function class. The recently introduced heteroscedastic location-scale noise functional models (LSNMs) combine expressive power with identifiability guarantees. LSNM model selection based on maximizing likelihood achieves state-of-the-art accuracy, when the noise distributions are correctly specified. However, through an extensive empirical evaluation, we demonstrate that the accuracy deteriorates sharply when the form of the noise distribution is misspecified by the user. Our analysis shows that the failure occurs mainly when the conditional variance in the anti-causal direction is smaller than that in the causal direction. As an alternative, we find that causal model selection through residual independence testing is much more robust to noise misspecification and misleading conditional variance.

----

## [0] M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models

**Authors**: *Wenxuan Zhang, Mahani Aljunied, Chang Gao, Yew Ken Chia, Lidong Bing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/117c5c8622b0d539f74f6d1fb082a2e9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/117c5c8622b0d539f74f6d1fb082a2e9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Despite the existence of various benchmarks for evaluating natural language processing models, we argue that human exams are a more suitable means of evaluating general intelligence for large language models (LLMs), as they inherently demand a much wider range of abilities such as language understanding, domain knowledge, and problem-solving skills. To this end, we introduce M3Exam, a novel benchmark sourced from real and official human exam questions for evaluating LLMs in a multilingual, multimodal, and multilevel context. M3Exam exhibits three unique characteristics: (1) multilingualism, encompassing questions from multiple countries that require strong multilingual proficiency and cultural knowledge; (2) multimodality, accounting for the multimodal nature of many exam questions to test the model's multimodal understanding capability; and (3) multilevel structure, featuring exams from three critical educational periods to comprehensively assess a model's proficiency at different levels. In total, M3Exam contains 12,317 questions in 9 diverse languages with three educational levels, where about 23\% of the questions require processing images for successful solving. We assess the performance of top-performing LLMs on M3Exam and find that current models, including GPT-4, still struggle with multilingual text, particularly in low-resource and non-Latin script languages. Multimodal LLMs also perform poorly with complex multimodal questions. We believe that M3Exam can be a valuable resource for comprehensively evaluating LLMs by examining their multilingual and multimodal abilities and tracking their development. Data and evaluation code is available at \url{https://github.com/DAMO-NLP-SG/M3Exam}.

----

## [0] CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

**Authors**: *Anthony Fuller, Koreen Millard, James R. Green*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)

**Abstract**:

A vital and rapidly growing application, remote sensing offers vast yet sparsely labeled, spatially aligned multimodal data; this makes self-supervised learning algorithms invaluable. We present CROMA: a framework that combines contrastive and reconstruction self-supervised objectives to learn rich unimodal and multimodal representations. Our method separately encodes masked-out multispectral optical and synthetic aperture radar samples—aligned in space and time—and performs cross-modal contrastive learning. Another encoder fuses these sensors, producing joint multimodal encodings that are used to predict the masked patches via a lightweight decoder. We show that these objectives are complementary when leveraged on spatially aligned multimodal data. We also introduce X- and 2D-ALiBi, which spatially biases our cross- and self-attention matrices. These strategies improve representations and allow our models to effectively extrapolate to images up to $17.6\times$ larger at test-time. CROMA outperforms the current SoTA multispectral model, evaluated on: four classification benchmarks—finetuning (avg.$\uparrow$ 1.8%), linear (avg.$\uparrow$ 2.4%) and nonlinear (avg.$\uparrow$ 1.4%) probing, $k$NN classification (avg.$\uparrow$ 3.5%), and $K$-means clustering (avg.$\uparrow$ 8.4%); and three segmentation benchmarks (avg.$\uparrow$ 6.4%). CROMA’s rich, optionally multimodal representations can be widely leveraged across remote sensing applications.

----

## [0] OpenAGI: When LLM Meets Domain Experts

**Authors**: *Yingqiang Ge, Wenyue Hua, Kai Mei, Jianchao Ji, Juntao Tan, Shuyuan Xu, Zelong Li, Yongfeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1190733f217404edc8a7f4e15a57f301-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1190733f217404edc8a7f4e15a57f301-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Human Intelligence (HI) excels at combining basic skills to solve complex tasks. This capability is vital for Artificial Intelligence (AI) and should be embedded in comprehensive AI Agents, enabling them to harness expert models for complex task-solving towards Artificial General Intelligence (AGI). Large Language Models (LLMs) show promising learning and reasoning abilities, and can effectively use external models, tools, plugins, or APIs to tackle complex problems. In this work, we introduce OpenAGI, an open-source AGI research and development platform designed for solving multi-step, real-world tasks. Specifically, OpenAGI uses a dual strategy, integrating standard benchmark tasks for benchmarking and evaluation, and open-ended tasks including more expandable models, tools, plugins, or APIs for creative problem-solving. Tasks are presented as natural language queries to the LLM, which then selects and executes appropriate models. We also propose a Reinforcement Learning from Task Feedback (RLTF) mechanism that uses task results to improve the LLM's task-solving ability, which creates a self-improving AI feedback loop. While we acknowledge that AGI is a broad and multifaceted research challenge with no singularly defined solution path, the integration of LLMs with domain-specific expert models, inspired by mirroring the blend of general and specialized intelligence in humans, offers a promising approach towards AGI. We are open-sourcing the OpenAGI project's code, dataset, benchmarks, evaluation methods, and the UI demo to foster community involvement in AGI advancement: https://github.com/agiresearch/OpenAGI.

----

## [0] Neural Frailty Machine: Beyond proportional hazard assumption in neural survival regressions

**Authors**: *Ruofan Wu, Jiawei Qiao, Mingzhe Wu, Wen Yu, Ming Zheng, Tengfei Liu, Tianyi Zhang, Weiqiang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11a7f429d75f9f8c6e9c630aeb6524b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11a7f429d75f9f8c6e9c630aeb6524b5-Abstract-Conference.html)

**Abstract**:

We present neural frailty machine (NFM), a powerful and flexible neural modeling framework for survival regressions. The NFM framework utilizes the classical idea of multiplicative frailty in survival analysis as a principled way of extending the proportional hazard assumption, at the same time being able to leverage the strong approximation power of neural architectures for handling nonlinear covariate dependence. Two concrete models are derived under the framework that extends neural proportional hazard models and nonparametric hazard regression models. Both models allow efficient training under the likelihood objective. Theoretically, for both proposed models, we establish statistical guarantees of neural function approximation with respect to nonparametric components via characterizing their rate of convergence. Empirically, we provide synthetic experiments that verify our theoretical statements. We also conduct experimental evaluations over $6$ benchmark datasets of different scales, showing that the proposed NFM models achieve predictive performance comparable to or sometimes surpassing state-of-the-art survival models. Our code is publicly availabel at https://github.com/Rorschach1989/nfm

----

## [0] Non-autoregressive Machine Translation with Probabilistic Context-free Grammar

**Authors**: *Shangtong Gui, Chenze Shao, Zhengrui Ma, Xishan Zhang, Yunji Chen, Yang Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11c7f1dd168439884b6dfb43a7891432-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11c7f1dd168439884b6dfb43a7891432-Abstract-Conference.html)

**Abstract**:

Non-autoregressive Transformer(NAT) significantly accelerates the inference of neural machine translation. However, conventional NAT models suffer from limited expression power and performance degradation compared to autoregressive (AT) models due to the assumption of conditional independence among target tokens. To address these limitations, we propose a novel approach called PCFG-NAT, which leverages a specially designed Probabilistic Context-Free Grammar (PCFG) to enhance the ability of NAT models to capture complex dependencies among output tokens. Experimental results on major machine translation benchmarks demonstrate that PCFG-NAT further narrows the gap in translation quality between NAT and AT models. Moreover, PCFG-NAT facilitates a deeper understanding of the generated sentences, addressing the lack of satisfactory explainability in neural machine translation. Code is publicly available at https://github.com/ictnlp/PCFG-NAT.

----

## [0] Constrained Policy Optimization with Explicit Behavior Density For Offline Reinforcement Learning

**Authors**: *Jing Zhang, Chi Zhang, Wenjia Wang, Bingyi Jing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11e1900e680f5fe1893a8e27362dbe2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11e1900e680f5fe1893a8e27362dbe2c-Abstract-Conference.html)

**Abstract**:

Due to the inability to interact with the environment, offline reinforcement learning (RL) methods face the challenge of estimating the Out-of-Distribution (OOD) points. Existing methods for addressing this issue either control policy to exclude the OOD action or make the $Q$ function pessimistic. However, these methods can be overly conservative or fail to identify OOD areas accurately. To overcome this problem, we propose a Constrained Policy optimization with Explicit Behavior density (CPED) method that utilizes a flow-GAN model to explicitly estimate the density of behavior policy. By estimating the explicit density, CPED can accurately identify the safe region and enable exploration within the region, resulting in less conservative learning policies.  We further provide theoretical results for both the flow-GAN estimator and performance guarantee for CPED by showing that CPED can find the optimal $Q$-function value. Empirically, CPED outperforms existing alternatives on various standard offline reinforcement learning tasks, yielding higher expected returns.

----

## [0] Large Language Models are Fixated by Red Herrings: Exploring Creative Problem Solving and Einstellung Effect using the Only Connect Wall Dataset

**Authors**: *Saeid Alavi Naeini, Raeid Saqur, Mozhgan Saeidi, John Giorgi, Babak Taati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11e3e0f1b29dcd31bd0952bfc1357f68-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/11e3e0f1b29dcd31bd0952bfc1357f68-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The quest for human imitative AI has been an enduring topic in AI research since inception. The technical evolution and emerging capabilities of the latest cohort of large language models (LLMs) have reinvigorated the subject beyond academia to cultural zeitgeist. While recent NLP evaluation benchmark tasks test some aspects of human-imitative behaviour (e.g., BIG-bench's `human-like behavior' tasks), few, if not none, examine creative problem solving abilities. Creative problem solving in humans is a well-studied topic in cognitive neuroscience with standardized tests that predominantly use ability to associate (heterogeneous) connections among clue words as a metric for creativity. Exposure to misleading stimuli --- distractors dubbed red herrings --- impede human performance in such tasks via the fixation effect and Einstellung paradigm. In cognitive neuroscience studies, such fixations are experimentally induced by pre-exposing participants to orthographically similar incorrect words to subsequent word-fragments or clues. The popular British quiz show Only Connect's Connecting Wall segment essentially mimics Mednick's Remote Associates Test (RAT) formulation with built-in, deliberate red herrings, that makes it an ideal proxy dataset to explore and study fixation effect and Einstellung paradigm from cognitive neuroscience in LLMs. In addition to presenting the novel Only Connect Wall (OCW) dataset, we also report results from our evaluation of selected pre-trained language models and LLMs (including OpenAI's GPT series) on creative problem solving tasks like grouping clue words by heterogeneous connections, and identifying correct open knowledge domain connections in respective groups. We synthetically generate two additional datasets: OCW-Randomized, OCW-WordNet to further analyze our red-herrings hypothesis in language models.The code and link to the dataset is available at url.

----

## [0] Formalizing locality for normative synaptic plasticity models

**Authors**: *Colin Bredenberg, Ezekiel Williams, Cristina Savin, Blake A. Richards, Guillaume Lajoie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/120339238f293d4ae53a7167403abc4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/120339238f293d4ae53a7167403abc4b-Abstract-Conference.html)

**Abstract**:

In recent years, many researchers have proposed new models for synaptic plasticity in the brain based on principles of machine learning. The central motivation has been the development of learning algorithms that are able to learn difficult tasks while qualifying as "biologically plausible". However, the concept of a biologically plausible learning algorithm is only heuristically defined as an algorithm that is potentially implementable by biological neural networks. Further, claims that neural circuits could implement any given algorithm typically rest on an amorphous concept of "locality" (both in space and time). As a result, it is unclear what many proposed local learning algorithms actually predict biologically, and which of these are consequently good candidates for experimental investigation. Here, we address this lack of clarity by proposing formal and operational definitions of locality. Specifically, we define different classes of locality, each of which makes clear what quantities cannot be included in a learning rule if an algorithm is to qualify as local with respect to a given (biological) constraint. We subsequently use this framework to distill testable predictions from various classes of biologically plausible synaptic plasticity models that are robust to arbitrary choices about neural network architecture. Therefore, our framework can be used to guide claims of biological plausibility and to identify potential means of experimentally falsifying a proposed learning algorithm for the brain.

----

## [0] Exact Verification of ReLU Neural Control Barrier Functions

**Authors**: *Hongchao Zhang, Junlin Wu, Yevgeniy Vorobeychik, Andrew Clark*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html)

**Abstract**:

Control Barrier Functions (CBFs) are a popular approach for safe control of nonlinear systems. In CBF-based control, the desired safety properties of the system are mapped to nonnegativity of a CBF, and the control input is chosen to ensure that the CBF remains nonnegative for all time. Recently, machine learning methods that represent CBFs as neural networks (neural control barrier functions, or NCBFs) have shown great promise due to the universal representability of neural networks. However, verifying that a learned CBF guarantees safety remains a challenging research problem. This paper presents novel exact conditions and algorithms for verifying safety of feedforward NCBFs with ReLU activation functions. The key challenge in doing so is that, due to the piecewise linearity of the ReLU function, the NCBF will be nondifferentiable at certain points, thus invalidating traditional safety verification methods that assume a smooth barrier function. We resolve this issue by leveraging a generalization of Nagumo's theorem for proving invariance of sets with nonsmooth boundaries to derive necessary and sufficient conditions for safety. Based on this condition, we propose an algorithm for safety verification of NCBFs that first decomposes the NCBF into piecewise linear segments and then solves a nonlinear program to verify safety of each segment as well as the intersections of the linear segments. We  mitigate the complexity by only considering the boundary of the safe region and by pruning  the segments with Interval Bound Propagation (IBP) and linear relaxation. We evaluate our approach through numerical studies with comparison to state-of-the-art SMT-based methods. Our code is available at https://github.com/HongchaoZhang-HZ/exactverif-reluncbf-nips23.

----

## [0] Normalization-Equivariant Neural Networks with Application to Image Denoising

**Authors**: *Sébastien Herbreteau, Emmanuel Moebel, Charles Kervrann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12143893d9d37c3569dda800b95cabd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12143893d9d37c3569dda800b95cabd9-Abstract-Conference.html)

**Abstract**:

In many information processing systems, it may be desirable to ensure that any change of the input, whether by shifting or  scaling, results in a corresponding change in the system response.  While deep neural networks are gradually replacing all traditional automatic processing methods, they surprisingly do not guarantee such normalization-equivariance (scale + shift) property, which can be detrimental in many applications. To address this issue, we propose a methodology for adapting existing neural networks so that normalization-equivariance holds by design. Our main claim is that not only ordinary convolutional layers, but also all activation functions, including the ReLU (rectified linear unit), which are applied element-wise to the pre-activated neurons, should be completely removed from neural networks and replaced by better conditioned alternatives. To this end, we introduce affine-constrained convolutions and channel-wise sort pooling layers as surrogates and show that these two architectural modifications do preserve normalization-equivariance without loss of performance. Experimental results in image denoising show that normalization-equivariant neural networks, in addition to their better conditioning, also provide much better generalization across noise levels.

----

## [0] Budgeting Counterfactual for Offline RL

**Authors**: *Yao Liu, Pratik Chaudhari, Rasool Fakoor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/121db870b0470dd63bb5bc59c724275a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/121db870b0470dd63bb5bc59c724275a-Abstract-Conference.html)

**Abstract**:

The main challenge of offline reinforcement learning, where data is limited, arises from a sequence of counterfactual reasoning dilemmas within the realm of potential actions: What if we were to choose a different course of action? These circumstances frequently give rise to extrapolation errors, which tend to accumulate exponentially with the problem horizon. Hence, it becomes crucial to acknowledge that not all decision steps are equally important to the final outcome, and to budget the number of counterfactual decisions a policy make in order to control the extrapolation. Contrary to existing approaches that use regularization on either the policy or value function, we propose an approach to explicitly bound the amount of out-of-distribution actions during training. Specifically, our method utilizes dynamic programming to decide where to extrapolate and where not to, with an upper bound on the decisions different from behavior policy. It balances between the potential for improvement from taking out-of-distribution actions and the risk of making errors due to extrapolation. Theoretically, we justify our method by the constrained optimality of the fixed point solution to our $Q$ updating rules. Empirically, we show that the overall performance of our method is better than the state-of-the-art offline RL methods on tasks in the widely-used D4RL benchmarks.

----

## [0] Federated Conditional Stochastic Optimization

**Authors**: *Xidong Wu, Jianhui Sun, Zhengmian Hu, Junyi Li, Aidong Zhang, Heng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1229eaae5bf1db93e1e4c539258eb472-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1229eaae5bf1db93e1e4c539258eb472-Abstract-Conference.html)

**Abstract**:

Conditional stochastic optimization has found applications in a wide range of machine learning tasks, such as invariant learning, AUPRC maximization, and meta-learning. As the demand for training models with large-scale distributed data grows in these applications, there is an increasing need for communication-efficient distributed optimization algorithms, such as federated learning algorithms. This paper considers the nonconvex conditional stochastic optimization in federated learning and proposes the first federated conditional stochastic optimization algorithm (FCSG) with a conditional stochastic gradient estimator and a momentum-based algorithm (\emph{i.e.}, FCSG-M). To match the lower bound complexity in the single-machine setting, we design an accelerated algorithm (Acc-FCSG-M) via the variance reduction to achieve the best sample and communication complexity. Compared with the existing optimization analysis for Meta-Learning in FL, federated conditional stochastic optimization considers the sample of tasks. Extensive experimental results on various tasks validate the efficiency of these algorithms.

----

## [0] LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections

**Authors**: *Muhammad Jehanzeb Mirza, Leonid Karlinsky, Wei Lin, Horst Possegger, Mateusz Kozinski, Rogério Feris, Horst Bischof*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/123a18dfd821c8b440f42a00a27648d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/123a18dfd821c8b440f42a00a27648d6-Abstract-Conference.html)

**Abstract**:

Recently, large-scale pre-trained Vision and Language (VL) models have set a new state-of-the-art (SOTA) in zero-shot visual classification enabling open-vocabulary recognition of potentially unlimited set of categories defined as simple language prompts. However, despite these great advances, the performance of these zero-shot classifiers still falls short of the results of dedicated (closed category set) classifiers trained with supervised fine-tuning. In this paper we show, for the first time, how to reduce this gap without any labels and without any paired VL data, using an unlabeled image collection and a set of texts auto-generated using a Large Language Model (LLM) describing the categories of interest and effectively substituting labeled visual instances of those categories. Using our label-free approach, we are able to attain significant performance improvements over the zero-shot performance of the base VL model and other contemporary methods and baselines on a wide variety of datasets, demonstrating absolute improvement of up to $11.7\%$ ($3.8\%$ on average) in the label-free setting. Moreover, despite our approach being label-free, we observe $1.3\%$ average gains over leading few-shot prompting baselines that do use 5-shot supervision.

----

## [0] Contextually Affinitive Neighborhood Refinery for Deep Clustering

**Authors**: *Chunlin Yu, Ye Shi, Jingya Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/123cfe7d8b7702ac97aaf4468fc05fa5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/123cfe7d8b7702ac97aaf4468fc05fa5-Abstract-Conference.html)

**Abstract**:

Previous endeavors in self-supervised learning have enlightened the research of deep clustering from an instance discrimination perspective. Built upon this foundation, recent studies further highlight the importance of grouping semantically similar instances. One effective method to achieve this is by promoting the semantic structure preserved by neighborhood consistency. However, the samples in the local neighborhood may be limited due to their close proximity to each other,  which may not provide substantial and diverse supervision signals. Inspired by the versatile re-ranking methods in the context of image retrieval, we propose to employ an efficient online re-ranking process to mine more informative neighbors in a Contextually Affinitive (ConAff) Neighborhood, and then encourage the cross-view neighborhood consistency. To further mitigate the intrinsic neighborhood noises near cluster boundaries, we propose a progressively relaxed boundary filtering strategy to circumvent the issues brought by noisy neighbors. Our method can be easily integrated into the generic self-supervised frameworks and outperforms the state-of-the-art methods on several popular benchmarks.

----

## [0] Differentiable Blocks World: Qualitative 3D Decomposition by Rendering Primitives

**Authors**: *Tom Monnier, Jake Austin, Angjoo Kanazawa, Alexei A. Efros, Mathieu Aubry*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html)

**Abstract**:

Given a set of calibrated images of a scene, we present an approach that produces a simple, compact, and actionable 3D world representation by means of 3D primitives. While many approaches focus on recovering high-fidelity 3D scenes, we focus on parsing a scene into mid-level 3D representations made of a small set of textured primitives. Such representations are interpretable, easy to manipulate and suited for physics-based simulations. Moreover, unlike existing primitive decomposition methods that rely on 3D input data, our approach operates directly on images through differentiable rendering. Specifically, we model primitives as textured superquadric meshes and optimize their parameters from scratch with an image rendering loss. We highlight the importance of modeling transparency for each primitive, which is critical for optimization and also enables handling varying numbers of primitives. We show that the resulting textured primitives faithfully reconstruct the input images and accurately model the visible 3D points, while providing amodal shape completions of unseen object regions. We compare our approach to the state of the art on diverse scenes from DTU, and demonstrate its robustness on real-life captures from BlendedMVS and Nerfstudio. We also showcase how our results can be used to effortlessly edit a scene or perform physical simulations. Code and video results are available at https://www.tmonnier.com/DBW.

----

## [0] Learning Shared Safety Constraints from Multi-task Demonstrations

**Authors**: *Konwoo Kim, Gokul Swamy, Zuxin Liu, Ding Zhao, Sanjiban Choudhury, Steven Z. Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/124dde499d62b58e97e42a45b26d7369-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/124dde499d62b58e97e42a45b26d7369-Abstract-Conference.html)

**Abstract**:

Regardless of the particular task we want to perform in an environment, there are often shared safety constraints we want our agents to respect. For example, regardless of whether it is making a sandwich or clearing the table, a kitchen robot should not break a plate. Manually specifying such a constraint can be both time-consuming and error-prone. We show how to learn constraints from expert demonstrations of safe task completion by extending inverse reinforcement learning (IRL) techniques to the space of constraints. Intuitively, we learn constraints that forbid highly rewarding behavior that the expert could have taken but chose not to. Unfortunately, the constraint learning problem is rather ill-posed and typically leads to overly conservative constraints that forbid all behavior that the expert did not take. We counter this by leveraging diverse demonstrations that naturally occur in multi-task setting to learn a tighter set of constraints. We validate our method with simulation experiments on high-dimensional continuous control tasks.

----

## [0] Don't Stop Pretraining? Make Prompt-based Fine-tuning Powerful Learner

**Authors**: *Zhengxiang Shi, Aldo Lipani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1289f9195d2ef8cfdfe5f50930c4a7c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1289f9195d2ef8cfdfe5f50930c4a7c4-Abstract-Conference.html)

**Abstract**:

Language models (LMs) trained on vast quantities of unlabelled data have greatly advanced the field of natural language processing (NLP). In this study, we re-visit the widely accepted notion in NLP that continued pre-training LMs on task-related texts improves the performance of fine-tuning (FT) in downstream tasks. Through experiments on eight single-sentence tasks and eight sentence-pair tasks in both semi-supervised and fully-supervised settings, we find that conventional continued pre-training does not consistently provide benefits and can even be detrimental for sentence-pair tasks or when prompt-based FT is used. To tackle these issues, we propose Prompt-based Continued Pre-training (PCP), which combines the idea of instruction tuning with conventional continued pre-training. Our approach aims to improve the performance of prompt-based FT by presenting both task-related texts and prompt templates to LMs through unsupervised pre-training objectives before fine-tuning for the target task. Our empirical evaluations on 21 benchmarks demonstrate that the PCP consistently improves the performance of state-of-the-art prompt-based FT approaches (up to 20.1% absolute) in both semi-supervised and fully-supervised settings, even with only hundreds of unlabelled examples. Additionally, prompt-based FT with PCP outperforms state-of-the-art semi-supervised approaches with greater simplicity, eliminating the need for an iterative process and extra data augmentation. Our further analysis explores the performance lower bound of the PCP and reveals that the advantages of PCP persist across different sizes of models and datasets.

----

## [0] GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning

**Authors**: *Haiteng Zhao, Shengchao Liu, Chang Ma, Hannan Xu, Jie Fu, Zhihong Deng, Lingpeng Kong, Qi Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/129033c7c08be683059559e8d6bfd460-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/129033c7c08be683059559e8d6bfd460-Abstract-Conference.html)

**Abstract**:

Molecule property prediction has gained significant attention in recent years. The main bottleneck is the label insufficiency caused by expensive lab experiments. In order to alleviate this issue and to better leverage textual knowledge for tasks, this study investigates the feasibility of employing natural language instructions to accomplish molecule-related tasks in a zero-shot setting. We discover that existing molecule-text models perform poorly in this setting due to inadequate treatment of instructions and limited capacity for graphs.  To overcome these issues, we propose GIMLET, which unifies language models for both graph and text data. By adopting generalized position embedding, our model is extended to encode both graph structures and instruction text without additional graph encoding modules. GIMLET also decouples encoding of the graph from tasks instructions in the attention mechanism, enhancing the generalization of graph features across novel tasks. We construct a dataset consisting of more than two thousand molecule tasks with corresponding instructions derived from task descriptions. We pretrain GIMLET on the molecule tasks along with instructions, enabling the model to transfer effectively to a broad range of tasks. Experimental results demonstrate that GIMLET significantly outperforms molecule-text baselines in instruction-based zero-shot learning, even achieving closed results to supervised GNN models on tasks such as toxcast and muv.

----

## [0] GEX: A flexible method for approximating influence via Geometric Ensemble

**Authors**: *Sungyub Kim, Kyungsu Kim, Eunho Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1297ca5c906f4bada8f5f6f4e80f9dd2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1297ca5c906f4bada8f5f6f4e80f9dd2-Abstract-Conference.html)

**Abstract**:

Through a deeper understanding of predictions of neural networks, Influence Function (IF) has been applied to various tasks such as detecting and relabeling mislabeled samples, dataset pruning, and separation of data sources in practice. However, we found standard approximations of IF suffer from performance degradation due to oversimplified influence distributions caused by their bilinear approximation, suppressing the expressive power of samples with a relatively strong influence. To address this issue, we propose a new interpretation of existing IF approximations as an average relationship between two linearized losses over parameters sampled from the Laplace approximation (LA). In doing so, we highlight two significant limitations of current IF approximations: the linearity of gradients and the singularity of Hessian. Accordingly, by improving each point, we introduce a new IF approximation method with the following features: i) the removal of linearization to alleviate the bilinear constraint and ii) the utilization of Geometric Ensemble (GE) tailored for non-linear losses. Empirically, our approach outperforms existing IF approximations for downstream tasks with lighter computation, thereby providing new feasibility of low-complexity/nonlinear-based IF design.

----

## [0] Offline Reinforcement Learning for Mixture-of-Expert Dialogue Management

**Authors**: *Dhawal Gupta, Yinlam Chow, Azamat Tulepbergenov, Mohammad Ghavamzadeh, Craig Boutilier*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12bcf58a1c09a0fcb5310f3589291ab4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12bcf58a1c09a0fcb5310f3589291ab4-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) has shown great promise for developing agents for dialogue management (DM) that are non-myopic, conduct rich conversations, and maximize overall user satisfaction. Despite the advancements in RL and language models (LMs), employing RL to drive conversational chatbots still poses significant challenges. A primary issue stems from RLâ€™s dependency on online exploration for effective learning, a process that can be costly. Moreover, engaging in online interactions with humans during the training phase can raise safety concerns, as the LM can potentially generate unwanted outputs. This issue is exacerbated by the combinatorial action spaces facing these algorithms, as most LM agents generate responses at the word level. We develop various RL algorithms, specialized in dialogue planning, that leverage recent Mixture-of-Expert Language Models (MoE-LMs)---models that capture diverse semantics, generate utterances reflecting different intents, and are amenable for multi-turn DM. By exploiting the MoE-LM structure, our methods significantly reduce the size of the action space and improve the efficacy of RL-based DM. We evaluate our methods in open-domain dialogue to demonstrate their effectiveness with respect to the diversity of intent in generated utterances and overall DM performance.

----

## [0] Binary Classification with Confidence Difference

**Authors**: *Wei Wang, Lei Feng, Yuchen Jiang, Gang Niu, Min-Ling Zhang, Masashi Sugiyama*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12c118ef87fde56a10bd858842781b34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12c118ef87fde56a10bd858842781b34-Abstract-Conference.html)

**Abstract**:

Recently, learning with soft labels has been shown to achieve better performance than learning with hard labels in terms of model generalization, calibration, and robustness. However, collecting pointwise labeling confidence for all training examples can be challenging and time-consuming in real-world scenarios. This paper delves into a novel weakly supervised binary classification problem called confidence-difference (ConfDiff) classification. Instead of pointwise labeling confidence, we are given only unlabeled data pairs with confidence difference that specifies the difference in the probabilities of being positive. We propose a risk-consistent approach to tackle this problem and show that the estimation error bound achieves the optimal convergence rate. We also introduce a risk correction approach to mitigate overfitting problems, whose consistency and convergence rate are also proven. Extensive experiments on benchmark data sets and a real-world recommender system data set validate the effectiveness of our proposed approaches in exploiting the supervision information of the confidence difference.

----

## [0] On student-teacher deviations in distillation: does it pay to disobey?

**Authors**: *Vaishnavh Nagarajan, Aditya Krishna Menon, Srinadh Bhojanapalli, Hossein Mobahi, Sanjiv Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12d286282e1be5431ea05262a21f415c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12d286282e1be5431ea05262a21f415c-Abstract-Conference.html)

**Abstract**:

Knowledge distillation (KD) has been widely used to improve the test accuracy of a "student" network, by training it to mimic the soft probabilities of a trained "teacher" network. Yet, it has been shown in recent work that, despite being trained to fit the teacher's probabilities, the student may not only significantly deviate from the teacher probabilities, but may also outdo than the teacher in performance. Our work aims to reconcile this seemingly paradoxical observation. Specifically, we characterize the precise nature of the student-teacher deviations, and argue how they can co-occur with better generalization.  First, through experiments on  image and language data, we identify that these probability deviations correspond to the student systematically exaggerating the confidence levels of the teacher.Next, we theoretically and empirically establish another form of exaggeration in some simple settings: KD exaggerates the implicit bias of gradient descent in converging faster along the top eigendirections of the data. Finally, we tie these two observations together: we demonstrate that the exaggerated bias of KD can simultaneously result in both (a) the exaggeration of confidence and (b) the improved generalization of the student, thus offering a resolution to the apparent paradox.  Our analysis brings existing theory and practice closer by considering the role of gradient descent in KD and by demonstrating the exaggerated bias effect in both theoretical and empirical settings.

----

## [0] Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis

**Authors**: *Victor Letzelter, Mathieu Fontaine, Mickaël Chen, Patrick Pérez, Slim Essid, Gaël Richard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12d7ba753894ed348904df1bf0ce02ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12d7ba753894ed348904df1bf0ce02ec-Abstract-Conference.html)

**Abstract**:

We introduce Resilient Multiple Choice Learning (rMCL), an extension of the MCL approach for conditional distribution estimation in regression settings where multiple targets may be sampled for each training input.Multiple Choice Learning is a simple framework to tackle multimodal density estimation, using the Winner-Takes-All (WTA) loss for a set of hypotheses. In regression settings, the existing MCL variants focus on merging the hypotheses, thereby eventually sacrificing the diversity of the predictions. In contrast, our method relies on a novel learned scoring scheme underpinned by a mathematical framework based on Voronoi tessellations of the output space, from which we can derive a probabilistic interpretation.After empirically validating rMCL with experiments on synthetic data, we further assess its merits on the sound source localization problem, demonstrating its practical usefulness and the relevance of its interpretation.

----

## [0] Graph of Circuits with GNN for Exploring the Optimal Design Space

**Authors**: *Aditya Shahane, Saripilli Swapna Manjiri, Ankesh Jain, Sandeep Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12da92b7c64176eb6eb6ad0ae31554fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12da92b7c64176eb6eb6ad0ae31554fd-Abstract-Conference.html)

**Abstract**:

The design automation of analog circuits poses significant challenges in terms of the large design space, complex interdependencies between circuit specifications, and resource-intensive simulations. To address these challenges, this paper presents an innovative framework called the Graph of Circuits Explorer (GCX). Leveraging graph structure learning along with graph neural networks, GCX enables the creation of a surrogate model that facilitates efficient exploration of the optimal design space within a semi-supervised learning framework which reduces the need for large labelled datasets. The proposed approach comprises three key stages. First, we learn the geometric representation of circuits and enrich it with technology information to create a comprehensive feature vector. Subsequently, integrating feature-based graph learning with few-shot and zero-shot learning  enhances the generalizability in predictions for unseen circuits. Finally, we introduce two algorithms namely, EASCO and ASTROG which upon integration with GCX optimize the available samples to yield the optimal circuit configuration meeting the designer's criteria. The effectiveness of the proposed approach is demonstrated through simulated performance evaluation of various circuits, using derived parameters in 180nm CMOS technology. Furthermore, the generalizability of the approach is extended to higher-order topologies and different technology nodes such as 65nm and 45nm CMOS process nodes.

----

## [0] Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data

**Authors**: *Xin Zheng, Miao Zhang, Chunyang Chen, Quoc Viet Hung Nguyen, Xingquan Zhu, Shirui Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13183a224208671a6fc33ba1aa661ec4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13183a224208671a6fc33ba1aa661ec4-Abstract-Conference.html)

**Abstract**:

Graph condensation, which reduces the size of a large-scale graph by synthesizing a small-scale condensed graph as its substitution, has immediate benefits for various graph learning tasks.However, existing graph condensation methods rely on the joint optimization of nodes and structures in the condensed graph, and overlook critical issues in effectiveness and generalization ability.In this paper, we advocate a new Structure-Free Graph Condensation paradigm, named SFGC, to distill a large-scale graph into a small-scale graph node set without explicit graph structures, i.e., graph-free data.Our idea is to implicitly encode topology structure information into the node attributes in the synthesized graph-free data, whose topology is reduced to an identity matrix.Specifically, SFGC contains two collaborative components: (1) a training trajectory meta-matching scheme for effectively synthesizing small-scale graph-free data;(2) a graph neural feature score metric for dynamically evaluating the quality of the condensed data. Through training trajectory meta-matching, SFGC aligns the long-term GNN learning behaviors between the large-scale graph and the condensed small-scale graph-free data, ensuring comprehensive and compact transfer of informative knowledge to the graph-free data.Afterward, the underlying condensed graph-free data would be dynamically evaluated with the graph neural feature score, which is a closed-form metric for ensuring the excellent expressiveness of the condensed graph-free data.Extensive experiments verify the superiority of SFGC across different condensation ratios.

----

## [0] Visual Programming for Step-by-Step Text-to-Image Generation and Evaluation

**Authors**: *Jaemin Cho, Abhay Zala, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13250eb13871b3c2c0a0667b54bad165-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13250eb13871b3c2c0a0667b54bad165-Abstract-Conference.html)

**Abstract**:

As large language models have demonstrated impressive performance in many domains, recent works have adopted language models (LMs) as controllers of visual modules for vision-and-language tasks. While existing work focuses on equipping LMs with visual understanding, we propose two novel interpretable/explainable visual programming frameworks for text-to-image (T2I) generation and evaluation. First, we introduce VPGen, an interpretable step-by-step T2I generation framework that decomposes T2I generation into three steps: object/count generation, layout generation, and image generation. We employ an LM to handle the first two steps (object/count generation and layout generation), by finetuning it on text-layout pairs. Our step-by-step T2I generation framework provides stronger spatial control than end-to-end models, the dominant approach for this task. Furthermore, we leverage the world knowledge of pretrained LMs, overcoming the limitation of previous layout-guided T2I works that can only handle predefined object classes. We demonstrate that our VPGen has improved control in counts/spatial relations/scales of objects than state-of-the-art T2I generation models. Second, we introduce VPEval, an interpretable and explainable evaluation framework for T2I generation based on visual programming. Unlike previous T2I evaluations with a single scoring model that is accurate in some skills but unreliable in others, VPEval produces evaluation programs that invoke a set of visual modules that are experts in different skills, and also provides visual+textual explanations of the evaluation results. Our analysis shows that VPEval provides a more human-correlated evaluation for skill-specific and open-ended prompts than widely used single model-based evaluation. We hope that our work encourages future progress on interpretable/explainable generation and evaluation for T2I models.

----

## [0] Auditing Fairness by Betting

**Authors**: *Ben Chugg, Santiago Cortes-Gomez, Bryan Wilder, Aaditya Ramdas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1338c277525011f20166cf740952bb47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1338c277525011f20166cf740952bb47-Abstract-Conference.html)

**Abstract**:

We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a  probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics---the ``testing by betting'' framework in particular. These connections ensure that our methods are interpretable, fast, and easy to implement. We demonstrate the efficacy of our approach on three benchmark fairness datasets.

----

## [0] Truly Scale-Equivariant Deep Nets with Fourier Layers

**Authors**: *Md Ashiqur Rahman, Raymond A. Yeh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1343edb2739a61a6e20bd8764e814b50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1343edb2739a61a6e20bd8764e814b50-Abstract-Conference.html)

**Abstract**:

In computer vision, models must be able to adapt to changes in image resolution to effectively carry out tasks such as image segmentation; This is known as scale-equivariance. Recent works have made progress in developing scale-equivariant convolutional neural networks, e.g., through weight-sharing and kernel resizing. However, these networks are not truly scale-equivariant in practice. Specifically, they do not consider anti-aliasing as they formulate the down-scaling operation in the continuous domain. To address this shortcoming, we directly formulate down-scaling in the discrete domain with consideration of anti-aliasing. We then propose a novel architecture based on Fourier layers to achieve truly scale-equivariant deep nets, i.e., absolute zero equivariance-error. Following prior works, we test this model on MNIST-scale and STL-10 datasets. Our proposed model achieves competitive classification performance while maintaining zero equivariance-error.

----

## [0] Projection-Free Methods for Stochastic Simple Bilevel Optimization with Convex Lower-level Problem

**Authors**: *Jincheng Cao, Ruichen Jiang, Nazanin Abolfazli, Erfan Yazdandoost Hamedani, Aryan Mokhtari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/136729ae4b0fee25a0d28077442506da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/136729ae4b0fee25a0d28077442506da-Abstract-Conference.html)

**Abstract**:

In this paper, we study a class of stochastic bilevel optimization problems, also known as stochastic simple bilevel optimization, where we minimize a smooth stochastic objective function over the optimal solution set of another stochastic convex optimization problem. We introduce novel stochastic bilevel optimization methods that locally approximate the solution set of the lower-level problem via a stochastic cutting plane, and then run a conditional gradient update with variance reduction techniques to control the error induced by using stochastic gradients. For the case that the upper-level function is convex, our method requires $\mathcal{O}(\max\\{1/\epsilon_f^{2},1/\epsilon_g^{2}\\}) $ stochastic oracle queries to obtain a solution that is $\epsilon_f$-optimal for the upper-level and $\epsilon_g$-optimal for the lower-level. This guarantee improves the previous best-known complexity of $\mathcal{O}(\max\\{1/\epsilon_f^{4},1/\epsilon_g^{4}\\})$. Moreover, for the case that the upper-level function is non-convex, our method requires at most $\mathcal{O}(\max\\{1/\epsilon_f^{3},1/\epsilon_g^{3}\\}) $ stochastic oracle queries to find an $(\epsilon_f, \epsilon_g)$-stationary point. In the finite-sum setting, we show that the number of stochastic oracle calls required by our method are  $\mathcal{O}(\sqrt{n}/\epsilon)$ and $\mathcal{O}(\sqrt{n}/\epsilon^{2})$ for the convex and non-convex settings, respectively, where $\epsilon=\min \\{\epsilon_f,\epsilon_g\\}$.

----

## [0] On the Implicit Bias of Linear Equivariant Steerable Networks

**Authors**: *Ziyu Chen, Wei Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/136a45cd9b841bf785625709a19c6508-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/136a45cd9b841bf785625709a19c6508-Abstract-Conference.html)

**Abstract**:

We study the implicit bias of gradient flow on linear equivariant steerable networks in group-invariant binary classification. Our findings reveal that the parameterized predictor converges in direction to the unique group-invariant classifier with a maximum margin defined by the input group action. Under a unitary assumption on the input representation, we establish the equivalence between steerable networks and data augmentation. Furthermore, we demonstrate the improved margin and generalization bound of steerable networks over their non-invariant counterparts.

----

## [0] Memory-Constrained Algorithms for Convex Optimization

**Authors**: *Moïse Blanchard, Junhui Zhang, Patrick Jaillet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1395b425d06a50e42fafe91cf04f3a98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1395b425d06a50e42fafe91cf04f3a98-Abstract-Conference.html)

**Abstract**:

We propose a family of recursive cutting-plane algorithms to solve feasibility problems with constrained memory, which can also be used for first-order convex optimization. Precisely, in order to find a point within a ball of radius $\epsilon$ with a separation oracle in dimension $d$---or to minimize $1$-Lipschitz convex functions to accuracy $\epsilon$ over the unit ball---our algorithms use $\mathcal O(\frac{d^2}{p}\ln \frac{1}{\epsilon})$ bits of memory, and make $\mathcal O((C\frac{d}{p}\ln \frac{1}{\epsilon})^p)$ oracle calls. The family is parametrized by $p\in[d]$ and provides an oracle-complexity/memory trade-off in the sub-polynomial regime $\ln\frac{1}{\epsilon}\gg\ln d$. While several works gave lower-bound trade-offs (impossibility results)---we explicit here their dependence with $\ln\frac{1}{\epsilon}$, showing that these also hold in any sub-polynomial regime---to the best of our knowledge this is the first class of algorithms that provides a positive trade-off between gradient descent and cutting-plane methods in any regime with $\epsilon\leq 1/\sqrt d$. The algorithms divide the $d$ variables into $p$ blocks and optimize over blocks sequentially, with approximate separation vectors constructed using a variant of Vaidya's method. In the regime $\epsilon \leq d^{-\Omega(d)}$, our algorithm with $p=d$ achieves the information-theoretic optimal memory usage and improves the oracle-complexity of gradient descent.

----

## [0] Nonparametric Boundary Geometry in Physics Informed Deep Learning

**Authors**: *Scott Alexander Cameron, Arnu Pretorius, Stephen J. Roberts*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13aef57cf532e88c476a10ff372e44e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13aef57cf532e88c476a10ff372e44e5-Abstract-Conference.html)

**Abstract**:

Engineering design problems frequently require solving systems ofpartial differential equations with boundary conditions specified onobject geometries in the form of a triangular mesh. These boundarygeometries are provided by a designer and are problem dependent.The efficiency of the design process greatly benefits from fast turnaroundtimes when repeatedly solving PDEs on various geometries. However,most current work that uses machine learning to speed up the solutionprocess relies heavily on a fixed parameterization of the geometry, whichcannot be changed after training. This severely limits the possibility ofreusing a trained model across a variety of design problems.In this work, we propose a novel neural operator architecture which acceptsboundary geometry, in the form of triangular meshes, as input and produces anapproximate solution to a given PDE as output. Once trained, the model can beused to rapidly estimate the PDE solution over a new geometry, without the need forretraining or representation of the geometry to a pre-specified parameterization.

----

## [0] Tracking Most Significant Shifts in Nonparametric Contextual Bandits

**Authors**: *Joe Suk, Samory Kpotufe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13b501c58ae3bfe9635a259f4414e943-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13b501c58ae3bfe9635a259f4414e943-Abstract-Conference.html)

**Abstract**:

We study nonparametric contextual bandits where Lipschitz mean reward functions may change over time.We first establish the minimax dynamic regret rate in this less understood setting in terms of number of changes $L$ and total-variation $V$, both capturing all changes in distribution over context space, and argue that state-of-the-art procedures are suboptimal in this setting.Next, we tend to the question of an _adaptivity_ for this setting, i.e. achieving the minimax rate without knowledge of $L$ or $V$. Quite importantly, we posit that the bandit  problem, viewed locally at a given context $X_t$, should not be affected by reward changes in other parts of context space $\cal X$. We therefore propose a notion of _change_, which we term _experienced significant shifts_, that better accounts for locality, and thus counts considerably less changes than $L$ and $V$. Furthermore, similar to recent work on non-stationary MAB (Suk & Kpotufe, 2022), _experienced significant shifts_ only count the most _significant_ changes in mean rewards, e.g., severe best-arm changes relevant to observed contexts.Our main result is to show that this more tolerant notion of change can in fact be adapted to.

----

## [0] Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss

**Authors**: *An Zhang, Leheng Sheng, Zhibo Cai, Xiang Wang, Tat-Seng Chua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13f1750b825659394a6499399e7637fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13f1750b825659394a6499399e7637fc-Abstract-Conference.html)

**Abstract**:

Contrastive Learning (CL) has achieved impressive performance in self-supervised learning tasks, showing superior generalization ability. Inspired by the success, adopting CL into collaborative filtering (CF) is prevailing in semi-supervised topK recommendations. The basic idea is to routinely conduct heuristic-based data augmentation and apply contrastive losses (e.g., InfoNCE) on the augmented views. Yet, some CF-tailored challenges make this adoption suboptimal, such as the issue of out-of-distribution, the risk of false negatives, and the nature of top-K evaluation. They necessitate the CL-based CF scheme to focus more on mining hard negatives and distinguishing false negatives from the vast unlabeled user-item interactions, for informative contrast signals. Worse still, there is limited understanding of contrastive loss in CF methods, especially w.r.t. its generalization ability. To bridge the gap, we delve into the reasons underpinning the success of contrastive loss in CF, and propose a principled Adversarial InfoNCE loss (AdvInfoNCE), which is a variant of InfoNCE, specially tailored for CF methods. AdvInfoNCE adaptively explores and assigns hardness to each negative instance in an adversarial fashion and further utilizes a fine-grained hardness-aware ranking criterion to empower the recommenderâ€™s generalization ability. Training CF models with AdvInfoNCE, we validate the effectiveness of AdvInfoNCE on both synthetic and real-world benchmark datasets, thus showing its generalization ability to mitigate out-of-distribution problems. Given the theoretical guarantees and empirical superiority of AdvInfoNCE over most contrastive loss functions, we advocate its adoption as a standard loss in recommender systems, particularly for the out-of-distribution tasks. Codes are available at https://github.com/LehengTHU/AdvInfoNCE.

----

## [0] The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-based Variable Importance

**Authors**: *Jon Donnelly, Srikar Katta, Cynthia Rudin, Edward P. Browne*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1403ab1a427050538ec59c7f570aec8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1403ab1a427050538ec59c7f570aec8b-Abstract-Conference.html)

**Abstract**:

Quantifying variable importance is essential for answering high-stakes questions in fields like genetics, public policy, and medicine. Current methods generally calculate variable importance for a given model trained on a given dataset. However, for a given dataset, there may be many models that explain the target outcome equally well; without accounting for all possible explanations, different researchers may arrive at many conflicting yet equally valid conclusions given the same data. Additionally, even when accounting for all possible explanations for a given dataset, these insights may not generalize because not all good explanations are stable across reasonable data perturbations. We propose a new variable importance framework that quantifies the importance of a variable across the set of all good models and is stable across the data distribution. Our framework is extremely flexible and can be integrated with most existing model classes and global variable importance metrics. We demonstrate through experiments that our framework recovers variable importance rankings for complex simulation setups where other methods fail. Further, we show that our framework accurately estimates the true importance of a variable for the underlying data distribution. We provide theoretical guarantees on the consistency and finite sample error rates for our estimator. Finally, we demonstrate its utility with a real-world case study exploring which genes are important for predicting HIV load in persons with HIV, highlighting an important gene that has not previously been studied in connection with HIV.

----

## [0] Model-Based Control with Sparse Neural Dynamics

**Authors**: *Ziang Liu, Genggeng Zhou, Jeff He, Tobia Marcucci, Fei-Fei Li, Jiajun Wu, Yunzhu Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/142cdba4b8d1e03f9ee131ac86bb0afc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/142cdba4b8d1e03f9ee131ac86bb0afc-Abstract-Conference.html)

**Abstract**:

Learning predictive models from observations using deep neural networks (DNNs) is a promising new approach to many real-world planning and control problems. However, common DNNs are too unstructured for effective planning, and current control methods typically rely on extensive sampling or local gradient descent. In this paper, we propose a new framework for integrated model learning and predictive control that is amenable to efficient optimization algorithms. Specifically, we start with a ReLU neural model of the system dynamics and, with minimal losses in prediction accuracy, we gradually sparsify it by removing redundant neurons. This discrete sparsification process is approximated as a continuous problem, enabling an end-to-end optimization of both the model architecture and the weight parameters. The sparsified model is subsequently used by a mixed-integer predictive controller, which represents the neuron activations as binary variables and employs efficient branch-and-bound algorithms. Our framework is applicable to a wide variety of DNNs, from simple multilayer perceptrons to complex graph neural dynamics. It can efficiently handle tasks involving complicated contact dynamics, such as object pushing, compositional object sorting, and manipulation of deformable objects. Numerical and hardware experiments show that, despite the aggressive sparsification, our framework can deliver better closed-loop performance than existing state-of-the-art methods.

----

## [0] AmadeusGPT: a natural language interface for interactive animal behavioral analysis

**Authors**: *Shaokai Ye, Jessy Lauer, Mu Zhou, Alexander Mathis, Mackenzie W. Mathis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1456560769bbc38e4f8c5055048ea712-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1456560769bbc38e4f8c5055048ea712-Abstract-Conference.html)

**Abstract**:

The process of quantifying and analyzing animal behavior involves translating the naturally occurring descriptive language of their actions into machine-readable code. Yet, codifying behavior analysis is often challenging without deep understanding of animal behavior and technical machine learning knowledge. To limit this gap, we introduce AmadeusGPT: a natural language interface that turns natural language descriptions of behaviors into machine-executable code. Large-language models (LLMs) such as GPT3.5 and GPT4 allow for interactive language-based queries that are potentially well suited for making interactive behavior analysis. However, the comprehension capability of these LLMs is limited by the context window size, which prevents it from remembering distant conversations. To overcome the context window limitation, we implement a novel dual-memory mechanism to allow communication between short-term and long-term memory using symbols as context pointers for retrieval and saving. Concretely, users directly use language-based definitions of behavior and our augmented GPT develops code based on the core AmadeusGPT API, which contains machine learning, computer vision, spatio-temporal reasoning, and visualization modules. Users then can interactively refine results, and seamlessly add new behavioral modules as needed. We used the MABe 2022 behavior challenge tasks to benchmark AmadeusGPT and show excellent performance. Note, an end-user would not need to write any code to achieve this. Thus, collectively AmadeusGPT presents a novel way to merge deep biological knowledge, large-language models, and core computer vision modules into a more naturally intelligent system. Code and demos can be found at: https://github.com/AdaptiveMotorControlLab/AmadeusGPT

----

## [0] Provably Efficient Algorithm for Nonstationary Low-Rank MDPs

**Authors**: *Yuan Cheng, Jing Yang, Yingbin Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/145c28cd4b1df9b426990fd68045f4f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/145c28cd4b1df9b426990fd68045f4f7-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) under changing environment models many real-world applications via nonstationary Markov Decision Processes (MDPs), and hence gains considerable interest. However, theoretical studies on nonstationary MDPs in the literature have mainly focused on tabular and linear (mixture) MDPs, which do not capture the nature of unknown representation in deep RL. In this paper, we make the first effort to investigate nonstationary RL under episodic low-rank MDPs, where both transition kernels and rewards may vary over time, and the low-rank model contains unknown representation in addition to the linear state embedding function. We first propose a parameter-dependent policy optimization algorithm called PORTAL,and further improve PORTAL to its parameter-free version of Ada-PORTAL, which is able to tune its hyper-parameters adaptively without any prior knowledge of nonstationarity. For both algorithms, we provide upper bounds on the average dynamic suboptimality gap, which show that as long as the nonstationarity is not significantly large, PORTAL and Ada-PORTAL are sample-efficient and can achieve arbitrarily small average dynamic suboptimality gap with polynomial sample complexity.

----

## [0] Time-uniform confidence bands for the CDF under nonstationarity

**Authors**: *Paul Mineiro, Steven R. Howard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/148bbc25b934211d80435b5cad5a7198-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/148bbc25b934211d80435b5cad5a7198-Abstract-Conference.html)

**Abstract**:

Estimation of a complete univariate distribution from a sequence of observations is a useful primitive for both manual and automated decision making. This problem has received extensive attention in the i.i.d. setting, but the arbitrary data dependent setting remains largely unaddressed. We present computationally felicitous time-uniform and value-uniform bounds on the CDF of the running averaged conditional distribution of a sequence of real-valued random variables. Consistent with known impossibility results, our CDF bounds are always valid but sometimes trivial when the instance is too hard, and we give an instance-dependent convergence guarantee.  The importance-weighted extension is appropriate for estimating complete counterfactual distributions of rewards given data from a randomized experiment, e.g., from an A/B test or a contextual bandit.

----

## [0] Risk-Averse Active Sensing for Timely Outcome Prediction under Cost Pressure

**Authors**: *Yuchao Qin, Mihaela van der Schaar, Changhee Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1498a03a04f9bcd3a7d44058fc5dc639-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1498a03a04f9bcd3a7d44058fc5dc639-Abstract-Conference.html)

**Abstract**:

Timely outcome prediction is essential in healthcare to enable early detection and intervention of adverse events. However, in longitudinal follow-ups to patients' health status, cost-efficient acquisition of patient covariates is usually necessary due to the significant expense involved in screening and lab tests. To balance the timely and accurate outcome predictions with acquisition costs, an effective active sensing strategy is crucial. In this paper, we propose a novel risk-averse active sensing approach RAS that addresses the composite decision problem of when to conduct the acquisition and which measurements to make. Our approach decomposes the policy into two sub-policies: acquisition scheduler and feature selector, respectively. Moreover, we introduce a novel risk-aversion training strategy to focus on the underrepresented subgroup of high-risk patients for whom timely and accurate prediction of disease progression is of greater value. Our method outperforms baseline active sensing approaches in experiments with both synthetic and real-world datasets, and we illustrate the significance of our policy decomposition and the necessity of a risk-averse sensing policy through case studies.

----

## [0] Single-Pass Pivot Algorithm for Correlation Clustering. Keep it simple!

**Authors**: *Konstantin Makarychev, Sayak Chakrabarty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/149ad6e32c08b73a3ecc3d11977fcc47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/149ad6e32c08b73a3ecc3d11977fcc47-Abstract-Conference.html)

**Abstract**:

We show that a simple single-pass semi-streaming variant of the Pivot algorithm for Correlation Clustering gives a (3+eps)-approximation using O(n/eps) words of memory. This is a slight improvement over the recent results of Cambus, Kuhn, Lindy, Pai, and Uitto, who gave a (3+eps)-approximation using O(n log n) words of memory, and Behnezhad, Charikar, Ma, and Tan, who gave a 5-approximation using O(n) words of memory. One of the main contributions of our paper is that the algorithm and its analysis are simple and easy to understand.

----

## [0] SPACE: Single-round Participant Amalgamation for Contribution Evaluation in Federated Learning

**Authors**: *Yi-Chung Chen, Hsi-Wen Chen, Shun-Gui Wang, Ming-Syan Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/14a812fa4b6bf244d055e37a7cd2f557-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/14a812fa4b6bf244d055e37a7cd2f557-Abstract-Conference.html)

**Abstract**:

The evaluation of participant contribution in federated learning (FL) has recently gained significant attention due to its applicability in various domains, such as incentive mechanisms, robustness enhancement, and client selection. Previous approaches have predominantly relied on the widely adopted Shapley value for participant evaluation. However, the computation of the Shapley value is expensive, despite using techniques like gradient-based model reconstruction and truncating unnecessary evaluations. Therefore, we present an efficient approach called Single-round Participants Amalgamation for Contribution Evaluation (SPACE). SPACE incorporates two novel components, namely Federated Knowledge Amalgamation and Prototype-based Model Evaluation to reduce the evaluation effort by eliminating the dependence on the size of the validation set and enabling participant evaluation within a single communication round. Experimental results demonstrate that SPACE outperforms state-of-the-art methods in terms of both running time and Pearsonâ€™s Correlation Coefficient (PCC). Furthermore, extensive experiments conducted on applications, client reweighting, and client selection highlight the effectiveness of SPACE. The code is available at https://github.com/culiver/SPACE.

----

## [0] SAME: Uncovering GNN Black Box with Structure-aware Shapley-based Multipiece Explanations

**Authors**: *Ziyuan Ye, Rihan Huang, Qilin Wu, Quanying Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/14cdc9013d80338bf81483a7736ea05c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/14cdc9013d80338bf81483a7736ea05c-Abstract-Conference.html)

**Abstract**:

Post-hoc explanation techniques on graph neural networks (GNNs) provide economical solutions for opening the black-box graph models without model retraining. Many GNN explanation variants have achieved state-of-the-art explaining results on a diverse set of benchmarks, while they rarely provide theoretical analysis for their inherent properties and explanatory capability. In this work, we propose $\underline{\text{S}}$tructure-$\underline{\text{A}}$ware Shapley-based $\underline{\text{M}}$ultipiece $\underline{\text{E}}$xplanation (SAME) method to address the structure-aware feature interactions challenges for GNNs explanation. Specifically, SAME leverages an expansion-based Monte Carlo tree search to explore the multi-grained structure-aware connected substructure. Afterward, the explanation results are encouraged to be informative of the graph properties by optimizing the combination of distinct single substructures. With the consideration of fair feature interactions in the process of investigating multiple connected important substructures, the explanation provided by SAME has the potential to be as explainable as the theoretically optimal explanation obtained by the Shapley value within polynomial time. Extensive experiments on real-world and synthetic benchmarks show that SAME improves the previous state-of-the-art fidelity performance by 12.9\% on BBBP, 7.01\% on MUTAG, 42.3\% on Graph-SST2, 38.9\% on Graph-SST5, 11.3\% on BA-2Motifs and 18.2\% on BA-Shapes under the same testing condition. Code is available at https://github.com/same2023neurips/same.

----

## [0] Federated Learning with Client Subsampling, Data Heterogeneity, and Unbounded Smoothness: A New Algorithm and Lower Bounds

**Authors**: *Michael Crawshaw, Yajie Bao, Mingrui Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/14ecbfb2216bab76195b60bfac7efb1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/14ecbfb2216bab76195b60bfac7efb1f-Abstract-Conference.html)

**Abstract**:

We study the problem of Federated Learning (FL) under client subsampling and data heterogeneity with an objective function that has potentially unbounded smoothness. This problem is motivated by empirical evidence that the class of relaxed smooth functions, where the Lipschitz constant of the gradient scales linearly with the gradient norm, closely resembles the loss functions of certain neural networks such as recurrent neural networks (RNNs) with possibly exploding gradient. We introduce EPISODE++, the first algorithm to solve this problem. It maintains historical statistics for each client to construct control variates and decide clipping behavior for sampled clients in the current round. We prove that EPISODE++ achieves linear speedup in the number of participating clients, reduced communication rounds, and resilience to data heterogeneity. Our upper bound proof relies on novel techniques of recursively bounding the client updates under unbounded smoothness and client subsampling, together with a refined high probability analysis. In addition, we prove a lower bound showing that the convergence rate of a special case of clipped minibatch SGD (without randomness in the stochastic gradient and with randomness in client subsampling) suffers from an explicit dependence on the maximum gradient norm of the objective in a sublevel set, which may be large. This effectively demonstrates that applying gradient clipping to minibatch SGD in our setting does not eliminate the problem of exploding gradients.  Our lower bound is based on new constructions of hard instances tailored to client subsampling and a novel analysis of the trajectory of the algorithm in the presence of clipping. Lastly, we provide an experimental evaluation of EPISODE++ when training RNNs on federated text classification tasks, demonstrating that EPISODE++ outperforms strong baselines in FL. The code is available at https://github.com/MingruiLiu-ML-Lab/episode_plusplus.

----

## [0] NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics

**Authors**: *Anwar Said, Roza G. Bayrak, Tyler Derr, Mudassir Shabbir, Daniel Moyer, Catie Chang, Xenofon D. Koutsoukos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/14f656f21d09a4114666f60a45aab1aa-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/14f656f21d09a4114666f60a45aab1aa-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning provides a valuable tool for analyzing high-dimensional functional neuroimaging data, and is proving effective in predicting various neurological conditions, psychiatric disorders, and cognitive patterns. In functional magnetic resonance imaging (MRI) research, interactions between brain regions are commonly modeled using graph-based representations. The potency of graph machine learning methods has been established across myriad domains, marking a transformative step in data interpretation and predictive modeling. Yet, despite their promise, the transposition of these techniques to the neuroimaging domain has been  challenging due to the expansive number of potential preprocessing pipelines and the large parameter search space for graph-based dataset construction. In this paper, we introduce NeuroGraph, a collection of graph-based neuroimaging datasets, and demonstrated its utility for predicting multiple categories of behavioral and cognitive traits. We delve deeply into the dataset generation search space by crafting 35 datasets that encompass static and dynamic brain connectivity, running in excess of 15 baseline methods for benchmarking. Additionally, we provide generic frameworks for learning on both static and dynamic graphs. Our extensive experiments lead to several key observations. Notably, using correlation vectors as node features, incorporating larger number of regions of interest, and employing sparser graphs lead to improved performance. To foster further advancements in graph-based data driven neuroimaging analysis, we offer a comprehensive open-source Python package that includes the benchmark datasets, baseline implementations, model training, and standard evaluation.

----

## [0] Quantifying the Cost of Learning in Queueing Systems

**Authors**: *Daniel Freund, Thodoris Lykouris, Wentao Weng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1502957929fc4257dd1b6daf7d869c2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1502957929fc4257dd1b6daf7d869c2f-Abstract-Conference.html)

**Abstract**:

Queueing systems are widely applicable stochastic models with use cases in communication networks, healthcare, service systems, etc. Although their optimal control has been extensively studied, most existing approaches assume perfect knowledge of the system parameters. Of course, this assumption rarely holds in practice where there is parameter uncertainty, thus motivating a recent line of work on bandit learning for queueing systems. This nascent stream of research focuses on the asymptotic performance of the proposed algorithms. In this paper, we argue that an asymptotic metric, which focuses on late-stage performance, is insufficient to capture the intrinsic statistical complexity of learning in queueing systems which typically occurs in the early stage. Instead, we propose the Cost of Learning in Queueing (CLQ), a new metric that quantifies the maximum increase in time-averaged queue length caused by parameter uncertainty.We characterize the CLQ of a single-queue multi-server system, and then extend these results to multi-queue multi-server systems and networks of queues. In establishing our results, we propose a unified analysis framework for CLQ that bridges Lyapunov and bandit analysis, provides guarantees for a wide range of algorithms, and could be of independent interest.

----

## [0] One-Line-of-Code Data Mollification Improves Optimization of Likelihood-based Generative Models

**Authors**: *Ba-Hien Tran, Giulio Franzese, Pietro Michiardi, Maurizio Filippone*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1516a7f7507d5550db5c7f29e995ec8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1516a7f7507d5550db5c7f29e995ec8c-Abstract-Conference.html)

**Abstract**:

Generative Models (GMs) have attracted considerable attention due to their tremendous success in various domains, such as computer vision where they are capable to generate impressive realistic-looking images. Likelihood-based GMs are attractive due to the possibility to generate new data by a single model evaluation. However, they typically achieve lower sample quality compared to state-of-the-art score-based Diffusion Models (DMs). This paper provides a significant step in the direction of addressing this limitation. The idea is to borrow one of the strengths of score-based DMs, which is the ability to perform accurate density estimation in low-density regions and to address manifold overfitting by means of data mollification. We propose a view of data mollification within likelihood-based GMs as a continuation method, whereby the optimization objective smoothly transitions from simple-to-optimize to the original target. Crucially, data mollification can be implemented by adding one line of code in the optimization loop, and we demonstrate that this provides a boost in generation quality of likelihood-based GMs, without computational overheads. We report results on real-world image data sets and UCI benchmarks with popular likelihood-based GMs, including variants of variational autoencoders and normalizing flows, showing large improvements in FID score and density estimation.

----

## [0] FLSL: Feature-level Self-supervised Learning

**Authors**: *Qing Su, Anton Netchaev, Hai Li, Shihao Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15212bd2265c4a3ab0dbc1b1982c1b69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15212bd2265c4a3ab0dbc1b1982c1b69-Abstract-Conference.html)

**Abstract**:

Current self-supervised learning (SSL) methods (e.g., SimCLR, DINO, VICReg, MOCOv3) target primarily on representations at instance level and do not generalize well to dense prediction tasks, such as object detection and segmentation. Towards aligning SSL with dense predictions, this paper demonstrates for the first time the underlying mean-shift clustering process of Vision Transformers (ViT), which aligns well with natural image semantics (e.g., a world of objects and stuffs). By employing transformer for joint embedding and clustering, we propose a bi-level feature clustering SSL method, coined Feature-Level Self-supervised Learning (FLSL). We present the formal definition of the FLSL problem and construct the objectives from the mean-shift and k-means perspectives. We show that FLSL promotes remarkable semantic cluster representations and learns an embedding scheme amenable to intra-view and inter-view feature clustering. Experiments show that FLSL yields significant improvements in dense prediction tasks, achieving 44.9 (+2.8)% AP and 46.5% AP in object detection, as well as 40.8 (+2.3)% AP and 42.1% AP in instance segmentation on MS-COCO, using Mask R-CNN with ViT-S/16 and ViT-S/8 as backbone, respectively. FLSL consistently outperforms existing SSL methods across additional benchmarks, including UAV object detection on UAVDT, and video instance segmentation on DAVIS 2017. We conclude by presenting visualization and various ablation studies to better understand the success of FLSL. The source code is available at https://github.com/ISL-CV/FLSL.

----

## [0] FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning

**Authors**: *Dipam Goswami, Yuyang Liu, Bartlomiej Twardowski, Joost van de Weijer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15294ba2dcfb4521274f7aa1c26f4dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15294ba2dcfb4521274f7aa1c26f4dd4-Abstract-Conference.html)

**Abstract**:

Exemplar-free class-incremental learning (CIL) poses several challenges since it prohibits the rehearsal of data from previous tasks and thus suffers from catastrophic forgetting. Recent approaches to incrementally learning the classifier by freezing the feature extractor after the first task have gained much attention. In this paper, we explore prototypical networks for CIL, which generate new class prototypes using the frozen feature extractor and classify the features based on the Euclidean distance to the prototypes. In an analysis of the feature distributions of classes, we show that classification based on Euclidean metrics is successful for jointly trained features. However, when learning from non-stationary data, we observe that the Euclidean metric is suboptimal and that feature distributions are heterogeneous. To address this challenge, we revisit the anisotropic Mahalanobis distance for CIL. In addition, we empirically show that modeling the feature covariance relations is better than previous attempts at sampling features from normal distributions and training a linear classifier. Unlike existing methods, our approach generalizes to both many- and few-shot CIL settings, as well as to domain-incremental settings. Interestingly, without updating the backbone network, our method obtains state-of-the-art results on several standard continual learning benchmarks. Code is available at https://github.com/dipamgoswami/FeCAM.

----

## [0] Learning non-Markovian Decision-Making from State-only Sequences

**Authors**: *Aoyang Qin, Feng Gao, Qing Li, Song-Chun Zhu, Sirui Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/154926e0b66e2b2a8c1120852f31a12d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/154926e0b66e2b2a8c1120852f31a12d-Abstract-Conference.html)

**Abstract**:

Conventional imitation learning assumes access to the actions of demonstrators, but these motor signals are often non-observable in naturalistic settings. Additionally, sequential decision-making behaviors in these settings can deviate from the assumptions of a standard Markov Decision Process (MDP). To address these challenges, we explore deep generative modeling of state-only sequences with non-Markov Decision Process (nMDP), where the policy is an energy-based prior in the latent space of the state transition generator. We develop maximum likelihood estimation to achieve model-based imitation, which involves short-run MCMC sampling from the prior and importance sampling for the posterior. The learned model enables $\textit{decision-making as inference}$: model-free policy execution is equivalent to prior sampling, model-based planning is posterior sampling initialized from the policy. We demonstrate the efficacy of the proposed method in a prototypical path planning task with non-Markovian constraints and show that the learned model exhibits strong performances in challenging domains from the MuJoCo suite.

----

## [0] Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts

**Authors**: *Zeyang Zhang, Xin Wang, Ziwei Zhang, Zhou Qin, Weigao Wen, Hui Xue, Haoyang Li, Wenwu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/154b90fcc9ba3dee96779c05c3108908-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/154b90fcc9ba3dee96779c05c3108908-Abstract-Conference.html)

**Abstract**:

Dynamic graph neural networks (DyGNNs) currently struggle with handling distribution shifts that are inherent in dynamic graphs.Existing work on DyGNNs with out-of-distribution settings only focuses on the time domain, failing to handle cases involving distribution shifts in the spectral domain. In this paper, we discover that there exist cases with distribution shifts unobservable in the time domain while observable in the spectral domain, and propose to study distribution shifts on dynamic graphs in the spectral domain for the first time.However, this investigation poses two key challenges: i) it is non-trivial to capture different graph patterns that are driven by various frequency components entangled in the spectral domain; and ii) it remains unclear how to handle distribution shifts with the discovered spectral patterns. To address these challenges, we propose Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts (SILD), which can handle distribution shifts on dynamic graphs by capturing and utilizing invariant and variant spectral patterns. Specifically, we first design a DyGNN with Fourier transform to obtain the ego-graph trajectory spectrums, allowing the mixed dynamic graph patterns to be transformed into separate frequency components. We then develop a disentangled spectrum mask to filter graph dynamics from various frequency components and discover the invariant and variant spectral patterns. Finally, we propose invariant spectral filtering, which encourages the model to rely on invariant patterns for generalization under distribution shifts. Experimental results on synthetic and real-world dynamic graph datasets demonstrate the superiority of our method for both node classification and link prediction tasks under distribution shifts.

----

## [0] Efficient Activation Function Optimization through Surrogate Modeling

**Authors**: *Garrett Bingham, Risto Miikkulainen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/154d63285d3ed7826e7f026c0b350d69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/154d63285d3ed7826e7f026c0b350d69-Abstract-Conference.html)

**Abstract**:

Carefully designed activation functions can improve the performance of neural networks in many machine learning tasks.  However, it is difficult for humans to construct optimal activation functions, and current activation function search algorithms are prohibitively expensive.  This paper aims to improve the state of the art through three steps: First, the benchmark datasets Act-Bench-CNN, Act-Bench-ResNet, and Act-Bench-ViT were created by training convolutional, residual, and vision transformer architectures from scratch with 2,913 systematically generated activation functions. Second, a characterization of the benchmark space was developed, leading to a new surrogate-based method for optimization. More specifically, the spectrum of the Fisher information matrix associated with the model's predictive distribution at initialization and the activation function's output distribution were found to be highly predictive of performance. Third, the surrogate was used to discover improved activation functions in several real-world tasks, with a surprising finding: a sigmoidal design that outperformed all other activation functions was discovered, challenging the status quo of always using rectifier nonlinearities in deep learning.  Each of these steps is a contribution in its own right; together they serve as a practical and theoretical foundation for further research on activation function optimization.

----

## [0] Data Market Design through Deep Learning

**Authors**: *Sai Srivatsa Ravindranath, Yanchen Jiang, David C. Parkes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1577ea3eaf8dacb99f64e4496c3ecddf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1577ea3eaf8dacb99f64e4496c3ecddf-Abstract-Conference.html)

**Abstract**:

The  data market design problem is a problem in economic theory to find a set of signaling schemes (statistical experiments) to maximize expected revenue to the information seller, where each experiment reveals some of the information known to a seller and has a corresponding price. Each buyer has their own decision to make in a world environment, and their subjective expected value for the information associated with a particular experiment comes from the improvement in this decision and depends on their prior and value for different outcomes. In a setting with multiple buyers, a buyer's expected value for an experiment may also depend on the information sold to others. We introduce the application of deep learning for the design of revenue-optimal data markets, looking to expand the frontiers of what can be understood and achieved. Relative to earlier work on deep learning for auction design, we must learn signaling schemes rather than allocation rules and handle  obedience constraints  — these arising from modeling the downstream actions of buyers — in addition to incentive constraints on bids.  Our experiments demonstrate that this new deep learning framework can almost precisely replicate all known solutions from theory, expand to more complex settings, and be used to establish the optimality of new designs for data markets and make conjectures in regard to the structure of optimal designs.

----

## [0] When Visual Prompt Tuning Meets Source-Free Domain Adaptive Semantic Segmentation

**Authors**: *Xinhong Ma, Yiming Wang, Hao Liu, Tianyu Guo, Yunhe Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/157c30da6a988e1cbef2095f7b9521db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/157c30da6a988e1cbef2095f7b9521db-Abstract-Conference.html)

**Abstract**:

Source-free domain adaptive semantic segmentation aims to adapt a pre-trained source model to the unlabeled target domain without accessing the private source data. Previous methods usually fine-tune the entire network, which suffers from expensive parameter tuning. To avoid this problem, we propose to utilize visual prompt tuning for parameter-efficient adaptation. However, the existing visual prompt tuning methods are unsuitable for source-free domain adaptive semantic segmentation due to the following two reasons: (1) Commonly used visual prompts like input tokens or pixel-level perturbations cannot reliably learn informative knowledge beneficial for semantic segmentation. (2) Visual prompts require sufficient labeled data to fill the gap between the pre-trained model and downstream tasks. To alleviate these problems, we propose a universal unsupervised visual prompt tuning  (Uni-UVPT) framework, which is applicable to various transformer-based backbones. Specifically, we first divide the source pre-trained backbone with frozen parameters into multiple stages, and propose a lightweight prompt adapter for progressively encoding informative knowledge into prompts and enhancing the generalization of target features between adjacent backbone stages. Cooperatively, a novel adaptive pseudo-label correction strategy with a multiscale consistency loss is designed to alleviate the negative effect of target samples with noisy pseudo labels and raise the capacity of visual prompts to spatial perturbations. Extensive experiments demonstrate that Uni-UVPT achieves state-of-the-art performance on GTA5 $\to$ Cityscapes and SYNTHIA $\to$ Cityscapes tasks and can serve as a universal and parameter-efficient framework for large-model unsupervised knowledge transfer. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/uni-uvpt and https://github.com/huawei-noah/noah-research/tree/master/uni-uvpt.

----

## [0] Benchmarking and Analyzing 3D-aware Image Synthesis with a Modularized Codebase

**Authors**: *Qiuyu Wang, Zifan Shi, Kecheng Zheng, Yinghao Xu, Sida Peng, Yujun Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1585da86b5a3c4fb15520a2b3682051f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1585da86b5a3c4fb15520a2b3682051f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Despite the rapid advance of 3D-aware image synthesis, existing studies usually adopt a mixture of techniques and tricks, leaving it unclear how each part contributes to the final performance in terms of generality. Following the most popular and effective paradigm in this field, which incorporates a neural radiance field (NeRF) into the generator of a generative adversarial network (GAN), we builda well-structured codebase through modularizing the generation process. Such a design allows researchers to develop and replace each module independently, and hence offers an opportunity to fairly compare various approaches and recognize their contributions from the module perspective. The reproduction of a range of cutting-edge algorithms demonstrates the availability of our modularized codebase. We also perform a variety of in-depth analyses, such as the comparison across different types of point feature, the necessity of the tailing upsampler in the generator, the reliance on the camera pose prior, etc., which deepen our understanding of existing methods and point out some further directions of the research work. Code and models will be made publicly available to facilitate the development and evaluation of this field.

----

## [0] RL-ViGen: A Reinforcement Learning Benchmark for Visual Generalization

**Authors**: *Zhecheng Yuan, Sizhe Yang, Pu Hua, Can Chang, Kaizhe Hu, Huazhe Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15c9f64ec172b046470d2a4d2b7669fc-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/15c9f64ec172b046470d2a4d2b7669fc-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Visual Reinforcement Learning (Visual RL), coupled with high-dimensional observations, has consistently confronted the long-standing challenge of out-of-distribution generalization. Despite the focus on algorithms aimed at resolving visual generalization problems, we argue that the devil is in the existing benchmarks as they are restricted to isolated tasks and generalization categories, undermining a comprehensive evaluation of agents' visual generalization capabilities. To bridge this gap, we introduce RL-ViGen: a novel Reinforcement Learning Benchmark for Visual Generalization, which contains diverse tasks and a wide spectrum of generalization types, thereby facilitating the derivation of more reliable conclusions. Furthermore, RL-ViGen incorporates the latest generalization visual RL algorithms into a unified framework, under which the experiment results indicate that no single existing algorithm has prevailed universally across tasks. Our aspiration is that Rl-ViGen will serve as a catalyst in this area, and lay a foundation for the future creation of universal visual generalization RL agents suitable for real-world scenarios.  Access to our code and implemented algorithms is provided at https://gemcollector.github.io/RL-ViGen/.

----

## [0] DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method

**Authors**: *Ahmed Khaled, Konstantin Mishchenko, Chi Jin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15ce36d35622f126f38e90167de1a350-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15ce36d35622f126f38e90167de1a350-Abstract-Conference.html)

**Abstract**:

This paper proposes a new easy-to-implement parameter-free gradient-based optimizer: DoWG (Distance over Weighted Gradients). We prove that DoWG is efficient---matching the convergence rate of optimally tuned gradient descent in convex optimization up to a logarithmic factor without tuning any parameters, and universal---automatically adapting to both smooth and nonsmooth problems. While popular algorithms following the AdaGrad framework compute a running average of the squared gradients, DoWG maintains a new distance-based weighted version of the running average, which is crucial to achieve the desired properties. To complement our theory, we also show empirically that DoWG trains at the edge of stability, and validate its effectiveness on practical machine learning tasks.

----

## [0] Multitask Learning with No Regret: from Improved Confidence Bounds to Active Learning

**Authors**: *Pier Giuseppe Sessa, Pierre Laforgue, Nicolò Cesa-Bianchi, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15d15045f93b44d933a260b249608d43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15d15045f93b44d933a260b249608d43-Abstract-Conference.html)

**Abstract**:

Multitask learning is a powerful framework that enables one to simultaneously learn multiple related tasks by sharing information between them. Quantifying uncertainty in the estimated tasks is of pivotal importance for many downstream applications, such as online or active learning. In this work, we provide novel confidence intervals for multitask regression in the challenging agnostic setting, i.e., when neither the similarity between tasks nor the tasks' features are available to the learner. The obtained intervals do not require i.i.d. data and can be directly applied to bound the regret in online learning. Through a refined analysis of the multitask information gain, we obtain new regret guarantees that, depending on a task similarity parameter, can significantly improve over treating tasks independently. We further propose a novel online learning algorithm that achieves such improved regret without knowing this parameter in advance, i.e., automatically adapting to task similarity. As a second key application of our results, we introduce a novel multitask active learning setup where several tasks must be simultaneously optimized, but only one of them can be queried for feedback by the learner at each round. For this problem, we design a no-regret algorithm that uses our confidence intervals to decide which task should be queried. Finally, we empirically validate our bounds and algorithms on synthetic and real-world (drug discovery) data.

----

## [0] Posterior Sampling with Delayed Feedback for Reinforcement Learning with Linear Function Approximation

**Authors**: *Nikki Lijing Kuang, Ming Yin, Mengdi Wang, Yu-Xiang Wang, Yian Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15d3d4a4bd808605e3a3c1ea0fd0eba4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15d3d4a4bd808605e3a3c1ea0fd0eba4-Abstract-Conference.html)

**Abstract**:

Recent studies in reinforcement learning (RL) have made significant progress by leveraging function approximation to alleviate the sample complexity hurdle for better performance. Despite the success, existing provably efficient algorithms typically rely on the accessibility of immediate feedback upon taking actions. The failure to account for the impact of delay in observations can significantly degrade the performance of real-world systems due to the regret blow-up. In this work, we tackle the challenge of delayed feedback in RL with linear function approximation by employing posterior sampling, which has been shown to empirically outperform the popular UCB algorithms in a wide range of regimes. We first introduce \textit{Delayed-PSVI}, an optimistic value-based algorithm that effectively explores the value function space via noise perturbation with posterior sampling. We provide the first analysis for posterior sampling algorithms with delayed feedback in RL and show our algorithm achieves $\widetilde{O}(\sqrt{d^3H^3 T} + d^2H^2 \mathbb{E}[\tau])$ worst-case regret in the presence of unknown stochastic delays. Here $\mathbb{E}[\tau]$ is the expected delay. To further improve its computational efficiency and to expand its applicability in high-dimensional RL problems, we incorporate a gradient-based approximate sampling scheme via Langevin dynamics for \textit{Delayed-LPSVI}, which maintains the same order-optimal regret guarantee with $\widetilde{O}(dHK)$ computational cost. Empirical evaluations are performed to demonstrate the statistical and computational efficacy of our algorithms.

----

## [0] Macro Placement by Wire-Mask-Guided Black-Box Optimization

**Authors**: *Yunqi Shi, Ke Xue, Song Lei, Chao Qian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15d6717f8bb33b3a74df26ce1eee0b9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15d6717f8bb33b3a74df26ce1eee0b9a-Abstract-Conference.html)

**Abstract**:

The development of very large-scale integration (VLSI) technology has posed new challenges for electronic design automation (EDA) techniques in chip floorplanning. During this process, macro placement is an important subproblem, which tries to determine the positions of all macros with the aim of minimizing half-perimeter wirelength (HPWL) and avoiding overlapping. Previous methods include packing-based, analytical and reinforcement learning methods. In this paper, we propose a new black-box optimization (BBO) framework (called WireMask-BBO) for macro placement, by using a wire-mask-guided greedy procedure for objective evaluation. Equipped with different BBO algorithms, WireMask-BBO empirically achieves significant improvements over previous methods, i.e., achieves significantly shorter HPWL by using much less time. Furthermore, it can fine-tune existing placements by treating them as initial solutions, which can bring up to 50% improvement in HPWL. WireMask-BBO has the potential to significantly improve the quality and efficiency of chip floorplanning, which makes it appealing to researchers and practitioners in EDA and will also promote the application of BBO. Our code is available at https://github.com/lamda-bbo/WireMask-BBO.

----

## [0] Reconciling Competing Sampling Strategies of Network Embedding

**Authors**: *Yuchen Yan, Baoyu Jing, Lihui Liu, Ruijie Wang, Jinning Li, Tarek F. Abdelzaher, Hanghang Tong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15dc2344ea9bdc01ffb8bb2d692e4018-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15dc2344ea9bdc01ffb8bb2d692e4018-Abstract-Conference.html)

**Abstract**:

Network embedding plays a significant role in a variety of applications. To capture the topology of the network, most of the existing network embedding algorithms follow a sampling training procedure, which maximizes the similarity (e.g., embedding vectors' dot product) between positively sampled node pairs and minimizes the similarity between negatively sampled node pairs in the embedding space. Typically, close node pairs function as positive samples while distant node pairs are usually considered as negative samples. However, under different or even competing sampling strategies, some methods champion sampling distant node pairs as positive samples to encapsulate longer distance information in link prediction, whereas others advocate adding close nodes into the negative sample set to boost the performance of node recommendation. In this paper, we seek to understand the intrinsic relationships between these competing strategies. To this end, we identify two properties (discrimination and monotonicity) that given any node pair proximity distribution, node embeddings should embrace.Moreover, we quantify the empirical error of the trained similarity score w.r.t. the sampling strategy, which leads to an important finding that the discrimination property and the monotonicity property for all node pairs can not be satisfied simultaneously in real-world applications. Guided by such analysis, a simple yet novel model (SENSEI) is proposed, which seamlessly fulfills the discrimination property and the partial monotonicity within the top-$K$ ranking list. Extensive experiments show that SENSEI outperforms the state-of-the-arts in plain network embedding.

----



[Go to the previous page](NIPS-2023-list200.md)

[Go to the next page](NIPS-2023-list202.md)

[Go to the catalog section](README.md)