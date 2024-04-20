## [200] DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection

**Authors**: *Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, Baoyuan Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e735e4b4f07de483cbe250130992726-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e735e4b4f07de483cbe250130992726-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

A critical yet frequently overlooked challenge in the field of deepfake detection is the lack of a standardized, unified, comprehensive benchmark. This issue leads to unfair performance comparisons and potentially misleading results. Specifically, there is a lack of uniformity in data processing pipelines, resulting in inconsistent data inputs for detection models. Additionally, there are noticeable differences in experimental settings, and evaluation strategies and metrics lack standardization. To fill this gap, we present the first comprehensive benchmark for deepfake detection, called \textit{DeepfakeBench}, which offers three key contributions: 1) a unified data management system to ensure consistent input across all detectors, 2) an integrated framework for state-of-the-art methods implementation, and 3) standardized evaluation metrics and protocols to promote transparency and reproducibility.  Featuring an extensible, modular-based codebase, \textit{DeepfakeBench} contains 15 state-of-the-art detection methods, 9 deepfake datasets, a series of deepfake detection evaluation protocols and analysis tools, as well as comprehensive evaluations.  Moreover, we provide new insights based on extensive analysis of these evaluations from various perspectives (\eg, data augmentations, backbones). We hope that our efforts could facilitate future research and foster innovation in this increasingly critical domain. All codes, evaluations, and analyses of our benchmark are publicly available at \url{https://github.com/SCLBD/DeepfakeBench}.

----

## [201] DreamWaltz: Make a Scene with Complex 3D Animatable Avatars

**Authors**: *Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, Lei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e769ec2c2cd99b6ad69c9d75113e386-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e769ec2c2cd99b6ad69c9d75113e386-Abstract-Conference.html)

**Abstract**:

We present DreamWaltz, a novel framework for generating and animating complex 3D avatars given text guidance and parametric human body prior. While recent methods have shown encouraging results for text-to-3D generation of common objects, creating high-quality and animatable 3D avatars remains challenging. To create high-quality 3D avatars, DreamWaltz proposes 3D-consistent occlusion-aware Score Distillation Sampling (SDS) to optimize implicit neural representations with canonical poses. It provides view-aligned supervision via 3D-aware skeleton conditioning which enables complex avatar generation without artifacts and multiple faces. For animation, our method learns an animatable 3D avatar representation from abundant image priors of diffusion model conditioned on various poses, which could animate complex non-rigged avatars given arbitrary poses without retraining. Extensive evaluations demonstrate that DreamWaltz is an effective and robust approach for creating 3D avatars that can take on complex shapes and appearances as well as novel poses for animation. The proposed framework further enables the creation of complex scenes with diverse compositions, including avatar-avatar, avatar-object and avatar-scene interactions. See https://dreamwaltz3d.github.io/ for more vivid 3D avatar and animation results.

----

## [202] Where2Explore: Few-shot Affordance Learning for Unseen Novel Categories of Articulated Objects

**Authors**: *Chuanruo Ning, Ruihai Wu, Haoran Lu, Kaichun Mo, Hao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0e7e2af2e5ba822c9ad35a37b31b5dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0e7e2af2e5ba822c9ad35a37b31b5dd4-Abstract-Conference.html)

**Abstract**:

Articulated object manipulation is a fundamental yet challenging task in robotics. Due to significant geometric and semantic variations across object categories, previous manipulation models struggle to generalize to novel categories. Few-shot learning is a promising solution for alleviating this issue by allowing robots to perform a few interactions with unseen objects. However, extant approaches often necessitate costly and inefficient test-time interactions with each unseen instance. Recognizing this limitation, we observe that despite their distinct shapes, different categories often share similar local geometries essential for manipulation, such as pullable handles and graspable edges - a factor typically underutilized in previous few-shot learning works. To harness this commonality, we introduce 'Where2Explore', an affordance learning framework that effectively explores novel categories with minimal interactions on a limited number of instances. Our framework explicitly estimates the geometric similarity across different categories, identifying local areas that differ from shapes in the training categories for efficient exploration while concurrently transferring affordance knowledge to similar parts of the objects. Extensive experiments in simulated and real-world environments demonstrate our framework's capacity for efficient few-shot exploration and generalization.

----

## [203] OpenProteinSet: Training data for structural biology at scale

**Authors**: *Gustaf Ahdritz, Nazim Bouatta, Sachin Kadyan, Lukas Jarosch, Daniel Berenberg, Ian Fisk, Andrew M. Watkins, Stephen Ra, Richard Bonneau, Mohammed AlQuraishi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0eb82171240776fe19da498bef3b1abe-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0eb82171240776fe19da498bef3b1abe-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Multiple sequence alignments (MSAs) of proteins encode rich biological information and have been workhorses in bioinformatic methods for tasks like protein design and protein structure prediction for decades. Recent breakthroughs like AlphaFold2 that use transformers to attend directly over large quantities of raw MSAs have reaffirmed their importance. Generation of MSAs is highly computationally intensive, however, and no datasets comparable to those used to train AlphaFold2 have been made available to the research community, hindering progress in machine learning for proteins. To remedy this problem, we introduce OpenProteinSet, an open-source corpus of more than 16 million MSAs, associated structural homologs from the Protein Data Bank, and AlphaFold2 protein structure predictions. We have previously demonstrated the utility of OpenProteinSet by successfully retraining AlphaFold2 on it. We expect OpenProteinSet to be broadly useful as training and validation data for 1) diverse tasks focused on protein structure, function, and design and 2) large-scale multimodal machine learning research.

----

## [204] Counting Distinct Elements in the Turnstile Model with Differential Privacy under Continual Observation

**Authors**: *Palak Jain, Iden Kalemaj, Sofya Raskhodnikova, Satchit Sivakumar, Adam D. Smith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ef1afa0daa888d695dcd5e9513bafa3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ef1afa0daa888d695dcd5e9513bafa3-Abstract-Conference.html)

**Abstract**:

Privacy is a central challenge for systems that learn from sensitive data sets, especially when a system's outputs must be continuously updated to reflect changing data. We consider the achievable error for differentially private continual release of a basic  statistic---the number of distinct items---in a stream where items may be both inserted and deleted (the turnstile model). With only insertions, existing algorithms have additive error just polylogarithmic in the length of the stream $T$. We uncover a much richer landscape in the turnstile model, even without considering memory restrictions. We show that every differentially private mechanism that handles insertions and deletions has worst-case additive error  at least $T^{1/4}$ even under  a relatively weak, event-level privacy definition. Then, we identify a parameter of the input stream, its maximum flippancy, that is low for natural data streams and for which we give tight parameterized error guarantees. Specifically, the maximum flippancy is the largest number of times that the contribution of a single item to the distinct elements count changes over the course of the stream. We present an item-level differentially private mechanism that, for all turnstile streams with  maximum flippancy  $w$, continually outputs the number of distinct elements with an $O(\sqrt{w} \cdot \mathsf{poly}\log T)$ additive error,  without requiring prior knowledge of $w$. We prove that this is the best achievable error bound  that depends only on $w$, for a large range of values of $w$. When $w$ is small, the error of our mechanism is similar to the polylogarithmic in $T$ error in the insertion-only setting, bypassing the hardness in the turnstile model.

----

## [205] Demystifying Softmax Gating Function in Gaussian Mixture of Experts

**Authors**: *Huy Nguyen, TrungTin Nguyen, Nhat Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ef6ffcb85a2d238fc4761860c31ded4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ef6ffcb85a2d238fc4761860c31ded4-Abstract-Conference.html)

**Abstract**:

Understanding the parameter estimation of softmax gating Gaussian mixture of experts has remained a long-standing open problem in the literature. It is mainly due to three fundamental theoretical challenges associated with the softmax gating function: (i) the identifiability only up to the translation of parameters; (ii) the intrinsic interaction via partial differential equations between the softmax gating and the expert functions in the Gaussian density; (iii) the complex dependence between the numerator and denominator of the conditional density of softmax gating Gaussian mixture of experts. We resolve these challenges by proposing novel Voronoi loss functions among parameters and establishing the convergence rates of maximum likelihood estimator (MLE) for solving parameter estimation in these models. When the true number of experts is unknown and over-specified, our findings show a connection between the convergence rate of the MLE and a solvability problem of a system of polynomial equations.

----

## [206] Hybrid Policy Optimization from Imperfect Demonstrations

**Authors**: *Hanlin Yang, Chao Yu, Peng Sun, Siji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f0a30c7b46be23a83317c5cb721fc43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f0a30c7b46be23a83317c5cb721fc43-Abstract-Conference.html)

**Abstract**:

Exploration is one of the main challenges in Reinforcement Learning (RL), especially in environments with sparse rewards. Learning from Demonstrations (LfD) is a promising approach to solving this problem by leveraging expert demonstrations. However, expert demonstrations of high quality  are usually costly or even impossible to collect in real-world applications. In this work, we propose a novel RL algorithm called HYbrid Policy Optimization (HYPO), which uses a small number of imperfect demonstrations to accelerate an agent's online learning process. The key idea is to train an offline guider policy using imitation learning in order to instruct an online agent policy to explore efficiently. Through mutual update of the guider policy and the agent policy, the agent can leverage suboptimal demonstrations for efficient exploration while avoiding the conservative policy caused by imperfect demonstrations. Empirical results show that HYPO significantly outperforms several baselines in various challenging tasks, such as MuJoCo with sparse rewards, Google Research Football, and the AirSim drone simulation.

----

## [207] What is Flagged in Uncertainty Quantification? Latent Density Models for Uncertainty Categorization

**Authors**: *Hao Sun, Boris van Breugel, Jonathan Crabbé, Nabeel Seedat, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f0c4f3d83c58df58380af3b0729354c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f0c4f3d83c58df58380af3b0729354c-Abstract-Conference.html)

**Abstract**:

Uncertainty quantification (UQ) is essential for creating trustworthy machine learning models. Recent years have seen a steep rise in UQ methods that can flag suspicious examples, however, it is often unclear what exactly these methods identify. In this work, we propose a framework for categorizing uncertain examples flagged by UQ methods. We introduce the confusion density matrix---a kernel-based approximation of the misclassification density---and use this to categorize suspicious examples identified by a given uncertainty method into three classes: out-of-distribution (OOD) examples, boundary (Bnd) examples, and examples in regions of high in-distribution misclassification (IDM). Through extensive experiments, we show that our framework provides a new and distinct perspective for assessing differences between uncertainty quantification methods, thereby forming a valuable assessment benchmark.

----

## [208] Datasets and Benchmarks for Nanophotonic Structure and Parametric Design Simulations

**Authors**: *Jungtaek Kim, Mingxuan Li, Oliver Hinder, Paul W. Leu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f12c9975ff4f2e44a5a26ef01b0b249-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f12c9975ff4f2e44a5a26ef01b0b249-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Nanophotonic structures have versatile applications including solar cells, anti-reflective coatings, electromagnetic interference shielding, optical filters, and light emitting diodes. To design and understand these nanophotonic structures, electrodynamic simulations are essential. These simulations enable us to model electromagnetic fields over time and calculate optical properties. In this work, we introduce frameworks and benchmarks to evaluate nanophotonic structures in the context of parametric structure design problems. The benchmarks are instrumental in assessing the performance of optimization algorithms and identifying an optimal structure based on target optical properties. Moreover, we explore the impact of varying grid sizes in electrodynamic simulations, shedding light on how evaluation fidelity can be strategically leveraged in enhancing structure designs.

----

## [209] Efficient Data Subset Selection to Generalize Training Across Models: Transductive and Inductive Networks

**Authors**: *Eeshaan Jain, Tushar Nandy, Gaurav Aggarwal, Ashish Tendulkar, Rishabh K. Iyer, Abir De*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f25eb6e9dc26c933a5d7516abf1eb8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f25eb6e9dc26c933a5d7516abf1eb8c-Abstract-Conference.html)

**Abstract**:

Existing subset selection methods for efficient learning predominantly employ discrete combinatorial and model-specific approaches, which lack generalizability--- for each new model, the algorithm has to be executed from the beginning. Therefore, for an unseen architecture, one cannot use the subset chosen for a different model. In this work, we propose $\texttt{SubSelNet}$, a non-adaptive subset selection framework, which tackles these problems. Here, we first introduce an attention-based neural gadget that leverages the graph structure of architectures and acts as a surrogate to trained deep neural networks for quick model prediction. Then, we use these predictions to build subset samplers. This naturally provides us two variants of $\texttt{SubSelNet}$. The first variant is transductive (called Transductive-$\texttt{SubSelNet}$), which computes the subset separately for each model by solving a small optimization problem. Such an optimization is still super fast, thanks to the replacement of explicit model training by the model approximator. The second variant is inductive (called Inductive-$\texttt{SubSelNet}$), which computes the subset using a trained subset selector, without any optimization.  Our experiments show that our model outperforms several methods across several real datasets.

----

## [210] NIS3D: A Completely Annotated Benchmark for Dense 3D Nuclei Image Segmentation

**Authors**: *Wei Zheng, Cheng Peng, Zeyuan Hou, Boyu Lyu, Mengfan Wang, Xuelong Mi, Shuoxuan Qiao, Yinan Wan, Guoqiang Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f2cd3d09a132757555b602e2dd43784-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f2cd3d09a132757555b602e2dd43784-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

3D segmentation of nuclei images is a fundamental task for many biological studies. Despite the rapid advances of large-volume 3D imaging acquisition methods and the emergence of sophisticated algorithms to segment the nuclei in recent years, a benchmark with all cells completely annotated is still missing, making it hard to accurately assess and further improve the performance of the algorithms. The existing nuclei segmentation benchmarks either worked on 2D only or annotated a small number of 3D cells, perhaps due to the high cost of 3D annotation for large-scale data. To fulfill the critical need, we constructed NIS3D, a 3D, high cell density, large-volume, and completely annotated Nuclei Image Segmentation benchmark, assisted by our newly designed semi-automatic annotation software. NIS3D provides more than 22,000 cells across multiple most-used species in this area. Each cell is labeled by three independent annotators, so we can measure the variability of each annotation. A confidence score is computed for each cell, allowing more nuanced testing and performance comparison. A comprehensive review on the methods of segmenting 3D dense nuclei was conducted. The benchmark was used to evaluate the performance of several selected state-of-the-art segmentation algorithms. The best of current methods is still far away from human-level accuracy, corroborating the necessity of generating such a benchmark. The testing results also demonstrated the strength and weakness of each method and pointed out the directions of further methodological development. The dataset can be downloaded here: https://github.com/yu-lab-vt/NIS3D.

----

## [211] HiBug: On Human-Interpretable Model Debug

**Authors**: *Muxi Chen, Yu Li, Qiang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f53ecc0d36a5d5d3d3e94d42c4b23ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f53ecc0d36a5d5d3d3e94d42c4b23ca-Abstract-Conference.html)

**Abstract**:

Machine learning models can frequently produce systematic errors on critical subsets (or slices) of data that share common attributes. Discovering and explaining such model bugs is crucial for reliable model deployment. However, existing bug discovery and interpretation methods usually involve heavy human intervention and annotation, which can be cumbersome and have low bug coverage.In this paper, we propose HiBug, an automated framework for interpretable model debugging. Our approach utilizes large pre-trained models, such as chatGPT, to suggest human-understandable attributes that are related to the targeted computer vision tasks. By leveraging pre-trained vision-language models, we can efficiently identify common visual attributes of underperforming data slices using human-understandable terms. This enables us to uncover rare cases in the training data, identify spurious correlations in the model, and use the interpretable debug results to select or generate new training data for model improvement. Experimental results demonstrate the efficacy of the HiBug framework.

----

## [212] A Theoretical Analysis of the Test Error of Finite-Rank Kernel Ridge Regression

**Authors**: *Tin Sum Cheng, Aurélien Lucchi, Anastasis Kratsios, Ivan Dokmanic, David Belius*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f580c1ace3b857a390575ca42de7938-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f580c1ace3b857a390575ca42de7938-Abstract-Conference.html)

**Abstract**:

Existing statistical learning guarantees for general kernel regressors often yield loose bounds when used with finite-rank kernels. Yet, finite-rank kernels naturally appear in a number of machine learning problems, e.g. when fine-tuning a pre-trained deep neural network's last layer to adapt it to a novel task when performing transfer learning.  We address this gap for finite-rank kernel ridge regression (KRR) by deriving sharp non-asymptotic upper and lower bounds for the KRR test error of any finite-rank KRR. Our bounds are tighter than previously derived bounds on finite-rank KRR and, unlike comparable results, they also remain valid for any regularization parameters.

----

## [213] Learning Invariant Representations with a Nonparametric Nadaraya-Watson Head

**Authors**: *Alan Wang, Minh Nguyen, Mert R. Sabuncu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f6931a9e339a012a9909306d7c758b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f6931a9e339a012a9909306d7c758b4-Abstract-Conference.html)

**Abstract**:

Machine learning models will often fail when deployed in an environment with a data distribution that is different than the training distribution. When multiple environments are available during training, many methods exist that learn representations which are invariant across the different distributions, with the hope that these representations will be transportable to unseen domains. In this work, we present a nonparametric strategy for learning invariant representations based on the recently-proposed Nadaraya-Watson (NW) head. The NW head makes a prediction by comparing the learned representations of the query to the elements of a support set that consists of labeled data. We demonstrate that by manipulating the support set, one can encode different causal assumptions. In particular, restricting the support set to a single environment encourages the model to learn invariant features that do not depend on the environment. We present a causally-motivated setup for our modeling and training strategy and validate on three challenging real-world domain generalization tasks in computer vision.

----

## [214] Conformalized matrix completion

**Authors**: *Yu Gui, Rina Barber, Cong Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0f7e4bb7a35dd4cb426203c91a4bfa10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0f7e4bb7a35dd4cb426203c91a4bfa10-Abstract-Conference.html)

**Abstract**:

Matrix completion aims to estimate missing entries in a data matrix, using the assumption of a low-complexity structure (e.g., low-rankness) so that imputation is possible. While many effective estimation algorithms exist in the literature, uncertainty quantification for this problem has proved to be challenging, and existing methods are extremely sensitive to model misspecification. In this work, we propose a distribution-free method for predictive inference in the matrix completion problem. Our method adapts the framework of conformal prediction, which provides prediction intervals with guaranteed distribution-free validity in the setting of regression, to the problem of matrix completion. Our resulting method, conformalized matrix completion (cmc), offers provable predictive coverage regardless of the accuracy of the low-rank model. Empirical results on simulated and real data demonstrate that cmc is robust to model misspecification while matching the performance of existing model-based methods when the model is correct.

----

## [215] Mixture Weight Estimation and Model Prediction in Multi-source Multi-target Domain Adaptation

**Authors**: *Yuyang Deng, Ilja Kuzborskij, Mehrdad Mahdavi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0fa81c3f0d57f95b8776de3a248ef0ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0fa81c3f0d57f95b8776de3a248ef0ed-Abstract-Conference.html)

**Abstract**:

We consider a problem of learning a model from multiple sources with the goal to performwell on a new target distribution.  Such problem arises inlearning with data collected from multiple sources (e.g. crowdsourcing) orlearning in distributed systems, where the data can be highly heterogeneous. Thegoal of learner is to mix these data sources in a target-distribution aware way andsimultaneously minimize the empirical risk on the mixed source.  The literature has made some tangible advancements in establishingtheory of learning on mixture domain.  However, there are still two unsolved problems. Firstly, how to estimate the optimal mixture of sources, given a target domain; Secondly, when there are numerous target domains, we have to solve empirical risk minimization for each target on possibly unique mixed source data , which is computationally expensive. In this paper we address both problems efficiently and with guarantees.We cast the first problem, mixture weight estimation as convex-nonconcave compositional minimax, and propose an efficient stochasticalgorithm with provable stationarity guarantees.Next, for the second problem, we identify that for certain regime,solving ERM for each target domain individually can be avoided, and instead parameters for a target optimalmodel can be viewed as a non-linear function ona space of the mixture coefficients.To this end, we show that in offline setting, a GD-trained overparameterized neural network can provably learn such function.Finally, we also consider an online setting and propose an label efficient online algorithm, which predicts parameters for new models given arbitrary sequence of mixing coefficients, while enjoying optimal regret.

----

## [216] CELLE-2: Translating Proteins to Pictures and Back with a Bidirectional Text-to-Image Transformer

**Authors**: *Emaad Khwaja, Yun Song, Aaron Agarunov, Bo Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0fb7c02d420c993385c7de44c2b5bf01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0fb7c02d420c993385c7de44c2b5bf01-Abstract-Conference.html)

**Abstract**:

We present CELL-E 2, a novel bidirectional transformer that can generate images depicting protein subcellular localization from the amino acid sequences (and vice versa). Protein localization is a challenging problem that requires integrating sequence and image information, which most existing methods ignore. CELL-E 2 extends the work of CELL-E, not only capturing the spatial complexity of protein localization and produce probability estimates of localization atop a nucleus image, but also being able to generate sequences from images, enabling de novo protein design. We train and finetune CELL-E 2 on two large-scale datasets of human proteins. We also demonstrate how to use CELL-E 2 to create hundreds of novel nuclear localization signals (NLS). Results and interactive demos are featured at https://bohuanglab.github.io/CELL-E_2/.

----

## [217] HeadSculpt: Crafting 3D Head Avatars with Text

**Authors**: *Xiao Han, Yukang Cao, Kai Han, Xiatian Zhu, Jiankang Deng, Yi-Zhe Song, Tao Xiang, Kwan-Yee K. Wong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0fb98d483fa580e0354bcdd3a003a3f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0fb98d483fa580e0354bcdd3a003a3f3-Abstract-Conference.html)

**Abstract**:

Recently, text-guided 3D generative methods have made remarkable advancements in producing high-quality textures and geometry, capitalizing on the proliferation of large vision-language and image diffusion models. However, existing methods still struggle to create high-fidelity 3D head avatars in two aspects: (1) They rely mostly on a pre-trained text-to-image diffusion model whilst missing the necessary 3D awareness and head priors. This makes them prone to inconsistency and geometric distortions in the generated avatars. (2) They fall short in fine-grained editing. This is primarily due to the inherited limitations from the pre-trained 2D image diffusion models, which become more pronounced when it comes to 3D head avatars. In this work, we address these challenges by introducing a versatile coarse-to-fine pipeline dubbed HeadSculpt for crafting (i.e., generating and editing) 3D head avatars from textual prompts. Specifically, we first equip the diffusion model with 3D awareness by leveraging landmark-based control and a learned textual embedding representing the back view appearance of heads, enabling 3D-consistent head avatar generations. We further propose a novel identity-aware editing score distillation strategy to optimize a textured mesh with a high-resolution differentiable rendering technique. This enables identity preservation while following the editing instruction.We showcase HeadSculpt's superior fidelity and editing capabilities through comprehensive experiments and comparisons with existing methods.

----

## [218] CBD: A Certified Backdoor Detector Based on Local Dominant Probability

**Authors**: *Zhen Xiang, Zidi Xiong, Bo Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0fbf046448d7eea18b982001320b9a10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0fbf046448d7eea18b982001320b9a10-Abstract-Conference.html)

**Abstract**:

Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.

----

## [219] SheetCopilot: Bringing Software Productivity to the Next Level through Large Language Models

**Authors**: *Hongxin Li, Jingran Su, Yuntao Chen, Qing Li, Zhaoxiang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ff30c4bf31db0119a6219e0d250e037-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ff30c4bf31db0119a6219e0d250e037-Abstract-Conference.html)

**Abstract**:

Computer end users have spent billions of hours completing daily tasks like tabular data processing and project timeline scheduling. Most of these tasks are repetitive and error-prone, yet most end users lack the skill to automate these burdensome works. With the advent of large language models (LLMs), directing software with natural language user requests become a reachable goal. In this work, we propose a SheetCopilot agent that takes natural language task and control spreadsheet to fulfill the requirements. We propose a set of atomic actions as an abstraction of spreadsheet software functionalities. We further design a state machine-based task planning framework for LLMs to robustly interact with spreadsheets. We curate a representative dataset containing 221 spreadsheet control tasks and establish a fully automated evaluation pipeline for rigorously benchmarking the ability of LLMs in software control tasks. Our SheetCopilot correctly completes 44.3\% of tasks for a single generation, outperforming the strong code generation baseline by a wide margin. Our project page: https://sheetcopilot.github.io/.

----

## [220] Beyond Uniform Sampling: Offline Reinforcement Learning with Imbalanced Datasets

**Authors**: *Zhang-Wei Hong, Aviral Kumar, Sathwik Karnik, Abhishek Bhandwaldar, Akash Srivastava, Joni Pajarinen, Romain Laroche, Abhishek Gupta, Pulkit Agrawal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ff3502bb29570b219967278db150a50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ff3502bb29570b219967278db150a50-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) enables learning a decision-making policy without interaction with the environment. This makes it particularly beneficial in situations where such interactions are costly. However, a known challenge for offline RL algorithms is the distributional mismatch between the state-action distributions of the learned policy and the dataset, which can significantly impact performance. State-of-the-art algorithms address it by constraining the policy to align with the state-action pairs in the dataset. However, this strategy struggles on datasets that predominantly consist of trajectories collected by low-performing policies and only a few trajectories from high-performing ones. Indeed, the constraint to align with the data leads the policy to imitate low-performing behaviors predominating the dataset. Our key insight to address this issue is to constrain the policy to the policy that collected the good parts of the dataset rather than all data. To this end, we optimize the importance sampling weights to emulate sampling data from a data distribution generated by a nearly optimal policy. Our method exhibits considerable performance gains (up to five times better) over the existing approaches in state-of-the-art offline RL algorithms over 72 imbalanced datasets with varying types of imbalance.

----

## [221] Variational Weighting for Kernel Density Ratios

**Authors**: *Sangwoong Yoon, Frank C. Park, Gunsu S. Yun, Iljung Kim, Yung-Kyun Noh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ff54b4ec4f70b3ae12c8621ca8a49f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ff54b4ec4f70b3ae12c8621ca8a49f4-Abstract-Conference.html)

**Abstract**:

Kernel density estimation (KDE) is integral to a range of generative and discriminative tasks in machine learning. Drawing upon tools from the multidimensional calculus of variations, we derive an optimal weight function that reduces bias in standard kernel density estimates for density ratios, leading to improved estimates of prediction posteriors and information-theoretic measures. In the process, we shed light on some fundamental aspects of density estimation, particularly from the perspective of algorithms that employ KDEs as their main building blocks.

----

## [222] Adversarial Examples Exist in Two-Layer ReLU Networks for Low Dimensional Linear Subspaces

**Authors**: *Odelia Melamed, Gilad Yehudai, Gal Vardi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/0ffd11b5bce666816802b86c77b54cf7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/0ffd11b5bce666816802b86c77b54cf7-Abstract-Conference.html)

**Abstract**:

Despite a great deal of research, it is still not well-understood why trained neural networks are highly vulnerable to adversarial examples.In this work we focus on two-layer neural networks trained using data which lie on a low dimensional linear subspace.We show that standard gradient methods lead to non-robust neural networks, namely, networks which have large gradients in directions orthogonal to the data subspace, and are susceptible to small adversarial $L_2$-perturbations in these directions.Moreover, we show that decreasing the initialization scale of the training algorithm, or adding $L_2$ regularization, can make the trained network more robust to adversarial perturbations orthogonal to the data.

----

## [223] Complexity of Derivative-Free Policy Optimization for Structured H∞ Control

**Authors**: *Xingang Guo, Darioush Keivan, Geir E. Dullerud, Peter J. Seiler, Bin Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1052b823a161aa2c808dd51c0f58dc37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1052b823a161aa2c808dd51c0f58dc37-Abstract-Conference.html)

**Abstract**:

The applications of direct policy search in reinforcement learning and continuous control have received increasing attention.In this work, we present novel theoretical results on the complexity of derivative-free policy optimization on an important class of robust control tasks, namely the structured $H_\infty$ synthesis with static output feedback. Optimal $H_\infty$ synthesis under structural constraints leads to a constrained nonconvex nonsmooth problem and is typicallyaddressed using subgradient-based policy search techniques that are built upon the concept of Goldstein subdifferential or other notions of enlarged subdifferential.  In this paper, we study the complexity of finding $(\delta,\epsilon)$-stationary points for such nonsmooth robust control design tasks using policy optimization methods which can only access the zeroth-order oracle (i.e. the $H_\infty$ norm of the closed-loop system). First, we study the exact oracle setting and identify the coerciveness of the cost function to prove high-probability feasibility/complexity bounds for derivative-free policy optimization on this problem. Next, we derive a sample complexity result for the multi-input multi-output (MIMO)  $H_\infty$-norm estimation. We combine this with our analysis to obtain the first sample complexity of model-free, trajectory-based, zeroth-order policy optimization on finding $(\delta,\epsilon)$-stationary points for structured $H_\infty$ control. Numerical results are also provided to demonstrate our theory.

----

## [224] Meet in the Middle: A New Pre-training Paradigm

**Authors**: *Anh Nguyen, Nikos Karampatziakis, Weizhu Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/105fdc31cc9eb927cc5a0110f4031287-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/105fdc31cc9eb927cc5a0110f4031287-Abstract-Conference.html)

**Abstract**:

Most language models (LMs) are trained and applied in an autoregressive left-to-right fashion, predicting the next token from the preceding ones. However, this ignores that the full sequence is available during training. In this paper, we introduce ``Meet in the Middle'' (MIM) a new pre-training paradigm that improves data efficiency by training in two directions, left-to-right and right-to-left, and encouraging the respective modelsto agree on their token distribution for each position. While the primary outcome is an improved left-to-right LM,we also obtain secondary benefits in the infilling task. There, we leverage the two pre-trained directions to propose an infilling procedure that builds the completion simultaneously from both sides. We conduct extensive experiments on both programming and natural languages and show that MIM significantly surpasses existing pre-training paradigms, in both left-to-right generation as well as infilling.Code and models available at https://github.com/microsoft/Meet-in-the-Middle

----

## [225] Score-based Source Separation with Applications to Digital Communication Signals

**Authors**: *Tejas Jayashankar, Gary C. F. Lee, Alejandro Lancho, Amir Weiss, Yury Polyanskiy, Gregory W. Wornell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/106b2434b8d496c6aed9235d478678af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/106b2434b8d496c6aed9235d478678af-Abstract-Conference.html)

**Abstract**:

We propose a new method for separating superimposed sources using diffusion-based generative models.  Our method relies only on separately trained statistical priors of independent sources to establish a new objective function guided by $\textit{maximum a posteriori}$ estimation with an $\textit{$\alpha$-posterior}$, across multiple levels of Gaussian smoothing.  Motivated by applications in radio-frequency (RF) systems, we are interested in sources with underlying discrete nature and the recovery of encoded bits from a signal of interest, as measured by the bit error rate (BER). Experimental results with RF mixtures demonstrate that our method results in a BER reduction of 95\%  over classical and existing learning-based methods.  Our analysis demonstrates that our proposed method yields solutions that asymptotically approach the modes of an underlying discrete distribution. Furthermore, our method can be viewed as a multi-source extension to the recently proposed score distillation sampling scheme, shedding additional light on its use beyond conditional sampling. The project webpage is available at https://alpha-rgs.github.io.

----

## [226] Fair Streaming Principal Component Analysis: Statistical and Algorithmic Viewpoint

**Authors**: *Junghyun Lee, Hanseul Cho, Se-Young Yun, Chulhee Yun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1074541383db5ef12d6ac66d2f8e8d34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1074541383db5ef12d6ac66d2f8e8d34-Abstract-Conference.html)

**Abstract**:

Fair Principal Component Analysis (PCA) is a problem setting where we aim to perform PCA while making the resulting representation fair in that the projected distributions, conditional on the sensitive attributes, match one another. However, existing approaches to fair PCA have two main problems: theoretically, there has been no statistical foundation of fair PCA in terms of learnability; practically, limited memory prevents us from using existing approaches, as they explicitly rely on full access to the entire data. On the theoretical side, we rigorously formulate fair PCA using a new notion called probably approximately fair and optimal (PAFO) learnability. On the practical side, motivated by recent advances in streaming algorithms for addressing memory limitation, we propose a new setting called fair streaming PCA along with a memory-efficient algorithm, fair noisy power method (FNPM). We then provide its statistical guarantee in terms of PAFO-learnability, which is the first of its kind in fair PCA literature. We verify our algorithm in the CelebA dataset without any pre-processing; while the existing approaches are inapplicable due to memory limitations, by turning it into a streaming setting, we show that our algorithm performs fair PCA efficiently and effectively.

----

## [227] DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models

**Authors**: *Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, Sibei Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/108030643e640ac050e0ed5e6aace48f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/108030643e640ac050e0ed5e6aace48f-Abstract-Conference.html)

**Abstract**:

A long-standing goal of AI systems is to perform complex multimodal reasoning like humans. Recently, large language models (LLMs) have made remarkable strides in such multi-step reasoning on the language modality solely by leveraging the chain of thought (CoT) to mimic human thinking. However, the transfer of these advancements to multimodal contexts introduces heightened challenges, including but not limited to the impractical need for labor-intensive annotation and the limitations in terms of flexibility, generalizability, and explainability. To evoke CoT reasoning in multimodality, this work first conducts an in-depth analysis of these challenges posed by multimodality and presents two key insights: “keeping critical thinking” and “letting everyone do their jobs” in multimodal CoT reasoning. Furthermore, this study proposes a novel DDCoT prompting that maintains a critical attitude through negative-space prompting and incorporates multimodality into reasoning by first dividing the reasoning responsibility of LLMs into reasoning and recognition and then integrating the visual recognition capability of visual models into the joint reasoning process. The rationales generated by DDCoT not only improve the reasoning abilities of both large and small language models in zero-shot prompting and fine-tuning learning, significantly outperforming state-of-the-art methods but also exhibit impressive generalizability and explainability.

----

## [228] Adversarially Robust Learning with Uncertain Perturbation Sets

**Authors**: *Tosca Lechner, Vinayak Pathak, Ruth Urner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1097a0aeaf00cacfa8f6aced24f3a8bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1097a0aeaf00cacfa8f6aced24f3a8bd-Abstract-Conference.html)

**Abstract**:

In many real-world settings exact perturbation sets to be used by an adversary are not plausibly available to a learner. While prior literature has studied both scenarios with completely known and completely unknown perturbation sets, we propose an in-between setting of learning with respect to a class of perturbation sets. We show that in this setting we can improve on previous results with completely unknown perturbation sets, while still addressing the concerns of not having perfect knowledge of these sets in real life. In particular, we give the first positive results for the learnability of infinite Littlestone classes when having access to a perfect-attack oracle. We also consider a setting of learning with abstention, where predictions are considered robustness violations, only when the wrong prediction is made within the perturbation set. We show there are classes for which perturbation-set unaware learning without query access is possible, but abstention is required.

----

## [229] Common Ground in Cooperative Communication

**Authors**: *Xiaoran Hao, Yash Jhaveri, Patrick Shafto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/10b7e27c8eb9571fbbd2ae6a9f8c3855-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/10b7e27c8eb9571fbbd2ae6a9f8c3855-Abstract-Conference.html)

**Abstract**:

Cooperative communication plays a fundamental role in theories of human-human interaction--cognition, culture, development, language, etc.--as well as human-robot interaction. The core challenge in cooperative communication is the problem of common ground: having enough shared knowledge and understanding to successfully communicate. Prior models of cooperative communication, however, uniformly assume the strongest form of common ground, perfect and complete knowledge sharing, and, therefore, fail to capture the core challenge of cooperative communication. We propose a general theory of cooperative communication that is mathematically principled and explicitly defines a spectrum of common ground possibilities, going well beyond that of perfect and complete knowledge sharing, on spaces that permit arbitrary representations of data and hypotheses. Our framework is a strict generalization of prior models of cooperative communication. After considering a parametric form of common ground and viewing the data selection and hypothesis inference processes of communication as encoding and decoding, we establish a connection to variational autoencoding, a powerful model in modern machine learning. Finally, we carry out a series of empirical simulations to support and elaborate on our theoretical results.

----

## [230] Keep Various Trajectories: Promoting Exploration of Ensemble Policies in Continuous Control

**Authors**: *Chao Li, Chen Gong, Qiang He, Xinwen Hou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/10cb15f4559b3d578b7f24966d48a137-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/10cb15f4559b3d578b7f24966d48a137-Abstract-Conference.html)

**Abstract**:

The combination of deep reinforcement learning (DRL) with ensemble methods has been proved to be highly effective in addressing complex sequential decision-making problems. This success can be primarily attributed to the utilization of multiple models, which enhances both the robustness of the policy and the accuracy of value function estimation. However, there has been limited analysis of the empirical success of current ensemble RL methods thus far. Our new analysis reveals that the sample efficiency of previous ensemble DRL algorithms may be limited by sub-policies that are not as diverse as they could be. Motivated by these findings, our study introduces a new ensemble RL algorithm, termed \textbf{T}rajectories-awar\textbf{E} \textbf{E}nsemble exploratio\textbf{N} (TEEN). The primary goal of TEEN is to  maximize the expected return while promoting more diverse trajectories. Through extensive experiments, we demonstrate that TEEN not only enhances the sample diversity of the ensemble policy compared to using sub-policies alone but also improves the performance over ensemble RL algorithms. On average, TEEN outperforms the baseline ensemble DRL algorithms by 41\% in performance on the tested representative environments.

----

## [231] ReSync: Riemannian Subgradient-based Robust Rotation Synchronization

**Authors**: *Huikang Liu, Xiao Li, Anthony Man-Cho So*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/10e9204f14c4daa08041343455435308-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/10e9204f14c4daa08041343455435308-Abstract-Conference.html)

**Abstract**:

This work presents ReSync, a Riemannian subgradient-based algorithm for solving the robust rotation synchronization problem, which arises in various engineering applications. ReSync solves a least-unsquared minimization formulation over the rotation group, which is nonsmooth and nonconvex, and aims at recovering the underlying rotations directly. We provide strong theoretical guarantees for ReSync under the random corruption setting. Specifically, we first show that the initialization procedure of ReSync yields a proper initial point that lies in a local region around the ground-truth rotations. We next establish the weak sharpness property of the aforementioned formulation and then utilize this property to derive the local linear convergence of ReSync to the ground-truth rotations. By combining these guarantees, we conclude that ReSync converges linearly to the ground-truth rotations under appropriate conditions. Experiment results demonstrate the effectiveness of ReSync.

----

## [232] On the Exploration of Local Significant Differences For Two-Sample Test

**Authors**: *Zhijian Zhou, Jie Ni, Jia-He Yao, Wei Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/10fc83943b4540a9524af6fc67a23fef-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed increasing attentions on two-sample test with diverse real applications, while this work takes one more step on the exploration of local significant differences for two-sample test. We propose the ME$_\text{MaBiD}$, an effective test for two-sample testing, and the basic idea is to exploit local information by multiple Mahalanobis kernels and introduce bi-directional hypothesis for testing. On the exploration of local significant differences, we first partition the embedding space into several rectangle regions via a new splitting criterion, which is relevant to test power and data correlation. We then explore local significant differences based on our bi-directional masked $p$-value together with the ME$_\text{MaBiD}$ test. Theoretically, we present the asymptotic distribution and lower bounds of test power for our ME$_\text{MaBiD}$ test, and control the familywise error rate on the exploration of local significant differences. We finally conduct extensive experiments  to validate the effectiveness of our proposed methods on two-sample test and the exploration of local significant differences.

----

## [233] Fine-Grained Cross-View Geo-Localization Using a Correlation-Aware Homography Estimator

**Authors**: *Xiaolong Wang, Runsen Xu, Zhuofan Cui, Zeyu Wan, Yu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/112d8e0c7563de6e3408b49a09b4d8a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/112d8e0c7563de6e3408b49a09b4d8a3-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce a novel approach to fine-grained cross-view geo-localization. Our method aligns a warped ground image with a corresponding GPS-tagged satellite image covering the same area using homography estimation. We first employ a differentiable spherical transform, adhering to geometric principles, to accurately align the perspective of the ground image with the satellite map. This transformation effectively places ground and aerial images in the same view and on the same plane, reducing the task to an image alignment problem. To address challenges such as occlusion, small overlapping range, and seasonal variations, we propose a robust correlation-aware homography estimator to align similar parts of the transformed ground image with the satellite image. Our method achieves sub-pixel resolution and meter-level GPS accuracy by mapping the center point of the transformed ground image to the satellite image using a homography matrix and determining the orientation of the ground camera using a point above the central axis. Operating at a speed of 30 FPS, our method outperforms state-of-the-art techniques, reducing the mean metric localization error by 21.3\% and 32.4\% in same-area and cross-area generalization tasks on the VIGOR benchmark, respectively, and by 34.4\% on the KITTI benchmark in same-area evaluation.

----

## [234] DataPerf: Benchmarks for Data-Centric AI Development

**Authors**: *Mark Mazumder, Colby R. Banbury, Xiaozhe Yao, Bojan Karlas, William Gaviria Rojas, Sudnya Frederick Diamos, Greg Diamos, Lynn He, Alicia Parrish, Hannah Rose Kirk, Jessica Quaye, Charvi Rastogi, Douwe Kiela, David Jurado, David Kanter, Rafael Mosquera, Will Cukierski, Juan Ciro, Lora Aroyo, Bilge Acun, Lingjiao Chen, Mehul Raje, Max Bartolo, Evan Sabri Eyuboglu, Amirata Ghorbani, Emmett D. Goodman, Addison Howard, Oana Inel, Tariq Kane, Christine R. Kirkpatrick, D. Sculley, Tzu-Sheng Kuo, Jonas W. Mueller, Tristan Thrush, Joaquin Vanschoren, Margaret Warren, Adina Williams, Serena Yeung, Newsha Ardalani, Praveen K. Paritosh, Ce Zhang, James Y. Zou, Carole-Jean Wu, Cody Coleman, Andrew Y. Ng, Peter Mattson, Vijay Janapa Reddi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/112db88215e25b3ae2750e9eefcded94-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/112db88215e25b3ae2750e9eefcded94-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning research has long focused on models rather than datasets, and prominent datasets are used for common ML tasks without regard to the breadth, difficulty, and faithfulness of the underlying problems. Neglecting the fundamental importance of data has given rise to inaccuracy, bias, and fragility in real-world applications, and research is hindered by saturation across existing dataset benchmarks. In response, we present DataPerf, a community-led benchmark suite for evaluating ML datasets and data-centric algorithms. We aim to foster innovation in data-centric AI through competition, comparability, and reproducibility. We enable the ML community to iterate on datasets, instead of just architectures, and we provide an open, online platform with multiple rounds of challenges to support this iterative development. The first iteration of DataPerf contains five benchmarks covering a wide spectrum of data-centric techniques, tasks, and modalities in vision, speech, acquisition, debugging, and diffusion prompting, and we support hosting new contributed benchmarks from the community. The benchmarks, online evaluation platform, and baseline implementations are open source, and the MLCommons Association will maintain DataPerf to ensure long-term benefits to academia and industry.

----

## [235] Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization

**Authors**: *Quanqi Hu, Dixian Zhu, Tianbao Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1160792eab11de2bbaf9e71fce191e8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1160792eab11de2bbaf9e71fce191e8c-Abstract-Conference.html)

**Abstract**:

This paper investigates new families of compositional optimization problems, called non-smooth weakly-convex finite-sum coupled compositional optimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau envelop of the objective function. Additionally, we also extend the algorithm for solving novel non-smooth weakly-convex tri-level finite-sum coupled compositional optimization problems,  which feature a nested arrangement of three functions. Lastly, we explore the applications of our algorithms in deep learning for two-way partial AUC maximization and multi-instance two-way partial AUC maximization, using empirical studies to showcase the effectiveness of the proposed algorithms.

----

## [236] Optimal Transport for Treatment Effect Estimation

**Authors**: *Hao Wang, Jiajun Fan, Zhichao Chen, Haoxuan Li, Weiming Liu, Tianqiao Liu, Quanyu Dai, Yichao Wang, Zhenhua Dong, Ruiming Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1160e7f31d0a74abbbe1bbf7924b949c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1160e7f31d0a74abbbe1bbf7924b949c-Abstract-Conference.html)

**Abstract**:

Estimating individual treatment effects from observational data is challenging due to treatment selection bias. Prevalent methods mainly mitigate this issue by aligning different treatment groups in the latent space, the core of which is the calculation of distribution discrepancy. However, two issues that are often overlooked can render these methods invalid:(1) mini-batch sampling effects (MSE), where the calculated discrepancy is erroneous in non-ideal mini-batches with outcome imbalance and outliers;(2) unobserved confounder effects (UCE), where the unobserved confounders are not considered in the discrepancy calculation.Both of these issues invalidate the calculated discrepancy, mislead the training of estimators, and thus impede the handling of treatment selection bias.To tackle these issues, we propose Entire Space CounterFactual Regression (ESCFR), which is a new take on optimal transport technology in the context of causality.Specifically, based on the canonical optimal transport framework, we propose a relaxed mass-preserving regularizer to address the MSE issue and design a proximal factual outcome regularizer to handle the UCE issue.Extensive experiments demonstrate that ESCFR estimates distribution discrepancy accurately, handles the treatment selection bias effectively, and outperforms prevalent competitors significantly.

----

## [237] Initialization Matters: Privacy-Utility Analysis of Overparameterized Neural Networks

**Authors**: *Jiayuan Ye, Zhenyu Zhu, Fanghui Liu, Reza Shokri, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1165af8b913fb836c6280b42d6e0084f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1165af8b913fb836c6280b42d6e0084f-Abstract-Conference.html)

**Abstract**:

We analytically investigate how over-parameterization of models in randomized machine learning algorithms impacts the information leakage about their training data. Specifically, we prove a privacy bound for the KL divergence between model distributions on worst-case neighboring datasets, and explore its dependence on the initialization, width, and depth of fully connected neural networks. We find that this KL privacy bound is largely determined by the expected squared gradient norm relative to model parameters during training. Notably, for the special setting of linearized network, our analysis indicates that the squared gradient norm (and therefore the escalation of privacy loss) is tied directly to the per-layer variance of the initialization distribution. By using this analysis, we demonstrate that privacy bound improves with increasing depth under certain initializations (LeCun and Xavier), while degrades with increasing depth under other initializations (He and NTK). Our work reveals a complex interplay between privacy and depth that depends on the chosen initialization distribution. We further prove excess empirical risk bounds under a fixed KL privacy budget, and show that the interplay between privacy utility trade-off and depth is similarly affected by the initialization.

----

## [238] Cause-Effect Inference in Location-Scale Noise Models: Maximum Likelihood vs Independence Testing

**Authors**: *Xiangyu Sun, Oliver Schulte*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11715d433f6f8b9106baae0df023deb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11715d433f6f8b9106baae0df023deb3-Abstract-Conference.html)

**Abstract**:

A fundamental problem of causal discovery is cause-effect inference, to learn the correct causal direction between two random variables. Significant progress has been made through modelling the effect as a function of its cause and a noise term, which allows us to leverage assumptions about the generating function class. The recently introduced heteroscedastic location-scale noise functional models (LSNMs) combine expressive power with identifiability guarantees. LSNM model selection based on maximizing likelihood achieves state-of-the-art accuracy, when the noise distributions are correctly specified. However, through an extensive empirical evaluation, we demonstrate that the accuracy deteriorates sharply when the form of the noise distribution is misspecified by the user. Our analysis shows that the failure occurs mainly when the conditional variance in the anti-causal direction is smaller than that in the causal direction. As an alternative, we find that causal model selection through residual independence testing is much more robust to noise misspecification and misleading conditional variance.

----

## [239] M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models

**Authors**: *Wenxuan Zhang, Mahani Aljunied, Chang Gao, Yew Ken Chia, Lidong Bing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/117c5c8622b0d539f74f6d1fb082a2e9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/117c5c8622b0d539f74f6d1fb082a2e9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Despite the existence of various benchmarks for evaluating natural language processing models, we argue that human exams are a more suitable means of evaluating general intelligence for large language models (LLMs), as they inherently demand a much wider range of abilities such as language understanding, domain knowledge, and problem-solving skills. To this end, we introduce M3Exam, a novel benchmark sourced from real and official human exam questions for evaluating LLMs in a multilingual, multimodal, and multilevel context. M3Exam exhibits three unique characteristics: (1) multilingualism, encompassing questions from multiple countries that require strong multilingual proficiency and cultural knowledge; (2) multimodality, accounting for the multimodal nature of many exam questions to test the model's multimodal understanding capability; and (3) multilevel structure, featuring exams from three critical educational periods to comprehensively assess a model's proficiency at different levels. In total, M3Exam contains 12,317 questions in 9 diverse languages with three educational levels, where about 23\% of the questions require processing images for successful solving. We assess the performance of top-performing LLMs on M3Exam and find that current models, including GPT-4, still struggle with multilingual text, particularly in low-resource and non-Latin script languages. Multimodal LLMs also perform poorly with complex multimodal questions. We believe that M3Exam can be a valuable resource for comprehensively evaluating LLMs by examining their multilingual and multimodal abilities and tracking their development. Data and evaluation code is available at \url{https://github.com/DAMO-NLP-SG/M3Exam}.

----

## [240] CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

**Authors**: *Anthony Fuller, Koreen Millard, James R. Green*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)

**Abstract**:

A vital and rapidly growing application, remote sensing offers vast yet sparsely labeled, spatially aligned multimodal data; this makes self-supervised learning algorithms invaluable. We present CROMA: a framework that combines contrastive and reconstruction self-supervised objectives to learn rich unimodal and multimodal representations. Our method separately encodes masked-out multispectral optical and synthetic aperture radar samples—aligned in space and time—and performs cross-modal contrastive learning. Another encoder fuses these sensors, producing joint multimodal encodings that are used to predict the masked patches via a lightweight decoder. We show that these objectives are complementary when leveraged on spatially aligned multimodal data. We also introduce X- and 2D-ALiBi, which spatially biases our cross- and self-attention matrices. These strategies improve representations and allow our models to effectively extrapolate to images up to $17.6\times$ larger at test-time. CROMA outperforms the current SoTA multispectral model, evaluated on: four classification benchmarks—finetuning (avg.$\uparrow$ 1.8%), linear (avg.$\uparrow$ 2.4%) and nonlinear (avg.$\uparrow$ 1.4%) probing, $k$NN classification (avg.$\uparrow$ 3.5%), and $K$-means clustering (avg.$\uparrow$ 8.4%); and three segmentation benchmarks (avg.$\uparrow$ 6.4%). CROMA’s rich, optionally multimodal representations can be widely leveraged across remote sensing applications.

----

## [241] OpenAGI: When LLM Meets Domain Experts

**Authors**: *Yingqiang Ge, Wenyue Hua, Kai Mei, Jianchao Ji, Juntao Tan, Shuyuan Xu, Zelong Li, Yongfeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1190733f217404edc8a7f4e15a57f301-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1190733f217404edc8a7f4e15a57f301-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Human Intelligence (HI) excels at combining basic skills to solve complex tasks. This capability is vital for Artificial Intelligence (AI) and should be embedded in comprehensive AI Agents, enabling them to harness expert models for complex task-solving towards Artificial General Intelligence (AGI). Large Language Models (LLMs) show promising learning and reasoning abilities, and can effectively use external models, tools, plugins, or APIs to tackle complex problems. In this work, we introduce OpenAGI, an open-source AGI research and development platform designed for solving multi-step, real-world tasks. Specifically, OpenAGI uses a dual strategy, integrating standard benchmark tasks for benchmarking and evaluation, and open-ended tasks including more expandable models, tools, plugins, or APIs for creative problem-solving. Tasks are presented as natural language queries to the LLM, which then selects and executes appropriate models. We also propose a Reinforcement Learning from Task Feedback (RLTF) mechanism that uses task results to improve the LLM's task-solving ability, which creates a self-improving AI feedback loop. While we acknowledge that AGI is a broad and multifaceted research challenge with no singularly defined solution path, the integration of LLMs with domain-specific expert models, inspired by mirroring the blend of general and specialized intelligence in humans, offers a promising approach towards AGI. We are open-sourcing the OpenAGI project's code, dataset, benchmarks, evaluation methods, and the UI demo to foster community involvement in AGI advancement: https://github.com/agiresearch/OpenAGI.

----

## [242] Neural Frailty Machine: Beyond proportional hazard assumption in neural survival regressions

**Authors**: *Ruofan Wu, Jiawei Qiao, Mingzhe Wu, Wen Yu, Ming Zheng, Tengfei Liu, Tianyi Zhang, Weiqiang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11a7f429d75f9f8c6e9c630aeb6524b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11a7f429d75f9f8c6e9c630aeb6524b5-Abstract-Conference.html)

**Abstract**:

We present neural frailty machine (NFM), a powerful and flexible neural modeling framework for survival regressions. The NFM framework utilizes the classical idea of multiplicative frailty in survival analysis as a principled way of extending the proportional hazard assumption, at the same time being able to leverage the strong approximation power of neural architectures for handling nonlinear covariate dependence. Two concrete models are derived under the framework that extends neural proportional hazard models and nonparametric hazard regression models. Both models allow efficient training under the likelihood objective. Theoretically, for both proposed models, we establish statistical guarantees of neural function approximation with respect to nonparametric components via characterizing their rate of convergence. Empirically, we provide synthetic experiments that verify our theoretical statements. We also conduct experimental evaluations over $6$ benchmark datasets of different scales, showing that the proposed NFM models achieve predictive performance comparable to or sometimes surpassing state-of-the-art survival models. Our code is publicly availabel at https://github.com/Rorschach1989/nfm

----

## [243] Non-autoregressive Machine Translation with Probabilistic Context-free Grammar

**Authors**: *Shangtong Gui, Chenze Shao, Zhengrui Ma, Xishan Zhang, Yunji Chen, Yang Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11c7f1dd168439884b6dfb43a7891432-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11c7f1dd168439884b6dfb43a7891432-Abstract-Conference.html)

**Abstract**:

Non-autoregressive Transformer(NAT) significantly accelerates the inference of neural machine translation. However, conventional NAT models suffer from limited expression power and performance degradation compared to autoregressive (AT) models due to the assumption of conditional independence among target tokens. To address these limitations, we propose a novel approach called PCFG-NAT, which leverages a specially designed Probabilistic Context-Free Grammar (PCFG) to enhance the ability of NAT models to capture complex dependencies among output tokens. Experimental results on major machine translation benchmarks demonstrate that PCFG-NAT further narrows the gap in translation quality between NAT and AT models. Moreover, PCFG-NAT facilitates a deeper understanding of the generated sentences, addressing the lack of satisfactory explainability in neural machine translation. Code is publicly available at https://github.com/ictnlp/PCFG-NAT.

----

## [244] Constrained Policy Optimization with Explicit Behavior Density For Offline Reinforcement Learning

**Authors**: *Jing Zhang, Chi Zhang, Wenjia Wang, Bingyi Jing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11e1900e680f5fe1893a8e27362dbe2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/11e1900e680f5fe1893a8e27362dbe2c-Abstract-Conference.html)

**Abstract**:

Due to the inability to interact with the environment, offline reinforcement learning (RL) methods face the challenge of estimating the Out-of-Distribution (OOD) points. Existing methods for addressing this issue either control policy to exclude the OOD action or make the $Q$ function pessimistic. However, these methods can be overly conservative or fail to identify OOD areas accurately. To overcome this problem, we propose a Constrained Policy optimization with Explicit Behavior density (CPED) method that utilizes a flow-GAN model to explicitly estimate the density of behavior policy. By estimating the explicit density, CPED can accurately identify the safe region and enable exploration within the region, resulting in less conservative learning policies.  We further provide theoretical results for both the flow-GAN estimator and performance guarantee for CPED by showing that CPED can find the optimal $Q$-function value. Empirically, CPED outperforms existing alternatives on various standard offline reinforcement learning tasks, yielding higher expected returns.

----

## [245] Large Language Models are Fixated by Red Herrings: Exploring Creative Problem Solving and Einstellung Effect using the Only Connect Wall Dataset

**Authors**: *Saeid Alavi Naeini, Raeid Saqur, Mozhgan Saeidi, John Giorgi, Babak Taati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/11e3e0f1b29dcd31bd0952bfc1357f68-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/11e3e0f1b29dcd31bd0952bfc1357f68-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The quest for human imitative AI has been an enduring topic in AI research since inception. The technical evolution and emerging capabilities of the latest cohort of large language models (LLMs) have reinvigorated the subject beyond academia to cultural zeitgeist. While recent NLP evaluation benchmark tasks test some aspects of human-imitative behaviour (e.g., BIG-bench's `human-like behavior' tasks), few, if not none, examine creative problem solving abilities. Creative problem solving in humans is a well-studied topic in cognitive neuroscience with standardized tests that predominantly use ability to associate (heterogeneous) connections among clue words as a metric for creativity. Exposure to misleading stimuli --- distractors dubbed red herrings --- impede human performance in such tasks via the fixation effect and Einstellung paradigm. In cognitive neuroscience studies, such fixations are experimentally induced by pre-exposing participants to orthographically similar incorrect words to subsequent word-fragments or clues. The popular British quiz show Only Connect's Connecting Wall segment essentially mimics Mednick's Remote Associates Test (RAT) formulation with built-in, deliberate red herrings, that makes it an ideal proxy dataset to explore and study fixation effect and Einstellung paradigm from cognitive neuroscience in LLMs. In addition to presenting the novel Only Connect Wall (OCW) dataset, we also report results from our evaluation of selected pre-trained language models and LLMs (including OpenAI's GPT series) on creative problem solving tasks like grouping clue words by heterogeneous connections, and identifying correct open knowledge domain connections in respective groups. We synthetically generate two additional datasets: OCW-Randomized, OCW-WordNet to further analyze our red-herrings hypothesis in language models.The code and link to the dataset is available at url.

----

## [246] Formalizing locality for normative synaptic plasticity models

**Authors**: *Colin Bredenberg, Ezekiel Williams, Cristina Savin, Blake A. Richards, Guillaume Lajoie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/120339238f293d4ae53a7167403abc4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/120339238f293d4ae53a7167403abc4b-Abstract-Conference.html)

**Abstract**:

In recent years, many researchers have proposed new models for synaptic plasticity in the brain based on principles of machine learning. The central motivation has been the development of learning algorithms that are able to learn difficult tasks while qualifying as "biologically plausible". However, the concept of a biologically plausible learning algorithm is only heuristically defined as an algorithm that is potentially implementable by biological neural networks. Further, claims that neural circuits could implement any given algorithm typically rest on an amorphous concept of "locality" (both in space and time). As a result, it is unclear what many proposed local learning algorithms actually predict biologically, and which of these are consequently good candidates for experimental investigation. Here, we address this lack of clarity by proposing formal and operational definitions of locality. Specifically, we define different classes of locality, each of which makes clear what quantities cannot be included in a learning rule if an algorithm is to qualify as local with respect to a given (biological) constraint. We subsequently use this framework to distill testable predictions from various classes of biologically plausible synaptic plasticity models that are robust to arbitrary choices about neural network architecture. Therefore, our framework can be used to guide claims of biological plausibility and to identify potential means of experimentally falsifying a proposed learning algorithm for the brain.

----

## [247] Exact Verification of ReLU Neural Control Barrier Functions

**Authors**: *Hongchao Zhang, Junlin Wu, Yevgeniy Vorobeychik, Andrew Clark*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html)

**Abstract**:

Control Barrier Functions (CBFs) are a popular approach for safe control of nonlinear systems. In CBF-based control, the desired safety properties of the system are mapped to nonnegativity of a CBF, and the control input is chosen to ensure that the CBF remains nonnegative for all time. Recently, machine learning methods that represent CBFs as neural networks (neural control barrier functions, or NCBFs) have shown great promise due to the universal representability of neural networks. However, verifying that a learned CBF guarantees safety remains a challenging research problem. This paper presents novel exact conditions and algorithms for verifying safety of feedforward NCBFs with ReLU activation functions. The key challenge in doing so is that, due to the piecewise linearity of the ReLU function, the NCBF will be nondifferentiable at certain points, thus invalidating traditional safety verification methods that assume a smooth barrier function. We resolve this issue by leveraging a generalization of Nagumo's theorem for proving invariance of sets with nonsmooth boundaries to derive necessary and sufficient conditions for safety. Based on this condition, we propose an algorithm for safety verification of NCBFs that first decomposes the NCBF into piecewise linear segments and then solves a nonlinear program to verify safety of each segment as well as the intersections of the linear segments. We  mitigate the complexity by only considering the boundary of the safe region and by pruning  the segments with Interval Bound Propagation (IBP) and linear relaxation. We evaluate our approach through numerical studies with comparison to state-of-the-art SMT-based methods. Our code is available at https://github.com/HongchaoZhang-HZ/exactverif-reluncbf-nips23.

----

## [248] Normalization-Equivariant Neural Networks with Application to Image Denoising

**Authors**: *Sébastien Herbreteau, Emmanuel Moebel, Charles Kervrann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12143893d9d37c3569dda800b95cabd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12143893d9d37c3569dda800b95cabd9-Abstract-Conference.html)

**Abstract**:

In many information processing systems, it may be desirable to ensure that any change of the input, whether by shifting or  scaling, results in a corresponding change in the system response.  While deep neural networks are gradually replacing all traditional automatic processing methods, they surprisingly do not guarantee such normalization-equivariance (scale + shift) property, which can be detrimental in many applications. To address this issue, we propose a methodology for adapting existing neural networks so that normalization-equivariance holds by design. Our main claim is that not only ordinary convolutional layers, but also all activation functions, including the ReLU (rectified linear unit), which are applied element-wise to the pre-activated neurons, should be completely removed from neural networks and replaced by better conditioned alternatives. To this end, we introduce affine-constrained convolutions and channel-wise sort pooling layers as surrogates and show that these two architectural modifications do preserve normalization-equivariance without loss of performance. Experimental results in image denoising show that normalization-equivariant neural networks, in addition to their better conditioning, also provide much better generalization across noise levels.

----

## [249] Budgeting Counterfactual for Offline RL

**Authors**: *Yao Liu, Pratik Chaudhari, Rasool Fakoor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/121db870b0470dd63bb5bc59c724275a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/121db870b0470dd63bb5bc59c724275a-Abstract-Conference.html)

**Abstract**:

The main challenge of offline reinforcement learning, where data is limited, arises from a sequence of counterfactual reasoning dilemmas within the realm of potential actions: What if we were to choose a different course of action? These circumstances frequently give rise to extrapolation errors, which tend to accumulate exponentially with the problem horizon. Hence, it becomes crucial to acknowledge that not all decision steps are equally important to the final outcome, and to budget the number of counterfactual decisions a policy make in order to control the extrapolation. Contrary to existing approaches that use regularization on either the policy or value function, we propose an approach to explicitly bound the amount of out-of-distribution actions during training. Specifically, our method utilizes dynamic programming to decide where to extrapolate and where not to, with an upper bound on the decisions different from behavior policy. It balances between the potential for improvement from taking out-of-distribution actions and the risk of making errors due to extrapolation. Theoretically, we justify our method by the constrained optimality of the fixed point solution to our $Q$ updating rules. Empirically, we show that the overall performance of our method is better than the state-of-the-art offline RL methods on tasks in the widely-used D4RL benchmarks.

----

## [250] Federated Conditional Stochastic Optimization

**Authors**: *Xidong Wu, Jianhui Sun, Zhengmian Hu, Junyi Li, Aidong Zhang, Heng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1229eaae5bf1db93e1e4c539258eb472-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1229eaae5bf1db93e1e4c539258eb472-Abstract-Conference.html)

**Abstract**:

Conditional stochastic optimization has found applications in a wide range of machine learning tasks, such as invariant learning, AUPRC maximization, and meta-learning. As the demand for training models with large-scale distributed data grows in these applications, there is an increasing need for communication-efficient distributed optimization algorithms, such as federated learning algorithms. This paper considers the nonconvex conditional stochastic optimization in federated learning and proposes the first federated conditional stochastic optimization algorithm (FCSG) with a conditional stochastic gradient estimator and a momentum-based algorithm (\emph{i.e.}, FCSG-M). To match the lower bound complexity in the single-machine setting, we design an accelerated algorithm (Acc-FCSG-M) via the variance reduction to achieve the best sample and communication complexity. Compared with the existing optimization analysis for Meta-Learning in FL, federated conditional stochastic optimization considers the sample of tasks. Extensive experimental results on various tasks validate the efficiency of these algorithms.

----

## [251] LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections

**Authors**: *Muhammad Jehanzeb Mirza, Leonid Karlinsky, Wei Lin, Horst Possegger, Mateusz Kozinski, Rogério Feris, Horst Bischof*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/123a18dfd821c8b440f42a00a27648d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/123a18dfd821c8b440f42a00a27648d6-Abstract-Conference.html)

**Abstract**:

Recently, large-scale pre-trained Vision and Language (VL) models have set a new state-of-the-art (SOTA) in zero-shot visual classification enabling open-vocabulary recognition of potentially unlimited set of categories defined as simple language prompts. However, despite these great advances, the performance of these zero-shot classifiers still falls short of the results of dedicated (closed category set) classifiers trained with supervised fine-tuning. In this paper we show, for the first time, how to reduce this gap without any labels and without any paired VL data, using an unlabeled image collection and a set of texts auto-generated using a Large Language Model (LLM) describing the categories of interest and effectively substituting labeled visual instances of those categories. Using our label-free approach, we are able to attain significant performance improvements over the zero-shot performance of the base VL model and other contemporary methods and baselines on a wide variety of datasets, demonstrating absolute improvement of up to $11.7\%$ ($3.8\%$ on average) in the label-free setting. Moreover, despite our approach being label-free, we observe $1.3\%$ average gains over leading few-shot prompting baselines that do use 5-shot supervision.

----

## [252] Contextually Affinitive Neighborhood Refinery for Deep Clustering

**Authors**: *Chunlin Yu, Ye Shi, Jingya Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/123cfe7d8b7702ac97aaf4468fc05fa5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/123cfe7d8b7702ac97aaf4468fc05fa5-Abstract-Conference.html)

**Abstract**:

Previous endeavors in self-supervised learning have enlightened the research of deep clustering from an instance discrimination perspective. Built upon this foundation, recent studies further highlight the importance of grouping semantically similar instances. One effective method to achieve this is by promoting the semantic structure preserved by neighborhood consistency. However, the samples in the local neighborhood may be limited due to their close proximity to each other,  which may not provide substantial and diverse supervision signals. Inspired by the versatile re-ranking methods in the context of image retrieval, we propose to employ an efficient online re-ranking process to mine more informative neighbors in a Contextually Affinitive (ConAff) Neighborhood, and then encourage the cross-view neighborhood consistency. To further mitigate the intrinsic neighborhood noises near cluster boundaries, we propose a progressively relaxed boundary filtering strategy to circumvent the issues brought by noisy neighbors. Our method can be easily integrated into the generic self-supervised frameworks and outperforms the state-of-the-art methods on several popular benchmarks.

----

## [253] Differentiable Blocks World: Qualitative 3D Decomposition by Rendering Primitives

**Authors**: *Tom Monnier, Jake Austin, Angjoo Kanazawa, Alexei A. Efros, Mathieu Aubry*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html)

**Abstract**:

Given a set of calibrated images of a scene, we present an approach that produces a simple, compact, and actionable 3D world representation by means of 3D primitives. While many approaches focus on recovering high-fidelity 3D scenes, we focus on parsing a scene into mid-level 3D representations made of a small set of textured primitives. Such representations are interpretable, easy to manipulate and suited for physics-based simulations. Moreover, unlike existing primitive decomposition methods that rely on 3D input data, our approach operates directly on images through differentiable rendering. Specifically, we model primitives as textured superquadric meshes and optimize their parameters from scratch with an image rendering loss. We highlight the importance of modeling transparency for each primitive, which is critical for optimization and also enables handling varying numbers of primitives. We show that the resulting textured primitives faithfully reconstruct the input images and accurately model the visible 3D points, while providing amodal shape completions of unseen object regions. We compare our approach to the state of the art on diverse scenes from DTU, and demonstrate its robustness on real-life captures from BlendedMVS and Nerfstudio. We also showcase how our results can be used to effortlessly edit a scene or perform physical simulations. Code and video results are available at https://www.tmonnier.com/DBW.

----

## [254] Learning Shared Safety Constraints from Multi-task Demonstrations

**Authors**: *Konwoo Kim, Gokul Swamy, Zuxin Liu, Ding Zhao, Sanjiban Choudhury, Steven Z. Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/124dde499d62b58e97e42a45b26d7369-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/124dde499d62b58e97e42a45b26d7369-Abstract-Conference.html)

**Abstract**:

Regardless of the particular task we want to perform in an environment, there are often shared safety constraints we want our agents to respect. For example, regardless of whether it is making a sandwich or clearing the table, a kitchen robot should not break a plate. Manually specifying such a constraint can be both time-consuming and error-prone. We show how to learn constraints from expert demonstrations of safe task completion by extending inverse reinforcement learning (IRL) techniques to the space of constraints. Intuitively, we learn constraints that forbid highly rewarding behavior that the expert could have taken but chose not to. Unfortunately, the constraint learning problem is rather ill-posed and typically leads to overly conservative constraints that forbid all behavior that the expert did not take. We counter this by leveraging diverse demonstrations that naturally occur in multi-task setting to learn a tighter set of constraints. We validate our method with simulation experiments on high-dimensional continuous control tasks.

----

## [255] Don't Stop Pretraining? Make Prompt-based Fine-tuning Powerful Learner

**Authors**: *Zhengxiang Shi, Aldo Lipani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1289f9195d2ef8cfdfe5f50930c4a7c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1289f9195d2ef8cfdfe5f50930c4a7c4-Abstract-Conference.html)

**Abstract**:

Language models (LMs) trained on vast quantities of unlabelled data have greatly advanced the field of natural language processing (NLP). In this study, we re-visit the widely accepted notion in NLP that continued pre-training LMs on task-related texts improves the performance of fine-tuning (FT) in downstream tasks. Through experiments on eight single-sentence tasks and eight sentence-pair tasks in both semi-supervised and fully-supervised settings, we find that conventional continued pre-training does not consistently provide benefits and can even be detrimental for sentence-pair tasks or when prompt-based FT is used. To tackle these issues, we propose Prompt-based Continued Pre-training (PCP), which combines the idea of instruction tuning with conventional continued pre-training. Our approach aims to improve the performance of prompt-based FT by presenting both task-related texts and prompt templates to LMs through unsupervised pre-training objectives before fine-tuning for the target task. Our empirical evaluations on 21 benchmarks demonstrate that the PCP consistently improves the performance of state-of-the-art prompt-based FT approaches (up to 20.1% absolute) in both semi-supervised and fully-supervised settings, even with only hundreds of unlabelled examples. Additionally, prompt-based FT with PCP outperforms state-of-the-art semi-supervised approaches with greater simplicity, eliminating the need for an iterative process and extra data augmentation. Our further analysis explores the performance lower bound of the PCP and reveals that the advantages of PCP persist across different sizes of models and datasets.

----

## [256] GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning

**Authors**: *Haiteng Zhao, Shengchao Liu, Chang Ma, Hannan Xu, Jie Fu, Zhihong Deng, Lingpeng Kong, Qi Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/129033c7c08be683059559e8d6bfd460-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/129033c7c08be683059559e8d6bfd460-Abstract-Conference.html)

**Abstract**:

Molecule property prediction has gained significant attention in recent years. The main bottleneck is the label insufficiency caused by expensive lab experiments. In order to alleviate this issue and to better leverage textual knowledge for tasks, this study investigates the feasibility of employing natural language instructions to accomplish molecule-related tasks in a zero-shot setting. We discover that existing molecule-text models perform poorly in this setting due to inadequate treatment of instructions and limited capacity for graphs.  To overcome these issues, we propose GIMLET, which unifies language models for both graph and text data. By adopting generalized position embedding, our model is extended to encode both graph structures and instruction text without additional graph encoding modules. GIMLET also decouples encoding of the graph from tasks instructions in the attention mechanism, enhancing the generalization of graph features across novel tasks. We construct a dataset consisting of more than two thousand molecule tasks with corresponding instructions derived from task descriptions. We pretrain GIMLET on the molecule tasks along with instructions, enabling the model to transfer effectively to a broad range of tasks. Experimental results demonstrate that GIMLET significantly outperforms molecule-text baselines in instruction-based zero-shot learning, even achieving closed results to supervised GNN models on tasks such as toxcast and muv.

----

## [257] GEX: A flexible method for approximating influence via Geometric Ensemble

**Authors**: *Sungyub Kim, Kyungsu Kim, Eunho Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1297ca5c906f4bada8f5f6f4e80f9dd2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1297ca5c906f4bada8f5f6f4e80f9dd2-Abstract-Conference.html)

**Abstract**:

Through a deeper understanding of predictions of neural networks, Influence Function (IF) has been applied to various tasks such as detecting and relabeling mislabeled samples, dataset pruning, and separation of data sources in practice. However, we found standard approximations of IF suffer from performance degradation due to oversimplified influence distributions caused by their bilinear approximation, suppressing the expressive power of samples with a relatively strong influence. To address this issue, we propose a new interpretation of existing IF approximations as an average relationship between two linearized losses over parameters sampled from the Laplace approximation (LA). In doing so, we highlight two significant limitations of current IF approximations: the linearity of gradients and the singularity of Hessian. Accordingly, by improving each point, we introduce a new IF approximation method with the following features: i) the removal of linearization to alleviate the bilinear constraint and ii) the utilization of Geometric Ensemble (GE) tailored for non-linear losses. Empirically, our approach outperforms existing IF approximations for downstream tasks with lighter computation, thereby providing new feasibility of low-complexity/nonlinear-based IF design.

----

## [258] Offline Reinforcement Learning for Mixture-of-Expert Dialogue Management

**Authors**: *Dhawal Gupta, Yinlam Chow, Azamat Tulepbergenov, Mohammad Ghavamzadeh, Craig Boutilier*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12bcf58a1c09a0fcb5310f3589291ab4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12bcf58a1c09a0fcb5310f3589291ab4-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) has shown great promise for developing agents for dialogue management (DM) that are non-myopic, conduct rich conversations, and maximize overall user satisfaction. Despite the advancements in RL and language models (LMs), employing RL to drive conversational chatbots still poses significant challenges. A primary issue stems from RLâ€™s dependency on online exploration for effective learning, a process that can be costly. Moreover, engaging in online interactions with humans during the training phase can raise safety concerns, as the LM can potentially generate unwanted outputs. This issue is exacerbated by the combinatorial action spaces facing these algorithms, as most LM agents generate responses at the word level. We develop various RL algorithms, specialized in dialogue planning, that leverage recent Mixture-of-Expert Language Models (MoE-LMs)---models that capture diverse semantics, generate utterances reflecting different intents, and are amenable for multi-turn DM. By exploiting the MoE-LM structure, our methods significantly reduce the size of the action space and improve the efficacy of RL-based DM. We evaluate our methods in open-domain dialogue to demonstrate their effectiveness with respect to the diversity of intent in generated utterances and overall DM performance.

----

## [259] Binary Classification with Confidence Difference

**Authors**: *Wei Wang, Lei Feng, Yuchen Jiang, Gang Niu, Min-Ling Zhang, Masashi Sugiyama*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12c118ef87fde56a10bd858842781b34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12c118ef87fde56a10bd858842781b34-Abstract-Conference.html)

**Abstract**:

Recently, learning with soft labels has been shown to achieve better performance than learning with hard labels in terms of model generalization, calibration, and robustness. However, collecting pointwise labeling confidence for all training examples can be challenging and time-consuming in real-world scenarios. This paper delves into a novel weakly supervised binary classification problem called confidence-difference (ConfDiff) classification. Instead of pointwise labeling confidence, we are given only unlabeled data pairs with confidence difference that specifies the difference in the probabilities of being positive. We propose a risk-consistent approach to tackle this problem and show that the estimation error bound achieves the optimal convergence rate. We also introduce a risk correction approach to mitigate overfitting problems, whose consistency and convergence rate are also proven. Extensive experiments on benchmark data sets and a real-world recommender system data set validate the effectiveness of our proposed approaches in exploiting the supervision information of the confidence difference.

----

## [260] On student-teacher deviations in distillation: does it pay to disobey?

**Authors**: *Vaishnavh Nagarajan, Aditya Krishna Menon, Srinadh Bhojanapalli, Hossein Mobahi, Sanjiv Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12d286282e1be5431ea05262a21f415c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12d286282e1be5431ea05262a21f415c-Abstract-Conference.html)

**Abstract**:

Knowledge distillation (KD) has been widely used to improve the test accuracy of a "student" network, by training it to mimic the soft probabilities of a trained "teacher" network. Yet, it has been shown in recent work that, despite being trained to fit the teacher's probabilities, the student may not only significantly deviate from the teacher probabilities, but may also outdo than the teacher in performance. Our work aims to reconcile this seemingly paradoxical observation. Specifically, we characterize the precise nature of the student-teacher deviations, and argue how they can co-occur with better generalization.  First, through experiments on  image and language data, we identify that these probability deviations correspond to the student systematically exaggerating the confidence levels of the teacher.Next, we theoretically and empirically establish another form of exaggeration in some simple settings: KD exaggerates the implicit bias of gradient descent in converging faster along the top eigendirections of the data. Finally, we tie these two observations together: we demonstrate that the exaggerated bias of KD can simultaneously result in both (a) the exaggeration of confidence and (b) the improved generalization of the student, thus offering a resolution to the apparent paradox.  Our analysis brings existing theory and practice closer by considering the role of gradient descent in KD and by demonstrating the exaggerated bias effect in both theoretical and empirical settings.

----

## [261] Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis

**Authors**: *Victor Letzelter, Mathieu Fontaine, Mickaël Chen, Patrick Pérez, Slim Essid, Gaël Richard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12d7ba753894ed348904df1bf0ce02ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12d7ba753894ed348904df1bf0ce02ec-Abstract-Conference.html)

**Abstract**:

We introduce Resilient Multiple Choice Learning (rMCL), an extension of the MCL approach for conditional distribution estimation in regression settings where multiple targets may be sampled for each training input.Multiple Choice Learning is a simple framework to tackle multimodal density estimation, using the Winner-Takes-All (WTA) loss for a set of hypotheses. In regression settings, the existing MCL variants focus on merging the hypotheses, thereby eventually sacrificing the diversity of the predictions. In contrast, our method relies on a novel learned scoring scheme underpinned by a mathematical framework based on Voronoi tessellations of the output space, from which we can derive a probabilistic interpretation.After empirically validating rMCL with experiments on synthetic data, we further assess its merits on the sound source localization problem, demonstrating its practical usefulness and the relevance of its interpretation.

----

## [262] Graph of Circuits with GNN for Exploring the Optimal Design Space

**Authors**: *Aditya Shahane, Saripilli Swapna Manjiri, Ankesh Jain, Sandeep Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/12da92b7c64176eb6eb6ad0ae31554fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/12da92b7c64176eb6eb6ad0ae31554fd-Abstract-Conference.html)

**Abstract**:

The design automation of analog circuits poses significant challenges in terms of the large design space, complex interdependencies between circuit specifications, and resource-intensive simulations. To address these challenges, this paper presents an innovative framework called the Graph of Circuits Explorer (GCX). Leveraging graph structure learning along with graph neural networks, GCX enables the creation of a surrogate model that facilitates efficient exploration of the optimal design space within a semi-supervised learning framework which reduces the need for large labelled datasets. The proposed approach comprises three key stages. First, we learn the geometric representation of circuits and enrich it with technology information to create a comprehensive feature vector. Subsequently, integrating feature-based graph learning with few-shot and zero-shot learning  enhances the generalizability in predictions for unseen circuits. Finally, we introduce two algorithms namely, EASCO and ASTROG which upon integration with GCX optimize the available samples to yield the optimal circuit configuration meeting the designer's criteria. The effectiveness of the proposed approach is demonstrated through simulated performance evaluation of various circuits, using derived parameters in 180nm CMOS technology. Furthermore, the generalizability of the approach is extended to higher-order topologies and different technology nodes such as 65nm and 45nm CMOS process nodes.

----

## [263] Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data

**Authors**: *Xin Zheng, Miao Zhang, Chunyang Chen, Quoc Viet Hung Nguyen, Xingquan Zhu, Shirui Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13183a224208671a6fc33ba1aa661ec4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13183a224208671a6fc33ba1aa661ec4-Abstract-Conference.html)

**Abstract**:

Graph condensation, which reduces the size of a large-scale graph by synthesizing a small-scale condensed graph as its substitution, has immediate benefits for various graph learning tasks.However, existing graph condensation methods rely on the joint optimization of nodes and structures in the condensed graph, and overlook critical issues in effectiveness and generalization ability.In this paper, we advocate a new Structure-Free Graph Condensation paradigm, named SFGC, to distill a large-scale graph into a small-scale graph node set without explicit graph structures, i.e., graph-free data.Our idea is to implicitly encode topology structure information into the node attributes in the synthesized graph-free data, whose topology is reduced to an identity matrix.Specifically, SFGC contains two collaborative components: (1) a training trajectory meta-matching scheme for effectively synthesizing small-scale graph-free data;(2) a graph neural feature score metric for dynamically evaluating the quality of the condensed data. Through training trajectory meta-matching, SFGC aligns the long-term GNN learning behaviors between the large-scale graph and the condensed small-scale graph-free data, ensuring comprehensive and compact transfer of informative knowledge to the graph-free data.Afterward, the underlying condensed graph-free data would be dynamically evaluated with the graph neural feature score, which is a closed-form metric for ensuring the excellent expressiveness of the condensed graph-free data.Extensive experiments verify the superiority of SFGC across different condensation ratios.

----

## [264] Visual Programming for Step-by-Step Text-to-Image Generation and Evaluation

**Authors**: *Jaemin Cho, Abhay Zala, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13250eb13871b3c2c0a0667b54bad165-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13250eb13871b3c2c0a0667b54bad165-Abstract-Conference.html)

**Abstract**:

As large language models have demonstrated impressive performance in many domains, recent works have adopted language models (LMs) as controllers of visual modules for vision-and-language tasks. While existing work focuses on equipping LMs with visual understanding, we propose two novel interpretable/explainable visual programming frameworks for text-to-image (T2I) generation and evaluation. First, we introduce VPGen, an interpretable step-by-step T2I generation framework that decomposes T2I generation into three steps: object/count generation, layout generation, and image generation. We employ an LM to handle the first two steps (object/count generation and layout generation), by finetuning it on text-layout pairs. Our step-by-step T2I generation framework provides stronger spatial control than end-to-end models, the dominant approach for this task. Furthermore, we leverage the world knowledge of pretrained LMs, overcoming the limitation of previous layout-guided T2I works that can only handle predefined object classes. We demonstrate that our VPGen has improved control in counts/spatial relations/scales of objects than state-of-the-art T2I generation models. Second, we introduce VPEval, an interpretable and explainable evaluation framework for T2I generation based on visual programming. Unlike previous T2I evaluations with a single scoring model that is accurate in some skills but unreliable in others, VPEval produces evaluation programs that invoke a set of visual modules that are experts in different skills, and also provides visual+textual explanations of the evaluation results. Our analysis shows that VPEval provides a more human-correlated evaluation for skill-specific and open-ended prompts than widely used single model-based evaluation. We hope that our work encourages future progress on interpretable/explainable generation and evaluation for T2I models.

----

## [265] Auditing Fairness by Betting

**Authors**: *Ben Chugg, Santiago Cortes-Gomez, Bryan Wilder, Aaditya Ramdas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1338c277525011f20166cf740952bb47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1338c277525011f20166cf740952bb47-Abstract-Conference.html)

**Abstract**:

We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a  probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics---the ``testing by betting'' framework in particular. These connections ensure that our methods are interpretable, fast, and easy to implement. We demonstrate the efficacy of our approach on three benchmark fairness datasets.

----

## [266] Truly Scale-Equivariant Deep Nets with Fourier Layers

**Authors**: *Md Ashiqur Rahman, Raymond A. Yeh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1343edb2739a61a6e20bd8764e814b50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1343edb2739a61a6e20bd8764e814b50-Abstract-Conference.html)

**Abstract**:

In computer vision, models must be able to adapt to changes in image resolution to effectively carry out tasks such as image segmentation; This is known as scale-equivariance. Recent works have made progress in developing scale-equivariant convolutional neural networks, e.g., through weight-sharing and kernel resizing. However, these networks are not truly scale-equivariant in practice. Specifically, they do not consider anti-aliasing as they formulate the down-scaling operation in the continuous domain. To address this shortcoming, we directly formulate down-scaling in the discrete domain with consideration of anti-aliasing. We then propose a novel architecture based on Fourier layers to achieve truly scale-equivariant deep nets, i.e., absolute zero equivariance-error. Following prior works, we test this model on MNIST-scale and STL-10 datasets. Our proposed model achieves competitive classification performance while maintaining zero equivariance-error.

----

## [267] Projection-Free Methods for Stochastic Simple Bilevel Optimization with Convex Lower-level Problem

**Authors**: *Jincheng Cao, Ruichen Jiang, Nazanin Abolfazli, Erfan Yazdandoost Hamedani, Aryan Mokhtari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/136729ae4b0fee25a0d28077442506da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/136729ae4b0fee25a0d28077442506da-Abstract-Conference.html)

**Abstract**:

In this paper, we study a class of stochastic bilevel optimization problems, also known as stochastic simple bilevel optimization, where we minimize a smooth stochastic objective function over the optimal solution set of another stochastic convex optimization problem. We introduce novel stochastic bilevel optimization methods that locally approximate the solution set of the lower-level problem via a stochastic cutting plane, and then run a conditional gradient update with variance reduction techniques to control the error induced by using stochastic gradients. For the case that the upper-level function is convex, our method requires $\mathcal{O}(\max\\{1/\epsilon_f^{2},1/\epsilon_g^{2}\\}) $ stochastic oracle queries to obtain a solution that is $\epsilon_f$-optimal for the upper-level and $\epsilon_g$-optimal for the lower-level. This guarantee improves the previous best-known complexity of $\mathcal{O}(\max\\{1/\epsilon_f^{4},1/\epsilon_g^{4}\\})$. Moreover, for the case that the upper-level function is non-convex, our method requires at most $\mathcal{O}(\max\\{1/\epsilon_f^{3},1/\epsilon_g^{3}\\}) $ stochastic oracle queries to find an $(\epsilon_f, \epsilon_g)$-stationary point. In the finite-sum setting, we show that the number of stochastic oracle calls required by our method are  $\mathcal{O}(\sqrt{n}/\epsilon)$ and $\mathcal{O}(\sqrt{n}/\epsilon^{2})$ for the convex and non-convex settings, respectively, where $\epsilon=\min \\{\epsilon_f,\epsilon_g\\}$.

----

## [268] On the Implicit Bias of Linear Equivariant Steerable Networks

**Authors**: *Ziyu Chen, Wei Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/136a45cd9b841bf785625709a19c6508-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/136a45cd9b841bf785625709a19c6508-Abstract-Conference.html)

**Abstract**:

We study the implicit bias of gradient flow on linear equivariant steerable networks in group-invariant binary classification. Our findings reveal that the parameterized predictor converges in direction to the unique group-invariant classifier with a maximum margin defined by the input group action. Under a unitary assumption on the input representation, we establish the equivalence between steerable networks and data augmentation. Furthermore, we demonstrate the improved margin and generalization bound of steerable networks over their non-invariant counterparts.

----

## [269] Memory-Constrained Algorithms for Convex Optimization

**Authors**: *Moïse Blanchard, Junhui Zhang, Patrick Jaillet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1395b425d06a50e42fafe91cf04f3a98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1395b425d06a50e42fafe91cf04f3a98-Abstract-Conference.html)

**Abstract**:

We propose a family of recursive cutting-plane algorithms to solve feasibility problems with constrained memory, which can also be used for first-order convex optimization. Precisely, in order to find a point within a ball of radius $\epsilon$ with a separation oracle in dimension $d$---or to minimize $1$-Lipschitz convex functions to accuracy $\epsilon$ over the unit ball---our algorithms use $\mathcal O(\frac{d^2}{p}\ln \frac{1}{\epsilon})$ bits of memory, and make $\mathcal O((C\frac{d}{p}\ln \frac{1}{\epsilon})^p)$ oracle calls. The family is parametrized by $p\in[d]$ and provides an oracle-complexity/memory trade-off in the sub-polynomial regime $\ln\frac{1}{\epsilon}\gg\ln d$. While several works gave lower-bound trade-offs (impossibility results)---we explicit here their dependence with $\ln\frac{1}{\epsilon}$, showing that these also hold in any sub-polynomial regime---to the best of our knowledge this is the first class of algorithms that provides a positive trade-off between gradient descent and cutting-plane methods in any regime with $\epsilon\leq 1/\sqrt d$. The algorithms divide the $d$ variables into $p$ blocks and optimize over blocks sequentially, with approximate separation vectors constructed using a variant of Vaidya's method. In the regime $\epsilon \leq d^{-\Omega(d)}$, our algorithm with $p=d$ achieves the information-theoretic optimal memory usage and improves the oracle-complexity of gradient descent.

----

## [270] Nonparametric Boundary Geometry in Physics Informed Deep Learning

**Authors**: *Scott Alexander Cameron, Arnu Pretorius, Stephen J. Roberts*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13aef57cf532e88c476a10ff372e44e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13aef57cf532e88c476a10ff372e44e5-Abstract-Conference.html)

**Abstract**:

Engineering design problems frequently require solving systems ofpartial differential equations with boundary conditions specified onobject geometries in the form of a triangular mesh. These boundarygeometries are provided by a designer and are problem dependent.The efficiency of the design process greatly benefits from fast turnaroundtimes when repeatedly solving PDEs on various geometries. However,most current work that uses machine learning to speed up the solutionprocess relies heavily on a fixed parameterization of the geometry, whichcannot be changed after training. This severely limits the possibility ofreusing a trained model across a variety of design problems.In this work, we propose a novel neural operator architecture which acceptsboundary geometry, in the form of triangular meshes, as input and produces anapproximate solution to a given PDE as output. Once trained, the model can beused to rapidly estimate the PDE solution over a new geometry, without the need forretraining or representation of the geometry to a pre-specified parameterization.

----

## [271] Tracking Most Significant Shifts in Nonparametric Contextual Bandits

**Authors**: *Joe Suk, Samory Kpotufe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13b501c58ae3bfe9635a259f4414e943-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13b501c58ae3bfe9635a259f4414e943-Abstract-Conference.html)

**Abstract**:

We study nonparametric contextual bandits where Lipschitz mean reward functions may change over time.We first establish the minimax dynamic regret rate in this less understood setting in terms of number of changes $L$ and total-variation $V$, both capturing all changes in distribution over context space, and argue that state-of-the-art procedures are suboptimal in this setting.Next, we tend to the question of an _adaptivity_ for this setting, i.e. achieving the minimax rate without knowledge of $L$ or $V$. Quite importantly, we posit that the bandit  problem, viewed locally at a given context $X_t$, should not be affected by reward changes in other parts of context space $\cal X$. We therefore propose a notion of _change_, which we term _experienced significant shifts_, that better accounts for locality, and thus counts considerably less changes than $L$ and $V$. Furthermore, similar to recent work on non-stationary MAB (Suk & Kpotufe, 2022), _experienced significant shifts_ only count the most _significant_ changes in mean rewards, e.g., severe best-arm changes relevant to observed contexts.Our main result is to show that this more tolerant notion of change can in fact be adapted to.

----

## [272] Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss

**Authors**: *An Zhang, Leheng Sheng, Zhibo Cai, Xiang Wang, Tat-Seng Chua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/13f1750b825659394a6499399e7637fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/13f1750b825659394a6499399e7637fc-Abstract-Conference.html)

**Abstract**:

Contrastive Learning (CL) has achieved impressive performance in self-supervised learning tasks, showing superior generalization ability. Inspired by the success, adopting CL into collaborative filtering (CF) is prevailing in semi-supervised topK recommendations. The basic idea is to routinely conduct heuristic-based data augmentation and apply contrastive losses (e.g., InfoNCE) on the augmented views. Yet, some CF-tailored challenges make this adoption suboptimal, such as the issue of out-of-distribution, the risk of false negatives, and the nature of top-K evaluation. They necessitate the CL-based CF scheme to focus more on mining hard negatives and distinguishing false negatives from the vast unlabeled user-item interactions, for informative contrast signals. Worse still, there is limited understanding of contrastive loss in CF methods, especially w.r.t. its generalization ability. To bridge the gap, we delve into the reasons underpinning the success of contrastive loss in CF, and propose a principled Adversarial InfoNCE loss (AdvInfoNCE), which is a variant of InfoNCE, specially tailored for CF methods. AdvInfoNCE adaptively explores and assigns hardness to each negative instance in an adversarial fashion and further utilizes a fine-grained hardness-aware ranking criterion to empower the recommenderâ€™s generalization ability. Training CF models with AdvInfoNCE, we validate the effectiveness of AdvInfoNCE on both synthetic and real-world benchmark datasets, thus showing its generalization ability to mitigate out-of-distribution problems. Given the theoretical guarantees and empirical superiority of AdvInfoNCE over most contrastive loss functions, we advocate its adoption as a standard loss in recommender systems, particularly for the out-of-distribution tasks. Codes are available at https://github.com/LehengTHU/AdvInfoNCE.

----

## [273] The Rashomon Importance Distribution: Getting RID of Unstable, Single Model-based Variable Importance

**Authors**: *Jon Donnelly, Srikar Katta, Cynthia Rudin, Edward P. Browne*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1403ab1a427050538ec59c7f570aec8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1403ab1a427050538ec59c7f570aec8b-Abstract-Conference.html)

**Abstract**:

Quantifying variable importance is essential for answering high-stakes questions in fields like genetics, public policy, and medicine. Current methods generally calculate variable importance for a given model trained on a given dataset. However, for a given dataset, there may be many models that explain the target outcome equally well; without accounting for all possible explanations, different researchers may arrive at many conflicting yet equally valid conclusions given the same data. Additionally, even when accounting for all possible explanations for a given dataset, these insights may not generalize because not all good explanations are stable across reasonable data perturbations. We propose a new variable importance framework that quantifies the importance of a variable across the set of all good models and is stable across the data distribution. Our framework is extremely flexible and can be integrated with most existing model classes and global variable importance metrics. We demonstrate through experiments that our framework recovers variable importance rankings for complex simulation setups where other methods fail. Further, we show that our framework accurately estimates the true importance of a variable for the underlying data distribution. We provide theoretical guarantees on the consistency and finite sample error rates for our estimator. Finally, we demonstrate its utility with a real-world case study exploring which genes are important for predicting HIV load in persons with HIV, highlighting an important gene that has not previously been studied in connection with HIV.

----

## [274] Model-Based Control with Sparse Neural Dynamics

**Authors**: *Ziang Liu, Genggeng Zhou, Jeff He, Tobia Marcucci, Fei-Fei Li, Jiajun Wu, Yunzhu Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/142cdba4b8d1e03f9ee131ac86bb0afc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/142cdba4b8d1e03f9ee131ac86bb0afc-Abstract-Conference.html)

**Abstract**:

Learning predictive models from observations using deep neural networks (DNNs) is a promising new approach to many real-world planning and control problems. However, common DNNs are too unstructured for effective planning, and current control methods typically rely on extensive sampling or local gradient descent. In this paper, we propose a new framework for integrated model learning and predictive control that is amenable to efficient optimization algorithms. Specifically, we start with a ReLU neural model of the system dynamics and, with minimal losses in prediction accuracy, we gradually sparsify it by removing redundant neurons. This discrete sparsification process is approximated as a continuous problem, enabling an end-to-end optimization of both the model architecture and the weight parameters. The sparsified model is subsequently used by a mixed-integer predictive controller, which represents the neuron activations as binary variables and employs efficient branch-and-bound algorithms. Our framework is applicable to a wide variety of DNNs, from simple multilayer perceptrons to complex graph neural dynamics. It can efficiently handle tasks involving complicated contact dynamics, such as object pushing, compositional object sorting, and manipulation of deformable objects. Numerical and hardware experiments show that, despite the aggressive sparsification, our framework can deliver better closed-loop performance than existing state-of-the-art methods.

----

## [275] AmadeusGPT: a natural language interface for interactive animal behavioral analysis

**Authors**: *Shaokai Ye, Jessy Lauer, Mu Zhou, Alexander Mathis, Mackenzie W. Mathis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1456560769bbc38e4f8c5055048ea712-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1456560769bbc38e4f8c5055048ea712-Abstract-Conference.html)

**Abstract**:

The process of quantifying and analyzing animal behavior involves translating the naturally occurring descriptive language of their actions into machine-readable code. Yet, codifying behavior analysis is often challenging without deep understanding of animal behavior and technical machine learning knowledge. To limit this gap, we introduce AmadeusGPT: a natural language interface that turns natural language descriptions of behaviors into machine-executable code. Large-language models (LLMs) such as GPT3.5 and GPT4 allow for interactive language-based queries that are potentially well suited for making interactive behavior analysis. However, the comprehension capability of these LLMs is limited by the context window size, which prevents it from remembering distant conversations. To overcome the context window limitation, we implement a novel dual-memory mechanism to allow communication between short-term and long-term memory using symbols as context pointers for retrieval and saving. Concretely, users directly use language-based definitions of behavior and our augmented GPT develops code based on the core AmadeusGPT API, which contains machine learning, computer vision, spatio-temporal reasoning, and visualization modules. Users then can interactively refine results, and seamlessly add new behavioral modules as needed. We used the MABe 2022 behavior challenge tasks to benchmark AmadeusGPT and show excellent performance. Note, an end-user would not need to write any code to achieve this. Thus, collectively AmadeusGPT presents a novel way to merge deep biological knowledge, large-language models, and core computer vision modules into a more naturally intelligent system. Code and demos can be found at: https://github.com/AdaptiveMotorControlLab/AmadeusGPT

----

## [276] Provably Efficient Algorithm for Nonstationary Low-Rank MDPs

**Authors**: *Yuan Cheng, Jing Yang, Yingbin Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/145c28cd4b1df9b426990fd68045f4f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/145c28cd4b1df9b426990fd68045f4f7-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) under changing environment models many real-world applications via nonstationary Markov Decision Processes (MDPs), and hence gains considerable interest. However, theoretical studies on nonstationary MDPs in the literature have mainly focused on tabular and linear (mixture) MDPs, which do not capture the nature of unknown representation in deep RL. In this paper, we make the first effort to investigate nonstationary RL under episodic low-rank MDPs, where both transition kernels and rewards may vary over time, and the low-rank model contains unknown representation in addition to the linear state embedding function. We first propose a parameter-dependent policy optimization algorithm called PORTAL,and further improve PORTAL to its parameter-free version of Ada-PORTAL, which is able to tune its hyper-parameters adaptively without any prior knowledge of nonstationarity. For both algorithms, we provide upper bounds on the average dynamic suboptimality gap, which show that as long as the nonstationarity is not significantly large, PORTAL and Ada-PORTAL are sample-efficient and can achieve arbitrarily small average dynamic suboptimality gap with polynomial sample complexity.

----

## [277] Time-uniform confidence bands for the CDF under nonstationarity

**Authors**: *Paul Mineiro, Steven R. Howard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/148bbc25b934211d80435b5cad5a7198-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/148bbc25b934211d80435b5cad5a7198-Abstract-Conference.html)

**Abstract**:

Estimation of a complete univariate distribution from a sequence of observations is a useful primitive for both manual and automated decision making. This problem has received extensive attention in the i.i.d. setting, but the arbitrary data dependent setting remains largely unaddressed. We present computationally felicitous time-uniform and value-uniform bounds on the CDF of the running averaged conditional distribution of a sequence of real-valued random variables. Consistent with known impossibility results, our CDF bounds are always valid but sometimes trivial when the instance is too hard, and we give an instance-dependent convergence guarantee.  The importance-weighted extension is appropriate for estimating complete counterfactual distributions of rewards given data from a randomized experiment, e.g., from an A/B test or a contextual bandit.

----

## [278] Risk-Averse Active Sensing for Timely Outcome Prediction under Cost Pressure

**Authors**: *Yuchao Qin, Mihaela van der Schaar, Changhee Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1498a03a04f9bcd3a7d44058fc5dc639-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1498a03a04f9bcd3a7d44058fc5dc639-Abstract-Conference.html)

**Abstract**:

Timely outcome prediction is essential in healthcare to enable early detection and intervention of adverse events. However, in longitudinal follow-ups to patients' health status, cost-efficient acquisition of patient covariates is usually necessary due to the significant expense involved in screening and lab tests. To balance the timely and accurate outcome predictions with acquisition costs, an effective active sensing strategy is crucial. In this paper, we propose a novel risk-averse active sensing approach RAS that addresses the composite decision problem of when to conduct the acquisition and which measurements to make. Our approach decomposes the policy into two sub-policies: acquisition scheduler and feature selector, respectively. Moreover, we introduce a novel risk-aversion training strategy to focus on the underrepresented subgroup of high-risk patients for whom timely and accurate prediction of disease progression is of greater value. Our method outperforms baseline active sensing approaches in experiments with both synthetic and real-world datasets, and we illustrate the significance of our policy decomposition and the necessity of a risk-averse sensing policy through case studies.

----

## [279] Single-Pass Pivot Algorithm for Correlation Clustering. Keep it simple!

**Authors**: *Konstantin Makarychev, Sayak Chakrabarty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/149ad6e32c08b73a3ecc3d11977fcc47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/149ad6e32c08b73a3ecc3d11977fcc47-Abstract-Conference.html)

**Abstract**:

We show that a simple single-pass semi-streaming variant of the Pivot algorithm for Correlation Clustering gives a (3+eps)-approximation using O(n/eps) words of memory. This is a slight improvement over the recent results of Cambus, Kuhn, Lindy, Pai, and Uitto, who gave a (3+eps)-approximation using O(n log n) words of memory, and Behnezhad, Charikar, Ma, and Tan, who gave a 5-approximation using O(n) words of memory. One of the main contributions of our paper is that the algorithm and its analysis are simple and easy to understand.

----

## [280] SPACE: Single-round Participant Amalgamation for Contribution Evaluation in Federated Learning

**Authors**: *Yi-Chung Chen, Hsi-Wen Chen, Shun-Gui Wang, Ming-Syan Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/14a812fa4b6bf244d055e37a7cd2f557-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/14a812fa4b6bf244d055e37a7cd2f557-Abstract-Conference.html)

**Abstract**:

The evaluation of participant contribution in federated learning (FL) has recently gained significant attention due to its applicability in various domains, such as incentive mechanisms, robustness enhancement, and client selection. Previous approaches have predominantly relied on the widely adopted Shapley value for participant evaluation. However, the computation of the Shapley value is expensive, despite using techniques like gradient-based model reconstruction and truncating unnecessary evaluations. Therefore, we present an efficient approach called Single-round Participants Amalgamation for Contribution Evaluation (SPACE). SPACE incorporates two novel components, namely Federated Knowledge Amalgamation and Prototype-based Model Evaluation to reduce the evaluation effort by eliminating the dependence on the size of the validation set and enabling participant evaluation within a single communication round. Experimental results demonstrate that SPACE outperforms state-of-the-art methods in terms of both running time and Pearsonâ€™s Correlation Coefficient (PCC). Furthermore, extensive experiments conducted on applications, client reweighting, and client selection highlight the effectiveness of SPACE. The code is available at https://github.com/culiver/SPACE.

----

## [281] SAME: Uncovering GNN Black Box with Structure-aware Shapley-based Multipiece Explanations

**Authors**: *Ziyuan Ye, Rihan Huang, Qilin Wu, Quanying Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/14cdc9013d80338bf81483a7736ea05c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/14cdc9013d80338bf81483a7736ea05c-Abstract-Conference.html)

**Abstract**:

Post-hoc explanation techniques on graph neural networks (GNNs) provide economical solutions for opening the black-box graph models without model retraining. Many GNN explanation variants have achieved state-of-the-art explaining results on a diverse set of benchmarks, while they rarely provide theoretical analysis for their inherent properties and explanatory capability. In this work, we propose $\underline{\text{S}}$tructure-$\underline{\text{A}}$ware Shapley-based $\underline{\text{M}}$ultipiece $\underline{\text{E}}$xplanation (SAME) method to address the structure-aware feature interactions challenges for GNNs explanation. Specifically, SAME leverages an expansion-based Monte Carlo tree search to explore the multi-grained structure-aware connected substructure. Afterward, the explanation results are encouraged to be informative of the graph properties by optimizing the combination of distinct single substructures. With the consideration of fair feature interactions in the process of investigating multiple connected important substructures, the explanation provided by SAME has the potential to be as explainable as the theoretically optimal explanation obtained by the Shapley value within polynomial time. Extensive experiments on real-world and synthetic benchmarks show that SAME improves the previous state-of-the-art fidelity performance by 12.9\% on BBBP, 7.01\% on MUTAG, 42.3\% on Graph-SST2, 38.9\% on Graph-SST5, 11.3\% on BA-2Motifs and 18.2\% on BA-Shapes under the same testing condition. Code is available at https://github.com/same2023neurips/same.

----

## [282] Federated Learning with Client Subsampling, Data Heterogeneity, and Unbounded Smoothness: A New Algorithm and Lower Bounds

**Authors**: *Michael Crawshaw, Yajie Bao, Mingrui Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/14ecbfb2216bab76195b60bfac7efb1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/14ecbfb2216bab76195b60bfac7efb1f-Abstract-Conference.html)

**Abstract**:

We study the problem of Federated Learning (FL) under client subsampling and data heterogeneity with an objective function that has potentially unbounded smoothness. This problem is motivated by empirical evidence that the class of relaxed smooth functions, where the Lipschitz constant of the gradient scales linearly with the gradient norm, closely resembles the loss functions of certain neural networks such as recurrent neural networks (RNNs) with possibly exploding gradient. We introduce EPISODE++, the first algorithm to solve this problem. It maintains historical statistics for each client to construct control variates and decide clipping behavior for sampled clients in the current round. We prove that EPISODE++ achieves linear speedup in the number of participating clients, reduced communication rounds, and resilience to data heterogeneity. Our upper bound proof relies on novel techniques of recursively bounding the client updates under unbounded smoothness and client subsampling, together with a refined high probability analysis. In addition, we prove a lower bound showing that the convergence rate of a special case of clipped minibatch SGD (without randomness in the stochastic gradient and with randomness in client subsampling) suffers from an explicit dependence on the maximum gradient norm of the objective in a sublevel set, which may be large. This effectively demonstrates that applying gradient clipping to minibatch SGD in our setting does not eliminate the problem of exploding gradients.  Our lower bound is based on new constructions of hard instances tailored to client subsampling and a novel analysis of the trajectory of the algorithm in the presence of clipping. Lastly, we provide an experimental evaluation of EPISODE++ when training RNNs on federated text classification tasks, demonstrating that EPISODE++ outperforms strong baselines in FL. The code is available at https://github.com/MingruiLiu-ML-Lab/episode_plusplus.

----

## [283] NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics

**Authors**: *Anwar Said, Roza G. Bayrak, Tyler Derr, Mudassir Shabbir, Daniel Moyer, Catie Chang, Xenofon D. Koutsoukos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/14f656f21d09a4114666f60a45aab1aa-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/14f656f21d09a4114666f60a45aab1aa-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning provides a valuable tool for analyzing high-dimensional functional neuroimaging data, and is proving effective in predicting various neurological conditions, psychiatric disorders, and cognitive patterns. In functional magnetic resonance imaging (MRI) research, interactions between brain regions are commonly modeled using graph-based representations. The potency of graph machine learning methods has been established across myriad domains, marking a transformative step in data interpretation and predictive modeling. Yet, despite their promise, the transposition of these techniques to the neuroimaging domain has been  challenging due to the expansive number of potential preprocessing pipelines and the large parameter search space for graph-based dataset construction. In this paper, we introduce NeuroGraph, a collection of graph-based neuroimaging datasets, and demonstrated its utility for predicting multiple categories of behavioral and cognitive traits. We delve deeply into the dataset generation search space by crafting 35 datasets that encompass static and dynamic brain connectivity, running in excess of 15 baseline methods for benchmarking. Additionally, we provide generic frameworks for learning on both static and dynamic graphs. Our extensive experiments lead to several key observations. Notably, using correlation vectors as node features, incorporating larger number of regions of interest, and employing sparser graphs lead to improved performance. To foster further advancements in graph-based data driven neuroimaging analysis, we offer a comprehensive open-source Python package that includes the benchmark datasets, baseline implementations, model training, and standard evaluation.

----

## [284] Quantifying the Cost of Learning in Queueing Systems

**Authors**: *Daniel Freund, Thodoris Lykouris, Wentao Weng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1502957929fc4257dd1b6daf7d869c2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1502957929fc4257dd1b6daf7d869c2f-Abstract-Conference.html)

**Abstract**:

Queueing systems are widely applicable stochastic models with use cases in communication networks, healthcare, service systems, etc. Although their optimal control has been extensively studied, most existing approaches assume perfect knowledge of the system parameters. Of course, this assumption rarely holds in practice where there is parameter uncertainty, thus motivating a recent line of work on bandit learning for queueing systems. This nascent stream of research focuses on the asymptotic performance of the proposed algorithms. In this paper, we argue that an asymptotic metric, which focuses on late-stage performance, is insufficient to capture the intrinsic statistical complexity of learning in queueing systems which typically occurs in the early stage. Instead, we propose the Cost of Learning in Queueing (CLQ), a new metric that quantifies the maximum increase in time-averaged queue length caused by parameter uncertainty.We characterize the CLQ of a single-queue multi-server system, and then extend these results to multi-queue multi-server systems and networks of queues. In establishing our results, we propose a unified analysis framework for CLQ that bridges Lyapunov and bandit analysis, provides guarantees for a wide range of algorithms, and could be of independent interest.

----

## [285] One-Line-of-Code Data Mollification Improves Optimization of Likelihood-based Generative Models

**Authors**: *Ba-Hien Tran, Giulio Franzese, Pietro Michiardi, Maurizio Filippone*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1516a7f7507d5550db5c7f29e995ec8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1516a7f7507d5550db5c7f29e995ec8c-Abstract-Conference.html)

**Abstract**:

Generative Models (GMs) have attracted considerable attention due to their tremendous success in various domains, such as computer vision where they are capable to generate impressive realistic-looking images. Likelihood-based GMs are attractive due to the possibility to generate new data by a single model evaluation. However, they typically achieve lower sample quality compared to state-of-the-art score-based Diffusion Models (DMs). This paper provides a significant step in the direction of addressing this limitation. The idea is to borrow one of the strengths of score-based DMs, which is the ability to perform accurate density estimation in low-density regions and to address manifold overfitting by means of data mollification. We propose a view of data mollification within likelihood-based GMs as a continuation method, whereby the optimization objective smoothly transitions from simple-to-optimize to the original target. Crucially, data mollification can be implemented by adding one line of code in the optimization loop, and we demonstrate that this provides a boost in generation quality of likelihood-based GMs, without computational overheads. We report results on real-world image data sets and UCI benchmarks with popular likelihood-based GMs, including variants of variational autoencoders and normalizing flows, showing large improvements in FID score and density estimation.

----

## [286] FLSL: Feature-level Self-supervised Learning

**Authors**: *Qing Su, Anton Netchaev, Hai Li, Shihao Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15212bd2265c4a3ab0dbc1b1982c1b69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15212bd2265c4a3ab0dbc1b1982c1b69-Abstract-Conference.html)

**Abstract**:

Current self-supervised learning (SSL) methods (e.g., SimCLR, DINO, VICReg, MOCOv3) target primarily on representations at instance level and do not generalize well to dense prediction tasks, such as object detection and segmentation. Towards aligning SSL with dense predictions, this paper demonstrates for the first time the underlying mean-shift clustering process of Vision Transformers (ViT), which aligns well with natural image semantics (e.g., a world of objects and stuffs). By employing transformer for joint embedding and clustering, we propose a bi-level feature clustering SSL method, coined Feature-Level Self-supervised Learning (FLSL). We present the formal definition of the FLSL problem and construct the objectives from the mean-shift and k-means perspectives. We show that FLSL promotes remarkable semantic cluster representations and learns an embedding scheme amenable to intra-view and inter-view feature clustering. Experiments show that FLSL yields significant improvements in dense prediction tasks, achieving 44.9 (+2.8)% AP and 46.5% AP in object detection, as well as 40.8 (+2.3)% AP and 42.1% AP in instance segmentation on MS-COCO, using Mask R-CNN with ViT-S/16 and ViT-S/8 as backbone, respectively. FLSL consistently outperforms existing SSL methods across additional benchmarks, including UAV object detection on UAVDT, and video instance segmentation on DAVIS 2017. We conclude by presenting visualization and various ablation studies to better understand the success of FLSL. The source code is available at https://github.com/ISL-CV/FLSL.

----

## [287] FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning

**Authors**: *Dipam Goswami, Yuyang Liu, Bartlomiej Twardowski, Joost van de Weijer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15294ba2dcfb4521274f7aa1c26f4dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15294ba2dcfb4521274f7aa1c26f4dd4-Abstract-Conference.html)

**Abstract**:

Exemplar-free class-incremental learning (CIL) poses several challenges since it prohibits the rehearsal of data from previous tasks and thus suffers from catastrophic forgetting. Recent approaches to incrementally learning the classifier by freezing the feature extractor after the first task have gained much attention. In this paper, we explore prototypical networks for CIL, which generate new class prototypes using the frozen feature extractor and classify the features based on the Euclidean distance to the prototypes. In an analysis of the feature distributions of classes, we show that classification based on Euclidean metrics is successful for jointly trained features. However, when learning from non-stationary data, we observe that the Euclidean metric is suboptimal and that feature distributions are heterogeneous. To address this challenge, we revisit the anisotropic Mahalanobis distance for CIL. In addition, we empirically show that modeling the feature covariance relations is better than previous attempts at sampling features from normal distributions and training a linear classifier. Unlike existing methods, our approach generalizes to both many- and few-shot CIL settings, as well as to domain-incremental settings. Interestingly, without updating the backbone network, our method obtains state-of-the-art results on several standard continual learning benchmarks. Code is available at https://github.com/dipamgoswami/FeCAM.

----

## [288] Learning non-Markovian Decision-Making from State-only Sequences

**Authors**: *Aoyang Qin, Feng Gao, Qing Li, Song-Chun Zhu, Sirui Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/154926e0b66e2b2a8c1120852f31a12d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/154926e0b66e2b2a8c1120852f31a12d-Abstract-Conference.html)

**Abstract**:

Conventional imitation learning assumes access to the actions of demonstrators, but these motor signals are often non-observable in naturalistic settings. Additionally, sequential decision-making behaviors in these settings can deviate from the assumptions of a standard Markov Decision Process (MDP). To address these challenges, we explore deep generative modeling of state-only sequences with non-Markov Decision Process (nMDP), where the policy is an energy-based prior in the latent space of the state transition generator. We develop maximum likelihood estimation to achieve model-based imitation, which involves short-run MCMC sampling from the prior and importance sampling for the posterior. The learned model enables $\textit{decision-making as inference}$: model-free policy execution is equivalent to prior sampling, model-based planning is posterior sampling initialized from the policy. We demonstrate the efficacy of the proposed method in a prototypical path planning task with non-Markovian constraints and show that the learned model exhibits strong performances in challenging domains from the MuJoCo suite.

----

## [289] Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts

**Authors**: *Zeyang Zhang, Xin Wang, Ziwei Zhang, Zhou Qin, Weigao Wen, Hui Xue, Haoyang Li, Wenwu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/154b90fcc9ba3dee96779c05c3108908-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/154b90fcc9ba3dee96779c05c3108908-Abstract-Conference.html)

**Abstract**:

Dynamic graph neural networks (DyGNNs) currently struggle with handling distribution shifts that are inherent in dynamic graphs.Existing work on DyGNNs with out-of-distribution settings only focuses on the time domain, failing to handle cases involving distribution shifts in the spectral domain. In this paper, we discover that there exist cases with distribution shifts unobservable in the time domain while observable in the spectral domain, and propose to study distribution shifts on dynamic graphs in the spectral domain for the first time.However, this investigation poses two key challenges: i) it is non-trivial to capture different graph patterns that are driven by various frequency components entangled in the spectral domain; and ii) it remains unclear how to handle distribution shifts with the discovered spectral patterns. To address these challenges, we propose Spectral Invariant Learning for Dynamic Graphs under Distribution Shifts (SILD), which can handle distribution shifts on dynamic graphs by capturing and utilizing invariant and variant spectral patterns. Specifically, we first design a DyGNN with Fourier transform to obtain the ego-graph trajectory spectrums, allowing the mixed dynamic graph patterns to be transformed into separate frequency components. We then develop a disentangled spectrum mask to filter graph dynamics from various frequency components and discover the invariant and variant spectral patterns. Finally, we propose invariant spectral filtering, which encourages the model to rely on invariant patterns for generalization under distribution shifts. Experimental results on synthetic and real-world dynamic graph datasets demonstrate the superiority of our method for both node classification and link prediction tasks under distribution shifts.

----

## [290] Efficient Activation Function Optimization through Surrogate Modeling

**Authors**: *Garrett Bingham, Risto Miikkulainen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/154d63285d3ed7826e7f026c0b350d69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/154d63285d3ed7826e7f026c0b350d69-Abstract-Conference.html)

**Abstract**:

Carefully designed activation functions can improve the performance of neural networks in many machine learning tasks.  However, it is difficult for humans to construct optimal activation functions, and current activation function search algorithms are prohibitively expensive.  This paper aims to improve the state of the art through three steps: First, the benchmark datasets Act-Bench-CNN, Act-Bench-ResNet, and Act-Bench-ViT were created by training convolutional, residual, and vision transformer architectures from scratch with 2,913 systematically generated activation functions. Second, a characterization of the benchmark space was developed, leading to a new surrogate-based method for optimization. More specifically, the spectrum of the Fisher information matrix associated with the model's predictive distribution at initialization and the activation function's output distribution were found to be highly predictive of performance. Third, the surrogate was used to discover improved activation functions in several real-world tasks, with a surprising finding: a sigmoidal design that outperformed all other activation functions was discovered, challenging the status quo of always using rectifier nonlinearities in deep learning.  Each of these steps is a contribution in its own right; together they serve as a practical and theoretical foundation for further research on activation function optimization.

----

## [291] Data Market Design through Deep Learning

**Authors**: *Sai Srivatsa Ravindranath, Yanchen Jiang, David C. Parkes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1577ea3eaf8dacb99f64e4496c3ecddf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1577ea3eaf8dacb99f64e4496c3ecddf-Abstract-Conference.html)

**Abstract**:

The  data market design problem is a problem in economic theory to find a set of signaling schemes (statistical experiments) to maximize expected revenue to the information seller, where each experiment reveals some of the information known to a seller and has a corresponding price. Each buyer has their own decision to make in a world environment, and their subjective expected value for the information associated with a particular experiment comes from the improvement in this decision and depends on their prior and value for different outcomes. In a setting with multiple buyers, a buyer's expected value for an experiment may also depend on the information sold to others. We introduce the application of deep learning for the design of revenue-optimal data markets, looking to expand the frontiers of what can be understood and achieved. Relative to earlier work on deep learning for auction design, we must learn signaling schemes rather than allocation rules and handle  obedience constraints  — these arising from modeling the downstream actions of buyers — in addition to incentive constraints on bids.  Our experiments demonstrate that this new deep learning framework can almost precisely replicate all known solutions from theory, expand to more complex settings, and be used to establish the optimality of new designs for data markets and make conjectures in regard to the structure of optimal designs.

----

## [292] When Visual Prompt Tuning Meets Source-Free Domain Adaptive Semantic Segmentation

**Authors**: *Xinhong Ma, Yiming Wang, Hao Liu, Tianyu Guo, Yunhe Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/157c30da6a988e1cbef2095f7b9521db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/157c30da6a988e1cbef2095f7b9521db-Abstract-Conference.html)

**Abstract**:

Source-free domain adaptive semantic segmentation aims to adapt a pre-trained source model to the unlabeled target domain without accessing the private source data. Previous methods usually fine-tune the entire network, which suffers from expensive parameter tuning. To avoid this problem, we propose to utilize visual prompt tuning for parameter-efficient adaptation. However, the existing visual prompt tuning methods are unsuitable for source-free domain adaptive semantic segmentation due to the following two reasons: (1) Commonly used visual prompts like input tokens or pixel-level perturbations cannot reliably learn informative knowledge beneficial for semantic segmentation. (2) Visual prompts require sufficient labeled data to fill the gap between the pre-trained model and downstream tasks. To alleviate these problems, we propose a universal unsupervised visual prompt tuning  (Uni-UVPT) framework, which is applicable to various transformer-based backbones. Specifically, we first divide the source pre-trained backbone with frozen parameters into multiple stages, and propose a lightweight prompt adapter for progressively encoding informative knowledge into prompts and enhancing the generalization of target features between adjacent backbone stages. Cooperatively, a novel adaptive pseudo-label correction strategy with a multiscale consistency loss is designed to alleviate the negative effect of target samples with noisy pseudo labels and raise the capacity of visual prompts to spatial perturbations. Extensive experiments demonstrate that Uni-UVPT achieves state-of-the-art performance on GTA5 $\to$ Cityscapes and SYNTHIA $\to$ Cityscapes tasks and can serve as a universal and parameter-efficient framework for large-model unsupervised knowledge transfer. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/uni-uvpt and https://github.com/huawei-noah/noah-research/tree/master/uni-uvpt.

----

## [293] Benchmarking and Analyzing 3D-aware Image Synthesis with a Modularized Codebase

**Authors**: *Qiuyu Wang, Zifan Shi, Kecheng Zheng, Yinghao Xu, Sida Peng, Yujun Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1585da86b5a3c4fb15520a2b3682051f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1585da86b5a3c4fb15520a2b3682051f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Despite the rapid advance of 3D-aware image synthesis, existing studies usually adopt a mixture of techniques and tricks, leaving it unclear how each part contributes to the final performance in terms of generality. Following the most popular and effective paradigm in this field, which incorporates a neural radiance field (NeRF) into the generator of a generative adversarial network (GAN), we builda well-structured codebase through modularizing the generation process. Such a design allows researchers to develop and replace each module independently, and hence offers an opportunity to fairly compare various approaches and recognize their contributions from the module perspective. The reproduction of a range of cutting-edge algorithms demonstrates the availability of our modularized codebase. We also perform a variety of in-depth analyses, such as the comparison across different types of point feature, the necessity of the tailing upsampler in the generator, the reliance on the camera pose prior, etc., which deepen our understanding of existing methods and point out some further directions of the research work. Code and models will be made publicly available to facilitate the development and evaluation of this field.

----

## [294] RL-ViGen: A Reinforcement Learning Benchmark for Visual Generalization

**Authors**: *Zhecheng Yuan, Sizhe Yang, Pu Hua, Can Chang, Kaizhe Hu, Huazhe Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15c9f64ec172b046470d2a4d2b7669fc-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/15c9f64ec172b046470d2a4d2b7669fc-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Visual Reinforcement Learning (Visual RL), coupled with high-dimensional observations, has consistently confronted the long-standing challenge of out-of-distribution generalization. Despite the focus on algorithms aimed at resolving visual generalization problems, we argue that the devil is in the existing benchmarks as they are restricted to isolated tasks and generalization categories, undermining a comprehensive evaluation of agents' visual generalization capabilities. To bridge this gap, we introduce RL-ViGen: a novel Reinforcement Learning Benchmark for Visual Generalization, which contains diverse tasks and a wide spectrum of generalization types, thereby facilitating the derivation of more reliable conclusions. Furthermore, RL-ViGen incorporates the latest generalization visual RL algorithms into a unified framework, under which the experiment results indicate that no single existing algorithm has prevailed universally across tasks. Our aspiration is that Rl-ViGen will serve as a catalyst in this area, and lay a foundation for the future creation of universal visual generalization RL agents suitable for real-world scenarios.  Access to our code and implemented algorithms is provided at https://gemcollector.github.io/RL-ViGen/.

----

## [295] DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method

**Authors**: *Ahmed Khaled, Konstantin Mishchenko, Chi Jin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15ce36d35622f126f38e90167de1a350-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15ce36d35622f126f38e90167de1a350-Abstract-Conference.html)

**Abstract**:

This paper proposes a new easy-to-implement parameter-free gradient-based optimizer: DoWG (Distance over Weighted Gradients). We prove that DoWG is efficient---matching the convergence rate of optimally tuned gradient descent in convex optimization up to a logarithmic factor without tuning any parameters, and universal---automatically adapting to both smooth and nonsmooth problems. While popular algorithms following the AdaGrad framework compute a running average of the squared gradients, DoWG maintains a new distance-based weighted version of the running average, which is crucial to achieve the desired properties. To complement our theory, we also show empirically that DoWG trains at the edge of stability, and validate its effectiveness on practical machine learning tasks.

----

## [296] Multitask Learning with No Regret: from Improved Confidence Bounds to Active Learning

**Authors**: *Pier Giuseppe Sessa, Pierre Laforgue, Nicolò Cesa-Bianchi, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15d15045f93b44d933a260b249608d43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15d15045f93b44d933a260b249608d43-Abstract-Conference.html)

**Abstract**:

Multitask learning is a powerful framework that enables one to simultaneously learn multiple related tasks by sharing information between them. Quantifying uncertainty in the estimated tasks is of pivotal importance for many downstream applications, such as online or active learning. In this work, we provide novel confidence intervals for multitask regression in the challenging agnostic setting, i.e., when neither the similarity between tasks nor the tasks' features are available to the learner. The obtained intervals do not require i.i.d. data and can be directly applied to bound the regret in online learning. Through a refined analysis of the multitask information gain, we obtain new regret guarantees that, depending on a task similarity parameter, can significantly improve over treating tasks independently. We further propose a novel online learning algorithm that achieves such improved regret without knowing this parameter in advance, i.e., automatically adapting to task similarity. As a second key application of our results, we introduce a novel multitask active learning setup where several tasks must be simultaneously optimized, but only one of them can be queried for feedback by the learner at each round. For this problem, we design a no-regret algorithm that uses our confidence intervals to decide which task should be queried. Finally, we empirically validate our bounds and algorithms on synthetic and real-world (drug discovery) data.

----

## [297] Posterior Sampling with Delayed Feedback for Reinforcement Learning with Linear Function Approximation

**Authors**: *Nikki Lijing Kuang, Ming Yin, Mengdi Wang, Yu-Xiang Wang, Yian Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15d3d4a4bd808605e3a3c1ea0fd0eba4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15d3d4a4bd808605e3a3c1ea0fd0eba4-Abstract-Conference.html)

**Abstract**:

Recent studies in reinforcement learning (RL) have made significant progress by leveraging function approximation to alleviate the sample complexity hurdle for better performance. Despite the success, existing provably efficient algorithms typically rely on the accessibility of immediate feedback upon taking actions. The failure to account for the impact of delay in observations can significantly degrade the performance of real-world systems due to the regret blow-up. In this work, we tackle the challenge of delayed feedback in RL with linear function approximation by employing posterior sampling, which has been shown to empirically outperform the popular UCB algorithms in a wide range of regimes. We first introduce \textit{Delayed-PSVI}, an optimistic value-based algorithm that effectively explores the value function space via noise perturbation with posterior sampling. We provide the first analysis for posterior sampling algorithms with delayed feedback in RL and show our algorithm achieves $\widetilde{O}(\sqrt{d^3H^3 T} + d^2H^2 \mathbb{E}[\tau])$ worst-case regret in the presence of unknown stochastic delays. Here $\mathbb{E}[\tau]$ is the expected delay. To further improve its computational efficiency and to expand its applicability in high-dimensional RL problems, we incorporate a gradient-based approximate sampling scheme via Langevin dynamics for \textit{Delayed-LPSVI}, which maintains the same order-optimal regret guarantee with $\widetilde{O}(dHK)$ computational cost. Empirical evaluations are performed to demonstrate the statistical and computational efficacy of our algorithms.

----

## [298] Macro Placement by Wire-Mask-Guided Black-Box Optimization

**Authors**: *Yunqi Shi, Ke Xue, Song Lei, Chao Qian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15d6717f8bb33b3a74df26ce1eee0b9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15d6717f8bb33b3a74df26ce1eee0b9a-Abstract-Conference.html)

**Abstract**:

The development of very large-scale integration (VLSI) technology has posed new challenges for electronic design automation (EDA) techniques in chip floorplanning. During this process, macro placement is an important subproblem, which tries to determine the positions of all macros with the aim of minimizing half-perimeter wirelength (HPWL) and avoiding overlapping. Previous methods include packing-based, analytical and reinforcement learning methods. In this paper, we propose a new black-box optimization (BBO) framework (called WireMask-BBO) for macro placement, by using a wire-mask-guided greedy procedure for objective evaluation. Equipped with different BBO algorithms, WireMask-BBO empirically achieves significant improvements over previous methods, i.e., achieves significantly shorter HPWL by using much less time. Furthermore, it can fine-tune existing placements by treating them as initial solutions, which can bring up to 50% improvement in HPWL. WireMask-BBO has the potential to significantly improve the quality and efficiency of chip floorplanning, which makes it appealing to researchers and practitioners in EDA and will also promote the application of BBO. Our code is available at https://github.com/lamda-bbo/WireMask-BBO.

----

## [299] Reconciling Competing Sampling Strategies of Network Embedding

**Authors**: *Yuchen Yan, Baoyu Jing, Lihui Liu, Ruijie Wang, Jinning Li, Tarek F. Abdelzaher, Hanghang Tong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15dc2344ea9bdc01ffb8bb2d692e4018-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15dc2344ea9bdc01ffb8bb2d692e4018-Abstract-Conference.html)

**Abstract**:

Network embedding plays a significant role in a variety of applications. To capture the topology of the network, most of the existing network embedding algorithms follow a sampling training procedure, which maximizes the similarity (e.g., embedding vectors' dot product) between positively sampled node pairs and minimizes the similarity between negatively sampled node pairs in the embedding space. Typically, close node pairs function as positive samples while distant node pairs are usually considered as negative samples. However, under different or even competing sampling strategies, some methods champion sampling distant node pairs as positive samples to encapsulate longer distance information in link prediction, whereas others advocate adding close nodes into the negative sample set to boost the performance of node recommendation. In this paper, we seek to understand the intrinsic relationships between these competing strategies. To this end, we identify two properties (discrimination and monotonicity) that given any node pair proximity distribution, node embeddings should embrace.Moreover, we quantify the empirical error of the trained similarity score w.r.t. the sampling strategy, which leads to an important finding that the discrimination property and the monotonicity property for all node pairs can not be satisfied simultaneously in real-world applications. Guided by such analysis, a simple yet novel model (SENSEI) is proposed, which seamlessly fulfills the discrimination property and the partial monotonicity within the top-$K$ ranking list. Extensive experiments show that SENSEI outperforms the state-of-the-arts in plain network embedding.

----

## [300] Zero-shot causal learning

**Authors**: *Hamed Nilforoshan, Michael Moor, Yusuf Roohani, Yining Chen, Anja Surina, Michihiro Yasunaga, Sara Oblak, Jure Leskovec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15ddb1773510075ef44981cdb204330b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15ddb1773510075ef44981cdb204330b-Abstract-Conference.html)

**Abstract**:

Predicting how different interventions will causally affect a specific individual is important in a variety of domains such as personalized medicine, public policy, and online marketing. There are a large number of methods to predict the effect of an existing intervention based on historical data from individuals who received it. However, in many settings it is important to predict the effects of novel interventions (e.g., a newly invented drug), which these methods do not address.Here, we consider zero-shot causal learning: predicting the personalized effects of a novel intervention. We propose CaML, a causal meta-learning framework which formulates the personalized prediction of each intervention's effect as a task. CaML trains a single meta-model across thousands of tasks, each constructed by sampling an intervention, its recipients, and its nonrecipients. By leveraging both intervention information (e.g., a drug's attributes) and individual features (e.g., a patient's history), CaML is able to predict the personalized effects of novel interventions that do not exist at the time of training. Experimental results on real world datasets in large-scale medical claims and cell-line perturbations demonstrate the effectiveness of our approach. Most strikingly, CaML's zero-shot predictions outperform even strong baselines trained directly on data from the test interventions.

----

## [301] Learning Modulated Transformation in GANs

**Authors**: *Ceyuan Yang, Qihang Zhang, Yinghao Xu, Jiapeng Zhu, Yujun Shen, Bo Dai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15f1dbc086bfd94d8c32557b573cbe18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15f1dbc086bfd94d8c32557b573cbe18-Abstract-Conference.html)

**Abstract**:

The success of style-based generators largely benefits from style modulation,which helps take care of the cross-instance variation within data. However, theinstance-wise stochasticity is typically introduced via regular convolution, wherekernels interact with features at some fixed locations, limiting its capacity formodeling geometric variation. To alleviate this problem, we equip the generatorin generative adversarial networks (GANs) with a plug-and-play module, termedas modulated transformation module (MTM). This module predicts spatial offsetsunder the control of latent codes, based on which the convolution operation canbe applied at variable locations for different instances, and hence offers the modelan additional degree of freedom to handle geometry deformation. Extensiveexperiments suggest that our approach can be faithfully generalized to variousgenerative tasks, including image generation, 3D-aware image synthesis, andvideo generation, and get compatible with state-of-the-art frameworks withoutany hyper-parameter tuning. It is noteworthy that, towards human generation onthe challenging TaiChi dataset, we improve the FID of StyleGAN3 from 21.36 to13.60, demonstrating the efficacy of learning modulated geometry transformation.Code and models are available at https://github.com/limbo0000/mtm.

----

## [302] Active Negative Loss Functions for Learning with Noisy Labels

**Authors**: *Xichen Ye, Xiaoqiang Li, Songmin Dai, Tong Liu, Yan Sun, Weiqin Tong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15f4cefb0e143c7ad9d40e879b0a9d0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15f4cefb0e143c7ad9d40e879b0a9d0c-Abstract-Conference.html)

**Abstract**:

Robust loss functions are essential for training deep neural networks in the presence of noisy labels. Some robust loss functions use Mean Absolute Error (MAE) as its necessary component. For example, the recently proposed Active Passive Loss (APL) uses MAE as its passive loss function. However, MAE treats every sample equally, slows down the convergence and can make training difficult. In this work, we propose a new class of theoretically robust passive loss functions different from MAE, namely Normalized Negative Loss Functions (NNLFs), which focus more on memorized clean samples. By replacing the MAE in APL with our proposed NNLFs, we improve APL and propose a new framework called Active Negative Loss (ANL). Experimental results on benchmark and real-world datasets demonstrate that the new set of loss functions created by our ANL framework can outperform state-of-the-art methods. The code is available athttps://github.com/Virusdoll/Active-Negative-Loss.

----

## [303] Compositional Generalization from First Principles

**Authors**: *Thaddäus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, Wieland Brendel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/15f6a10899f557ce53fe39939af6f930-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/15f6a10899f557ce53fe39939af6f930-Abstract-Conference.html)

**Abstract**:

Leveraging the compositional nature of our world to expedite learning and facilitate generalization is a hallmark of human perception. In machine learning, on the other hand, achieving compositional generalization has proven to be an elusive goal, even for models with explicit compositional priors. To get a better handle on compositional generalization, we here approach it from the bottom up: Inspired by identifiable representation learning, we investigate compositionality as a property of the data-generating process rather than the data itself. This reformulation enables us to derive mild conditions on only the support of the training distribution and the model architecture, which are sufficient for compositional generalization. We further demonstrate how our theoretical framework applies to real-world scenarios and validate our findings empirically. Our results set the stage for a principled theoretical study of compositional generalization.

----

## [304] PanoGRF: Generalizable Spherical Radiance Fields for Wide-baseline Panoramas

**Authors**: *Zheng Chen, Yan-Pei Cao, Yuan-Chen Guo, Chen Wang, Ying Shan, Song-Hai Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16049e0c3f47899091ac46f8b3afb178-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16049e0c3f47899091ac46f8b3afb178-Abstract-Conference.html)

**Abstract**:

Achieving an immersive experience enabling users to explore virtual environments with six degrees of freedom (6DoF) is essential for various applications such as virtual reality (VR). Wide-baseline panoramas are commonly used in these applications to reduce network bandwidth and storage requirements. However, synthesizing novel views from these panoramas remains a key challenge. Although existing neural radiance field methods can produce photorealistic views under narrow-baseline and dense image captures, they tend to overfit the training views when dealing with wide-baseline panoramas due to the difficulty in learning accurate geometry from sparse $360^{\circ}$ views. To address this problem, we propose PanoGRF, Generalizable Spherical Radiance Fields for Wide-baseline Panoramas, which construct spherical radiance fields incorporating $360^{\circ}$ scene priors. Unlike generalizable radiance fields trained on perspective images, PanoGRF avoids the information loss from panorama-to-perspective conversion and directly aggregates geometry and appearance features of 3D sample points from each panoramic view based on spherical projection. Moreover, as some regions of the panorama are only visible from one view while invisible from others under wide baseline settings, PanoGRF incorporates $360^{\circ}$ monocular depth priors into spherical depth estimation to improve the geometry features. Experimental results on multiple panoramic datasets demonstrate that PanoGRF significantly outperforms state-of-the-art generalizable view synthesis methods for wide-baseline panoramas (e.g., OmniSyn) and perspective images (e.g., IBRNet, NeuRay).

----

## [305] A Heat Diffusion Perspective on Geodesic Preserving Dimensionality Reduction

**Authors**: *Guillaume Huguet, Alexander Tong, Edward De Brouwer, Yanlei Zhang, Guy Wolf, Ian Adelstein, Smita Krishnaswamy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16063a1c0f0cddd4894585cf44cebb2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16063a1c0f0cddd4894585cf44cebb2c-Abstract-Conference.html)

**Abstract**:

Diffusion-based manifold learning methods have proven useful in representation learning and dimensionality reduction of modern high dimensional, high throughput, noisy datasets. Such datasets are especially present in fields like biology and physics. While it is thought that these methods preserve underlying manifold structure of data by learning a proxy for geodesic distances, no specific theoretical links have been established. Here, we establish such a link via results in Riemannian geometry explicitly connecting heat diffusion to manifold distances. In this process, we also formulate a more general heat kernel based manifold embedding method that we call heat geodesic embeddings. This novel perspective makes clearer the choices available in manifold learning and denoising. Results show that our method outperforms existing state of the art in preserving ground truth manifold distances,  and preserving cluster structure in toy datasets. We also showcase our method on single cell RNA-sequencing datasets with both continuum and cluster structure, where our method enables interpolation of withheld timepoints of data. Finally, we show that parameters of our more general method can be configured to give results similar to PHATE (a state-of-the-art diffusion based manifold learning method) as well as SNE (an attraction/repulsion neighborhood based method that forms the basis of t-SNE).

----

## [306] Finite-Time Analysis of Single-Timescale Actor-Critic

**Authors**: *Xuyang Chen, Lin Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/160adf2dc118a920e7858484b92a37d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/160adf2dc118a920e7858484b92a37d8-Abstract-Conference.html)

**Abstract**:

Actor-critic methods have achieved significant success in many challenging applications. However, its finite-time convergence is still poorly understood in the most practical single-timescale form. Existing works on analyzing single-timescale actor-critic have been limited to i.i.d. sampling or tabular setting for simplicity. We investigate the more practical online single-timescale actor-critic algorithm on continuous state space, where the critic assumes linear function approximation and updates with a single Markovian sample per actor step.  Previous analysis has been unable to establish the convergence for such a challenging scenario. We demonstrate that the online single-timescale actor-critic method provably finds an $\epsilon$-approximate stationary point with $\widetilde{\mathcal{O}}(\epsilon^{-2})$ sample complexity under standard assumptions, which can be further improved to $\mathcal{O}(\epsilon^{-2})$ under the i.i.d. sampling. Our novel framework systematically evaluates and controls the error propagation between the actor and critic. It offers a promising approach for analyzing other single-timescale reinforcement learning algorithms as well.

----

## [307] VanillaNet: the Power of Minimalism in Deep Learning

**Authors**: *Hanting Chen, Yunhe Wang, Jianyuan Guo, Dacheng Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16336d94a5ffca8de019087ab7fe403f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16336d94a5ffca8de019087ab7fe403f-Abstract-Conference.html)

**Abstract**:

At the heart of foundation models is the philosophy of "more is different", exemplified by the astonishing success in computer vision and natural language processing. However, the challenges of optimization and inherent complexity of transformer models call for a paradigm shift towards simplicity. In this study, we introduce VanillaNet, a neural network architecture that embraces elegance in design. By avoiding high depth, shortcuts, and intricate operations like self-attention, VanillaNet is refreshingly concise yet remarkably powerful. Each layer is carefully crafted to be compact and straightforward, with nonlinear activation functions pruned after training to restore the original architecture. VanillaNet overcomes the challenges of inherent complexity, making it ideal for resource-constrained environments. Its easy-to-understand and highly simplified architecture opens new possibilities for efficient deployment. Extensive experimentation demonstrates that VanillaNet delivers performance on par with renowned deep neural networks and vision transformers, showcasing the power of minimalism in deep learning. This visionary journey of VanillaNet has significant potential to redefine the landscape and challenge the status quo of foundation model, setting a new path for elegant and effective model design. Pre-trained models and codes are available at https://github.com/huawei-noah/VanillaNet and https://gitee.com/mindspore/models/tree/master/research/cv/vanillanet

----

## [308] Probabilistic inverse optimal control for non-linear partially observable systems disentangles perceptual uncertainty and behavioral costs

**Authors**: *Dominik Straub, Matthias Schultheis, Heinz Koeppl, Constantin A. Rothkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16347f6e665376fd9a9a290dbfe0db5b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16347f6e665376fd9a9a290dbfe0db5b-Abstract-Conference.html)

**Abstract**:

Inverse optimal control can be used to characterize behavior in sequential decision-making tasks. Most existing work, however, is limited to fully observable or linear systems, or requires the action signals to be known. Here, we introduce a probabilistic approach to inverse optimal control for partially observable stochastic non-linear systems with unobserved action signals, which unifies previous approaches to inverse optimal control with maximum causal entropy formulations. Using an explicit model of the noise characteristics of the sensory and motor systems of the agent in conjunction with local linearization techniques, we derive an approximate likelihood function for the model parameters, which can be computed within a single forward pass. We present quantitative evaluations on stochastic and partially observable versions of two classic control tasks and two human behavioral tasks. Importantly, we show that our method can disentangle perceptual factors and behavioral costs despite the fact that epistemic and pragmatic actions are intertwined in sequential decision-making under uncertainty, such as in active sensing and active learning. The proposed method has broad applicability, ranging from imitation learning to sensorimotor neuroscience.

----

## [309] TIES-Merging: Resolving Interference When Merging Models

**Authors**: *Prateek Yadav, Derek Tam, Leshem Choshen, Colin A. Raffel, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1644c9af28ab7916874f6fd6228a9bcf-Abstract-Conference.html)

**Abstract**:

Transfer learning – i.e., further fine-tuning a pre-trained model on a downstream task – can confer significant advantages, including improved downstream performance, faster convergence, and better sample efficiency. These advantages have led to a proliferation of task-specific fine-tuned models, which typically can only perform a single task and do not benefit from one another. Recently, model merging techniques have emerged as a solution to combine multiple task-specific models into a single multitask model without performing additional training. However, existing merging methods often ignore the interference between parameters of different models, resulting in large performance drops when merging multiple models. In this paper, we demonstrate that prior merging techniques inadvertently lose valuable information due to two major sources of interference: (a) interference due to redundant parameter values and (b) disagreement on the sign of a given parameter’s values across models. To address this, we propose our method, TrIm, Elect Sign & Merge (TIES-Merging), which introduces three novel steps when merging models: (1) resetting parameters that only changed a small amount during fine-tuning, (2) resolving sign conflicts, and (3) merging only the parameters that are in alignment with the final agreed-upon sign. We find that TIES-Merging outperforms existing methods in diverse settings covering a range of modalities, domains, number of tasks, model sizes, architectures, and fine-tuning settings. We further analyze the impact of different types of interference on model parameters, highlight the importance of signs, and show that estimating the signs using the validation data could further improve performance.

----

## [310] 3D-IntPhys: Towards More Generalized 3D-grounded Visual Intuitive Physics under Challenging Scenes

**Authors**: *Haotian Xue, Antonio Torralba, Josh Tenenbaum, Dan Yamins, Yunzhu Li, Hsiao-Yu Tung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/164687cb815daae754d33364716e65e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/164687cb815daae754d33364716e65e6-Abstract-Conference.html)

**Abstract**:

Given a visual scene, humans have strong intuitions about how a scene can evolve over time under given actions. The intuition, often termed visual intuitive physics, is a critical ability that allows us to make effective plans to manipulate the scene to achieve desired outcomes without relying on extensive trial and error. In this paper, we present a framework capable of learning 3D-grounded visual intuitive physics models from videos of complex scenes with fluids. Our method is composed of a conditional Neural Radiance Field (NeRF)-style visual frontend and a 3D point-based dynamics prediction backend, using which we can impose strong relational and structural inductive bias to capture the structure of the underlying environment. Unlike existing intuitive point-based dynamics works that rely on the supervision of dense point trajectory from simulators, we relax the requirements and only assume access to multi-view RGB images and (imperfect) instance masks acquired using color prior. This enables the proposed model to handle scenarios where accurate point estimation and tracking are hard or impossible. We generate datasets including three challenging scenarios involving fluid, granular materials, and rigid objects in the simulation. The datasets do not include any dense particle information so most previous 3D-based intuitive physics pipelines can barely deal with that. We show our model can make long-horizon future predictions by learning from raw images and significantly outperforms models that do not employ an explicit 3D representation space. We also show that once trained, our model can achieve strong generalization in complex scenarios under extrapolate settings.

----

## [311] Entropy-based Training Methods for Scalable Neural Implicit Samplers

**Authors**: *Weijian Luo, Boya Zhang, Zhihua Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1646e34971facbcda3727d1dc28ab635-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1646e34971facbcda3727d1dc28ab635-Abstract-Conference.html)

**Abstract**:

Efficiently sampling from un-normalized target distributions is a fundamental problem in scientific computing and machine learning. Traditional approaches such as Markov Chain Monte Carlo (MCMC) guarantee asymptotically unbiased samples from such distributions but suffer from computational inefficiency, particularly when dealing with high-dimensional targets, as they require numerous iterations to generate a batch of samples. In this paper, we introduce an efficient and scalable neural implicit sampler that overcomes these limitations. The implicit sampler can generate large batches of samples with low computational costs by leveraging a neural transformation that directly maps easily sampled latent vectors to target samples without the need for iterative procedures. To train the neural implicit samplers, we introduce two novel methods: the KL training method and the Fisher training method. The former method minimizes the Kullback-Leibler divergence, while the latter minimizes the Fisher divergence between the sampler and the target distributions. By employing the two training methods, we effectively optimize the neural implicit samplers to learn and generate from the desired target distribution. To demonstrate the effectiveness, efficiency, and scalability of our proposed samplers, we evaluate them on three sampling benchmarks with different scales. These benchmarks include sampling from 2D targets, Bayesian inference, and sampling from high-dimensional energy-based models (EBMs). Notably, in the experiment involving high-dimensional EBMs, our sampler produces samples that are comparable to those generated by MCMC-based methods while being more than 100 times more efficient, showcasing the efficiency of our neural sampler. Besides the theoretical contributions and strong empirical performances, the proposed neural samplers and corresponding training methods will shed light on further research on developing efficient samplers for various applications beyond the ones explored in this study.

----

## [312] Direct Diffusion Bridge using Data Consistency for Inverse Problems

**Authors**: *Hyungjin Chung, Jeongsol Kim, Jong Chul Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/165b0e600b1721bd59526131eb061092-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/165b0e600b1721bd59526131eb061092-Abstract-Conference.html)

**Abstract**:

Diffusion model-based inverse problem solvers have shown impressive performance, but are limited in speed, mostly as they require reverse diffusion sampling starting from noise. Several recent works have tried to alleviate this problem by building a diffusion process, directly bridging the clean and the corrupted for specific inverse problems. In this paper, we first unify these existing works under the name Direct Diffusion Bridges (DDB), showing that while motivated by different theories, the resulting algorithms only differ in the choice of parameters. Then, we highlight a critical limitation of the current DDB framework, namely that it does not ensure data consistency. To address this problem, we propose a modified inference procedure that imposes data consistency without the need for fine-tuning. We term the resulting method data Consistent DDB (CDDB), which outperforms its inconsistent counterpart in terms of both perception and distortion metrics, thereby effectively pushing the Pareto-frontier toward the optimum. Our proposed method achieves state-of-the-art results on both evaluation criteria, showcasing its superiority over existing methods. Code is open-sourced here.

----

## [313] Mask Propagation for Efficient Video Semantic Segmentation

**Authors**: *Yuetian Weng, Mingfei Han, Haoyu He, Mingjie Li, Lina Yao, Xiaojun Chang, Bohan Zhuang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/167bcf2af2cd08fcf75b932022db0311-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/167bcf2af2cd08fcf75b932022db0311-Abstract-Conference.html)

**Abstract**:

Video Semantic Segmentation (VSS) involves assigning a semantic label to each pixel in a video sequence. Prior work in this field has demonstrated promising results by extending image semantic segmentation models to exploit temporal relationships across video frames; however, these approaches often incur significant computational costs. In this paper, we propose an efficient mask propagation framework for VSS, called MPVSS. Our approach first employs a strong query-based image segmentor on sparse key frames to generate accurate binary masks and class predictions. We then design a flow estimation module utilizing the learned queries to generate a set of segment-aware flow maps, each associated with a mask prediction from the key frame. Finally, the mask-flow pairs are warped to serve as the mask predictions for the non-key frames. By reusing predictions from key frames, we circumvent the need to process a large volume of video frames individually with resource-intensive segmentors, alleviating temporal redundancy and significantly reducing computational costs. Extensive experiments on VSPW and Cityscapes demonstrate that our mask propagation framework achieves SOTA accuracy and efficiency trade-offs. For instance, our best model with Swin-L backbone outperforms the SOTA MRCFA using MiT-B5 by 4.0% mIoU, requiring only 26% FLOPs on the VSPW dataset. Moreover, our framework reduces up to 4Ã— FLOPs compared to the per-frame Mask2Former baseline with only up to 2% mIoU degradation on the Cityscapes validation set. Code is available at https://github.com/ziplab/MPVSS.

----

## [314] Private Distribution Learning with Public Data: The View from Sample Compression

**Authors**: *Shai Ben-David, Alex Bie, Clément L. Canonne, Gautam Kamath, Vikrant Singhal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1687466683649e8bdcdec0e3f5c8de64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1687466683649e8bdcdec0e3f5c8de64-Abstract-Conference.html)

**Abstract**:

We study the problem of private distribution learning with access to public data. In this setup, which we refer to as *public-private learning*, the learner is given public and private samples drawn from an unknown distribution $p$ belonging to a class $\mathcal Q$, with the goal of outputting an estimate of $p$ while adhering to privacy constraints (here, pure differential privacy) only with respect to the private samples.     We show that the public-private learnability of a class $\mathcal Q$ is connected to the existence of a sample compression scheme for $\mathcal Q$, as well as to an intermediate notion we refer to as \emph{list learning}. Leveraging this connection: (1) approximately recovers previous results on Gaussians over $\mathbb R^d$; and (2) leads to new ones, including sample complexity upper bounds for arbitrary $k$-mixtures of Gaussians over $\mathbb R^d$, results for agnostic and distribution-shift resistant learners, as well as closure properties for public-private learnability under taking mixtures and products of distributions. Finally, via the connection to list learning, we show that for Gaussians in $\mathbb R^d$, at least $d$ public samples are necessary for private learnability, which is close to the known upper bound of $d+1$ public samples.

----

## [315] ChessGPT: Bridging Policy Learning and Language Modeling

**Authors**: *Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, Jun Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16b14e3f288f076e0ca73bdad6405f77-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/16b14e3f288f076e0ca73bdad6405f77-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

When solving decision-making tasks, humans typically depend on information from two key sources: (1) Historical policy data, which provides interaction replay from the environment, and (2) Analytical insights in natural language form, exposing the invaluable thought process or strategic considerations. Despite this, the majority of preceding research focuses on only one source: they either use historical replay exclusively to directly learn policy or value functions, or engaged in language model training utilizing mere language corpus. In this paper, we argue that a powerful autonomous agent should cover both sources. Thus, we propose ChessGPT, a GPT model bridging policy learning and language modeling by integrating data from these two sources in Chess games. Specifically, we build a large-scale game and language dataset related to chess. Leveraging the dataset, we showcase two model examples ChessCLIP and ChessGPT, integrating policy learning and language modeling. Finally, we propose a full evaluation framework for evaluating language model's chess ability. Experimental results validate our model and dataset's effectiveness. We open source our code, model, and dataset at https://github.com/waterhorse1/ChessGPT.

----

## [316] Fitting trees to 𝓁1-hyperbolic distances

**Authors**: *Joon-Hyeok Yim, Anna A. Gilbert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16bce4070c4e23434451b180348e3814-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16bce4070c4e23434451b180348e3814-Abstract-Conference.html)

**Abstract**:

Building trees to represent or to fit distances is a critical component of phylogenetic analysis, metric embeddings, approximation algorithms, geometric graph neural nets, and the analysis of hierarchical data. Much of the previous algorithmic work, however, has focused on generic metric spaces (i.e., those with no \emph{a priori} constraints). Leveraging several ideas from the mathematical analysis of hyperbolic geometry and geometric group theory, we study the tree fitting problem as finding the relation between the hyperbolicity (ultrametricity) vector and the error of tree (ultrametric) embedding. That is, we define a vector of hyperbolicity (ultrametric) values over all triples of points and compare the $\ell_p$ norms of this vector with the $\ell_q$ norm of the distortion of the best tree fit to the distances. This formulation allows us to define the average hyperbolicity (ultrametricity) in terms of a normalized $\ell_1$ norm of the hyperbolicity vector. Furthermore, we can interpret the classical tree fitting result of Gromov as a $p = q = \infty$ result. We present an algorithm \textsc{HCCRootedTreeFit} such that the $\ell_1$ error of the output embedding is analytically bounded in terms of the $\ell_1$-norm of the hyperbolicity vector (i.e., $p = q = 1$) and that this result is tight. Furthermore, this algorithm has significantly different theoretical and empirical performance as compared to Gromov's result and related algorithms. Finally, we show using \textsc{HCCRootedTreeFit} and related tree fitting algorithms, that supposedly standard data sets for hierarchical data analysis and geometric graph neural networks have radically different tree fits than those of synthetic, truly tree-like data sets, suggesting that a much more refined analysis of these standard data sets is called for.

----

## [317] Learning Robust Statistics for Simulation-based Inference under Model Misspecification

**Authors**: *Daolang Huang, Ayush Bharti, Amauri H. Souza, Luigi Acerbi, Samuel Kaski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16c5b4102a6b6eb061e502ce6736ad8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16c5b4102a6b6eb061e502ce6736ad8a-Abstract-Conference.html)

**Abstract**:

Simulation-based inference (SBI) methods such as approximate Bayesian computation (ABC),  synthetic likelihood, and neural posterior estimation (NPE) rely on simulating statistics to infer parameters of intractable likelihood models. However, such methods are known to yield untrustworthy and misleading inference outcomes under model misspecification, thus hindering their widespread applicability. In this work, we propose the first general approach to handle model misspecification that works across different classes of SBI methods. Leveraging the fact that the choice of statistics determines the degree of misspecification in SBI, we introduce a regularized loss function that penalizes those statistics that increase the mismatch between the data and the model. Taking NPE and ABC as use cases, we demonstrate the superior performance of our method on high-dimensional time-series models that are artificially misspecified. We also apply our method to real data from the field of radio propagation where the model is known to be misspecified. We show empirically that the method yields robust inference in misspecified scenarios, whilst still being accurate when the model is well-specified.

----

## [318] Block-State Transformers

**Authors**: *Jonathan Pilault, Mahan Fathi, Orhan Firat, Chris Pal, Pierre-Luc Bacon, Ross Goroshin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16ccd203e9e3696a7ab0dcf568316379-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16ccd203e9e3696a7ab0dcf568316379-Abstract-Conference.html)

**Abstract**:

State space models (SSMs) have shown impressive results on tasks that require modeling long-range dependencies and efficiently scale to long sequences owing to their subquadratic runtime complexity.Originally designed for continuous signals, SSMs have shown superior performance on a plethora of tasks, in vision and audio; however, SSMs still lag Transformer performance in Language Modeling tasks.In this work, we propose a hybrid layer named Block-State Transformer (BST), that internally combines an SSM sublayer for long-range contextualization, and a Block Transformer sublayer for short-term representation of sequences.We study three different, and completely parallelizable, variants that integrate SSMs and block-wise attention.We show that our model outperforms similar Transformer-based architectures on language modeling perplexity and generalizes to longer sequences. In addition, the Block-State Transformer demonstrates a more than tenfold increase in speed at the layer level compared to the Block-Recurrent Transformer when model parallelization is employed.

----

## [319] Explaining Predictive Uncertainty with Information Theoretic Shapley Values

**Authors**: *David S. Watson, Joshua O'Hara, Niek Tax, Richard Mudd, Ido Guy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/16e4be78e61a3897665fa01504e9f452-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/16e4be78e61a3897665fa01504e9f452-Abstract-Conference.html)

**Abstract**:

Researchers in explainable artificial intelligence have developed numerous methods for helping users understand the predictions of complex supervised learning models. By contrast, explaining the $\textit{uncertainty}$ of model outputs has received relatively little attention. We adapt the popular Shapley value framework to explain various types of predictive uncertainty, quantifying each feature's contribution to the conditional entropy of individual model outputs. We consider games with modified characteristic functions and find deep connections between the resulting Shapley values and fundamental quantities from information theory and conditional independence testing. We outline inference procedures for finite sample error rate control with provable guarantees, and implement efficient algorithms that perform well in a range of experiments on real and simulated data. Our method has applications to covariate shift detection, active learning, feature selection, and active feature-value acquisition.

----

## [320] Learning to Taste: A Multimodal Wine Dataset

**Authors**: *Thoranna Bender, Simon Møe Sørensen, Alireza Kashani, Kristjan Eldjarn Hjorleifsson, Grethe Hyldig, Søren Hauberg, Serge J. Belongie, Frederik Warburg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/170035f97007fdfa665880107b56f384-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/170035f97007fdfa665880107b56f384-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present WineSensed, a large multimodal wine dataset for studying the relations between visual perception, language, and flavor. The dataset encompasses 897k images of wine labels and 824k reviews of wines curated from the Vivino platform. It has over 350k unique vintages, annotated with year, region, rating, alcohol percentage, price, and grape composition. We obtained fine-grained flavor annotations on a subset by conducting a wine-tasting experiment with 256 participants who were asked to rank wines based on their similarity in flavor, resulting in more than 5k pairwise flavor distances. We propose a low-dimensional concept embedding algorithm that combines human experience with automatic machine similarity kernels. We demonstrate that this shared concept embedding space improves upon separate embedding spaces for coarse flavor classification  (alcohol percentage, country, grape, price, rating) and representing human perception of flavor.

----

## [321] CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning

**Authors**: *Charles Guille-Escuret, Pau Rodríguez, David Vázquez, Ioannis Mitliagkas, João Monteiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1700ad4e6252e8f2955909f96367b34d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1700ad4e6252e8f2955909f96367b34d-Abstract-Conference.html)

**Abstract**:

Handling out-of-distribution (OOD) samples has become a major stake in the real-world deployment of machine learning systems. This work explores the use of self-supervised contrastive learning to the simultaneous detection of two types of OOD samples: unseen classes and adversarial perturbations. First, we pair self-supervised contrastive learning with the maximum mean discrepancy (MMD) two-sample test. This approach enables us to robustly test whether two independent sets of samples originate from the same distribution, and we demonstrate its effectiveness by discriminating between CIFAR-10 and CIFAR-10.1 with higher confidence than previous work. Motivated by this success, we introduce CADet (Contrastive Anomaly Detection), a novel method for OOD detection of single samples. CADet draws inspiration from MMD, but leverages the similarity between contrastive transformations of a same sample. CADet outperforms existing adversarial detection methods in identifying adversarially perturbed samples on ImageNet and achieves comparable performance to unseen label detection methods on two challenging benchmarks: ImageNet-O and iNaturalist. Significantly, CADet is fully self-supervised and requires neither labels for in-distribution samples nor access to OOD examples.

----

## [322] PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning

**Authors**: *Neeratyoy Mallik, Edward Bergman, Carl Hvarfner, Danny Stoll, Maciej Janowski, Marius Lindauer, Luigi Nardi, Frank Hutter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1704fe7aaff33a54802b83a016050ab8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1704fe7aaff33a54802b83a016050ab8-Abstract-Conference.html)

**Abstract**:

Hyperparameters of Deep Learning (DL) pipelines are crucial for their downstream performance. While a large number of methods for Hyperparameter Optimization (HPO) have been developed, their incurred costs are often untenable for modern DL.Consequently, manual experimentation is still the most prevalent approach to optimize hyperparameters, relying on the researcher's intuition, domain knowledge, and cheap preliminary explorations.To resolve this misalignment between HPO algorithms and DL researchers, we propose PriorBand, an HPO algorithm tailored to DL, able to utilize both expert beliefs and cheap proxy tasks. Empirically, we demonstrate PriorBand's efficiency across a range of DL benchmarks and show its gains under informative expert input and robustness against poor expert beliefs.

----

## [323] Towards Efficient Image Compression Without Autoregressive Models

**Authors**: *Muhammad Salman Ali, Yeongwoong Kim, Maryam Qamar, Sung-Chang Lim, Donghyun Kim, Chaoning Zhang, Sung-Ho Bae, Hui Yong Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/170dc3e41f2d03e327e04dbab0fccbfb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/170dc3e41f2d03e327e04dbab0fccbfb-Abstract-Conference.html)

**Abstract**:

Recently, learned image compression (LIC) has garnered increasing interest with its rapidly improving performance surpassing conventional codecs. A key ingredient of LIC is a hyperprior-based entropy model, where the underlying joint probability of the latent image features is modeled as a product of Gaussian distributions from each latent element. Since latents from the actual images are not spatially independent, autoregressive (AR) context based entropy models were proposed to handle the discrepancy between the assumed distribution and the actual distribution. Though the AR-based models have proven effective, the computational complexity is significantly increased due to the inherent sequential nature of the algorithm. In this paper, we present a novel alternative to the AR-based approach that can provide a significantly better trade-off between performance and complexity. To minimize the discrepancy, we introduce a correlation loss that forces the latents to be spatially decorrelated and better fitted to the independent probability model. Our correlation loss is proved to act as a general plug-in for the hyperprior (HP) based learned image compression methods. The performance gain from our correlation loss is ‘free’ in terms of computation complexity for both inference time and decoding time. To our knowledge, our method  gives the best trade-off between the complexity and performance: combined with the Checkerboard-CM, it attains 90% and when combined with ChARM-CM, it attains 98% of the AR-based BD-Rate gains yet is around 50 times and 30 times faster than AR-based methods respectively

----

## [324] De novo Drug Design using Reinforcement Learning with Multiple GPT Agents

**Authors**: *Xiuyuan Hu, Guoqing Liu, Yang Zhao, Hao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1737656c4dc65027939e47e4587ce95e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1737656c4dc65027939e47e4587ce95e-Abstract-Conference.html)

**Abstract**:

De novo drug design is a pivotal issue in pharmacology and a new area of focus in AI for science research. A central challenge in this field is to generate molecules with specific properties while also producing a wide range of diverse candidates. Although advanced technologies such as transformer models and reinforcement learning have been applied in drug design, their potential has not been fully realized. Therefore, we propose MolRL-MGPT, a reinforcement learning algorithm with multiple GPT agents for drug molecular generation. To promote molecular diversity, we encourage the agents to collaborate in searching for desirable molecules in diverse directions. Our algorithm has shown promising results on the GuacaMol benchmark and exhibits efficacy in designing inhibitors against SARS-CoV-2 protein targets. The codes are available at: https://github.com/HXYfighter/MolRL-MGPT.

----

## [325] Pointwise uncertainty quantification for sparse variational Gaussian process regression with a Brownian motion prior

**Authors**: *Luke Travis, Kolyan Ray*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/176a579942089c4cdc70136c567932ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/176a579942089c4cdc70136c567932ab-Abstract-Conference.html)

**Abstract**:

We study pointwise estimation and uncertainty quantification for a sparse variational Gaussian process method with eigenvector inducing variables. For a rescaled Brownian motion prior, we derive theoretical guarantees and limitations for the frequentist size and coverage of pointwise credible sets. For sufficiently many inducing variables, we precisely characterize the asymptotic frequentist coverage, deducing when credible sets from this variational method are conservative and when overconfident/misleading. We numerically illustrate the applicability of our results and discuss connections with other common Gaussian process priors.

----

## [326] Few-shot Generation via Recalling Brain-Inspired Episodic-Semantic Memory

**Authors**: *Zhibin Duan, Zhiyi Lv, Chaojie Wang, Bo Chen, Bo An, Mingyuan Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17826a22eb8b58494dfdfca61e772c39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17826a22eb8b58494dfdfca61e772c39-Abstract-Conference.html)

**Abstract**:

Aimed at adapting a generative model to a novel generation task with only a few given data samples, the capability of few-shot generation is crucial for many real-world applications with limited data, \emph{e.g.}, artistic domains.Instead of training from scratch, recent works tend to leverage the prior knowledge stored in previous datasets, which is quite similar to the memory mechanism of human intelligence, but few of these works directly imitate the memory-recall mechanism that humans make good use of in accomplishing creative tasks,  \emph{e.g.}, painting and writing.Inspired by the memory mechanism of human brain, in this work, we carefully design a variational structured memory module (VSM), which can simultaneously store both episodic and semantic memories to assist existing generative models efficiently recall these memories during sample generation.Meanwhile, we introduce a bionic memory updating strategy for the conversion between episodic and semantic memories, which can also model the uncertainty during conversion.Then, we combine the developed VSM with various generative models under the Bayesian framework, and evaluate these memory-augmented generative models with few-shot generation tasks, demonstrating the effectiveness of our methods.

----

## [327] Balancing memorization and generalization in RNNs for high performance brain-machine Interfaces

**Authors**: *Joseph T. Costello, Hisham Temmar, Luis Cubillos, Matthew Mender, Dylan Wallace, Matt S. Willsey, Parag G. Patil, Cynthia A. Chestek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17a234c91f746d9625a75cf8a8731ee2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17a234c91f746d9625a75cf8a8731ee2-Abstract-Conference.html)

**Abstract**:

Brain-machine interfaces (BMIs) can restore motor function to people with paralysis but are currently limited by the accuracy of real-time decoding algorithms. Recurrent neural networks (RNNs) using modern training techniques have shown promise in accurately predicting movements from neural signals but have yet to be rigorously evaluated against other decoding algorithms in a closed-loop setting. Here we compared RNNs to other neural network architectures in real-time, continuous decoding of finger movements using intracortical signals from nonhuman primates. Across one and two finger online tasks, LSTMs (a type of RNN) outperformed convolutional and transformer-based neural networks, averaging 18% higher throughput than the convolution network. On simplified tasks with a reduced movement set, RNN decoders were allowed to memorize movement patterns and matched able-bodied control. Performance gradually dropped as the number of distinct movements increased but did not go below fully continuous decoder performance. Finally, in a two-finger task where one degree-of-freedom had poor input signals, we recovered functional control using RNNs trained to act both like a movement classifier and continuous decoder. Our results suggest that RNNs can enable functional real-time BMI control by learning and generating accurate movement patterns.

----

## [328] Saddle-to-Saddle Dynamics in Diagonal Linear Networks

**Authors**: *Scott Pesme, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17a9ab4190289f0e1504bbb98d1d111a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17a9ab4190289f0e1504bbb98d1d111a-Abstract-Conference.html)

**Abstract**:

In this paper we fully describe the trajectory of gradient flow over $2$-layer diagonal linear networks for the regression setting in the limit of vanishing initialisation. We show that the limiting flow successively jumps from a saddle of the training loss to another until reaching the minimum $\ell_1$-norm solution. We explicitly characterise the visited saddles as well as the jump times through a recursive algorithm reminiscent of the LARS algorithm used for computing the Lasso path.  Starting from the zero vector, coordinates are successively activated until the minimum $\ell_1$-norm solution is recovered, revealing an incremental learning. Our proof leverages a convenient arc-length time-reparametrisation which enables to keep track of the transitions between the jumps. Our analysis requires negligible assumptions on the data, applies to both under and overparametrised settings and covers complex cases where there is no monotonicity of the number of active coordinates. We provide numerical experiments to support our findings.

----

## [329] Encoding Human Behavior in Information Design through Deep Learning

**Authors**: *Guanghui Yu, Wei Tang, Saumik Narayanan, Chien-Ju Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17d0a21da4ec2c12b4f07fa2e34e4d6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17d0a21da4ec2c12b4f07fa2e34e4d6c-Abstract-Conference.html)

**Abstract**:

We initiate the study of $\textit{behavioral information design}$ through deep learning. In information design, a $\textit{sender}$ aims to persuade a $\textit{receiver}$ to take certain actions by strategically revealing information. We address scenarios in which the receiver might exhibit different behavior patterns other than the standard Bayesian rational assumption. We propose HAIDNet, a neural-network-based optimization framework for information design that can adapt to multiple representations of human behavior. Through extensive simulation, we show that HAIDNet can not only recover information policies that are near-optimal compared with known analytical solutions, but also can extend to designing information policies for settings that are computationally challenging (e.g., when there are multiple receivers) or for settings where there are no known solutions in general (e.g., when the receiver behavior does not follow the Bayesian rational assumption). We also conduct real-world human-subject experiments and demonstrate that our framework can capture human behavior from data and lead to more effective information policy for real-world human receivers.

----

## [330] Collaboratively Learning Linear Models with Structured Missing Data

**Authors**: *Chen Cheng, Gary Cheng, John C. Duchi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/17f158c25b08758cf650130f7f173e51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/17f158c25b08758cf650130f7f173e51-Abstract-Conference.html)

**Abstract**:

We study the problem of collaboratively learning least squares estimates for $m$ agents. Each agent observes a different subset of the features---e.g., containing data collected from sensors of varying resolution. Our goal is to determine how to coordinate the agents in order to produce the best estimator for each agent. We propose a distributed, semi-supervised algorithm Collab, consisting of three steps: local training, aggregation, and distribution. Our procedure does not require communicating the labeled data, making it communication efficient and useful in settings where the labeled data is inaccessible. Despite this handicap, our procedure is nearly asymptotically, local-minimax optimal---even among estimators allowed to communicate the labeled data such as imputation methods. We test our method on US Census data. We also discuss generalizations of our method to non-Gaussian feature settings, non-linear settings, and Federated Learning.

----

## [331] Generating Behaviorally Diverse Policies with Latent Diffusion Models

**Authors**: *Shashank Hegde, Sumeet Batra, K. R. Zentner, Gaurav S. Sukhatme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/180d4373aca26bd86bf45fc50d1a709f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/180d4373aca26bd86bf45fc50d1a709f-Abstract-Conference.html)

**Abstract**:

Recent progress in Quality Diversity Reinforcement Learning (QD-RL) has enabled learning a collection of behaviorally diverse, high performing policies. However, these methods typically involve storing thousands of policies, which results in high space-complexity and poor scaling to additional behaviors. Condensing the archive into a single model while retaining the performance and coverage of theoriginal collection of policies has proved challenging. In this work, we propose using diffusion models to distill the archive into a single generative model over policy parameters. We show that our method achieves a compression ratio of 13x while recovering 98% of the original rewards and 89% of the original humanoid archive coverage. Further, the conditioning mechanism of diffusion models allowsfor flexibly selecting and sequencing behaviors, including using language. Project website: https://sites.google.com/view/policydiffusion/home.

----

## [332] Incentives in Private Collaborative Machine Learning

**Authors**: *Rachael Hwee Ling Sim, Yehong Zhang, Nghia Hoang, Xinyi Xu, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/180f1a1de4244c009ff0848c55ae54a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/180f1a1de4244c009ff0848c55ae54a5-Abstract-Conference.html)

**Abstract**:

Collaborative machine learning involves training models on data from multiple parties but must incentivize their participation. Existing data valuation methods fairly value and reward each party based on  shared data or model parameters but neglect the privacy risks involved. To address this, we introduce differential privacy (DP) as an incentive. Each party can select its required DP guarantee and perturb its sufficient statistic (SS) accordingly. The mediator values the perturbed SS by the Bayesian surprise it elicits about the model parameters. As our valuation function enforces a privacy-valuation trade-off, parties are deterred from selecting excessive DP guarantees that reduce the utility of the grand coalition's model. Finally, the mediator rewards each party with different posterior samples of the model parameters. Such rewards still satisfy existing incentives like fairness but additionally preserve DP and a high similarity to the grand coalition's posterior. We empirically demonstrate the effectiveness and practicality of our approach on synthetic and real-world datasets.

----

## [333] VideoComposer: Compositional Video Synthesis with Motion Controllability

**Authors**: *Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/180f6184a3458fa19c28c5483bc61877-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/180f6184a3458fa19c28c5483bc61877-Abstract-Conference.html)

**Abstract**:

The pursuit of controllability as a higher standard of visual content creation has yielded remarkable progress in customizable image synthesis. However, achieving controllable video synthesis remains challenging due to the large variation of temporal dynamics and the requirement of cross-frame temporal consistency. Based on the paradigm of compositional generation, this work presents VideoComposer that allows users to flexibly compose a video with textual conditions, spatial conditions, and more importantly temporal conditions. Specifically, considering the characteristic of video data, we introduce the motion vector from compressed videos as an explicit control signal to provide guidance regarding temporal dynamics. In addition, we develop a Spatio-Temporal Condition encoder (STC-encoder) that serves as a unified interface to effectively incorporate the spatial and temporal relations of sequential inputs, with which the model could make better use of temporal conditions and hence achieve higher inter-frame consistency. Extensive experimental results suggest that VideoComposer is able to control the spatial and temporal patterns simultaneously within a synthesized video in various forms, such as text description, sketch sequence, reference video, or even simply hand-crafted motions. The code and models are publicly available athttps://videocomposer.github.io.

----

## [334] Look Beneath the Surface: Exploiting Fundamental Symmetry for Sample-Efficient Offline RL

**Authors**: *Peng Cheng, Xianyuan Zhan, Zhi-Hao Wu, Wenjia Zhang, Youfang Lin, Shoucheng Song, Han Wang, Li Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/181a027913d36bc0a8857c0da661d621-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/181a027913d36bc0a8857c0da661d621-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) offers an appealing approach to real-world tasks by learning policies from pre-collected datasets without interacting with the environment. However, the performance of existing offline RL algorithms heavily depends on the scale and state-action space coverage of datasets. Real-world data collection is often expensive and uncontrollable, leading to small and narrowly covered datasets and posing significant challenges for practical deployments of offline RL. In this paper, we provide a new insight that leveraging the fundamental symmetry of system dynamics can substantially enhance offline RL performance under small datasets. Specifically, we propose a Time-reversal symmetry (T-symmetry) enforced Dynamics Model (TDM), which establishes consistency between a pair of forward and reverse latent dynamics. TDM provides both well-behaved representations for small datasets and a new reliability measure for OOD samples based on compliance with the T-symmetry. These can be readily used to construct a new offline RL algorithm (TSRL) with less conservative policy constraints and a reliable latent space data augmentation procedure. Based on extensive experiments, we find TSRL achieves great performance on small benchmark datasets with as few as 1% of the original samples, which significantly outperforms the recent offline RL algorithms in terms of data efficiency and generalizability. Code is available at:https://github.com/pcheng2/TSRL

----

## [335] Initialization-Dependent Sample Complexity of Linear Predictors and Neural Networks

**Authors**: *Roey Magen, Ohad Shamir*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/18210aa6209b9adfc97b8c17c3741d95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/18210aa6209b9adfc97b8c17c3741d95-Abstract-Conference.html)

**Abstract**:

We provide several new results on the sample complexity of vector-valued linear predictors (parameterized by a matrix), and more generally neural networks. Focusing on size-independent bounds, where only the Frobenius norm distance of the parameters from some fixed reference matrix $W_0$ is controlled, we show that the sample complexity behavior can be surprisingly different than what we may expect considering the well-studied setting of scalar-valued linear predictors. This also leads to new sample complexity bounds for feed-forward neural networks, tackling some open questions in the literature, and establishing a new convex linear prediction problem that is provably learnable without uniform convergence.

----

## [336] Incentivizing Honesty among Competitors in Collaborative Learning and Optimization

**Authors**: *Florian E. Dorner, Nikola Konstantinov, Georgi Pashaliev, Martin T. Vechev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/182b39a4458fb4a9a8d6871a6671ff3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/182b39a4458fb4a9a8d6871a6671ff3e-Abstract-Conference.html)

**Abstract**:

Collaborative learning techniques have the potential to enable training machine learning models that are superior to models trained on a single entityâ€™s data. However, in many cases, potential participants in such collaborative schemes are competitors on a downstream task, such as firms that each aim to attract customers by providing the best recommendations. This can incentivize dishonest updates that damage other participants' models, potentially undermining the benefits of collaboration. In this work, we formulate a game that models such interactions and study two learning tasks within this framework: single-round mean estimation and multi-round SGD on strongly-convex objectives. For a natural class of player actions, we show that rational clients are incentivized to strongly manipulate their updates, preventing learning. We then propose mechanisms that incentivize honest communication and ensure learning quality comparable to full cooperation. Lastly, we empirically demonstrate the effectiveness of our incentive scheme on a standard non-convex federated learning benchmark. Our work shows that explicitly modeling the incentives and actions of dishonest clients, rather than assuming them malicious, can enable strong robustness guarantees for collaborative learning.

----

## [337] SNAP: Self-Supervised Neural Maps for Visual Positioning and Semantic Understanding

**Authors**: *Paul-Edouard Sarlin, Eduard Trulls, Marc Pollefeys, Jan Hosang, Simon Lynen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/182c433412b33c14e32a7c4fc2c3e290-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/182c433412b33c14e32a7c4fc2c3e290-Abstract-Conference.html)

**Abstract**:

Semantic 2D maps are commonly used by humans and machines for navigation purposes, whether it's walking or driving. However, these maps have limitations: they lack detail, often contain inaccuracies, and are difficult to create and maintain, especially in an automated fashion. Can we use  raw imagery to automatically create better maps that can be easily interpreted by both humans and machines? We introduce SNAP, a deep network that learns rich 2D neural maps from ground-level and overhead images. We train our model to align neural maps estimated from different inputs, supervised only with camera poses over tens of millions of StreetView images. SNAP can resolve the location of challenging image queries beyond the reach of traditional methods, outperforming the state of the art in localization by a large margin. Moreover, our neural maps encode not only geometry and appearance but also high-level semantics, discovered without explicit supervision. This enables effective pre-training for data-efficient semantic scene understanding, with the potential to unlock cost-efficient creation of more detailed maps.

----

## [338] Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research

**Authors**: *Cole Gulino, Justin Fu, Wenjie Luo, George Tucker, Eli Bronstein, Yiren Lu, Jean Harb, Xinlei Pan, Yan Wang, Xiangyu Chen, John D. Co-Reyes, Rishabh Agarwal, Rebecca Roelofs, Yao Lu, Nico Montali, Paul Mougin, Zoey Yang, Brandyn White, Aleksandra Faust, Rowan McAllister, Dragomir Anguelov, Benjamin Sapp*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1838feeb71c4b4ea524d0df2f7074245-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1838feeb71c4b4ea524d0df2f7074245-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Simulation is an essential tool to develop and benchmark autonomous vehicle planning software in a safe and cost-effective manner. However, realistic simulation requires accurate modeling of multi-agent interactive behaviors to be trustworthy, behaviors which can be highly nuanced and complex. To address these challenges, we introduce Waymax, a new data-driven simulator for autonomous driving in multi-agent scenes, designed for large-scale simulation and testing.  Waymax uses publicly-released, real-world driving data (e.g., the Waymo Open Motion Dataset) to initialize or play back a diverse set of multi-agent simulated scenarios.   It runs entirely on hardware accelerators such as TPUs/GPUs and supports in-graph simulation for training, making it suitable for modern large-scale, distributed machine learning workflows. To support online training and evaluation, Waymax includes several learned and hard-coded behavior models that allow for realistic interaction within simulation. To supplement Waymax, we benchmark a suite of popular imitation and reinforcement learning algorithms with ablation studies on different design decisions, where we highlight the effectiveness of routes as guidance for planning agents and the ability of RL to overfit against simulated agents.

----

## [339] Equal Opportunity of Coverage in Fair Regression

**Authors**: *Fangxin Wang, Lu Cheng, Ruocheng Guo, Kay Liu, Philip S. Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1849b94ed817ae7043a6b6934ef410c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1849b94ed817ae7043a6b6934ef410c1-Abstract-Conference.html)

**Abstract**:

We study fair machine learning (ML) under predictive uncertainty to enable reliable and trustworthy decision-making. The seminal work of 'equalized coverage' proposed an uncertainty-aware fairness notion. However, it does not guarantee equal coverage rates across more fine-grained groups (e.g., low-income females) conditioning on the true label and is biased in the assessment of uncertainty. To tackle these limitations, we propose a new uncertainty-aware fairness -- Equal Opportunity of Coverage (EOC) -- that aims to achieve two properties: (1) coverage rates for different groups with similar outcomes are close, and (2) the coverage rate for the entire population remains at a predetermined level. Further, the prediction intervals should be narrow to be informative. We propose Binned Fair Quantile Regression (BFQR), a distribution-free post-processing method to improve EOC with reasonable width for any trained ML models. It first calibrates a hold-out set to bound deviation from EOC, then leverages conformal prediction to maintain EOC on a test set, meanwhile optimizing prediction interval width. Experimental results demonstrate the effectiveness of our method in improving EOC.

----

## [340] Nonparametric Teaching for Multiple Learners

**Authors**: *Chen Zhang, Xiaofeng Cao, Weiyang Liu, Ivor W. Tsang, James T. Kwok*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/184a03a3ad07e8897c62461c02634b02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/184a03a3ad07e8897c62461c02634b02-Abstract-Conference.html)

**Abstract**:

We study the problem of teaching multiple learners simultaneously in the nonparametric iterative teaching setting, where the teacher iteratively provides examples to the learner for accelerating the acquisition of a target concept. This problem is motivated by the gap between current single-learner teaching setting and the real-world scenario of human instruction where a teacher typically imparts knowledge to multiple students. Under the new problem formulation, we introduce a novel framework -- Multi-learner Nonparametric Teaching (MINT). In MINT, the teacher aims to instruct multiple learners, with each learner focusing on learning a scalar-valued target model. To achieve this, we frame the problem as teaching a vector-valued target model and extend the target model space from a scalar-valued reproducing kernel Hilbert space used in single-learner scenarios to a vector-valued space. Furthermore, we demonstrate that MINT offers significant teaching speed-up over repeated single-learner teaching, particularly when the multiple learners can communicate with each other. Lastly, we conduct extensive experiments to validate the practicality and efficiency of MINT.

----

## [341] EvoPrompting: Language Models for Code-Level Neural Architecture Search

**Authors**: *Angelica Chen, David Dohan, David R. So*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/184c1e18d00d7752805324da48ad25be-Abstract-Conference.html)

**Abstract**:

Given the recent impressive accomplishments of language models (LMs) for code generation, we explore the use of LMs as general adaptive mutation and crossover operators for an evolutionary neural architecture search (NAS) algorithm.While NAS still proves too difficult a task for LMs to succeed at solely through prompting, we find that the combination of evolutionary prompt engineering with soft prompt-tuning, a method we term EvoPrompting, consistently finds diverse and high performing models. We first demonstrate that EvoPrompting is effective on the computationally efficient MNIST-1D dataset, where EvoPrompting produces convolutional architecture variants that outperform both those designed by human experts and naive few-shot prompting in terms of accuracy and model size. We then apply our method to searching for graph neural networks on the CLRS Algorithmic Reasoning Benchmark, where EvoPrompting is able to design novel architectures that outperform current state-of-the-art models on 21 out of 30 algorithmic reasoning tasks while maintaining similar model size. EvoPrompting is successful at designing accurate and efficient neural network architectures across a variety of machine learning tasks, while also being general enough for easy adaptation to other tasks beyond neural network design.

----

## [342] Global-correlated 3D-decoupling Transformer for Clothed Avatar Reconstruction

**Authors**: *Zechuan Zhang, Li Sun, Zongxin Yang, Ling Chen, Yi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1857d2e8f51ed219ca0c2663239b38e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1857d2e8f51ed219ca0c2663239b38e5-Abstract-Conference.html)

**Abstract**:

Reconstructing 3D clothed human avatars from single images is a challenging task, especially when encountering complex poses and loose clothing. Current methods exhibit limitations in performance, largely attributable to their dependence on insufficient 2D image features and inconsistent query methods. Owing to this, we present the Global-correlated 3D-decoupling Transformer for clothed Avatar reconstruction (GTA), a novel transformer-based architecture that reconstructs clothed human avatars from monocular images. Our approach leverages transformer architectures by utilizing a Vision Transformer model as an encoder for capturing global-correlated image features. Subsequently, our innovative 3D-decoupling decoder employs cross-attention to decouple tri-plane features, using learnable embeddings as queries for cross-plane generation. To effectively enhance feature fusion with the tri-plane 3D feature and human body prior, we propose a hybrid prior fusion strategy combining spatial and prior-enhanced queries, leveraging the benefits of spatial localization and human body prior knowledge. Comprehensive experiments on CAPE and THuman2.0 datasets illustrate that our method outperforms state-of-the-art approaches in both geometry and texture reconstruction, exhibiting high robustness to challenging poses and loose clothing, and producing higher-resolution textures.  Codes are available at https://github.com/River-Zhang/GTA.

----

## [343] TopP&R: Robust Support Estimation Approach for Evaluating Fidelity and Diversity in Generative Models

**Authors**: *Pum Jun Kim, Yoojin Jang, Jisu Kim, Jaejun Yoo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/185969291540b3cd86e70c51e8af5d08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/185969291540b3cd86e70c51e8af5d08-Abstract-Conference.html)

**Abstract**:

We propose a robust and reliable evaluation metric for generative models called Topological Precision and Recall (TopP&R, pronounced “topper”), which systematically estimates supports by retaining only topologically and statistically significant features with a certain level of confidence. Existing metrics, such as Inception Score (IS), Frechet Inception Distance (FID), and various Precision and Recall (P&R) variants, rely heavily on support estimates derived from sample features. However, the reliability of these estimates has been overlooked, even though the quality of the evaluation hinges entirely on their accuracy. In this paper, we demonstrate that current methods not only fail to accurately assess sample quality when support estimation is unreliable, but also yield inconsistent results. In contrast, TopP&R reliably evaluates the sample quality and ensures statistical consistency in its results. Our theoretical and experimental findings reveal that TopP&R provides a robust evaluation, accurately capturing the true trend of change in samples, even in the presence of outliers and non-independent and identically distributed (Non-IID) perturbations where other methods result in inaccurate support estimations. To our knowledge, TopP&R is the first evaluation metric specifically focused on the robust estimation of supports, offering statistical consistency under noise conditions.

----

## [344] A Unified Detection Framework for Inference-Stage Backdoor Defenses

**Authors**: *Xun Xian, Ganghua Wang, Jayanth Srinivasa, Ashish Kundu, Xuan Bi, Mingyi Hong, Jie Ding*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1868a3c73d0d2a44c42458575fa8514c-Abstract-Conference.html)

**Abstract**:

Backdoor attacks involve inserting poisoned samples during training, resulting in a model containing a hidden backdoor that can trigger specific behaviors without impacting performance on normal samples. These attacks are challenging to detect, as the backdoored model appears normal until activated by the backdoor trigger, rendering them particularly stealthy. In this study, we devise a unified inference-stage detection framework to defend against backdoor attacks. We first rigorously formulate the inference-stage backdoor detection problem, encompassing various existing methods, and discuss several challenges and limitations. We then propose a framework with provable guarantees on the false positive rate or the probability of misclassifying a clean sample.  Further, we derive the most powerful detection rule to maximize the detection power, namely the rate of accurately identifying a backdoor sample, given a false positive rate under classical learning scenarios. Based on the theoretically optimal detection rule, we suggest a practical and effective approach for real-world applications based on the latent representations of backdoored deep nets. We extensively evaluate our method on 14 different backdoor attacks using Computer Vision (CV) and Natural Language Processing (NLP) benchmark datasets. The experimental findings align with our theoretical results. We significantly surpass the state-of-the-art methods, e.g., up to 300\% improvement on the detection power as evaluated by AUCROC, over the state-of-the-art defense against advanced adaptive backdoor attacks.

----

## [345] Non-Stationary Bandits with Auto-Regressive Temporal Dependency

**Authors**: *Qinyi Chen, Negin Golrezaei, Djallel Bouneffouf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/186a213d720568b31f9b59c085a23e5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/186a213d720568b31f9b59c085a23e5a-Abstract-Conference.html)

**Abstract**:

Traditional multi-armed bandit (MAB) frameworks, predominantly examined under stochastic or adversarial settings, often overlook the temporal dynamics inherent in many real-world applications such as recommendation systems and online advertising. This paper introduces a novel non-stationary MAB framework that captures the temporal structure of these real-world dynamics through an auto-regressive (AR) reward structure. We propose an algorithm that integrates two key mechanisms: (i) an alternation mechanism adept at leveraging temporal dependencies to dynamically balance exploration and exploitation, and (ii) a restarting mechanism designed to discard out-of-date information. Our algorithm achieves a regret upper bound that nearly matches the lower bound, with regret measured against a robust dynamic benchmark. Finally, via a real-world case study on tourism demand prediction, we demonstrate both the efficacy of our algorithm and the broader applicability of our techniques to more complex, rapidly evolving time series.

----

## [346] Globally solving the Gromov-Wasserstein problem for point clouds in low dimensional Euclidean spaces

**Authors**: *Martin Ryner, Jan Kronqvist, Johan Karlsson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/188409d2ad91db4fb13644d024d99074-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/188409d2ad91db4fb13644d024d99074-Abstract-Conference.html)

**Abstract**:

This paper presents a framework for computing the Gromov-Wasserstein problem between two sets of points in low dimensional spaces, where the discrepancy is the squared Euclidean norm.The Gromov-Wasserstein problem is a generalization of the optimal transport problem that finds the assignment between two sets preserving pairwise distances as much as possible. This can be used to quantify the similarity between two formations or shapes, a common problem in AI and machine learning.The problem can be formulated as a Quadratic Assignment Problem (QAP), which is in general computationally intractable even for small problems. Our framework addresses this challenge by reformulating the QAP as an optimization problem with a low-dimensional domain, leveraging the fact that the problem can be expressed as a concave quadratic optimization problem with low rank. The method scales well with the number of points, and it can be used to find the global solution for large-scale problems with thousands of points.We compare the computational complexity of our approach with state-of-the-art methods on synthetic problems and apply it to a near-symmetrical problem which is of particular interest in computational biology.

----

## [347] Combinatorial Optimization with Policy Adaptation using Latent Space Search

**Authors**: *Félix Chalumeau, Shikha Surana, Clément Bonnet, Nathan Grinsztajn, Arnu Pretorius, Alexandre Laterre, Tom Barrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/18d3a2f3068d6c669dcae19ceca1bc24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/18d3a2f3068d6c669dcae19ceca1bc24-Abstract-Conference.html)

**Abstract**:

Combinatorial Optimization underpins many real-world applications and yet, designing performant algorithms to solve these complex, typically NP-hard, problems remains a significant research challenge. Reinforcement Learning (RL) provides a versatile framework for designing heuristics across a broad spectrum of problem domains. However, despite notable progress, RL has not yet supplanted industrial solvers as the go-to solution. Current approaches emphasize pre-training heuristics that construct solutions, but often rely on search procedures with limited variance, such as stochastically sampling numerous solutions from a single policy, or employing computationally expensive fine-tuning of the policy on individual problem instances. Building on the intuition that performant search at inference time should be anticipated during pre-training, we propose COMPASS, a novel RL approach that parameterizes a distribution of diverse and specialized policies conditioned on a continuous latent space. We evaluate COMPASS across three canonical problems - Travelling Salesman, Capacitated Vehicle Routing, and Job-Shop Scheduling - and demonstrate that our search strategy (i) outperforms state-of-the-art approaches in 9 out of 11 standard benchmarking tasks and (ii) generalizes better, surpassing all other approaches on a set of 18 procedurally transformed instance distributions.

----

## [348] SubseasonalClimateUSA: A Dataset for Subseasonal Forecasting and Benchmarking

**Authors**: *Soukayna Mouatadid, Paulo Orenstein, Genevieve Flaspohler, Miruna Oprescu, Judah Cohen, Franklyn Wang, Sean Knight, Maria Geogdzhayeva, Sam Levang, Ernest Fraenkel, Lester Mackey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/18ef499ee57c4822e1e3ea9b9948af18-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/18ef499ee57c4822e1e3ea9b9948af18-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Subseasonal forecasting of the weather two to six weeks in advance is critical for resource allocation and climate adaptation but poses many challenges for the forecasting community. At this forecast horizon, physics-based dynamical models have limited skill, and the targets for prediction depend in a complex manner on both local weather variables and global climate variables. Recently, machine learning methods have shown promise in advancing the state of the art but  only at the cost of complex data curation, integrating expert knowledge with aggregation across multiple relevant data sources, file formats, and temporal and spatial resolutions.To streamline this process and accelerate future development, we introduce SubseasonalClimateUSA, a curated dataset for training and benchmarking subseasonal forecasting models in the United States. We use this dataset to benchmark a diverse suite of models, including operational dynamical models, classical meteorological baselines, and ten state-of-the-art machine learning and deep learning-based methods from the literature. Overall, our benchmarks suggest simple and effective ways to extend the accuracy of current operational models. SubseasonalClimateUSA is regularly updated and accessible via the https://github.com/microsoft/subseasonal_data/ Python package.

----

## [349] RenderMe-360: A Large Digital Asset Library and Benchmarks Towards High-fidelity Head Avatars

**Authors**: *Dongwei Pan, Long Zhuo, Jingtan Piao, Huiwen Luo, Wei Cheng, Yuxin Wang, Siming Fan, Shengqi Liu, Lei Yang, Bo Dai, Ziwei Liu, Chen Change Loy, Chen Qian, Wayne Wu, Dahua Lin, Kwan-Yee Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1909ac72220bf5016b6c93f08b66cf36-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1909ac72220bf5016b6c93f08b66cf36-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Synthesizing high-fidelity head avatars is a central problem for computer vision and graphics. While head avatar synthesis algorithms have advanced rapidly, the best ones still face great obstacles in real-world scenarios. One of the vital causes is the inadequate datasets  -- 1) current public datasets can only support researchers to explore high-fidelity head avatars in one or two task directions; 2) these datasets usually contain digital head assets with limited data volume, and narrow distribution over different attributes, such as expressions, ages, and accessories. In this paper, we present RenderMe-360, a comprehensive 4D human head dataset to drive advance in head avatar algorithms across different scenarios. It contains massive data assets, with 243+ million complete head frames and over 800k video sequences from 500 different identities captured by multi-view cameras at 30 FPS. It is a large-scale digital library for head avatars with three key attributes: 1) High Fidelity: all subjects are captured in 360 degrees via 60 synchronized, high-resolution 2K cameras. 2) High Diversity: The collected subjects vary from different ages, eras, ethnicities, and cultures, providing abundant materials with distinctive styles in appearance and geometry. Moreover, each subject is asked to perform various dynamic motions, such as expressions and head rotations, which further extend the richness of assets. 3) Rich Annotations: the dataset provides annotations with different granularities: cameras' parameters, background matting, scan, 2D/3D facial landmarks, FLAME fitting, and text description.    Based on the dataset, we build a comprehensive benchmark for head avatar research, with 16 state-of-the-art methods performed on five main tasks: novel view synthesis, novel expression synthesis, hair rendering, hair editing, and talking head generation. Our experiments uncover the strengths and flaws of state-of-the-art methods. RenderMe-360 opens the door for future exploration in modern head avatars. All of the data, code, and models will be publicly available at https://renderme-360.github.io/.

----

## [350] Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation

**Authors**: *Wei Jin, Haitao Mao, Zheng Li, Haoming Jiang, Chen Luo, Hongzhi Wen, Haoyu Han, Hanqing Lu, Zhengyang Wang, Ruirui Li, Zhen Li, Monica Cheng, Rahul Goutam, Haiyang Zhang, Karthik Subbian, Suhang Wang, Yizhou Sun, Jiliang Tang, Bing Yin, Xianfeng Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/193df57a2366d032fb18dcac0698d09a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/193df57a2366d032fb18dcac0698d09a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Modeling customer shopping intentions is a crucial task for e-commerce, as it directly impacts user experience and engagement. Thus,  accurately understanding customer preferences is essential for providing personalized recommendations. Session-based recommendation, which utilizes customer session data to predict their next interaction, has become increasingly popular. However, existing session datasets have limitations in terms of item attributes, user diversity, and dataset scale. As a result, they cannot comprehensively capture the spectrum of user behaviors and preferences.To bridge this gap, we present the Amazon Multilingual Multi-locale Shopping Session Dataset, namely Amazon-M2. It is the first multilingual dataset consisting of millions of user sessions from six different locales, where the major languages of products are English, German, Japanese, French, Italian, and Spanish.Remarkably, the dataset can help us enhance personalization and understanding of user preferences, which can benefit various existing tasks as well as enable new tasks. To test the potential of the dataset, we introduce three tasks in this work:(1) next-product recommendation, (2) next-product recommendation with domain shifts, and (3) next-product title generation.With the above tasks, we benchmark a range of algorithms on our proposed dataset, drawing new insights for further research and practice. In addition, based on the proposed dataset and tasks, we hosted a competition in the KDD CUP 2023 https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge and have attracted thousands of users and submissions. The winning solutions and the associated workshop can be accessed at our website~https://kddcup23.github.io/.

----

## [351] Adversarial Resilience in Sequential Prediction via Abstention

**Authors**: *Surbhi Goel, Steve Hanneke, Shay Moran, Abhishek Shetty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1967f962c7c2083618236d80eeb9d1ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1967f962c7c2083618236d80eeb9d1ac-Abstract-Conference.html)

**Abstract**:

We study the problem of sequential prediction in the stochastic setting with an adversary that is allowed to inject clean-label adversarial (or out-of-distribution) examples. Algorithms designed to handle purely stochastic data tend to fail in the presence of such adversarial examples, often leading to erroneous predictions. This is undesirable in many high-stakes applications such as medical recommendations, where abstaining from predictions on adversarial examples is preferable to misclassification. On the other hand, assuming fully adversarial data leads to very pessimistic bounds that are often vacuous in practice.     To move away from these pessimistic guarantees, we propose a new model of sequential prediction that sits between the purely stochastic and fully adversarial settings by allowing the learner to abstain from making a prediction at no cost on adversarial examples, thereby asking the learner to make predictions with certainty. Assuming access to the marginal distribution on the non-adversarial examples, we design a learner whose error scales with the VC dimension (mirroring the stochastic setting) of the hypothesis class, as opposed to the Littlestone dimension which characterizes the fully adversarial setting. Furthermore, we design learners for VC dimension~1 classes and the class of axis-aligned rectangles, which work even in the absence of access to the marginal distribution. Our key technical contribution is a novel measure for quantifying uncertainty for learning VC classes, which may be of independent interest.

----

## [352] Simplicity Bias in 1-Hidden Layer Neural Networks

**Authors**: *Depen Morwani, Jatin Batra, Prateek Jain, Praneeth Netrapalli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/196c4e02b7464c554f0f5646af5d502e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/196c4e02b7464c554f0f5646af5d502e-Abstract-Conference.html)

**Abstract**:

Recent works have demonstrated that neural networks exhibit extreme *simplicity bias* (SB). That is,  they learn *only the simplest* features  to solve a task at hand, even in the presence of other, more robust but more complex features. Due to the lack of a general and rigorous definition of *features*, these works showcase SB on *semi-synthetic* datasets such as Color-MNIST , MNIST-CIFAR where defining features is relatively easier. In this work, we rigorously define as well as thoroughly establish SB for *one hidden layer* neural networks in the infinite width regime. More concretely, (i) we define SB as the network essentially being a function of a low dimensional projection of the inputs (ii) theoretically, we show that when the data is linearly separable, the network primarily depends on only the linearly separable ($1$-dimensional) subspace even in the presence of an arbitrarily large number of other, more complex features which could have led to a significantly more robust classifier,  (iii) empirically, we show that models trained on *real* datasets such as Imagenet and Waterbirds-Landbirds indeed depend on a low dimensional projection of the inputs, thereby demonstrating SB on these datasets, iv) finally, we present a natural ensemble approach that encourages diversity in  models by training successive models on features not used by earlier models, and demonstrate that it yields models that are significantly more robust to Gaussian noise.

----

## [353] AVOIDDS: Aircraft Vision-based Intruder Detection Dataset and Simulator

**Authors**: *Elysia Q. Smyers, Sydney M. Katz, Anthony Corso, Mykel J. Kochenderfer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19a260641ebaf68d412f427e591bb74a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/19a260641ebaf68d412f427e591bb74a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Designing robust machine learning systems remains an open problem, and there is a need for benchmark problems that cover both environmental changes and evaluation on a downstream task. In this work, we introduce AVOIDDS, a realistic object detection benchmark for the vision-based aircraft detect-and-avoid problem. We provide a labeled dataset consisting of 72,000 photorealistic images of intruder aircraft with various lighting conditions, weather conditions, relative geometries, and geographic locations.  We also provide an interface that evaluates trained models on slices of this dataset to identify changes in performance with respect to changing environmental conditions. Finally, we implement a fully-integrated, closed-loop simulator of the vision-based detect-and-avoid problem to evaluate trained models with respect to the downstream collision avoidance task. This benchmark will enable further research in the design of robust machine learning systems for use in safety-critical applications. The AVOIDDS dataset and code are publicly available at https://purl.stanford.edu/hj293cv5980 and https://github.com/sisl/VisionBasedAircraftDAA, respectively.

----

## [354] Temporally Disentangled Representation Learning under Unknown Nonstationarity

**Authors**: *Xiangchen Song, Weiran Yao, Yewen Fan, Xinshuai Dong, Guangyi Chen, Juan Carlos Niebles, Eric Xing, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19a567abaec3990cb40d7a013556fecd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19a567abaec3990cb40d7a013556fecd-Abstract-Conference.html)

**Abstract**:

In unsupervised causal representation learning for sequential data with time-delayed latent causal influences, strong identifiability results for the disentanglement of causally-related latent variables have been established in stationary settings by leveraging temporal structure.However, in nonstationary setting, existing work only partially addressed the problem by either utilizing observed auxiliary variables (e.g., class labels and/or domain indexes) as side information or assuming simplified latent causal dynamics. Both constrain the method to a limited range of scenarios.In this study, we further explored the Markov Assumption under time-delayed causally related process in nonstationary setting and showed that under mild conditions, the independent latent components can be recovered from their nonlinear mixture up to a permutation and a component-wise transformation, without the observation of auxiliary variables. We then introduce NCTRL, a principled estimation framework, to reconstruct time-delayed latent causal variables and identify their relations from measured sequential data only.Empirical evaluations demonstrated the reliable identification of time-delayed latent causal influences, with our methodology substantially outperforming existing baselines that fail to exploit the nonstationarity adequately and then, consequently, cannot distinguish distribution shifts.

----

## [355] Accelerated Quasi-Newton Proximal Extragradient: Faster Rate for Smooth Convex Optimization

**Authors**: *Ruichen Jiang, Aryan Mokhtari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19c9708f31ec44b5b1cbd67f91d05d95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19c9708f31ec44b5b1cbd67f91d05d95-Abstract-Conference.html)

**Abstract**:

In this paper, we propose an accelerated quasi-Newton proximal extragradient method for solving unconstrained smooth convex optimization problems. With access only to the gradients of the objective, we prove that our method can achieve a convergence rate of $\mathcal{O}\bigl(\min\\{\frac{1}{k^2}, \frac{\sqrt{d\log k}}{k^{2.5}}\\}\bigr)$, where $d$ is the problem dimension and $k$ is the number of iterations. In particular, in the regime where $k = \mathcal{O}(d)$, our method matches the _optimal rate_ of $\mathcal{O}(\frac{1}{k^2})$ by Nesterov's accelerated gradient (NAG). Moreover, in the the regime where $k = \Omega(d \log d)$, it outperforms NAG and converges at a _faster rate_ of $\mathcal{O}\bigl(\frac{\sqrt{d\log k}}{k^{2.5}}\bigr)$. To the best of our knowledge, this result is the first to demonstrate a provable gain for a quasi-Newton-type method over NAG in the convex setting.  To achieve such results, we build our method on a recent variant of the Monteiro-Svaiter acceleration framework and adopt an online learning perspective to update the Hessian approximation matrices, in which we relate the convergence rate of our method to the dynamic regret of a specific online convex optimization problem in the space of matrices.

----

## [356] Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference

**Authors**: *Tao Lei, Junwen Bai, Siddhartha Brahma, Joshua Ainslie, Kenton Lee, Yanqi Zhou, Nan Du, Vincent Y. Zhao, Yuexin Wu, Bo Li, Yu Zhang, Ming-Wei Chang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19d7204af519eae9993f7f72377a0ec0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19d7204af519eae9993f7f72377a0ec0-Abstract-Conference.html)

**Abstract**:

We propose Conditional Adapter (CoDA), a parameter-efficient transfer learning method that also improves inference efficiency. CoDA generalizes beyond standard adapter approaches to enable a new way of balancing speed and accuracy using conditional computation.Starting with an existing dense pretrained model, CoDA adds sparse activation together with a small number of new parameters and a light-weight training phase.Our experiments demonstrate that the CoDA approach provides an unexpectedly efficient way to transfer knowledge.Across a variety of language, vision, and speech tasks, CoDA achieves a 2x to 8x inference speed-up compared to the state-of-the-art Adapter approaches with moderate to no accuracy loss and the same parameter efficiency.

----

## [357] Time-Independent Information-Theoretic Generalization Bounds for SGLD

**Authors**: *Futoshi Futami, Masahiro Fujisawa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19dbb86f771ddbf9986cf0c9b1c61c17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19dbb86f771ddbf9986cf0c9b1c61c17-Abstract-Conference.html)

**Abstract**:

We provide novel information-theoretic generalization bounds for stochastic gradient Langevin dynamics (SGLD) under the assumptions of smoothness and dissipativity, which are widely used in sampling and non-convex optimization studies.Our bounds are time-independent and decay to zero as the sample size increases, regardless of the number of iterations and whether the step size is fixed.Unlike previous studies, we derive the generalization error bounds by focusing on the time evolution of the Kullback--Leibler divergence, which is related to the stability of datasets and is the upper bound of the mutual information between output parameters and an input dataset.Additionally, we establish the first information-theoretic generalization bound when the training and test loss are the same by showing that a loss function of SGLD is sub-exponential.This bound is also time-independent and removes the problematic step size dependence in existing work, leading to an improved excess risk bound by combining our analysis with the existing non-convex optimization error bounds.

----

## [358] Topology-Aware Uncertainty for Image Segmentation

**Authors**: *Saumya Gupta, Yikai Zhang, Xiaoling Hu, Prateek Prasanna, Chao Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19ded4cfc36a7feb7fce975393d378fd-Abstract-Conference.html)

**Abstract**:

Segmentation of curvilinear structures such as vasculature and road networks is challenging due to relatively weak signals and complex geometry/topology. To facilitate and accelerate large scale annotation, one has to adopt semi-automatic approaches such as proofreading by experts. In this work, we focus on uncertainty estimation for such tasks, so that highly uncertain, and thus error-prone structures can be identified for human annotators to verify. Unlike most existing works, which provide pixel-wise uncertainty maps, we stipulate it is crucial to estimate uncertainty in the units of topological structures, e.g., small pieces of connections and branches. To achieve this, we leverage tools from topological data analysis, specifically discrete Morse theory (DMT), to first capture the structures, and then reason about their uncertainties. To model the uncertainty, we (1) propose a joint prediction model that estimates the uncertainty of a structure while taking the neighboring structures into consideration (inter-structural uncertainty); (2) propose a novel Probabilistic DMT to model the inherent uncertainty within each structure (intra-structural uncertainty) by sampling its representations via a perturb-and-walk scheme. On various 2D and 3D datasets, our method produces better structure-wise uncertainty maps compared to existing works. Code available at: https://github.com/Saumya-Gupta-26/struct-uncertainty

----

## [359] Multiplication-Free Transformer Training via Piecewise Affine Operations

**Authors**: *Atli Kosson, Martin Jaggi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/19df21cd4931bd0caaa4d8480e9a59cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/19df21cd4931bd0caaa4d8480e9a59cd-Abstract-Conference.html)

**Abstract**:

Multiplications are responsible for most of the computational cost involved in neural network training and inference. Recent research has thus looked for ways to reduce the cost associated with them. Inspired by Mogami 2020, we replace multiplication with a cheap piecewise affine approximation that is achieved by adding the bit representation of the floating point numbers together as integers. We show that transformers can be trained with the resulting modified matrix multiplications on both vision and language tasks with little to no performance impact, and without changes to the training hyperparameters. We further replace all non-linearities in the networks making them fully and jointly piecewise affine in both inputs and weights. Finally, we show that we can eliminate all multiplications in the entire training process, including operations in the forward pass, backward pass and optimizer update, demonstrating the first successful training of modern neural network architectures in a fully multiplication-free fashion.

----

## [360] A Unified Framework for Uniform Signal Recovery in Nonlinear Generative Compressed Sensing

**Authors**: *Junren Chen, Jonathan Scarlett, Michael Ng, Zhaoqiang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a04df6a405210aab4986994b873db9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a04df6a405210aab4986994b873db9b-Abstract-Conference.html)

**Abstract**:

In generative compressed sensing (GCS), we want to recover a signal $\mathbf{x^*}\in\mathbb{R}^n$ from $m$ measurements ($m\ll n$) using a generative prior $\mathbf{x^*}\in G(\mathbb{B}_2^k(r))$, where $G$ is typically an $L$-Lipschitz continuous generative model and $\mathbb{B}_2^k(r)$ represents the radius-$r$ $\ell_2$-ball in $\mathbb{R}^k$. Under nonlinear measurements, most prior results are non-uniform, i.e., they hold with high probability for a fixed $\mathbf{x^*}$ rather than for all $\mathbf{x^*}$ simultaneously. In this paper, we build a unified framework to derive uniform recovery guarantees for nonlinear GCS where the observation model is nonlinear and possibly discontinuous or unknown. Our framework accommodates GCS with 1-bit/uniformly quantized observations and single index model as canonical examples. Specifically, using a single   realization of the sensing ensemble and generalized Lasso,   all $\mathbf{x^*}\in G(\mathbb{B}_2^k(r))$ can be recovered up to an $\ell_2$-error at most $\epsilon$ using roughly $\tilde{O}({k}/{\epsilon^2})$ samples, with omitted logarithmic factors typically being dominated by $\log L$.  Notably, this almost coincides with existing non-uniform guarantees up to logarithmic factors, hence the uniformity costs very little.    As part of our technical contributions, we introduce Lipschitz approximation to handle discontinuous observation models.    We also develop a concentration inequality  that produces tighter bound for product process whose index sets have low metric entropy.  Experimental results are presented to corroborate our theory.

----

## [361] Tempo Adaptation in Non-stationary Reinforcement Learning

**Authors**: *Hyunin Lee, Yuhao Ding, Jongmin Lee, Ming Jin, Javad Lavaei, Somayeh Sojoudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a0672689a693e0764f93f900488b3d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a0672689a693e0764f93f900488b3d9-Abstract-Conference.html)

**Abstract**:

We first raise and tackle a ``time synchronization'' issue between the agent and the environment in non-stationary reinforcement learning (RL), a crucial factor hindering its real-world applications. In reality, environmental changes occur over wall-clock time ($t$) rather than episode progress ($k$), where wall-clock time signifies the actual elapsed time within the fixed duration $t \in [0, T]$. In existing works, at episode $k$, the agent rolls a trajectory and trains a policy before transitioning to episode $k+1$. In the context of the time-desynchronized environment, however, the agent at time $t_{k}$ allocates $\Delta t$ for trajectory generation and training, subsequently moves to the next episode at $t_{k+1}=t_{k}+\Delta t$. Despite a fixed total number of episodes ($K$), the agent accumulates different trajectories influenced by the choice of interaction times ($t_1,t_2,...,t_K$), significantly impacting the suboptimality gap of the policy. We propose a Proactively Synchronizing Tempo ($\texttt{ProST}$) framework that computes a suboptimal sequence {$t_1,t_2,...,t_K$} (= { $t_{1:K}$}) by minimizing an upper bound on its performance measure, i.e., the dynamic regret. Our main contribution is that we show that a suboptimal {$t_{1:K}$} trades-off between the policy training time (agent tempo) and how fast the environment changes (environment tempo). Theoretically, this work develops a suboptimal {$t_{1:K}$} as a function of the degree of the environment's non-stationarity while also achieving a sublinear dynamic regret. Our experimental evaluation on various high-dimensional non-stationary environments shows that the $\texttt{ProST}$ framework achieves a higher online return at suboptimal {$t_{1:K}$} than the existing methods.

----

## [362] Unsupervised Semantic Correspondence Using Stable Diffusion

**Authors**: *Eric Hedlin, Gopal Sharma, Shweta Mahajan, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a074a28c3a6f2056562d00649ae6416-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a074a28c3a6f2056562d00649ae6416-Abstract-Conference.html)

**Abstract**:

Text-to-image diffusion models are now capable of generating images that are often indistinguishable from real images. To generate such images, these models must understand the semantics of the objects they are asked to generate. In this work we show that, without any training, one can leverage this semantic knowledge within diffusion models to find semantic correspondences â€“ locations in multiple images that have the same semantic meaning. Specifically, given an image, we optimize the prompt embeddings of these models for maximum attention on the regions of interest. These optimized embeddings capture semantic information about the location, which can then be transferred to another image. By doing so we obtain results on par with the strongly supervised state of the art on the PF-Willow dataset and significantly outperform (20.9% relative for the SPair-71k dataset) any existing weakly- or unsupervised method on PF-Willow, CUB-200 and SPair-71k datasets.

----

## [363] Efficient Subgame Refinement for Extensive-form Games

**Authors**: *Zhenxing Ge, Zheng Xu, Tianyu Ding, Wenbin Li, Yang Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a2b4aba905a16733ff199888ac8eec4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a2b4aba905a16733ff199888ac8eec4-Abstract-Conference.html)

**Abstract**:

Subgame solving is an essential technique in addressing large imperfect information games, with various approaches developed to enhance the performance of refined strategies in the abstraction of the target subgame. However, directly applying existing subgame solving techniques may be difficult, due to the intricate nature and substantial size of many real-world games. To overcome this issue, recent subgame solving methods allow for subgame solving on limited knowledge order subgames, increasing their applicability in large games; yet this may still face obstacles due to extensive information set sizes. To address this challenge, we propose a generative subgame solving (GS2) framework, which utilizes a generation function to identify a subset of the earliest-reached nodes, reducing the size of the subgame. Our method is supported by a theoretical analysis and employs a diversity-based generation function to enhance safety. Experiments conducted on medium-sized games as well as the challenging large game of GuanDan demonstrate a significant improvement over the blueprint.

----

## [364] NeRF-IBVS: Visual Servo Based on NeRF for Visual Localization and Navigation

**Authors**: *Yuanze Wang, Yichao Yan, Dianxi Shi, Wenhan Zhu, Jianqiang Xia, Jeff Tan, Songchang Jin, Ke Gao, Xiaobo Li, Xiaokang Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a57081f257da7b440b8eda72a0b12d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a57081f257da7b440b8eda72a0b12d4-Abstract-Conference.html)

**Abstract**:

Visual localization is a fundamental task in computer vision and robotics. Training existing visual localization methods requires a large number of posed images to generalize to novel views, while state-of-the-art methods generally require dense ground truth 3D labels for supervision. However, acquiring a  large number of posed images and dense 3D labels in the real world is challenging and costly. In this paper, we present a novel visual localization method that achieves accurate localization while using only a few posed images compared to other localization methods. To achieve this, we first use a few posed images with coarse pseudo-3D labels provided by NeRF to train a coordinate regression network. Then a coarse pose is estimated from the regression network with PNP. Finally, we use the image-based visual servo (IBVS) with the scene prior provided by NeRF for pose optimization. Furthermore, our method can provide effective navigation prior, which enable navigation based on IBVS without using custom markers and depth sensor. Extensive experiments on 7-Scenes  and 12-Scenes datasets demonstrate that our method outperforms state-of-the-art methods under the same setting, with only 5\% to 25\% training data.  Furthermore, our framework can be naturally extended to the visual navigation task based on IBVS, and its effectiveness is verified in simulation experiments.

----

## [365] How Does Adaptive Optimization Impact Local Neural Network Geometry?

**Authors**: *Kaiqi Jiang, Dhruv Malik, Yuanzhi Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a5e6d0441a8e1eda9a50717b0870f94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a5e6d0441a8e1eda9a50717b0870f94-Abstract-Conference.html)

**Abstract**:

Adaptive optimization methods are well known to achieve superior convergence relative to vanilla gradient methods. The traditional viewpoint in optimization, particularly in convex optimization, explains this improved performance by arguing that, unlike vanilla gradient schemes, adaptive algorithms mimic the behavior of a second-order method by adapting to the *global* geometry of the loss function. We argue that in the context of neural network optimization, this traditional viewpoint is insufficient. Instead, we advocate for a *local* trajectory analysis. For iterate trajectories produced by running a generic optimization algorithm OPT, we introduce $R^{\text{OPT}}\_{\text{med}}$, a statistic that is analogous to the condition number of the loss Hessian evaluated at the iterates. Through extensive experiments on language models where adaptive algorithms converge faster than vanilla gradient methods like SGD, we show that adaptive methods such as Adam bias the trajectories towards regions where $R^{\text{Adam}}_{\text{med}}$ is small, where one might expect faster optimization. By contrast, SGD (with momentum) biases the trajectories towards regions where $R^{\text{SGD}}\_{\text{med}}$ is comparatively large. We complement these empirical observations with a theoretical result that provably demonstrates this phenomenon in the simplified setting of a two-layer linear network. We view our findings as evidence for the need of a new explanation of the success of adaptive methods, one that is different than the conventional wisdom.

----

## [366] Are Diffusion Models Vision-And-Language Reasoners?

**Authors**: *Benno Krojer, Elinor Poole-Dayan, Vikram Voleti, Chris Pal, Siva Reddy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a675d804f50509b8e21d0d3ca709d03-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a675d804f50509b8e21d0d3ca709d03-Abstract-Conference.html)

**Abstract**:

Text-conditioned image generation models have recently shown immense qualitative success using denoising diffusion processes. However, unlike discriminative vision-and-language models, it is a non-trivial task to subject these diffusion-based generative models to automatic fine-grained quantitative evaluation of high-level phenomena such as compositionality.Towards this goal, we perform two innovations. First, we transform diffusion-based models (in our case, Stable Diffusion) for any image-text matching (ITM) task using a novel method called DiffusionITM.Second, we introduce the Generative-Discriminative Evaluation Benchmark (GDBench) benchmark with 7 complex vision-and-language tasks, bias evaluation and detailed analysis.We find that Stable Diffusion + DiffusionITM is competitive on many tasks and outperforms CLIP on compositional tasks like like CLEVR and Winoground.We further boost its compositional performance with a transfer setup by fine-tuning on MS-COCO while retaining generative capabilities. We also measure the stereotypical bias in diffusion models, and find that Stable Diffusion 2.1 is, for the most part, less biased than Stable Diffusion 1.5.Overall, our results point in an exciting direction bringing discriminative and generative model evaluation closer. We will release code and benchmark setup soon.

----

## [367] ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation

**Authors**: *Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a87980b9853e84dfb295855b425c262-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a87980b9853e84dfb295855b425c262-Abstract-Conference.html)

**Abstract**:

Score distillation sampling (SDS) has shown great promise in text-to-3D generation by distilling pretrained large-scale text-to-image diffusion models, but suffers from over-saturation, over-smoothing, and low-diversity problems. In this work, we propose to model the 3D parameter as a random variable instead of a constant as in SDS and present *variational score distillation* (VSD), a principled particle-based variational framework to explain and address the aforementioned issues in text-to-3D generation. We show that SDS is a special case of VSD and leads to poor samples with both small and large CFG weights. In comparison, VSD works well with various CFG weights as ancestral sampling from diffusion models and simultaneously improves the diversity and sample quality with a common CFG weight (i.e., 7.5). We further present various improvements in the design space for text-to-3D such as distillation time schedule and density initialization, which are orthogonal to the distillation algorithm yet not well explored. Our overall approach, dubbed *ProlificDreamer*, can generate high rendering resolution (i.e., 512$\times$512) and high-fidelity NeRF with rich structure and complex effects (e.g., smoke and drops). Further, initialized from NeRF, meshes fine-tuned by VSD are meticulously detailed and photo-realistic.

----

## [368] SAMoSSA: Multivariate Singular Spectrum Analysis with Stochastic Autoregressive Noise

**Authors**: *Abdullah Alomar, Munther A. Dahleh, Sean Mann, Devavrat Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1a8d295871250443f9747d239925b89d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1a8d295871250443f9747d239925b89d-Abstract-Conference.html)

**Abstract**:

The well-established practice of time series analysis involves estimating deterministic, non-stationary trend and seasonality components followed by learning the residual stochastic, stationary components. Recently, it has been shown that one can learn the deterministic non-stationary components accurately using  multivariate Singular Spectrum Analysis (mSSA) in the absence of a correlated stationary component; meanwhile, in the absence of deterministic non-stationary components, the Autoregressive (AR) stationary component can also be learnt readily, e.g. via Ordinary Least Squares (OLS). However, a theoretical underpinning of multi-stage learning algorithms involving both deterministic and stationary components has been absent in the literature despite its pervasiveness. We resolve this open question by establishing desirable theoretical guarantees for a natural two-stage algorithm, where mSSA is first applied to estimate the non-stationary components despite the presence of a correlated stationary AR component, which is subsequently learned from the residual time series. We provide a finite-sample forecasting consistency bound for the proposed algorithm, SAMoSSA, which is data-driven and thus requires minimal parameter tuning. To establish theoretical guarantees, we overcome three hurdles: (i) we characterize the spectra of Page matrices of stable AR processes, thus extending the analysis of mSSA; (ii) we extend the analysis of AR process identification in the presence of arbitrary bounded perturbations; (iii) we characterize the out-of-sample or forecasting error, as opposed to solely considering model identification. Through representative empirical studies, we validate the superior performance of SAMoSSA compared to existing baselines. Notably, SAMoSSA's ability to account for AR noise structure yields improvements ranging from 5% to 37% across various benchmark datasets.

----

## [369] Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection

**Authors**: *Ruiying Lu, YuJie Wu, Long Tian, Dongsheng Wang, Bo Chen, Xiyang Liu, Ruimin Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1abc87c67cc400a67b869358e627fe37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1abc87c67cc400a67b869358e627fe37-Abstract-Conference.html)

**Abstract**:

Unsupervised image Anomaly Detection (UAD) aims to learn robust and discriminative representations of normal samples. While separate solutions per class endow expensive computation and limited generalizability, this paper focuses on building a unified framework for multiple classes. Under such a challenging setting, popular reconstruction-based networks with continuous latent representation assumption always suffer from the "identical shortcut" issue, where both normal and abnormal samples can be well recovered and difficult to distinguish. To address this pivotal issue, we propose a hierarchical vector quantized prototype-oriented Transformer under a probabilistic framework. First, instead of learning the continuous representations, we preserve the typical normal patterns as discrete iconic prototypes, and confirm the importance of Vector Quantization in preventing the model from falling into the shortcut. The vector quantized iconic prototypes are integrated into the Transformer for reconstruction, such that the abnormal data point is flipped to a normal data point. Second, we investigate an exquisite hierarchical framework to relieve the codebook collapse issue and replenish frail normal patterns.  Third, a prototype-oriented optimal transport method is proposed to better regulate the prototypes and hierarchically evaluate the abnormal score. By evaluating on MVTec-AD and VisA datasets, our model surpasses the state-of-the-art alternatives and possesses good interpretability. The code is available at https://github.com/RuiyingLu/HVQ-Trans.

----

## [370] MCUFormer: Deploying Vision Tranformers on Microcontrollers with Limited Memory

**Authors**: *Yinan Liang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, Jiwen Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1ae4999aefb509d75d8608e07280922c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1ae4999aefb509d75d8608e07280922c-Abstract-Conference.html)

**Abstract**:

Due to the high price and heavy energy consumption of GPUs, deploying deep models on IoT devices such as microcontrollers makes significant contributions for ecological AI. Conventional methods successfully enable convolutional neural network inference of high resolution images on microcontrollers, while the framework for vision transformers that achieve the state-of-the-art performance in many vision applications still remains unexplored. In this paper, we propose a hardware-algorithm co-optimizations method called MCUFormer to deploy vision transformers on microcontrollers with extremely limited memory, where we jointly design transformer architecture and construct the inference operator library to fit the memory resource constraint. More specifically, we generalize the one-shot network architecture search (NAS) to discover the optimal architecture with highest task performance given the memory budget from the microcontrollers, where we enlarge the existing search space of vision transformers by considering the low-rank decomposition dimensions and patch resolution for memory reduction. For the construction of the inference operator library of vision transformers, we schedule the memory buffer during inference through operator integration, patch embedding decomposition, and token overwriting, allowing the memory buffer to be fully utilized to adapt to the forward pass of the vision transformer. Experimental results demonstrate that our MCUFormer achieves 73.62\% top-1 accuracy on ImageNet for image classification with 320KB memory on STM32F746 microcontroller. Code is available at https://github.com/liangyn22/MCUFormer.

----

## [371] Towards Accelerated Model Training via Bayesian Data Selection

**Authors**: *Zhijie Deng, Peng Cui, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1af3e0bf5905e33789979f666c31192d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1af3e0bf5905e33789979f666c31192d-Abstract-Conference.html)

**Abstract**:

Mislabeled, duplicated, or biased data in real-world scenarios can lead to prolonged training and even hinder model convergence. Traditional solutions prioritizing easy or hard samples lack the flexibility to handle such a variety simultaneously. Recent work has proposed a more reasonable data selection principle by examining the data's impact on the model's generalization loss. However, its practical adoption relies on less principled approximations and additional holdout data. This work solves these problems by leveraging a lightweight Bayesian treatment and incorporating off-the-shelf zero-shot predictors built on large-scale pre-trained models. The resulting algorithm is efficient and easy to implement. We perform extensive empirical studies on challenging benchmarks with considerable data noise and imbalance in the online batch selection scenario, and observe superior training efficiency over competitive baselines. Notably, on the challenging WebVision benchmark, our method can achieve similar predictive performance with significantly fewer training iterations than leading data selection methods.

----

## [372] CSOT: Curriculum and Structure-Aware Optimal Transport for Learning with Noisy Labels

**Authors**: *Wanxing Chang, Ye Shi, Jingya Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b0da24d136f46bfaee78e8da907127e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b0da24d136f46bfaee78e8da907127e-Abstract-Conference.html)

**Abstract**:

Learning with noisy labels (LNL) poses a significant challenge in training a well-generalized model while avoiding overfitting to corrupted labels.Recent advances have achieved impressive performance by identifying clean labels and correcting corrupted labels for training.However, the current approaches rely heavily on the modelâ€™s predictions and evaluate each sample independently without considering either the global or local structure of the sample distribution.These limitations typically result in a suboptimal solution for the identification and correction processes, which eventually leads to models overfitting to incorrect labels.In this paper, we propose a novel optimal transport (OT) formulation, called Curriculum and Structure-aware Optimal Transport (CSOT). CSOT concurrently considers the inter- and intra-distribution structure of the samples to construct a robust denoising and relabeling allocator.During the training process, the allocator incrementally assigns reliable labels to a fraction of the samples with the highest confidence. These labels have both global discriminability and local coherence.Notably, CSOT is a new OT formulation with a nonconvex objective function and curriculum constraints, so it is not directly compatible with classical OT solvers. Here, we develop a lightspeed computational method that involves a scaling iteration within a generalized conditional gradient framework to solve CSOT efficiently.Extensive experiments demonstrate the superiority of our method over the current state-of-the-arts in LNL.

----

## [373] In-Context Learning Unlocked for Diffusion Models

**Authors**: *Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang (Atlas) Wang, Mingyuan Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b3750390ca8b931fb9ca988647940cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b3750390ca8b931fb9ca988647940cb-Abstract-Conference.html)

**Abstract**:

We present Prompt Diffusion, a framework for enabling in-context learning in diffusion-based generative models. Given a pair of task-specific example images, such as depth from/to image and scribble from/to image, and a text guidance, our model automatically understands the underlying task and performs the same task on a new query image following the text guidance. To achieve this, we propose a vision-language prompt that can model a wide range of vision-language tasks and a diffusion model that takes it as input. The diffusion model is trained jointly on six different tasks using these prompts. The resulting Prompt Diffusion model becomes the first diffusion-based vision-language foundation model capable of in-context learning. It demonstrates high-quality in-context generation for the trained tasks and effectively generalizes to new, unseen vision tasks using their respective prompts. Our model also shows compelling text-guided image editing results. Our framework aims to facilitate research into in-context learning for computer vision. We share our code and pre-trained models at https://github.com/Zhendong-Wang/Prompt-Diffusion.

----

## [374] Object-Centric Slot Diffusion

**Authors**: *Jindong Jiang, Fei Deng, Gautam Singh, Sungjin Ahn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b3ceb8a495a63ced4a48f8429ccdcd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b3ceb8a495a63ced4a48f8429ccdcd8-Abstract-Conference.html)

**Abstract**:

The recent success of transformer-based image generative models in object-centric learning highlights the importance of powerful image generators for handling complex scenes. However, despite the high expressiveness of diffusion models in image generation, their integration into object-centric learning remains largely unexplored in this domain. In this paper, we explore the feasibility and potential of integrating diffusion models into object-centric learning and investigate the pros and cons of this approach. We introduce Latent Slot Diffusion (LSD), a novel model that serves dual purposes: it is the first object-centric learning model to replace conventional slot decoders with a latent diffusion model conditioned on object slots, and it is also the first unsupervised compositional conditional diffusion model that operates without the need for supervised annotations like text. Through experiments on various object-centric tasks, including the first application of the FFHQ dataset in this field, we demonstrate that LSD significantly outperforms state-of-the-art transformer-based decoders, particularly in more complex scenes, and exhibits superior unsupervised compositional generation quality. In addition, we conduct a preliminary investigation into the integration of pre-trained diffusion models in LSD and demonstrate its effectiveness in real-world image segmentation and generation. Project page is available at https://latentslotdiffusion.github.io

----

## [375] NAS-X: Neural Adaptive Smoothing via Twisting

**Authors**: *Dieterich Lawson, Michael Li, Scott W. Linderman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b3d005a2cb0e71e698e0b13ac657473-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b3d005a2cb0e71e698e0b13ac657473-Abstract-Conference.html)

**Abstract**:

Sequential latent variable models (SLVMs) are essential tools in statistics and machine learning, with applications ranging from healthcare to neuroscience. As their flexibility increases, analytic inference and model learning can become challenging, necessitating approximate methods. Here we introduce neural adaptive smoothing via twisting (NAS-X), a method that extends reweighted wake-sleep (RWS) to the sequential setting by using smoothing sequential Monte Carlo (SMC) to estimate intractable posterior expectations. Combining RWS and smoothing SMC allows NAS-X to provide low-bias and low-variance gradient estimates, and fit both discrete and continuous latent variable models. We illustrate the theoretical advantages of NAS-X over previous methods and explore these advantages empirically in a variety of tasks, including a challenging application to mechanistic models of neuronal dynamics. These experiments show that NAS-X substantially outperforms previous VI- and RWS-based methods in inference and model learning, achieving lower parameter error and tighter likelihood bounds.

----

## [376] Reflexion: language agents with verbal reinforcement learning

**Authors**: *Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning. We propose \emph{Reflexion}, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback. Concretely, Reflexion agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning). For example, Reflexion achieves a 91\% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80\%. We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, and provide insights into how they affect performance. We release all code, demos, and datasets at \url{https://github.com/noahshinn024/reflexion}.

----

## [377] Demographic Parity Constrained Minimax Optimal Regression under Linear Model

**Authors**: *Kazuto Fukuchi, Jun Sakuma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b4acad19cc425a7352a71d4e4468393-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b4acad19cc425a7352a71d4e4468393-Abstract-Conference.html)

**Abstract**:

We explore the minimax optimal error associated with a demographic parity-constrained regression problem within the context of a linear model. Our proposed model encompasses a broader range of discriminatory bias sources compared to the model presented by Chzhen and Schreuder. Our analysis reveals that the minimax optimal error for the demographic parity-constrained regression problem under our model is characterized by $\Theta(\frac{dM}{n})$, where $n$ denotes the sample size, $d$ represents the dimensionality, and $M$ signifies the number of demographic groups arising from sensitive attributes. Moreover, we demonstrate that the minimax error increases in conjunction with a larger bias present in the model.

----

## [378] GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization

**Authors**: *Vicente Vivanco Cepeda, Gaurav Kumar Nayak, Mubarak Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html)

**Abstract**:

Worldwide Geo-localization aims to pinpoint the precise location of images taken anywhere on Earth. This task has considerable challenges due to the immense variation in geographic landscapes. The image-to-image retrieval-based approaches fail to solve this problem on a global scale as it is not feasible to construct a large gallery of images covering the entire world. Instead, existing approaches divide the globe into discrete geographic cells, transforming the problem into a classification task. However, their performance is limited by the predefined classes and often results in inaccurate localizations when an image's location significantly deviates from its class center. To overcome these limitations, we propose GeoCLIP, a novel CLIP-inspired Image-to-GPS retrieval approach that enforces alignment between the image and its corresponding GPS locations. GeoCLIP's location encoder models the Earth as a continuous function by employing positional encoding through random Fourier features and constructing a hierarchical representation that captures information at varying resolutions to yield a semantically rich high-dimensional feature suitable to use even beyond geo-localization. To the best of our knowledge, this is the first work employing GPS encoding for geo-localization. We demonstrate the efficacy of our method via extensive experiments and ablations on benchmark datasets. We achieve competitive performance with just 20% of training data, highlighting its effectiveness even in limited-data settings. Furthermore, we qualitatively demonstrate geo-localization using a text query by leveraging the CLIP backbone of our image encoder. The project webpage is available at: https://vicentevivan.github.io/GeoCLIP

----

## [379] RECESS Vaccine for Federated Learning: Proactive Defense Against Model Poisoning Attacks

**Authors**: *Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, Hui Li, Xiaodong Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b80fe066fdbceb3a2960117bac33917-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b80fe066fdbceb3a2960117bac33917-Abstract-Conference.html)

**Abstract**:

Model poisoning attacks greatly jeopardize the application of federated learning (FL). The effectiveness of existing defenses is susceptible to the latest model poisoning attacks, leading to a decrease in prediction accuracy. Besides, these defenses are intractable to distinguish benign outliers from malicious gradients, which further compromises the model generalization. In this work, we propose a novel defense including detection and aggregation, named RECESS, to serve as a “vaccine” for FL against model poisoning attacks. Different from the passive analysis in previous defenses, RECESS proactively queries each participating client with a delicately constructed aggregation gradient, accompanied by the detection of malicious clients according to their responses with higher accuracy. Further, RECESS adopts a newly proposed trust scoring based mechanism to robustly aggregate gradients. Rather than previous methods of scoring in each iteration, RECESS takes into account the correlation of clients’ performance over multiple iterations to estimate the trust score, bringing in a significant increase in detection fault tolerance. Finally, we extensively evaluate RECESS on typical model architectures and four datasets under various settings including white/black-box, cross-silo/device FL, etc. Experimental results show the superiority of RECESS in terms of reducing accuracy loss caused by the latest model poisoning attacks over five classic and two state-of-the-art defenses.

----

## [380] Minimum norm interpolation by perceptra: Explicit regularization and implicit bias

**Authors**: *Jiyoung Park, Ian Pelakh, Stephan Wojtowytsch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b8612e11c75456c90963fd408d75c4d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b8612e11c75456c90963fd408d75c4d-Abstract-Conference.html)

**Abstract**:

We investigate how shallow ReLU networks interpolate between known regions. Our analysis shows that empirical risk minimizers converge to a minimum norm interpolant as the number of data points and parameters tends to infinity when a weight decay regularizer is penalized with a coefficient which vanishes at a precise rate as the network width and the number of data points grow. With and without explicit regularization, we numerically study the implicit bias of common optimization algorithms towards known minimum norm interpolants.

----

## [381] Spectral Co-Distillation for Personalized Federated Learning

**Authors**: *Zihan Chen, Howard H. Yang, Tony Q. S. Quek, Kai Fong Ernest Chong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b86cf4b15cd83b6520d851eb7298228-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b86cf4b15cd83b6520d851eb7298228-Abstract-Conference.html)

**Abstract**:

Personalized federated learning (PFL) has been widely investigated to address the challenge of data heterogeneity, especially when a single generic model is inadequate in satisfying the diverse performance requirements of local clients simultaneously. Existing PFL methods are inherently based on the idea that the relations between the generic global and personalized local models are captured by the similarity of model weights. Such a similarity is primarily based on either partitioning the model architecture into generic versus personalized components or modeling client relationships via model weights. To better capture similar (yet distinct) generic versus personalized model representations, we propose $\textit{spectral distillation}$, a novel distillation method based on model spectrum information. Building upon spectral distillation, we also introduce a co-distillation framework that establishes a two-way bridge between generic and personalized model training. Moreover, to utilize the local idle time in conventional PFL, we propose a wait-free local training protocol. Through extensive experiments on multiple datasets over diverse heterogeneous data settings, we demonstrate the outperformance and efficacy of our proposed spectral co-distillation method, as well as our wait-free training protocol.

----

## [382] DVSOD: RGB-D Video Salient Object Detection

**Authors**: *Jingjing Li, Wei Ji, Size Wang, Wenbo Li, Li Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1b88e65f737256d437e56764d39ba06d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1b88e65f737256d437e56764d39ba06d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Salient object detection (SOD) aims to identify standout elements in a scene, with recent advancements primarily focused on integrating depth data (RGB-D) or temporal data from videos to enhance SOD in complex scenes. However, the unison of two types of crucial information remains largely underexplored due to data constraints. To bridge this gap, we in this work introduce the DViSal dataset, fueling further research in the emerging field of RGB-D video salient object detection (DVSOD). Our dataset features 237 diverse RGB-D videos alongside comprehensive annotations, including object and instance-level markings, as well as bounding boxes and scribbles. These resources enable a broad scope for potential research directions. We also conduct benchmarking experiments using various SOD models, affirming the efficacy of multimodal video input for salient object detection. Lastly, we highlight some intriguing findings and promising future research avenues. To foster growth in this field, our dataset and benchmark results are publicly accessible at: https://dvsod.github.io/.

----

## [383] Gradient Informed Proximal Policy Optimization

**Authors**: *Sanghyun Son, Laura Yu Zheng, Ryan Sullivan, Yi-Ling Qiao, Ming C. Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1bd8cfc0e4c53869b7f1d0ed4b1e78e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1bd8cfc0e4c53869b7f1d0ed4b1e78e1-Abstract-Conference.html)

**Abstract**:

We introduce a novel policy learning method that integrates analytical gradients from differentiable environments with the Proximal Policy Optimization (PPO) algorithm. To incorporate analytical gradients into the PPO framework, we introduce the concept of an α-policy that stands as a locally superior policy. By adaptively modifying the α value, we can effectively manage the influence of analytical policy gradients during learning. To this end, we suggest metrics for assessing the variance and bias of analytical gradients, reducing dependence on these gradients when high variance or bias is detected. Our proposed approach outperforms baseline algorithms in various scenarios, such as function optimization, physics simulations, and traffic control environments. Our code can be found online: https://github.com/SonSang/gippo.

----

## [384] SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model

**Authors**: *Di Wang, Jing Zhang, Bo Du, Minqiang Xu, Lin Liu, Dacheng Tao, Liangpei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1be3843e534ee06d3a70c7f62b983b31-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1be3843e534ee06d3a70c7f62b983b31-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The success of the Segment Anything Model (SAM) demonstrates the significance of data-centric machine learning. However, due to the difficulties and high costs associated with annotating Remote Sensing (RS) images, a large amount of valuable RS data remains unlabeled, particularly at the pixel level. In this study, we leverage SAM and existing RS object detection datasets to develop an efficient pipeline for generating a large-scale RS segmentation dataset, dubbed SAMRS. SAMRS totally possesses 105,090 images and 1,668,241 instances, surpassing existing high-resolution RS segmentation datasets in size by several orders of magnitude. It provides object category, location, and instance information that can be used for semantic segmentation, instance segmentation, and object detection, either individually or in combination. We also provide a comprehensive analysis of SAMRS from various aspects.  Moreover, preliminary experiments highlight the importance of conducting segmentation pre-training with SAMRS to address task discrepancies and alleviate the limitations posed by limited training data during fine-tuning. The code and dataset will be available at https://github.com/ViTAE-Transformer/SAMRS

----

## [385] Blockwise Parallel Transformers for Large Context Models

**Authors**: *Hao Liu, Pieter Abbeel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1bfd87d2d92f0556819467dc08034f76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1bfd87d2d92f0556819467dc08034f76-Abstract-Conference.html)

**Abstract**:

Transformers have emerged as the cornerstone of state-of-the-art natural language processing models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands posed by the self-attention mechanism and the large feedforward network in Transformers limit their ability to handle long sequences, thereby creating challenges for tasks involving multiple long sequences or long-term dependencies. We present a distinct approach, Blockwise Parallel Transformer (BPT), that leverages blockwise computation of self-attention and feedforward network fusion to minimize memory costs. By processing longer input sequences while maintaining memory efficiency, BPT enables training sequences 32 times longer than vanilla Transformers and up to 4 times longer than previous memory-efficient methods. Extensive experiments on language modeling and reinforcement learning tasks demonstrate the effectiveness of BPT in reducing memory requirements and improving performance.

----

## [386] Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization

**Authors**: *Fu Luo, Xi Lin, Fei Liu, Qingfu Zhang, Zhenkun Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html)

**Abstract**:

Neural combinatorial optimization (NCO) is a promising learning-based approach for solving challenging combinatorial optimization problems without specialized algorithm design by experts. However, most constructive NCO methods cannot solve problems with large-scale instance sizes, which significantly diminishes their usefulness for real-world applications. In this work, we propose a novel Light Encoder and Heavy Decoder (LEHD) model with a strong generalization ability to address this critical issue. The LEHD model can learn to dynamically capture the relationships between all available nodes of varying sizes, which is beneficial for model generalization to problems of various scales. Moreover, we develop a data-efficient training scheme and a flexible solution construction mechanism for the proposed LEHD model. By training on small-scale problem instances, the LEHD model can generate nearly optimal solutions for the Travelling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP) with up to 1000 nodes, and also generalizes well to solve real-world TSPLib and CVRPLib problems. These results confirm our proposed LEHD model can significantly improve the state-of-the-art performance for constructive NCO.

----

## [387] Topological Obstructions and How to Avoid Them

**Authors**: *Babak Esmaeili, Robin Walters, Heiko Zimmermann, Jan-Willem van de Meent*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c12ccfc7720f6b680edea17300bfc2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c12ccfc7720f6b680edea17300bfc2b-Abstract-Conference.html)

**Abstract**:

Incorporating geometric inductive biases into models can aid interpretability and generalization, but encoding to a specific geometric structure can be challenging due to the imposed topological constraints. In this paper, we theoretically and empirically characterize obstructions to training encoders with geometric latent spaces. We show that local optima can arise due to singularities (e.g. self-intersection) or due to an incorrect degree or winding number. We then discuss how normalizing flows can potentially circumvent these obstructions by defining multimodal variational distributions. Inspired by this observation, we propose a new flow-based model that maps data points to multimodal distributions over geometric spaces and empirically evaluate our model on 2 domains. We observe improved stability during training and a higher chance of converging to a homeomorphic encoder.

----

## [388] The Double-Edged Sword of Implicit Bias: Generalization vs Robustness in ReLU Networks

**Authors**: *Spencer Frei, Gal Vardi, Peter L. Bartlett, Nati Srebro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c26c389d60ec419fd24b5fee5b35796-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c26c389d60ec419fd24b5fee5b35796-Abstract-Conference.html)

**Abstract**:

In this work, we study the implications of the implicit bias of gradient flow on generalization and adversarial robustness in ReLU networks.  We focus on a setting where the data consists of clusters and the correlations between cluster means are small, and show that in two-layer ReLU networks gradient flow is biased towards solutions that generalize well, but are vulnerable to adversarial examples. Our results hold even in cases where the network is highly overparameterized.  Despite the potential for harmful overfitting in such settings, we prove that the implicit bias of gradient flow prevents it.  However, the implicit bias also leads to non-robust solutions (susceptible to small adversarial $\ell_2$-perturbations), even though robust networks that fit the data exist.

----

## [389] PromptRestorer: A Prompting Image Restoration Method with Degradation Perception

**Authors**: *Cong Wang, Jinshan Pan, Wei Wang, Jiangxin Dong, Mengzhu Wang, Yakun Ju, Junyang Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c364d98a5cdc426fd8c76fbb2c10e34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c364d98a5cdc426fd8c76fbb2c10e34-Abstract-Conference.html)

**Abstract**:

We show that raw degradation features can effectively guide deep restoration models, providing accurate degradation priors to facilitate better restoration. While networks that do not consider them for restoration forget gradually degradation during the learning process, model capacity is severely hindered. To address this, we propose a Prompting image Restorer, termed as PromptRestorer. Specifically, PromptRestorer contains two branches: a restoration branch and a prompting branch. The former is used to restore images, while the latter perceives degradation priors to prompt the restoration branch with reliable perceived content to guide the restoration process for better recovery. To better perceive the degradation which is extracted by a pre-trained model from given degradation observations, we propose a prompting degradation perception modulator, which adequately considers the characters of the self-attention mechanism and pixel-wise modulation, to better perceive the degradation priors from global and local perspectives. To control the propagation of the perceived content for the restoration branch, we propose gated degradation perception propagation, enabling the restoration branch to adaptively learn more useful features for better recovery. Extensive experimental results show that our PromptRestorer achieves state-of-the-art results on 4 image restoration tasks, including image deraining, deblurring, dehazing, and desnowing.

----

## [390] Beyond MLE: Convex Learning for Text Generation

**Authors**: *Chenze Shao, Zhengrui Ma, Min Zhang, Yang Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html)

**Abstract**:

Maximum likelihood estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution that best explain the observed data. In the context of text generation, MLE is often used to train generative language models, which can then be used to generate new text. However, we argue that MLE is not always necessary and optimal, especially for closed-ended text generation tasks like machine translation. In these tasks, the goal of model is to generate the most appropriate response, which does not necessarily require it to estimate the entire data distribution with MLE. To this end, we propose a novel class of training objectives based on convex functions, which enables text generation models to focus on highly probable outputs without having to estimate the entire data distribution. We investigate the theoretical properties of the optimal predicted distribution when applying convex functions to the loss, demonstrating that convex functions can sharpen the optimal distribution, thereby enabling the model to better capture outputs with high probabilities. Experiments on various text generation tasks and models show the effectiveness of our approach. It enables autoregressive models to bridge the gap between greedy and beam search, and facilitates the learning of non-autoregressive models with a maximum improvement of 9+ BLEU points. Moreover, our approach also exhibits significant impact on large language models (LLMs), substantially enhancing their generative capability on various tasks. Source code is available at \url{https://github.com/ictnlp/Convex-Learning}.

----

## [391] Bandit Task Assignment with Unknown Processing Time

**Authors**: *Shinji Ito, Daisuke Hatano, Hanna Sumita, Kei Takemura, Takuro Fukunaga, Naonori Kakimura, Ken-ichi Kawarabayashi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c5ee7343f396954377c2c16dda33a96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c5ee7343f396954377c2c16dda33a96-Abstract-Conference.html)

**Abstract**:

This study considers a novel problem setting, referred to as \textit{bandit task assignment}, that incorporates the processing time of each task in the bandit setting. In this problem setting, a player sequentially chooses a set of tasks to start so that the set of processing tasks satisfies a given combinatorial constraint. The reward and processing time for each task follow unknown distributions, values of which are revealed only after the task has been completed. The problem generalizes the stochastic combinatorial semi-bandit problem and the budget-constrained bandit problem. For this problem setting, we propose an algorithm based on upper confidence bounds~(UCB) combined with a phased-update approach. The proposed algorithm admits a gap-dependent regret upper bound of $O(MN(1/\Delta){\log T})$ and a gap-free regret upper bound of $\tilde{O}( \sqrt{MNT} )$, where $N$ is the number of the tasks, $M$ is the maximum number of tasks run at the same time, $T$ is the time horizon, and $\Delta$ is the gap between expected per-round rewards of the optimal and best suboptimal sets of tasks. These regret bounds nearly match lower bounds.

----

## [392] Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved with Text

**Authors**: *Wanrong Zhu, Jack Hessel, Anas Awadalla, Samir Yitzhak Gadre, Jesse Dodge, Alex Fang, Youngjae Yu, Ludwig Schmidt, William Yang Wang, Yejin Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c6bed78d3813886d3d72595dbecb80b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c6bed78d3813886d3d72595dbecb80b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In-context vision and language models like Flamingo support arbitrarily interleaved sequences of images and text as input.This format not only enables few-shot learning via interleaving independent supervised (image, text) examples, but also, more complex prompts involving interaction between images, e.g., ``What do image A and image B have in common?''To support this interface, pretraining occurs over web corpora that similarly contain interleaved images+text.To date, however, large-scale data of this form have not been publicly available.We release Multimodal C4, an augmentation of the popular text-only C4 corpus with images interleaved.We use a linear assignment algorithm to place images into longer bodies of text using CLIP features, a process that we show outperforms alternatives.Multimodal C4 spans everyday topics like cooking, travel, technology, etc. A manual inspection of a random sample of documents shows that a vast majority (88\%) of images are topically relevant, and that linear assignment frequently selects individual sentences specifically well-aligned with each image (80\%). After filtering NSFW images, ads, etc., the resulting corpus consists of 101.2M documents with 571M images interleaved in 43B English tokens.

----

## [393] Towards Self-Interpretable Graph-Level Anomaly Detection

**Authors**: *Yixin Liu, Kaize Ding, Qinghua Lu, Fuyi Li, Leo Yu Zhang, Shirui Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c6f06863df46de009a7a41b41c95cad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c6f06863df46de009a7a41b41c95cad-Abstract-Conference.html)

**Abstract**:

Graph-level anomaly detection (GLAD) aims to identify graphs that exhibit notable dissimilarity compared to the majority in a collection. However, current works primarily focus on evaluating graph-level abnormality while failing to provide meaningful explanations for the predictions, which largely limits their reliability and application scope. In this paper, we investigate a new challenging problem, explainable GLAD, where the learning objective is to predict the abnormality of each graph sample with corresponding explanations, i.e., the vital subgraph that leads to the predictions. To address this challenging problem, we propose a Self-Interpretable Graph aNomaly dETection model (SIGNET for short) that detects anomalous graphs as well as generates informative explanations simultaneously. Specifically, we first introduce the multi-view subgraph information bottleneck (MSIB) framework, serving as the design basis of our self-interpretable GLAD approach. This way SIGNET is able to not only measure the abnormality of each graph based on cross-view mutual information but also provide informative graph rationales by extracting bottleneck subgraphs from the input graph and its dual hypergraph in a self-supervised way. Extensive experiments on 16 datasets demonstrate the anomaly detection capability and self-interpretability of SIGNET.

----

## [394] AMAG: Additive, Multiplicative and Adaptive Graph Neural Network For Forecasting Neuron Activity

**Authors**: *Jingyuan Li, Leo Scholl, Trung Le, Pavithra Rajeswaran, Amy Orsborn, Eli Shlizerman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c70ba3591d0694a535089e1c25888d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c70ba3591d0694a535089e1c25888d7-Abstract-Conference.html)

**Abstract**:

Latent Variable Models (LVMs) propose to model the dynamics of neural populations by capturing low-dimensional structures that represent features involved in neural activity. Recent LVMs are based on deep learning methodology where a deep neural network is trained to reconstruct the same neural activity given as input and as a result to build the latent representation. Without taking past or future activity into account such a task is non-causal. In contrast, the task of forecasting neural activity based on given input extends the reconstruction task. LVMs that are trained on such a task could potentially capture temporal causality constraints within its latent representation. Forecasting has received less attention than reconstruction due to recording challenges such as limited neural measurements and trials. In this work, we address modeling neural population dynamics via the forecasting task and improve forecasting performance by including a prior, which consists of pairwise neural unit interaction as a multivariate dynamic system. Our proposed model---Additive, Multiplicative, and Adaptive Graph Neural Network (AMAG)---leverages additive and multiplicative message-passing operations analogous to the interactions in neuronal systems and adaptively learns the interaction among neural units to forecast their future activity. We demonstrate the advantage of AMAG compared to non-GNN based methods on synthetic data and multiple modalities of neural recordings (field potentials from penetrating electrodes or surface-level micro-electrocorticography) from four rhesus macaques. Our results show the ability of AMAG to recover ground truth spatial interactions and yield estimation for future dynamics of the neural population.

----

## [395] PackQViT: Faster Sub-8-bit Vision Transformers via Full and Packed Quantization on the Mobile

**Authors**: *Peiyan Dong, Lei Lu, Chao Wu, Cheng Lyu, Geng Yuan, Hao Tang, Yanzhi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1c92edb990a05f2269f0cc3afbb4c952-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1c92edb990a05f2269f0cc3afbb4c952-Abstract-Conference.html)

**Abstract**:

While Vision Transformers (ViTs) have undoubtedly made impressive strides in computer vision (CV), their intricate network structures necessitate substantial computation and memory resources. A decision-making process for CV tasks typically entails performing computations with low latency, which is a tricky problem for ViT models.Model quantization is a widely-used technique to optimize the hardware efficiency of deep neural networks.Full quantization under Sub-8-bit precision, in particular, is a promising solution to reduce inference latency significantly. Unfortunately, current commodity hardware, such as CPUs and GPUs, still struggles to efficiently execute these sub-8-bit quantized networks, as their SIMD instructions only support a granularity of 8 bits or wider.Also, there is a scarcity of literature that presents a full quantization paradigm for ViTs.In this paper, we propose an activation-aware fully sub-8-bit quantization-aware training (QAT) framework called PackQViT for efficient yet accurate ViT acceleration on mobile devices to facilitate real-time AI-powered decision-making.Specifically, in revisiting data activation within the ViT dataflow, two characteristics are relevant to quantization strategy and precision: the long-tailed distribution and systematic channel-wise outliers.In response, we employ either log2 quantization or clipping to address the long-tailed distribution and incorporate outlier-aware training for residual link quantization to regulate the various channel-wise outliers more consistently.Notably, due to the systematic fixed pattern, outlier-aware training approach can predict the channel indices and regularized scales of outliers in advance, thus avoiding the runtime data-adaptive selection during inference.Furthermore, we employ Int-$2^{n}$-Softmax, Int-LayerNorm, and Integer GELU to enable integer-only computation flow. Finally, we develop a SIMD-based 4-bit packed multiplier to achieve end-to-end ViT acceleration on mobile phones.Compared to prior studies on ViT quantization using 8-bit precision, PackQViT surpasses other works by an improved accuracy ranging from 0.4\% to 17.9\% for various widely used ViTs on ImageNet dataset; under 4-bit precision, PackQViT demonstrates 0.4%$\sim$2.8% higher accuracy. Compared to the baseline multiplier, our implementations on the Realme GT Android smartphone with Snapdragon 870 SoC CPU achieve 2.6x$\sim$3.7x speedup under 8-bit scenario and 3.8x$\sim$5.9x speedup under 4-bit which ensures practical real-time performance.

----

## [396] Extending the Design Space of Graph Neural Networks by Rethinking Folklore Weisfeiler-Lehman

**Authors**: *Jiarui Feng, Lecheng Kong, Hao Liu, Dacheng Tao, Fuhai Li, Muhan Zhang, Yixin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1cac8326ce3fbe79171db9754211530c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1cac8326ce3fbe79171db9754211530c-Abstract-Conference.html)

**Abstract**:

Message passing neural networks (MPNNs) have emerged as the most popular framework of graph neural networks (GNNs) in recent years. However, their expressive power is limited by the 1-dimensional Weisfeiler-Lehman (1-WL) test. Some works are inspired by $k$-WL/FWL (Folklore WL) and design the corresponding neural versions. Despite the high expressive power, there are serious limitations in this line of research. In particular, (1) $k$-WL/FWL requires at least $O(n^k)$ space complexity, which is impractical for large graphs even when $k=3$; (2) The design space of $k$-WL/FWL is rigid, with the only adjustable hyper-parameter being $k$. To tackle the first limitation, we propose an extension, $(k, t)$-FWL. We theoretically prove that even if we fix the space complexity to $O(n^k)$ (for any $k \geq 2$) in $(k, t)$-FWL, we can construct an expressiveness hierarchy up to solving the graph isomorphism problem. To tackle the second problem, we propose $k$-FWL+, which considers any equivariant set as neighbors instead of all nodes, thereby greatly expanding the design space of $k$-FWL. Combining these two modifications results in a flexible and powerful framework $(k, t)$-FWL+. We demonstrate $(k, t)$-FWL+ can implement most existing models with matching expressiveness. We then introduce an instance of $(k,t)$-FWL+ called Neighborhood$^2$-FWL (N$^2$-FWL), which is practically and theoretically sound. We prove that N$^2$-FWL is no less powerful than 3-WL, and can encode many substructures while only requiring $O(n^2)$ space. Finally, we design its neural version named **N$^2$-GNN** and evaluate its performance on various tasks. N$^2$-GNN achieves record-breaking results on ZINC-Subset (**0.059**), outperforming previous SOTA results by 10.6\%. Moreover, N$^2$-GNN achieves new SOTA results on the BREC dataset (**71.8\%**) among all existing high-expressive GNN methods.

----

## [397] Off-Policy Evaluation for Human Feedback

**Authors**: *Qitong Gao, Ge Gao, Juncheng Dong, Vahid Tarokh, Min Chi, Miroslav Pajic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1cb57fcf7ff3f6d37eebae5becc9ea6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1cb57fcf7ff3f6d37eebae5becc9ea6d-Abstract-Conference.html)

**Abstract**:

Off-policy evaluation (OPE) is important for closing the gap between offline training and evaluation of reinforcement learning (RL), by estimating performance and/or rank of target (evaluation) policies using offline trajectories only. It can improve the safety and efficiency of data collection and policy testing procedures in situations where online deployments are expensive, such as healthcare. However, existing OPE methods fall short in estimating human feedback (HF) signals, as HF may be conditioned over multiple underlying factors and are only sparsely available; as opposed to the agent-defined environmental rewards (used in policy optimization), which are usually determined over parametric functions or distributions. Consequently, the nature of HF signals makes extrapolating accurate OPE estimations to be challenging. To resolve this, we introduce an OPE for HF (OPEHF) framework that revives existing OPE methods in order to accurately evaluate the HF signals. Specifically, we develop an immediate human reward (IHR) reconstruction approach, regularized by environmental knowledge distilled in a latent space that captures the underlying dynamics of state transitions as well as issuing HF signals. Our approach has been tested over two real-world experiments, adaptive in-vivo neurostimulation and intelligent tutoring, and a simulation environment (visual Q&A). Results show that our approach significantly improves the performance toward estimating HF signals accurately, compared to directly applying (variants of) existing OPE methods.

----

## [398] Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion

**Authors**: *Yash Bhalgat, Iro Laina, João F. Henriques, Andrea Vedaldi, Andrew Zisserman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1cb5b3d64bdf3c6642c8d9a8fbecd019-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1cb5b3d64bdf3c6642c8d9a8fbecd019-Abstract-Conference.html)

**Abstract**:

Instance segmentation in 3D is a challenging task due to the lack of large-scale annotated datasets. In this paper, we show that this task can be addressed effectively by leveraging instead 2D pre-trained models for instance segmentation. We propose a novel approach to lift 2D segments to 3D and fuse them by means of a neural field representation, which encourages multi-view consistency across frames. The core of our approach is a slow-fast clustering objective function, which is scalable and well-suited for scenes with a large number of objects. Unlike previous approaches, our method does not require an upper bound on the number of objects or object tracking across frames. To demonstrate the scalability of the slow-fast clustering, we create a new semi-realistic dataset called the Messy Rooms dataset, which features scenes with up to 500 objects per scene. Our approach outperforms the state-of-the-art on challenging scenes from the ScanNet, Hypersim, and Replica datasets, as well as on our newly created Messy Rooms dataset, demonstrating the effectiveness and scalability of our slow-fast clustering method.

----

## [399] GALOPA: Graph Transport Learning with Optimal Plan Alignment

**Authors**: *Yejiang Wang, Yuhai Zhao, Daniel Zhengkui Wang, Ling Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/1d35af80e775e342f4cd3792e4405837-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/1d35af80e775e342f4cd3792e4405837-Abstract-Conference.html)

**Abstract**:

Self-supervised learning on graph aims to learn graph representations in an unsupervised manner. While graph contrastive learning (GCL - relying on graph augmentation for creating perturbation views of anchor graphs and maximizing/minimizing similarity for positive/negative pairs) is a popular self-supervised method, it faces challenges in finding label-invariant augmented graphs and determining the exact extent of similarity between sample pairs to be achieved. In this work, we propose an alternative self-supervised solution that (i) goes beyond the label invariance assumption without distinguishing between positive/negative samples, (ii) can calibrate the encoder for preserving not only the structural information inside the graph, but the matching information between different graphs, (iii) learns isometric embeddings that preserve the distance between graphs, a by-product of our objective. Motivated by optimal transport theory, this scheme relays on an observation that the optimal transport plans between node representations at the output space, which measure the matching probability between two distributions, should be consistent to the plans between the corresponding graphs at the input space. The experimental findings include: (i) The plan alignment strategy significantly outperforms the counterpart using the transport distance; (ii) The proposed model shows superior performance using only node attributes as calibration signals, without relying on edge information; (iii) Our model maintains robust results even under high perturbation rates; (iv) Extensive experiments on various benchmarks validate the effectiveness of the proposed method.

----



[Go to the previous page](NIPS-2023-list1.md)

[Go to the next page](NIPS-2023-list3.md)

[Go to the catalog section](README.md)