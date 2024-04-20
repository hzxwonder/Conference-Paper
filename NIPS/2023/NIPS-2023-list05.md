## [800] RVD: A Handheld Device-Based Fundus Video Dataset for Retinal Vessel Segmentation

**Authors**: *Md. Wahiduzzaman Khan, Hongwei Sheng, Hu Zhang, Heming Du, Sen Wang, Minas Theodore Coroneo, Farshid Hajati, Sahar Shariflou, Michael Kalloniatis, Jack Phu, Ashish Agar, Zi Huang, S. Mojtaba Golzan, Xin Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a71ee306d6991f2f87dd414e0bdf851-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a71ee306d6991f2f87dd414e0bdf851-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Retinal vessel segmentation is generally grounded in image-based datasets collected with bench-top devices. The static images naturally lose the dynamic characteristics of retina fluctuation, resulting in diminished dataset richness, and the usage of bench-top devices further restricts dataset scalability due to its limited accessibility. Considering these limitations, we introduce the first video-based retinal dataset by employing handheld devices for data acquisition. The dataset comprises 635 smartphone-based fundus videos collected from four different clinics, involving 415 patients from 50 to 75 years old. It delivers comprehensive and precise annotations of retinal structures in both spatial and temporal dimensions, aiming to advance the landscape of vasculature segmentation. Specifically, the dataset provides three levels of spatial annotations: binary vessel masks for overall retinal structure delineation, general vein-artery masks for distinguishing the vein and artery, and fine-grained vein-artery masks for further characterizing the granularities of each artery and vein. In addition, the dataset offers temporal annotations that capture the vessel pulsation characteristics, assisting in detecting ocular diseases that require fine-grained recognition of hemodynamic fluctuation. In application, our dataset exhibits a significant domain shift with respect to data captured by bench-top devices, thus posing great challenges to existing methods. Thanks to rich annotations and data scales, our dataset potentially paves the path for more advanced retinal analysis and accurate disease diagnosis. In the experiments, we provide evaluation metrics and benchmark results on our dataset, reflecting both the potential and challenges it offers for vessel segmentation tasks. We hope this challenging dataset would significantly contribute to the development of eye disease diagnosis and early prevention.

----

## [801] LayoutGPT: Compositional Visual Planning and Generation with Large Language Models

**Authors**: *Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun R. Akula, Xuehai He, Sugato Basu, Xin Eric Wang, William Yang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html)

**Abstract**:

Attaining a high degree of user controllability in visual generation often requires intricate, fine-grained inputs like layouts. However, such inputs impose a substantial burden on users when compared to simple text inputs. To address the issue, we study how Large Language Models (LLMs) can serve as visual planners by generating layouts from text conditions, and thus collaborate with visual generative models. We propose LayoutGPT, a method to compose in-context visual demonstrations in style sheet language to enhance visual planning skills of LLMs. We show that LayoutGPT can generate plausible layouts in multiple domains, ranging from 2D images to 3D indoor scenes. LayoutGPT also shows superior performance in converting challenging language concepts like numerical and spatial relations to layout arrangements for faithful text-to-image generation. When combined with a downstream image generation model, LayoutGPT outperforms text-to-image models/systems by 20-40\% and achieves comparable performance as human users in designing visual layouts for numerical and spatial correctness. Lastly, LayoutGPT achieves comparable performance to supervised methods in 3D indoor scene synthesis, demonstrating its effectiveness and potential in multiple visual domains.

----

## [802] Data Pruning via Moving-one-Sample-out

**Authors**: *Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3abe23bf7e295b44369c24465d68987a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3abe23bf7e295b44369c24465d68987a-Abstract-Conference.html)

**Abstract**:

In this paper, we propose a novel data-pruning approach called moving-one-sample-out (MoSo), which aims to identify and remove the least informative samples from the training set. The core insight behind MoSo is to determine the importance of each sample by assessing its impact on the optimal empirical risk. This is achieved by measuring the extent to which the empirical risk changes when a particular sample is excluded from the training set. Instead of using the computationally expensive leaving-one-out-retraining procedure, we propose an efficient first-order approximator that only requires gradient information from different training stages. The key idea behind our approximation is that samples with gradients that are consistently aligned with the average gradient of the training set are more informative and should receive higher scores, which could be intuitively understood as follows: if the gradient from a specific sample is consistent with the average gradient vector, it implies that optimizing the network using the sample will yield a similar effect on all remaining samples.  Experimental results demonstrate that MoSo effectively mitigates severe performance degradation at high pruning ratios and achieves satisfactory performance across various settings. Experimental results demonstrate that MoSo effectively mitigates severe performance degradation at high pruning ratios and outperforms state-of-the-art methods by a large margin across various settings.

----

## [803] Alternation makes the adversary weaker in two-player games

**Authors**: *Volkan Cevher, Ashok Cutkosky, Ali Kavis, Georgios Piliouras, Stratis Skoulakis, Luca Viano*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3acb49252187efa352a1ae0e4b066ced-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3acb49252187efa352a1ae0e4b066ced-Abstract-Conference.html)

**Abstract**:

Motivated by alternating game-play in two-player games, we study an altenating variant of the \textit{Online Linear Optimization} (OLO). In alternating OLO,  a \textit{learner} at each round $t \in [n]$ selects a vector $x^t$ and then an \textit{adversary} selects a cost-vector $c^t \in [-1,1]^n$. The learner then experiences cost $(c^t + c^{t-1})^\top x^t$ instead of $(c^t)^\top x^t$ as in standard OLO. We establish that under this small twist, the $\Omega(\sqrt{T})$ lower bound on the regret is no longer valid. More precisely, we present two online learning algorithms for alternating OLO that respectively admit $\mathcal{O}((\log n)^{4/3} T^{1/3})$ regret for the $n$-dimensional simplex and $\mathcal{O}(\rho \log T)$ regret for the ball of radius $\rho>0$. Our results imply that in alternating game-play, an agent can always guarantee $\mathcal{\tilde{O}}((\log n)^{4/3} T^{1/3})$ regardless the strategies of the other agent while the regret bound improves to $\mathcal{O}(\log T)$ in case the agent admits only two actions.

----

## [804] Spuriosity Didn't Kill the Classifier: Using Invariant Predictions to Harness Spurious Features

**Authors**: *Cian Eastwood, Shashank Singh, Andrei Liviu Nicolicioiu, Marin Vlastelica Pogancic, Julius von Kügelgen, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3acbe9dc3a1e8d48a57b16e9aef91879-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3acbe9dc3a1e8d48a57b16e9aef91879-Abstract-Conference.html)

**Abstract**:

To avoid failures on out-of-distribution data, recent works have sought to extract features that have an invariant or stable relationship with the label across domains, discarding "spurious" or unstable features whose relationship with the label changes across domains. However, unstable features often carry complementary information that could boost performance if used correctly in the test domain. In this work, we show how this can be done without test-domain labels. In particular, we prove that pseudo-labels based on stable features provide sufficient guidance for doing so, provided that stable and unstable features are conditionally independent given the label. Based on this theoretical insight, we propose Stable Feature Boosting (SFB), an algorithm for: (i) learning a predictor that separates stable and conditionally-independent unstable features; and (ii) using the stable-feature predictions to adapt the unstable-feature predictions in the test domain. Theoretically, we prove that SFB can learn an asymptotically-optimal predictor without test-domain labels. Empirically, we demonstrate the effectiveness of SFB on real and synthetic data.

----

## [805] A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints

**Authors**: *Kareem Ahmed, Kai-Wei Chang, Guy Van den Broeck*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3accfe8332366a6f740d8740cd4cd653-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3accfe8332366a6f740d8740cd4cd653-Abstract-Conference.html)

**Abstract**:

Neuro-symbolic AI bridges the gap between purely symbolic and neural approaches to learning. This often requires maximizing the likelihood of a symbolic constraint w.r.t the neural network's output distribution. Such output distributions are typically assumed to be fully-factorized. This limits the applicability of neuro-symbolic learning to the more expressive auto-regressive distributions, e.g., transformers. Under such distributions, computing the likelihood of even simple constraints is #P-hard. Instead of attempting to enforce the constraint on the entire likelihood distribution, we propose to do so on a random, local approximation thereof. More precisely, we approximate the likelihood of the constraint with the pseudolikelihood of the constraint centered around a model sample. Our approach is factorizable, allowing us to reuse solutions to sub-problems---a main tenet for the efficient computation of neuro-symbolic losses. It also provides a local, high fidelity approximation of the likelihood: it exhibits low entropy and KL-divergence around the model sample. We tested our approach on Sudoku and shortest-path prediction cast as auto-regressive generation, and observe that we greatly improve upon the base model's ability to predict logically-consistent outputs. We also tested our approach on the task of detoxifying large language models. We observe that using a simple constraint disallowing a list of toxic words, we are able to steer the model's outputs away from toxic generations, achieving SoTA compared to previous approaches.

----

## [806] Physics-Informed Bayesian Optimization of Variational Quantum Circuits

**Authors**: *Kim Nicoli, Christopher J. Anders, Lena Funcke, Tobias Hartung, Karl Jansen, Stefan Kühn, Klaus-Robert Müller, Paolo Stornati, Pan Kessel, Shinichi Nakajima*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3adb85a348a18cdd74ce99fbbab20301-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3adb85a348a18cdd74ce99fbbab20301-Abstract-Conference.html)

**Abstract**:

In this paper, we propose a novel and powerful method to harness Bayesian optimization for variational quantum eigensolvers (VQEs) - a hybrid quantum-classical protocol used to approximate the ground state of a quantum Hamiltonian. Specifically, we derive a VQE-kernel which incorporates important prior information about quantum circuits: the kernel feature map of the VQE-kernel exactly matches the known functional form of the VQE's objective function and thereby significantly reduces the posterior uncertainty.Moreover, we propose a novel acquisition function for Bayesian optimization called \emph{Expected Maximum Improvement over Confident Regions} (EMICoRe) which can actively exploit the inductive bias of the VQE-kernel by treating regions with low predictive uncertainty as indirectly "observed". As a result, observations at as few as three points in the search domain are sufficient to determine the complete objective function along an entire one-dimensional subspace of the optimization landscape. Our numerical experiments demonstrate that our approach improves over state-of-the-art baselines.

----

## [807] Rubik's Cube: High-Order Channel Interactions with a Hierarchical Receptive Field

**Authors**: *Naishan Zheng, Man Zhou, Chong Zhou, Chen Change Loy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ae86071c169649bff21188c536163dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ae86071c169649bff21188c536163dc-Abstract-Conference.html)

**Abstract**:

Image restoration techniques, spanning from the convolution to the transformer paradigm, have demonstrated robust spatial representation capabilities to deliver high-quality performance.Yet, many of these methods, such as convolution and the Feed Forward Network (FFN) structure of transformers, primarily leverage the basic first-order channel interactions and have not maximized the potential benefits of higher-order modeling. To address this limitation, our research dives into understanding relationships within the channel dimension and introduces a simple yet efficient, high-order channel-wise operator tailored for image restoration. Instead of merely mimicking high-order spatial interaction, our approach offers several added benefits: Efficiency: It adheres to the zero-FLOP and zero-parameter principle, using a spatial-shifting mechanism across channel-wise groups. Simplicity: It turns the favorable channel interaction and aggregation capabilities into element-wise multiplications and convolution units with $1 \times 1$ kernel. Our new formulation expands the first-order channel-wise interactions seen in previous works to arbitrary high orders, generating a hierarchical receptive field akin to a Rubik's cube through the combined action of shifting and interactions. Furthermore, our proposed Rubik's cube convolution is a flexible operator that can be incorporated into existing image restoration networks, serving as a drop-in replacement for the standard convolution unit with fewer parameters overhead. We conducted experiments across various low-level vision tasks, including image denoising, low-light image enhancement, guided image super-resolution, and image de-blurring. The results consistently demonstrate that our Rubik's cube operator enhances performance across all tasks. Code is publicly available at https://github.com/zheng980629/RubikCube.

----

## [808] Closing the Computational-Statistical Gap in Best Arm Identification for Combinatorial Semi-bandits

**Authors**: *Ruo-Chun Tzeng, Po-An Wang, Alexandre Proutière, Chi-Jen Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ae8a7d6fc6d0d45e7c1ad9d4b063a01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ae8a7d6fc6d0d45e7c1ad9d4b063a01-Abstract-Conference.html)

**Abstract**:

We study the best arm identification problem in combinatorial semi-bandits in the fixed confidence setting. We present Perturbed Frank-Wolfe Sampling (P-FWS), an algorithm that (i) runs in polynomial time, (ii) achieves the instance-specific minimal sample complexity in the high confidence regime, and (iii) enjoys polynomial sample complexity guarantees in the moderate confidence regime. To our best knowledge, existing algorithms cannot achieve (ii) and (iii) simultaneously in vanilla bandits. With P-FWS, we close the computational-statistical gap in best arm identification in combinatorial semi-bandits. The design of P-FWS starts from the optimization problem that defines the information-theoretical and instance-specific sample complexity lower bound. P-FWS solves this problem in an online manner using, in each round, a single iteration of the Frank-Wolfe algorithm. Structural properties of the problem are leveraged to make the P-FWS successive updates computationally efficient. In turn, P-FWS only relies on a simple linear maximization oracle.

----

## [809] Imitation Learning from Imperfection: Theoretical Justifications and Algorithms

**Authors**: *Ziniu Li, Tian Xu, Zeyu Qin, Yang Yu, Zhi-Quan Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3af25aa3de8b7b02ddbd1b6be5031be8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3af25aa3de8b7b02ddbd1b6be5031be8-Abstract-Conference.html)

**Abstract**:

Imitation learning (IL) algorithms excel in acquiring high-quality policies from expert data for sequential decision-making tasks. But, their effectiveness is hampered when faced with limited expert data. To tackle this challenge, a novel framework called (offline) IL with supplementary data has been proposed, which enhances learning by incorporating an additional yet imperfect dataset obtained inexpensively from sub-optimal policies. Nonetheless, learning becomes challenging due to the potential inclusion of out-of-expert-distribution samples. In this work, we propose a mathematical formalization of this framework, uncovering its limitations. Our theoretical analysis reveals that a naive approach—applying the behavioral cloning (BC) algorithm concept to the combined set of expert and supplementary data—may fall short of vanilla BC, which solely relies on expert data. This deficiency arises due to the distribution shift between the two data sources. To address this issue, we propose a new importance-sampling-based technique for selecting data within the expert distribution. We prove that the proposed method eliminates the gap of the naive approach, highlighting its efficacy when handling imperfect data. Empirical studies demonstrate that our method outperforms previous state-of-the-art methods in tasks including robotic locomotion control, Atari video games, and image classification. Overall, our work underscores the potential of improving IL by leveraging diverse data sources through effective data selection.

----

## [810] Detection Based Part-level Articulated Object Reconstruction from Single RGBD Image

**Authors**: *Yuki Kawana, Tatsuya Harada*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3af8c40dcf1bc94fa570a5e42edf219d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3af8c40dcf1bc94fa570a5e42edf219d-Abstract-Conference.html)

**Abstract**:

We propose an end-to-end trainable, cross-category method for reconstructing multiple man-made articulated objects from a single RGBD image, focusing on part-level shape reconstruction and pose and kinematics estimation. We depart from previous works that rely on learning instance-level latent space, focusing on man-made articulated objects with predefined part counts. Instead, we propose a novel alternative approach that employs part-level representation, representing instances as combinations of detected parts. While our detect-then-group approach effectively handles instances with diverse part structures and various part counts, it faces issues of false positives, varying part sizes and scales, and an increasing model size due to end-to-end training. To address these challenges, we propose 1) test-time kinematics-aware part fusion to improve detection performance while suppressing false positives, 2) anisotropic scale normalization for part shape learning to accommodate various part sizes and scales, and 3) a balancing strategy for cross-refinement between feature space and output space to improve part detection while maintaining model size. Evaluation on both synthetic and real data demonstrates that our method successfully reconstructs variously structured multiple instances that previous works cannot handle, and outperforms prior works in shape reconstruction and kinematics estimation.

----

## [811] FlatMatch: Bridging Labeled Data and Unlabeled Data with Cross-Sharpness for Semi-Supervised Learning

**Authors**: *Zhuo Huang, Li Shen, Jun Yu, Bo Han, Tongliang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b11c5cc84b6da2838db348b37dbd1a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b11c5cc84b6da2838db348b37dbd1a2-Abstract-Conference.html)

**Abstract**:

Semi-Supervised Learning (SSL) has been an effective way to leverage abundant unlabeled data with extremely scarce labeled data. However, most SSL methods are commonly based on instance-wise consistency between different data transformations. Therefore, the label guidance on labeled data is hard to be propagated to unlabeled data. Consequently, the learning process on labeled data is much faster than on unlabeled data which is likely to fall into a local minima that does not favor unlabeled data, leading to sub-optimal generalization performance. In this paper, we propose FlatMatch which minimizes a cross-sharpness measure to ensure consistent learning performance between the two datasets. Specifically, we increase the empirical risk on labeled data to obtain a worst-case model which is a failure case needing to be enhanced. Then, by leveraging the richness of unlabeled data, we penalize the prediction difference (i.e., cross-sharpness) between the worst-case model and the original model so that the learning direction is beneficial to generalization on unlabeled data. Therefore, we can calibrate the learning process without being limited to insufficient label information. As a result, the mismatched learning performance can be mitigated, further enabling the effective exploitation of unlabeled data and improving SSL performance. Through comprehensive validation, we show FlatMatch achieves state-of-the-art results in many SSL settings.

----

## [812] Neural Sculpting: Uncovering hierarchically modular task structure in neural networks through pruning and network analysis

**Authors**: *Shreyas Malakarjun Patil, Loizos Michael, Constantine Dovrolis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b1675de6b49cc00084374213f8c38ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b1675de6b49cc00084374213f8c38ae-Abstract-Conference.html)

**Abstract**:

Natural target functions and tasks typically exhibit hierarchical modularity -- they can be broken down into simpler sub-functions that are organized in a hierarchy. Such sub-functions have two important features: they have a distinct set of inputs (input-separability) and they are reused as inputs higher in the hierarchy (reusability). Previous studies have established that hierarchically modular neural networks, which are inherently sparse, offer benefits such as learning efficiency, generalization, multi-task learning, and transfer. However, identifying the underlying sub-functions and their hierarchical structure for a given task can be challenging. The high-level question in this work is: if we learn a task using a sufficiently deep neural network, how can we uncover the underlying hierarchy of sub-functions in that task? As a starting point, we examine the domain of Boolean functions, where it is easier to determine whether a task is hierarchically modular. We propose an approach based on iterative unit and edge pruning (during training), combined with network analysis for module detection and hierarchy inference. Finally, we demonstrate that this method can uncover the hierarchical modularity of a wide range of Boolean functions and two vision tasks based on the MNIST digits dataset.

----

## [813] Elastic Decision Transformer

**Authors**: *Yueh-Hua Wu, Xiaolong Wang, Masashi Hamaya*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b3889d313ba9476c12c2d77ea66b24f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b3889d313ba9476c12c2d77ea66b24f-Abstract-Conference.html)

**Abstract**:

This paper introduces Elastic Decision Transformer (EDT), a significant advancement over the existing Decision Transformer (DT) and its variants. Although DT purports to generate an optimal trajectory, empirical evidence suggests it struggles with trajectory stitching, a process involving the generation of an optimal or near-optimal trajectory from the best parts of a set of sub-optimal trajectories. The proposed EDT differentiates itself by facilitating trajectory stitching during action inference at test time, achieved by adjusting the history length maintained in DT. Further, the EDT optimizes the trajectory by retaining a longer history when the previous trajectory is optimal and a shorter one when it is sub-optimal, enabling it to "stitch" with a more optimal trajectory. Extensive experimentation demonstrates EDT's ability to bridge the performance gap between DT-based and Q Learning-based approaches. In particular, the EDT outperforms Q Learning-based methods in a multi-task regime on the D4RL locomotion benchmark and Atari games.

----

## [814] Asymptotically Optimal Quantile Pure Exploration for Infinite-Armed Bandits

**Authors**: *Evelyn Xiao-Yue Gong, Mark Sellke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html)

**Abstract**:

We study pure exploration with infinitely many bandit arms generated \iid from an unknown distribution. Our goal is to efficiently select a single high quality arm whose average reward is, with probability $1-\delta$, within $\varepsilon$ of being with the top $\eta$-fraction of arms; this is a natural adaptation of the classical PAC guarantee for infinite action sets. We consider both the fixed confidence and fixed budget settings, aiming respectively for optimal \emph{expected} and \emph{fixed} sample complexity.For fixed confidence, we give an algorithm with expected sample complexity $O\left(\frac{\log (1/\eta)\log (1/\delta)}{\eta\varepsilon^2}\right)$. This is optimal except for the $\log (1/\eta)$ factor, and the $\delta$-dependence closes a quadratic gap in the literature. For fixed budget, we show the asymptotically optimal sample complexity as $\delta\to 0$ is $c^{-1}\log(1/\delta)\big(\log\log(1/\delta)\big)^2$ to leading order; equivalently, the optimal failure probability with exactly $N$ samples decays as $\exp\big(-(1\pm o(1))\frac{cN}{\log^2 N}\big)$.The value of $c$ depends explicitly on the problem parameters (including the unknown arm distribution) through a certain Fisher information distance. Even the strictly super-linear dependence on $\log(1/\delta)$ was not known and resolves a question of Grossman-Moshkovitz (FOCS 2015).

----

## [815] Learning Probabilistic Symmetrization for Architecture Agnostic Equivariance

**Authors**: *Jinwoo Kim, Dat Nguyen, Ayhan Suleymanzade, Hyeokjun An, Seunghoon Hong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b5c7c9c5c7bd77eb73d0baec7a07165-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b5c7c9c5c7bd77eb73d0baec7a07165-Abstract-Conference.html)

**Abstract**:

We present a novel framework to overcome the limitations of equivariant architectures in learning functions with group symmetries. In contrary to equivariant architectures, we use an arbitrary base model such as an MLP or a transformer and symmetrize it to be equivariant to the given group by employing a small equivariant network that parameterizes the probabilistic distribution underlying the symmetrization. The distribution is end-to-end trained with the base model which can maximize performance while reducing sample complexity of symmetrization. We show that this approach ensures not only equivariance to given group but also universal approximation capability in expectation. We implement our method on various base models, including patch-based transformers that can be initialized from pretrained vision transformers, and test them for a wide range of symmetry groups including permutation and Euclidean groups and their combinations. Empirical tests show competitive results against tailored equivariant architectures, suggesting the potential for learning equivariant functions for diverse groups using a non-equivariant universal base architecture. We further show evidence of enhanced learning in symmetric modalities, like graphs, when pretrained from non-symmetric modalities, like vision. Code is available at https://github.com/jw9730/lps.

----

## [816] Distributionally Robust Linear Quadratic Control

**Authors**: *Bahar Taskesen, Dan A. Iancu, Çagil Koçyigit, Daniel Kuhn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b7a66b2d1258e892c89f485b8f896e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b7a66b2d1258e892c89f485b8f896e0-Abstract-Conference.html)

**Abstract**:

Linear-Quadratic-Gaussian (LQG) control is a fundamental control paradigm that is studied in various fields such as engineering, computer science, economics, and neuroscience. It involves controlling a system with linear dynamics and imperfect observations, subject to additive noise, with the goal of minimizing a quadratic cost function for the state and control variables. In this work, we consider a generalization of the discrete-time, finite-horizon LQG problem, where the noise distributions are unknown and belong to Wasserstein ambiguity sets centered at nominal (Gaussian) distributions. The objective is to minimize a worst-case cost across all distributions in the ambiguity set, including non-Gaussian distributions. Despite the added complexity, we prove that a control policy that is linear in the observations is optimal for this problem, as in the classic LQG problem. We propose a numerical solution method that efficiently characterizes this optimal control policy. Our method uses the Frank-Wolfe algorithm to identify the least-favorable distributions within the Wasserstein ambiguity sets and computes the controller's optimal policy using Kalman filter estimation under these distributions.

----

## [817] Fully Dynamic k-Clustering in Õ(k) Update Time

**Authors**: *Sayan Bhattacharya, Martín Costa, Silvio Lattanzi, Nikos Parotsidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b7ba46201bf15e5c3935272afae50db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b7ba46201bf15e5c3935272afae50db-Abstract-Conference.html)

**Abstract**:

We present a $O(1)$-approximate fully dynamic algorithm for the $k$-median and $k$-means problems on metric spaces with amortized update time $\tilde O(k)$ and worst-case query time $\tilde O(k^2)$. We complement our theoretical analysis with the first in-depth experimental study for the dynamic $k$-median problem on general metrics, focusing on comparing our dynamic algorithm to the current state-of-the-art by Henzinger and Kale [ESA'20]. Finally, we also provide a lower bound for dynamic $k$-median which shows that any $O(1)$-approximate algorithm with $\tilde O(\text{poly}(k))$ query time must have $\tilde \Omega(k)$ amortized update time, even in the incremental setting.

----

## [818] FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models

**Authors**: *Lihe Yang, Xiaogang Xu, Bingyi Kang, Yinghuan Shi, Hengshuang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html)

**Abstract**:

Semantic segmentation has witnessed tremendous progress due to the proposal of various advanced network architectures. However, they are extremely hungry for delicate annotations to train, and the acquisition is laborious and unaffordable. Therefore, we present FreeMask in this work, which resorts to synthetic images from generative models to ease the burden of both data collection and annotation procedures. Concretely, we first synthesize abundant training images conditioned on the semantic masks provided by realistic datasets. This yields extra well-aligned image-mask training pairs for semantic segmentation models. We surprisingly observe that, solely trained with synthetic images, we already achieve comparable performance with real ones (e.g., 48.3 vs. 48.5 mIoU on ADE20K, and 49.3 vs. 50.5 on COCO-Stuff). Then, we investigate the role of synthetic images by joint training with real images, or pre-training for real images. Meantime, we design a robust filtering principle to suppress incorrectly synthesized regions. In addition, we propose to inequally treat different semantic masks to prioritize those harder ones and sample more corresponding synthetic images for them. As a result, either jointly trained or pre-trained with our filtered and re-sampled synthesized images, segmentation models can be greatly enhanced, e.g., from 48.7 to 52.0 on ADE20K.

----

## [819] RS-Del: Edit Distance Robustness Certificates for Sequence Classifiers via Randomized Deletion

**Authors**: *Zhuoqun Huang, Neil G. Marchant, Keane Lucas, Lujo Bauer, Olga Ohrimenko, Benjamin I. P. Rubinstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ba82362eb0aa75487069f19fde794fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ba82362eb0aa75487069f19fde794fe-Abstract-Conference.html)

**Abstract**:

Randomized smoothing is a leading approach for constructing classifiers that are certifiably robust against adversarial examples. Existing work on randomized smoothing has focused on classifiers with continuous inputs, such as images, where $\ell_p$-norm bounded adversaries are commonly studied. However, there has been limited work for classifiers with discrete or variable-size inputs, such as for source code, which require different threat models and smoothing mechanisms. In this work, we adapt randomized smoothing for discrete sequence classifiers to provide certified robustness against edit distance-bounded adversaries. Our proposed smoothing mechanism randomized deletion (RS-Del) applies random deletion edits, which are (perhaps surprisingly) sufficient to confer robustness against adversarial deletion, insertion and substitution edits. Our proof of certification deviates from the established Neyman-Pearson approach, which is intractable in our setting, and is instead organized around longest common subsequences. We present a case study on malware detectionâ€”a binary classification problem on byte sequences where classifier evasion is a well-established threat model. When applied to the popular MalConv malware detection model, our smoothing mechanism RS-Del achieves a certified accuracy of 91% at an edit distance radius of 128 bytes.

----

## [820] Flow: Per-instance Personalized Federated Learning

**Authors**: *Kunjal Panchal, Sunav Choudhary, Nisarg Parikh, Lijun Zhang, Hui Guan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3baf4eeffad860ca9c54aeab632716b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3baf4eeffad860ca9c54aeab632716b4-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) suffers from data heterogeneity, where the diverse data distributions across clients make it challenging to train a single global model effectively. Existing personalization approaches aim to address the data heterogeneity issue by creating a personalized model for each client from the global model that fits their local data distribution. However, these personalized models may achieve lower accuracy than the global model in some clients, resulting in limited performance improvement compared to that without personalization. To overcome this limitation, we propose a per-instance personalization FL algorithm Flow. Flow creates dynamic personalized models that are adaptive not only to each client’s data distributions but also to each client’s data instances. The personalized model allows each instance to dynamically determine whether it prefers the local parameters or its global counterpart to make correct predictions, thereby improving clients’accuracy. We provide theoretical analysis on the convergence of Flow and empirically demonstrate the superiority of Flow in improving clients’ accuracy compared to state-of-the-art personalization approaches on both vision and language-based tasks.

----

## [821] MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing

**Authors**: *Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu, Lihua Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3baf7a39d07e9f4f1e258a412df94521-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3baf7a39d07e9f4f1e258a412df94521-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

4D human perception plays an essential role in a myriad of applications, such as home automation and metaverse avatar simulation. However, existing solutions which mainly rely on cameras and wearable devices are either privacy intrusive or inconvenient to use. To address these issues, wireless sensing has emerged as a promising alternative, leveraging LiDAR, mmWave radar, and WiFi signals for device-free human sensing. In this paper, we propose MM-Fi, the first multi-modal non-intrusive 4D human dataset with 27 daily or rehabilitation action categories, to bridge the gap between wireless sensing and high-level human perception tasks. MM-Fi consists of over 320k synchronized frames of five modalities from 40 human subjects. Various annotations are provided to support potential sensing tasks, e.g., human pose estimation and action recognition. Extensive experiments have been conducted to compare the sensing capacity of each or several modalities in terms of multiple tasks. We envision that MM-Fi can contribute to wireless sensing research with respect to action recognition, human pose estimation, multi-modal learning, cross-modal supervision, and interdisciplinary healthcare research.

----

## [822] Live Graph Lab: Towards Open, Dynamic and Real Transaction Graphs with NFT

**Authors**: *Zhen Zhang, Bingqiao Luo, Shengliang Lu, Bingsheng He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3be31c1a2fdcb7b748c53c3f4cb0e9d2-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3be31c1a2fdcb7b748c53c3f4cb0e9d2-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Numerous studies have been conducted to investigate the properties of large-scale temporal graphs. Despite the ubiquity of these graphs in real-world scenarios, it's usually impractical for us to obtain the whole real-time graphs due to privacy concerns and technical limitations. In this paper, we introduce the concept of {\it Live Graph Lab} for temporal graphs, which enables open, dynamic and real transaction graphs from blockchains. Among them, Non-fungible tokens (NFTs) have become one of the most prominent parts of blockchain over the past several years. With more than \$40 billion market capitalization, this decentralized ecosystem produces massive, anonymous and real transaction activities, which naturally forms a complicated transaction network. However, there is limited understanding about the characteristics of this emerging NFT ecosystem from a temporal graph analysis perspective. To mitigate this gap, we instantiate a live graph with NFT transaction network and investigate its dynamics to provide new observations and insights. Specifically, through downloading and parsing the NFT transaction activities, we obtain a temporal graph with more than 4.5 million nodes and 124 million edges. Then, a series of measurements are presented to understand the properties of the NFT ecosystem. Through comparisons with social, citation, and web networks, our analyses give intriguing findings and point out potential directions for future exploration. Finally, we also study machine learning models in this live graph to enrich the current datasets and provide new opportunities for the graph community. The source codes and dataset are available at https://livegraphlab.github.io.

----

## [823] CMMA: Benchmarking Multi-Affection Detection in Chinese Multi-Modal Conversations

**Authors**: *Yazhou Zhang, Yang Yu, Qing Guo, Benyou Wang, Dongming Zhao, Sagar Uprety, Dawei Song, Qiuchi Li, Jing Qin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3be60b4a739b95a07a944a1a2c41e05e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3be60b4a739b95a07a944a1a2c41e05e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Human communication has a multi-modal and multi-affection nature. The inter-relatedness of different emotions and sentiments poses a challenge to jointly detect multiple human affections with multi-modal clues. Recent advances in this field employed multi-task learning paradigms to render the inter-relatedness across tasks, but the scarcity of publicly available resources sets a limit to the potential of works. To fill this gap, we build the first Chinese Multi-modal Multi-Affection conversation (CMMA) dataset, which contains 3,000 multi-party conversations and 21,795 multi-modal utterances collected from various styles of TV-series. CMMA contains a wide variety of affection labels, including sentiment, emotion, sarcasm and humor, as well as the novel inter-correlations values between certain pairs of tasks. Moreover, it provides the topic and speaker information in conversations, which promotes better modeling of conversational context. On the dataset, we empirically analyze the influence of different data modalities and conversational contexts on different affection analysis tasks, and exhibit the practical benefit of inter-task correlations. The full dataset will be publicly available for research\footnote{https://github.com/annoymity2022/Chinese-Dataset}

----

## [824] Inverse Preference Learning: Preference-based RL without a Reward Function

**Authors**: *Joey Hejna, Dorsa Sadigh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3be7859b36d9440372cae0a293f2e4cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3be7859b36d9440372cae0a293f2e4cc-Abstract-Conference.html)

**Abstract**:

Reward functions are difficult to design and often hard to align with human intent. Preference-based Reinforcement Learning (RL) algorithms address these problems by learning reward functions from human feedback. However, the majority of preference-based RL methods na\"ively combine supervised reward models with off-the-shelf RL algorithms. Contemporary approaches have sought to improve performance and query complexity by using larger and more complex reward architectures such as transformers. Instead of using highly complex architectures, we develop a new and parameter-efficient algorithm, Inverse Preference Learning (IPL), specifically designed for learning from offline preference data. Our key insight is that for a fixed policy, the $Q$-function encodes all information about the reward function, effectively making them interchangeable. Using this insight, we completely eliminate the need for a learned reward function. Our resulting algorithm is simpler and more parameter-efficient. Across a suite of continuous control and robotics benchmarks, IPL attains competitive performance compared to more complex approaches that leverage transformer-based and non-Markovian reward functions while having fewer algorithmic hyperparameters and learned network parameters. Our code is publicly released.

----

## [825] Matrix Compression via Randomized Low Rank and Low Precision Factorization

**Authors**: *Rajarshi Saha, Varun Srivastava, Mert Pilanci*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3bf4b55960aaa23553cd2a6bdc6e1b57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3bf4b55960aaa23553cd2a6bdc6e1b57-Abstract-Conference.html)

**Abstract**:

Matrices are exceptionally useful in various fields of study as they provide a convenient framework to organize and manipulate data in a structured manner. However, modern matrices can  involve billions of elements, making their storage and processing quite demanding in terms of computational resources and memory usage. Although prohibitively large, such matrices are often approximately low rank. We propose an algorithm that exploits this structure to obtain a low rank decomposition of any matrix $\mathbf{A}$ as $\mathbf{A} \approx \mathbf{L}\mathbf{R}$, where $\mathbf{L}$ and $\mathbf{R}$ are the low rank factors. The total number of elements in $\mathbf{L}$ and $\mathbf{R}$ can be significantly less than that in $\mathbf{A}$. Furthermore, the entries of $\mathbf{L}$ and $\mathbf{R}$ are quantized to low precision formats -- compressing $\mathbf{A}$ by giving us a low rank and low precision factorization. Our algorithm first computes an approximate basis of the range space of $\mathbf{A}$ by randomly sketching its columns, followed by a quantization of the vectors constituting this basis. It then computes approximate projections of the columns of $\mathbf{A}$ onto this quantized basis. We derive upper bounds on the approximation error of our algorithm, and analyze the impact of target rank and quantization bit-budget. The tradeoff between compression ratio and approximation accuracy allows for flexibility in choosing these parameters based on specific application requirements. We empirically demonstrate the efficacy of our algorithm in image compression, nearest neighbor classification of image and text embeddings, and compressing the layers of LlaMa-$7$b. Our results illustrate that we can achieve compression ratios as aggressive as one bit per matrix coordinate, all while surpassing or maintaining the performance of traditional compression techniques.

----

## [826] OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping

**Authors**: *Huijie Wang, Tianyu Li, Yang Li, Li Chen, Chonghao Sima, Zhenbo Liu, Bangjun Wang, Peijin Jia, Yuting Wang, Shengyin Jiang, Feng Wen, Hang Xu, Ping Luo, Junchi Yan, Wei Zhang, Hongyang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c0a4c8c236144f1b99b7e1531debe9c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c0a4c8c236144f1b99b7e1531debe9c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Accurately depicting the complex traffic scene is a vital component for autonomous vehicles to execute correct judgments. However, existing benchmarks tend to oversimplify the scene by solely focusing on lane perception tasks. Observing that human drivers rely on both lanes and traffic signals to operate their vehicles safely, we present OpenLane-V2, the first dataset on topology reasoning for traffic scene structure. The objective of the presented dataset is to advance research in understanding the structure of road scenes by examining the relationship between perceived entities, such as traffic elements and lanes. Leveraging existing datasets, OpenLane-V2 consists of 2,000 annotated road scenes that describe traffic elements and their correlation to the lanes. It comprises three primary sub-tasks, including the 3D lane detection inherited from OpenLane, accompanied by corresponding metrics to evaluate the modelâ€™s performance. We evaluate various state-of-the-art methods, and present their quantitative and qualitative results on OpenLane-V2 to indicate future avenues for investigating topology reasoning in traffic scenes.

----

## [827] Prompt-augmented Temporal Point Process for Streaming Event Sequence

**Authors**: *Siqiao Xue, Yan Wang, Zhixuan Chu, Xiaoming Shi, Caigao Jiang, Hongyan Hao, Gangwei Jiang, Xiaoyun Feng, James Zhang, Jun Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c129892b4f9c8326aba665425a470c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c129892b4f9c8326aba665425a470c5-Abstract-Conference.html)

**Abstract**:

Neural Temporal Point Processes (TPPs)  are the prevalent paradigm for modeling continuous-time event sequences, such as user activities on the web and financial transactions. In real world applications, the event data typically comes in a streaming manner, where the distribution of the patterns may shift over time. Under the privacy and memory constraints commonly seen in real scenarios, how to continuously monitor a TPP to learn the streaming event sequence is an important yet under-investigated problem. In this work, we approach this problem by adopting Continual Learning (CL), which aims to enable a model to continuously learn a sequence of tasks without catastrophic forgetting. While CL for event sequence is less well studied, we present a simple yet effective framework, PromptTPP, by integrating the base TPP with a continuous-time retrieval prompt pool. In our proposed framework, prompts are small learnable parameters, maintained in a memory space and jointly optimized with the base TPP so that the model is properly instructed to learn event streams arriving sequentially without buffering past examples or task-specific attributes. We formalize a novel and realistic experimental setup for modeling event streams, where PromptTPP consistently sets state-of-the-art performance across two real user behavior datasets.

----

## [828] Leveraging Locality and Robustness to Achieve Massively Scalable Gaussian Process Regression

**Authors**: *Robert Allison, Anthony Stephenson, Samuel F, Edward O. Pyzer-Knapp*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c2b60a3f269c404e9329ee119f2d34a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c2b60a3f269c404e9329ee119f2d34a-Abstract-Conference.html)

**Abstract**:

The accurate predictions and principled uncertainty measures provided by GP regression incur $O(n^3)$ cost which is prohibitive for modern-day large-scale applications. This has motivated extensive work on computationally efficient approximations. We introduce a new perspective by exploring robustness properties and limiting behaviour of GP nearest-neighbour (GPnn) prediction. We demonstrate through theory and simulation that as the data-size $n$ increases, accuracy of estimated parameters and GP model assumptions become increasingly irrelevant to GPnn predictive accuracy. Consequently, it is sufficient to spend small amounts of work on parameter estimation in order to achieve high MSE accuracy, even in the presence of gross misspecification. In contrast, as $n \rightarrow \infty$, uncertainty calibration and NLL are shown to remain sensitive to just one parameter, the additive noise-variance; but we show that this source of inaccuracy can be corrected for, thereby achieving both well-calibrated uncertainty measures and accurate predictions at remarkably low computational cost. We exhibit a very simple GPnn regression algorithm with stand-out performance compared to other state-of-the-art GP approximations as measured on large UCI datasets. It operates at a small fraction of those other methods' training costs, for example on a basic laptop taking about 30 seconds to train on a dataset of size $n = 1.6 \times 10^6$.

----

## [829] Building the Bridge of Schrödinger: A Continuous Entropic Optimal Transport Benchmark

**Authors**: *Nikita Gushchin, Alexander Kolesov, Petr Mokrov, Polina Karpikova, Andrei Spiridonov, Evgeny Burnaev, Alexander Korotin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c4688b6a76f25f2311daa0d75a58f1a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c4688b6a76f25f2311daa0d75a58f1a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Over the last several years, there has been significant progress in developing neural solvers for the Schrödinger Bridge (SB) problem and applying them to generative modelling. This new research field is justifiably fruitful as it is interconnected with the practically well-performing diffusion models and theoretically grounded entropic optimal transport (EOT). Still, the area lacks non-trivial tests allowing a researcher to understand how well the methods solve SB or its equivalent continuous EOT problem. We fill this gap and propose a novel way to create pairs of probability distributions for which the ground truth OT solution is known by the construction. Our methodology is generic and works for a wide range of OT formulations, in particular, it covers the EOT which is equivalent to SB (the main interest of our study). This development allows us to create continuous benchmark distributions with the known EOT and SB solutions on high-dimensional spaces such as spaces of images. As an illustration, we use these benchmark pairs to test how well existing neural EOT/SB solvers actually compute the EOT solution. Our code for constructing benchmark pairs under different setups is available at: https://github.com/ngushchin/EntropicOTBenchmark

----

## [830] Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark

**Authors**: *Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, Yaodong Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Artificial intelligence (AI) systems possess significant potential to drive societal progress. However, their deployment often faces obstacles due to substantial safety concerns. Safe reinforcement learning (SafeRL) emerges as a solution to optimize policies while simultaneously adhering to multiple constraints, thereby addressing the challenge of integrating reinforcement learning in safety-critical scenarios. In this paper, we present an environment suite called Safety-Gymnasium, which encompasses safety-critical tasks in both single and multi-agent scenarios, accepting vector and vision-only input. Additionally, we offer a library of algorithms named Safe Policy Optimization (SafePO), comprising 16 state-of-the-art SafeRL algorithms. This comprehensive library can serve as a validation tool for the research community. By introducing this benchmark, we aim to facilitate the evaluation and comparison of safety performance, thus fostering the development of reinforcement learning for safer, more reliable, and responsible real-world applications. The website of this project can be accessed at https://sites.google.com/view/safety-gymnasium.

----

## [831] Direct Training of SNN using Local Zeroth Order Method

**Authors**: *Bhaskar Mukhoty, Velibor Bojkovic, William de Vazelhes, Xiaohan Zhao, Giulia De Masi, Huan Xiong, Bin Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c5e64f26a97db6a2b0bbb788236431e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c5e64f26a97db6a2b0bbb788236431e-Abstract-Conference.html)

**Abstract**:

Spiking neural networks are becoming increasingly popular for their low energy requirement in real-world tasks with accuracy comparable to traditional ANNs. SNN training algorithms face the loss of gradient information and non-differentiability due to the Heaviside function in minimizing the model loss over model parameters. To circumvent this problem, the surrogate method employs a differentiable approximation of the Heaviside function in the backward pass, while the forward pass continues to use the Heaviside as the spiking function. We propose to use the zeroth-order technique at the local or neuron level in training SNNs, motivated by its regularizing and potential energy-efficient effects and establish a theoretical connection between it and the existing surrogate methods. We perform experimental validation of the technique on standard static datasets (CIFAR-10, CIFAR-100, ImageNet-100) and neuromorphic datasets (DVS-CIFAR-10, DVS-Gesture, N-Caltech-101, NCARS) and obtain results that offer improvement over the state-of-the-art results. The proposed method also lends itself to efficient implementations of the back-propagation method, which could provide 3-4 times overall speedup in training time. The code is available at \url{https://github.com/BhaskarMukhoty/LocalZO}.

----

## [832] Discover and Align Taxonomic Context Priors for Open-world Semi-Supervised Learning

**Authors**: *Yu Wang, Zhun Zhong, Pengchong Qiao, Xuxin Cheng, Xiawu Zheng, Chang Liu, Nicu Sebe, Rongrong Ji, Jie Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c646b713f5de2cf1ab1939d49a4036d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c646b713f5de2cf1ab1939d49a4036d-Abstract-Conference.html)

**Abstract**:

Open-world Semi-Supervised Learning (OSSL) is a realistic and challenging task, aiming to classify unlabeled samples from both seen and novel classes using partially labeled samples from the seen classes. Previous works typically explore the relationship of samples as priors on the pre-defined single-granularity labels to help novel class recognition. In fact, classes follow a taxonomy and samples can be classified at multiple levels of granularity, which contains more underlying relationships for supervision. We thus argue that learning with single-granularity labels results in sub-optimal representation learning and inaccurate pseudo labels, especially with unknown classes. In this paper, we take the initiative to explore and propose a uniformed framework, called Taxonomic context prIors Discovering and Aligning (TIDA), which exploits the relationship of samples under various granularity. It allows us to discover multi-granularity semantic concepts as taxonomic context priors (i.e., sub-class, target-class, and super-class), and then collaboratively leverage them to enhance representation learning and improve the quality of pseudo labels.Specifically, TIDA comprises two components: i) A taxonomic context discovery module that constructs a set of hierarchical prototypes in the latent space to discover the underlying taxonomic context priors; ii) A taxonomic context-based prediction alignment module that enforces consistency across hierarchical predictions to build the reliable relationship between classes among various granularity and provide additions supervision. We demonstrate that these two components are mutually beneficial for an effective OSSL framework, which is theoretically explained from the perspective of the EM algorithm. Extensive experiments on seven commonly used datasets show that TIDA can significantly improve the performance and achieve a new state of the art. The source codes are publicly available at https://github.com/rain305f/TIDA.

----

## [833] Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models

**Authors**: *Julien Siems, Konstantin Ditschuneit, Winfried Ripken, Alma Lindborg, Maximilian Schambach, Johannes S. Otterbach, Martin Genzel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c6696d70d364337cf98dcb7c652a770-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c6696d70d364337cf98dcb7c652a770-Abstract-Conference.html)

**Abstract**:

Generalized Additive Models (GAMs) have recently experienced a resurgence in popularity due to their interpretability, which arises from expressing the target value as a sum of non-linear transformations of the features. Despite the current enthusiasm for GAMs, their susceptibility to concurvity — i.e., (possibly non-linear) dependencies between the features — has hitherto been largely overlooked. Here, we demonstrate how concurvity can severly impair the interpretability of GAMs and propose a remedy: a conceptually simple, yet effective regularizer which penalizes pairwise correlations of the non-linearly transformed feature variables. This procedure is applicable to any differentiable additive model, such as Neural Additive Models or NeuralProphet, and enhances interpretability by eliminating ambiguities due to self-canceling feature contributions. We validate the effectiveness of our regularizer in experiments on synthetic as well as real-world datasets for time-series and tabular data. Our experiments show that concurvity in GAMs can be reduced without significantly compromising prediction quality, improving interpretability and reducing variance in the feature importances.

----

## [834] Mutual Information Regularized Offline Reinforcement Learning

**Authors**: *Xiao Ma, Bingyi Kang, Zhongwen Xu, Min Lin, Shuicheng Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c6bd2021c10462c5164638d22f3d5d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c6bd2021c10462c5164638d22f3d5d8-Abstract-Conference.html)

**Abstract**:

The major challenge of offline RL is the distribution shift that appears when out-of-distribution actions are queried, which makes the policy improvement direction biased by extrapolation errors. Most existing methods address this problem by penalizing the policy or value for deviating from the behavior policy during policy improvement or evaluation. In this work, we propose a novel MISA framework to approach offline RL from the perspective of Mutual Information between States and Actions in the dataset by directly constraining the policy improvement direction. MISA constructs lower bounds of mutual information parameterized by the policy and Q-values. We show that optimizing this lower bound is equivalent to maximizing the likelihood of a one-step improved policy on the offline dataset. Hence, we constrain the policy improvement direction to lie in the data manifold. The resulting algorithm simultaneously augments the policy evaluation and improvement by adding mutual information regularizations. MISA is a general framework that unifies conservative Q-learning (CQL) and behavior regularization methods (e.g., TD3+BC) as special cases. We introduce 3 different variants of MISA, and empirically demonstrate that tighter mutual information lower bound gives better offline RL performance. In addition, our extensive experiments show MISA significantly outperforms a wide range of baselines on various tasks of the D4RL benchmark, e.g., achieving 742.9 total points on gym-locomotion tasks. Our code is attached and will be released upon publication.

----

## [835] Have it your way: Individualized Privacy Assignment for DP-SGD

**Authors**: *Franziska Boenisch, Christopher Mühl, Adam Dziedzic, Roy Rinberg, Nicolas Papernot*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html)

**Abstract**:

When training a machine learning model with differential privacy, one sets a privacy budget. This uniform budget represents an overall maximal privacy violation that any user is willing to face by contributing their data to the training set. We argue that this approach is limited because different users may have different privacy expectations. Thus, setting a uniform privacy budget across all points may be overly conservative for some users or, conversely, not sufficiently protective for others. In this paper, we capture these preferences through individualized privacy budgets. To demonstrate their practicality, we introduce a variant of Differentially Private Stochastic Gradient Descent (DP-SGD) which supports such individualized budgets. DP-SGD is the canonical approach to training models with differential privacy. We modify its data sampling and gradient noising mechanisms to arrive at our approach, which we call Individualized DP-SGD (IDP-SGD). Because IDP-SGD provides privacy guarantees tailored to the preferences of individual users and their data points, we empirically find it to improve privacy-utility trade-offs.

----

## [836] Penguin: Parallel-Packed Homomorphic Encryption for Fast Graph Convolutional Network Inference

**Authors**: *Ran Ran, Nuo Xu, Tao Liu, Wei Wang, Gang Quan, Wujie Wen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3cc685788a311fa35d8d41df93e288ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3cc685788a311fa35d8d41df93e288ca-Abstract-Conference.html)

**Abstract**:

The marriage of Graph Convolutional Network (GCN) and Homomorphic Encryption (HE) enables the inference of graph data on the cloud with significantly enhanced client data privacy. However, the tremendous computation and memory overhead associated with HE operations challenges the practicality of HE-based GCN inference. GCN inference involves a sequence of expensive matrix-matrix multiplications, and we observe that directly applying the state-of-the-art HE-based secure matrix-matrix multiplication solutions to accelerate HE-GCN inference is far less efficient as it does not exploit the unique aggregation mechanism of two-dimension graph node-features in GCN layer computation. As a result, in this paper, we propose a novel HE-based ciphertext packing technique, i.e., Penguin, that can take advantage of the unique computation pattern during the HE-GCN inference to significantly reduce the computation and memory overhead associated with HE operations.Specifically, Penguin employs (i) an effective two-dimension parallel packing technique for feature ciphertext with optimal graph node partitioning and graph feature interleaving, and (ii) an interleaved assembly technique that can effectively make use of the blank slots to merge ciphertexts after feature reduction and significantly reduce the costly rotation operation.We provide theoretical analysis and experimental validation to demonstrate the speedup achieved by Penguin in accelerating GCN inference using popular GCN models and datasets. Our results show that Penguin can achieve up to $\sim10\times$ speedup and around $\sim79$% reduction in computational memory overhead, significantly outperforming state-of-the-art solutions. To the best of our knowledge, this is the first work that can ensure the protection of both graph structure and features when accelerating HE-GCN inference on encrypted data. Our code is publicly available at https://github.com/ranran0523/Penguin.

----

## [837] Learning Dynamic Attribute-factored World Models for Efficient Multi-object Reinforcement Learning

**Authors**: *Fan Feng, Sara Magliacane*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3cc87f2bd3e3b4df8f9217326761c322-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3cc87f2bd3e3b4df8f9217326761c322-Abstract-Conference.html)

**Abstract**:

In many reinforcement learning tasks, the agent has to learn to interact with many objects of different types and generalize to unseen combinations and numbers of objects. Often a task is a composition of previously learned tasks (e.g. block stacking).These are examples of compositional generalization, in which we compose object-centric representations to solve complex tasks. Recent works have shown the benefits of object-factored representations and hierarchical abstractions for improving sample efficiency in these settings. On the other hand, these methods do not fully exploit the benefits of factorization in terms of object attributes. In this paper, we address this opportunity and introduce the Dynamic Attribute FacTored RL (DAFT-RL) framework. In DAFT-RL, we leverage object-centric representation learning to extract objects from visual inputs. We learn to classify them into classes and infer their latent parameters. For each class of object, we learn a class template graph that describes how the dynamics and reward of an object of this class factorize according to its attributes. We also learn an interaction pattern graph that describes how objects of different classes interact with each other at the attribute level. Through these graphs and a dynamic interaction graph that models the interactions between objects, we can learn a policy that can then be directly applied in a new environment by estimating the interactions and latent parameters.We evaluate DAFT-RL in three benchmark datasets and show our framework outperforms the state-of-the-art in generalizing across unseen objects with varying attributes and latent parameters, as well as in the composition of previously learned tasks.

----

## [838] Statistical Insights into HSIC in High Dimensions

**Authors**: *Tao Zhang, Yaowu Zhang, Tingyou Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3cfc102893d47c46295cb437949dccb5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3cfc102893d47c46295cb437949dccb5-Abstract-Conference.html)

**Abstract**:

Measuring the nonlinear dependence between random vectors and testing for their statistical independence is a fundamental problem in statistics. One of the most popular dependence measures is the Hilbert-Schmidt independence criterion (HSIC), which has attracted increasing attention in recent years. However, most existing works have focused on either fixed or very high-dimensional covariates. In this work, we bridge the gap between these two scenarios and provide statistical insights into the performance of HSIC when the dimensions grow at different rates. We first show that, under the null hypothesis, the rescaled HSIC converges in distribution to a standard normal distribution. Then we provide a general condition for the HSIC based tests to have nontrivial power in high dimensions. By decomposing this condition, we illustrate how the ability of HSIC to measure nonlinear dependence changes with increasing dimensions. Moreover, we demonstrate that, depending on the sample size, the covariate dimensions and the dependence structures within covariates, the HSIC can capture different types of associations between random vectors. We also conduct extensive numerical studies to validate our theoretical results.

----

## [839] Fair Adaptive Experiments

**Authors**: *Waverly Wei, Xinwei Ma, Jingshen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d007df4ae13adf9001f8969555b11bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d007df4ae13adf9001f8969555b11bd-Abstract-Conference.html)

**Abstract**:

Randomized experiments have been the gold standard for assessing the effectiveness of a treatment, policy, or intervention, spanning various fields, including social sciences, biomedical studies, and e-commerce. The classical complete randomization approach assigns treatments based on a pre-specified probability and may lead to inefficient use of data. Adaptive experiments improve upon complete randomization by sequentially learning and updating treatment assignment probabilities using accrued evidence during the experiment. Hence, they can help achieve efficient data use and higher estimation efficiency. However, their application can also raise fairness and equity concerns, as assignment probabilities may vary drastically across groups of participants. Furthermore, when treatment is expected to be extremely beneficial to certain groups of participants, it is more appropriate to expose many of these participants to favorable treatment. In response to these challenges, we propose a fair adaptive experiment strategy that simultaneously enhances data use efficiency, achieves an ``envy-free'' treatment assignment guarantee, and improves the overall welfare of participants. An important feature of our proposed strategy is that we do not impose parametric modeling assumptions on the outcome variables, making it more versatile and applicable to a wider array of applications. Through our theoretical investigation, we characterize the convergence rate of the estimated treatment effects and the associated standard deviations at the group level and further prove that our adaptive treatment assignment algorithm, despite not having a closed-form expression, approaches the optimal allocation rule asymptotically. Our proof strategy takes into account the fact that the allocation decisions in our design depend on sequentially accumulated data, which poses a significant challenge in characterizing the properties and conducting statistical inference of our method. We further provide simulation evidence and two synthetic data studies to showcase the performance of our fair adaptive experiment strategy.

----

## [840] Alexa Arena: A User-Centric Interactive Platform for Embodied AI

**Authors**: *Qiaozi Gao, Govind Thattai, Suhaila Shakiah, Xiaofeng Gao, Shreyas Pansare, Vasu Sharma, Gaurav S. Sukhatme, Hangjie Shi, Bofei Yang, Desheng Zhang, Lucy Hu, Karthika Arumugam, Shui Hu, Matthew Wen, Dinakar Guthy, Shunan Chung, Rohan Khanna, Osman Ipek, Leslie Ball, Kate Bland, Heather Rocker, Michael Johnston, Reza Ghanadan, Dilek Hakkani-Tur, Prem Natarajan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d0758f0b95e19abc68c1c8070d36510-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d0758f0b95e19abc68c1c8070d36510-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Alexa Arena, a user-centric simulation platform to facilitate research in building assistive conversational embodied agents. Alexa Arena features multi-room layouts and an abundance of interactable objects. With user-friendly graphics and control mechanisms, the platform supports the development of gamified robotic tasks readily accessible to general human users, allowing high-efficiency data collection and EAI system evaluation. Along with the platform, we introduce a dialog-enabled task completion benchmark with online human evaluations.

----

## [841] Synthetic Combinations: A Causal Inference Framework for Combinatorial Interventions

**Authors**: *Abhineet Agarwal, Anish Agarwal, Suhas Vijaykumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d17b7f7d52c83ab6e97e2dc0bda2e71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d17b7f7d52c83ab6e97e2dc0bda2e71-Abstract-Conference.html)

**Abstract**:

We consider a setting where there are $N$ heterogeneous units and $p$ interventions. Our goal is to learn unit-specific potential outcomes for any combination of these $p$ interventions, i.e., $N \times 2^p$ causal parameters. Choosing a combination of interventions is a problem that naturally arises in a variety of applications such as factorial design experiments and recommendation engines (e.g., showing a set of movies that maximizes engagement for a given user). Running $N \times 2^p$ experiments to estimate the various parameters is likely expensive and/or infeasible as $N$ and $p$ grow. Further, with observational data there is likely confounding, i.e., whether or not a unit is seen under a combination is correlated with its potential outcome under that combination. We study this problem under a novel model that imposes latent structure across both units and combinations of interventions. Specifically, we assume latent similarity in potential outcomes across units (i.e., the matrix of potential outcomes is approximately rank $r$) and regularity in how combinations of interventions interact (i.e., the coefficients in the Fourier expansion of the potential outcomes is approximately $s$ sparse). We establish identification for all $N \times 2^p$ parameters despite unobserved confounding. We propose an estimation procedure, Synthetic Combinations, and establish finite-sample consistency under precise conditions on the observation pattern. We show that Synthetic Combinations is able to consistently estimate unit-specific potential outcomes given a total of $\text{poly}(r) \times \left( N + s^2p\right)$ observations. In comparison, previous methods that do not exploit structure across both units and combinations have poorer sample complexity scaling as $\min(N \times s^2p, \ \ r \times (N + 2^p))$.

----

## [842] PUCA: Patch-Unshuffle and Channel Attention for Enhanced Self-Supervised Image Denoising

**Authors**: *Hyemi Jang, Junsung Park, Dahuin Jung, Jaihyun Lew, Ho Bae, Sungroh Yoon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d226fb8fbd6ee6ec70d0427f1319707-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d226fb8fbd6ee6ec70d0427f1319707-Abstract-Conference.html)

**Abstract**:

Although supervised image denoising networks have shown remarkable performance on synthesized noisy images, they often fail in practice due to the difference between real and synthesized noise. Since clean-noisy image pairs from the real world are extremely costly to gather, self-supervised learning, which utilizes noisy input itself as a target, has been studied. To prevent a self-supervised denoising model from learning identical mapping, each output pixel should not be influenced by its corresponding input pixel; This requirement is known as J-invariance. Blind-spot networks (BSNs) have been a prevalent choice to ensure J-invariance in self-supervised image denoising. However, constructing variations of BSNs by injecting additional operations such as downsampling can expose blinded information, thereby violating J-invariance. Consequently, convolutions designed specifically for BSNs have been allowed only, limiting architectural flexibility. To overcome this limitation, we propose PUCA, a novel J-invariant U-Net architecture, for self-supervised denoising. PUCA leverages patch-unshuffle/shuffle to dramatically expand receptive fields while maintaining J-invariance and dilated attention blocks (DABs) for global context incorporation. Experimental results demonstrate that PUCA achieves state-of-the-art performance, outperforming existing methods in self-supervised image denoising.

----

## [843] Projection Regret: Reducing Background Bias for Novelty Detection via Diffusion Models

**Authors**: *Sungik Choi, Hankook Lee, Honglak Lee, Moontae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d27d607586984908900eaa8ce19c96c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d27d607586984908900eaa8ce19c96c-Abstract-Conference.html)

**Abstract**:

Novelty detection is a fundamental task of machine learning which aims to detect abnormal (i.e. out-of-distribution (OOD)) samples. Since diffusion models have recently emerged as the de facto standard generative framework with surprising generation results, novelty detection via diffusion models has also gained much attention. Recent methods have mainly utilized the reconstruction property of in-distribution samples. However, they often suffer from detecting OOD samples that share similar background information to the in-distribution data. Based on our observation that diffusion models can project any sample to an in-distribution sample with similar background information, we propose Projection Regret (PR), an efficient novelty detection method that mitigates the bias of non-semantic information. To be specific, PR computes the perceptual distance between the test image and its diffusion-based projection to detect abnormality. Since the perceptual distance often fails to capture semantic changes when the background information is dominant, we cancel out the background bias by comparing it against recursive projections. Extensive experiments demonstrate that PR outperforms the prior art of generative-model-based novelty detection methods by a significant margin.

----

## [844] Versatile Energy-Based Probabilistic Models for High Energy Physics

**Authors**: *Taoli Cheng, Aaron C. Courville*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d4c0a618d0acd7921493e4f30395c22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d4c0a618d0acd7921493e4f30395c22-Abstract-Conference.html)

**Abstract**:

As a classical generative modeling approach, energy-based models have the natural advantage of flexibility in the form of the energy function. Recently, energy-based models have achieved great success in modeling high-dimensional data in computer vision and natural language processing. In line with these advancements, we build a multi-purpose energy-based probabilistic model for High Energy Physics events at the Large Hadron Collider.  This framework builds on a powerful generative model and describes higher-order inter-particle interactions. It suits different encoding architectures and builds on implicit generation. As for applicative aspects, it can serve as a powerful parameterized event generator for physics simulation, a generic anomalous signal detector free from spurious correlations, and an augmented event classifier for particle identification.

----

## [845] User-Level Differential Privacy With Few Examples Per User

**Authors**: *Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Raghu Meka, Chiyuan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d57795f0e263aa69577f1bbceade46b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d57795f0e263aa69577f1bbceade46b-Abstract-Conference.html)

**Abstract**:

Previous work on user-level differential privacy (DP) [Ghazi et al. NeurIPS 2021, Bun et al. STOC 2023] obtained generic algorithms that work for various learning tasks. However, their focus was on the *example-rich* regime, where the users have so many examples that each user could themselves solve the problem. In this work we consider the *example-scarce* regime, where each user has only a few examples, and obtain the following results:* For approximate-DP, we give a generic transformation of any item-level DP algorithm to a user-level DP algorithm. Roughly speaking, the latter gives a (multiplicative) savings of $O_{\varepsilon,\delta}(\sqrt{m})$ in terms of the number of users required for achieving the same utility, where $m$ is the number of examples per user. This algorithm, while recovering most known bounds for specific problems, also gives new bounds, e.g., for PAC learning. * For pure-DP, we present a simple technique for adapting the exponential mechanism [McSherry & Talwar, FOCS 2007] to the user-level setting. This gives new bounds for a variety of tasks, such as private PAC learning, hypothesis selection, and distribution learning. For some of these problems, we show that our bounds are near-optimal.

----

## [846] Neural Lighting Simulation for Urban Scenes

**Authors**: *Ava Pun, Gary Sun, Jingkang Wang, Yun Chen, Ze Yang, Sivabalan Manivasagam, Wei-Chiu Ma, Raquel Urtasun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d7259031023c5aa463187c4a31c95c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d7259031023c5aa463187c4a31c95c8-Abstract-Conference.html)

**Abstract**:

Different outdoor illumination conditions drastically alter the appearance of urban scenes, and they can harm the performance of image-based robot perception systems if not seen during training. Camera simulation provides a cost-effective solution to create a large dataset of images captured under different lighting conditions. Towards this goal, we propose LightSim, a neural lighting camera simulation system that enables diverse, realistic, and controllable data generation. LightSim automatically builds lighting-aware digital twins at scale from collected raw sensor data and decomposes the scene into dynamic actors and static background with accurate geometry, appearance, and estimated scene lighting. These digital twins enable actor insertion, modification, removal, and rendering from a new viewpoint, all in a lighting-aware manner. LightSim then combines physically-based and learnable deferred rendering to perform realistic relighting of modified scenes, such as altering the sun location and modifying the shadows or changing the sun brightness, producing spatially- and temporally-consistent camera videos. Our experiments show that LightSim generates more realistic relighting results  than prior work. Importantly,  training perception models on data generated by LightSim can significantly improve their performance. Our project page is available at https://waabi.ai/lightsim/.

----

## [847] Learning to Compress Prompts with Gist Tokens

**Authors**: *Jesse Mu, Xiang Li, Noah D. Goodman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html)

**Abstract**:

Prompting is the primary way to utilize the multitask capabilities of language models (LMs), but prompts occupy valuable space in the input context window, and repeatedly encoding the same prompt is computationally inefficient. Finetuning and distillation methods allow for specialization of LMs without prompting, but require retraining the model for each task. To avoid this trade-off entirely, we present gisting, which trains an LM to compress prompts into smaller sets of "gist" tokens which can be cached and reused for compute efficiency. Gist models can be trained with no additional cost over standard instruction finetuning by simply modifying Transformer attention masks to encourage prompt compression. On decoder (LLaMA-7B) and encoder-decoder (FLAN-T5-XXL) LMs, gisting enables up to 26x compression of prompts, resulting in up to 40% FLOPs reductions, 4.2% wall time speedups, and storage savings, all with minimal loss in output quality.

----

## [848] A Heavy-Tailed Algebra for Probabilistic Programming

**Authors**: *Feynman T. Liang, Liam Hodgkinson, Michael W. Mahoney*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d8f7945cd7f4446cb05a390d4c00558-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d8f7945cd7f4446cb05a390d4c00558-Abstract-Conference.html)

**Abstract**:

Despite the successes of probabilistic models based on passing noise through neural networks, recent work has identified that such methods often fail to capture tail behavior accurately---unless the tails of the base distribution are appropriately calibrated.  To overcome this deficiency, we propose a systematic approach for analyzing the tails of random variables, and we illustrate how this approach can be used during the static analysis (before drawing samples) pass of a probabilistic programming language (PPL) compiler.  To characterize how the tails change under various operations, we develop an algebra which acts on a three-parameter family of tail asymptotics and which is based on the generalized Gamma distribution.  Our algebraic operations are closed under addition and multiplication; they are capable of distinguishing sub-Gaussians with differing scales; and they handle ratios sufficiently well to reproduce the tails of most important statistical distributions directly from their definitions.  Our empirical results confirm that inference algorithms that leverage our heavy-tailed algebra attain superior performance across a number of density modeling and variational inference (VI) tasks.

----

## [849] AIMS: All-Inclusive Multi-Level Segmentation for Anything

**Authors**: *Lu Qi, Jason Kuen, Weidong Guo, Jiuxiang Gu, Zhe Lin, Bo Du, Yu Xu, Ming-Hsuan Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3da292ced54290c19fc55d9dba3da793-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3da292ced54290c19fc55d9dba3da793-Abstract-Conference.html)

**Abstract**:

Despite the progress of image segmentation for accurate visual entity segmentation, completing the diverse requirements of image editing applications for different-level region-of-interest selections remains unsolved. In this paper, we propose a new task, All-Inclusive Multi-Level Segmentation (AIMS), which segments visual regions into three levels: part, entity, and relation (two entities with some semantic relationships). We also build a unified AIMS model through multi-dataset multi-task training to address the two major challenges of annotation inconsistency and task correlation. Specifically, we propose task complementarity, association, and prompt mask encoder for three-level predictions. Extensive experiments demonstrate the effectiveness and generalization capacity of our method compared to other state-of-the-art methods on a single dataset or the concurrent work on segment anything. We will make our code and training model publicly available.

----

## [850] Performance Bounds for Policy-Based Average Reward Reinforcement Learning Algorithms

**Authors**: *Yashaswini Murthy, Mehrdad Moharrami, R. Srikant*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3da8e709fa1a7d9e23bee89d3c25b5b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3da8e709fa1a7d9e23bee89d3c25b5b4-Abstract-Conference.html)

**Abstract**:

Many policy-based reinforcement learning (RL) algorithms can be viewed as instantiations of approximate policy iteration (PI), i.e., where policy improvement and policy evaluation are both performed approximately. In applications where the average reward objective is the meaningful performance metric, often discounted reward formulations are used with the discount factor being close to $1,$ which is equivalent to making the expected horizon very large. However, the corresponding theoretical bounds for error performance scale with the square of the horizon. Thus, even after dividing the total reward by the length of the horizon, the corresponding performance bounds for average reward problems go to infinity. Therefore, an open problem has been to obtain meaningful performance bounds for approximate PI and RL algorithms for the average-reward setting.  In this paper, we solve this open problem by obtaining the first non-trivial finite time error bounds for average-reward MDPs which go to zero in the limit as policy evaluation and policy improvement errors go to zero.

----

## [851] Understanding Few-Shot Learning: Measuring Task Relatedness and Adaptation Difficulty via Attributes

**Authors**: *Minyang Hu, Hong Chang, Zong Guo, Bingpeng Ma, Shiguang Shan, Xilin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3df38ca67befaed9c03b95ffee07d9f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3df38ca67befaed9c03b95ffee07d9f8-Abstract-Conference.html)

**Abstract**:

Few-shot learning (FSL) aims to learn novel tasks with very few labeled samples by leveraging experience from \emph{related} training tasks.    In this paper, we try to understand FSL by exploring two key questions:    (1) How to quantify the relationship between \emph{ training} and \emph{novel} tasks?    (2) How does the relationship affect the \emph{adaptation difficulty} on novel tasks for different models?    To answer the first question, we propose Task Attribute Distance (TAD) as a metric to quantify the task relatedness via attributes.    Unlike other metrics, TAD is independent of models, making it applicable to different FSL models.    To address the second question, we utilize TAD metric to establish a theoretical connection between task relatedness and task adaptation difficulty.    By deriving the generalization error bound on a novel task, we discover how TAD measures the adaptation difficulty on novel tasks for different models.    To validate our theoretical results, we conduct experiments on three benchmarks.    Our experimental results confirm that TAD metric effectively quantifies the task relatedness and reflects the adaptation difficulty on novel tasks for various FSL methods, even if some of them do not learn attributes explicitly or human-annotated attributes are not provided.    Our code is available at     \href{https://github.com/hu-my/TaskAttributeDistance}{https://github.com/hu-my/TaskAttributeDistance}.

----

## [852] Locally Invariant Explanations: Towards Stable and Unidirectional Explanations through Local Invariant Learning

**Authors**: *Amit Dhurandhar, Karthikeyan Natesan Ramamurthy, Kartik Ahuja, Vijay Arya*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html)

**Abstract**:

Locally interpretable model agnostic explanations (LIME) method is one of the most popular methods used to explain black-box models at a per example level. Although many variants have been proposed, few provide a simple way to produce high fidelity explanations that are also stable and intuitive. In this work, we provide a novel perspective by proposing a model agnostic local explanation method inspired by the invariant risk minimization (IRM) principle -- originally proposed for (global) out-of-distribution generalization -- to provide such high fidelity explanations that are also stable and unidirectional across nearby examples. Our method is based on a game theoretic formulation where we theoretically show that our approach has a strong tendency to eliminate features where the gradient of the black-box function abruptly changes sign in the locality of the example we want to explain, while in other cases it is more careful and will choose a more conservative (feature) attribution, a behavior which can be highly desirable for recourse. Empirically, we show on tabular, image and text data that the quality of our explanations with neighborhoods formed using random perturbations are much better than LIME and in some cases even comparable to other methods that use realistic neighbors sampled from the data manifold. This is desirable given that learning a manifold to either create realistic neighbors or to project explanations is typically expensive or may even be impossible. Moreover, our algorithm is simple and efficient to train, and can ascertain stable input features for local decisions of a black-box without access to side information such as a (partial) causal graph as has been seen in some recent works.

----

## [853] Quantification of Uncertainty with Adversarial Models

**Authors**: *Kajetan Schweighofer, Lukas Aichberger, Mykyta Ielanskyi, Günter Klambauer, Sepp Hochreiter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e0b96206965f5f05b0b4550c0e73ff0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e0b96206965f5f05b0b4550c0e73ff0-Abstract-Conference.html)

**Abstract**:

Quantifying uncertainty is important for actionable predictions in real-world applications. A crucial part of predictive uncertainty quantification is the estimation of epistemic uncertainty, which is defined as an integral of the product between a divergence function and the posterior. Current methods such as Deep Ensembles or MC dropout underperform at estimating the epistemic uncertainty, since they primarily consider the posterior when sampling models. We suggest Quantification of Uncertainty with Adversarial Models (QUAM) to better estimate the epistemic uncertainty. QUAM identifies regions where the whole product under the integral is large, not just the posterior. Consequently, QUAM has lower approximation error of the epistemic uncertainty compared to previous methods. Models for which the product is large correspond to adversarial models (not adversarial examples!). Adversarial models have both a high posterior as well as a high divergence between their predictions and that of a reference model. Our experiments show that QUAM excels in capturing epistemic uncertainty for deep learning models and outperforms previous methods on challenging tasks in the vision domain.

----

## [854] NeuroGF: A Neural Representation for Fast Geodesic Distance and Path Queries

**Authors**: *Qijian Zhang, Junhui Hou, Yohanes Yudhi Adikusuma, Wenping Wang, Ying He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e22abb329d44080460b0eb11bf21da1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e22abb329d44080460b0eb11bf21da1-Abstract-Conference.html)

**Abstract**:

Geodesics play a critical role in many geometry processing applications. Traditional algorithms for computing geodesics on 3D mesh models are often inefficient and slow, which make them impractical for scenarios requiring extensive querying of arbitrary point-to-point geodesics. Recently, deep implicit functions have gained popularity for 3D geometry representation, yet there is still no research on neural implicit representation of geodesics. To bridge this gap, we make the first attempt to represent geodesics using implicit learning frameworks. Specifically, we propose neural geodesic field (NeuroGF), which can be learned to encode all-pairs geodesics of a given 3D mesh model, enabling to efficiently and accurately answer queries of arbitrary point-to-point geodesic distances and paths. Evaluations on common 3D object models and real-captured scene-level meshes demonstrate our exceptional performances in terms of representation accuracy and querying efficiency. Besides, NeuroGF also provides a convenient way of jointly encoding both 3D geometry and geodesics in a unified representation. Moreover, the working mode of per-model overfitting is further extended to generalizable learning frameworks that can work on various input formats such as unstructured point clouds, which also show satisfactory performances for unseen shapes and categories. Our code and data are available at https://github.com/keeganhk/NeuroGF.

----

## [855] A Trichotomy for Transductive Online Learning

**Authors**: *Steve Hanneke, Shay Moran, Jonathan Shafer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e32af2df2cd13dfbcbe6e8d38111068-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e32af2df2cd13dfbcbe6e8d38111068-Abstract-Conference.html)

**Abstract**:

We present new upper and lower bounds on the number of learner mistakes in the `transductive' online learning setting of Ben-David, Kushilevitz and Mansour (1997).    This setting is similar to standard online learning, except that the adversary fixes a sequence of instances $x_1,\dots,x_n$ to be labeled at the start of the game, and this sequence is known to the learner.    Qualitatively, we prove a \emph{trichotomy}, stating that the minimal number of mistakes made by the learner as $n$ grows can take only one of precisely three possible values: $n$, $\Theta\left(\log (n)\right)$, or $\Theta(1)$.    Furthermore, this behavior is determined by a combination of the VC dimension and the Littlestone dimension.    Quantitatively, we show a variety of bounds relating the number of mistakes to well-known combinatorial dimensions.    In particular, we improve the known lower bound on the constant in the $\Theta(1)$ case from $\Omega\left(\sqrt{\log(d)}\right)$ to $\Omega(\log(d))$ where $d$ is the Littlestone dimension.    Finally, we extend our results to cover multiclass classification and the agnostic setting.

----

## [856] Evolutionary Neural Architecture Search for Transformer in Knowledge Tracing

**Authors**: *Shangshang Yang, Xiaoshan Yu, Ye Tian, Xueming Yan, Haiping Ma, Xingyi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e53d82a1113e3d240059a9195668edc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e53d82a1113e3d240059a9195668edc-Abstract-Conference.html)

**Abstract**:

Knowledge tracing (KT) aims to trace students' knowledge states by predicting whether students answer correctly on exercises. Despite the excellent performance of existing Transformer-based KT approaches, they are criticized for the manually selected input features for fusion and the defect of single global context modelling to directly capture students' forgetting behavior in KT, when the related records are distant from the current record in terms of time. To address the issues, this paper first considers adding convolution operations to the Transformer to enhance its local context modelling ability used for students' forgetting behavior, then proposes an evolutionary neural architecture search approach to automate the input feature selection and automatically determine where to apply which operation for achieving the balancing of the local/global context modelling. In the search space, the original global path containing the attention module in Transformer is replaced with the sum of a global path and a local path that could contain different convolutions, and the selection of input features is also considered. To search the best architecture, we employ an effective evolutionary algorithm to explore the search space and also suggest a search space reduction strategy to accelerate the convergence of the algorithm. Experimental results on the two largest and most challenging education datasets demonstrate the effectiveness of the architecture found by the proposed approach.

----

## [857] Learning threshold neurons via edge of stability

**Authors**: *Kwangjun Ahn, Sébastien Bubeck, Sinho Chewi, Yin Tat Lee, Felipe Suarez, Yi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e592c571de69a43d7a870ea89c7e33a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e592c571de69a43d7a870ea89c7e33a-Abstract-Conference.html)

**Abstract**:

Existing analyses of neural network training often operate under the unrealistic assumption of an extremely small learning rate. This lies in stark contrast to practical wisdom and empirical studies, such as the work of J. Cohen et al. (ICLR 2021), which exhibit startling new phenomena (the "edge of stability"' or "unstable convergence") and potential benefits for generalization in the large learning rate regime. Despite a flurry of recent works on this topic, however, the latter effect is still poorly understood. In this paper, we take a step towards understanding genuinely non-convex training dynamics with large learning rates by performing a detailed analysis of gradient descent for simplified models of two-layer neural networks. For these models, we provably establish the edge of stability phenomenon and discover a sharp phase transition for the step size below which the neural network fails to learn ``threshold-like'' neurons (i.e., neurons with a non-zero first-layer bias). This elucidates one possible mechanism by which the edge of stability can in fact lead to better generalization, as threshold neurons are basic building blocks with useful inductive bias for many tasks.

----

## [858] k-Means Clustering with Distance-Based Privacy

**Authors**: *Alessandro Epasto, Vahab Mirrokni, Shyam Narayanan, Peilin Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e8d9bf1dd1eb9d3d9d500fb3543c87b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e8d9bf1dd1eb9d3d9d500fb3543c87b-Abstract-Conference.html)

**Abstract**:

In this paper, we initiate the study of Euclidean clustering with Distance-based privacy. Distance-based privacy is motivated by the fact that it is often only needed to protect the privacy of exact, rather than approximate, locations. We provide constant-approximate algorithms for $k$-means and $k$-median clustering, with additive error depending only on the attacker's precision bound $\rho$, rather than the radius $\Lambda$ of the space. In addition, we empirically demonstrate that our algorithm performs significantly better than previous differentially private clustering algorithms, as well as naive distance-based private clustering baselines.

----

## [859] StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

**Authors**: *Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html)

**Abstract**:

In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker adaptation. This work achieves the first human-level TTS on both single and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with large SLMs. The audio demos and source code are available at https://styletts2.github.io/.

----

## [860] Large Language Models Are Zero-Shot Time Series Forecasters

**Authors**: *Nate Gruver, Marc Finzi, Shikai Qiu, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3eb7ca52e8207697361b2c0fb3926511-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3eb7ca52e8207697361b2c0fb3926511-Abstract-Conference.html)

**Abstract**:

By encoding time series as a string of numerical digits, we can frame time series forecasting as next-token prediction in text. Developing this approach, we find that large language models (LLMs) such as GPT-3 and LLaMA-2 can surprisingly zero-shot extrapolate time series at a level comparable to or exceeding the performance of purpose-built time series models trained on the downstream tasks. To facilitate this performance, we propose procedures for effectively tokenizing time series data and converting discrete distributions over tokens into highly flexible densities over continuous values. We argue the success of LLMs for time series stems from their ability to naturally represent multimodal distributions, in conjunction with biases for simplicity, and repetition, which align with the salient features in many time series, such as repeated seasonal trends. We also show how LLMs can naturally handle missing data without imputation through non-numerical text, accommodate textual side information, and answer questions to help explain predictions.  While we find that increasing model size generally improves performance on time series, we show GPT-4 can perform worse than GPT-3 because of how it tokenizes numbers, and poor uncertainty calibration, which is likely the result of alignment interventions such as RLHF.

----

## [861] Learning Mixtures of Gaussians Using the DDPM Objective

**Authors**: *Kulin Shah, Sitan Chen, Adam R. Klivans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ec077b4af90f2556b517b556e186f64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ec077b4af90f2556b517b556e186f64-Abstract-Conference.html)

**Abstract**:

Recent works have shown that diffusion models can learn essentially any distribution provided one can perform score estimation.Yet it remains poorly understood under what settings score estimation is possible, let alone when practical gradient-based algorithms for this task can provably succeed. In this work, we give the first provably efficient results for one of the most fundamental distribution families, Gaussian mixture models.We prove that GD on the denoising diffusion probabilistic model (DDPM) objective can efficiently recover the ground truth parameters of the mixture model in the following two settings:1. We show GD with random initialization learns mixtures of two spherical Gaussians in $d$ dimensions with $1/\text{poly}(d)$-separated centers.2. We show GD with a warm start learns mixtures of $K$ spherical Gaussians with $\Omega(\sqrt{\log(\min(K,d))})$-separated centers.A key ingredient in our proofs is a new connection between score-based methods and two other approaches to distribution learning, EM and spectral methods.

----

## [862] Graph Convolutional Kernel Machine versus Graph Convolutional Networks

**Authors**: *Zhihao Wu, Zhao Zhang, Jicong Fan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ec6c6fc9065aa57785eb05dffe7c3db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ec6c6fc9065aa57785eb05dffe7c3db-Abstract-Conference.html)

**Abstract**:

Graph convolutional networks (GCN) with one or two hidden layers have been widely used in handling graph data that are prevalent in various disciplines. Many studies showed that the gain of making GCNs deeper is tiny or even negative. This implies that the complexity of graph data is often limited and shallow models are often sufficient to extract expressive features for various tasks such as node classification. Therefore, in this work, we present a framework called graph convolutional kernel machine (GCKM) for graph-based machine learning. GCKMs are built upon kernel functions integrated with graph convolution. An example is the graph convolutional kernel support vector machine (GCKSVM) for node classification, for which we analyze the generalization error bound and discuss the impact of the graph structure. Compared to GCNs, GCKMs require much less effort in architecture design, hyperparameter tuning, and optimization. More importantly, GCKMs are guaranteed to obtain globally optimal solutions and have strong generalization ability and high interpretability. GCKMs are composable, can be extended to large-scale data, and are applicable to various tasks (e.g., node or graph classification, clustering, feature extraction, dimensionality reduction). The numerical results on benchmark datasets show that, besides the aforementioned advantages, GCKMs have at least competitive accuracy compared to GCNs.

----

## [863] First Order Stochastic Optimization with Oblivious Noise

**Authors**: *Ilias Diakonikolas, Sushrut Karmalkar, Jong Ho Park, Christos Tzamos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ec90b0eec9c1151c152ba865713f184-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ec90b0eec9c1151c152ba865713f184-Abstract-Conference.html)

**Abstract**:

We initiate the study of stochastic optimization with oblivious noise, broadly generalizing the standard heavy-tailed noise setup.In our setting, in addition to random observation noise, the stochastic gradient may be subject to independent \emph{oblivious noise}, which may not have bounded moments and is not necessarily centered. Specifically, we assume access to a noisy oracle for the stochastic gradient of $f$ at $x$,  which returns a vector $\nabla f(\gamma, x) + \xi$, where $\gamma$ is the  bounded variance observation noise and $\xi$ is the oblivious noise that is independent of $\gamma$ and $x$. The only assumption we make on the oblivious noise $\xi$ is that $\Pr[\xi = 0] \ge \alpha$, for some $\alpha \in (0, 1)$.In this setting, it is not information-theoretically possible to recover a single solution close to the target when the fraction of inliers $\alpha$ is less than $1/2$. Our main result is an efficient {\em list-decodable} learner that recovers a small list of candidates at least one of which is close to the true solution. On the other hand, if $\alpha = 1-\epsilon$, where $0< \epsilon < 1/2$ is sufficiently smallconstant, the algorithm recovers a single solution.Along the way, we develop a rejection-sampling-based algorithm to perform noisy location estimation, which may be of independent interest.

----

## [864] CHAMMI: A benchmark for channel-adaptive models in microscopy imaging

**Authors**: *Zitong Sam Chen, Chau Pham, Siqi Wang, Michael Doron, Nikita Moshkov, Bryan A. Plummer, Juan C. Caicedo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ecca655ac67685fdc2155da0eceda6b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ecca655ac67685fdc2155da0eceda6b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Most neural networks assume that input images have a fixed number of channels (three for RGB images). However, there are many settings where the number of channels may vary, such as microscopy images where the number of channels changes depending on instruments and experimental goals. Yet, there has not been a systemic attempt to create and evaluate neural networks that are invariant to the number and type of channels. As a result, trained models remain specific to individual studies and are hardly reusable for other microscopy settings. In this paper, we present a benchmark for investigating channel-adaptive models in microscopy imaging, which consists of 1) a dataset of varied-channel single-cell images, and 2) a biologically relevant evaluation framework. In addition, we adapted several existing techniques to create channel-adaptive models and compared their performance on this benchmark to fixed-channel, baseline models. We find that channel-adaptive models can generalize better to out-of-domain tasks and can be computationally efficient. We contribute a curated dataset and an evaluation API to facilitate objective comparisons in future research and applications.

----

## [865] A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs

**Authors**: *Xingyue Huang, Miguel Romero, Ismail Ilkan Ceylan, Pablo Barceló*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3eceb70f47690051d6769739fbf6294b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3eceb70f47690051d6769739fbf6294b-Abstract-Conference.html)

**Abstract**:

Graph neural networks are prominent models for representation learning over graph-structured data. While the capabilities and limitations of these models are well-understood for simple graphs, our understanding remains incomplete in the context of knowledge graphs. Our goal is to provide a systematic understanding of the landscape of graph neural networks for knowledge graphs pertaining to the prominent task of link prediction. Our analysis entails a unifying perspective on seemingly unrelated models and unlocks a series of other models. The expressive power of various models is characterized via a corresponding relational Weisfeiler-Leman algorithm. This analysis is extended to provide a precise logical characterization of the class of functions captured by a class of graph neural networks. The theoretical findings presented in this paper explain the benefits of some widely employed practical design choices, which are validated empirically.

----

## [866] Bayes beats Cross Validation: Efficient and Accurate Ridge Regression via Expectation Maximization

**Authors**: *Shu Yu Tew, Mario Boley, Daniel F. Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3eec5006051d9544e717067de3220198-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3eec5006051d9544e717067de3220198-Abstract-Conference.html)

**Abstract**:

We present a novel method for tuning the regularization hyper-parameter, $\lambda$, of a ridge regression that is faster to compute than leave-one-out cross-validation (LOOCV) while yielding estimates of the regression parameters of equal, or particularly in the setting of sparse covariates, superior quality to those obtained by minimising the LOOCV risk. The LOOCV risk can suffer from multiple and bad local minima for finite $n$ and thus requires the specification of a set of candidate $\lambda$, which can fail to provide good solutions. In contrast, we show that the proposed method is guaranteed to find a unique optimal solution for large enough $n$, under relatively mild conditions, without requiring the specification of any difficult to determine hyper-parameters. This is based on a Bayesian formulation of ridge regression that we prove to have a unimodal posterior for large enough $n$, allowing for  both the optimal $\lambda$ and the regression coefficients to be jointly learned within an iterative expectation maximization (EM) procedure. Importantly, we show that by utilizing an appropriate preprocessing step, a single iteration of the main EM loop can be implemented in $O(\min(n, p))$ operations, for input data with $n$ rows and $p$ columns. In contrast, evaluating a single value of $\lambda$ using fast LOOCV costs $O(n \min(n, p))$ operations when using the same preprocessing. This advantage amounts to an asymptotic improvement of a factor of $l$ for $l$ candidate values for $\lambda$ (in the regime $q, p \in O(\sqrt{n})$ where $q$ is the number of regression targets).

----

## [867] Segment Everything Everywhere All at Once

**Authors**: *Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, Yong Jae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c726172b86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c726172b86-Abstract-Conference.html)

**Abstract**:

In this work, we present SEEM, a promotable and interactive model for segmenting everything everywhere all at once in an image. In SEEM, we propose a novel and versatile decoding mechanism that enables diverse prompting for all types of segmentation tasks, aiming at a universal interface that behaves like large language models (LLMs). More specifically, SEEM is designed with four desiderata:i) Versatility. We introduce a new visual prompt to unify different spatial queries including points, boxes, scribbles, and masks, which can further generalize to a different referring image; ii) Compositionality. We learn a joint visual-semantic space between text and visual prompts, which facilitates the dynamic composition of two prompt types required for various segmentation tasks, as shown in Fig. 1;iii) Interactivity. We further incorporate learnable memory prompts into the decoder to retain segmentation history through mask-guided cross-attention from the decoder to image features; iv) Semantic awareness. We use a text encoder to encode text queries and mask labels into the same semantic space for open-vocabulary segmentation. We conduct a comprehensive empirical study to validate the effectiveness of SEEM across diverse segmentation tasks. The results demonstrate that SEEM exhibits robust generalizing to unseen user intents as it learns to compose prompts of different types in a unified representation space. Our approach achieves competitive performance on interactive segmentation, generic segmentation, referring segmentation, and video object segmentation on 9 datasets with minimum 1/100 supervision in a single set of weights.

----

## [868] PUe: Biased Positive-Unlabeled Learning Enhancement by Causal Inference

**Authors**: *Xutao Wang, Hanting Chen, Tianyu Guo, Yunhe Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3efb4bdc6bfe13e1ff95b4407c37961d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3efb4bdc6bfe13e1ff95b4407c37961d-Abstract-Conference.html)

**Abstract**:

Positive-Unlabeled (PU) learning aims to achieve high-accuracy binary classification with limited labeled positive examples and numerous unlabeled ones. Existing cost-sensitive-based methods often rely on strong assumptions that examples with an observed positive label were selected entirely at random. In fact, the uneven distribution of labels is prevalent in real-world PU problems, indicating that most actual positive and unlabeled data are subject to selection bias. In this paper, we propose a PU learning enhancement (PUe) algorithm based on causal inference theory, which employs normalized propensity scores and normalized inverse probability weighting (NIPW) techniques to reconstruct the loss function, thus obtaining a consistent, unbiased estimate of the classifier and enhancing the model's performance. Moreover, we investigate and propose a method for estimating propensity scores in deep learning using regularization techniques when the labeling mechanism is unknown. Our experiments on three benchmark datasets demonstrate the proposed PUe algorithm significantly improves the accuracy of classifiers on non-uniform label distribution datasets compared to advanced cost-sensitive PU methods. Codes are available at https://github.com/huawei-noah/Noah-research/tree/master/PUe and https://gitee.com/mindspore/models/tree/master/research/cv/PUe.

----

## [869] Sparse Modular Activation for Efficient Sequence Modeling

**Authors**: *Liliang Ren, Yang Liu, Shuohang Wang, Yichong Xu, Chenguang Zhu, ChengXiang Zhai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f0739410e1c9c5da04fa10c1f3f86b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f0739410e1c9c5da04fa10c1f3f86b6-Abstract-Conference.html)

**Abstract**:

Recent hybrid models combining Linear State Space Models (SSMs) with self-attention mechanisms have demonstrated impressive results across a range of sequence modeling tasks. However, current approaches apply attention modules statically and uniformly to all elements in the input sequences, leading to sub-optimal quality-efficiency trade-offs. To address this limitation, we introduce Sparse Modular Activation (SMA), a general mechanism enabling neural networks to sparsely and dynamically activate sub-modules for sequence elements in a differentiable manner. Through allowing each element to skip non-activated sub-modules, SMA reduces computation and memory consumption of neural networks at both training and inference stages. To validate the effectiveness of SMA on sequence modeling, we design a novel neural architecture, SeqBoat, which employs SMA to sparsely activate a Gated Attention Unit (GAU) based on the state representations learned from an SSM. By constraining the GAU to only conduct local attention on the activated inputs, SeqBoat can achieve linear inference complexity with theoretically infinite attention span, and provide substantially better quality-efficiency trade-off than the chunking-based models. With experiments on a wide range of tasks, including long sequence modeling, speech classification and language modeling, SeqBoat brings new state-of-the-art results among hybrid models with linear complexity, and reveals the amount of attention needed for each task through the learned sparse activation patterns. Our code is publicly available at https://github.com/renll/SeqBoat.

----

## [870] BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting

**Authors**: *Patrick Emami, Abhijeet Sahu, Peter Graf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f17bf868966df01ca125e5bbc9ee24e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f17bf868966df01ca125e5bbc9ee24e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Short-term forecasting of residential and commercial building energy consumption is widely used in power systems and continues to grow in importance. Data-driven short-term load forecasting (STLF), although promising, has suffered from a lack of open, large-scale datasets with high building diversity. This has hindered exploring the pretrain-then-fine-tune paradigm for STLF. To help address this, we present BuildingsBench, which consists of: 1) Buildings-900K, a large-scale dataset of 900K simulated buildings representing the U.S. building stock; and 2) an evaluation platform with over 1,900 real residential and commercial buildings from 7 open datasets. BuildingsBench benchmarks two under-explored tasks: zero-shot STLF, where a pretrained model is evaluated on unseen buildings without fine-tuning, and transfer learning, where a pretrained model is fine-tuned on a target building. The main finding of our benchmark analysis is that synthetically pretrained models generalize surprisingly well to real commercial buildings. An exploration of the effect of increasing dataset size and diversity on zero-shot commercial building performance reveals a power-law with diminishing returns. We also show that fine-tuning pretrained models on real commercial and residential buildings improves performance for a majority of target buildings. We hope that BuildingsBench encourages and facilitates future research on generalizable STLF. All datasets and code can be accessed from https://github.com/NREL/BuildingsBench.

----

## [871] Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

**Authors**: *Steven Adriaensen, Herilalaina Rakotoarison, Samuel Müller, Frank Hutter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html)

**Abstract**:

Learning curve extrapolation aims to predict model performance in later epochs of training, based on the performance in earlier epochs.In this work, we argue that, while the inherent uncertainty in the extrapolation of learning curves warrants a Bayesian approach, existing methods are (i) overly restrictive, and/or (ii) computationally expensive. We describe the first application of prior-data fitted neural networks (PFNs) in this context. A PFN is a transformer, pre-trained on data generated from a prior, to perform approximate Bayesian inference in a single forward pass. We propose LC-PFN, a PFN trained to extrapolate 10 million artificial right-censored learning curves generated from a parametric prior proposed in prior art using MCMC. We demonstrate that LC-PFN can approximate the posterior predictive distribution more accurately than MCMC, while being over 10 000 times faster. We also show that the same LC-PFN achieves competitive performance extrapolating a total of 20 000 real learning curves from four learning curve benchmarks (LCBench, NAS-Bench-201, Taskset, and PD1) that stem from training a wide range of model architectures (MLPs, CNNs, RNNs, and Transformers) on 53 different datasets with varying input modalities (tabular, image, text, and protein data). Finally, we investigate its potential in the context of model selection and find that a simple LC-PFN based predictive early stopping criterion obtains 2 - 6x speed-ups on 45 of these datasets, at virtually no overhead.

----

## [872] Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective

**Authors**: *Zeyu Zhang, Yi Su, Hui Yuan, Yiran Wu, Rishab Balasubramanian, Qingyun Wu, Huazheng Wang, Mengdi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f1b6e97a5eb3b10e6b0c99b022988eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f1b6e97a5eb3b10e6b0c99b022988eb-Abstract-Conference.html)

**Abstract**:

Off-policy Learning to Rank (LTR) aims to optimize a ranker from data collected by a deployed logging policy. However, existing off-policy learning to rank methods often make strong assumptions about how users generate the click data, i.e., the click model, and hence need to tailor their methods specifically under different click models. In this paper, we unified the ranking process under general stochastic click models as a Markov Decision Process (MDP), and the optimal ranking could be learned with offline reinforcement learning (RL) directly. Building upon this, we leverage offline RL techniques for off-policy LTR and propose the Click Model-Agnostic Unified Off-policy Learning to Rank (CUOLR) method, which could be easily applied to a wide range of click models. Through a dedicated formulation of the MDP, we show that offline RL algorithms can adapt to various click models without complex debiasing techniques and prior knowledge of the model. Results on various large-scale datasets demonstrate that CUOLR consistently outperforms the state-of-the-art off-policy learning to rank algorithms while maintaining consistency and robustness under different click models.

----

## [873] Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints

**Authors**: *Dohyeong Kim, Kyungjae Lee, Songhwai Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f20f2b0315c72201e23512fdbd1ee91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f20f2b0315c72201e23512fdbd1ee91-Abstract-Conference.html)

**Abstract**:

In safety-critical robotic tasks, potential failures must be reduced, and multiple constraints must be met, such as avoiding collisions, limiting energy consumption, and maintaining balance.Thus, applying safe reinforcement learning (RL) in such robotic tasks requires to handle multiple constraints and use risk-averse constraints rather than risk-neutral constraints.To this end, we propose a trust region-based safe RL algorithm for multiple constraints called a safe distributional actor-critic (SDAC).Our main contributions are as follows: 1) introducing a gradient integration method to manage infeasibility issues in multi-constrained problems, ensuring theoretical convergence, and 2) developing a TD($\lambda$) target distribution to estimate risk-averse constraints with low biases. We evaluate SDAC through extensive experiments involving multi- and single-constrained robotic tasks.While maintaining high scores, SDAC shows 1.93 times fewer steps to satisfy all constraints in multi-constrained tasks and 1.78 times fewer constraint violations in single-constrained tasks compared to safe RL baselines.Code is available at: https://github.com/rllab-snu/Safe-Distributional-Actor-Critic.

----

## [874] The Contextual Lasso: Sparse Linear Models via Deep Neural Networks

**Authors**: *Ryan Thompson, Amir Dezfouli, Robert Kohn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f226824426a4d6ae3d3efad8883fc53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f226824426a4d6ae3d3efad8883fc53-Abstract-Conference.html)

**Abstract**:

Sparse linear models are one of several core tools for interpretable machine learning, a field of emerging importance as predictive models permeate decision-making in many domains. Unfortunately, sparse linear models are far less flexible as functions of their input features than black-box models like deep neural networks. With this capability gap in mind, we study a not-uncommon situation where the input features dichotomize into two groups: explanatory features, which are candidates for inclusion as variables in an interpretable model, and contextual features, which select from the candidate variables and determine their effects. This dichotomy leads us to the contextual lasso, a new statistical estimator that fits a sparse linear model to the explanatory features such that the sparsity pattern and coefficients vary as a function of the contextual features. The fitting process learns this function nonparametrically via a deep neural network. To attain sparse coefficients, we train the network with a novel lasso regularizer in the form of a projection layer that maps the network's output onto the space of $\ell_1$-constrained linear models. An extensive suite of experiments on real and synthetic data suggests that the learned models, which remain highly transparent, can be sparser than the regular lasso without sacrificing the predictive power of a standard deep neural network.

----

## [875] No Representation Rules Them All in Category Discovery

**Authors**: *Sagar Vaze, Andrea Vedaldi, Andrew Zisserman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f52ab4322e967efd312c38a68d07f01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f52ab4322e967efd312c38a68d07f01-Abstract-Conference.html)

**Abstract**:

In this paper we tackle the problem of Generalized Category Discovery (GCD). Specifically, given a dataset with labelled and unlabelled images, the task is to cluster all images in the unlabelled subset, whether or not they belong to the labelled categories. Our first contribution is to recognise that most existing GCD benchmarks only contain labels for a single clustering of the data, making it difficult to ascertain whether models are leveraging the available labels to solve the GCD task, or simply solving an unsupervised clustering problem. As such, we present a synthetic dataset, named 'Clevr-4', for category discovery. Clevr-4 contains four equally valid partitions of the data, i.e based on object 'shape', 'texture' or 'color' or 'count'. To solve the task, models are required to extrapolate the taxonomy specified by labelled set, rather than simply latch onto a single natural grouping of the data. We use this dataset to demonstrate the limitations of unsupervised clustering in the GCD setting, showing that even very strong unsupervised models fail on Clevr-4. We further use Clevr-4 to examine the weaknesses of existing GCD algorithms, and propose a new method which addresses these shortcomings, leveraging consistent findings from the representation learning literature to do so. Our simple solution, which is based on `Mean Teachers' and termed $\mu$GCD, substantially outperforms implemented baselines on Clevr-4. Finally, when we transfer these findings to real data on the challenging Semantic Shift Benchmark suite, we find that $\mu$GCD outperforms all prior work, setting a new state-of-the-art.

----

## [876] CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions

**Authors**: *Juan M. Cardenas, Ben Adcock, Nick C. Dexter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f8c7eb848ffec848f3ed2b7ca44915d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f8c7eb848ffec848f3ed2b7ca44915d-Abstract-Conference.html)

**Abstract**:

We introduce a general framework for active learning in regression problems. Our framework extends the standard setup by allowing for general types of data, rather than merely pointwise samples of the target function. This generalization covers many cases of practical interest, such as data acquired in transform domains (e.g., Fourier data), vector-valued data (e.g., gradient-augmented data), data acquired along continuous curves, and, multimodal data (i.e., combinations of different types of measurements).  Our framework considers random sampling according to a finite number of sampling measures and arbitrary nonlinear approximation spaces (model classes). We introduce the concept of \textit{generalized Christoffel functions} and show how these can be used to optimize the sampling measures. We prove that this leads to near-optimal sample complexity in various important cases. This paper focuses on applications in scientific computing, where active learning is often desirable, since it is usually expensive to generate data. We demonstrate the efficacy of our framework for gradient-augmented learning with polynomials, Magnetic Resonance Imaging (MRI) using generative models and adaptive sampling for solving PDEs using Physics-Informed Neural Networks (PINNs).

----

## [877] Two Heads are Better Than One: A Simple Exploration Framework for Efficient Multi-Agent Reinforcement Learning

**Authors**: *Jiahui Li, Kun Kuang, Baoxiang Wang, Xingchen Li, Fei Wu, Jun Xiao, Long Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fa2d2b637122007845a2fbb7c21453b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fa2d2b637122007845a2fbb7c21453b-Abstract-Conference.html)

**Abstract**:

Exploration strategy plays an important role in reinforcement learning, especially in sparse-reward tasks. In cooperative multi-agent reinforcement learning~(MARL), designing a suitable exploration strategy is much more challenging due to the large state space and the complex interaction among agents. Currently, mainstream exploration methods in MARL either contribute to exploring the unfamiliar states which are large and sparse, or measuring the interaction among agents with high computational costs. We found an interesting phenomenon that different kinds of exploration plays a different role in different MARL scenarios, and choosing a suitable one is often more effective than designing an exquisite algorithm. In this paper, we propose a exploration method that incorporate the \underline{C}uri\underline{O}sity-based and \underline{IN}fluence-based exploration~(COIN) which is simple but effective in various situations. First, COIN measures the influence of each agent on the other agents based on mutual information theory and designs it as intrinsic rewards which are applied to each individual value function. Moreover, COIN computes the curiosity-based intrinsic rewards via prediction errors which are added to the extrinsic reward. For integrating the two kinds of intrinsic rewards, COIN utilizes a novel framework in which they complement each other and lead to a sufficient and effective exploration on cooperative MARL tasks. We perform extensive experiments on different challenging benchmarks, and results across different scenarios show the superiority of our method.

----

## [878] Cross-Scale MAE: A Tale of Multiscale Exploitation in Remote Sensing

**Authors**: *Maofeng Tang, Andrei Cozma, Konstantinos Georgiou, Hairong Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fadcbd0437f4717723ff3f6f7216800-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fadcbd0437f4717723ff3f6f7216800-Abstract-Conference.html)

**Abstract**:

Remote sensing images present unique challenges to image analysis due to the extensive geographic coverage, hardware limitations, and misaligned multi-scale images. This paper revisits the classical multi-scale representation learning prob- lem but under the general framework of self-supervised learning for remote sensing image understanding. We present Cross-Scale MAE, a self-supervised model built upon the Masked Auto-Encoder (MAE). During pre-training, Cross-Scale MAE employs scale augmentation techniques and enforces cross-scale consistency constraints through both contrastive and generative losses to ensure consistent and meaningful representations well-suited for a wide range of downstream tasks. Further, our implementation leverages the xFormers library to accelerate network pre-training on a single GPU while maintaining the quality of learned represen- tations. Experimental evaluations demonstrate that Cross-Scale MAE exhibits superior performance compared to standard MAE and other state-of-the-art remote sensing MAE methods.

----

## [879] MotionGPT: Human Motion as a Foreign Language

**Authors**: *Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html)

**Abstract**:

Though the advancement of pre-trained large language models unfolds, the exploration of building a unified model for language and other multimodal data, such as motion, remains challenging and untouched so far. Fortunately, human motion displays a semantic coupling akin to human language, often perceived as a form of body language. By fusing language data with large-scale motion models, motion-language pre-training that can enhance the performance of motion-related tasks becomes feasible. Driven by this insight, we propose MotionGPT, a unified, versatile, and user-friendly motion-language model to handle multiple motion-relevant tasks. Specifically, we employ the discrete vector quantization for human motion and transfer 3D motion into motion tokens, similar to the generation process of word tokens. Building upon this "motion vocabulary", we perform language modeling on both motion and text in a unified manner, treating human motion as a specific language. Moreover, inspired by prompt learning, we pre-train MotionGPT with a mixture of motion-language data and fine-tune it on prompt-based question-and-answer tasks. Extensive experiments demonstrate that MotionGPT achieves state-of-the-art performances on multiple motion tasks including text-driven motion generation, motion captioning, motion prediction, and motion in-between.

----

## [880] Model-Free Reinforcement Learning with the Decision-Estimation Coefficient

**Authors**: *Dylan J. Foster, Noah Golowich, Jian Qian, Alexander Rakhlin, Ayush Sekhari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fcd0f8747f9217c6dbc45ed138b1fde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fcd0f8747f9217c6dbc45ed138b1fde-Abstract-Conference.html)

**Abstract**:

We consider the problem of interactive decision making, encompassing structured bandits and reinforcementlearning with general function approximation. Recently, Foster et al. (2021) introduced theDecision-Estimation Coefficient, a measure of statistical complexity that lower bounds the optimal regret for interactive decisionmaking, as well as a meta-algorithm, Estimation-to-Decisions, which achieves upperbounds in terms of the same quantity. Estimation-to-Decisions is a reduction, which liftsalgorithms for (supervised) online estimation into algorithms fordecision making. In this paper, we show that by combining Estimation-to-Decisions witha specialized form of "optimistic" estimation introduced byZhang (2022), it is possible to obtain guaranteesthat improve upon those of Foster et al. (2021) byaccommodating more lenient notions of estimation error. We use this approach to derive regret bounds formodel-free reinforcement learning with value function approximation, and give structural results showing when it can and cannot help more generally.

----

## [881] FlowPG: Action-constrained Policy Gradient with Normalizing Flows

**Authors**: *Janaka Chathuranga Brahmanage, Jiajing Ling, Akshat Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fd9fe8ec6d7238bf71784797399bb61-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fd9fe8ec6d7238bf71784797399bb61-Abstract-Conference.html)

**Abstract**:

Action-constrained reinforcement learning (ACRL) is a popular approach for solving safety-critical and resource-allocation related decision making problems. A major challenge in ACRL is to ensure agent taking a valid action satisfying constraints in each RL step. Commonly used approach of using a projection layer on top of the policy network requires solving an optimization program which can result in longer training time, slow convergence, and zero gradient problem. To address this, first we use a normalizing flow model to learn an invertible, differentiable mapping between the feasible action space and the support of a simple distribution on a latent variable, such as Gaussian. Second, learning the flow model requires sampling from the feasible action space, which is also challenging. We develop multiple methods, based on Hamiltonian Monte-Carlo and probabilistic sentential decision diagrams for such action sampling for convex and non-convex constraints. Third, we integrate the learned normalizing flow with the DDPG algorithm. By design, a well-trained normalizing flow will transform policy output into a valid action without requiring an optimization solver. Empirically, our approach results in significantly fewer constraint violations (upto an order-of-magnitude for several instances) and is multiple times faster on a variety of continuous control tasks.

----

## [882] Distributionally Robust Bayesian Optimization with φ-divergences

**Authors**: *Hisham Husain, Vu Nguyen, Anton van den Hengel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3feb8ed3c33c3310b45f80be7dfef707-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3feb8ed3c33c3310b45f80be7dfef707-Abstract-Conference.html)

**Abstract**:

The study of robustness has received much attention due to its inevitability in data-driven settings where many systems face uncertainty. One such example of concern is Bayesian Optimization (BO), where uncertainty is multi-faceted, yet there only exists a limited number of works dedicated to this direction. In particular, there is the work of Kirschner et al., which bridges the existing literature of Distributionally Robust Optimization (DRO) by casting the BO problem from the lens of DRO. While this work is pioneering, it admittedly suffers from various practical shortcomings such as finite contexts assumptions, leaving behind the main question \textit{Can one devise a computationally tractable algorithm for solving this DRO-BO problem}? In this work, we tackle this question to a large degree of generality by considering robustness against data-shift in $\varphi$-divergences, which subsumes many popular choices, such as the $\chi^2$-divergence, Total Variation, and the extant Kullback-Leibler (KL) divergence. We show that the DRO-BO problem in this setting is equivalent to a finite-dimensional optimization problem which, even in the continuous context setting, can be easily implemented with provable sublinear regret bounds. We then show experimentally that our method surpasses existing methods, attesting to the theoretical results.

----

## [883] Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems

**Authors**: *Sihan Zeng, Thinh T. Doan, Justin Romberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ff48dde82306fe8f26f3e51dd1054d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ff48dde82306fe8f26f3e51dd1054d7-Abstract-Conference.html)

**Abstract**:

The aim of this paper is to improve the understanding of the optimization landscape for policy optimization problems in reinforcement learning. Specifically, we show that the superlevel set of the objective function with respect to the policy parameter is always a connected set both in the tabular setting and under policies represented by a class of neural networks. In addition, we show that the optimization objective as a function of the policy parameter and reward satisfies a stronger “equiconnectedness” property. To our best knowledge, these are novel and previously unknown discoveries.We present an application of the connectedness of these superlevel sets to the derivation of minimax theorems for robust reinforcement learning. We show that any minimax optimization program which is convex on one side and is equiconnected on the other side observes the minimax equality (i.e. has a Nash equilibrium). We find that this exact structure is exhibited by an interesting class of robust reinforcement learning problems under an adversarial reward attack, and the validity of its minimax equality immediately follows. This is the first time such a result is established in the literature.

----

## [884] Towards Efficient and Accurate Winograd Convolution via Full Quantization

**Authors**: *Tianqi Chen, Weixiang Xu, Weihan Chen, Peisong Wang, Jian Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/400a2e6a82520b690810b97fd67fcc4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/400a2e6a82520b690810b97fd67fcc4e-Abstract-Conference.html)

**Abstract**:

The Winograd algorithm is an efficient convolution implementation, which performs calculations in the transformed domain. To further improve the computation efficiency, recent works propose to combine it with model quantization. Although Post-Training Quantization has the advantage of low computational cost and has been successfully applied in many other scenarios, a severe accuracy drop exists when utilizing it in Winograd convolution. Besides, despite the Winograd algorithm consisting of four stages, most existing methods only quantize the element-wise multiplication stage, leaving a considerable portion of calculations in full precision.In this paper, observing the inconsistency among different transformation procedures, we present PTQ-Aware Winograd (PAW) to optimize them collaboratively under a unified objective function. Moreover, we explore the full quantization of faster Winograd (tile size $\geq4$) for the first time. We further propose a hardware-friendly method called Factorized Scale Quantization (FSQ), which can effectively balance the significant range differences in the Winograd domain. Experiments demonstrate the effectiveness of our method, e.g., with 8-bit quantization and a tile size of 6, our method outperforms the previous Winograd PTQ method by 8.27\% and 5.38\% in terms of the top-1 accuracy on ResNet-18 and ResNet-34, respectively.

----

## [885] Quantum Bayesian Optimization

**Authors**: *Zhongxiang Dai, Gregory Kang Ruey Lau, Arun Verma, Yao Shu, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/401aa72e0e3be680348a5b0ffdb1a5aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/401aa72e0e3be680348a5b0ffdb1a5aa-Abstract-Conference.html)

**Abstract**:

Kernelized bandits, also known as Bayesian optimization (BO), has been a prevalent method for optimizing complicated black-box reward functions. Various BO algorithms have been theoretically shown to enjoy upper bounds on their cumulative regret which are sub-linear in the number $T$ of iterations, and a regret lower bound of $\Omega(\sqrt{T})$ has been derived which represents the unavoidable regrets for any classical BO algorithm. Recent works on quantum bandits have shown that with the aid of quantum computing, it is possible to achieve tighter regret upper bounds better than their corresponding classical lower bounds. However, these works are restricted to either multi-armed or linear bandits, and are hence not able to solve sophisticated real-world problems with non-linear reward functions. To this end, we introduce the quantum-Gaussian process-upper confidence bound (Q-GP-UCB) algorithm. To the best of our knowledge, our Q-GP-UCB is the first BO algorithm able to achieve a regret upper bound of $\mathcal{O}(\text{poly}\log T)$, which is significantly smaller than its regret lower bound of $\Omega(\sqrt{T})$ in the classical setting. Moreover, thanks to our novel analysis of the confidence ellipsoid, our Q-GP-UCB with the linear kernel achieves a smaller regret than the quantum linear UCB algorithm from the previous work. We use simulations, as well as an experiment using a real quantum computer, to verify that the theoretical quantum speedup achieved by our Q-GP-UCB is also potentially relevant in practice.

----

## [886] Interpretable Reward Redistribution in Reinforcement Learning: A Causal Approach

**Authors**: *Yudi Zhang, Yali Du, Biwei Huang, Ziyan Wang, Jun Wang, Meng Fang, Mykola Pechenizkiy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/402e12102d6ec3ea3df40ce1b23d423a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/402e12102d6ec3ea3df40ce1b23d423a-Abstract-Conference.html)

**Abstract**:

A major challenge in reinforcement learning is to determine which state-action pairs are responsible for future rewards that are delayed. Reward redistribution serves as a solution to re-assign credits for each time step from observed sequences.  While the majority of current approaches construct the reward redistribution in an uninterpretable manner, we propose to explicitly model the contributions of state and action from a causal perspective, resulting in an interpretable reward redistribution and preserving policy invariance. In this paper, we start by studying the role of causal generative models in reward redistribution by characterizing the generation of Markovian rewards and trajectory-wise long-term return and further propose a framework, called Generative Return Decomposition (GRD), for policy optimization in delayed reward scenarios. Specifically, GRD first identifies the unobservable Markovian rewards and causal relations in the generative process. Then,  GRD makes use of the identified causal generative model to form a compact representation to train policy over the most favorable subspace of the state space of the agent. Theoretically, we show that the unobservable Markovian reward function is identifiable, as well as the underlying causal structure and causal models. Experimental results show that our method outperforms state-of-the-art methods and the provided visualization further demonstrates the interpretability of our method.The project page is located at https://reedzyd.github.io/GenerativeReturnDecomposition/.

----

## [887] Guarantees for Self-Play in Multiplayer Games via Polymatrix Decomposability

**Authors**: *Revan MacQueen, James R. Wright*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/40386e4770bebd63fdf47cbc67341c0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/40386e4770bebd63fdf47cbc67341c0b-Abstract-Conference.html)

**Abstract**:

Self-play is a technique for machine learning in multi-agent systems where a learning algorithm learns by interacting with copies of itself. Self-play is useful for generating large quantities of data for learning, but has the drawback that the agents the learner will face post-training may have dramatically different behavior than the learner came to expect by interacting with itself. For the special case of two-player constant-sum games, self-play that reaches Nash equilibrium is guaranteed to produce strategies that perform well against any post-training opponent; however, no such guarantee exists for multiplayer games. We show that in games that approximately decompose into a set of two-player constant-sum games (called constant-sum polymatrix games) where global $\epsilon$-Nash equilibria are boundedly far from Nash equilibria in each subgame (called subgame stability), any no-external-regret algorithm that learns by self-play will produce a strategy with bounded vulnerability. For the first time, our results identify a structural property of multiplayer games that enable performance guarantees for the strategies produced by a broad class of self-play algorithms. We demonstrate our findings through experiments on Leduc poker.

----

## [888] VCC: Scaling Transformers to 128K Tokens or More by Prioritizing Important Tokens

**Authors**: *Zhanpeng Zeng, Cole Hawkins, Mingyi Hong, Aston Zhang, Nikolaos Pappas, Vikas Singh, Shuai Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html)

**Abstract**:

Transformers are central in modern natural language processing and computer vision applications. Despite recent works devoted to reducing the quadratic cost of such models with respect to sequence length, dealing with ultra long sequences (e.g., $>$16K tokens) remains challenging. Applications such as answering questions based on a book or summarizing a scientific article are inefficient or infeasible. Here, we propose to significantly improve the efficiency of Transformers for ultra long sequences, by compressing the sequence into a much smaller representation at each layer. Specifically, by exploiting the fact that in many tasks, only a small subset of special tokens, which we call VIP-tokens, are most relevant to the final prediction, we propose a VIP-token centric compression (VCC) scheme which selectively compresses the sequence based on their impact on approximating the representation of the VIP-tokens. Compared with competitive baselines, our algorithm is not only efficient (achieving more than $3\times$ compute efficiency gain compared to baselines on 4K and 16K lengths), but also offers competitive/better performance on a large number of tasks. Further, we show that our algorithm scales to 128K tokens (or more) while consistently offering accuracy improvement. Code is available at https://github.com/mlpen/VCC.

----

## [889] Greatness in Simplicity: Unified Self-Cycle Consistency for Parser-Free Virtual Try-On

**Authors**: *Chenghu Du, Junyin Wang, Shuqing Liu, Shengwu Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4065a881baab1744bfba208a4361bbb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4065a881baab1744bfba208a4361bbb1-Abstract-Conference.html)

**Abstract**:

Image-based virtual try-on tasks remain challenging, primarily due to inherent complexities associated with non-rigid garment deformation modeling and strong feature entanglement of clothing within human body. Recent groundbreaking formulations, such as in-painting, cycle consistency, and knowledge distillation, have facilitated self-supervised generation of try-on images. However, these paradigms necessitate the disentanglement of garment features within human body features through auxiliary tasks, such as leveraging 'teacher knowledge' and dual generators. The potential presence of irresponsible prior knowledge in the auxiliary task can serve as a significant bottleneck for the main generator (e.g., 'student model') in the downstream task. Moreover, existing garment deformation methods lack the ability to perceive the correlation between the garment and the human body in the real world, leading to unrealistic alignment effects. To tackle these limitations, we present a new parser-free virtual try-on network based on unified self-cycle consistency (USC-PFN), which enables robust translation between different garments using just a single generator, faithfully replicating non-rigid geometric deformation of garments in real-life scenarios. Specifically, we first propose a self-cycle consistency architecture with a circular mode. It utilizes real unpaired garment-person images exclusively as input for training, effectively eliminating the impact of irresponsible prior knowledge at the model input end. Additionally, we formulate a Markov Random Field to simulate a more natural and realistic garment deformation. Furthermore, USC-PFN can leverage a general generator for self-supervised cycle training. Experiments demonstrate that our method achieves state-of-the-art performance on a popular virtual try-on benchmark.

----

## [890] VPGTrans: Transfer Visual Prompt Generator across LLMs

**Authors**: *Ao Zhang, Hao Fei, Yuan Yao, Wei Ji, Li Li, Zhiyuan Liu, Tat-Seng Chua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html)

**Abstract**:

Since developing a new multimodal LLM (MLLM) by pre-training on tremendous image-text pairs from scratch can be exceedingly resource-consuming, connecting an existing LLM with a comparatively lightweight visual prompt generator (VPG) becomes a feasible paradigm. However, further tuning the VPG component of the MLLM still incurs significant computational costs, such as thousands of GPU hours and millions of training data points. An alternative solution is transferring an existing VPG from one MLLM to the target MLLM. In this work, we investigate VPG transferability across LLMs for the first time, aiming to reduce the cost of VPG training.  Specifically, we explore VPG transfer across different LLM sizes (e.g., small-to-large) and types.  We identify key factors to maximize transfer efficiency, based on which we develop a simple yet highly effective two-stage transfer framework, called VPGTrans. Notably, it enables VPG transfer from BLIP-2 OPT 2.7B to BLIP-2 OPT 6.7B with less than 10% of the GPU hours using only 10.7% of the training data compared to training a VPG for OPT 6.7B from scratch. Furthermore, we provide a series of intriguing findings and discuss potential explanations behind them. Finally, we showcase the practical value of our VPGTrans approach, by customizing two novel MLLMs, including VL-LLaMA and VL-Vicuna, with recently released LLaMA and Vicuna LLMs.

----

## [891] Nearest Neighbour with Bandit Feedback

**Authors**: *Stephen Pasteris, Chris Hicks, Vasilios Mavroudis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4078c8b648dc107aedbdf561dd4edc2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4078c8b648dc107aedbdf561dd4edc2a-Abstract-Conference.html)

**Abstract**:

In this paper we adapt the nearest neighbour rule to the contextual bandit problem. Our algorithm handles the fully adversarial setting in which no assumptions at all are made about the data-generation process. When combined with a sufficiently fast data-structure for (perhaps approximate) adaptive nearest neighbour search, such as a navigating net, our algorithm is extremely efficient - having a per trial running time polylogarithmic in both the number of trials and actions, and taking only quasi-linear space. We give generic regret bounds for our algorithm and further analyse them when applied to the stochastic bandit problem in euclidean space. A side result of this paper is that, when applied to the online classification problem with stochastic labels, our algorithm can, under certain conditions, have sublinear regret whilst only finding a single nearest neighbour per trial - in stark contrast to the k-nearest neighbours algorithm.

----

## [892] Generative Neural Fields by Mixtures of Neural Implicit Functions

**Authors**: *Tackgeun You, Mijeong Kim, Jungtaek Kim, Bohyung Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/407fb8c5f3fda374c57d1bb18313ea5d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/407fb8c5f3fda374c57d1bb18313ea5d-Abstract-Conference.html)

**Abstract**:

We propose a novel approach to learning the generative neural fields represented by linear combinations of implicit basis networks. Our algorithm learns basis networks in the form of implicit neural representations and their coefficients in a latent space by either conducting meta-learning or adopting auto-decoding paradigms. The proposed method easily enlarges the capacity of generative neural fields by increasing the number of basis networks while maintaining the size of a network for inference to be small through their weighted model averaging. Consequently, sampling instances using the model is efficient in terms of latency and memory footprint. Moreover, we customize denoising diffusion probabilistic model for a target task to sample latent mixture coefficients, which allows our final model to generate unseen data effectively. Experiments show that our approach achieves competitive generation performance on diverse benchmarks for images, voxel data, and NeRF scenes without sophisticated designs for specific modalities and domains.

----

## [893] MAViL: Masked Audio-Video Learners

**Authors**: *Po-Yao Huang, Vasu Sharma, Hu Xu, Chaitanya Ryali, Haoqi Fan, Yanghao Li, Shang-Wen Li, Gargi Ghosh, Jitendra Malik, Christoph Feichtenhofer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/40b60852a4abdaa696b5a1a78da34635-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/40b60852a4abdaa696b5a1a78da34635-Abstract-Conference.html)

**Abstract**:

We present Masked Audio-Video Learners (MAViL) to learn audio-visual representations with three complementary forms of self-supervision: (1) reconstructing masked raw audio and video inputs, (2) intra-modal and inter-modal contrastive learning with masking, and (3) self-training to predict aligned and contextualized audio-video representations learned from the first two objectives. Empirically, MAViL achieves state-of-the-art audio-video classification performance on AudioSet (53.3 mAP) and VGGSound (67.1\% accuracy), surpassing recent self-supervised models and supervised models that utilize external labeled data. Notably, pre-training with MAViL not only enhances performance in multimodal classification and retrieval tasks, but it also improves the representations of each modality in isolation, without relying on information from the other modality during uni-modal fine-tuning or inference. The code and models are available at https://github.com/facebookresearch/MAViL.

----

## [894] Combating Representation Learning Disparity with Geometric Harmonization

**Authors**: *Zhihan Zhou, Jiangchao Yao, Feng Hong, Ya Zhang, Bo Han, Yanfeng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/40bb79c081828bebdc39d65a82367246-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/40bb79c081828bebdc39d65a82367246-Abstract-Conference.html)

**Abstract**:

Self-supervised learning (SSL) as an effective paradigm of representation learning has achieved tremendous success on various curated datasets in diverse scenarios. Nevertheless, when facing the long-tailed distribution in real-world applications, it is still hard for existing methods to capture transferable and robust representation. The attribution is that the vanilla SSL methods that pursue the sample-level uniformity easily leads to representation learning disparity, where head classes with the huge sample number dominate the feature regime but tail classes with the small sample number passively collapse. To address this problem, we propose a novel Geometric Harmonization (GH) method to encourage the category-level uniformity in representation learning, which is more benign to the minority and almost does not hurt the majority under long-tailed distribution. Specially, GH measures the population statistics of the embedding space on top of self-supervised learning, and then infer an fine-grained instance-wise calibration to constrain the space expansion of head classes and avoid the passive collapse of tail classes. Our proposal does not alter the setting of SSL and can be easily integrated into existing methods in a low-cost manner. Extensive results on a range of benchmark datasets show the effectiveness of \methodspace with high tolerance to the distribution skewness.

----

## [895] BioMassters: A Benchmark Dataset for Forest Biomass Estimation using Multi-modal Satellite Time-series

**Authors**: *Andrea Nascetti, Ritu Yadav, Kirill Brodt, Qixun Qu, Hongwei Fan, Yuri Shendryk, Isha Shah, Christine Chung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/40daf2a00278c4bea1b26cd4c8a654f8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/40daf2a00278c4bea1b26cd4c8a654f8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Above Ground Biomass is an important variable as forests play a crucial role in mitigating climate change as they act as an efficient, natural and cost-effective carbon sink. Traditional field and airborne LiDAR measurements have been proven to provide reliable estimations of forest biomass. Nevertheless, the use of these techniques at a large scale can be challenging and expensive. Satellite data have been widely used as a valuable tool in estimating biomass on a global scale. However, the full potential of dense multi-modal satellite time series data, in combination with modern deep learning approaches, has yet to be fully explored. The aim of the "BioMassters" data challenge and benchmark dataset is to investigate the potential of multi-modal satellite data (Sentinel-1 SAR and Sentinel-2 MSI) to estimate forest biomass at a large scale using the Finnish Forest Centre's open forest and nature airborne LiDAR data as a reference. The performance of the top three baseline models shows the potential of deep learning to produce accurate and higher-resolution biomass maps. Our benchmark dataset is publically available at https://huggingface.co/datasets/nascetti-a/BioMassters (doi:10.57967/hf/1009) and the implementation of the top three winning models are available at https://github.com/drivendataorg/the-biomassters.

----

## [896] Online Inventory Problems: Beyond the iid Setting with Online Convex Optimization

**Authors**: *Massil Hihat, Stéphane Gaïffas, Guillaume Garrigos, Simon Bussy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41128e5b3a7622da5b17588757599077-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41128e5b3a7622da5b17588757599077-Abstract-Conference.html)

**Abstract**:

We study multi-product inventory control problems where a manager makes sequential replenishment decisions based on partial historical information in order to minimize its cumulative losses. Our motivation is to consider general demands, losses and dynamics to go beyond standard models which usually rely on newsvendor-type losses, fixed dynamics, and unrealistic i.i.d. demand assumptions. We propose MaxCOSD, an online algorithm that has provable guarantees even for problems with non-i.i.d. demands and stateful dynamics, including for instance perishability. We consider what we call non-degeneracy assumptions on the demand process, and argue that they are necessary to allow learning.

----

## [897] On kernel-based statistical learning theory in the mean field limit

**Authors**: *Christian Fiedler, Michael Herty, Sebastian Trimpe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/411fa9d368b5485be4c6bb62615b365e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/411fa9d368b5485be4c6bb62615b365e-Abstract-Conference.html)

**Abstract**:

In many applications of machine learning, a large number of variables are considered. Motivated by machine learning of interacting particle systems, we consider the situation when the number of input variables goes to infinity. First, we continue the recent investigation of the mean field limit of kernels and their reproducing kernel Hilbert spaces, completing the existing theory. Next, we provide results relevant for approximation with such kernels in the mean field limit, including a representer theorem. Finally, we use these kernels in the context of statistical learning in the mean field limit, focusing on Support Vector Machines. In particular, we show mean field convergence of empirical and infinite-sample solutions as well as the convergence of the corresponding risks. On the one hand, our results establish rigorous mean field limits in the context of kernel methods, providing new theoretical tools and insights for large-scale problems. On the other hand, our setting corresponds to a new form of limit of learning problems, which seems to have not been investigated yet in the statistical learning theory literature.

----

## [898] Benchmarking Encoder-Decoder Architectures for Biplanar X-ray to 3D Bone Shape Reconstruction

**Authors**: *Mahesh Shakya, Bishesh Khanal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Various deep learning models have been proposed for 3D bone shape reconstruction from two orthogonal (biplanar) X-ray images.However, it is unclear how these models compare against each other since they are evaluated on different anatomy, cohort and (often privately held) datasets.Moreover, the impact of the commonly optimized image-based segmentation metrics such as dice score on the estimation of clinical parameters relevant in 2D-3D bone shape reconstruction is not well known.To move closer toward clinical translation, we propose a benchmarking framework that evaluates tasks relevant to real-world clinical scenarios, including reconstruction of fractured bones, bones with implants, robustness to population shift, and error in estimating clinical parameters.Our open-source platform provides reference implementations of 8 models (many of whose implementations were not publicly available), APIs to easily collect and preprocess 6 public datasets, and the implementation of automatic clinical parameter and landmark extraction methods. We present an extensive evaluation of 8 2D-3D models on equal footing using 6 public datasets comprising images for four different anatomies.Our results show that attention-based methods that capture global spatial relationships tend to perform better across all anatomies and datasets; performance on clinically relevant subgroups may be overestimated without disaggregated reporting; ribs are substantially more difficult to reconstruct compared to femur, hip and spine; and the dice score improvement does not always bring corresponding improvement in the automatic estimation of clinically relevant parameters.

----

## [899] 3D-LLM: Injecting the 3D World into Large Language Models

**Authors**: *Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, Chuang Gan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/413885e70482b95dcbeeddc1daf39177-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/413885e70482b95dcbeeddc1daf39177-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) and Vision-Language Models (VLMs) have been proved to excel at multiple tasks, such as commonsense reasoning. Powerful as these models can be, they are not grounded in the 3D physical world, which involves richer concepts such as spatial relationships, affordances, physics, layout, and so on. In this work, we propose to inject the 3D world into large language models, and introduce a whole new family of 3D-LLMs. Specifically, 3D-LLMs can take 3D point clouds and their features as input and perform a diverse set of 3D-related tasks, including captioning, dense captioning, 3D question answering, task decomposition, 3Dgrounding, 3D-assisted dialog, navigation, and so on. Using three types of prompting mechanisms that we design, we are able to collect over 300k 3D-language data covering these tasks. To efficiently train 3D-LLMs, we first utilize a 3D feature extractor that obtains 3D features from rendered multi-view images. Then, we use 2D VLMs as our backbones to train our 3D-LLMs. By introducing a 3D localization mechanism, 3D-LLMs could better capture 3D spatial information.  Experiments on ScanQA  show that our model outperforms state-of-the-art baselines by a large margin (\textit{e.g.}, the BLEU-1 score surpasses state-of-the-art score by 9\%). Furthermore, experiments on our held-in datasets for 3D captioning, task composition, and 3D-assisted dialogue show that our model outperforms 2D VLMs. Qualitative examples also show that our model could perform more tasks beyond the scope of existing LLMs and VLMs. Our model and data will be publicly available.

----

## [900] An Optimal and Scalable Matrix Mechanism for Noisy Marginals under Convex Loss Functions

**Authors**: *Yingtai Xiao, Guanlin He, Danfeng Zhang, Daniel Kifer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/414f4c9fe9653e5de98fad6964d50315-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/414f4c9fe9653e5de98fad6964d50315-Abstract-Conference.html)

**Abstract**:

Noisy marginals are a common form of confidentiality-protecting data release and are useful for many downstream tasks such as contingency table analysis, construction of Bayesian networks, and even synthetic data generation. Privacy mechanisms that provide unbiased noisy answers to linear queries (such as marginals) are known as matrix mechanisms.We propose ResidualPlanner, a matrix mechanism for marginals with Gaussian noise that is both optimal and scalable. ResidualPlanner can optimize for many loss functions that can be written as a convex function of marginal variances (prior work was restricted to just one predefined objective function). ResidualPlanner can optimize the accuracy of marginals in large scale settings in seconds, even when the previous state of the art (HDMM) runs out of memory. It even runs on datasets with 100 attributes in a couple of minutes. Furthermore ResidualPlanner can efficiently compute variance/covariance values for each marginal (prior methods quickly run out of memory, even for relatively small datasets).

----

## [901] Recovering Unbalanced Communities in the Stochastic Block Model with Application to Clustering with a Faulty Oracle

**Authors**: *Chandra Sekhar Mukherjee, Pan Peng, Jiapeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41623b137cd34807f56028aa9f6f84a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41623b137cd34807f56028aa9f6f84a7-Abstract-Conference.html)

**Abstract**:

The stochastic block model (SBM) is a fundamental model for studying graph clustering or community detection in networks. It has received great attention in the last decade and the balanced case, i.e., assuming all clusters have large size, has been well studied. However, our understanding of SBM with unbalanced communities (arguably, more relevant in practice) is still limited. In this paper, we provide a simple SVD-based algorithm for recovering the communities in the SBM with communities of varying sizes.We improve upon a result of Ailon, Chen and Xu [ICML 2013; JMLR 2015] by removing the assumption that there is a large interval such that the sizes of clusters do not fall in, and also remove the dependency of the size of the recoverable clusters on the number of underlying clusters. We further complement our theoretical improvements with experimental comparisons.Under the planted clique conjecture, the size of the clusters that can be recovered by our algorithm is nearly optimal (up to poly-logarithmic factors) when the probability parameters are constant. As a byproduct, we obtain an efficient clustering algorithm with sublinear query complexity in a faulty oracle model, which is capable of detecting all clusters larger than $\tilde{\Omega}({\sqrt{n}})$, even in the presence of $\Omega(n)$ small clusters in the graph. In contrast, previous efficient algorithms that use a sublinear number of queries are incapable of recovering any large clusters if there are more than $\tilde{\Omega}(n^{2/5})$ small clusters.

----

## [902] Transition-constant Normalization for Image Enhancement

**Authors**: *Jie Huang, Man Zhou, Jinghao Zhang, Gang Yang, Mingde Yao, Chongyi Li, Zhiwei Xiong, Feng Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4163873c9ad623a87989d0a6eefe9442-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4163873c9ad623a87989d0a6eefe9442-Abstract-Conference.html)

**Abstract**:

Normalization techniques that capture image style by statistical representation have become a popular component in deep neural networks.Although image enhancement can be considered as a form of style transformation, there has been little exploration of how normalization affect the enhancement performance. To fully leverage the potential of normalization, we present a novel Transition-Constant Normalization (TCN) for various image enhancement tasks.Specifically, it consists of two streams of normalization operations arranged under an invertible constraint, along with a feature sub-sampling operation that satisfies the normalization constraint.TCN enjoys several merits, including being parameter-free, plug-and-play, and incurring no additional computational costs.We provide various formats to utilize TCN for image enhancement, including seamless  integration with enhancement networks, incorporation into encoder-decoder architectures for downsampling, and implementation of efficient architectures.Through extensive experiments on multiple image enhancement tasks, like low-light enhancement, exposure correction, SDR2HDR translation, and image dehazing, our TCN consistently demonstrates performance improvements.Besides, it showcases extensive ability in other tasks including pan-sharpening and medical segmentation.The code is available at  \textit{\textcolor{blue}{https://github.com/huangkevinj/TCNorm}}.

----

## [903] Unexpected Improvements to Expected Improvement for Bayesian Optimization

**Authors**: *Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, Eytan Bakshy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/419f72cbd568ad62183f8132a3605a2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/419f72cbd568ad62183f8132a3605a2a-Abstract-Conference.html)

**Abstract**:

Expected Improvement (EI) is arguably the most popular acquisition function in Bayesian optimization and has found countless successful applications, but its performance is often exceeded by that of more recent methods. Notably, EI and its variants, including for the parallel and multi-objective settings, are challenging to optimize because their acquisition values vanish numerically in many regions. This difficulty generally increases as the number of observations, dimensionality of the search space, or the number of constraints grow, resulting in performance that is inconsistent across the literature and most often sub-optimal. Herein, we propose LogEI, a new family of acquisition functions whose members either have identical or approximately equal optima as their canonical counterparts, but are substantially easier to optimize numerically. We demonstrate that numerical pathologies manifest themselves in “classic” analytic EI, Expected Hypervolume Improvement (EHVI), as well as their constrained, noisy, and parallel variants, and propose corresponding reformulations that remedy these pathologies. Our empirical results show that members of the LogEI family of acquisition functions substantially improve on the optimization performance of their canonical counterparts and surprisingly, are on par with or exceed the performance of recent state-of-the-art acquisition functions, highlighting the understated role of numerical optimization in the literature.

----

## [904] Pseudo-Likelihood Inference

**Authors**: *Theo Gruner, Boris Belousov, Fabio Muratore, Daniel Palenicek, Jan R. Peters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41aa1c9f57ea83d7c41f0d3e98ed3dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41aa1c9f57ea83d7c41f0d3e98ed3dd4-Abstract-Conference.html)

**Abstract**:

Simulation-Based Inference (SBI) is a common name for an emerging family of approaches that infer the model parameters when the likelihood is intractable. Existing SBI methods either approximate the likelihood, such as Approximate Bayesian Computation (ABC) or directly model the posterior, such as Sequential Neural Posterior Estimation (SNPE). While ABC is efficient on low-dimensional problems, on higher-dimensional tasks, it is generally outperformed by SNPE, which leverages function approximation. In this paper, we propose Pseudo-Likelihood Inference (PLI), a new method that brings neural approximation into ABC, making it competitive on challenging Bayesian system identification tasks. By utilizing integral probability metrics, we introduce a smooth likelihood kernel with an adaptive bandwidth that is updated based on information-theoretic trust regions. Thanks to this formulation, our method (i) allows for optimizing neural posteriors via gradient descent, (ii) does not rely on summary statistics, and (iii) enables multiple observations as input. In comparison to SNPE, it leads to improved performance when more data is available. The effectiveness of PLI is evaluated on four classical SBI benchmark tasks and on a highly dynamic physical system, showing particular advantages on stochastic simulations and multi-modal posterior landscapes.

----

## [905] Calibrating "Cheap Signals" in Peer Review without a Prior

**Authors**: *Yuxuan Lu, Yuqing Kong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41badd36e935f8a80175e95d8bc6192e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41badd36e935f8a80175e95d8bc6192e-Abstract-Conference.html)

**Abstract**:

Peer review lies at the core of the academic process, but even well-intentioned reviewers can still provide noisy ratings. While ranking papers by average ratings may reduce noise, varying noise levels and systematic biases stemming from ``cheap'' signals (e.g. author identity, proof length) can lead to unfairness. Detecting and correcting bias is challenging, as ratings are subjective and unverifiable. Unlike previous works relying on prior knowledge or historical data, we propose a one-shot noise calibration process without any prior information. We ask reviewers to predict others' scores and use these predictions for calibration. Assuming reviewers adjust their predictions according to the noise, we demonstrate that the calibrated score results in a more robust ranking compared to average ratings, even with varying noise levels and biases.In detail, we show that the error probability of the calibrated score approaches zero as the number of reviewers increases and is significantly lower compared to average ratings when the number of reviewers is small.

----

## [906] SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds

**Authors**: *Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys, Yun Fu, Yanzhi Wang, Sergey Tulyakov, Jian Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41bcc9d3bddd9c90e1f44b29e26d97ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41bcc9d3bddd9c90e1f44b29e26d97ff-Abstract-Conference.html)

**Abstract**:

Text-to-image diffusion models can create stunning images from natural language descriptions that rival the work of professional artists and photographers. However, these models are large, with complex network architectures and tens of denoising iterations, making them computationally expensive and slow to run. As a result, high-end GPUs and cloud-based inference are required to run diffusion models at scale. This is costly and has privacy implications, especially when user data is sent to a third party. To overcome these challenges, we present a generic approach that, for the first time, unlocks running text-to-image diffusion models on mobile devices in **less than 2 seconds**.  We achieve so by introducing efficient network architecture and improving step distillation. Specifically, we propose an efficient UNet by identifying the redundancy of the original model and reducing the computation of the image decoder via data distillation. Further, we enhance the step distillation by exploring training strategies and introducing regularization from classifier-free guidance. Our extensive experiments on MS-COCO show that our model with $8$ denoising steps achieves better FID and CLIP scores than Stable Diffusion v$1.5$ with $50$ steps. Our work democratizes content creation by bringing powerful text-to-image diffusion models to the hands of users.

----

## [907] LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference

**Authors**: *Hongwu Peng, Ran Ran, Yukui Luo, Jiahui Zhao, Shaoyi Huang, Kiran Thorat, Tong Geng, Chenghong Wang, Xiaolin Xu, Wujie Wen, Caiwen Ding*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41bd71e7bf7f9fe68f1c936940fd06bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41bd71e7bf7f9fe68f1c936940fd06bd-Abstract-Conference.html)

**Abstract**:

The growth of Graph Convolution Network (GCN) model sizes has revolutionized numerous applications, surpassing human performance in areas such as personal healthcare and financial systems. The  deployment of GCNs in the cloud raises privacy concerns due to potential adversarial attacks on client data. To address security concerns, Privacy-Preserving Machine Learning (PPML) using Homomorphic Encryption (HE) secures sensitive client data. However, it introduces substantial computational overhead in practical applications. To tackle those challenges, we present LinGCN, a framework designed to reduce multiplication depth and optimize the performance of HE based GCN inference. LinGCN is structured around three key elements: (1) A differentiable structural linearization algorithm, complemented by a parameterized discrete indicator function, co-trained with model weights to meet the optimization goal. This strategy promotes fine-grained node-level non-linear location selection, resulting in a model with minimized multiplication depth. (2) A compact node-wise polynomial replacement policy with a second-order trainable activation function, steered towards superior convergence by a two-level distillation approach from an all-ReLU based teacher model. (3) an enhanced HE solution that enables finer-grained operator fusion for node-wise activation functions, further reducing multiplication level consumption in HE-based inference. Our experiments on the NTU-XVIEW skeleton joint dataset reveal that LinGCN excels in latency, accuracy, and scalability for homomorphically encrypted inference, outperforming solutions such as CryptoGCN. Remarkably, LinGCN achieves a 14.2Ã— latency speedup relative to CryptoGCN, while preserving an inference accuracy of ~75\% and notably reducing multiplication depth. Additionally, LinGCN proves scalable for larger models, delivering a substantial 85.78\% accuracy with 6371s latency, a 10.47\% accuracy improvement over CryptoGCN.

----

## [908] Spectral Evolution and Invariance in Linear-width Neural Networks

**Authors**: *Zhichao Wang, Andrew Engel, Anand D. Sarwate, Ioana Dumitriu, Tony Chiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41ed4bd197d0a5fa036d361c1fc606ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41ed4bd197d0a5fa036d361c1fc606ad-Abstract-Conference.html)

**Abstract**:

We investigate the spectral properties of linear-width feed-forward neural networks, where the sample size is asymptotically proportional to network width. Empirically, we show that the spectra of weight in this high dimensional regime are invariant when trained by gradient descent for small constant learning rates; we provide a theoretical justification for this observation and prove the invariance of the bulk spectra for both conjugate and neural tangent kernels. We demonstrate similar characteristics when training with stochastic gradient descent with small learning rates. When the learning rate is large, we exhibit the emergence of an outlier whose corresponding eigenvector is aligned with the training data structure. We also show that after adaptive gradient training, where a lower test error and feature learning emerge, both weight and kernel matrices exhibit heavy tail behavior. Simple examples are provided to explain when heavy tails can have better generalizations. We exhibit different spectral properties such as invariant bulk, spike, and heavy-tailed distribution from a two-layer neural network using different training strategies, and then correlate them to the feature learning. Analogous phenomena also appear when we train conventional neural networks with real-world data. We conclude that monitoring the evolution of the spectra during training is an essential step toward understanding the training dynamics and feature learning.

----

## [909] Paxion: Patching Action Knowledge in Video-Language Foundation Models

**Authors**: *Zhenhailong Wang, Ansel Blume, Sha Li, Genglin Liu, Jaemin Cho, Zineng Tang, Mohit Bansal, Heng Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/420492060687ca7448398c4c3fa10366-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/420492060687ca7448398c4c3fa10366-Abstract-Conference.html)

**Abstract**:

Action knowledge involves the understanding of textual, visual, and temporal aspects of actions. We introduce the Action Dynamics Benchmark (ActionBench) containing two carefully designed probing tasks: Action Antonym and Video Reversal, which targets multimodal alignment capabilities and temporal understanding skills of the model, respectively. Despite recent video-language models’ (VidLM) impressive performance on various benchmark tasks, our diagnostic tasks reveal their surprising deficiency (near-random performance) in action knowledge, suggesting that current models rely on object recognition abilities as a shortcut for action understanding. To remedy this, we propose a novel framework, Paxion, along with a new Discriminative Video Dynamics Modeling (DVDM) objective. The Paxion framework utilizes a Knowledge Patcher network to encode new action knowledge and a Knowledge Fuser component to integrate the Patcher into frozen VidLMs without compromising their existing capabilities. Due to limitations of the widely-used Video-Text Contrastive (VTC) loss for learning action knowledge, we introduce the DVDM objective to train the Knowledge Patcher. DVDM forces the model to encode the correlation between the action text and the correct ordering of video frames. Our extensive analyses show that Paxion and DVDM together effectively fill the gap in action knowledge understanding (~50% → 80%), while maintaining or improving performance on a wide spectrum of both object- and action-centric downstream tasks.

----

## [910] ProPILE: Probing Privacy Leakage in Large Language Models

**Authors**: *Siwon Kim, Sangdoo Yun, Hwaran Lee, Martin Gubri, Sungroh Yoon, Seong Joon Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/420678bb4c8251ab30e765bc27c3b047-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/420678bb4c8251ab30e765bc27c3b047-Abstract-Conference.html)

**Abstract**:

The rapid advancement and widespread use of large language models (LLMs) have raised significant concerns regarding the potential leakage of personally identifiable information (PII). These models are often trained on vast quantities of web-collected data, which may inadvertently include sensitive personal data. This paper presents ProPILE, a novel probing tool designed to empower data subjects, or the owners of the PII, with awareness of potential PII leakage in LLM-based services. ProPILE lets data subjects formulate prompts based on their own PII to evaluate the level of privacy intrusion in LLMs. We demonstrate its application on the OPT-1.3B model trained on the publicly available Pile dataset. We show how hypothetical data subjects may assess the likelihood of their PII being included in the Pile dataset being revealed. ProPILE can also be leveraged by LLM service providers to effectively evaluate their own levels of PII leakage with more powerful prompts specifically tuned for their in-house models. This tool represents a pioneering step towards empowering the data subjects for their awareness and control over their own data on the web.

----

## [911] Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension

**Authors**: *Moritz Haas, David Holzmüller, Ulrike von Luxburg, Ingo Steinwart*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/421f83663c02cdaec8c3c38337709989-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/421f83663c02cdaec8c3c38337709989-Abstract-Conference.html)

**Abstract**:

The success of over-parameterized neural networks trained to near-zero training error has caused great interest in the phenomenon of benign overfitting, where estimators are statistically consistent even though they interpolate noisy training data. While benign overfitting in fixed dimension has been established for some learning methods, current literature suggests that for regression with typical kernel methods and wide neural networks, benign overfitting requires a high-dimensional setting, where the dimension grows with the sample size. In this paper, we show that the smoothness of the estimators, and not the dimension, is the key: benign overfitting is possible if and only if the estimator's derivatives are large enough. We generalize existing inconsistency results to non-interpolating models and more kernels to show that benign overfitting with moderate derivatives is impossible in fixed dimension. Conversely, we show that benign overfitting is possible for regression with a sequence of spiky-smooth kernels with large derivatives. Using neural tangent kernels, we translate our results to wide neural networks. We prove that while infinite-width networks do not overfit benignly with the ReLU activation, this can be fixed by adding small high-frequency fluctuations to the activation function. Our experiments verify that such neural networks, while overfitting, can indeed generalize well even on low-dimensional data sets.

----

## [912] The Goldilocks of Pragmatic Understanding: Fine-Tuning Strategy Matters for Implicature Resolution by LLMs

**Authors**: *Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, Edward Grefenstette*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4241fec6e94221526b0a9b24828bb774-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4241fec6e94221526b0a9b24828bb774-Abstract-Conference.html)

**Abstract**:

Despite widespread use of LLMs as conversational agents, evaluations of performance fail to capture a crucial aspect of communication: interpreting language in context---incorporating its pragmatics. Humans interpret language using beliefs and prior knowledge about the world. For example, we intuitively understand the response "I wore gloves" to the question "Did you leave fingerprints?" as meaning "No". To investigate whether LLMs have the ability to make this type of inference, known as an implicature, we design a simple task and evaluate four categories of widely used state-of-the-art models. We find that, despite only evaluating on utterances that require a binary inference (yes or no), models in three of these categories perform close to random. However, LLMs instruction-tuned at the example-level perform significantly better. These results suggest that certain fine-tuning strategies are far better at inducing pragmatic understanding in models. We present our findings as the starting point for further research into evaluating how LLMs interpret language in context and to drive the development of more pragmatic and useful models of human discourse.

----

## [913] Slow and Weak Attractor Computation Embedded in Fast and Strong E-I Balanced Neural Dynamics

**Authors**: *Xiaohan Lin, Liyuan Li, Boxin Shi, Tiejun Huang, Yuanyuan Mi, Si Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/425ee25d6c22ef98b67328273b8f95d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/425ee25d6c22ef98b67328273b8f95d5-Abstract-Conference.html)

**Abstract**:

Attractor networks require neuronal connections to be highly structured in order to maintain attractor states that represent information, while excitation and inhibition balanced networks (E-INNs) require neuronal connections to be random and sparse to generate irregular neuronal firings. Despite being regarded as canonical models of neural circuits, both types of networks are usually studied in isolation, and it remains unclear how they coexist in the brain, given their very different structural demands. In this study, we investigate the compatibility of continuous attractor neural networks (CANNs) and E-INNs. In line with recent experimental data, we find that a neural circuit can exhibit both the traits of CANNs and E-INNs if the neuronal synapses consist of two sets: one set is strong and fast for irregular firing, and the other set is weak and slow for attractor dynamics. Our results from simulations and theoretical analysis reveal that the network also exhibits enhanced performance compared to the case of using only one set of synapses, with accelerated convergence of attractor states and retained E-I balanced condition for localized input. We also apply the network model to solve a real-world tracking problem and demonstrate that it can track fast-moving objects well. We hope that this study provides insight into how structured neural computations are realized by irregular firings of neurons.

----

## [914] Test-time Training for Matching-based Video Object Segmentation

**Authors**: *Juliette Bertrand, Giorgos Kordopatis-Zilos, Yannis Kalantidis, Giorgos Tolias*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4267d84ca2f6fbb4aa5172b76b433aca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4267d84ca2f6fbb4aa5172b76b433aca-Abstract-Conference.html)

**Abstract**:

The video object segmentation (VOS) task involves the segmentation of an object over time based on a single initial mask. Current state-of-the-art approaches use a memory of previously processed frames and rely on matching to estimate segmentation masks of subsequent frames. Lacking any adaptation mechanism, such methods are prone to test-time distribution shifts. This work focuses on matching-based VOS under distribution shifts such as video corruptions, stylization, and sim-to-real transfer. We explore test-time training strategies that are agnostic to the specific task as well as strategies that are designed specifically for VOS. This includes a variant based on mask cycle consistency tailored to matching-based VOS methods. The experimental results on common benchmarks demonstrate that the proposed test-time training yields significant improvements in performance. In particular for the sim-to-real scenario and despite using only a single test video, our approach manages to recover a substantial portion of the performance gain achieved through training on real videos. Additionally, we introduce DAVIS-C, an augmented version of the popular DAVIS test set, featuring extreme distribution shifts like image-/video-level corruptions and stylizations. Our results illustrate that test-time training enhances performance even in these challenging cases.

----

## [915] Causal Effect Regularization: Automated Detection and Removal of Spurious Correlations

**Authors**: *Abhinav Kumar, Amit Deshpande, Amit Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42770daf4a3384b712ea9c36e9279998-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42770daf4a3384b712ea9c36e9279998-Abstract-Conference.html)

**Abstract**:

In many classification datasets, the task labels are spuriously correlated with some input attributes. Classifiers trained on such datasets often rely on these attributes for prediction, especially when the spurious correlation is high, and thus fail togeneralize whenever there is a shift in the attributes’ correlation at deployment. If we assume that the spurious attributes are known a priori, several methods have been proposed to learn a classifier that is invariant to the specified attributes. However, in real-world data, information about spurious attributes is typically unavailable. Therefore, we propose a method that automatically identifies spurious attributes by estimating their causal effect on the label and then uses a regularization objective to mitigate the classifier’s reliance on them. Although causal effect of an attribute on the label is not always identified, we present two commonly occurring data-generating processes where the effect can be identified. Compared to recent work for identifying spurious attributes, we find that our method, AutoACER, ismore accurate in removing the attribute from the learned model, especially when spurious correlation is high. Specifically, across synthetic, semi-synthetic, and real-world datasets, AutoACER shows significant improvement in a metric used to quantify the dependence of a classifier on spurious attributes ($\Delta$Prob), while obtaining better or similar accuracy. Empirically we find that AutoACER mitigatesthe reliance on spurious attributes even under noisy estimation of causal effects or when the causal effect is not identified. To explain the empirical robustness of our method, we create a simple linear classification task with two sets of attributes: causal and spurious. Under this setting, we prove that AutoACER only requires the ranking of estimated causal effects to be correct across attributes to select thecorrect classifier.

----

## [916] Multi-resolution Spectral Coherence for Graph Generation with Score-based Diffusion

**Authors**: *Hyuna Cho, Minjae Jeong, Sooyeon Jeon, Sungsoo Ahn, Won Hwa Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/427f20d90386fd27804f1831d6a3d48f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/427f20d90386fd27804f1831d6a3d48f-Abstract-Conference.html)

**Abstract**:

Successful graph generation depends on the accurate estimation of the joint distribution of graph components such as nodes and edges from training data. While recent deep neural networks have demonstrated sampling of realistic graphs together with diffusion models, however, they still suffer from oversmoothing problems which are inherited from conventional graph convolution and thus high-frequency characteristics of nodes and edges become intractable. To overcome such issues and generate graphs with high fidelity, this paper introduces a novel approach that captures the dependency between nodes and edges at multiple resolutions in the spectral space. By modeling the joint distribution of node and edge signals in a shared graph wavelet space, together with a score-based diffusion model, we propose a Wavelet Graph Diffusion Model (Wave-GD) which lets us sample synthetic graphs with real-like frequency characteristics of nodes and edges. Experimental results on four representative benchmark datasets validate the superiority of the Wave-GD over existing approaches, highlighting its potential for a wide range of applications that involve graph data.

----

## [917] Real-World Image Super-Resolution as Multi-Task Learning

**Authors**: *Wenlong Zhang, Xiaohui Li, Guangyuan Shi, Xiangyu Chen, Yu Qiao, Xiaoyun Zhang, Xiao-Ming Wu, Chao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42806406dd99e30c3796bc98b2670fa2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42806406dd99e30c3796bc98b2670fa2-Abstract-Conference.html)

**Abstract**:

In this paper, we take a new look at real-world image super-resolution (real-SR) from a multi-task learning perspective. We demonstrate that the conventional formulation of real-SR can be viewed as solving multiple distinct degradation tasks using a single shared model. This poses a challenge known as task competition or task conflict in multi-task learning, where certain tasks dominate the learning process, resulting in poor performance on other tasks. This problem is exacerbated in the case of real-SR, due to the involvement of numerous degradation tasks. To address the issue of task competition in real-SR, we propose a task grouping approach. Our approach efficiently identifies the degradation tasks where a real-SR model falls short and groups these unsatisfactory tasks into multiple task groups. We then utilize the task groups to fine-tune the real-SR model in a simple way, which effectively mitigates task competition and facilitates knowledge transfer. Extensive experiments demonstrate our method achieves significantly enhanced performance across a wide range of degradation scenarios.

----

## [918] Exact Representation of Sparse Networks with Symmetric Nonnegative Embeddings

**Authors**: *Sudhanshu Chanpuriya, Ryan A. Rossi, Anup B. Rao, Tung Mai, Nedim Lipka, Zhao Song, Cameron Musco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/428ceef2cd8a53add7213e04d1746479-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/428ceef2cd8a53add7213e04d1746479-Abstract-Conference.html)

**Abstract**:

Graph models based on factorization of the adjacency matrix often fail to capture network structures related to links between dissimilar nodes (heterophily). We introduce a novel graph factorization model that leverages two nonnegative vectors per node to interpretably account for links between both similar and dissimilar nodes. We prove that our model can exactly represent any graph with low arboricity, a property that many real-world networks satisfy; our proof also applies to related models but has much greater scope than the closest prior bound, which is based on low max degree. Our factorization also has compelling properties besides expressiveness: due to its symmetric structure and nonnegativity, fitting the model inherently finds node communities, and the model's link predictions can be interpreted in terms of these communities. In experiments on real-world networks, we demonstrate our factorization's effectiveness on a variety of tasks, including community detection and link prediction.

----

## [919] Data-Centric Learning from Unlabeled Graphs with Diffusion Model

**Authors**: *Gang Liu, Eric Inae, Tong Zhao, Jiaxin Xu, Tengfei Luo, Meng Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4290cccf23be59e42a575d026ccbeeb8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4290cccf23be59e42a575d026ccbeeb8-Abstract-Conference.html)

**Abstract**:

Graph property prediction tasks are important and numerous. While each task offers a small size of labeled examples, unlabeled graphs have been collected from various sources and at a large scale. A conventional approach is training a model with the unlabeled graphs on self-supervised tasks and then fine-tuning the model on the prediction tasks. However, the self-supervised task knowledge could not be aligned or sometimes conflicted with what the predictions needed. In this paper, we propose to extract the knowledge underlying the large set of unlabeled graphs as a specific set of useful data points to augment each property prediction model. We use a diffusion model to fully utilize the unlabeled graphs and design two new objectives to guide the model's denoising process with each task's labeled data to generate task-specific graph examples and their labels. Experiments demonstrate that our data-centric approach performs significantly better than fifteen existing various methods on fifteen tasks. The performance improvement brought by unlabeled data is visible as the generated labeled examples unlike the self-supervised learning.

----

## [920] Wasserstein Gradient Flows for Optimizing Gaussian Mixture Policies

**Authors**: *Hanna Ziesche, Leonel Rozo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/429b5216a4d08850c586fbf809e17877-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/429b5216a4d08850c586fbf809e17877-Abstract-Conference.html)

**Abstract**:

Robots often rely on a repertoire of previously-learned motion policies for performing tasks of diverse complexities.  When facing unseen task conditions or when new task requirements arise, robots must adapt their motion policies accordingly. In this context, policy optimization is the \emph{de facto} paradigm to adapt robot policies as a function of task-specific objectives.  Most commonly-used motion policies carry particular structures that are often overlooked in policy optimization algorithms.  We instead propose to leverage the structure of probabilistic policies by casting the policy optimization as an optimal transport problem. Specifically, we focus on robot motion policies that build on Gaussian mixture models (GMMs) and formulate the policy optimization as a Wassertein gradient flow over the GMMs space. This naturally allows us to constrain the policy updates via the $L^2$-Wasserstein distance between GMMs to enhance the stability of the policy optimization process. Furthermore, we leverage the geometry of the Bures-Wasserstein manifold to optimize the Gaussian distributions of the GMM policy via Riemannian optimization. We evaluate our approach on common robotic settings: Reaching motions, collision-avoidance behaviors, and multi-goal tasks. Our results show that our method outperforms common policy optimization baselines in terms of task success rate and low-variance solutions.

----

## [921] Change point detection and inference in multivariate non-parametric models under mixing conditions

**Authors**: *Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, Oscar Hernan Madrid Padilla, Yi Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42a0de6b8a1809ceba8fdad1661be06c-Abstract-Conference.html)

**Abstract**:

This paper addresses the problem of localizing and inferring multiple change points, in non-parametric multivariate time series settings. Specifically, we consider a multivariate time series with potentially short-range dependence, whose underlying distributions have HÃ¶lder smooth densities and can change over time in a piecewise-constant manner. The change points, which correspond to the times when the distribution changes, are unknown. We present the limiting distributions of the change point estimators under the scenarios where the minimal jump size vanishes or remains constant.  Such results have not been revealed in the literature in non-parametric change point settings. As byproducts, we develop a sharp estimator that can accurately localize the change points in multivariate non-parametric time series, and a consistent block-type long-run variance estimator.  Numerical studies are provided to complement our theoretical findings.

----

## [922] Near-optimal learning with average Hölder smoothness

**Authors**: *Guy Kornowski, Steve Hanneke, Aryeh Kontorovich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42afce512806ab874b9f99ed9a08055e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42afce512806ab874b9f99ed9a08055e-Abstract-Conference.html)

**Abstract**:

We generalize the notion of average Lipschitz smoothness proposed by Ashlagi et al. (COLT 2021) by extending it to Hölder smoothness. This measure of the "effective smoothness" of a function is sensitive to the underlying distribution and can be dramatically smaller than its classic "worst-case" Hölder constant.We consider both the realizable and the agnostic (noisy) regression settings, proving upper and lower risk bounds in terms of the average Hölder smoothness; these rates improve upon both previously known rates even in the special case of average Lipschitz smoothness.Moreover, our lower bound is tight in the realizable setting up to log factors, thus we establish the minimax rate.From an algorithmic perspective, since our notion of average smoothness is defined with respect to the unknown underlying distribution, the learner does not have an explicit representation of the function class, hence is unable to execute ERM. Nevertheless, we provide distinct learning algorithms that achieve both (nearly) optimal learning rates.Our results hold in any totally bounded metric space, and are stated in terms of its intrinsic geometry.Overall, our results show that the classic worst-case notion of Hölder smoothness can be essentially replaced by its average, yielding considerably sharper guarantees.

----

## [923] Neural-Logic Human-Object Interaction Detection

**Authors**: *Liulei Li, Jianan Wei, Wenguan Wang, Yi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42b7c2f6d320d1fe1afa899a6319d6d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42b7c2f6d320d1fe1afa899a6319d6d7-Abstract-Conference.html)

**Abstract**:

The interaction decoder utilized in prevalent Transformer-based HOI detectors typically accepts pre-composed human-object pairs as inputs. Though achieving remarkable performance, such a paradigm lacks feasibility and cannot explore novel combinations over entities during decoding. We present LogicHOI, a new HOI detector that leverages neural-logic reasoning and Transformer to infer feasible interactions between. entities. Specifically, we modify. self-attention mechanism in the vanilla Transformer, enabling it to reason over the ⟨ human, action, object ⟩ triplet and constitute novel interactions. Meanwhile, such a reasoning process is guided by two crucial properties for understanding HOI: affordances (the potential actions an object can facilitate) and proxemics (the spatial relations between humans and objects). We formulate these two properties in first-order logic and ground them into continuous space to constrain the learning process of our approach, leading to improved performance and zero-shot generalization capabilities. We evaluate L OGIC HOI on V-COCO and HICO-DET under both normal and zero-shot setups, achieving significant improvements over existing methods.

----

## [924] Which Models have Perceptually-Aligned Gradients? An Explanation via Off-Manifold Robustness

**Authors**: *Suraj Srinivas, Sebastian Bordt, Himabindu Lakkaraju*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42bbe2bfdbbfcadda643e8f89025716c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42bbe2bfdbbfcadda643e8f89025716c-Abstract-Conference.html)

**Abstract**:

One of the remarkable properties of robust computer vision models is that their input-gradients are often aligned with human perception, referred to in the literature as perceptually-aligned gradients (PAGs). Despite only being trained for classification, PAGs cause robust models to have rudimentary generative capabilities, including image generation, denoising, and in-painting. However, the underlying mechanisms behind these phenomena remain unknown. In this work, we provide a first explanation of PAGs via \emph{off-manifold robustness}, which states that models must be more robust off- the data manifold than they are on-manifold. We first demonstrate theoretically that off-manifold robustness leads input gradients to lie approximately on the data manifold, explaining their perceptual alignment. We then show that Bayes optimal models satisfy off-manifold robustness, and confirm the same empirically for robust models trained via gradient norm regularization, randomized smoothing, and adversarial training with projected gradient descent. Quantifying the perceptual alignment of model gradients via their similarity with the gradients of generative models, we show that off-manifold robustness correlates well with perceptual alignment. Finally, based on the levels of on- and off-manifold robustness, we identify three different regimes of robustness that affect both perceptual alignment and model accuracy: weak robustness, bayes-aligned robustness, and excessive robustness. Code is available at https://github.com/tml-tuebingen/pags.

----

## [925] Inferring the Future by Imagining the Past

**Authors**: *Kartik Chandra, Tony Chen, Tzu-Mao Li, Jonathan Ragan-Kelley, Josh Tenenbaum*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42c3438f432bc62014ce65af880e0d94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42c3438f432bc62014ce65af880e0d94-Abstract-Conference.html)

**Abstract**:

A single panel of a comic book can say a lot: it can depict not only where the characters currently are, but also their motions, their motivations, their emotions, and what they might do next. More generally, humans routinely infer complex sequences of past and future events from a static snapshot of a dynamic scene, even in situations they have never seen before.In this paper, we model how humans make such rapid and flexible inferences. Building on a long line of work in cognitive science, we offer a Monte Carlo algorithm whose inferences correlate well with human intuitions in a wide variety of domains, while only using a small, cognitively-plausible number of samples. Our key technical insight is a surprising connection between our inference problem and Monte Carlo path tracing, which allows us to apply decades of ideas from the computer graphics community to this seemingly-unrelated theory of mind task.

----

## [926] The Grand Illusion: The Myth of Software Portability and Implications for ML Progress

**Authors**: *Fraser Mince, Dzung Dinh, Jonas Kgomo, Neil Thompson, Sara Hooker*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42c40aff7814e9796266e12053b1c610-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42c40aff7814e9796266e12053b1c610-Abstract-Conference.html)

**Abstract**:

Pushing the boundaries of machine learning often requires exploring different hardware and software combinations. However, this ability to experiment with different systems can be at odds with the drive for efficiency, which has produced increasingly specialized AI hardware and incentivized consolidation around a narrow set of ML frameworks. Exploratory research can be further restricted if software and hardware are co-evolving, making it even harder to stray away from a given tooling stack. While this friction increasingly impacts the rate of innovation in machine learning, to our knowledge the lack of portability in tooling has not been quantified. In this work we ask: How portable are popular ML software frameworks? We conduct a large scale study of the portability of mainstream ML frameworks across different hardware types. Our findings paint an uncomfortable picture -- frameworks can lose more than 40% of their key functions when ported to other hardware. Worse, even when functions are portable, the slowdown in their performance can be extreme. Collectively, our results reveal how costly straying from a narrow set of hardware-software combinations can be - and thus how specialization incurs an exploration cost that can impede innovation in machine learning research.

----

## [927] Computing Optimal Nash Equilibria in Multiplayer Games

**Authors**: *Youzhi Zhang, Bo An, Venkatramanan Siva Subrahmanian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42cac45fb00f7038c892f1a1bfc216d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42cac45fb00f7038c892f1a1bfc216d3-Abstract-Conference.html)

**Abstract**:

Designing efficient algorithms to compute a Nash Equilibrium (NE) in multiplayer games is still an open challenge. In this paper, we focus on computing an NE that optimizes a given objective function. For example, when there is a team of players independently playing against an adversary in a game (e.g., several groups in a forest trying to interdict illegal loggers in green security games), these team members may need to find an NE minimizing the adversaryâ€™s utility. Finding an optimal NE in multiplayer games can be formulated as a mixed-integer bilinear program by introducing auxiliary variables to represent bilinear terms, leading to a huge number of bilinear terms, making it hard to solve. To overcome this challenge, we first propose a general framework for this formulation based on a set of correlation plans. We then develop a novel algorithm called CRM based on this framework, which uses correlation plans with their relations to strictly reduce the feasible solution space after the convex relaxation of bilinear terms while minimizing the number of correlation plans to significantly reduce the number of bilinear terms. We show that our techniques can significantly reduce the time complexity and CRM can be several orders of magnitude faster than the state-of-the-art baseline.

----

## [928] AND: Adversarial Neural Degradation for Learning Blind Image Super-Resolution

**Authors**: *Fangzhou Luo, Xiaolin Wu, Yanhui Guo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html)

**Abstract**:

Learnt deep neural networks for image super-resolution fail easily if the assumed degradation model in training mismatches that of the real degradation source at the inference stage. Instead of attempting to exhaust all degradation variants in simulation, which is unwieldy and impractical, we propose a novel adversarial neural degradation (AND) model that can, when trained in conjunction with a deep restoration neural network under a minmax criterion, generate a wide range of highly nonlinear complex degradation effects without any explicit supervision. The AND model has a unique advantage over the current state of the art in that it can generalize much better to unseen degradation variants and hence deliver significantly improved restoration performance on real-world images.

----

## [929] Into the LAION's Den: Investigating Hate in Multimodal Datasets

**Authors**: *Abeba Birhane, Vinay Uday Prabhu, Sanghyun Han, Vishnu Boddeti, Sasha Luccioni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/42f225509e8263e2043c9d834ccd9a2b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/42f225509e8263e2043c9d834ccd9a2b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

`Scale the model, scale the data, scale the compute' is the reigning sentiment in the world of generative AI today. While the impact of model scaling has been extensively studied, we are only beginning to scratch the surface of data scaling and its consequences. This is especially of critical importance in the context of vision-language datasets such as LAION. These datasets are continually growing in size and are built based on large-scale internet dumps such as the Common Crawl, which is known to have numerous drawbacks ranging from quality, legality, and content. The datasets then serve as the backbone for large generative models, contributing to the operationalization and perpetuation of harmful societal and historical biases and stereotypes. In this paper, we investigate the effect of scaling datasets on hateful content through a comparative audit of two datasets: LAION-400M and LAION-2B. Our results show that hate content increased by nearly 12% with dataset scale, measured both qualitatively and quantitatively using a metric that we term as Hate Content Rate (HCR). We also found that filtering dataset contents based on Not Safe For Work (NSFW) values calculated based on images alone does not exclude all the harmful content in alt-text. Instead, we found that trace amounts of hateful, targeted, and aggressive text remain even when carrying out conservative filtering. We end with a reflection and a discussion of the significance of our results for dataset curation and usage in the AI community.Code and the meta-data assets curated in this paper are publicly available at https://github.com/vinayprabhu/hate_scaling. Content warning: This paper contains examples of hateful text that might be disturbing, distressing, and/or offensive.

----

## [930] SE(3) Diffusion Model-based Point Cloud Registration for Robust 6D Object Pose Estimation

**Authors**: *Haobo Jiang, Mathieu Salzmann, Zheng Dang, Jin Xie, Jian Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43069caa6776eac8bca4bfd74d4a476d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43069caa6776eac8bca4bfd74d4a476d-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce an SE(3) diffusion model-based point cloud registration framework for 6D object pose estimation in real-world scenarios. Our approach formulates the 3D registration task as a denoising diffusion process, which progressively refines the pose of the source point cloud to obtain a precise alignment with the model point cloud. Training our framework involves two operations: An SE(3) diffusion process and an SE(3) reverse process. The SE(3) diffusion process gradually perturbs the optimal rigid transformation of a pair of point clouds by continuously injecting noise (perturbation transformation). By contrast, the SE(3) reverse process focuses on learning a denoising network that refines the noisy transformation step-by-step, bringing it closer to the optimal transformation for accurate pose estimation. Unlike standard diffusion models used in linear Euclidean spaces, our diffusion model operates on the SE(3) manifold. This requires exploiting the linear Lie algebra $\mathfrak{se}(3)$ associated with SE(3) to constrain the transformation transitions during the diffusion and reverse processes. Additionally, to effectively train our denoising network, we derive a registration-specific variational lower bound as the optimization objective for model learning. Furthermore, we show that our denoising network can be constructed with a surrogate registration model, making our approach applicable to different deep registration networks. Extensive experiments demonstrate that our diffusion registration framework presents outstanding pose estimation performance on the real-world TUD-L, LINEMOD, and Occluded-LINEMOD datasets.

----

## [931] RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions

**Authors**: *Lingdong Kong, Shaoyuan Xie, Hanjiang Hu, Lai Xing Ng, Benoit Cottereau, Wei Tsang Ooi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Depth estimation from monocular images is pivotal for real-world visual perception systems. While current learning-based depth estimation models train and test on meticulously curated data, they often overlook out-of-distribution (OoD) situations. Yet, in practical settings -- especially safety-critical ones like autonomous driving -- common corruptions can arise. Addressing this oversight, we introduce a comprehensive robustness test suite, RoboDepth, encompassing 18 corruptions spanning three categories: i) weather and lighting conditions; ii) sensor failures and movement; and iii) data processing anomalies. We subsequently benchmark 42 depth estimation models across indoor and outdoor scenes to assess their resilience to these corruptions. Our findings underscore that, in the absence of a dedicated robustness evaluation framework, many leading depth estimation models may be susceptible to typical corruptions. We delve into design considerations for crafting more robust depth estimation models, touching upon pre-training, augmentation, modality, model capacity, and learning paradigms. We anticipate our benchmark will establish a foundational platform for advancing robust OoD depth estimation.

----

## [932] Fed-CO2: Cooperation of Online and Offline Models for Severe Data Heterogeneity in Federated Learning

**Authors**: *Zhongyi Cai, Ye Shi, Wei Huang, Jingya Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/431d53d513461ff155d5bc8faa9a440c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/431d53d513461ff155d5bc8faa9a440c-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) has emerged as a promising distributed learning paradigm that enables multiple clients to learn a global model collaboratively without sharing their private data. However, the effectiveness of FL is highly dependent on the quality of the data that is being used for training. In particular, data heterogeneity issues, such as label distribution skew and feature skew, can significantly impact the performance of FL. Previous studies in FL have primarily focused on addressing label distribution skew data heterogeneity, while only a few recent works have made initial progress in tackling feature skew issues. Notably, these two forms of data heterogeneity have been studied separately and have not been well explored within a unified FL framework. To address this gap, we propose Fed-CO$_2$, a universal FL framework that handles both label distribution skew and feature skew within a Cooperation mechanism between the Online and Offline models. Specifically, the online model learns general knowledge that is shared among all clients, while the offline model is trained locally to learn the specialized knowledge of each individual client. To further enhance model cooperation in the presence of feature shifts, we design an intra-client knowledge transfer mechanism that reinforces mutual learning between the online and offline models, and an inter-client knowledge transfer mechanism to increase the modelsâ€™ domain generalization ability. Extensive experiments show that our Fed-CO$_2$ outperforms a wide range of existing personalized federated learning algorithms in terms of handling label distribution skew and feature skew, both individually and collectively. The empirical results are supported by our convergence analyses in a simplified setting.

----

## [933] Combating Bilateral Edge Noise for Robust Link Prediction

**Authors**: *Zhanke Zhou, Jiangchao Yao, Jiaxu Liu, Xiawei Guo, Quanming Yao, LI He, Liang Wang, Bo Zheng, Bo Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/435986a8cc3e0667648df5d1c2d55c83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/435986a8cc3e0667648df5d1c2d55c83-Abstract-Conference.html)

**Abstract**:

Although link prediction on graphs has achieved great success with the development of graph neural networks (GNNs), the potential robustness under the edge noise is still less investigated. To close this gap, we first conduct an empirical study to disclose that the edge noise bilaterally perturbs both input topology and target label, yielding severe performance degradation and representation collapse. To address this dilemma, we propose an information-theory-guided principle, Robust Graph Information Bottleneck (RGIB), to extract reliable supervision signals and avoid representation collapse. Different from the basic information bottleneck, RGIB further decouples and balances the mutual dependence among graph topology, target labels, and representation, building new learning objectives for robust representation against the bilateral noise. Two instantiations, RGIB-SSL and RGIB-REP, are explored to leverage the merits of different methodologies, i.e., self-supervised learning and data reparameterization, for implicit and explicit data denoising, respectively. Extensive experiments on six datasets and three GNNs with diverse noisy scenarios verify the effectiveness of our RGIB instantiations. The code is publicly available at: https://github.com/tmlr-group/RGIB.

----

## [934] SyncTREE: Fast Timing Analysis for Integrated Circuit Design through a Physics-informed Tree-based Graph Neural Network

**Authors**: *Yuting Hu, Jiajie Li, Florian Klemme, Gi-Joon Nam, Tengfei Ma, Hussam Amrouch, Jinjun Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/435e8fbbfc2c6072d4f3a5cb6e56a39a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/435e8fbbfc2c6072d4f3a5cb6e56a39a-Abstract-Conference.html)

**Abstract**:

Nowadays integrated circuits (ICs) are underpinning all major information technology innovations including the current trends of artificial intelligence (AI). Modern IC designs often involve analyses of complex phenomena (such as timing, noise, and power etc.) for tens of billions of electronic components, like resistance (R), capacitance (C), transistors and gates, interconnected in various complex structures. Those analyses often need to strike a balance between accuracy and speed as those analyses need to be carried out many times throughout the entire IC design cycles. With the advancement of AI, researchers also start to explore news ways in leveraging AI to improve those analyses. This paper focuses on one of the most important analyses, timing analysis for interconnects. Since IC interconnects can be represented as an RC-tree, a specialized graph as tree, we design a novel tree-based graph neural network, SyncTREE, to speed up the timing analysis by incorporating both the structural and physical properties of electronic circuits. Our major innovations include (1) a two-pass message-passing (bottom-up and top-down) for graph embedding, (2) a tree contrastive loss to guide learning, and (3) a closed formular-based approach to conduct fast timing.  Our experiments show that, compared to conventional GNN models, SyncTREE achieves the best timing prediction in terms of both delays and slews, all in reference to the industry golden numerical analyses results on real IC design data.

----

## [935] Hierarchical Open-vocabulary Universal Image Segmentation

**Authors**: *Xudong Wang, Shufan Li, Konstantinos Kallidromitis, Yusuke Kato, Kazuki Kozuka, Trevor Darrell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43663f64775ae439ec52b64305d219d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43663f64775ae439ec52b64305d219d3-Abstract-Conference.html)

**Abstract**:

Open-vocabulary image segmentation aims to partition an image into semantic regions according to arbitrary text descriptions. However, complex visual scenes can be naturally decomposed into simpler parts and abstracted at multiple lev4 els of granularity, introducing inherent segmentation ambiguity. Unlike existing methods that typically sidestep this ambiguity and treat it as an external factor, our approach actively incorporates a hierarchical representation encompassing different semantic-levels into the learning process. We propose a decoupled text-image fusion mechanism and representation learning modules for both “things” and “stuff”. Additionally, we systematically examine the differences that exist in the textual and visual features between these types of categories. Our resulting model, named HIPIE, tackles HIerarchical, oPen-vocabulary, and unIvErsal segmentation tasks within a unified framework. Benchmarked on diverse datasets, e.g., ADE20K,COCO, Pascal-VOC Part, and RefCOCO/RefCOCOg, HIPIE achieves the state-of14 the-art results at various levels of image comprehension, including semantic-level (e.g., semantic segmentation), instance-level (e.g., panoptic/referring segmentationand object detection), as well as part-level (e.g., part/subpart segmentation) tasks.

----

## [936] Fairly Recommending with Social Attributes: A Flexible and Controllable Optimization Approach

**Authors**: *Jinqiu Jin, Haoxuan Li, Fuli Feng, Sihao Ding, Peng Wu, Xiangnan He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/436d042b2dd81214d23ae43eb196b146-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/436d042b2dd81214d23ae43eb196b146-Abstract-Conference.html)

**Abstract**:

Item-side group fairness (IGF) requires a recommendation model to treat different item groups similarly, and has a crucial impact on information diffusion, consumption activity, and market equilibrium. Previous IGF notions only focus on the direct utility of the item exposures, i.e., the exposure numbers across different item groups. Nevertheless, the item exposures also facilitate utility gained from the neighboring users via social influence, called social utility, such as information sharing on the social media. To fill this gap, this paper introduces two social attribute-aware IGF metrics, which require similar user social attributes on the exposed items across the different item groups. In light of the trade-off between the direct utility and social utility, we formulate a new multi-objective optimization problem for training recommender models with flexible trade-off while ensuring controllable accuracy. To solve this problem, we develop a gradient-based optimization algorithm and theoretically show that the proposed algorithm can find Pareto optimal solutions with varying trade-off and guaranteed accuracy. Extensive experiments on two real-world datasets validate the effectiveness of our approach.

----

## [937] Look Ma, No Hands! Agent-Environment Factorization of Egocentric Videos

**Authors**: *Matthew Chang, Aditya Prakash, Saurabh Gupta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/437cd2749391ad40f67e4dd1d87c4596-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/437cd2749391ad40f67e4dd1d87c4596-Abstract-Conference.html)

**Abstract**:

The analysis and use of egocentric videos for robotics tasks is made challenging by occlusion and the visual mismatch between the human hand and a robot end-effector. Past work views the human hand as a nuisance and removes it from the scene. However, the hand also provides a valuable signal for learning. In this work, we propose to extract a factored representation of the scene that separates the agent (human hand) and the environment. This alleviates both occlusion and mismatch while preserving the signal, thereby easing the design of models for downstream robotics tasks. At the heart of this factorization is our proposed Video Inpainting via Diffusion Model (VIDM) that leverages both a prior on real-world images (through a large-scale pre-trained diffusion model) and the appearance of the object in earlier frames of the video (through attention). Our experiments demonstrate the effectiveness of VIDM at improving the in-painting quality in egocentric videos and the power of our factored representation for numerous tasks: object detection, 3D reconstruction of manipulated objects, and learning of reward functions, policies, and affordances from videos.

----

## [938] Generating Images with Multimodal Language Models

**Authors**: *Jing Yu Koh, Daniel Fried, Russ Salakhutdinov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43a69d143273bd8215578bde887bb552-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43a69d143273bd8215578bde887bb552-Abstract-Conference.html)

**Abstract**:

We propose a method to fuse frozen text-only large language models (LLMs) with pre-trained image encoder and decoder models, by mapping between their embedding spaces. Our model demonstrates a wide suite of multimodal capabilities: image retrieval, novel image generation, and multimodal dialogue. Ours is the first approach capable of conditioning on arbitrarily interleaved image and text inputs to generate coherent image (and text) outputs. To achieve strong performance on image generation, we propose an efficient mapping network to ground the LLM to an off-the-shelf text-to-image generation model. This mapping network translates hidden representations of text into the embedding space of the visual models, enabling us to leverage the strong text representations of the LLM for visual outputs. Our approach outperforms baseline generation models on tasks with longer and more complex language. In addition to novel image generation, our model is also capable of image retrieval from a prespecified dataset, and decides whether to retrieve or generate at inference time. This is done with a learnt decision module which conditions on the hidden representations of the LLM. Our model exhibits a wider range of capabilities compared to prior multimodal language models. It can process image-and-text inputs, and produce retrieved images, generated images, and generated text â€” outperforming non-LLM based generation models across several text-to-image tasks that measure context dependence.

----

## [939] MoVie: Visual Model-Based Policy Adaptation for View Generalization

**Authors**: *Sizhe Yang, Yanjie Ze, Huazhe Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43b77cef2a83a25aa27d3271d209e4fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43b77cef2a83a25aa27d3271d209e4fd-Abstract-Conference.html)

**Abstract**:

Visual Reinforcement Learning (RL) agents trained on limited views face significant challenges in generalizing their learned abilities to unseen views. This inherent difficulty is known as the problem of $\textit{view generalization}$. In this work, we systematically categorize this fundamental problem into four distinct and highly challenging scenarios that closely resemble real-world situations. Subsequently, we propose a straightforward yet effective approach to enable successful adaptation of visual $\textbf{Mo}$del-based policies for $\textbf{Vie}$w generalization ($\textbf{MoVie}$) during test time, without any need for explicit reward signals and any modification during training time. Our method demonstrates substantial advancements across all four scenarios encompassing a total of $\textbf{18}$ tasks sourced from DMControl, xArm, and Adroit, with a relative improvement of $\mathbf{33}$%, $\mathbf{86}$%, and $\mathbf{152}$% respectively. The superior results highlight the immense potential of our approach for real-world robotics applications. Code and videos are available at https://yangsizhe.github.io/MoVie/.

----

## [940] Does Visual Pretraining Help End-to-End Reasoning?

**Authors**: *Chen Sun, Calvin Luo, Xingyi Zhou, Anurag Arnab, Cordelia Schmid*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43ba0466af2b1ac76aa85d8fbec714e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43ba0466af2b1ac76aa85d8fbec714e3-Abstract-Conference.html)

**Abstract**:

We aim to investigate whether end-to-end learning of visual reasoning can be achieved with general-purpose neural networks, with the help of visual pretraining. A positive result would refute the common belief that explicit visual abstraction (e.g. object detection) is essential for compositional generalization on visual reasoning, and confirm the feasibility of a neural network ''generalist'' to solve visual recognition and reasoning tasks. We propose a simple and general self-supervised framework which ''compresses'' each video frame into a small set of tokens with a transformer network, and reconstructs the remaining frames based on the compressed temporal context. To minimize the reconstruction loss, the network must learn a compact representation for each image, as well as capture temporal dynamics and object permanence from temporal context. We perform evaluation on two visual reasoning benchmarks, CATER and ACRE. We observe that pretraining is essential to achieve compositional generalization for end-to-end visual reasoning. Our proposed framework outperforms traditional supervised pretraining, including image classification and explicit object detection, by large margins.

----

## [941] Newton-Cotes Graph Neural Networks: On the Time Evolution of Dynamic Systems

**Authors**: *Lingbing Guo, Weiqing Wang, Zhuo Chen, Ningyu Zhang, Zequn Sun, Yixuan Lai, Qiang Zhang, Huajun Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43e8fd8b9581faa71a6a61602bc28435-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43e8fd8b9581faa71a6a61602bc28435-Abstract-Conference.html)

**Abstract**:

Reasoning system dynamics is one of the most important analytical approaches for many scientific studies. With the initial state of a system as input, the recent graph neural networks (GNNs)-based methods are capable of predicting the future state distant in time with high accuracy. Although these methods have diverse designs in modeling the coordinates and interacting forces of the system, we show that they actually share a common paradigm that learns the integration of the velocity over the interval between the initial and terminal coordinates. However, their integrand is constant w.r.t. time. Inspired by this observation, we propose a new approach to predict the integration based on several velocity estimations with Newton–Cotes formulas and prove its effectiveness theoretically. Extensive experiments on several benchmarks empirically demonstrate consistent and significant improvement compared with the state-of-the-art methods.

----

## [942] Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation

**Authors**: *Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html)

**Abstract**:

Program synthesis has been long studied with recent approaches focused on directly using the power of Large Language Models (LLMs) to generate code. Programming benchmarks, with curated synthesis problems and test-cases, are used to measure the performance of various LLMs on code synthesis. However, these test-cases can be limited in both quantity and quality for fully assessing the functional correctness of the generated code. Such limitation in the existing benchmarks begs the following question: In the era of LLMs, is the code generated really correct? To answer this, we propose EvalPlus â€“ a code synthesis evaluation framework to rigorously benchmark the functional correctness of LLM-synthesized code. EvalPlus augments a given evaluation dataset with large amounts of test-cases newly produced by an automatic test input generator, powered by both LLM- and mutation-based strategies. While EvalPlus is general, we extend the test-cases of the popular HumanEval benchmark by 80x to build HumanEval+. Our extensive evaluation across 26 popular LLMs (e.g., GPT-4 and ChatGPT) demonstrates that HumanEval+ is able to catch significant amounts of previously undetected wrong code synthesized by LLMs, reducing the pass@k by up-to 19.3-28.9%. We also surprisingly found that test insufficiency can lead to mis-ranking. For example, both WizardCoder-CodeLlama and Phind-CodeLlama now outperform ChatGPT on HumanEval+, while none of them could on HumanEval. Our work not only indicates that prior popular code synthesis evaluation results do not accurately reflect the true performance of LLMs for code synthesis, but also opens up a new direction to improve such programming benchmarks through automated testing. We have open-sourced our tools, enhanced datasets as well as all LLM-generated code at https://github.com/evalplus/evalplus to facilitate and accelerate future LLM-for-code research.

----

## [943] LeanDojo: Theorem Proving with Retrieval-Augmented Language Models

**Authors**: *Kaiyu Yang, Aidan M. Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan J. Prenger, Animashree Anandkumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4441469427094f8873d0fecb0c4e1cee-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/4441469427094f8873d0fecb0c4e1cee-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large language models (LLMs) have shown promise in proving formal theorems using proof assistants such as Lean. However, existing methods are difficult to reproduce or build on, due to private code, data, and large compute requirements. This has created substantial barriers to research on machine learning methods for theorem proving. This paper removes these barriers by introducing LeanDojo: an open-source Lean playground consisting of toolkits, data, models, and benchmarks. LeanDojo extracts data from Lean and enables interaction with the proof environment programmatically. It contains fine-grained annotations of premises in proofs, providing valuable data for premise selectionâ€”a key bottleneck in theorem proving. Using this data, we develop ReProver (Retrieval-Augmented Prover): an LLM-based prover augmented with retrieval for selecting premises from a vast math library. It is inexpensive and needs only one GPU week of training. Our retriever leverages LeanDojo's program analysis capability to identify accessible premises and hard negative examples, which makes retrieval much more effective. Furthermore, we construct a new benchmark consisting of 98,734 theorems and proofs extracted from Lean's math library. It features challenging data split requiring the prover to generalize to theorems relying on novel premises that are never used in training. We use this benchmark for training and evaluation, and experimental results demonstrate the effectiveness of ReProver over non-retrieval baselines and GPT-4. We thus provide the first set of open-source LLM-based theorem provers without any proprietary datasets and release it under a permissive MIT license to facilitate further research.

----

## [944] Cognitive Steering in Deep Neural Networks via Long-Range Modulatory Feedback Connections

**Authors**: *Talia Konkle, George A. Alvarez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/444b09beab8438d4a58e9bc694dca32a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/444b09beab8438d4a58e9bc694dca32a-Abstract-Conference.html)

**Abstract**:

Given the rich visual information available in each glance, humans can internally direct their visual attention to enhance goal-relevant information---a capacity often absent in standard vision models.  Here we introduce cognitively and biologically-inspired long-range modulatory pathways to enable `cognitive steeringâ€™ in vision models.  First, we show that models equipped with these feedback pathways naturally show improved image recognition, adversarial robustness, and increased brain alignment, relative to baseline models. Further,  these feedback projections from the final layer of the vision backbone provide a meaningful steering interface, where goals can be specified as vectors in the output space.  We show that there are effective ways to steer the model that dramatically improve recognition of categories in composite images of multiple categories, succeeding where baseline feed-forward models without flexible steering fail. And, our multiplicative modulatory motif prevents rampant hallucination of the top-down goal category, dissociating what the model is looking for, from what it is looking at. Thus, these long-range modulatory pathways enable new behavioral capacities for goal-directed visual encoding, offering a flexible communication interface between cognitive and visual systems.

----

## [945] Neuro-symbolic Learning Yielding Logical Constraints

**Authors**: *Zenan Li, Yunpeng Huang, Zhaoyu Li, Yuan Yao, Jingwei Xu, Taolue Chen, Xiaoxing Ma, Jian Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4459c3c143db74ee52afebdf56836375-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4459c3c143db74ee52afebdf56836375-Abstract-Conference.html)

**Abstract**:

Neuro-symbolic systems combine the abilities of neural perception and logical reasoning. However, end-to-end learning of neuro-symbolic systems is still an unsolved challenge. This paper proposes a natural framework that fuses neural network training, symbol grounding, and logical constraint synthesis into a coherent and efficient end-to-end learning process. The capability of this framework comes from the improved interactions between the neural and the symbolic parts of the system in both the training and inference stages. Technically, to bridge the gap between the continuous neural network and the discrete logical constraint, we introduce a difference-of-convex programming technique to relax the logical constraints while maintaining their precision. We also employ cardinality constraints as the language for logical constraint learning and incorporate a trust region method to avoid the degeneracy of logical constraint in learning. Both theoretical analyses and empirical evaluations substantiate the effectiveness of the proposed framework.

----

## [946] Exploiting Connections between Lipschitz Structures for Certifiably Robust Deep Equilibrium Models

**Authors**: *Aaron J. Havens, Alexandre Araujo, Siddharth Garg, Farshad Khorrami, Bin Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4462db5eee6823b2abad0d1f955e187a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4462db5eee6823b2abad0d1f955e187a-Abstract-Conference.html)

**Abstract**:

Recently, deep equilibrium models (DEQs) have drawn increasing attention from the machine learning community. However, DEQs are much less understood in terms of certified robustness than their explicit network counterparts. In this paper, we advance the understanding of certified robustness of DEQs via exploiting the connections between various Lipschitz network parameterizations for both explicit and implicit models. Importantly, we show that various popular Lipschitz network structures, including convex potential layers (CPL), SDP-based Lipschitz layers (SLL), almost orthogonal layers (AOL), Sandwich layers, and monotone DEQs (MonDEQ) can all be reparameterized as special cases of the Lipschitz-bounded equilibrium networks (LBEN) without changing the prescribed Lipschitz constant in the original network parameterization. A key feature of our reparameterization technique is that it preserves the Lipschitz prescription used in different structures. This opens the possibility of achieving improved certified robustness of DEQs via a combination of network reparameterization, structure-preserving regularization, and LBEN-based fine-tuning. We also support our theoretical understanding with new empirical results, which show that our proposed method improves the certified robust accuracy of DEQs on classification tasks. All codes and experiments are made available at \url{https://github.com/AaronHavens/ExploitingLipschitzDEQ}.

----

## [947] A Combinatorial Algorithm for Approximating the Optimal Transport in the Parallel and MPC Settings

**Authors**: *Nathaniel Lahn, Sharath Raghvendra, Kaiyi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/448444518637da106d978ae7409d9789-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/448444518637da106d978ae7409d9789-Abstract-Conference.html)

**Abstract**:

Optimal Transport is a popular distance metric for measuring similarity between distributions. Exact and approximate combinatorial algorithms for computing the optimal transport distance are hard to parallelize. This has motivated the development of numerical solvers (e.g. Sinkhorn method) that can exploit GPU parallelism and produce approximate solutions. We introduce the first parallel combinatorial algorithm to find an additive $\varepsilon$-approximation of the OT distance. The parallel complexity of our algorithm is $O(\log(n)/ \varepsilon^2)$ where $n$ is the total support size for the input distributions. In Massive Parallel Computation (MPC) frameworks such as Hadoop and MapReduce, our algorithm computes an $\varepsilon$-approximate transport plan in $O(\log (\log (n/\varepsilon))/\varepsilon^2)$ rounds with $O(n/\varepsilon)$ space per machine; all prior algorithms in the MPC framework take $\Omega(\log n)$ rounds. We also provide a GPU-friendly matrix-based interpretation of our algorithm where each step of the algorithm is row or column manipulation of the matrix. Experiments suggest that our combinatorial algorithm is faster than the state-of-the-art approximate solvers in the GPU, especially for higher values of $n$.

----

## [948] RegBN: Batch Normalization of Multimodal Data with Regularization

**Authors**: *Morteza Ghahremani, Christian Wachinger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4488bf8354049b1cd592b6418dc30466-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4488bf8354049b1cd592b6418dc30466-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed a surge of interest in integrating high-dimensional data captured by multisource sensors, driven by the impressive success of neural networks in integrating multimodal data. However, the integration of heterogeneous multimodal data poses a significant challenge, as confounding effects and dependencies among such heterogeneous data sources introduce unwanted variability and bias, leading to suboptimal performance of multimodal models. Therefore, it becomes crucial to normalize the low- or high-level features extracted from data modalities before their fusion takes place. This paper introduces RegBN, a novel approach for multimodal Batch Normalization with REGularization. RegBN uses the Frobenius norm as a regularizer term to address the side effects of confounders and underlying dependencies among different data sources. The proposed method generalizes well across multiple modalities and eliminates the need for learnable parameters, simplifying training and inference. We validate the effectiveness of RegBN on eight databases from five research areas, encompassing diverse modalities such as language, audio, image, video, depth, tabular, and 3D MRI. The proposed method demonstrates broad applicability across different architectures such as multilayer perceptrons, convolutional neural networks, and vision transformers, enabling effective normalization of both low- and high-level features in multimodal neural networks. RegBN is available at https://mogvision.github.io/RegBN.

----

## [949] LLM-Pruner: On the Structural Pruning of Large Language Models

**Authors**: *Xinyin Ma, Gongfan Fang, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in both the deployment, inference, and training stages. With LLM being a general-purpose task solver, we explore its compression in a task-agnostic manner, which aims to preserve the multi-task solving and language generation ability of the original LLM. One challenge to achieving this is the enormous size of the training corpus of LLM, which makes both data transfer and model post-training over-burdensome. Thus, we tackle the compression of LLMs within the bound of two constraints: being task-agnostic and minimizing the reliance on the original training dataset. Our method, named LLM-pruner, adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionality. To this end, the performance of pruned models can be efficiently recovered through tuning techniques, LoRA, in merely 3 hours, requiring only 50K data. We validate the LLM-Pruner on three LLMs, including LLaMA, Vicuna, and ChatGLM, and demonstrate that the compressed models still exhibit satisfactory capabilities in zero-shot classification and generation. The code will be made public.

----

## [950] Nearly Optimal VC-Dimension and Pseudo-Dimension Bounds for Deep Neural Network Derivatives

**Authors**: *Yahong Yang, Haizhao Yang, Yang Xiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/449a016a6ce6fba3fe50d05482abf836-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/449a016a6ce6fba3fe50d05482abf836-Abstract-Conference.html)

**Abstract**:

This paper addresses the problem of  nearly optimal Vapnik--Chervonenkis dimension (VC-dimension) and pseudo-dimension estimations of the derivative functions of deep neural networks (DNNs). Two important applications of these estimations include: 1) Establishing a nearly tight approximation result of DNNs in the Sobolev space; 2)  Characterizing the generalization error of machine learning methods with loss functions involving function derivatives. This theoretical investigation fills the gap of learning error estimations for a wide range of physics-informed machine learning models and applications including generative models, solving partial differential equations, operator learning, network compression, distillation, regularization, etc.

----

## [951] ClimateSet: A Large-Scale Climate Model Dataset for Machine Learning

**Authors**: *Julia Kaltenborn, Charlotte E. E. Lange, Venkatesh Ramesh, Philippe Brouillard, Yaniv Gurwicz, Chandni Nagda, Jakob Runge, Peer Nowack, David Rolnick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44a6769fe6c695f8dfb347c649f7c9f0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/44a6769fe6c695f8dfb347c649f7c9f0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Climate models have been key for assessing the impact of climate change and simulating future climate scenarios. The machine learning (ML) community has taken an increased interest in supporting climate scientists’ efforts on various tasks such as climate model emulation, downscaling, and prediction tasks. Many of those tasks have been addressed on datasets created with single climate models. However, both the climate science and ML communities have suggested that to address those tasks at scale, we need large, consistent, and ML-ready climate model datasets. Here, we introduce ClimateSet, a dataset containing the inputs and outputs of 36 climate models from the Input4MIPs and CMIP6 archives. In addition, we provide a modular dataset pipeline for retrieving and preprocessing additional climate models and scenarios. We showcase the potential of our dataset by using it as a benchmark for ML-based climate model emulation. We gain new insights about the performance and generalization capabilities of the different ML models by analyzing their performance across different climate models. Furthermore, the dataset can be used to train an ML emulator on several climate models instead of just one. Such a “super emulator” can quickly project new climate change scenarios, complementing existing scenarios already provided to policymakers. We believe ClimateSet will create the basis needed for the ML community to tackle climate-related tasks at scale.

----

## [952] Near-Optimal Bounds for Learning Gaussian Halfspaces with Random Classification Noise

**Authors**: *Ilias Diakonikolas, Jelena Diakonikolas, Daniel Kane, Puqian Wang, Nikos Zarifis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44c150733f9c5b6f98cb0caad0c664c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/44c150733f9c5b6f98cb0caad0c664c7-Abstract-Conference.html)

**Abstract**:

We study the problem of learning general (i.e., not necessarily homogeneous) halfspaces with Random Classification Noise under the Gaussian distribution. We establish nearly-matching algorithmic and Statistical Query (SQ) lower bound results revealing a surprising information-computation gap for this basic problem. Specifically, the sample complexity of this learning problem is $\widetilde{\Theta}(d/\epsilon)$, where $d$ is the dimension and $\epsilon$ is the excess error. Our positive result is a computationally efficient learning algorithm with sample complexity$\tilde{O}(d/\epsilon + d/\max(p, \epsilon))^2)$, where $p$ quantifies the bias of the target halfspace. On the lower bound side, we show that any efficient SQ algorithm (or low-degree test)for the problem requires sample complexity at least $\Omega(d^{1/2}/(\max(p, \epsilon))^2)$. Our lower bound suggests that this quadratic dependence on $1/\epsilon$ is inherent for efficient algorithms.

----

## [953] Explain Any Concept: Segment Anything Meets Concept-Based Explanation

**Authors**: *Ao Sun, Pingchuan Ma, Yuanyuan Yuan, Shuai Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44cdeb5ab7da31d9b5cd88fd44e3da84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/44cdeb5ab7da31d9b5cd88fd44e3da84-Abstract-Conference.html)

**Abstract**:

EXplainable AI (XAI) is an essential topic to improve human understanding of deep neural networks (DNNs) given their black-box internals. For computer vision tasks, mainstream pixel-based XAI methods explain DNN decisions by identifying important pixels, and emerging concept-based XAI explore forming explanations with concepts (e.g., a head in an image). However, pixels are generally hard to interpret and sensitive to the imprecision of XAI methods, whereas “concepts” in prior works require human annotation or are limited to pre-defined concept sets. On the other hand, driven by large-scale pre-training, Segment Anything Model (SAM) has been demonstrated as a powerful and promotable framework for performing precise and comprehensive instance segmentation, enabling automatic preparation of concept sets from a given image. This paper for the first time explores using SAM to augment concept-based XAI. We offer an effective and flexible concept-based explanation method, namely Explain Any Concept (EAC), which explains DNN decisions with any concept. While SAM is highly effective and offers an “out-of-the-box” instance segmentation, it is costly when being integrated into defacto XAI pipelines. We thus propose a lightweight per-input equivalent (PIE) scheme, enabling efficient explanation with a surrogate model. Our evaluation  over two popular datasets (ImageNet and COCO) illustrate the highly encouraging performance of EAC over commonly-used XAI methods.

----

## [954] Data-Driven Network Neuroscience: On Data Collection and Benchmark

**Authors**: *Jiaxing Xu, Yunhan Yang, David Tse Jung Huang, Sophi Shilpa Gururajapathy, Yiping Ke, Miao Qiao, Alan Wang, Haribalan Kumar, Josh McGeown, Eryn Kwon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44e3a3115ca26e5127851acd0cedd0d9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/44e3a3115ca26e5127851acd0cedd0d9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper presents a comprehensive and quality collection of functional human brain network data for potential research in the intersection of neuroscience, machine learning, and graph analytics. Anatomical and functional MRI images have been used to understand the functional connectivity of the human brain and are particularly important in identifying underlying neurodegenerative conditions such as Alzheimer's, Parkinson's, and Autism. Recently, the study of the brain in the form of brain networks using machine learning and graph analytics has become increasingly popular, especially to predict the early onset of these conditions. A brain network, represented as a graph, retains rich structural and positional information that traditional examination methods are unable to capture. However, the lack of publicly accessible brain network data prevents researchers from data-driven explorations. One of the main difficulties lies in the complicated domain-specific preprocessing steps and the exhaustive computation required to convert the data from MRI images into brain networks. We bridge this gap by collecting a large amount of MRI images from public databases and a private source, working with domain experts to make sensible design choices, and preprocessing the MRI images to produce a collection of brain network datasets. The datasets originate from 6 different sources, cover 4 brain conditions, and consist of a total of 2,702 subjects. We test our graph datasets on 12 machine learning models to provide baselines and validate the data quality on a recent graph analysis model. To lower the barrier to entry and promote the research in this interdisciplinary field, we release our brain network data and complete preprocessing details including codes at https://doi.org/10.17608/k6.auckland.21397377 and https://github.com/brainnetuoa/datadrivennetwork_neuroscience.

----

## [955] No-Regret Learning with Unbounded Losses: The Case of Logarithmic Pooling

**Authors**: *Eric Neyman, Tim Roughgarden*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/44ecfb60950e868a13172b935b7964a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/44ecfb60950e868a13172b935b7964a9-Abstract-Conference.html)

**Abstract**:

For each of $T$ time steps, $m$ experts report probability distributions over $n$ outcomes; we wish to learn to aggregate these forecasts in a way that attains a no-regret guarantee. We focus on the fundamental and practical aggregation method known as *logarithmic pooling* -- a weighted average of log odds -- which is in a certain sense the optimal choice of pooling method if one is interested in minimizing log loss (as we take to be our loss function). We consider the problem of learning the best set of parameters (i.e. expert weights) in an online adversarial setting.  We assume (by necessity) that the adversarial choices of outcomes and forecasts are consistent, in the sense that experts report calibrated forecasts. Imposing this constraint creates a (to our knowledge) novel semi-adversarial setting in which the adversary retains a large amount of flexibility. In this setting, we present an algorithm based on online mirror descent that learns expert weights in a way that attains $O(\sqrt{T} \log T)$ expected regret as compared with the best weights in hindsight.

----

## [956] PanoGen: Text-Conditioned Panoramic Environment Generation for Vision-and-Language Navigation

**Authors**: *Jialu Li, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html)

**Abstract**:

Vision-and-Language Navigation requires the agent to follow language instructions to navigate through 3D environments. One main challenge in Vision-and-Language Navigation is the limited availability of photorealistic training environments, which makes it hard to generalize to new and unseen environments. To address this problem, we propose PanoGen, a generation method that can potentially create an infinite number of diverse panoramic environments conditioned on text. Specifically, we collect room descriptions by captioning the room images in existing Matterport3D environments, and leverage a state-of-the-art text-to-image diffusion model to generate the new panoramic environments. We use recursive outpainting over the generated images to create consistent 360-degree panorama views. Our new panoramic environments share similar semantic information with the original environments by conditioning on text descriptions, which ensures the co-occurrence of objects in the panorama follows human intuition, and creates enough diversity in room appearance and layout with image outpainting. Lastly, we explore two ways of utilizing PanoGen in VLN pre-training and fine-tuning. We generate instructions for paths in our PanoGen environments with a speaker built on a pre-trained vision-and-language model for VLN pre-training, and augment the visual observation with our panoramic environments during agents' fine-tuning to avoid overfitting to seen environments. Empirically, learning with our PanoGen environments achieves the new state-of-the-art on the Room-to-Room, Room-for-Room, and CVDN datasets. Besides, we find that pre-training with our PanoGen speaker data is especially effective for CVDN, which has under-specified instructions and needs commonsense knowledge to reach the target. Lastly, we show that the agent can benefit from training with more generated panoramic environments, suggesting promising results for scaling up the PanoGen environments to enhance agents' generalization to unseen environments.

----

## [957] Scaling laws for language encoding models in fMRI

**Authors**: *Richard Antonello, Aditya R. Vaidya, Alexander Huth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4533e4a352440a32558c1c227602c323-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4533e4a352440a32558c1c227602c323-Abstract-Conference.html)

**Abstract**:

Representations from transformer-based unidirectional language models are known to be effective at predicting brain responses to natural language. However, most studies comparing language models to brains have used GPT-2 or similarly sized language models. Here we tested whether larger open-source models such as those from the OPT and LLaMA families are better at predicting brain responses recorded using fMRI. Mirroring scaling results from other contexts, we found that brain prediction performance scales logarithmically with model size from 125M to 30B parameter models, with ~15% increased encoding performance as measured by correlation with a held-out test set across 3 subjects. Similar log-linear behavior was observed when scaling the size of the fMRI training set. We also characterized scaling for acoustic encoding models that use HuBERT, WavLM, and Whisper, and we found comparable improvements with model size. A noise ceiling analysis of these large, high-performance encoding models showed that performance is nearing the theoretical maximum for brain areas such as the precuneus and higher auditory cortex. These results suggest that increasing scale in both models and data will yield incredibly effective models of language processing in the brain, enabling better scientific understanding as well as applications such as decoding.

----

## [958] Optimal Rates for Bandit Nonstochastic Control

**Authors**: *Y. Jennifer Sun, Stephen H. Newman, Elad Hazan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45591d6727f0e127295f8d16adba6b23-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/45591d6727f0e127295f8d16adba6b23-Abstract-Conference.html)

**Abstract**:

Linear Quadratic Regulator (LQR) and Linear Quadratic Gaussian (LQG) control are foundational and extensively researched problems in optimal control. We investigate LQR and LQG problems with semi-adversarial perturbations and time-varying adversarial bandit loss functions. The best-known sublinear regret algorithm~\cite{gradu2020non} has a $T^{\frac{3}{4}}$ time horizon dependence, and its authors posed an open question about whether a tight rate of $\sqrt{T}$ could be achieved. We answer in the affirmative, giving an algorithm for bandit LQR and LQG which attains optimal regret, up to logarithmic factors. A central component of our method is a new scheme for bandit convex optimization with memory, which is of independent interest.

----

## [959] Flow-Attention-based Spatio-Temporal Aggregation Network for 3D Mask Detection

**Authors**: *Yuxin Cao, Yian Li, Yumeng Zhu, Derui Wang, Minhui Xue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/456f9445d0fa1a932d19584ab788c787-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/456f9445d0fa1a932d19584ab788c787-Abstract-Conference.html)

**Abstract**:

Anti-spoofing detection has become a necessity for face recognition systems due to the security threat posed by spoofing attacks. Despite great success in traditional attacks, most deep-learning-based methods perform poorly in 3D masks, which can highly simulate real faces in appearance and structure, suffering generalizability insufficiency while focusing only on the spatial domain with single frame input. This has been mitigated by the recent introduction of a biomedical technology called rPPG (remote photoplethysmography). However, rPPG-based methods are sensitive to noisy interference and require at least one second (> 25 frames) of observation time, which induces high computational overhead. To address these challenges, we propose a novel 3D mask detection framework, called FASTEN (Flow-Attention-based Spatio-Temporal aggrEgation Network). We tailor the network for focusing more on fine-grained details in large movements, which can eliminate redundant spatio-temporal feature interference and quickly capture splicing traces of 3D masks in fewer frames. Our proposed network contains three key modules: 1) a facial optical flow network to obtain non-RGB inter-frame flow information; 2) flow attention to assign different significance to each frame; 3) spatio-temporal aggregation to aggregate high-level spatial features and temporal transition features. Through extensive experiments, FASTEN only requires five frames of input and outperforms eight competitors for both intra-dataset and cross-dataset evaluations in terms of multiple detection metrics. Moreover, FASTEN has been deployed in real-world mobile devices for practical 3D mask detection.

----

## [960] On the Last-iterate Convergence in Time-varying Zero-sum Games: Extra Gradient Succeeds where Optimism Fails

**Authors**: *Yi Feng, Hu Fu, Qun Hu, Ping Li, Ioannis Panageas, Bo Peng, Xiao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/457ab261562014550e53351422f69834-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/457ab261562014550e53351422f69834-Abstract-Conference.html)

**Abstract**:

Last-iterate convergence has received extensive study in two player zero-sum games starting from bilinear, convex-concave up to settings that satisfy the MVI condition. Typical methods that exhibit last-iterate convergence for the aforementioned games include extra-gradient (EG) and optimistic gradient descent ascent (OGDA). However, all the established last-iterate convergence results hold for the restrictive setting where the underlying repeated game does not change over time.Recently, a line of research has focused on regret analysis of OGDA  in time-varying games, i.e., games where payoffs evolve with time; the last-iterate behavior of OGDA and EG in time-varying environments remains unclear though. In this paper, we study the last-iterate behavior of various algorithms in two types of unconstrained, time-varying, bilinear zero-sum games: periodic and convergent perturbed games. These models expand upon the usual repeated game formulation and incorporate external environmental factors, such as the seasonal effects on species competition and vanishing external noise. In periodic games, we prove that EG will converge while OGDA and momentum method will diverge. This is quite surprising, as to the best of our knowledge, it is the first result that indicates EG and OGDA have qualitatively different last-iterate behaviors and do not exhibit similar behavior. In convergent perturbed games, we prove all these algorithms converge as long as the game itself stabilizes with a faster rate than $1/t$.

----

## [961] Taking the neural sampling code very seriously: A data-driven approach for evaluating generative models of the visual system

**Authors**: *Suhas Shrinivasan, Konstantin-Klemens Lurz, Kelli Restivo, George H. Denfield, Andreas S. Tolias, Edgar Y. Walker, Fabian H. Sinz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/458d9f2dd5c7565af60143630dc62f10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/458d9f2dd5c7565af60143630dc62f10-Abstract-Conference.html)

**Abstract**:

Prevailing theories of perception hypothesize that the brain implements perception via Bayesian inference in a generative model of the world.One prominent theory, the Neural Sampling Code (NSC), posits that neuronal responses to a stimulus represent samples from the posterior distribution over latent world state variables that cause the stimulus.Although theoretically elegant, NSC does not specify the exact form of the generative model or prescribe how to link the theory to recorded neuronal activity.Previous works assume simple generative models and test their qualitative agreement with neurophysiological data.Currently, there is no precise alignment of the normative theory with neuronal recordings, especially in response to natural stimuli, and a quantitative, experimental evaluation of models under NSC has been lacking.Here, we propose a novel formalization of NSC, that (a) allows us to directly fit NSC generative models to recorded neuronal activity in response to natural images, (b) formulate richer and more flexible generative models, and (c) employ standard metrics to quantitatively evaluate different generative models under NSC.Furthermore, we derive a stimulus-conditioned predictive model of neuronal responses from the trained generative model using our formalization that we compare to neural system identification models.We demonstrate our approach by fitting and comparing classical- and flexible deep learning-based generative models on population recordings from the macaque primary visual cortex (V1) to natural images, and show that the flexible models outperform classical models in both their generative- and predictive-model performance.Overall, our work is an important step towards a quantitative evaluation of NSC. It provides a framework that lets us \textit{learn} the generative model directly from neuronal population recordings, paving the way for an experimentally-informed understanding of probabilistic computational principles underlying perception and behavior.

----

## [962] Can semi-supervised learning use all the data effectively? A lower bound perspective

**Authors**: *Alexandru Tifrea, Gizem Yüce, Amartya Sanyal, Fanny Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/458fa8ee331566383d8e74bdb647f829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/458fa8ee331566383d8e74bdb647f829-Abstract-Conference.html)

**Abstract**:

Prior theoretical and empirical works have established that semi-supervised learning algorithms can leverage the unlabeled data to improve over the labeled sample complexity of supervised learning (SL) algorithms. However, existing theoretical work focuses on regimes where the unlabeled data is sufficient to learn a good decision boundary using unsupervised learning (UL) alone. This begs the question: Can SSL algorithms simultaneously improve upon both UL and SL? To this end, we derive a tight lower bound for 2-Gaussian mixture models that explicitly depends on the labeled and the unlabeled dataset size as well as the signal-to-noise ratio of the mixture distribution. Surprisingly, our result implies that no SSL algorithm improves upon the minimax-optimal statistical error rates of SL or UL algorithms for these distributions. Nevertheless, in our real-world experiments, SSL algorithms can often outperform UL and SL algorithms. In summary, our work suggests that while it is possible to prove the performance gains of SSL algorithms, this would require careful tracking of constants in the theoretical analysis.

----

## [963] Evolving Standardization for Continual Domain Generalization over Temporal Drift

**Authors**: *Mixue Xie, Shuang Li, Longhui Yuan, Chi Harold Liu, Zehui Dai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/459a911eb49cd2e0192055ee156d04e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/459a911eb49cd2e0192055ee156d04e5-Abstract-Conference.html)

**Abstract**:

The capability of generalizing to out-of-distribution data is crucial for the deployment of machine learning models in the real world. Existing domain generalization (DG) mainly embarks on offline and discrete scenarios, where multiple source domains are simultaneously accessible and the distribution shift among domains is abrupt and violent. Nevertheless, such setting may not be universally applicable to all real-world applications, as there are cases where the data distribution gradually changes over time due to various factors, e.g., the process of aging. Additionally, as the domain constantly evolves, new domains will continually emerge. Re-training and updating models with both new and previous domains using existing DG methods can be resource-intensive and inefficient. Therefore, in this paper, we present a problem formulation for Continual Domain Generalization over Temporal Drift (CDGTD). CDGTD addresses the challenge of gradually shifting data distributions over time, where domains arrive sequentially and models can only access the data of the current domain. The goal is to generalize to unseen domains that are not too far into the future. To this end, we propose an Evolving Standardization (EvoS) method, which characterizes the evolving pattern of feature distribution and mitigates the distribution shift by standardizing features with generated statistics of corresponding domain. Specifically, inspired by the powerful ability of transformers to model sequence relations, we design a multi-scale attention module (MSAM) to learn the evolving pattern under sliding time windows of different lengths. MSAM can generate statistics of current domain based on the statistics of previous domains and the learned evolving pattern. Experiments on multiple real-world datasets including images and texts validate the efficacy of our EvoS.

----

## [964] Learning the Efficient Frontier

**Authors**: *Philippe Chatigny, Ivan Sergienko, Ryan Ferguson, Jordan Weir, Maxime Bergeron*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45a7ca247462d9e465ee88c8a302ca70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/45a7ca247462d9e465ee88c8a302ca70-Abstract-Conference.html)

**Abstract**:

The efficient frontier (EF) is a fundamental resource allocation problem where one has to find an optimal portfolio maximizing a reward at a given level of risk. This optimal solution is traditionally found by solving a convex optimization problem. In this paper, we introduce NeuralEF: a fast neural approximation framework that robustly forecasts the result of the EF convex optimizations problems with respect to heterogeneous linear constraints and variable number of optimization inputs. By reformulating an optimization problem as a sequence to sequence problem, we show that NeuralEF is a viable solution to accelerate large-scale simulation while handling discontinuous behavior.

----

## [965] Dissecting Chain-of-Thought: Compositionality through In-Context Filtering and Learning

**Authors**: *Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris Papailiopoulos, Samet Oymak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45e15bae91a6f213d45e203b8a29be48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/45e15bae91a6f213d45e203b8a29be48-Abstract-Conference.html)

**Abstract**:

Chain-of-thought (CoT) is a method that enables language models to handle complex reasoning tasks by decomposing them into simpler steps. Despite its success, the underlying mechanics of CoT are not yet fully understood. In an attempt to shed light on this, our study investigates the impact of CoT on the ability of transformers to in-context learn a simple to study, yet general family of compositional functions: multi-layer perceptrons (MLPs). In this setting, we find that the success of CoT can be attributed to breaking down in-context learning of a compositional function into two distinct phases: focusing on and filtering data related to each step of the composition and in-context learning the single-step composition function. Through both experimental and theoretical evidence, we demonstrate how CoT significantly reduces the sample complexity of in-context learning (ICL) and facilitates the learning of complex functions that non-CoT methods struggle with. Furthermore, we illustrate how transformers can transition from vanilla in-context learning to mastering a compositional function with CoT by simply incorporating additional layers that perform the necessary data-filtering for CoT via the attention mechanism. In addition to these test-time benefits, we show CoT helps accelerate pretraining by learning shortcuts to represent complex functions and filtering plays an important role in this process. These findings collectively provide insights into the mechanics of CoT, inviting further investigation of its role in complex reasoning tasks.

----

## [966] Improving multimodal datasets with image captioning

**Authors**: *Thao Nguyen, Samir Yitzhak Gadre, Gabriel Ilharco, Sewoong Oh, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45e604a3e33d10fba508e755faa72345-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/45e604a3e33d10fba508e755faa72345-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Massive web datasets play a key role in the success of large vision-language models like CLIP and Flamingo. However, the raw web data is noisy, and existing filtering methods to reduce noise often come at the expense of data diversity. Our work focuses on caption quality as one major source of noise, and studies how generated captions can increase the utility of web-scraped datapoints with nondescript text. Through exploring different mixing strategies for raw and generated captions, we outperform the best filtering method proposed by the DataComp benchmark by 2% on ImageNet and 4% on average across 38 tasks, given a candidate pool of 128M image-text pairs. Our best approach is also 2x better at Flickr and MS-COCO retrieval. We then analyze what makes synthetic captions an effective source of text supervision. In experimenting with different image captioning models, we also demonstrate that the performance of a model on standard image captioning benchmarks (e.g., NoCaps CIDEr) is not a reliable indicator of the utility of the captions it generates for multimodal training. Finally, our experiments with using generated captions at DataComp's large scale (1.28B image-text pairs) offer insights into the limitations of synthetic text, as well as the importance of image curation with increasing training data quantity. The synthetic captions used in our experiments are now available on HuggingFace.

----

## [967] ClimSim: A large multi-scale dataset for hybrid physics-ML climate emulation

**Authors**: *Sungduk Yu, Walter M. Hannah, Liran Peng, Jerry Lin, Mohamed Aziz Bhouri, Ritwik Gupta, Björn Lütjens, Justus C. Will, Gunnar Behrens, Julius Busecke, Nora Loose, Charles Stern, Tom Beucler, Bryce E. Harrop, Benjamin R. Hillman, Andrea M. Jenney, Savannah L. Ferretti, Nana Liu, Animashree Anandkumar, Noah D. Brenowitz, Veronika Eyring, Nicholas Geneva, Pierre Gentine, Stephan Mandt, Jaideep Pathak, Akshay Subramaniam, Carl Vondrick, Rose Yu, Laure Zanna, Tian Zheng, Ryan Abernathey, Fiaz Ahmed, David C. Bader, Pierre Baldi, Elizabeth A. Barnes, Christopher S. Bretherton, Peter M. Caldwell, Wayne Chuang, Yilun Han, Yu Huang, Fernando Iglesias-Suarez, Sanket R. Jantre, Karthik Kashinath, Marat Khairoutdinov, Thorsten Kurth, Nicholas J. Lutsko, Po-Lun Ma, Griffin Mooers, J. David Neelin, David A. Randall, Sara Shamekh, Mark Taylor, Nathan M. Urban, Janni Yuval, Guang Zhang, Mike Pritchard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/45fbcc01349292f5e059a0b8b02c8c3f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/45fbcc01349292f5e059a0b8b02c8c3f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Modern climate projections lack adequate spatial and temporal resolution due to computational constraints. A consequence is inaccurate and imprecise predictions of critical processes such as storms. Hybrid methods that combine physics with machine learning (ML) have introduced a new generation of higher fidelity climate simulators that can sidestep Moore's Law by outsourcing compute-hungry, short, high-resolution simulations to ML emulators. However, this hybrid ML-physics simulation approach requires domain-specific treatment and has been inaccessible to ML experts because of lack of training data and relevant, easy-to-use workflows. We present ClimSim, the largest-ever dataset designed for hybrid ML-physics research. It comprises multi-scale climate simulations, developed by a consortium of climate scientists and ML researchers. It consists of 5.7 billion pairs of multivariate input and output vectors that isolate the influence of locally-nested, high-resolution, high-fidelity physics on a host climate simulator's macro-scale physical state.The dataset is global in coverage, spans multiple years at high sampling frequency, and is designed such that resulting emulators are compatible with downstream coupling into operational climate simulators. We implement a range of deterministic and stochastic regression baselines to highlight the ML challenges and their scoring. The data (https://huggingface.co/datasets/LEAP/ClimSim_high-res) and code (https://leap-stc.github.io/ClimSim) are released openly to support the development of hybrid ML-physics and high-fidelity climate simulations for the benefit of science and society.

----

## [968] Relative Entropic Optimal Transport: a (Prior-aware) Matching Perspective to (Unbalanced) Classification

**Authors**: *Liangliang Shi, Haoyu Zhen, Gu Zhang, Junchi Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4621451c25a7aa175dc00e5dd4a243a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4621451c25a7aa175dc00e5dd4a243a3-Abstract-Conference.html)

**Abstract**:

Classification is a fundamental problem in machine learning, and considerable efforts have been recently devoted to the demanding long-tailed setting due to its prevalence in nature. Departure from the Bayesian framework, this paper rethinks classification from a matching perspective by studying the matching probability between samples and labels with optimal transport (OT) formulation. Specifically, we first propose a new variant of optimal transport, called Relative Entropic Optimal Transport (RE-OT), which guides the coupling solution to a known prior information matrix. We gives some theoretical results and their proof for RE-OT and surprisingly find RE-OT can help to deblur for barycenter images. Then we adopt inverse RE-OT for training long-tailed data and find that the loss derived from RE-OT has a similar form to Softmax-based cross-entropy loss, indicating a close connection between optimal transport and classification and the potential for transferring concepts between these two academic fields, such as barycentric projection in OT, which can map the labels back to the feature space. We further derive an epoch-varying RE-OT loss, and do the experiments on unbalanced image classification,  molecule classification, instance segmentation and representation learning. Experimental results show its effectiveness.

----

## [969] Connecting Multi-modal Contrastive Representations

**Authors**: *Zehan Wang, Yang Zhao, Xize Cheng, Haifeng Huang, Jiageng Liu, Aoxiong Yin, Li Tang, Linjun Li, Yongqi Wang, Ziang Zhang, Zhou Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46362971bfc3a97e6a271f2eb90fba17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46362971bfc3a97e6a271f2eb90fba17-Abstract-Conference.html)

**Abstract**:

Multi-modal Contrastive Representation (MCR) learning aims to encode different modalities into a semantically aligned shared space. This paradigm shows remarkable generalization ability on numerous downstream tasks across various modalities. However, the reliance on massive high-quality data pairs limits its further development on more modalities. This paper proposes a novel training-efficient method for learning MCR without paired data called Connecting Multi-modal Contrastive Representations (C-MCR). Specifically, given two existing MCRs pre-trained on $(\mathcal{A}$, $\mathcal{B})$ and $(\mathcal{B}$, $\mathcal{C})$ modality pairs, we project them to a new space and use the data from the overlapping modality $\mathcal{B}$ to aligning the two MCRs in the new space. Meanwhile, since the modality pairs $(\mathcal{A}$, $\mathcal{B})$ and $(\mathcal{B}$, $\mathcal{C})$ are already aligned within each MCR, the connection learned by overlapping modality can also be transferred to non-overlapping modality pair $(\mathcal{A}$, $\mathcal{C})$. To unleash the potential of C-MCR, we further introduce a semantic-enhanced inter- and intra-MCR connection method. We first enhance the semantic consistency and completion of embeddings across different modalities for more robust alignment. Then we utilize the inter-MCR alignment to establish the connection, and employ the intra-MCR alignment to better maintain the connection for inputs from non-overlapping modalities. To demonstrate the effectiveness of C-MCR, we take the field of audio-visual and 3D-language learning as examples. Specifically, we connect CLIP and CLAP via texts to derive audio-visual representations, and integrate CLIP and ULIP via images for 3D-language representations. Remarkably, without using any paired data, C-MCR for audio-visual achieves state-of-the-art performance on audio-image retrieval, audio-visual source localization, and counterfactual audio-image recognition tasks. Furthermore, C-MCR for 3D-language also attains advanced zero-shot 3D point cloud classification accuracy on ModelNet40. Our project page is available at \url{https://c-mcr.github.io/C-MCR/}

----

## [970] Boosting Learning for LDPC Codes to Improve the Error-Floor Performance

**Authors**: *Heeyoul Kwak, Daeyoung Yun, Yongjune Kim, Sang-Hyo Kim, Jong-Seon No*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/463a91da3c832bd28912cd0d1b8d9974-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/463a91da3c832bd28912cd0d1b8d9974-Abstract-Conference.html)

**Abstract**:

Low-density parity-check (LDPC) codes have been successfully commercialized in communication systems due to their strong error correction capabilities and simple decoding process. However, the error-floor phenomenon of LDPC codes, in which the error rate stops decreasing rapidly at a certain level, presents challenges for achieving extremely low error rates and deploying LDPC codes in scenarios demanding ultra-high reliability. In this work, we propose training methods for neural min-sum (NMS) decoders to eliminate the error-floor effect. First, by leveraging the boosting learning technique of ensemble networks, we divide the decoding network into two neural decoders and train the post decoder to be specialized for uncorrected words that the first decoder fails to correct. Secondly, to address the vanishing gradient issue in training, we introduce a block-wise training schedule that locally trains a block of weights while retraining the preceding block. Lastly, we show that assigning different weights to unsatisfied check nodes effectively lowers the error-floor with a minimal number of weights. By applying these training methods to standard LDPC codes, we achieve the best error-floor performance compared to other decoding methods. The proposed NMS decoder, optimized solely through novel training methods without additional modules, can be integrated into existing LDPC decoders without incurring extra hardware costs. The source code is available at https://github.com/ghy1228/LDPCErrorFloor.

----

## [971] Learning Score-based Grasping Primitive for Human-assisting Dexterous Grasping

**Authors**: *Tianhao Wu, Mingdong Wu, Jiyao Zhang, Yunchong Gan, Hao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/464012c83279e19be4cd42c25f341c92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/464012c83279e19be4cd42c25f341c92-Abstract-Conference.html)

**Abstract**:

The use of anthropomorphic robotic hands for assisting individuals in situations where human hands may be unavailable or unsuitable has gained significant importance. In this paper, we propose a novel task called human-assisting dexterous grasping that aims to train a policy for controlling a robotic hand's fingers to assist users in grasping objects. Unlike conventional dexterous grasping, this task presents a more complex challenge as the policy needs to adapt to diverse user intentions, in addition to the object's geometry.  We address this challenge by proposing an approach consisting of two sub-modules: a hand-object-conditional grasping primitive called Grasping Gradient Field (GraspGF), and a history-conditional residual policy.  GraspGF learns 'how' to grasp by estimating the gradient of a synthesised success grasping example set, while the residual policy determines 'when' and at what speed the grasping action should be executed based on the trajectory history. Experimental results demonstrate the superiority of our proposed method compared to baselines, highlighting the user-awareness and practicality in real-world applications. The codes and demonstrations can be viewed at https://sites.google.com/view/graspgf.

----

## [972] Maximize to Explore: One Objective Function Fusing Estimation, Planning, and Exploration

**Authors**: *Zhihan Liu, Miao Lu, Wei Xiong, Han Zhong, Hao Hu, Shenao Zhang, Sirui Zheng, Zhuoran Yang, Zhaoran Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4640d5da5888238b9de7e0dbacd2c605-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4640d5da5888238b9de7e0dbacd2c605-Abstract-Conference.html)

**Abstract**:

In reinforcement learning (RL), balancing exploration and exploitation is crucial for achieving an optimal policy in a sample-efficient way. To this end, existing sample- efficient algorithms typically consist of three components: estimation, planning, and exploration. However, to cope with general function approximators, most of them involve impractical algorithmic components to incentivize exploration, such as data-dependent level-set constraints or complicated sampling procedures. To address this challenge, we propose an easy-to-implement RL framework called Maximize to Explore (MEX), which only needs to optimize unconstrainedly a single objective that integrates the estimation and planning components while balancing exploration and exploitation automatically. Theoretically, we prove that the MEX achieves a sublinear regret with general function approximators and is extendable to the zero-sum Markov game setting. Meanwhile, we adapt deep RL baselines to design practical versions of MEX in both the model-based and model-free settings, which outperform baselines in various MuJoCo environments with sparse reward by a stable margin. Compared with existing sample-efficient algorithms with general function approximators, MEX achieves similar sample efficiency while also enjoying a lower computational cost and is more compatible with modern deep RL methods.

----

## [973] Hokoff: Real Game Dataset from Honor of Kings and its Offline Reinforcement Learning Benchmarks

**Authors**: *Yun Qu, Boyuan Wang, Jianzhun Shao, Yuhang Jiang, Chen Chen, Zhenbin Ye, Liu Linc, Yang Feng, Lin Lai, Hongyang Qin, Minwen Deng, Juchao Zhuo, Deheng Ye, Qiang Fu, Yang Guang, Wei Yang, Lanxiao Huang, Xiangyang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/464fefa022aaefc85d901317bbf13f85-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/464fefa022aaefc85d901317bbf13f85-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The advancement of Offline Reinforcement Learning (RL) and Offline Multi-Agent Reinforcement Learning (MARL) critically depends on the availability of high-quality, pre-collected offline datasets that represent real-world complexities and practical applications. However, existing datasets often fall short in their simplicity and lack of realism. To address this gap, we propose Hokoff, a comprehensive set of pre-collected datasets that covers both offline RL and offline MARL, accompanied by a robust framework, to facilitate further research. This data is derived from Honor of Kings, a recognized Multiplayer Online Battle Arena (MOBA) game known for its intricate nature, closely resembling real-life situations. Utilizing this framework, we benchmark a variety of offline RL and offline MARL algorithms. We also introduce a novel baseline algorithm tailored for the inherent hierarchical action space of the game. We reveal the incompetency of current offline RL approaches in handling task complexity, generalization and multi-task learning.

----

## [974] Learning and Collusion in Multi-unit Auctions

**Authors**: *Simina Brânzei, Mahsa Derakhshan, Negin Golrezaei, Yanjun Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4661b55200c03a8c4bb9c2974b4fb12d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4661b55200c03a8c4bb9c2974b4fb12d-Abstract-Conference.html)

**Abstract**:

In a carbon auction, licenses for CO2 emissions are allocated among multiple interested players. Inspired by this setting, we consider repeated multi-unit auctions with uniform pricing, which are widely used in practice. Our contribution is to analyze these auctions in both the offline and online settings, by designing efficient bidding algorithms with low regret and giving regret lower bounds. We also analyze the quality of the equilibria  in  two  main variants of the auction, finding that one  variant is susceptible to collusion among the bidders while the other is not.

----

## [975] One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization

**Authors**: *Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, Hao Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4683beb6bab325650db13afd05d1a14a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4683beb6bab325650db13afd05d1a14a-Abstract-Conference.html)

**Abstract**:

Single image 3D reconstruction is an important but challenging task that requires extensive knowledge of our natural world. Many existing methods solve this problem by optimizing a neural radiance field under the guidance of 2D diffusion models but suffer from lengthy optimization time, 3D inconsistency results, and poor geometry. In this work, we propose a novel method that takes a single image of any object as input and generates a full 360-degree 3D textured mesh in a single feed-forward pass. Given a single image, we first use a view-conditioned 2D diffusion model, Zero123, to generate multi-view images for the input view, and then aim to lift them up to 3D space. Since traditional reconstruction methods struggle with inconsistent multi-view predictions, we build our 3D reconstruction module upon an SDF-based generalizable neural surface reconstruction method and propose several critical training strategies to enable the reconstruction of 360-degree meshes. Without costly optimizations, our method reconstructs 3D shapes in significantly less time than existing methods. Moreover, our method favors better geometry, generates more 3D consistent results, and adheres more closely to the input image. We evaluate our approach on both synthetic data and in-the-wild images and demonstrate its superiority in terms of both mesh quality and runtime. In addition, our approach can seamlessly support the text-to-3D task by integrating with off-the-shelf text-to-image diffusion models.

----

## [976] VeriX: Towards Verified Explainability of Deep Neural Networks

**Authors**: *Min Wu, Haoze Wu, Clark W. Barrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46907c2ff9fafd618095161d76461842-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46907c2ff9fafd618095161d76461842-Abstract-Conference.html)

**Abstract**:

We present VeriX (Verified eXplainability), a system for producing optimal robust explanations and generating counterfactuals along decision boundaries of machine learning models. We build such explanations and counterfactuals iteratively using constraint solving techniques and a heuristic based on feature-level sensitivity ranking. We evaluate our method on image recognition benchmarks and a real-world scenario of autonomous aircraft taxiing.

----

## [977] Generalized test utilities for long-tail performance in extreme multi-label classification

**Authors**: *Erik Schultheis, Marek Wydmuch, Wojciech Kotlowski, Rohit Babbar, Krzysztof Dembczynski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46994b3d6dd0fd5fca5f780af6259db5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46994b3d6dd0fd5fca5f780af6259db5-Abstract-Conference.html)

**Abstract**:

Extreme multi-label classification (XMLC) is the task of selecting a small subset of relevant labels from a very large set of possible labels. As such, it is characterized by long-tail labels, i.e., most labels have very few positive instances. With standard performance measures such as precision@k, a classifier can ignore tail labels and still report good performance. However, it is often argued that correct predictions in the tail are more "interesting" or "rewarding," but the community has not yet settled on a metric capturing this intuitive concept. The existing propensity-scored metrics fall short on this goal by confounding the problems of long-tail and missing labels. In this paper, we analyze generalized metrics budgeted "at k" as an alternative solution. To tackle the challenging problem of optimizing these metrics, we formulate it in the expected test utility (ETU) framework, which aims to optimize the expected performance on a given test set. We derive optimal prediction rules and construct their computationally efficient approximations with provable regret guarantees and being robust against model misspecification. Our algorithm, based on block coordinate descent, scales effortlessly to XMLC problems and obtains promising results in terms of long-tail performance.

----

## [978] Compositional Foundation Models for Hierarchical Planning

**Authors**: *Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi S. Jaakkola, Joshua B. Tenenbaum, Leslie Pack Kaelbling, Akash Srivastava, Pulkit Agrawal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html)

**Abstract**:

To make effective decisions in novel environments with long-horizon goals, it is crucial to engage in hierarchical reasoning across spatial and temporal scales. This entails planning abstract subgoal sequences, visually reasoning about the underlying plans, and executing actions in accordance with the devised plan through visual-motor control. We propose Compositional Foundation Models for Hierarchical Planning (HiP), a foundation model which leverages multiple expert foundation model trained on language, vision and action data individually jointly together to solve long-horizon tasks. We use a large language model to construct symbolic plans that are grounded in the environment through a large video diffusion model. Generated video plans are then grounded to visual-motor control, through an inverse dynamics model that infers actions from generated videos. To enable effective reasoning within this hierarchy, we enforce consistency between the models via iterative refinement. We illustrate the efficacy and adaptability of our approach in three different long-horizon table-top manipulation tasks.

----

## [979] Diffusion Model for Graph Inverse Problems: Towards Effective Source Localization on Complex Networks

**Authors**: *Xin Yan, Hui Fang, Qiang He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46ab9d9645b6975b947231ddb48da1ab-Abstract-Conference.html)

**Abstract**:

Information diffusion problems, such as the spread of epidemics or rumors, are widespread in society. The inverse problems of graph diffusion, which involve locating the sources and identifying the paths of diffusion based on currently observed diffusion graphs, are crucial to controlling the spread of information. The problem of localizing the source of diffusion is highly ill-posed, presenting a major obstacle in accurately assessing the uncertainty involved. Besides, while comprehending how information diffuses through a graph is crucial, there is a scarcity of research on reconstructing the paths of information propagation. To tackle these challenges, we propose a probabilistic model called DDMSL (Discrete Diffusion Model for Source Localization). Our approach is based on the natural diffusion process of information propagation over complex networks, which can be formulated using a message-passing function. First, we model the forward diffusion of information using Markov chains. Then, we design a reversible residual network to construct a denoising-diffusion model in discrete space for both source localization and reconstruction of information diffusion paths. We provide rigorous theoretical guarantees for DDMSL and demonstrate its effectiveness through extensive experiments on five real-world datasets.

----

## [980] UniT: A Unified Look at Certified Robust Training against Text Adversarial Perturbation

**Authors**: *Muchao Ye, Ziyi Yin, Tianrong Zhang, Tianyu Du, Jinghui Chen, Ting Wang, Fenglong Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46b065f7d301a15a23909f6cad409a97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46b065f7d301a15a23909f6cad409a97-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed a surge of certified robust training pipelines against text adversarial perturbation constructed by synonym substitutions. Given a base model, existing pipelines provide prediction certificates either in the discrete word space or the continuous latent space. However, they are isolated from each other with a structural gap. We observe that existing training frameworks need unification to provide stronger certified robustness. Additionally,  they mainly focus on building the certification process but neglect to improve the robustness of the base model. To mitigate the aforementioned limitations, we propose a unified framework named UniT that enables us to train flexibly in either fashion by working in the word embedding space. It can provide a stronger robustness guarantee obtained directly from the word embedding space without extra modules. In addition, we introduce the decoupled regularization (DR) loss to improve the robustness of the base model, which includes two separate robustness regularization terms for the feature extraction and classifier modules. Experimental results on widely used text classification datasets further demonstrate the effectiveness of the designed unified framework and the proposed DR loss for improving the certified robust accuracy.

----

## [981] Convergence of Alternating Gradient Descent for Matrix Factorization

**Authors**: *Rachel A. Ward, Tamara G. Kolda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46c10f6c8ea5aa6f267bcdabcb123f97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46c10f6c8ea5aa6f267bcdabcb123f97-Abstract-Conference.html)

**Abstract**:

We consider alternating gradient descent (AGD) with fixed step size applied to the asymmetric matrix factorization objective.  We show that, for a rank-$r$ matrix $A \in \mathbb{R}^{m \times n}$,  $T = C ( \frac{\sigma_1(A)}{\sigma_r(A)} )^2 \log(1/\epsilon)$  iterations of alternating gradient descent suffice to reach an $\epsilon$-optimal factorization   $\| A - X_{T} Y_{T}' \|^2 \leq \epsilon \| A \|^2$   with high probability  starting from an atypical random initialization. The  factors have rank $d \geq r$ so that $X_{T}\in \mathbb{R}^{m \times d}$ and $Y_{T} \in\mathbb{R}^{n \times d}$, and mild overparameterization suffices for the constant  $C$ in the iteration complexity $T$ to be an absolute constant.   Experiments suggest that our proposed initialization is not merely of theoretical benefit, but rather significantly improves the convergence rate of gradient descent in practice. Our proof is conceptually simple: a uniform Polyak-Lojasiewicz (PL) inequality and uniform Lipschitz smoothness constant are guaranteed for a sufficient number of iterations, starting from our random initialization.  Our proof method should be useful for extending and simplifying convergence analyses for a broader class of nonconvex low-rank factorization problems.

----

## [982] SPRING: Studying Papers and Reasoning to play Games

**Authors**: *Yue Wu, So Yeon Min, Shrimai Prabhumoye, Yonatan Bisk, Russ Salakhutdinov, Amos Azaria, Tom M. Mitchell, Yuanzhi Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46c2a9a6f2b2be68682013eb1173c801-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46c2a9a6f2b2be68682013eb1173c801-Abstract-Conference.html)

**Abstract**:

Open-world survival games pose significant challenges for AI algorithms due to their multi-tasking, deep exploration, and goal prioritization requirements. Despite reinforcement learning (RL) being popular for solving games, its high sample complexity limits its effectiveness in complex open-world games like Crafter or Minecraft. We propose a novel approach, SPRING, to read Crafter's original academic paper and use the knowledge learned to reason and play the game through a large language model (LLM).Prompted with the LaTeX source as game context and a description of the agent's current observation, our SPRING framework employs a directed acyclic graph (DAG) with game-related questions as nodes and dependencies as edges. We identify the optimal action to take in the environment by traversing the DAG and calculating LLM responses for each node in topological order, with the LLM's answer to final node directly translating to environment actions.In our experiments, we study the quality of in-context "reasoning" induced by different forms of prompts under the setting of the Crafter environment. Our experiments suggest that LLMs, when prompted with consistent chain-of-thought, have great potential in completing sophisticated high-level trajectories. Quantitatively, SPRING with GPT-4 outperforms all state-of-the-art RL baselines, trained for 1M steps, without any training. Finally, we show the potential of Crafter as a test bed for LLMs. Code at github.com/holmeswww/SPRING

----

## [983] Hybrid Search for Efficient Planning with Completeness Guarantees

**Authors**: *Kalle Kujanpää, Joni Pajarinen, Alexander Ilin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46d26daeb05fbbcfe5f3d8f7ca756e16-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46d26daeb05fbbcfe5f3d8f7ca756e16-Abstract-Conference.html)

**Abstract**:

Solving complex planning problems has been a long-standing challenge in computer science. Learning-based subgoal search methods have shown promise in tackling these problems, but they often suffer from a lack of completeness guarantees, meaning that they may fail to find a solution even if one exists. In this paper, we propose an efficient approach to augment a subgoal search method to achieve completeness in discrete action spaces. Specifically, we augment the high-level search with low-level actions to execute a multi-level (hybrid) search, which we call complete subgoal search. This solution achieves the best of both worlds: the practical efficiency of high-level search and the completeness of low-level search. We apply the proposed search method to a recently proposed subgoal search algorithm and evaluate the algorithm trained on offline data on complex planning problems. We demonstrate that our complete subgoal search not only guarantees completeness but can even improve performance in terms of search expansions for instances that the high-level could solve without low-level augmentations. Our approach makes it possible to apply subgoal-level planning for systems where completeness is a critical requirement.

----

## [984] Diversified Outlier Exposure for Out-of-Distribution Detection via Informative Extrapolation

**Authors**: *Jianing Zhu, Yu Geng, Jiangchao Yao, Tongliang Liu, Gang Niu, Masashi Sugiyama, Bo Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46d943bc6a15a57c923829efc0db7c7a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46d943bc6a15a57c923829efc0db7c7a-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) detection is important for deploying reliable machine learning models on real-world applications. Recent advances in outlier exposure have shown promising results on OOD detection via fine-tuning model with informatively sampled auxiliary outliers. However, previous methods assume that the collected outliers can be sufficiently large and representative to cover the boundary between ID and OOD data, which might be impractical and challenging. In this work, we propose a novel framework, namely, Diversified Outlier Exposure (DivOE), for effective OOD detection via informative extrapolation based on the given auxiliary outliers. Specifically, DivOE introduces a new learning objective, which diversifies the auxiliary distribution by explicitly synthesizing more informative outliers for extrapolation during training. It leverages a multi-step optimization method to generate novel outliers beyond the original ones, which is compatible with many variants of outlier exposure. Extensive experiments and analyses have been conducted to characterize and demonstrate the effectiveness of the proposed DivOE. The code is publicly available at: https://github.com/tmlr-group/DivOE.

----

## [985] Attacks on Online Learners: a Teacher-Student Analysis

**Authors**: *Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/46e37aeccafc3b4b697b17b8a36f3b30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/46e37aeccafc3b4b697b17b8a36f3b30-Abstract-Conference.html)

**Abstract**:

Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.

----

## [986] Delayed Algorithms for Distributed Stochastic Weakly Convex Optimization

**Authors**: *Wenzhi Gao, Qi Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/470e23d14e330ab0daa5387916b95f9c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/470e23d14e330ab0daa5387916b95f9c-Abstract-Conference.html)

**Abstract**:

This paper studies delayed stochastic algorithms for weakly convex optimization in a distributed network with workers connected to a master node.  Recently, Xu~et~al.~2022  showed that an inertial stochastic subgradient method   converges at a rate of $\mathcal{O}(\tau_{\text{max}}/\sqrt{K})$ which depends on the maximum information delay $\tau_{\text{max}}$. In this work, we show that the delayed stochastic subgradient method ($\texttt{DSGD}$) obtains a tighter convergence rate which depends on the expected delay $\bar{\tau}$. Furthermore, for an important class of composition weakly convex problems, we develop a new delayed stochastic prox-linear ($\texttt{DSPL}$) method in which the delays only affect the high-order term in the rate and hence,  are negligible after a certain number of $\texttt{DSPL}$  iterations.  In addition, we demonstrate the robustness of our proposed algorithms against arbitrary delays.  By incorporating a simple safeguarding step in both methods, we achieve convergence rates that depend solely on the number of workers, eliminating the effect of delays. Our numerical experiments further confirm the empirical superiority of our proposed methods.

----

## [987] Grounding Neural Inference with Satisfiability Modulo Theories

**Authors**: *Zifan Wang, Saranya Vijayakumar, Kaiji Lu, Vijay Ganesh, Somesh Jha, Matt Fredrikson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47167991e38c65a72914763c11cd8d23-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47167991e38c65a72914763c11cd8d23-Abstract-Conference.html)

**Abstract**:

Recent techniques that integrate solver layers into Deep Neural Networks (DNNs) have shown promise in bridging a long-standing gap between inductive learning and symbolic reasoning techniques. In this paper we present a set of techniques for integrating Satisfiability Modulo Theories (SMT) solvers into the forward and backward passes of a deep network layer, called SMTLayer.Using this approach, one can encode rich domain knowledge into the network in the form of mathematical formulas.In the forward pass, the solver uses symbols produced by prior layers, along with these formulas, to construct inferences; in the backward pass, the solver informs updates to the network, driving it towards representations that are compatible with the solver's theory.Notably, the solver need not be differentiable. We implement SMTLayer as a Pytorch module, and our empirical results show that it leads to models that 1) require fewer training samples than conventional models, 2) that are robust to certain types of covariate shift, and 3) that ultimately learn representations that are consistent with symbolic knowledge, and thus naturally interpretable.

----

## [988] D2CSG: Unsupervised Learning of Compact CSG Trees with Dual Complements and Dropouts

**Authors**: *Fenggen Yu, Qimin Chen, Maham Tanveer, Ali Mahdavi-Amiri, Hao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4732d425125832887f6c5a9675d49ead-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4732d425125832887f6c5a9675d49ead-Abstract-Conference.html)

**Abstract**:

We present D$^2$CSG, a neural model composed of two dual and complementary network branches, with dropouts, for unsupervised learning of compact constructive solid geometry (CSG) representations of 3D CAD shapes. Our network is trained to reconstruct a 3D shape by a fixed-order assembly of quadric primitives, with both branches producing a union of primitive intersections or inverses. A key difference between D$^2$CSG and all prior neural CSG models is its dedicated residual branch to assemble the potentially complex shape complement, which is subtracted from an overall shape modeled by the cover branch. With the shape complements, our network is provably general, while the weight dropout further improves compactness of the CSG tree by removing redundant primitives. We demonstrate both quantitatively and qualitatively that D$^2$CSG produces compact CSG reconstructions with superior quality and more natural primitives than all existing alternatives, especially over complex and high-genus CAD shapes.

----

## [989] Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering

**Authors**: *Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, Bill Byrne*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html)

**Abstract**:

Knowledge-based Visual Question Answering (KB-VQA) requires VQA systems to utilize knowledge from external knowledge bases to answer visually-grounded questions. Retrieval-Augmented Visual Question Answering (RA-VQA), a strong framework to tackle KB-VQA, first retrieves related documents with Dense Passage Retrieval (DPR) and then uses them to answer questions. This paper proposes Fine-grained Late-interaction Multi-modal Retrieval (FLMR) which significantly improves knowledge retrieval in RA-VQA. FLMR addresses two major limitations in RA-VQA's retriever: (1) the image representations obtained via image-to-text transforms can be incomplete and inaccurate and (2) similarity scores between queries and documents are computed with one-dimensional embeddings, which can be insensitive to finer-grained similarities.FLMR overcomes these limitations by obtaining image representations that complement those from the image-to-text transform using a vision model aligned with an existing text-based retriever through a simple alignment network. FLMR also encodes images and questions using multi-dimensional embeddings to capture finer-grained similarities between queries and documents. FLMR significantly improves the original RA-VQA retriever's PRRecall@5 by approximately 8\%. Finally, we equipped RA-VQA with two state-of-the-art large multi-modal/language models to achieve $\sim62$% VQA score in the OK-VQA dataset.

----

## [990] Iteratively Learn Diverse Strategies with State Distance Information

**Authors**: *Wei Fu, Weihua Du, Jingwei Li, Sunli Chen, Jingzhao Zhang, Yi Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/473aadf077f8464dbae7e9600d9be6c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/473aadf077f8464dbae7e9600d9be6c4-Abstract-Conference.html)

**Abstract**:

In complex reinforcement learning (RL) problems, policies with similar rewards may have substantially different behaviors. It remains a fundamental challenge to optimize rewards while also discovering as many diverse strategies as possible, which can be crucial in many practical applications. Our study examines two design choices for tackling this challenge, i.e., diversity measure and computation framework. First, we find that with existing diversity measures, visually indistinguishable policies can still yield high diversity scores. To accurately capture the behavioral difference, we propose to incorporate the state-space distance information into the diversity measure. In addition, we examine two common computation frameworks for this problem, i.e., population-based training (PBT) and iterative learning (ITR). We show that although PBT is the precise problem formulation, ITR can achieve comparable diversity scores with higher computation efficiency, leading to improved solution quality in practice. Based on our analysis, we further combine ITR with two tractable realizations of the state-distance-based diversity measures and develop a novel diversity-driven RL algorithm, State-based Intrinsic-reward Policy Optimization (SIPO), with provable convergence properties. We empirically examine SIPO across three domains from robot locomotion to multi-agent games. In all of our testing environments, SIPO consistently produces strategically diverse and human-interpretable policies that cannot be discovered by existing baselines.

----

## [991] Neural Fields with Hard Constraints of Arbitrary Differential Order

**Authors**: *Fangcheng Zhong, Kyle Fogarty, Param Hanji, Tianhao Wu, Alejandro Sztrajman, Andrew Spielberg, Andrea Tagliasacchi, Petra Bosilj, Cengiz Öztireli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47547ee84e3fbbcbbbbad7c1fd9a973b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47547ee84e3fbbcbbbbad7c1fd9a973b-Abstract-Conference.html)

**Abstract**:

While deep learning techniques have become extremely popular for solving a broad range of optimization problems, methods to enforce hard constraints during optimization, particularly on deep neural networks, remain underdeveloped. Inspired by the rich literature on meshless interpolation and its extension to spectral collocation methods in scientific computing, we develop a series of approaches for enforcing hard constraints on neural fields, which we refer to as Constrained Neural Fields (CNF). The constraints can be specified as a linear operator applied to the neural field and its derivatives. We also design specific model representations and training strategies for problems where standard models may encounter difficulties, such as conditioning of the system, memory consumption, and capacity of the network when being constrained. Our approaches are demonstrated in a wide range of real-world applications. Additionally, we develop a framework that enables highly efficient model and constraint specification, which can be readily applied to any downstream task where hard constraints need to be explicitly satisfied during optimization.

----

## [992] Thinker: Learning to Plan and Act

**Authors**: *Stephen Chung, Ivan Anokhin, David Krueger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4761fab863f0900d90cf601fce6d5155-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4761fab863f0900d90cf601fce6d5155-Abstract-Conference.html)

**Abstract**:

We propose the Thinker algorithm, a novel approach that enables reinforcement learning agents to autonomously interact with and utilize a learned world model. The Thinker algorithm wraps the environment with a world model and introduces new actions designed for interacting with the world model. These model-interaction actions enable agents to perform planning by proposing alternative plans to the world model before selecting a final action to execute in the environment. This approach eliminates the need for handcrafted planning algorithms by enabling the agent to learn how to plan autonomously and allows for easy interpretation of the agent's plan with visualization. We demonstrate the algorithm's effectiveness through experimental results in the game of Sokoban and the Atari 2600 benchmark, where the Thinker algorithm achieves state-of-the-art performance and competitive results, respectively. Visualizations of agents trained with the Thinker algorithm demonstrate that they have learned to plan effectively with the world model to select better actions. Thinker is the first work showing that an RL agent can learn to plan with a learned world model in complex environments.

----

## [993] Near-Optimal k-Clustering in the Sliding Window Model

**Authors**: *David P. Woodruff, Peilin Zhong, Samson Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/476ab8f369e489c04187ba84f68cfa68-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/476ab8f369e489c04187ba84f68cfa68-Abstract-Conference.html)

**Abstract**:

Clustering is an important technique for identifying structural information in large-scale data analysis, where the underlying dataset may be too large to store. In many applications, recent data can provide more accurate information and thus older data past a certain time is expired. The sliding window model captures these desired properties and thus there has been substantial interest in clustering in the sliding window model. In this paper, we give the first algorithm that achieves near-optimal $(1+\varepsilon)$-approximation to $(k,z)$-clustering in the sliding window model. Our algorithm uses $\frac{k}{\min(\varepsilon^4,\varepsilon^{2+z})}\,\text{polylog}\frac{n\Delta}{\varepsilon}$ words of space when the points are from $[\Delta]^d$, thus significantly improving on works by Braverman et. al. (SODA 2016), Borassi et. al. (NeurIPS 2021), and Epasto et. al. (SODA 2022).Along the way, we develop a data structure for clustering called an online coreset, which outputs a coreset not only for the end of a stream, but also for all prefixes of the stream. Our online coreset samples $\frac{k}{\min(\varepsilon^4,\varepsilon^{2+z})}\,\text{polylog}\frac{n\Delta}{\varepsilon}$ points from the stream. We then show that any online coreset requires $\Omega\left(\frac{k}{\varepsilon^2}\log n\right)$ samples, which shows a separation between the problem of constructing an offline coreset, i.e., constructing online coresets is strictly harder. Our results also extend to general metrics on $[\Delta]^d$ and are near-optimal in light of a $\Omega\left(\frac{k}{\varepsilon^{2+z}}\right)$ lower bound for the size of an offline coreset.

----

## [994] SynMob: Creating High-Fidelity Synthetic GPS Trajectory Dataset for Urban Mobility Analysis

**Authors**: *Yuanshao Zhu, Yongchao Ye, Ying Wu, Xiangyu Zhao, James Jian Qiao Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4786c0d1b9687a841bc579b0b8b01b8e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/4786c0d1b9687a841bc579b0b8b01b8e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Urban mobility analysis has been extensively studied in the past decade using a vast amount of GPS trajectory data, which reveals hidden patterns in movement and human activity within urban landscapes. Despite its significant value, the availability of such datasets often faces limitations due to privacy concerns, proprietary barriers, and quality inconsistencies. To address these challenges, this paper presents a synthetic trajectory dataset with high fidelity, offering a general solution to these data accessibility issues. Specifically, the proposed dataset adopts a diffusion model as its synthesizer, with the primary aim of accurately emulating the spatial-temporal behavior of the original trajectory data. These synthesized data can retain the geo-distribution and statistical properties characteristic of real-world datasets. Through rigorous analysis and case studies, we validate the high similarity and utility between the proposed synthetic trajectory dataset and real-world counterparts. Such validation underscores the practicality of synthetic datasets for urban mobility analysis and advocates for its wider acceptance within the research community. Finally, we publicly release the trajectory synthesizer and datasets, aiming to enhance the quality and availability of synthetic trajectory datasets and encourage continued contributions to this rapidly evolving field. The dataset is released for public online availability https://github.com/Applied-Machine-Learning-Lab/SynMob.

----

## [995] Window-Based Distribution Shift Detection for Deep Neural Networks

**Authors**: *Guy Bar-Shalom, Yonatan Geifman, Ran El-Yaniv*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4791edcba96fbd82a8962b0f790b52c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4791edcba96fbd82a8962b0f790b52c9-Abstract-Conference.html)

**Abstract**:

To deploy and operate deep neural models in production, the quality of their predictions, which might be contaminated benignly or manipulated maliciously by input distributional deviations, must be monitored and assessed. Specifically, we study the case of monitoring the healthy operation of a deep neural network (DNN) receiving a stream of data, with the aim of detecting input distributional deviations over which the quality of the network's predictions is potentially damaged. Using selective prediction principles, we propose a distribution deviation detection method for DNNs. The proposed method is derived from a tight coverage generalization bound computed over a sample of instances drawn from the true underlying distribution. Based on this bound, our detector continuously monitors the operation of the network over a test window and fires off an alarm whenever a deviation is detected. Our novel detection method performs on-par or better than the state-of-the-art, while consuming substantially lower computation time (five orders of magnitude reduction) and space complexity. Unlike previous methods, which require at least linear dependence on the size of the source distribution for each detection, rendering them inapplicable to ``Google-Scale'' datasets, our approach eliminates this dependence, making it suitable for real-world applications. Code is available at https://github.com/BarSGuy/Window-Based-Distribution-Shift-Detection.

----

## [996] Towards Label Position Bias in Graph Neural Networks

**Authors**: *Haoyu Han, Xiaorui Liu, Feng Shi, MohamadAli Torkamani, Charu C. Aggarwal, Jiliang Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4798eef078de031518beaf54f4b5fb5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4798eef078de031518beaf54f4b5fb5f-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have emerged as a powerful tool for semi-supervised node classification tasks. However, recent studies have revealed various biases in GNNs stemming from both node features and graph topology. In this work, we uncover a new bias - label position bias, which indicates that the node closer to the labeled nodes tends to perform better. We introduce a new metric, the Label Proximity Score, to quantify this bias, and find that it is closely related to performance disparities. To address the label position bias, we propose a novel optimization framework for learning a label position unbiased graph structure, which can be applied to existing GNNs. Extensive experiments demonstrate that our proposed method not only outperforms backbone methods but also significantly mitigates the issue of label position bias in GNNs.

----

## [997] Label Robust and Differentially Private Linear Regression: Computational and Statistical Efficiency

**Authors**: *Xiyang Liu, Prateek Jain, Weihao Kong, Sewoong Oh, Arun Sai Suggala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47e74fca60b4af4846b7abab188b85f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47e74fca60b4af4846b7abab188b85f2-Abstract-Conference.html)

**Abstract**:

We study the canonical problem of linear regression under $(\varepsilon,\delta)$-differential privacy when the datapoints are sampled i.i.d.~from a distribution and a fraction of response variables are adversarially corrupted. We provide the first provably efficient -- both computationally and statistically -- method for this problem, assuming standard assumptions on the data distribution. Our algorithm is a variant of the popular differentially private stochastic gradient descent (DP-SGD) algorithm with two key innovations: a full-batch gradient descent to improve sample complexity and a novel adaptive clipping to guarantee robustness. Our method requires only linear time in input size, and still matches the information theoretical optimal sample complexity up to a data distribution dependent condition number factor.  Interestingly, the same algorithm, when applied to a setting where there is no adversarial corruption, still improves upon the existing state-of-the-art and achieves a near optimal sample complexity.

----

## [998] Explainable and Efficient Randomized Voting Rules

**Authors**: *Soroush Ebadian, Aris Filos-Ratsikas, Mohamad Latifian, Nisarg Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47eb2874a790d5b1f554b9bb93b3de9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47eb2874a790d5b1f554b9bb93b3de9d-Abstract-Conference.html)

**Abstract**:

With a rapid growth in the deployment of AI tools for making critical decisions (or aiding humans in doing so), there is a growing demand to be able to explain to the stakeholders how these tools arrive at a decision. Consequently, voting is frequently used to make such decisions due to its inherent explainability. Recent work suggests that using randomized (as opposed to deterministic) voting rules can lead to significant efficiency gains measured via the distortion framework. However, rules that use intricate randomization can often become too complex to explain to the stakeholders; losing explainability can eliminate the key advantage of voting over black-box AI tools, which may outweigh the efficiency gains.We study the efficiency gains which can be unlocked by using voting rules that add a simple randomization step to a deterministic rule, thereby retaining explainability. We focus on two such families of rules, randomized positional scoring rules and random committee member rules, and show, theoretically and empirically, that they indeed achieve explainability and efficiency simultaneously to some extent.

----

## [999] Conformal PID Control for Time Series Prediction

**Authors**: *Anastasios Angelopoulos, Emmanuel J. Candès, Ryan J. Tibshirani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract-Conference.html)

**Abstract**:

We study the problem of uncertainty quantification for time series prediction, with the goal of providing easy-to-use  algorithms with formal guarantees. The algorithms we present build upon ideas from conformal prediction and control theory, are able to prospectively model conformal scores in an online setting, and adapt to the presence of systematic errors due to seasonality, trends, and general distribution shifts. Our theory both simplifies and strengthens existing analyses in online conformal prediction. Experiments on 4-week-ahead forecasting of statewide COVID-19 death counts in the U.S. show an improvement in coverage over the ensemble forecaster used inofficial CDC communications. We also run experiments on predicting electricity demand, market returns, and temperature using autoregressive, Theta, Prophet, and Transformer models. We provide an extendable codebase for testing our methods and for the integration of new algorithms, data sets, and forecasting rules at this link.

----



[Go to the previous page](NIPS-2023-list04.md)

[Go to the next page](NIPS-2023-list06.md)

[Go to the catalog section](README.md)