## [0] NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks

**Authors**: *Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81e1cdaa570954321d8b06be386cc3d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81e1cdaa570954321d8b06be386cc3d4-Abstract-Conference.html)

**Abstract**:

While multi-exit neural networks are regarded as a promising solution for making efficient inference via early exits, combating adversarial attacks remains a challenging problem. In multi-exit networks, due to the high dependency among different submodels, an adversarial example targeting a specific exit not only degrades the performance of the target exit but also reduces the performance of all other exits concurrently. This makes multi-exit networks highly vulnerable to simple adversarial attacks. In this paper, we propose NEO-KD, a knowledge-distillation-based adversarial training strategy that tackles this fundamental challenge based on two key contributions. NEO-KD first resorts to neighbor knowledge distillation to guide the output of the adversarial examples to tend to the ensemble outputs of neighbor exits of clean data. NEO-KD also employs exit-wise orthogonal knowledge distillation for reducing adversarial transferability across different submodels. The result is a significantly improved robustness against adversarial attacks. Experimental results on various datasets/models show that our method achieves the best adversarial accuracy with reduced computation budgets, compared to the baselines relying on existing adversarial training or knowledge distillation techniques for multi-exit networks.

----

## [0] Self-Evaluation Guided Beam Search for Reasoning

**Authors**: *Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, Michael Qizhe Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html)

**Abstract**:

Breaking down a problem into intermediate steps has demonstrated impressive performance in Large Language Model (LLM) reasoning. However, the growth of the reasoning chain introduces uncertainty and error accumulation, making it challenging to elicit accurate final results. To tackle this challenge of uncertainty in multi-step reasoning, we introduce a stepwise self-evaluation mechanism to guide and calibrate the reasoning process of LLMs. We propose a decoding algorithm integrating the self-evaluation guidance via stochastic beam search. The self-evaluation guidance serves as a better-calibrated automatic criterion, facilitating an efficient search in the reasoning space and resulting in superior prediction quality. Stochastic beam search balances exploitation and exploration of the search space with temperature-controlled randomness. Our approach surpasses the corresponding Codex-backboned baselines in few-shot accuracy by $6.34$%, $9.56$%, and $5.46$% on the GSM8K, AQuA, and StrategyQA benchmarks, respectively. Experiment results with Llama-2 on arithmetic reasoning demonstrate the efficiency of our method in outperforming the baseline methods with comparable computational budgets. Further analysis in multi-step reasoning finds our self-evaluation guidance pinpoints logic failures and leads to higher consistency and robustness. Our code is publicly available at [https://guideddecoding.github.io/](https://guideddecoding.github.io/).

----

## [0] ViSt3D: Video Stylization with 3D CNN

**Authors**: *Ayush Pande, Gaurav Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8203a5156918d467328d5a90147ab307-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8203a5156918d467328d5a90147ab307-Abstract-Conference.html)

**Abstract**:

Visual stylization has been a very popular research area in recent times. While image stylization has seen a rapid advancement in the recent past, video stylization, while being more challenging, is relatively less explored. The immediate method of stylizing videos by stylizing each frame independently has been tried with some success. To the best of our knowledge, we present the first approach to video stylization using 3D CNN directly, building upon insights from 2D image stylization. Stylizing video is highly challenging, as the appearance and video motion, which includes both camera and subject motions, are inherently entangled in the representations learnt by a 3D CNN. Hence, a naive extension of 2D CNN stylization methods to 3D CNN does not work. To perform stylization with 3D CNN, we propose to explicitly disentangle motion and appearance, stylize the appearance part, and then add back the motion component and decode the final stylized video. In addition, we propose a dataset, curated from existing datasets, to train video stylization networks. We also provide an independently collected test set to study the generalization of video stylization methods. We provide results on this test dataset comparing the proposed method with 2D stylization methods applied frame by frame. We show successful stylization with 3D CNN for the first time, and obtain better stylization in terms of texture cf.\ the existing 2D methods.

----

## [0] Smoothed Online Learning for Prediction in Piecewise Affine Systems

**Authors**: *Adam Block, Max Simchowitz, Russ Tedrake*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82096f4f6f897529ecd3eabea603e9cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82096f4f6f897529ecd3eabea603e9cc-Abstract-Conference.html)

**Abstract**:

The problem of piecewise affine (PWA) regression and planning is of foundational importance to the study of online learning, control, and robotics, where it provides a theoretically and empirically tractable setting to study systems undergoing sharp changes in the dynamics.  Unfortunately, due to the discontinuities that arise when crossing into different ``pieces,'' learning in general sequential settings is impossible and practical algorithms are forced to resort to heuristic approaches.  This paper builds on the recently developed smoothed online learning framework and provides the first algorithms for prediction and simulation in PWA systems whose regret is polynomial in all relevant problem parameters under a weak smoothness assumption; moreover, our algorithms are efficient in the number of calls to an optimization oracle.  We further apply our results to the problems of one-step prediction and multi-step simulation regret in piecewise affine dynamical systems, where the learner is tasked with simulating trajectories and regret is measured in terms of the Wasserstein distance between simulated and true data.  Along the way, we develop several technical tools of more general interest.

----

## [0] Adversarial Attacks on Online Learning to Rank with Click Feedback

**Authors**: *Jinhang Zuo, Zhiyao Zhang, Zhiyong Wang, Shuai Li, Mohammad Hajiesmaili, Adam Wierman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/820e42f39773c6cbbd875553db45658f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/820e42f39773c6cbbd875553db45658f-Abstract-Conference.html)

**Abstract**:

Online learning to rank (OLTR) is a sequential decision-making problem where a learning agent selects an ordered list of items and receives feedback through user clicks. Although potential attacks against OLTR algorithms may cause serious losses in real-world applications, there is limited knowledge about adversarial attacks on OLTR. This paper studies attack strategies against multiple variants of OLTR. Our first result provides an attack strategy against the UCB algorithm on classical stochastic bandits with binary feedback, which solves the key issues caused by bounded and discrete feedback that previous works cannot handle. Building on this result, we design attack algorithms against UCB-based OLTR algorithms in position-based and cascade models. Finally, we propose a general attack strategy against any algorithm under the general click model. Each attack algorithm manipulates the learning agent into choosing the target attack item $T-o(T)$ times, incurring a cumulative cost of $o(T)$. Experiments on synthetic and real data further validate the effectiveness of our proposed attack algorithms.

----

## [0] RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths

**Authors**: *Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, Ping Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/821655c7dc4836838cd8524d07f9d6fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/821655c7dc4836838cd8524d07f9d6fd-Abstract-Conference.html)

**Abstract**:

Text-to-image generation has recently witnessed remarkable achievements. We introduce a text-conditional image diffusion model, termed RAPHAEL, to generate highly artistic images, which accurately portray the text prompts, encompassing multiple nouns, adjectives, and verbs. This is achieved by  stacking tens of mixture-of-experts (MoEs) layers, i.e., space-MoE  and time-MoE layers,  enabling billions of diffusion paths (routes) from the network input to the output. Each path intuitively functions as a "painter" for depicting a particular textual concept onto a specified image region at a diffusion timestep. Comprehensive experiments reveal that RAPHAEL outperforms recent cutting-edge models, such as Stable Diffusion, ERNIE-ViLG 2.0, DeepFloyd, and DALL-E 2, in terms of both image quality and aesthetic appeal.  Firstly, RAPHAEL exhibits superior performance in switching images across diverse styles, such as Japanese comics, realism, cyberpunk, and ink illustration. Secondly, a single model with three billion parameters, trained on 1,000 A100 GPUs for two months, achieves a state-of-the-art zero-shot FID score of 6.61 on the COCO dataset. Furthermore, RAPHAEL significantly surpasses its counterparts in human evaluation on the ViLG-300 benchmark. We believe that RAPHAEL holds the potential to propel the frontiers of image generation research in both academia and industry, paving the way for future breakthroughs in this rapidly evolving field. More details can be found on a webpage: https://raphael-painter.github.io/.

----

## [0] Towards Evaluating Transfer-based Attacks Systematically, Practically, and Fairly

**Authors**: *Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/823e43f5537d8c1894afd1f6ab00a927-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/823e43f5537d8c1894afd1f6ab00a927-Abstract-Conference.html)

**Abstract**:

The adversarial vulnerability of deep neural networks (DNNs) has drawn great attention due to the security risk of applying these models in real-world applications. Based on transferability of adversarial examples, an increasing number of transfer-based methods have been developed to fool black-box DNN models whose architecture and parameters are inaccessible. Although tremendous effort has been exerted, there still lacks a standardized benchmark that could be taken advantage of to compare these methods systematically, fairly, and practically. Our investigation shows that the evaluation of some methods needs to be more reasonable and more thorough to verify their effectiveness, to avoid, for example, unfair comparison and insufficient consideration of possible substitute/victim models. Therefore, we establish a transfer-based attack benchmark (TA-Bench) which implements 30+ methods. In this paper, we evaluate and compare them comprehensively on 10 popular substitute/victim models on ImageNet. New insights about the effectiveness of these methods are gained and guidelines for future evaluations are provided.

----

## [0] Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger

**Authors**: *Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, George Karypis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8249b30d877c91611fd8c7aa6ac2b5fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8249b30d877c91611fd8c7aa6ac2b5fe-Abstract-Conference.html)

**Abstract**:

Per-example gradient clipping is a key algorithmic step that enables practical differential private (DP) training for deep learning models. The choice of clipping threshold $R$, however, is vital for achieving high accuracy under DP. We propose an easy-to-use replacement, called automatic clipping, that eliminates the need to tune $R$ for any DP optimizers, including DP-SGD, DP-Adam, DP-LAMB and many others.The automatic variants are as private and computationally efficient as existing DP optimizers, but require no DP-specific hyperparameters and thus make DP training as amenable as the standard non-private training. We give a rigorous convergence analysis of automatic DP-SGD in the non-convex setting, showing that it can enjoy an asymptotic convergence rate that matches the standard SGD, under a symmetric gradient noise assumption of the per-sample gradients (commonly used in the non-DP literature). We demonstrate on various language and vision tasks that automatic clipping outperforms or matches the state-of-the-art, and can be easily employed with minimal changes to existing codebases.

----

## [0] Error Discovery By Clustering Influence Embeddings

**Authors**: *Fulton Wang, Julius Adebayo, Sarah Tan, Diego García-Olano, Narine Kokhlikyan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8278a2e5f9db8489cd908d20c43f1f87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8278a2e5f9db8489cd908d20c43f1f87-Abstract-Conference.html)

**Abstract**:

We present a method for identifying groups of test examples---slices---on which a model under-performs, a task now known as slice discovery. We formalize coherence---a requirement that erroneous predictions, within a slice, should be wrong for the same reason---as a key property that any slice discovery method should satisfy.  We then use influence functions to derive a new slice discovery method, InfEmbed, which satisfies coherence by returning slices whose examples are influenced similarly by the training data.  InfEmbed is simple, and consists of applying K-Means clustering to a novel representation we deem influence embeddings. We show InfEmbed outperforms current state-of-the-art methods on 2 benchmarks, and is effective for model debugging across several case studies.

----

## [0] BadTrack: A Poison-Only Backdoor Attack on Visual Object Tracking

**Authors**: *Bin Huang, Jiaqian Yu, Yiwei Chen, Siyang Pan, Qiang Wang, Zhi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/828bb8f42d4ab15322b9315151959c61-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/828bb8f42d4ab15322b9315151959c61-Abstract-Conference.html)

**Abstract**:

Visual object tracking (VOT) is one of the most fundamental tasks in computer vision community. State-of-the-art VOT trackers extract positive and negative examples that are used to guide the tracker to distinguish the object from the background. In this paper, we show that this characteristic can be exploited to introduce new threats and hence propose a simple yet effective poison-only backdoor attack. To be specific, we poison a small part of the training data by attaching a predefined trigger pattern to the background region of each video frame, so that the trigger appears almost exclusively in the extracted negative examples. To the best of our knowledge, this is the first work that reveals the threat of poison-only backdoor attack on VOT trackers. We experimentally show that our backdoor attack can significantly degrade the performance of both two-stream Siamese and one-stream Transformer trackers on the poisoned data while gaining comparable performance with the benign trackers on the clean data.

----

## [0] Spiking PointNet: Spiking Neural Networks for Point Clouds

**Authors**: *Dayong Ren, Zhe Ma, Yuanpei Chen, Weihang Peng, Xiaode Liu, Yuhan Zhang, Yufei Guo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8296d5800a8e68e58ad0472b393be80e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8296d5800a8e68e58ad0472b393be80e-Abstract-Conference.html)

**Abstract**:

Recently, Spiking Neural Networks (SNNs), enjoying extreme energy efficiency, have drawn much research attention on 2D visual recognition and shown gradually increasing application potential. However, it still remains underexplored whether SNNs can be generalized to 3D recognition. To this end, we present Spiking PointNet in the paper, the first spiking neural model for efficient deep learning on point clouds. We discover that the two huge obstacles limiting the application of SNNs in point clouds are: the intrinsic optimization obstacle of SNNs that impedes the training of a big spiking model with large time steps, and the expensive memory and computation cost of PointNet that makes training a big spiking point model unrealistic. To solve the problems simultaneously, we present a trained-less but learning-more paradigm for Spiking PointNet with theoretical justifications and in-depth experimental analysis. In specific, our Spiking PointNet is trained with only a single time step but can obtain better performance with multiple time steps inference, compared to the one trained directly with multiple time steps. We conduct various experiments on ModelNet10, ModelNet40 to demonstrate the effectiveness of Sipiking PointNet. Notably, our Spiking PointNet even can outperform its ANN counterpart, which is rare in the SNN field thus providing a potential research direction for the following work. Moreover, Spiking PointNet shows impressive speedup and storage saving in the training phase. Our code is open-sourced at https://github.com/DayongRen/Spiking-PointNet.

----

## [0] A Sublinear-Time Spectral Clustering Oracle with Improved Preprocessing Time

**Authors**: *Ranran Shen, Pan Peng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82aec8518602748540a42b783468c94d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82aec8518602748540a42b783468c94d-Abstract-Conference.html)

**Abstract**:

We address the problem of designing a sublinear-time spectral clustering oracle for graphs that exhibit strong clusterability. Such graphs contain $k$ latent clusters, each characterized by a large inner conductance (at least $\varphi$) and a small outer conductance (at most $\varepsilon$). Our aim is to preprocess the graph to enable clustering membership queries, with the key requirement that both preprocessing and query answering should be performed in sublinear time, and the resulting partition should be consistent with a $k$-partition that is close to the ground-truth clustering. Previous oracles have relied on either a $\textrm{poly}(k)\log n$ gap between inner and outer conductances or exponential (in $k/\varepsilon$) preprocessing time. Our algorithm relaxes these assumptions, albeit at the cost of a slightly higher misclassification ratio. We also show that our clustering oracle is robust against a few random edge deletions. To validate our theoretical bounds, we conducted experiments on synthetic networks.

----

## [0] Boosting with Tempered Exponential Measures

**Authors**: *Richard Nock, Ehsan Amid, Manfred K. Warmuth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82d3258eb58ceac31744a88005b7ddef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82d3258eb58ceac31744a88005b7ddef-Abstract-Conference.html)

**Abstract**:

One of the most popular ML algorithms, AdaBoost, can bederived from the dual of a relative entropyminimization problem subject to the fact that the positive weightson the examples sum to one. Essentially, harder examples receive higher probabilities. We generalize this setup to the recently introduced *temperedexponential measure*s (TEMs) where normalization is enforced on a specific power of the measure and not the measure itself.TEMs are indexed by a parameter $t$ and generalize exponential families ($t=1$). Our algorithm, $t$-AdaBoost, recovers AdaBoost as a special case ($t=1$). We show that $t$-AdaBoost retains AdaBoost's celebrated exponential convergence rate when $t\in [0,1)$ while allowing a slight improvement of the rate's hidden constant compared to $t=1$. $t$-AdaBoost partially computes on a generalization of classical arithmetic over the reals and brings notable properties like guaranteed bounded leveraging coefficients for $t\in [0,1)$. From the loss that $t$-AdaBoost minimizes (a generalization of the exponential loss), we show how to derive a new family of *tempered* losses for the induction of domain-partitioning classifiers like decision trees. Crucially, strict properness is ensured for all while their boosting rates span the full known spectrum. Experiments using $t$-AdaBoost+trees display that significant leverage can be achieved by tuning $t$.

----

## [0] DP-HyPO: An Adaptive Private Framework for Hyperparameter Optimization

**Authors**: *Hua Wang, Sheng Gao, Huanyu Zhang, Weijie J. Su, Milan Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82d7d58cba24731c0ca952dff1de46ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82d7d58cba24731c0ca952dff1de46ae-Abstract-Conference.html)

**Abstract**:

Hyperparameter optimization, also known as hyperparameter tuning, is a widely recognized technique for improving model performance. Regrettably, when training private ML models, many practitioners often overlook the privacy risks associated with hyperparameter optimization, which could potentially expose sensitive information about the underlying dataset.Currently, the sole existing approach to allow privacy-preserving hyperparameter optimization is to uniformly and randomly select hyperparameters for a number of runs, subsequently reporting the best-performing hyperparameter.In contrast, in non-private settings, practitioners commonly utilize "adaptive" hyperparameter optimization methods such as Gaussian Process-based optimization, which select the next candidate based on information gathered from previous outputs.This substantial contrast between private and non-private hyperparameter optimization underscores a critical concern. In our paper, we introduce DP-HyPO, a pioneering framework for "adaptive" private hyperparameter optimization, aiming to bridge the gap between private and non-private hyperparameter optimization. To accomplish this, we provide a comprehensive differential privacy analysis of our framework. Furthermore, we empirically demonstrate the effectiveness of DP-HyPO on a diverse set of real-world datasets.

----

## [0] Active Learning-Based Species Range Estimation

**Authors**: *Christian Lange, Elijah Cole, Grant Van Horn, Oisin Mac Aodha*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82eec786fdfbbfa53450c5feb7d1ac92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82eec786fdfbbfa53450c5feb7d1ac92-Abstract-Conference.html)

**Abstract**:

We propose a new active learning approach for efficiently estimating the geographic range of a species from a limited number of on the ground observations. We model the range of an unmapped species of interest as the weighted combination of estimated ranges obtained from a set of different species. We show that it is possible to generate this candidate set of ranges by using models that have been trained on large weakly supervised community collected observation data. From this, we develop a new active querying approach that sequentially selects geographic locations to visit that best reduce our uncertainty over an unmapped speciesâ€™ range. We conduct a detailed evaluation of our approach and compare it to existing active learning methods using an evaluation dataset containing expert-derived ranges for one thousand species. Our results demonstrate that our method outperforms alternative active learning methods and approaches the performance of end-to-end trained models, even when only using a fraction of the data. This highlights the utility of active learning via transfer learned spatial representations for species range estimation. It also emphasizes the value of leveraging emerging large-scale crowdsourced datasets, not only for modeling a species' range, but also for actively discovering them.

----

## [0] One-Step Diffusion Distillation via Deep Equilibrium Models

**Authors**: *Zhengyang Geng, Ashwini Pokle, J. Zico Kolter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82f05a105c928c10706213952bf0c8b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82f05a105c928c10706213952bf0c8b7-Abstract-Conference.html)

**Abstract**:

Diffusion models excel at producing high-quality samples but naively require hundreds of iterations, prompting multiple attempts to distill the generation process into a faster network. However, many existing approaches suffer from a variety of challenges: the process for distillation training can be complex, often requiring multiple training stages, and the resulting models perform poorly when utilized in single-step generative applications. In this paper, we introduce a simple yet effective means of distilling diffusion models *directly* from the initial noise to the resulting image. Of particular importance to our approach is to leverage a new Deep Equilibrium (DEQ) model as the distilled architecture: the Generative Equilibrium Transformer (GET). Our method enables fully offline training with just noise/image pairs from the diffusion model while achieving superior performance compared to existing one-step methods on comparable training budgets. We demonstrate that the DEQ architecture is crucial to this capability, as GET matches a $5\times$ larger ViT in terms of FID scores while striking a critical balance of computational cost and image quality. Code, checkpoints, and datasets are available [here](https://github.com/locuslab/get).

----

## [0] Discrete-Smoothness in Online Algorithms with Predictions

**Authors**: *Yossi Azar, Debmalya Panigrahi, Noam Touitou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82f0dae85424eb743017c90380e7ab9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82f0dae85424eb743017c90380e7ab9b-Abstract-Conference.html)

**Abstract**:

In recent years, there has been an increasing focus on designing online algorithms with (machine-learned) predictions. The ideal learning-augmented algorithm is comparable to the optimum when given perfect predictions (consistency), to the best online approximation for arbitrary predictions (robustness), and should interpolate between these extremes as a smooth function of the prediction error. In this paper, we quantify these guarantees in terms of a general property that we call discrete-smoothness, and achieve discrete-smooth algorithms for online covering, specifically the facility location and set cover problems. For set cover, our work improves the results of Bamas, Maggiori, and Svensson (2020) by augmenting consistency and robustness with smoothness guarantees. For facility location, our work improves on prior work by Almanza et al. (2021) by generalizing to nonuniform costs and also providing smoothness guarantees by augmenting consistency and robustness.

----

## [0] A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning

**Authors**: *Valeriia Cherepanova, Roman Levin, Gowthami Somepalli, Jonas Geiping, C. Bayan Bruss, Andrew Gordon Wilson, Tom Goldstein, Micah Goldblum*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82f39c7409155b74d15d73b048f06771-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/82f39c7409155b74d15d73b048f06771-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Academic tabular benchmarks often contain small sets of curated features. In contrast, data scientists typically collect as many features as possible into their datasets, and even engineer new features from existing ones. To prevent over-fitting in subsequent downstream modeling, practitioners commonly use automated feature selection methods that identify a reduced subset of informative features. Existing benchmarks for tabular feature selection consider classical downstream models, toy synthetic datasets, or do not evaluate feature selectors on the basis of downstream performance. We construct a challenging feature selection benchmark evaluated on downstream neural networks including transformers, using real datasets and multiple methods for generating extraneous features. We also propose an input-gradient-based analogue of LASSO for neural networks that outperforms classical feature selection methods on challenging problems such as selecting from corrupted or second-order features.

----

## [0] Riemannian Projection-free Online Learning

**Authors**: *Zihao Hu, Guanghui Wang, Jacob D. Abernethy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8305a0049227f7dd2bb91e11090f8cfa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8305a0049227f7dd2bb91e11090f8cfa-Abstract-Conference.html)

**Abstract**:

The projection operation is a critical component in a wide range of optimization algorithms, such as online gradient descent (OGD), for enforcing constraints and achieving optimal regret bounds. However, it suffers from computational complexity limitations in high-dimensional settings or when dealing with ill-conditioned constraint sets. Projection-free algorithms address this issue by replacing the projection oracle with more efficient optimization subroutines. But to date, these methods have been developed primarily in the Euclidean setting, and while there has been growing interest in optimization on  Riemannian manifolds, there has been essentially no work in trying to utilize projection-free tools here. An apparent issue is that non-trivial affine functions  are generally non-convex in such domains. In this paper, we present methods for obtaining sub-linear regret guarantees in online geodesically convex optimization  on curved spaces for two scenarios: when we have access to (a) a separation oracle or (b) a linear optimization oracle. For geodesically convex losses, and  when a separation oracle is available, our algorithms achieve $O(T^{\frac{1}{2}})$, $O(T^{\frac{3}{4}})$ and $O(T^{\frac{1}{2}})$ adaptive regret guarantees in the full  information setting, the bandit setting with one-point feedback and the bandit setting with two-point feedback, respectively. When a linear optimization oracle is   available, we obtain regret rates of $O(T^{\frac{3}{4}})$ for geodesically convex losses and $O(T^{\frac{2}{3}}\log T)$ for strongly geodesically convex losses.

----

## [0] SwiFT: Swin 4D fMRI Transformer

**Authors**: *Peter Yongho Kim, Junbeom Kwon, Sunghwan Joo, Sangyoon Bae, Donggyu Lee, Yoonho Jung, Shinjae Yoo, Jiook Cha, Taesup Moon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8313b1920ee9c78d846c5798c1ce48be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8313b1920ee9c78d846c5798c1ce48be-Abstract-Conference.html)

**Abstract**:

Modeling spatiotemporal brain dynamics from high-dimensional data, such as functional Magnetic Resonance Imaging (fMRI), is a formidable task in neuroscience. Existing approaches for fMRI analysis utilize hand-crafted features, but the process of feature extraction risks losing essential information in fMRI scans. To address this challenge, we present SwiFT (Swin 4D fMRI Transformer), a Swin Transformer architecture that can learn brain dynamics directly from fMRI volumes in a memory and computation-efficient manner. SwiFT achieves this by implementing a 4D window multi-head self-attention mechanism and absolute positional embeddings. We evaluate SwiFT using multiple large-scale resting-state fMRI datasets, including the Human Connectome Project (HCP), Adolescent Brain Cognitive Development (ABCD), and UK Biobank (UKB) datasets, to predict sex, age, and cognitive intelligence. Our experimental outcomes reveal that SwiFT consistently outperforms recent state-of-the-art models. Furthermore, by leveraging its end-to-end learning capability, we show that contrastive loss-based self-supervised pre-training of SwiFT can enhance performance on downstream tasks. Additionally, we employ an explainable AI method to identify the brain regions associated with sex classification. To our knowledge, SwiFT is the first Swin Transformer architecture to process dimensional spatiotemporal brain functional data in an end-to-end fashion. Our work holds substantial potential in facilitating scalable learning of functional brain imaging in neuroscience research by reducing the hurdles associated with applying Transformer models to high-dimensional fMRI.

----

## [0] Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent

**Authors**: *Giannis Daras, Yuval Dagan, Alex Dimakis, Constantinos Daskalakis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/831406cfe7e4a0aed5ac5c8a8389d1f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/831406cfe7e4a0aed5ac5c8a8389d1f5-Abstract-Conference.html)

**Abstract**:

Imperfect score-matching leads to a shift between the training and the sampling distribution of diffusion models. Due to the recursive nature of the generation process, errors in previous steps yield sampling iterates that drift away from the training distribution. However, the standard training objective via Denoising Score Matching (DSM) is only designed to optimize over non-drifted data. To train on drifted data, we propose to enforce a \emph{Consistency} property (CP) which states that predictions of the model on its owngenerated data are consistent across time. Theoretically, we show that the differential equation that describes CP together with the one that describes a conservative vector field, have a unique solution given some initial condition. Consequently, if the score is learned well on non-drifted points via DSM (enforcing the true initial condition) then enforcing CP on drifted points propagates true score values. Empirically, we show that enforcing CP improves the generation quality for conditional and unconditional generation on CIFAR-10, and in AFHQ and FFHQ. We open-source our code and models: https://github.com/giannisdaras/cdm.

----

## [0] A High-Resolution Dataset for Instance Detection with Multi-View Object Capture

**Authors**: *Qianqian Shen, Yunhan Zhao, Nahyun Kwon, Jeeeun Kim, Yanan Li, Shu Kong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/832ea0ff01bd512aab28bf416db9489c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/832ea0ff01bd512aab28bf416db9489c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Instance detection (InsDet) is a long-lasting problem in robotics and computer vision, aiming to detect object instances (predefined by some visual examples) in a cluttered scene. Despite its practical significance, its advancement is overshadowed by Object Detection, which aims to detect objects belonging to some predefined classes. One major reason is that current InsDet datasets are too small in scale by today's standards. For example, the popular InsDet dataset GMU (published in 2016) has only 23 instances, far less than  COCO (80 classes), a well-known object detection dataset published in 2014. We are motivated to introduce a new InsDet dataset and protocol. First, we define a realistic setup for InsDet: training data consists of multi-view instance captures, along with diverse scene images allowing synthesizing training images by pasting instance images on them with free box annotations. Second, we release a real-world database, which contains multi-view capture of 100 object instances, and high-resolution (6k$\times$8k) testing images. Third, we extensively study baseline methods for InsDet on our dataset, analyze their performance and suggest future work. Somewhat surprisingly, using the off-the-shelf  class-agnostic segmentation model (Segment Anything Model, SAM) and the self-supervised feature representation DINOv2 performs the best, achieving $>$10 AP better than end-to-end trained InsDet models that repurpose object detectors (e.g., FasterRCNN and RetinaNet).

----

## [0] AVIDa-hIL6: A Large-Scale VHH Dataset Produced from an Immunized Alpaca for Predicting Antigen-Antibody Interactions

**Authors**: *Hirofumi Tsuruta, Hiroyuki Yamazaki, Ryota Maeda, Ryotaro Tamura, Jennifer N. Wei, Zelda E. Mariet, Poomarin Phloyphisut, Hidetoshi Shimokawa, Joseph R. Ledsam, Lucy J. Colwell, Akihiro Imura*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8339dacd9df7ffe9623760f74169dd1e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8339dacd9df7ffe9623760f74169dd1e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Antibodies have become an important class of therapeutic agents to treat human diseases.To accelerate therapeutic antibody discovery, computational methods, especially machine learning, have attracted considerable interest for predicting specific interactions between antibody candidates and target antigens such as viruses and bacteria.However, the publicly available datasets in existing works have notable limitations, such as small sizes and the lack of non-binding samples and exact amino acid sequences.To overcome these limitations, we have developed AVIDa-hIL6, a large-scale dataset for predicting antigen-antibody interactions in the variable domain of heavy chain of heavy chain antibodies (VHHs), produced from an alpaca immunized with the human interleukin-6 (IL-6) protein, as antigens.By leveraging the simple structure of VHHs, which facilitates identification of full-length amino acid sequences by DNA sequencing technology, AVIDa-hIL6 contains 573,891 antigen-VHH pairs with amino acid sequences.All the antigen-VHH pairs have reliable labels for binding or non-binding, as generated by a novel labeling method.Furthermore, via introduction of artificial mutations, AVIDa-hIL6 contains 30 different mutants in addition to wild-type IL-6 protein.This characteristic provides opportunities to develop machine learning models for predicting changes in antibody binding by antigen mutations.We report experimental benchmark results on AVIDa-hIL6 by using machine learning models.The results indicate that the existing models have potential, but further research is needed to generalize them to predict effective antibodies against unknown mutants.The dataset is available at https://avida-hil6.cognanous.com.

----

## [0] Token-Scaled Logit Distillation for Ternary Weight Generative Language Models

**Authors**: *Minsoo Kim, Sihwa Lee, Janghwan Lee, Sukjin Hong, Du-Seong Chang, Wonyong Sung, Jungwook Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8342218a4ec08b8c19661725e9cd6c0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8342218a4ec08b8c19661725e9cd6c0b-Abstract-Conference.html)

**Abstract**:

Generative Language Models (GLMs) have shown impressive performance in tasks such as text generation, understanding, and reasoning. However, the large model size poses challenges for practical deployment. To solve this problem, Quantization-Aware Training (QAT) has become increasingly popular. However, current QAT methods for generative models have resulted in a noticeable loss of accuracy. To counteract this issue, we propose a novel knowledge distillation method specifically designed for GLMs. Our method, called token-scaled logit distillation, prevents overfitting and provides superior learning from the teacher model and ground truth. This research marks the first evaluation of ternary weight quantization-aware training of large-scale GLMs with less than 1.0 degradation in perplexity and achieves enhanced accuracy in tasks like common-sense QA and arithmetic reasoning  as well as natural language understanding. Our code is available at https://github.com/aiha-lab/TSLD.

----

## [0] Efficient Exploration in Continuous-time Model-based Reinforcement Learning

**Authors**: *Lenart Treven, Jonas Hübotter, Bhavya Sukhija, Florian Dörfler, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/836012122f3de08aeeae67369b087964-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/836012122f3de08aeeae67369b087964-Abstract-Conference.html)

**Abstract**:

Reinforcement learning algorithms typically consider discrete-time dynamics, even though the underlying systems are often continuous in time. In this paper, we introduce a model-based reinforcement learning algorithm that represents continuous-time dynamics using nonlinear ordinary differential equations (ODEs). We capture epistemic uncertainty using well-calibrated probabilistic models, and use the optimistic principle for exploration. Our regret bounds surface the importance of the measurement selection strategy (MSS), since in continuous time we not only must decide how to explore, but also when to observe the underlying system. Our analysis demonstrates that the regret is sublinear when modeling ODEs with Gaussian Processes (GP) for common choices of MSS, such as equidistant sampling. Additionally, we propose an adaptive, data-dependent, practical MSS that, when combined with GP dynamics, also achieves sublinear regret with significantly fewer samples. We showcase the benefits of continuous-time modeling over its discrete-time counterpart, as well as our proposed adaptive MSS over standard baselines, on several applications.

----

## [0] On Learning Necessary and Sufficient Causal Graphs

**Authors**: *Hengrui Cai, Yixin Wang, Michael I. Jordan, Rui Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/837b396039248acb08c385bebb6291b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/837b396039248acb08c385bebb6291b4-Abstract-Conference.html)

**Abstract**:

The causal revolution has stimulated interest in understanding complex relationships in various fields. Most of the existing methods aim to discover causal relationships among all variables within a complex large-scale graph. However, in practice, only a small subset of variables in the graph are relevant to the outcomes of interest. Consequently, causal estimation with the full causal graph---particularly given limited data---could lead to numerous falsely discovered, spurious variables that exhibit high correlation with, but exert no causal impact on, the target outcome. In this paper, we propose learning a class of necessary and sufficient causal graphs (NSCG) that exclusively comprises causally relevant variables for an outcome of interest, which we term causal features. The key idea is to employ probabilities of causation to systematically evaluate the importance of features in the causal graph, allowing us to identify a subgraph relevant to the outcome of interest. To learn NSCG from data, we develop a necessary and sufficient causal structural learning (NSCSL) algorithm, by establishing theoretical properties and relationships between probabilities of causation and natural causal effects of features. Across empirical studies of simulated and real data, we demonstrate that NSCSL outperforms existing algorithms and can reveal crucial yeast genes for target heritable traits of interest.

----

## [0] Renku: a platform for sustainable data science

**Authors**: *Rok Roskar, Chandrasekhar Ramakrishnan, Michele Volpi, Fernando Pérez-Cruz, Lilian Gasser, Firat Ozdemir, Patrick Paitz, Mohammad Alisafaee, Philipp Fischer, Ralf Grubenmann, Eliza Harris, Tasko Olevski, Carl Remlinger, Luis Salamanca, Elisabet Capon Garcia, Lorenzo Cavazzi, Jakub Chrobasik, Darlin Cordoba Osnas, Alessandro Degano, Jimena Dupre, Wesley Johnson, Eike Kettner, Laura Kinkead, Sean D. Murphy, Flora Thiebaut, Olivier Verscheure*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/838694e9ab6b0a193b84daaafcac0eed-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/838694e9ab6b0a193b84daaafcac0eed-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Data and code working together is fundamental to machine learning (ML), but the context around datasets and interactions between datasets and code are in general captured only rudimentarily. Context such as how the dataset was prepared and created, what source data were used, what code was used in processing, how the dataset evolved, and where it has been used and reused can provide much insight, but this information is often poorly documented. That is unfortunate since it makes datasets into black-boxes with potentially hidden characteristics that have downstream consequences. We argue that making dataset preparation more accessible and dataset usage easier to record and document would have significant benefits for the ML community: it would allow for greater diversity in datasets by inviting modification to published sources, simplify use of alternative datasets and, in doing so, make results more transparent and robust, while allowing for all contributions to be adequately credited. We present a platform, Renku, designed to support and encourage such sustainable development and use of data, datasets, and code, and we demonstrate its benefits through a few illustrative projects which span the spectrum from dataset creation to dataset consumption and showcasing.

----

## [0] Slimmed Asymmetrical Contrastive Learning and Cross Distillation for Lightweight Model Training

**Authors**: *Jian Meng, Li Yang, Kyungmin Lee, Jinwoo Shin, Deliang Fan, Jae-sun Seo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8393d955a00c463a982cefe77d0404e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8393d955a00c463a982cefe77d0404e1-Abstract-Conference.html)

**Abstract**:

Contrastive learning (CL) has been widely investigated with various learning mechanisms and achieves strong capability in learning representations of data in a self-supervised manner using unlabeled data. A common fashion of contrastive learning on this line is employing mega-sized encoders to achieve comparable performance as the supervised learning counterpart. Despite the success of the labelless training, current contrastive learning algorithms *failed* to achieve good performance with lightweight (compact) models, e.g., MobileNet, while the requirements of the heavy encoders impede the energy-efficient computation, especially for resource-constrained AI applications. Motivated by this, we propose a new self-supervised CL scheme, named SACL-XD, consisting of two technical components, **S**limmed **A**symmetrical **C**ontrastive **L**earning (SACL) and **Cross**-**D**istillation (XD), which collectively enable efficient CL with compact models. While relevant prior works employed a strong pre-trained model as the teacher of unsupervised knowledge distillation to a lightweight encoder, our proposed method trains CL models from scratch and outperforms them even without such an expensive requirement. Compared to the SoTA lightweight CL training (distillation) algorithms, SACL-XD achieves 1.79% ImageNet-1K accuracy improvement on MobileNet-V3 with 64$\times$ training FLOPs reduction.

----

## [0] Beyond Unimodal: Generalising Neural Processes for Multimodal Uncertainty Estimation

**Authors**: *Myong Chol Jung, He Zhao, Joanna Dipnall, Lan Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/839e23e5b1c52cfd1268f4023a3af0d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/839e23e5b1c52cfd1268f4023a3af0d6-Abstract-Conference.html)

**Abstract**:

Uncertainty estimation is an important research area to make deep neural networks (DNNs) more trustworthy. While extensive research on uncertainty estimation has been conducted with unimodal data, uncertainty estimation for multimodal data remains a challenge. Neural processes (NPs) have been demonstrated to be an effective uncertainty estimation method for unimodal data by providing the reliability of Gaussian processes with efficient and powerful DNNs. While NPs hold significant potential for multimodal uncertainty estimation, the adaptation of NPs for multimodal data has not been carefully studied. To bridge this gap, we propose Multimodal Neural Processes (MNPs) by generalising NPs for multimodal uncertainty estimation. Based on the framework of NPs, MNPs consist of several novel and principled mechanisms tailored to the characteristics of multimodal data. In extensive empirical evaluation, our method achieves state-of-the-art multimodal uncertainty estimation performance, showing its appealing robustness against noisy samples and reliability in out-of-distribution detection with faster computation time compared to the current state-of-the-art multimodal uncertainty estimation method.

----

## [0] Trans-Dimensional Generative Modeling via Jump Diffusion Models

**Authors**: *Andrew Campbell, William Harvey, Christian Weilbach, Valentin De Bortoli, Thomas Rainforth, Arnaud Doucet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83a10a480fbec91c88f6a9293b4d2b05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83a10a480fbec91c88f6a9293b4d2b05-Abstract-Conference.html)

**Abstract**:

We propose a new class of generative model that naturally handles data of varying dimensionality by jointly modeling the state and dimension of each datapoint. The generative process is formulated as a jump diffusion process that makes jumps between different dimensional spaces. We first define a dimension destroying forward noising process, before deriving the dimension creating time-reversed generative process along with a novel evidence lower bound training objective for learning to approximate it.Simulating our learned approximation to the time-reversed generative process then provides an effective way of sampling data of varying dimensionality by jointly generating state values and dimensions. We demonstrate our approach on molecular and video datasets of varying dimensionality, reporting better compatibility with test-time diffusion guidance imputation tasks and improved interpolation capabilities versus fixed dimensional models that generate state values and dimensions separately.

----

## [0] Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication

**Authors**: *Chaofei Fan, Nick Hahn, Foram Kamdar, Donald T. Avansino, Guy H. Wilson, Leigh R. Hochberg, Krishna V. Shenoy, Jaimie M. Henderson, Francis R. Willett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83a14a36de4502bac5b580db36e81858-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83a14a36de4502bac5b580db36e81858-Abstract-Conference.html)

**Abstract**:

Intracortical brain-computer interfaces (iBCIs) have shown promise for restoring rapid communication to people with neurological disorders such as amyotrophic lateral sclerosis (ALS). However, to maintain high performance over time, iBCIs typically need frequent recalibration to combat changes in the neural recordings that accrue over days. This requires iBCI users to stop using the iBCI and engage in supervised data collection, making the iBCI system hard to use. In this paper, we propose a method that enables self-recalibration of communication iBCIs without interrupting the user. Our method leverages large language models (LMs) to automatically correct errors in iBCI outputs. The self-recalibration process uses these corrected outputs ("pseudo-labels") to continually update the iBCI decoder online. Over a period of more than one year (403 days), we evaluated our Continual Online Recalibration with Pseudo-labels (CORP) framework with one clinical trial participant. CORP achieved  a stable decoding accuracy of 93.84% in an online handwriting iBCI task, significantly outperforming other baseline methods. Notably, this is the longest-running iBCI stability demonstration involving a human participant. Our results provide the first  evidence for long-term stabilization of a plug-and-play, high-performance communication iBCI, addressing a major barrier for the clinical translation of iBCIs.

----

## [0] Towards robust and generalizable representations of extracellular data using contrastive learning

**Authors**: *Ankit Vishnubhotla, Charlotte Loh, Akash Srivastava, Liam Paninski, Cole L. Hurwitz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83c637c3bc0ca88eda6cf4f5f45bdced-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83c637c3bc0ca88eda6cf4f5f45bdced-Abstract-Conference.html)

**Abstract**:

Contrastive learning is quickly becoming an essential tool in neuroscience for extracting robust and meaningful representations of neural activity. Despite numerous applications to neuronal population data, there has been little exploration of how these methods can be adapted to key primary data analysis tasks such as spike sorting or cell-type classification. In this work, we propose a novel contrastive learning framework, CEED (Contrastive Embeddings for Extracellular Data), for high-density extracellular recordings. We demonstrate that through careful design of the network architecture and data augmentations, it is possible to generically extract representations that far outperform current specialized approaches. We validate our method across multiple high-density extracellular recordings. All code used to run CEED can be found at https://github.com/ankitvishnu23/CEED.

----

## [0] Rethinking Conditional Diffusion Sampling with Progressive Guidance

**Authors**: *Anh-Dung Dinh, Daochang Liu, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83ca9e252329e7b0704ead93893e6b1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83ca9e252329e7b0704ead93893e6b1b-Abstract-Conference.html)

**Abstract**:

This paper tackles two critical challenges encountered in classifier guidance for diffusion generative models, i.e., the lack of diversity and the presence of adversarial effects. These issues often result in a scarcity of diverse samples or the generation of non-robust features. The underlying cause lies in the mechanism of classifier guidance, where discriminative gradients push samples to be recognized as conditions aggressively. This inadvertently suppresses information with common features among relevant classes, resulting in a limited pool of features with less diversity or the absence of robust features for image construction.We propose a generalized classifier guidance method called Progressive Guidance, which mitigates the problems by allowing relevant classes' gradients to contribute to shared information construction when the image is noisy in early sampling steps. In the later sampling stage, we progressively enhance gradients to refine the details in the image toward the primary condition. This helps to attain a high level of diversity and robustness compared to the vanilla classifier guidance. Experimental results demonstrate that our proposed method further improves the image quality while offering a significant level of diversity as well as robust features.

----

## [0] State-Action Similarity-Based Representations for Off-Policy Evaluation

**Authors**: *Brahma S. Pavse, Josiah Hanna*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83dc5747870ea454cab25e30bef4eb8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83dc5747870ea454cab25e30bef4eb8a-Abstract-Conference.html)

**Abstract**:

In reinforcement learning, off-policy evaluation (OPE) is the problem of estimating the expected return of an evaluation policy given a fixed dataset that was collected by running one or more different policies. One of the more empirically successful algorithms for OPE has been the fitted q-evaluation (FQE) algorithm that uses temporal difference updates to learn an action-value function, which is then used to estimate the expected return of the evaluation policy. Typically, the original fixed dataset is fed directly into FQE to learn the action-value function of the evaluation policy. Instead, in this paper, we seek to enhance the data-efficiency of FQE by first transforming the fixed dataset using a learned encoder, and then feeding the transformed dataset into FQE. To learn such an encoder, we introduce an OPE-tailored state-action behavioral similarity metric, and use this metric and the fixed dataset to  learn an encoder that models this metric. Theoretically, we show that this metric allows us to bound the error in the resulting OPE estimate. Empirically, we show that other state-action similarity metrics lead to representations that cannot represent the action-value function of the evaluation policy, and that our state-action representation method boosts the data-efficiency of FQE and lowers OPE error relative to other OPE-based representation learning methods on challenging OPE tasks. We also empirically show that the learned representations significantly mitigate divergence of FQE under varying distribution shifts. Our code is available here: https://github.com/Badger-RL/ROPE.

----

## [0] Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs

**Authors**: *Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin Chen-Chuan Chang, Fei Huang, Reynold Cheng, Yongbin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83fc8fab1710363050bbd1d4b8cc0021-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/83fc8fab1710363050bbd1d4b8cc0021-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Text-to-SQL parsing, which aims at converting natural language instructions into executable SQLs, has gained increasing attention in recent years.  In particular, GPT-4 and Claude-2 have shown impressive results in this task. However, most of the prevalent benchmarks, i.e., Spider, and WikiSQL, focus on database schema with few rows of database contents leaving the gap between academic study and real-world applications. To mitigate this gap, we present BIRD, a BIg benchmark for laRge-scale Database grounded in text-to-SQL tasks, containing 12,751 pairs of text-to-SQL data and 95 databases with a total size of 33.4 GB, spanning 37 professional domains. Our emphasis on database values highlights the new challenges of dirty database contents, external knowledge between NL questions and database contents,  and SQL efficiency, particularly in the context of massive databases. To solve these problems, text-to-SQL models must feature database value comprehension in addition to semantic parsing. The experimental results demonstrate the significance of database values in generating accurate text-to-SQLs for big databases. Furthermore, even the most popular and effective text-to-SQL models, i.e. GPT-4, only achieve 54.89% in execution accuracy, which is still far from the human result of 92.96%, proving that challenges still stand. We also provide an efficiency analysis to offer insights into generating text-to-efficient-SQLs that are beneficial to industries. We believe that BIRD will contribute to advancing real-world applications of text-to-SQL research.The leaderboard and source code are available: https://bird-bench.github.io/.

----

## [0] CARE-MI: Chinese Benchmark for Misinformation Evaluation in Maternity and Infant Care

**Authors**: *Tong Xiang, Liangzhi Li, Wangyue Li, Mingbai Bai, Lu Wei, Bowen Wang, Noa Garcia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84062fe53d23e0791c6dbb456783e4a9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/84062fe53d23e0791c6dbb456783e4a9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The recent advances in natural language processing (NLP), have led to a new trend of applying large language models (LLMs) to real-world scenarios. While the latest LLMs are astonishingly fluent when interacting with humans, they suffer from the misinformation problem by unintentionally generating factually false statements. This can lead to harmful consequences, especially when produced within sensitive contexts, such as healthcare. Yet few previous works have focused on evaluating misinformation in the long-form (LF) generation of LLMs, especially for knowledge-intensive topics. Moreover, although LLMs have been shown to perform well in different languages, misinformation evaluation has been mostly conducted in English. To this end, we present a benchmark, CARE-MI, for evaluating LLM misinformation in: 1) a sensitive topic, specifically the maternity and infant care domain; and 2) a language other than English, namely Chinese. Most importantly, we provide an innovative paradigm for building LF generation evaluation benchmarks that can be transferred to other knowledge-intensive domains and low-resourced languages. Our proposed benchmark fills the gap between the extensive usage of LLMs and the lack of datasets for assessing the misinformation generated by these models. It contains 1,612 expert-checked questions, accompanied with human-selected references. Using our benchmark, we conduct extensive experiments and found that current Chinese LLMs are far from perfect in the topic of maternity and infant care. In an effort to minimize the reliance on human resources for performance evaluation, we offer off-the-shelf judgment models for automatically assessing the LF output of LLMs given benchmark questions. Moreover, we compare potential solutions for LF generation evaluation and provide insights for building better automated metrics.

----

## [0] Explore In-Context Learning for 3D Point Cloud Understanding

**Authors**: *Zhongbin Fang, Xiangtai Li, Xia Li, Joachim M. Buhmann, Chen Change Loy, Mengyuan Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8407d254b5baacf69ee977aa34f0e521-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8407d254b5baacf69ee977aa34f0e521-Abstract-Conference.html)

**Abstract**:

With the rise of large-scale models trained on broad data, in-context learning has become a new learning paradigm that has demonstrated significant potential in natural language processing and computer vision tasks. Meanwhile, in-context learning is still largely unexplored in the 3D point cloud domain. Although masked modeling has been successfully applied for in-context learning in 2D vision, directly extending it to 3D point clouds remains a formidable challenge. In the case of point clouds, the tokens themselves are the point cloud positions (coordinates) that are masked during inference. Moreover, position embedding in previous works may inadvertently introduce information leakage. To address these challenges, we introduce a novel framework, named Point-In-Context, designed especially for in-context learning in 3D point clouds, where both inputs and outputs are modeled as coordinates for each task. Additionally, we propose the Joint Sampling module, carefully designed to work in tandem with the general point sampling operator, effectively resolving the aforementioned technical issues. We conduct extensive experiments to validate the versatility and adaptability of our proposed methods in handling a wide range of tasks. Furthermore, with a more effective prompt selection strategy, our framework surpasses the results of individually trained models.

----

## [0] Learning Re-sampling Methods with Parameter Attribution for Image Super-resolution

**Authors**: *Xiaotong Luo, Yuan Xie, Yanyun Qu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8434e0db3227276c00ef2b18c7f01c65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8434e0db3227276c00ef2b18c7f01c65-Abstract-Conference.html)

**Abstract**:

Single image super-resolution (SISR) has made a significant breakthrough benefiting from the prevalent rise of deep neural networks and large-scale training samples. The mainstream deep SR models primarily focus on network architecture design as well as optimization schemes, while few pay attention to the training data. In fact, most of the existing SR methods train the model on uniformly sampled patch pairs from the whole image. However, the uneven image content makes the training data present an unbalanced distribution, i.e., the easily reconstructed region (smooth) occupies the majority of the data, while the hard reconstructed region (edge or texture) has rarely few samples. Based on this phenomenon, we consider rethinking the current paradigm of merely using uniform data sampling way for training SR models. In this paper, we propose a simple yet effective Bi-Sampling Parameter Attribution (BSPA) method for accurate image SR. Specifically, the bi-sampling consists of uniform sampling and inverse sampling, which is introduced to reconcile the unbalanced inherent data bias. The former aims to keep the intrinsic data distribution, and the latter is designed to enhance the feature extraction ability of the model on the hard samples. Moreover, integrated gradient is introduced to attribute the contribution of each parameter in the alternate models trained by both sampling data so as to filter the trivial parameters for further dynamic refinement. By progressively decoupling the allocation of parameters, the SR model can learn a more compact representation. Extensive experiments on publicly available datasets demonstrate that our proposal can effectively boost the performance of baseline methods from the data re-sampling view.

----

## [0] Interactive Visual Reasoning under Uncertainty

**Authors**: *Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/844f722dbbcb27933ff5baf58a1f00c8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/844f722dbbcb27933ff5baf58a1f00c8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

One of the fundamental cognitive abilities of humans is to quickly resolve uncertainty by generating hypotheses and testing them via active trials. Encountering a novel phenomenon accompanied by ambiguous cause-effect relationships, humans make hypotheses against data, conduct inferences from observation, test their theory via experimentation, and correct the proposition if inconsistency arises. These iterative processes persist until the underlying mechanism becomes clear. In this work, we devise the  IVRE (pronounced as "ivory") environment for evaluating artificial agents' reasoning ability under uncertainty. IVRE is an interactive environment featuring rich scenarios centered around Blicket detection. Agents in IVRE are placed into environments with various ambiguous action-effect pairs and asked to determine each object's role. They are encouraged to propose effective and efficient experiments to validate their hypotheses based on observations and actively gather new information. The game ends when all uncertainties are resolved or the maximum number of trials is consumed. By evaluating modern artificial agents in IVRE, we notice a clear failure of today's learning methods compared to humans. Such inefficacy in interactive reasoning ability under uncertainty calls for future research in building human-like intelligence.

----

## [0] Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport

**Authors**: *Jaemoo Choi, Jaewoong Choi, Myungjoo Kang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84706cdfc192cd0351daf48f379847e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/84706cdfc192cd0351daf48f379847e6-Abstract-Conference.html)

**Abstract**:

Optimal Transport (OT) problem investigates a transport map that bridges two distributions while minimizing a given cost function. In this regard, OT between tractable prior distribution and data has been utilized for generative modeling tasks. However, OT-based methods are susceptible to outliers and face optimization challenges during training. In this paper, we propose a novel generative model based on the semi-dual formulation of Unbalanced Optimal Transport (UOT). Unlike OT, UOT relaxes the hard constraint on distribution matching. This approach provides better robustness against outliers, stability during training, and faster convergence. We validate these properties empirically through experiments. Moreover, we study the theoretical upper-bound of divergence between distributions in UOT. Our model outperforms existing OT-based generative models, achieving FID scores of 2.97 on CIFAR-10 and 6.36 on CelebA-HQ-256. The code is available at \url{https://github.com/Jae-Moo/UOTM}.

----

## [0] Generalized Belief Transport

**Authors**: *Junqi Wang, Pei Wang, Patrick Shafto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/847bb9bb1351f557a52d2ecdacb7e2d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/847bb9bb1351f557a52d2ecdacb7e2d3-Abstract-Conference.html)

**Abstract**:

Human learners have ability to adopt appropriate learning approaches depending on constraints such as prior on the hypothesis, urgency of decision, and drift of the environment. However, existing learning models are typically considered individually rather than in relation to one and other. To build agents that have the ability to move between different modes of learning over time, it is important to understand how learning models are related as points in a broader space of possibilities. We introduce a mathematical framework, Generalized Belief Transport (GBT), that unifies and generalizes prior models, including Bayesian inference, cooperative communication and classification, as parameterizations of three learning constraints within Unbalanced Optimal Transport (UOT). We visualize the space of learning models encoded by GBT as a cube which includes classic learning models as special points. We derive critical properties of this parameterized space including proving continuity and differentiability which is the basis for model interpolation, and study limiting behavior of the parameters, which allows attaching learning models on the boundaries. Moreover, we investigate the long-run behavior of GBT, explore convergence properties of models in GBT mathematical and computationally, document the ability to learn in the presence of distribution drift, and formulate conjectures about general behavior. We conclude with open questions and implications for more unified models of learning.

----

## [0] Lie Point Symmetry and Physics-Informed Networks

**Authors**: *Tara Akhound-Sadegh, Laurence Perreault Levasseur, Johannes Brandstetter, Max Welling, Siamak Ravanbakhsh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8493c860bec41705f7743d5764301b94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8493c860bec41705f7743d5764301b94-Abstract-Conference.html)

**Abstract**:

Symmetries have been leveraged to improve the generalization of neural networks through different mechanisms from data augmentation to equivariant architectures. However, despite their potential, their integration into neural solvers for partial differential equations (PDEs) remains largely unexplored. We explore the integration of PDE symmetries, known as Lie point symmetries, in a major family of neural solvers known as physics-informed neural networks (PINNs). We propose a loss function that informs the network about Lie point symmetries in the same way that PINN models try to enforce the underlying PDE through a loss function. Intuitively, our symmetry loss ensures that the infinitesimal generators of the Lie group conserve the PDE solutions.. Effectively, this means that once the network learns a solution, it also learns the neighbouring solutions generated by Lie point symmetries.Empirical evaluations indicate that the inductive bias introduced by the Lie point symmetries of the PDEs greatly boosts the sample efficiency of PINNs.

----

## [0] Norm-based Generalization Bounds for Sparse Neural Networks

**Authors**: *Tomer Galanti, Mengjia Xu, Liane Galanti, Tomaso A. Poggio*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8493e190ff1bbe3837eca821190b61ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8493e190ff1bbe3837eca821190b61ff-Abstract-Conference.html)

**Abstract**:

In this paper, we derive norm-based generalization bounds for sparse ReLU neural networks, including convolutional neural networks. These bounds differ from previous ones because they consider the sparse structure of the neural network architecture and the norms of the convolutional filters, rather than the norms of the (Toeplitz) matrices associated with the convolutional layers. Theoretically, we demonstrate that these bounds are significantly tighter than standard norm-based generalization bounds. Empirically, they offer relatively tight estimations of generalization for various simple classification problems. Collectively, these findings suggest that the sparsity of the underlying target function and the model's architecture plays a crucial role in the success of deep learning.

----

## [0] Intelligent Knee Sleeves: A Real-time Multimodal Dataset for 3D Lower Body Motion Estimation Using Smart Textile

**Authors**: *Wenwen Zhang, Arvin Tashakori, Zenan Jiang, Amir Servati, Harishkumar Narayana, Saeid Soltanian, Rou Yi Yeap, Meng Han Ma, Lauren Toy, Peyman Servati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84948f178cfd3f6a0ffecda8fdcb3488-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/84948f178cfd3f6a0ffecda8fdcb3488-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The kinematics of human movements and locomotion are closely linked to the activation and contractions of muscles. To investigate this, we present a multimodal dataset with benchmarks collected using a novel pair of Intelligent Knee Sleeves (Texavie MarsWear Knee Sleeves) for human pose estimation. Our system utilizes synchronized datasets that comprise time-series data from the Knee Sleeves and the corresponding ground truth labels from visualized motion capture camera system. We employ these to generate 3D human models solely based on the wearable data of individuals performing different activities. We demonstrate the effectiveness of this camera-free system and machine learning algorithms in the assessment of various movements and exercises, including extension to unseen exercises and individuals. The results show an average error of 7.21 degrees across all eight lower body joints when compared to the ground truth, indicating the effectiveness and reliability of the Knee Sleeve system for the prediction of different lower body joints beyond knees. The results enable human pose estimation in a seamless manner without being limited by visual occlusion or the field of view of cameras. Our results show the potential of multimodal wearable sensing in a variety of applications from home fitness to sports, healthcare, and physical rehabilitation focusing on pose and movement estimation.

----

## [0] Neural Injective Functions for Multisets, Measures and Graphs via a Finite Witness Theorem

**Authors**: *Tal Amir, Steven J. Gortler, Ilai Avni, Ravina Ravina, Nadav Dym*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84b686f7cc7b7751e9aaac0da74f755a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/84b686f7cc7b7751e9aaac0da74f755a-Abstract-Conference.html)

**Abstract**:

Injective multiset functions have a key role in the theoretical study of machine learning on multisets and graphs. Yet, there remains a gap between the provably injective multiset functions considered in theory, which typically rely on polynomial moments, and the multiset functions used in practice, which rely on $\textit{neural moments}$ â€” whose injectivity on multisets has not been studied to date.In this paper, we bridge this gap by showing that moments of neural networks do define injective multiset functions,  provided that an analytic non-polynomial activation is used. The number of moments required by our theory is optimal essentially up to a multiplicative factor of two. To prove this result, we state and prove a $\textit{finite witness theorem}$, which is of independent interest. As a corollary to our main theorem, we derive new approximation results for functions on multisets and measures, and new separation results for graph neural networks. We also provide two negative results: (1) moments of piecewise-linear neural networks cannot be injective multiset functions; and (2) even when moment-based multiset functions are injective, they can never be bi-Lipschitz.

----

## [0] Online Ad Procurement in Non-stationary Autobidding Worlds

**Authors**: *Jason Cheuk Nam Liang, Haihao Lu, Baoyu Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html)

**Abstract**:

Today's online advertisers procure digital ad impressions through interacting with autobidding platforms: advertisers convey high level procurement goals via setting levers such as budget, target return-on-investment, max cost per click, etc.. Then ads platforms subsequently procure impressions on advertisers' behalf, and report final procurement conversions (e.g. click) to advertisers. In practice, advertisers may receive minimal information on platforms' procurement details, and procurement outcomes  are subject to non-stationary factors like seasonal patterns, occasional system corruptions, and market trends which make it difficult for advertisers to optimize lever decisions effectively. Motivated by this,  we present an online learning framework that helps advertisers dynamically optimize ad platform lever decisions while subject to general long-term constraints in a realistic bandit feedback environment with non-stationary procurement outcomes. In particular, we introduce a primal-dual algorithm for online decision making with multi-dimension decision variables, bandit feedback and long-term uncertain constraints. We show that our algorithm achieves low regret in many worlds when procurement outcomes are generated through procedures that are stochastic, adversarial, adversarially corrupted, periodic, and ergodic, respectively, without having to know which procedure is the ground truth. Finally, we emphasize that our proposed algorithm and theoretical results extend beyond the applications of online advertising.

----

## [0] Precise asymptotic generalization for multiclass classification with overparameterized linear models

**Authors**: *David X. Wu, Anant Sahai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84f44b36ceb4fbc9bb269959f4796eed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/84f44b36ceb4fbc9bb269959f4796eed-Abstract-Conference.html)

**Abstract**:

We study the asymptotic generalization of an overparameterized linear model for multiclass classification under the Gaussian covariates bi-level model introduced in Subramanian et al. (NeurIPS'22), where the number of data points, features, and classes all grow together. We fully resolve the conjecture posed in Subramanian et al. '22, matching the predicted regimes for which the model does and does not generalize. Furthermore, our new lower bounds are akin to an information-theoretic strong converse: they establish that the misclassification rate goes to 0 or 1 asymptotically. One surprising consequence of our tight results is that the min-norm interpolating classifier can be asymptotically suboptimal relative to noninterpolating classifiers in the regime where the min-norm interpolating regressor is known to be optimal. The key to our tight analysis is a new variant of the Hanson-Wright inequality which is broadly useful for multiclass problems with sparse labels. As an application, we show that the same type of analysis can be used to analyze the related multi-label classification problem under the same bi-level ensemble.

----

## [0] Break It Down: Evidence for Structural Compositionality in Neural Networks

**Authors**: *Michael A. Lepori, Thomas Serre, Ellie Pavlick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85069585133c4c168c865e65d72e9775-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85069585133c4c168c865e65d72e9775-Abstract-Conference.html)

**Abstract**:

Though modern neural networks have achieved impressive performance in both vision and language tasks, we know little about the functions that they implement. One possibility is that neural networks implicitly break down complex tasks into subroutines, implement modular solutions to these subroutines, and compose them into an overall solution to a task --- a property we term structural compositionality.  Another possibility is that they may simply learn to match new inputs to learned templates, eliding task decomposition entirely. Here, we leverage model pruning techniques to investigate this question in both vision and language across a variety of architectures, tasks, and pretraining regimens. Our results demonstrate that models oftentimes implement solutions to subroutines via modular subnetworks, which can be ablated while maintaining the functionality of other subnetworks. This suggests that neural networks may be able to learn compositionality, obviating the need for specialized symbolic mechanisms.

----

## [0] Focused Transformer: Contrastive Training for Context Scaling

**Authors**: *Szymon Tworkowski, Konrad Staniszewski, Mikolaj Pacek, Yuhuai Wu, Henryk Michalewski, Piotr Milos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8511d06d5590f4bda24d42087802cc81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8511d06d5590f4bda24d42087802cc81-Abstract-Conference.html)

**Abstract**:

Large language models have an exceptional capability to incorporate new information in a contextual manner. However, the full potential of such an approach is often restrained due to a limitation in the effective context length. One solution to this issue is to endow an attention layer with access to an additional context, which comprises of (key, value) pairs. Yet, as the number of documents increases, the proportion of relevant keys to irrelevant ones decreases, leading the model to focus more on the irrelevant keys. We identify a significant challenge, dubbed the distraction issue, where keys linked to different semantic values might overlap, making them hard to distinguish. To tackle this problem, we introduce the Focused Transformer (FoT), a technique that employs a training process inspired by contrastive learning. This novel approach enhances the structure of the (key, value) space, enabling an extension of the context length. Our method allows for fine-tuning pre-existing, large-scale models to lengthen their effective context. This is demonstrated by our fine-tuning of $3 B$ and $7 B$ OpenLLaMA checkpoints. The resulting models, which we name LongLLaMA, exhibit advancements in tasks requiring a long context. We further illustrate that our LongLLaMA models adeptly manage a $256 k$ context length for passkey retrieval.

----

## [0] Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone

**Authors**: *Zeyinzi Jiang, Chaojie Mao, Ziyuan Huang, Ao Ma, Yiliang Lv, Yujun Shen, Deli Zhao, Jingren Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8514a5203b87cba5e440bd62ab18f2b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8514a5203b87cba5e440bd62ab18f2b4-Abstract-Conference.html)

**Abstract**:

Parameter-efficient tuning has become a trend in transferring large-scale foundation models to downstream applications. Existing methods typically embed some light-weight tuners into the backbone, where both the design and the learning of the tuners are highly dependent on the base model. This work offers a new tuning paradigm, dubbed Res-Tuning, which intentionally unbinds tuners from the backbone. With both theoretical and empirical evidence, we show that popular tuning approaches have their equivalent counterparts under our unbinding formulation, and hence can be integrated into our framework effortlessly. Thanks to the structural disentanglement, we manage to free the design of tuners from the network architecture, facilitating flexible combination of various tuning strategies. We further propose a memory-efficient variant of Res-Tuning, where the bypass i.e., formed by a sequence of tuners) is effectively detached from the main branch, such that the gradients are back-propagated only to the tuners but not to the backbone. Such a detachment also allows one-time backbone forward for multi-task inference. Extensive experiments on both discriminative and generative tasks demonstrate the superiority of our method over existing alternatives from the perspectives of efficacy and efficiency. Project page: https://res-tuning.github.io/.

----

## [0] Learning Energy-Based Prior Model with Diffusion-Amortized MCMC

**Authors**: *Peiyu Yu, Yaxuan Zhu, Sirui Xie, Xiaojian Ma, Ruiqi Gao, Song-Chun Zhu, Ying Nian Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85381f4549b5ddf1d48e2e287d7d3d15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85381f4549b5ddf1d48e2e287d7d3d15-Abstract-Conference.html)

**Abstract**:

Latent space EBMs, also known as energy-based priors, have drawn growing interests in the field of generative modeling due to its flexibility in the formulation and strong modeling power of the latent space. However, the common practice of learning latent space EBMs with non-convergent short-run MCMC for prior and posterior sampling is hindering the model from further progress; the degenerate MCMC sampling quality in practice often leads to degraded generation quality and instability in training, especially with highly multi-modal and/or high-dimensional target distributions. To remedy this sampling issue, in this paper we introduce a simple but effective diffusion-based amortization method for long-run MCMC sampling and develop a novel learning algorithm for the latent space EBM based on it. We provide theoretical evidence that the learned amortization of MCMC is a valid long-run MCMC sampler. Experiments on several image modeling benchmark datasets demonstrate the superior performance of our method compared with strong counterparts.

----

## [0] Perception Test: A Diagnostic Benchmark for Multimodal Video Models

**Authors**: *Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adrià Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fréchette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, João Carreira*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8540fba4abdc7f9f7a7b1cc6cd60e409-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8540fba4abdc7f9f7a7b1cc6cd60e409-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We propose a novel multimodal video benchmark - the Perception Test - to evaluate the perception and reasoning skills of pre-trained multimodal models (e.g. Flamingo, BEiT-3, or GPT-4). Compared to existing benchmarks that focus on computational tasks (e.g. classification, detection or tracking), the Perception Test focuses on skills (Memory, Abstraction, Physics, Semantics) and types of reasoning (descriptive, explanatory, predictive, counterfactual) across video, audio, and text modalities, to provide a comprehensive and efficient evaluation tool. The benchmark probes pre-trained models for their transfer capabilities, in a zero-shot / few-shot or limited finetuning regime. For these purposes, the Perception Test introduces 11.6k real-world videos, 23s average length, designed to show perceptually interesting situations, filmed by around 100 participants worldwide. The videos are densely annotated with six types of labels (multiple-choice and grounded video question-answers, object and point tracks, temporal action and sound segments), enabling both language and non-language evaluations. The fine-tuning and validation splits of the benchmark are publicly available (CC-BY license), in addition to a challenge server with a held-out test split. Human baseline results compared to state-of-the-art video QA models show a significant gap in performance (91.4% vs 45.8%), suggesting that there is significant room for improvement in multimodal video understanding.Dataset, baselines code, and challenge server are available at https://github.com/deepmind/perception_test

----

## [0] Directed Cyclic Graph for Causal Discovery from Multivariate Functional Data

**Authors**: *Saptarshi Roy, Raymond K. W. Wong, Yang Ni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/854a9ab0f323b841955e70ca383b27d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/854a9ab0f323b841955e70ca383b27d1-Abstract-Conference.html)

**Abstract**:

Discovering causal relationship using multivariate functional data has received a significant amount of attention very recently. In this article, we introduce a functional linear structural equation model for causal structure learning when the underlying graph involving the multivariate functions may have cycles. To enhance interpretability, our model involves a low-dimensional causal embedded space such that all the relevant causal information in the multivariate functional data is preserved in this lower-dimensional subspace. We prove that the proposed model is causally identifiable under standard assumptions that are often made in the causal discovery literature. To carry out inference of our model, we develop a fully Bayesian framework with suitable prior specifications and uncertainty quantification through posterior summaries. We illustrate the superior performance of our method over existing methods in terms of causal graph estimation through extensive simulation studies. We also demonstrate the proposed method using a brain EEG dataset.

----

## [0] Do SSL Models Have Déjà Vu? A Case of Unintended Memorization in Self-supervised Learning

**Authors**: *Casey Meehan, Florian Bordes, Pascal Vincent, Kamalika Chaudhuri, Chuan Guo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/854b6ec839294bf332db0d86e2f83c3f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/854b6ec839294bf332db0d86e2f83c3f-Abstract-Conference.html)

**Abstract**:

Self-supervised learning (SSL) algorithms can produce useful image representations by learning to associate different parts of natural images with one another. However, when taken to the extreme, SSL models can unintendedly memorize specific parts in individual training samples rather than learning semantically meaningful associations. In this work, we perform a systematic study of the unintended memorization of image-specific information in SSL models -- which we refer to as déjà vu memorization. Concretely, we show that given the trained model and a crop of a training image containing only the background (e.g., water, sky, grass), it is possible to infer the foreground object with high accuracy or even visually reconstruct it. Furthermore, we show that déjà vu memorization is common to different SSL algorithms, is exacerbated by certain design choices, and cannot be detected by conventional techniques for evaluating representation quality. Our study of déjà vu memorization reveals previously unknown privacy risks in SSL models, as well as suggests potential practical mitigation strategies.

----

## [0] Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes

**Authors**: *Ali Younis, Erik B. Sudderth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract-Conference.html)

**Abstract**:

Particle filters flexibly represent multiple posterior modes nonparametrically, via a collection of weighted samples, but have classically been applied to tracking problems with known dynamics and observation likelihoods.  Such generative models may be inaccurate or unavailable for high-dimensional observations like images.  We instead leverage training data to discriminatively learn particle-based representations of uncertainty in latent object states, conditioned on arbitrary observations via deep neural network encoders.   While prior discriminative particle filters have used heuristic relaxations of discrete particle resampling, or biased learning by truncating gradients at resampling steps, we achieve unbiased and low-variance gradient estimates by representing posteriors as continuous mixture densities.  Our theory and experiments expose dramatic failures of existing reparameterization-based estimators for mixture gradients, an issue we address via an importance-sampling gradient estimator. Unlike standard recurrent neural networks, our mixture density particle filter represents multimodal uncertainty in continuous latent states, improving accuracy and robustness.  On a range of challenging tracking and robot localization problems, our approach achieves dramatic improvements in accuracy, will also showing much greater stability across multiple training runs.

----

## [0] Benchmarking Robustness to Adversarial Image Obfuscations

**Authors**: *Florian Stimberg, Ayan Chakrabarti, Chun-Ta Lu, Hussein Hazimeh, Otilia Stretcu, Wei Qiao, Yintao Liu, Merve Kaya, Cyrus Rashtchian, Ariel Fuxman, Mehmet Tek, Sven Gowal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85c123f6da0fa159eb249e6a2e171903-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/85c123f6da0fa159eb249e6a2e171903-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Automated content filtering and moderation is an important tool that allows online platforms to build striving user communities that facilitate cooperation and prevent abuse. Unfortunately, resourceful actors try to bypass automated filters in a bid to post content that violate platform policies and codes of conduct. To reach this goal, these malicious actors may obfuscate policy violating images (e.g., overlay harmful images by carefully selected benign images or visual patterns) to prevent machine learning models from reaching the correct decision. In this paper, we invite researchers to tackle this specific issue and present a new image benchmark. This benchmark, based on ImageNet, simulates the type of obfuscations created by malicious actors. It goes beyond Image-Net-C and ImageNet-C-bar by proposing general, drastic, adversarial modifications that preserve the original content intent. It aims to tackle a more common adversarial threat than the one considered by lp-norm bounded adversaries. We evaluate 33 pretrained models on the benchmark and train models with different augmentations, architectures and training methods on subsets of the obfuscations to measure generalization. Our hope is that this benchmark will encourage researchers to test their models and methods and try to find new approaches that are more robust to these obfuscations.

----

## [0] Learning Curves for Deep Structured Gaussian Feature Models

**Authors**: *Jacob A. Zavatone-Veth, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85d456fd41f3eec83bd3b0c337037a0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85d456fd41f3eec83bd3b0c337037a0e-Abstract-Conference.html)

**Abstract**:

In recent years, significant attention in deep learning theory has been devoted to analyzing when models that interpolate their training data can still generalize well to unseen examples. Many insights have been gained from studying models with multiple layers of Gaussian random features, for which one can compute precise generalization asymptotics. However, few works have considered the effect of weight anisotropy; most assume that the random features are generated using independent and identically distributed Gaussian weights, and allow only for structure in the input data. Here, we use the replica trick from statistical physics to derive learning curves for models with many layers of structured Gaussian features. We show that allowing correlations between the rows of the first layer of features can aid generalization, while structure in later layers is generally detrimental. Our results shed light on how weight structure affects generalization in a simple class of solvable models.

----

## [0] Mirror Diffusion Models for Constrained and Watermarked Generation

**Authors**: *Guan-Horng Liu, Tianrong Chen, Evangelos A. Theodorou, Molei Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85f5c7372625d1e0df0e3996f85062d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85f5c7372625d1e0df0e3996f85062d6-Abstract-Conference.html)

**Abstract**:

Modern successes of diffusion models in learning complex, high-dimensional data distributions are attributed, in part, to their capability to construct diffusion processes with analytic transition kernels and score functions. The tractability results in a simulation-free framework with stable regression losses, from which reversed, generative processes can be learned at scale. However, when data is confined to a constrained set as opposed to a standard Euclidean space, these desirable characteristics appear to be lost based on prior attempts. In this work, we propose Mirror Diffusion Models (MDM), a new class of diffusion models that generate data on convex constrained sets without losing any tractability. This is achieved by learning diffusion processes in a dual space constructed from a mirror map, which, crucially, is a standard Euclidean space. We derive efficient computation of mirror maps for popular constrained sets, such as simplices and $\ell_2$-balls, showing significantly improved performance of MDM over existing methods. For safety and privacy purposes, we also explore constrained sets as a new mechanism to embed invisible but quantitative information (i.e., watermarks) in generated data, for which MDM serves as a compelling approach. Our work brings new algorithmic opportunities for learning tractable diffusion on complex domains.

----

## [0] Training Transitive and Commutative Multimodal Transformers with LoReTTa

**Authors**: *Manuel Tran, Yashin Dicente Cid, Amal Lahiani, Fabian J. Theis, Tingying Peng, Eldad Klaiman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/860a092bb4d9d81d3133a01c50c01578-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/860a092bb4d9d81d3133a01c50c01578-Abstract-Conference.html)

**Abstract**:

Training multimodal foundation models is challenging due to the limited availability of multimodal datasets. While many public datasets pair images with text, few combine images with audio or text with audio. Even rarer are datasets that align all three modalities at once. Critical domains such as healthcare, infrastructure, or transportation are particularly affected by missing modalities. This makes it difficult to integrate all modalities into a large pre-trained neural network that can be used out-of-the-box or fine-tuned for different downstream tasks. We introduce LoReTTa ($\textbf{L}$inking m$\textbf{O}$dalities with a t$\textbf{R}$ansitive and commutativ$\textbf{E}$ pre-$\textbf{T}$raining s$\textbf{T}$r$\textbf{A}$tegy) to address this understudied problem. Our self-supervised framework unifies causal modeling and masked modeling with the rules of commutativity and transitivity. This allows us to transition within and between modalities. As a result, our pre-trained models are better at exploring the true underlying joint probability distribution. Given a dataset containing only the disjoint combinations $(A, B)$ and $(B, C)$, LoReTTa can model the relation $A \leftrightarrow C$ with $A \leftrightarrow B \leftrightarrow C$. In particular, we show that a transformer pre-trained with LoReTTa can handle any mixture of modalities at inference time, including the never-seen pair $(A, C)$ and the triplet $(A, B, C)$. We extensively evaluate our approach on a synthetic, medical, and reinforcement learning dataset. Across different domains, our universal multimodal transformer consistently outperforms strong baselines such as GPT, BERT, and CLIP on tasks involving the missing modality tuple.

----

## [0] SaVeNet: A Scalable Vector Network for Enhanced Molecular Representation Learning

**Authors**: *Sarp Aykent, Tian Xia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/860c1c657deafe09f64c013c2888bd7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/860c1c657deafe09f64c013c2888bd7b-Abstract-Conference.html)

**Abstract**:

Geometric representation learning of molecules is challenging yet essential for applications in multiple domains. Despite the impressive breakthroughs made by geometric deep learning in various molecular representation learning tasks, effectively capturing complicated geometric features across spatial dimensions is still underexplored due to the significant difficulties in modeling efficient geometric representations and learning the inherent correlation in 3D structural modeling. These include computational inefficiency, underutilization of vectorial embeddings, and limited generalizability to integrate various geometric properties. To address the raised concerns, we introduce an efficient and effective framework, Scalable Vector Network (SaVeNet), designed to accommodate a range of geometric requirements without depending on costly embeddings. In addition, the proposed framework scales effectively with introduced direction noise. Theoretically, we analyze the desired properties (i.e., invariance and equivariant) and framework efficiency of the SaVeNet. Empirically, we conduct a comprehensive series of experiments to evaluate the efficiency and expressiveness of the proposed model. Our efficiency-focused experiments underscore the model's empirical superiority over existing methods. Experimental results on synthetic and real-world datasets demonstrate the expressiveness of our model, which achieves state-of-the-art performance across various tasks within molecular representation learning.

----

## [0] Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense

**Authors**: *Zunzhi You, Daochang Liu, Bohyung Han, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8629b0fff229b8a27efb1422e990605f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8629b0fff229b8a27efb1422e990605f-Abstract-Conference.html)

**Abstract**:

Recent advancements in masked image modeling (MIM) have made it a prevailing framework for self-supervised visual representation learning. The MIM pretrained models, like most deep neural network methods, remain vulnerable to adversarial attacks, limiting their practical application, and this issue has received little research attention. In this paper, we investigate how this powerful self-supervised learning paradigm can provide adversarial robustness to downstream classifiers. During the exploration, we find that noisy image modeling (NIM), a simple variant of MIM that adopts denoising as the pre-text task, reconstructs noisy images surprisingly well despite severe corruption. Motivated by this observation, we propose an adversarial defense method, referred to as De^3, by exploiting the pretrained decoder for denoising. Through De^3, NIM is able to enhance adversarial robustness beyond providing pretrained features. Furthermore, we incorporate a simple modification, sampling the noise scale hyperparameter from random distributions, and enable the defense to achieve a better and tunable trade-off between accuracy and robustness. Experimental results demonstrate that, in terms of adversarial robustness, NIM is superior to MIM thanks to its effective denoising capability. Moreover, the defense provided by NIM achieves performance on par with adversarial training while offering the extra tunability advantage. Source code and models are available at https://github.com/youzunzhi/NIM-AdvDef.

----

## [0] UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild

**Authors**: *Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, Ran Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/862f45ccecb2275851bc8acebb8b4d65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/862f45ccecb2275851bc8acebb8b4d65-Abstract-Conference.html)

**Abstract**:

Achieving machine autonomy and human control often represent divergent objectives in the design of interactive AI systems. Visual generative foundation models such as Stable Diffusion show promise in navigating these goals, especially when prompted with arbitrary languages. However, they often fall short in generating images with spatial, structural, or geometric controls. The integration of such controls, which can accommodate various visual conditions in a single unified model, remains an unaddressed challenge. In response, we introduce UniControl, a new generative foundation model that consolidates a wide array of controllable condition-to-image (C2I) tasks within a singular framework, while still allowing for arbitrary language prompts. UniControl enables pixel-level-precise image generation, where visual conditions primarily influence the generated structures and language prompts guide the style and context. To equip UniControl with the capacity to handle diverse visual conditions, we augment pretrained text-to-image diffusion models and introduce a task-aware HyperNet to modulate the diffusion models, enabling the adaptation to different C2I tasks simultaneously. Trained on nine unique C2I tasks, UniControl demonstrates impressive zero-shot generation abilities with unseen visual conditions. Experimental results show that UniControl often surpasses the performance of single-task-controlled methods of comparable model sizes. This control versatility positions UniControl as a significant advancement in the realm of controllable visual generation.

----

## [0] Unlocking Deterministic Robustness Certification on ImageNet

**Authors**: *Kai Hu, Andy Zou, Zifan Wang, Klas Leino, Matt Fredrikson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/863da9d40547f1d1b18859519ce2dee4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/863da9d40547f1d1b18859519ce2dee4-Abstract-Conference.html)

**Abstract**:

Despite the promise of Lipschitz-based methods for provably-robust deep learning with deterministic guarantees, current state-of-the-art results are limited to feed-forward Convolutional Networks (ConvNets) on low-dimensional data, such as CIFAR-10. This paper investigates strategies for expanding certifiably robust training to larger, deeper models.A key challenge in certifying deep networks is efficient calculation of the Lipschitz bound for residual blocks found in ResNet and ViT architectures.We show that fast ways of bounding the Lipschitz constant for conventional ResNets are loose, and show how to address this by designing a new residual block, leading to the *Linear ResNet* (LiResNet) architecture.We then introduce *Efficient Margin MAximization* (EMMA), a loss function that stabilizes robust training by penalizing worst-case adversarial examples from multiple classes simultaneously.Together, these contributions yield new *state-of-the-art* robust accuracy on CIFAR-10/100 and Tiny-ImageNet under $\ell_2$ perturbations.Moreover, for the first time, we are able to scale up fast deterministic robustness guarantees to ImageNet, demonstrating that this approach to robust learning can be applied to real-world applications.

----

## [0] Bayesian Optimisation of Functions on Graphs

**Authors**: *Xingchen Wan, Pierre Osselin, Henry Kenlay, Binxin Ru, Michael A. Osborne, Xiaowen Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86419aba4e5eafd2b1009a2e3c540bb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86419aba4e5eafd2b1009a2e3c540bb0-Abstract-Conference.html)

**Abstract**:

The increasing availability of graph-structured data motivates the task of optimising over functions defined on the node set of graphs. Traditional graph search algorithms can be applied in this case, but they may be sample-inefficient and do not make use of information about the function values; on the other hand, Bayesian optimisation is a class of promising black-box solvers with superior sample efficiency, but it has scarcely been applied to such novel setups. To fill this gap, we propose a novel Bayesian optimisation framework that optimises over functions defined on generic, large-scale and potentially unknown graphs. Through the learning of suitable kernels on graphs, our framework has the advantage of adapting to the behaviour of the target function. The local modelling approach further guarantees the efficiency of our method. Extensive experiments on both synthetic and real-world graphs demonstrate the effectiveness of the proposed optimisation framework.

----

## [0] RIO: A Benchmark for Reasoning Intention-Oriented Objects in Open Environments

**Authors**: *Mengxue Qu, Yu Wu, Wu Liu, Xiaodan Liang, Jingkuan Song, Yao Zhao, Yunchao Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Intention-oriented object detection aims to detect desired objects based on specific intentions or requirements. For instance, when we desire to "lie down and rest", we instinctively seek out a suitable option such as a "bed" or a "sofa" that can fulfill our needs. Previous work in this area is limited either by the number of intention descriptions or by the affordance vocabulary available for intention objects. These limitations make it challenging to handle intentions in open environments effectively. To facilitate this research, we construct a comprehensive dataset called Reasoning Intention-Oriented Objects (RIO). In particular, RIO is specifically designed to incorporate diverse real-world scenarios and a wide range of object categories. It offers the following key features: 1) intention descriptions in RIO are represented as natural sentences rather than a mere word or verb phrase, making them more practical and meaningful; 2) the intention descriptions are contextually relevant to the scene, enabling a broader range of potential functionalities associated with the objects; 3) the dataset comprises a total of 40,214 images and 130,585 intention-object pairs. With the proposed RIO, we evaluate the ability of some existing models to reason intention-oriented objects in open environments.

----

## [0] Supervised Pretraining Can Learn In-Context Reinforcement Learning

**Authors**: *Jonathan Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, Emma Brunskill*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8644b61a9bc87bf7844750a015feb600-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8644b61a9bc87bf7844750a015feb600-Abstract-Conference.html)

**Abstract**:

Large transformer models trained on diverse datasets have shown a remarkable ability to learn in-context, achieving high few-shot performance on tasks they were not explicitly trained to solve. In this paper, we study the in-context learning capabilities of transformers in decision-making problems, i.e., reinforcement learning (RL) for bandits and Markov decision processes. To do so, we introduce and study the Decision-Pretrained Transformer (DPT), a supervised pretraining method where a transformer predicts an optimal action given a query state and an in-context dataset of interactions from a diverse set of tasks. While simple, this procedure produces a model with several surprising capabilities. We find that the trained transformer can solve a range of RL problems in-context, exhibiting both exploration online and conservatism offline, despite not being explicitly trained to do so. The model also generalizes beyond the pretraining distribution to new tasks and automatically adapts its decision-making strategies to unknown structure. Theoretically, we show DPT can be viewed as an efficient implementation of Bayesian posterior sampling, a provably sample-efficient RL algorithm. We further leverage this connection to provide guarantees on the regret of the in-context algorithm yielded by DPT, and prove that it can learn faster than algorithms used to generate the pretraining data. These results suggest a promising yet simple path towards instilling strong in-context decision-making abilities in transformers.

----

## [0] L2T-DLN: Learning to Teach with Dynamic Loss Network

**Authors**: *Zhaoyang Hai, Liyuan Pan, Xiabi Liu, Zhengzheng Liu, Mirna Yunita*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8667f264f88c7938a73a53ab01eb1327-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8667f264f88c7938a73a53ab01eb1327-Abstract-Conference.html)

**Abstract**:

With the concept of teaching being introduced to the machine learning community, a teacher model start using dynamic loss functions to teach the training of a student model. The dynamic intends to set adaptive loss functions to different phases of student model learning. In existing works, the teacher model 1) merely determines the loss function based on the present states of the student model, e.g., disregards the experience of the teacher; 2) only utilizes the states of the student model, e.g., training iteration number and loss/accuracy from training/validation sets, while ignoring the states of the loss function. In this paper, we first formulate the loss adjustment as a temporal task by designing a teacher model with memory units, and, therefore, enables the student learning to be guided by the experience of the teacher model. Then, with a Dynamic Loss Network, we can additionally use the states of the loss to assist the teacher learning in enhancing the interactions between the teacher and the student model.   Extensive experiments demonstrate our approach can enhance student learning and improve the performance of various deep models on real-world tasks, including classification, objective detection, and semantic segmentation scenario.

----

## [0] Structured Federated Learning through Clustered Additive Modeling

**Authors**: *Jie Ma, Tianyi Zhou, Guodong Long, Jing Jiang, Chengqi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8668fdc7b2ddf55a0e235824c66f2eee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8668fdc7b2ddf55a0e235824c66f2eee-Abstract-Conference.html)

**Abstract**:

Heterogeneous federated learning without assuming any structure is challenging due to the conflicts among non-identical data distributions of clients. In practice, clients often comprise near-homogeneous clusters so training a server-side model per cluster mitigates the conflicts. However, FL with client clustering often suffers from “clustering collapse'', i.e., one cluster's model excels on increasing clients, and reduces to single-model FL. Moreover, cluster-wise models hinder knowledge sharing between clusters and each model depends on fewer clients. Furthermore, the static clustering assumption on data may not hold for dynamically changing models, which are sensitive to cluster imbalance/initialization or outliers. To address these challenges, we propose ''Clustered Additive Modeling  (CAM)'', which applies a globally shared model $\Theta_g$ on top of the cluster-wise models $\Theta_{1:K}$, i.e., $y=h(x;\Theta_g)+f(x;\Theta_k)$ for clients of cluster-$k$. The global model captures the features shared by all clusters so $\Theta_{1:K}$ are enforced to focus on the difference among clusters. To train CAM, we develop a novel Fed-CAM algorithm that alternates between client clustering and training global/cluster models to predict the residual of each other.  We can easily modify any existing clustered FL methods by CAM and significantly improve their performance without ‘’clustering collapse'' in different non-IID settings. We also provide a convergence analysis of Fed-CAM algorithm.

----

## [0] An Inductive Bias for Tabular Deep Learning

**Authors**: *Ege Beyazit, Jonathan Kozaczuk, Bo Li, Vanessa Wallace, Bilal Fadlallah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8671b6dffc08b4fcf5b8ce26799b2bef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8671b6dffc08b4fcf5b8ce26799b2bef-Abstract-Conference.html)

**Abstract**:

Deep learning methods have achieved state-of-the-art performance in most modeling tasks involving images, text and audio, however, they typically underperform tree-based methods on tabular data. In this paper, we hypothesize that a significant contributor to this performance gap is the interaction between irregular target functions resulting from the heterogeneous nature of tabular feature spaces, and the well-known tendency of neural networks to learn smooth functions. Utilizing tools from spectral analysis, we show that functions described by tabular datasets often have high irregularity, and that they can be smoothed by transformations such as scaling and ranking in order to improve performance. However, because these transformations tend to lose information or negatively impact the loss landscape during optimization, they need to be rigorously fine-tuned for each feature to achieve performance gains. To address these problems, we propose introducing frequency reduction as an inductive bias. We realize this bias as a neural network layer that promotes learning low-frequency representations of the input features, allowing the network to operate in a space where the target function is more regular. Our proposed method introduces less computational complexity than a fully connected layer, while significantly improving neural network performance, and speeding up its convergence on 14 tabular datasets.

----

## [0] Fairness-guided Few-shot Prompting for Large Language Models

**Authors**: *Huan Ma, Changqing Zhang, Yatao Bian, Lemao Liu, Zhirui Zhang, Peilin Zhao, Shu Zhang, Huazhu Fu, Qinghua Hu, Bingzhe Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8678da90126aa58326b2fc0254b33a8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8678da90126aa58326b2fc0254b33a8c-Abstract-Conference.html)

**Abstract**:

Large language models have demonstrated surprising ability to perform in-context learning, i.e., these models can be directly applied to solve numerous downstream tasks by conditioning on a prompt constructed by a few input-output examples. However, prior research has shown that in-context learning can suffer from high instability due to variations in training examples, example order, and prompt formats. Therefore, the construction of an appropriate prompt is essential for improving the performance of in-context learning. In this paper,  we revisit this problem from the view of predictive bias. Specifically, we introduce a metric to evaluate the predictive bias of a fixed prompt against labels or a given attributes. Then we empirically show that prompts with higher bias always lead to unsatisfactory predictive quality. Based on this observation, we propose a novel search strategy based on the greedy search to identify the near-optimal prompt for improving the performance of in-context learning. We perform comprehensive experiments with state-of-the-art mainstream models such as GPT-3 on various downstream tasks. Our results indicate that our method can enhance the model's in-context learning performance in an effective and interpretable manner.

----

## [0] Efficient Robust Bayesian Optimization for Arbitrary Uncertain inputs

**Authors**: *Lin Yang, Junlong Lyu, Wenlong Lyu, Zhitang Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/868f2f9a9950f7b0538b3ce7eb4c8eb8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/868f2f9a9950f7b0538b3ce7eb4c8eb8-Abstract-Conference.html)

**Abstract**:

Bayesian Optimization (BO) is a sample-efficient optimization algorithm widely employed across various applications. In some challenging BO tasks, input uncertainty arises due to the inevitable randomness in the optimization process, such as machining errors, execution noise, or contextual variability. This uncertainty deviates the input from the intended value before evaluation, resulting in significant performance fluctuations in the final result. In this paper, we introduce a novel robust Bayesian Optimization algorithm, AIRBO, which can effectively identify a robust optimum that performs consistently well under arbitrary input uncertainty. Our method directly models the uncertain inputs of arbitrary distributions by empowering the Gaussian Process with the Maximum Mean Discrepancy (MMD) and further accelerates the posterior inference via Nystrom approximation. Rigorous theoretical regret bound is established under MMD estimation error and extensive experiments on synthetic functions and real problems demonstrate that our approach can handle various input uncertainties and achieve a state-of-the-art performance.

----

## [0] HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution

**Authors**: *Eric Nguyen, Michael Poli, Marjan Faizi, Armin W. Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton M. Rabideau, Yoshua Bengio, Stefano Ermon, Christopher Ré, Stephen Baccus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html)

**Abstract**:

Genomic (DNA) sequences encode an enormous amount of information for gene regulation and protein synthesis. Similar to natural language models, researchers have proposed foundation models in genomics to learn generalizable features from unlabeled genome data that can then be fine-tuned for downstream tasks such as identifying regulatory elements. Due to the quadratic scaling of attention, previous Transformer-based genomic models have used 512 to 4k tokens as context (<0.001% of the human genome), significantly limiting the modeling of long-range interactions in DNA. In addition, these methods rely on tokenizers or fixed k-mers to aggregate meaningful DNA units, losing single nucleotide resolution (i.e. DNA "characters") where subtle genetic variations can completely alter protein function via single nucleotide polymorphisms (SNPs). Recently, Hyena, a large language model based on implicit convolutions was shown to match attention in quality while allowing longer context lengths and lower time complexity. Leveraging Hyena’s new long-range capabilities, we present HyenaDNA, a genomic foundation model pretrained on the human reference genome with context lengths of up to 1 million tokens at the single nucleotide-level – an up to 500x increase over previous dense attention-based models. HyenaDNA scales sub-quadratically in sequence length (training up to 160x faster than Transformer), uses single nucleotide tokens, and has full global context at each layer. We explore what longer context enables - including the first use of in-context learning in genomics for simple adaptation to novel tasks without updating pretrained model weights. On fine-tuned benchmarks from the Nucleotide Transformer, HyenaDNA reaches state-of-the-art (SotA) on 12 of 18 datasets using a model with orders of magnitude less parameters and pretraining data.1 On the GenomicBenchmarks, HyenaDNA surpasses SotA on 7 of 8 datasets on average by +10 accuracy points. Code at https://github.com/HazyResearch/hyena-dna.

----

## [0] Learning a 1-layer conditional generative model in total variation

**Authors**: *Ajil Jalal, Justin Singh Kang, Ananya Uppal, Kannan Ramchandran, Eric Price*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86b8ad667206fb9a52ae575fbf1cd6be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86b8ad667206fb9a52ae575fbf1cd6be-Abstract-Conference.html)

**Abstract**:

A conditional generative model is a method for sampling from a conditional distribution $p(y \mid x)$.  For example, one may want to sample an image of a cat given the label ``cat''.  A feed-forward conditional generative model is a function $g(x, z)$ that takes the input $x$ and a random seed $z$, and outputs a sample $y$ from $p(y \mid x)$.  Ideally the distribution of outputs $(x, g(x, z))$ would be close in total variation to the ideal distribution $(x, y)$.Generalization bounds for other learning models require assumptions on the distribution of $x$, even in simple settings like linear regression with Gaussian noise. We show these assumptions are unnecessary in our model, for both linear regression and single-layer ReLU networks.  Given samples $(x, y)$, we show how to learn a 1-layer ReLU conditional generative model in total variation. As our result has no assumption on the distribution of inputs $x$, if we are given access to the internal activations of a deep generative model, we can compose our 1-layer guarantee to progressively learn the deep model using a near-linear number of samples.

----

## [0] Model Shapley: Equitable Model Valuation with Black-box Access

**Authors**: *Xinyi Xu, Thanh Lam, Chuan Sheng Foo, Bryan Kian Hsiang Low*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86bcae6da75c72e32f30a5553f094c06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86bcae6da75c72e32f30a5553f094c06-Abstract-Conference.html)

**Abstract**:

Valuation methods of data and machine learning (ML) models are essential to the establishment of AI marketplaces. Importantly, certain practical considerations (e.g., operational constraints, legal restrictions) favor the use of model valuation over data valuation. Also, existing marketplaces that involve trading of pre-trained ML models call for an equitable model valuation method to price them. In particular, we investigate the black-box access setting which allows querying a model (to observe predictions) without disclosing model-specific information (e.g., architecture and parameters). By exploiting a Dirichlet abstraction of a model’s predictions, we propose a novel and equitable model valuation method called model Shapley. We also leverage a Lipschitz continuity of model Shapley to design a learning approach for predicting the model Shapley values (MSVs) of many vendors’ models (e.g., 150) in a large-scale marketplace. We perform extensive empirical validation on the effectiveness of model Shapley using various real-world datasets and heterogeneous model types.

----

## [0] Robust Concept Erasure via Kernelized Rate-Distortion Maximization

**Authors**: *Somnath Basu Roy Chowdhury, Nicholas Monath, Kumar Avinava Dubey, Amr Ahmed, Snigdha Chaturvedi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86bd650f85480c595ecab29081a3774e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86bd650f85480c595ecab29081a3774e-Abstract-Conference.html)

**Abstract**:

Distributed representations provide a vector space that captures meaningful relationships between data instances. The distributed nature of these representations, however, entangles together multiple attributes or concepts of data instances (e.g., the topic or sentiment of a text, characteristics of the author (age, gender, etc), etc).  Recent work has proposed the task of concept erasure, in which rather than making a concept predictable, the goal is to remove an attribute from distributed representations while retaining other information from the original representation space as much as possible.  In this paper, we propose a new distance metric learning-based objective, the Kernelized Rate-Distortion Maximizer (KRaM), for performing concept erasure. KRaM fits a transformation of representations to match a specified distance measure (defined by a labeled concept to erase) using a modified rate-distortion function. Specifically, KRaM's objective function aims to make instances with similar concept labels dissimilar in the learned representation space while retaining other information.  We find that optimizing KRaM effectively erases various types of concepts—categorical, continuous, and vector-valued variables—from data representations across diverse domains. We also provide a theoretical analysis of several properties of KRaM's objective. To assess the quality of the learned representations, we propose an alignment score to evaluate their similarity with the original representation space. Additionally, we conduct experiments to showcase KRaM's efficacy in various settings, from erasing binary gender variables in word embeddings to vector-valued variables in GPT-3 representations.

----

## [0] BiMatting: Efficient Video Matting via Binarization

**Authors**: *Haotong Qin, Lei Ke, Xudong Ma, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, Xianglong Liu, Fisher Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86c070ce724102ee876d1935590e111a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86c070ce724102ee876d1935590e111a-Abstract-Conference.html)

**Abstract**:

Real-time video matting on edge devices faces significant computational resource constraints, limiting the widespread use of video matting in applications such as online conferences and short-form video production. Binarization is a powerful compression approach that greatly reduces computation and memory consumption by using 1-bit parameters and bitwise operations. However, binarization of the video matting model is not a straightforward process, and our empirical analysis has revealed two primary bottlenecks: severe representation degradation of the encoder and massive redundant computations of the decoder. To address these issues, we propose BiMatting, an accurate and efficient video matting model using binarization. Specifically, we construct shrinkable and dense topologies of the binarized encoder block to enhance the extracted representation. We sparsify the binarized units to reduce the low-information decoding computation. Through extensive experiments, we demonstrate that BiMatting outperforms other binarized video matting models, including state-of-the-art (SOTA) binarization methods, by a significant margin. Our approach even performs comparably to the full-precision counterpart in visual quality. Furthermore, BiMatting achieves remarkable savings of 12.4$\times$ and 21.6$\times$ in computation and storage, respectively, showcasing its potential and advantages in real-world resource-constrained scenarios. Our code and models are released at https://github.com/htqin/BiMatting .

----

## [0] One Fits All: Power General Time Series Analysis by Pretrained LM

**Authors**: *Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html)

**Abstract**:

Although we have witnessed great success of pre-trained models in natural language processing (NLP) and computer vision (CV), limited progress has been made for general time series analysis. Unlike NLP and CV where a unified model can be used to perform different tasks, specially designed approach still dominates in each time series analysis task such as classification, anomaly detection, forecasting, and few-shot learning. The main challenge that blocks the development of pre-trained model for time series analysis is the lack of a large amount of data for training. In this work, we address this challenge by leveraging language or CV models, pre-trained from billions of tokens, for time series analysis. Specifically, we refrain from altering the self-attention and feedforward layers of the residual blocks in the pre-trained language or image model. This model, known as the Frozen Pretrained Transformer (FPT), is evaluated through fine-tuning on all major types of tasks involving time series. Our results demonstrate that pre-trained models on natural language or images can lead to a comparable or state-of-the-art performance in all main time series analysis tasks, as illustrated in Figure1. We also found both theoretically and empirically that the self-attention module behaviors similarly to principle component analysis (PCA), an observation that helps explains how transformer bridges the domain gap and a crucial step towards understanding the universality of a pre-trained transformer. The code is publicly available at https://anonymous.4open.science/r/Pretrained-LM-for-TSForcasting-C561.

----

## [0] Parameterizing Non-Parametric Meta-Reinforcement Learning Tasks via Subtask Decomposition

**Authors**: *Suyoung Lee, Myungsik Cho, Youngchul Sung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86c1fd74fa25bd6be0072937803e0bd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86c1fd74fa25bd6be0072937803e0bd1-Abstract-Conference.html)

**Abstract**:

Meta-reinforcement learning (meta-RL) techniques have demonstrated remarkable success in generalizing deep reinforcement learning across a range of tasks. Nevertheless, these methods often struggle to generalize beyond tasks with parametric variations. To overcome this challenge, we propose Subtask Decomposition and Virtual Training (SDVT), a novel meta-RL approach that decomposes each non-parametric task into a collection of elementary subtasks and parameterizes the task based on its decomposition. We employ a  Gaussian mixture VAE to meta-learn the decomposition process, enabling the agent to reuse policies acquired from common subtasks. Additionally, we propose a virtual training procedure, specifically designed for non-parametric task variability, which generates hypothetical subtask compositions, thereby enhancing generalization to previously unseen subtask compositions. Our method significantly improves performance on the Meta-World ML-10 and ML-45 benchmarks, surpassing current state-of-the-art techniques.

----

## [0] Near-Optimal Algorithms for Gaussians with Huber Contamination: Mean Estimation and Linear Regression

**Authors**: *Ilias Diakonikolas, Daniel Kane, Ankit Pensia, Thanasis Pittas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86c283920335ed1fec3edee227e05fbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86c283920335ed1fec3edee227e05fbf-Abstract-Conference.html)

**Abstract**:

We study the fundamental problems of Gaussian mean estimation and linear regression with Gaussian covariates in the presence of Huber contamination. Our main contribution is the design of the first sample near-optimal and almost linear-time algorithms with optimal error guarantees for both these problems. Specifically, for Gaussian robust mean estimation on $\mathbb R^d$ with contamination parameter $\epsilon \in (0, \epsilon_0)$ for a small absolute constant $\epsilon_0$, we give an algorithm with sample complexity $n = \tilde{O}(d/\epsilon^2)$ and almost linear runtime that approximates the target mean within $\ell_2$-error $O(\epsilon)$. This improves on prior work that achieved this error guarantee with polynomially suboptimal sample and time complexity. For robust linear regression, we give the first algorithm with sample complexity $n = \tilde{O}(d/\epsilon^2)$ and almost linear runtime that approximates the target regressor within $\ell_2$-error $O(\epsilon)$. This is the first polynomial sample and time algorithm achieving the optimal error guarantee, answering an open question in the literature. At the technical level, we develop a methodology that yields almost-linear time algorithms for multi-directional filtering that may be of broader interest.

----

## [0] Causal Discovery from Subsampled Time Series with Proxy Variables

**Authors**: *Mingzhou Liu, Xinwei Sun, Lingjing Hu, Yizhou Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86cba2b31d17f237866a2e6c52c7878a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86cba2b31d17f237866a2e6c52c7878a-Abstract-Conference.html)

**Abstract**:

Inferring causal structures from time series data is the central interest of many scientific inquiries. A major barrier to such inference is the problem of subsampling, i.e., the frequency of measurement is much lower than that of causal influence. To overcome this problem, numerous methods have been proposed, yet either was limited to the linear case or failed to achieve identifiability. In this paper, we propose a constraint-based algorithm that can identify the entire causal structure from subsampled time series, without any parametric constraint. Our observation is that the challenge of subsampling arises mainly from hidden variables at the unobserved time steps. Meanwhile, every hidden variable has an observed proxy, which is essentially itself at some observable time in the future, benefiting from the temporal structure. Based on these, we can leverage the proxies to remove the bias induced by the hidden variables and hence achieve identifiability. Following this intuition, we propose a proxy-based causal discovery algorithm. Our algorithm is nonparametric and can achieve full causal identification. Theoretical advantages are reflected in synthetic and real-world experiments.

----

## [0] "Why Not Looking backward?" A Robust Two-Step Method to Automatically Terminate Bayesian Optimization

**Authors**: *Shuang Li, Ke Li, Wei Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/870c1e0589822bf37590b84984c345c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/870c1e0589822bf37590b84984c345c4-Abstract-Conference.html)

**Abstract**:

Bayesian Optimization (BO) is a powerful method for tackling expensive black-box optimization problems. As a sequential model-based optimization strategy, BO iteratively explores promising solutions until a predetermined budget, either iterations or time, is exhausted. The decision on when to terminate BO significantly influences both the quality of solutions and its computational efficiency. In this paper, we propose a simple, yet theoretically grounded, two-step method for automatically terminating BO. Our core concept is to proactively identify if the search is within a convex region by examining previously observed samples. BO is halted once the local regret within this convex region falls below a predetermined threshold. To enhance numerical stability, we propose an approximation method for calculating the termination indicator by solving a bilevel optimization problem. We conduct extensive empirical studies on diverse benchmark problems, including synthetic functions, reinforcement learning, and hyperparameter optimization. Experimental results demonstrate that our proposed method saves up to $\approx 80\%$ computational budget yet is with an order of magnitude smaller performance degradation, comparing against the other peer methods. In addition, our proposed termination method is robust in terms of the setting of its termination criterion.

----

## [0] Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models

**Authors**: *Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Jianfeng Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have achieved remarkable progress in solving various natural language processing tasks due to emergent reasoning abilities. However, LLMs have inherent limitations as they are incapable of accessing up-to-date information (stored on the Web or in task-specific knowledge bases), using external tools, and performing precise mathematical and logical reasoning. In this paper, we present Chameleon, an AI system that mitigates these limitations by augmenting LLMs with plug-and-play modules for compositional reasoning. Chameleon synthesizes programs by composing various tools (e.g., LLMs, off-the-shelf vision models, web search engines, Python functions, and heuristic-based modules) for accomplishing complex reasoning tasks. At the heart of Chameleon is an LLM-based planner that assembles a sequence of tools to execute to generate the final response. We showcase the effectiveness of Chameleon on two multi-modal knowledge-intensive reasoning tasks: ScienceQA and TabMWP. Chameleon, powered by GPT-4, achieves an 86.54% overall accuracy on ScienceQA, improving the best published few-shot result by 11.37%. On TabMWP, GPT-4-powered Chameleon improves the accuracy by 17.0%, lifting the state of the art to 98.78%. Our analysis also shows that the GPT-4-powered planner exhibits more consistent and rational tool selection via inferring potential constraints from instructions, compared to a ChatGPT-powered planner.

----

## [0] Diffusion Models and Semi-Supervised Learners Benefit Mutually with Few Labels

**Authors**: *Zebin You, Yong Zhong, Fan Bao, Jiacheng Sun, Chongxuan Li, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8735753cc18f6baa92d1f069fd8b14a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8735753cc18f6baa92d1f069fd8b14a0-Abstract-Conference.html)

**Abstract**:

In an effort to further advance semi-supervised generative and classification tasks, we propose a simple yet effective training strategy called *dual pseudo training* (DPT), built upon strong semi-supervised learners and diffusion models. DPT operates in three stages: training a classifier on partially labeled data to predict pseudo-labels; training a conditional generative model using these pseudo-labels to generate pseudo images; and retraining the classifier with a mix of real and pseudo images.  Empirically, DPT consistently achieves SOTA performance of semi-supervised generation and classification across various settings. In particular, with one or two labels per class, DPT achieves a Fr√©chet Inception Distance (FID) score of 3.08 or 2.52 on ImageNet $256\times256$. Besides, DPT outperforms competitive semi-supervised baselines substantially on  ImageNet classification tasks, *achieving top-1 accuracies of 59.0 (+2.8), 69.5 (+3.0), and 74.4 (+2.0)* with one, two, or five labels per class, respectively. Notably, our results demonstrate that diffusion can generate realistic images with only a few labels (e.g., $<0.1$%) and generative augmentation remains viable for semi-supervised classification. Our code is available at *https://github.com/ML-GSAI/DPT*.

----

## [0] Pre-Training Protein Encoder via Siamese Sequence-Structure Diffusion Trajectory Prediction

**Authors**: *Zuobai Zhang, Minghao Xu, Aurélie C. Lozano, Vijil Chenthamarakshan, Payel Das, Jian Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/873c86d9a979ab80d8e2919510d4446b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/873c86d9a979ab80d8e2919510d4446b-Abstract-Conference.html)

**Abstract**:

Self-supervised pre-training methods on proteins have recently gained attention, with most approaches focusing on either protein sequences or structures, neglecting the exploration of their joint distribution, which is crucial for a comprehensive understanding of protein functions by integrating co-evolutionary information and structural characteristics. In this work, inspired by the success of denoising diffusion models in generative tasks, we propose the DiffPreT approach to pre-train a protein encoder by sequence-structure joint diffusion modeling. DiffPreT guides the encoder to recover the native protein sequences and structures from the perturbed ones along the joint diffusion trajectory, which acquires the joint distribution of sequences and structures. Considering the essential protein conformational variations, we enhance DiffPreT by a method called Siamese Diffusion Trajectory Prediction (SiamDiff) to capture the correlation between different conformers of a protein. SiamDiff attains this goal by maximizing the mutual information between representations of diffusion trajectories of structurally-correlated conformers. We study the effectiveness of DiffPreT and SiamDiff on both atom- and residue-level structure-based protein understanding tasks. Experimental results show that the performance of DiffPreT is consistently competitive on all tasks, and SiamDiff achieves new state-of-the-art performance, considering the mean ranks on all tasks. Code will be released upon acceptance.

----

## [0] Progressive Ensemble Distillation: Building Ensembles for Efficient Inference

**Authors**: *Don Kurian Dennis, Abhishek Shetty, Anish Prasad Sevekari, Kazuhito Koishida, Virginia Smith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87425754bcc35f2bc62ef4a421a772d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87425754bcc35f2bc62ef4a421a772d6-Abstract-Conference.html)

**Abstract**:

Knowledge distillation is commonly used to compress an ensemble of models into a single model. In this work we study the problem of progressive ensemble distillation: Given a large, pretrained teacher model , we seek to decompose the model into an ensemble of smaller, low-inference cost student models . The resulting ensemble allows for flexibly tuning accuracy vs. inference cost, which can be useful for a multitude of applications in efficient inference. Our method, B-DISTIL, uses a boosting procedure that allows function composition based aggregation rules to construct expressive ensembles with similar performance as using much smaller student models. We demonstrate the effectiveness of B-DISTIL by decomposing pretrained models across a variety of image, speech, and sensor datasets. Our method comes with strong theoretical guarantees in terms of convergence as well as generalization.

----

## [0] Differentially Private Approximate Near Neighbor Counting in High Dimensions

**Authors**: *Alexandr Andoni, Piotr Indyk, Sepideh Mahabadi, Shyam Narayanan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87571720167f7e88827c40e468e3101f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87571720167f7e88827c40e468e3101f-Abstract-Conference.html)

**Abstract**:

Range counting (e.g., counting the number of data points falling into a given query ball) under differential privacy has been studied extensively. However, the current algorithms for this problem are subject to the following dichotomy. One class of algorithms suffers from an additive error that is a fixed polynomial in the number of points. Another class of algorithms allows for polylogarithmic additive error, but the error grows exponentially in the dimension. To achieve the latter, the problem is relaxed to allow a “fuzzy” definition of the range boundary, e.g., a count of the points in a ball of radius $r$ might also include points in a ball of radius $cr$ for some $c>1$. In this paper we present an efficient algorithm that offers a sweet spot between these two classes. The algorithm has an additive error that is an arbitrary small power of the data set size, depending on how fuzzy the range boundary is, as well as a small ($1+o(1)$) multiplicative error. Crucially, the amount of noise added has no dependence on the dimension. Our algorithm introduces a variant of Locality-Sensitive Hashing, utilizing it in a novel manner.

----

## [0] Reinforcement-Enhanced Autoregressive Feature Transformation: Gradient-steered Search in Continuous Space for Postfix Expressions

**Authors**: *Dongjie Wang, Meng Xiao, Min Wu, Pengfei Wang, Yuanchun Zhou, Yanjie Fu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8797d13e5998acfab387d4bf0a5b9b00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8797d13e5998acfab387d4bf0a5b9b00-Abstract-Conference.html)

**Abstract**:

Feature transformation aims to generate new pattern-discriminative feature space from original features to improve downstream machine learning  (ML) task performances. However, the discrete search space for the optimal feature explosively grows on the basis of combinations of features and operations from low-order forms to high-order forms. Existing methods, such as exhaustive search, expansion reduction, evolutionary algorithms, reinforcement learning, and iterative greedy, suffer from large search space. Overly emphasizing efficiency in algorithm design usually sacrifice stability or robustness. To fundamentally fill this gap, we reformulate discrete feature transformation as a continuous space optimization task and develop an embedding-optimization-reconstruction framework. This framework includes four steps: 1) reinforcement-enhanced data preparation, aiming to prepare high-quality transformation-accuracy training data; 2) feature transformation operation sequence embedding, intending to encapsulate the knowledge of prepared training data within a continuous space; 3) gradient-steered optimal embedding search, dedicating to uncover potentially superior embeddings within the learned space; 4) transformation operation sequence reconstruction, striving to reproduce the feature transformation solution to pinpoint the optimal feature space. Finally, extensive experiments and case studies are performed to demonstrate the effectiveness and robustness of the proposed method. The code and data are publicly accessible https://www.dropbox.com/sh/imh8ckui7va3k5u/AACulQegVx0MuywYyoCqSdVPa?dl=0.

----

## [0] FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning

**Authors**: *Kun Song, Huimin Ma, Bochao Zou, Huishuai Zhang, Weiran Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87cf37e2085655bad7bad0a014e0edad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87cf37e2085655bad7bad0a014e0edad-Abstract-Conference.html)

**Abstract**:

Due to the limited availability of data, existing few-shot learning methods trained from scratch fail to achieve satisfactory performance. In contrast, large-scale pre-trained models such as CLIP demonstrate remarkable few-shot and zero-shot capabilities. To enhance the performance of pre-trained models for downstream tasks, fine-tuning the model on downstream data is frequently necessary. However, fine-tuning the pre-trained model leads to a decrease in its generalizability in the presence of distribution shift, while the limited number of samples in few-shot learning makes the model highly susceptible to overfitting. Consequently, existing methods for fine-tuning few-shot learning primarily focus on fine-tuning the model's classification head or introducing additional structure. In this paper, we introduce a fine-tuning approach termed Feature Discrimination Alignment (FD-Align). Our method aims to bolster the model's generalizability by preserving the consistency of spurious features across the fine-tuning process. Extensive experimental results validate the efficacy of our approach for both ID and OOD tasks. Once fine-tuned, the model can seamlessly integrate with existing methods, leading to performance improvements. Our code can be found in https://github.com/skingorz/FD-Align.

----

## [0] A Step Towards Worldwide Biodiversity Assessment: The BIOSCAN-1M Insect Dataset

**Authors**: *Zahra Gharaee, ZeMing Gong, Nicholas Pellegrino, Iuliia Zarubiieva, Joakim Bruslund Haurum, Scott C. Lowe, Jaclyn T. A. McKeown, Chris C. Y. Ho, Joschka McLeod, Yi-Yun C. Wei, Jireh Agda, Sujeevan Ratnasingham, Dirk Steinke, Angel X. Chang, Graham W. Taylor, Paul W. Fieguth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87dbbdc3a685a97ad28489a1d57c45c1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/87dbbdc3a685a97ad28489a1d57c45c1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In an effort to catalog insect biodiversity, we propose a new large dataset of hand-labelled insect images, the BIOSCAN-1M Insect Dataset. Each record is taxonomically classified by an expert, and also has associated genetic information including raw nucleotide barcode sequences and assigned barcode index numbers, which are genetic-based proxies for species classification. This paper presents a curated million-image dataset, primarily to train computer-vision models capable of providing image-based taxonomic assessment, however, the dataset also presents compelling characteristics, the study of which would be of interest to the broader machine learning community. Driven by the biological nature inherent to the dataset, a characteristic long-tailed class-imbalance distribution is exhibited. Furthermore, taxonomic labelling is a hierarchical classification scheme, presenting a highly fine-grained classification problem at lower levels. Beyond spurring interest in biodiversity research within the machine learning community, progress on creating an image-based taxonomic classifier will also further the ultimate goal of all BIOSCAN research: to lay the foundation for a comprehensive survey of global biodiversity. This paper introduces the dataset and explores the classification task through the implementation and analysis of a baseline classifier. The code repository of the BIOSCAN-1M-Insect dataset is available at https://github.com/zahrag/BIOSCAN-1M

----

## [0] D-Separation for Causal Self-Explanation

**Authors**: *Wei Liu, Jun Wang, Haozhao Wang, Ruixuan Li, Zhiying Deng, Yuankai Zhang, Yang Qiu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87e82678c0d6e5b729398426f82e9af6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87e82678c0d6e5b729398426f82e9af6-Abstract-Conference.html)

**Abstract**:

Rationalization aims to strengthen the interpretability of NLP models by extracting a subset of human-intelligible pieces of their inputting texts. Conventional works generally employ the maximum mutual information (MMI) criterion to find the rationale that is most indicative of the target label. However, this criterion can be influenced by spurious features that correlate with the causal rationale or the target label. Instead of attempting to rectify the issues of the MMI criterion, we propose a novel criterion to uncover the causal rationale, termed the Minimum Conditional Dependence (MCD) criterion, which is grounded on our finding that the non-causal features and the target label are \emph{d-separated} by the causal rationale. By minimizing the dependence between the non-selected parts of the input and the target label conditioned on the selected rationale candidate, all the causes of the label are compelled to be selected. In this study, we employ a simple and practical measure for dependence, specifically the KL-divergence, to validate our proposed MCD criterion.  Empirically, we demonstrate that MCD improves the F1 score by up to 13.7% compared to previous state-of-the-art MMI-based methods.Our code is in an anonymous repository: https://anonymous.4open.science/r/MCD-CE88.

----

## [0] History Filtering in Imperfect Information Games: Algorithms and Complexity

**Authors**: *Christopher Solinas, Douglas Rebstock, Nathan R. Sturtevant, Michael Buro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87ee1bbac4635e7c948f3eea83c1f262-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87ee1bbac4635e7c948f3eea83c1f262-Abstract-Conference.html)

**Abstract**:

Historically applied exclusively to perfect information games, depth-limited search with value functions has been key to recent advances in AI for imperfect information games. Most prominent approaches with strong theoretical guarantees require subgame decomposition - a process in which a subgame is computed from public information and player beliefs. However, subgame decomposition can itself require non-trivial computations, and its tractability depends on the existence of efficient algorithms for either full enumeration or generation of the histories that form the root of the subgame. Despite this, no formal analysis of the tractability of such computations has been established in prior work, and application domains have often consisted of games, such as poker, for which enumeration is trivial on modern hardware.Applying these ideas to more complex domains requires understanding their cost. In this work, we introduce and analyze the computational aspects and tractability of filtering histories for subgame decomposition. We show that constructing a single history from the root of the subgame is generally intractable, and then provide a necessary and sufficient condition for efficient enumeration. We also introduce a novel Markov Chain Monte Carlo-based generation algorithm for trick-taking card games - a domain where enumeration is often prohibitively expensive. Our experiments demonstrate its improved scalability in the trick-taking card game Oh Hell.These contributions clarify when and how depth-limited search via subgame decomposition can be an effective tool for sequential decision-making in imperfect information settings.

----

## [0] Constant Approximation for Individual Preference Stable Clustering

**Authors**: *Anders Aamand, Justin Y. Chen, Allen Liu, Sandeep Silwal, Pattara Sukprasert, Ali Vakilian, Fred Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/881259965dacb9f42967aae84a157283-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/881259965dacb9f42967aae84a157283-Abstract-Conference.html)

**Abstract**:

Individual preference (IP) stability, introduced by Ahmadi et al. (ICML 2022), is a natural clustering objective inspired by stability and fairness constraints. A clustering is $\alpha$-IP stable if the average distance of every data point to its own cluster is at most $\alpha$ times the average distance to any other cluster. Unfortunately, determining if a dataset admits a $1$-IP stable clustering is NP-Hard. Moreover, before this work, it was unknown if an $o(n)$-IP stable clustering always exists, as the prior state of the art only guaranteed an $O(n)$-IP stable clustering. We close this gap in understanding and show that an $O(1)$-IP stable clustering always exists for general metrics, and we give an efficient algorithm which outputs such a clustering.  We also introduce generalizations of IP stability beyond average distance and give efficient near optimal algorithms in the cases where we consider the maximum and minimum distances within and between clusters.

----

## [0] Intervention Generalization: A View from Factor Graph Models

**Authors**: *Gecia Bravo Hermsdorff, David S. Watson, Jialin Yu, Jakob Zeitler, Ricardo Silva*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88139fdcc82fc597090620d77b023282-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88139fdcc82fc597090620d77b023282-Abstract-Conference.html)

**Abstract**:

One of the goals of causal inference is to generalize from past experiments and observational data to novel conditions. While it is in principle possible to eventually learn a mapping from a novel experimental condition to an outcome of interest, provided a sufficient variety of experiments is available in the training data, coping with a large combinatorial space of possible interventions is hard. Under a typical sparse experimental design, this mapping is ill-posed without relying on heavy regularization or prior distributions. Such assumptions may or may not be reliable, and can be hard to defend or test. In this paper, we take a close look at how to warrant a leap from past experiments to novel conditions based on minimal assumptions about the factorization of the distribution of the manipulated system, communicated in the well-understood language of factor graph models. A postulated interventional factor model (IFM) may not always be informative, but it conveniently abstracts away a need for explicitly modeling unmeasured confounding and feedback mechanisms, leading to directly testable claims. Given an IFM and datasets from a collection of experimental regimes, we derive conditions for identifiability of the expected outcomes of new regimes never observed in these training data. We implement our framework using several efficient algorithms, and apply them on a range of semi-synthetic experiments.

----

## [0] Decision Tree for Locally Private Estimation with Public Data

**Authors**: *Yuheng Ma, Han Zhang, Yuchao Cai, Hanfang Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88237ac4e9941b1be5c6d3c1ad408184-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88237ac4e9941b1be5c6d3c1ad408184-Abstract-Conference.html)

**Abstract**:

We propose conducting locally differentially private (LDP) estimation with the aid of a small amount of public data to enhance the performance of private estimation. Specifically, we introduce an efficient algorithm called Locally differentially Private Decision Tree (LPDT) for LDP regression. We first use the public data to grow a decision tree partition and then fit an estimator according to the partition privately.  From a theoretical perspective, we show that LPDT is $\varepsilon$-LDP and has a mini-max optimal convergence rate under a mild assumption of similarity between public and private data, whereas the lower bound of the convergence rate of LPDT without public data is strictly slower, which implies that the public data helps to improve the convergence rates of LDP estimation. We conduct experiments on both synthetic and real-world data to demonstrate the superior performance of LPDT compared with other state-of-the-art LDP regression methods. Moreover, we show that LPDT remains effective despite considerable disparities between public and private data.

----

## [0] DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization

**Authors**: *Haoran Ye, Jiarui Wang, Zhiguang Cao, Helan Liang, Yong Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html)

**Abstract**:

Ant Colony Optimization (ACO) is a meta-heuristic algorithm that has been successfully applied to various Combinatorial Optimization Problems (COPs). Traditionally, customizing ACO for a specific problem requires the expert design of knowledge-driven heuristics. In this paper, we propose DeepACO, a generic framework that leverages deep reinforcement learning to automate heuristic designs. DeepACO serves to strengthen the heuristic measures of existing ACO algorithms and dispense with laborious manual design in future ACO applications. As a neural-enhanced meta-heuristic, DeepACO consistently outperforms its ACO counterparts on eight COPs using a single neural model and a single set of hyperparameters. As a Neural Combinatorial Optimization method, DeepACO performs better than or on par with problem-specific methods on canonical routing problems. Our code is publicly available at https://github.com/henry-yeh/DeepACO.

----

## [0] Offline Imitation Learning with Variational Counterfactual Reasoning

**Authors**: *Zexu Sun, Bowei He, Jinxin Liu, Xu Chen, Chen Ma, Shuai Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8833c8aa10542d24d693bbaf6a4598f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8833c8aa10542d24d693bbaf6a4598f5-Abstract-Conference.html)

**Abstract**:

In offline imitation learning (IL), an agent aims to learn an optimal expert behavior policy without additional online environment interactions. However, in many real-world scenarios, such as robotics manipulation, the offline dataset is collected from suboptimal behaviors without rewards. Due to the scarce expert data, the agents usually suffer from simply memorizing poor trajectories and are vulnerable to the variations in the environments, lacking the capability of generalizing to new environments.To automatically generate high-quality expert data and improve the generalization ability of the agent, we propose a framework named \underline{O}ffline \underline{I}mitation \underline{L}earning with \underline{C}ounterfactual data \underline{A}ugmentation (OILCA) by doing counterfactual inference. In particular, we leverage identifiable variational autoencoder to generate \textit{counterfactual} samples for expert data augmentation. We theoretically analyze the influence of the generated expert data and the improvement of generalization. Moreover, we conduct extensive experiments to demonstrate that our approach significantly outperforms various baselines on both \textsc{DeepMind Control Suite} benchmark for in-distribution performance and \textsc{CausalWorld} benchmark for out-of-distribution generalization.

----

## [0] LART: Neural Correspondence Learning with Latent Regularization Transformer for 3D Motion Transfer

**Authors**: *Haoyu Chen, Hao Tang, Radu Timofte, Luc Van Gool, Guoying Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88593e16b09104fb6010d370c081d7bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88593e16b09104fb6010d370c081d7bc-Abstract-Conference.html)

**Abstract**:

3D motion transfer aims at transferring the motion from a dynamic input sequence to a static 3D object and outputs an identical motion of the target with high-fidelity and realistic visual effects. In this work, we propose a novel 3D Transformer framework called LART for 3D motion transfer. With carefully-designed architectures, LART is able to implicitly learn the correspondence via a flexible geometry perception. Thus, unlike other existing methods, LART does not require any key point annotations or pre-defined correspondence between the motion source and target meshes and can also handle large-size full-detailed unseen 3D targets. Besides, we introduce a novel latent metric regularization on the Transformer for better motion generation. Our rationale lies in the observation that the decoded motions can be approximately expressed as linearly geometric distortion at the frame level. The metric preservation of motions could be translated to the formation of linear paths in the underlying latent space as a rigorous constraint to control the synthetic motions occurring in the construction of the latent space. The proposed LART shows a high learning efficiency with the need for a few samples from the AMASS dataset to generate motions with plausible visual effects. The experimental results verify the potential of our generative model in applications of motion transfer, content generation, temporal interpolation, and motion denoising. The code is made available: https://github.com/mikecheninoulu/LART.

----

## [0] Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data

**Authors**: *Jang-Hyun Kim, Sangdoo Yun, Hyun Oh Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/886ed40d7882c9f891824e42a452c228-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/886ed40d7882c9f891824e42a452c228-Abstract-Conference.html)

**Abstract**:

Diagnosing and cleaning data is a crucial step for building robust machine learning systems. However, identifying problems within large-scale datasets with real-world distributions is challenging due to the presence of complex issues such as label errors, under-representation, and outliers. In this paper, we propose a unified approach for identifying the problematic data by utilizing a largely ignored source of information: a relational structure of data in the feature-embedded space. To this end, we present scalable and effective algorithms for detecting label errors and outlier data based on the relational graph structure of data. We further introduce a visualization tool that provides contextual information of a data point in the feature-embedded space, serving as an effective tool for interactively diagnosing data. We evaluate the label error and outlier/out-of-distribution (OOD) detection performances of our approach on the large-scale image, speech, and language domain tasks, including ImageNet, ESC-50, and SST2. Our approach achieves state-of-the-art detection performance on all tasks considered and demonstrates its effectiveness in debugging large-scale real-world datasets across various domains. We release codes at https://github.com/snu-mllab/Neural-Relation-Graph.

----

## [0] Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory

**Authors**: *Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, Rui Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/887262aeb3eafb01ef0fd0e3a87a8831-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/887262aeb3eafb01ef0fd0e3a87a8831-Abstract-Conference.html)

**Abstract**:

With direct access to human-written reference as memory, retrieval-augmented generation has achieved much progress in a wide range of text generation tasks. Since better memory would typically prompt better generation (we define this as primal problem). The traditional approach for memory retrieval involves selecting memory that exhibits the highest similarity to the input. However, this method is constrained by the quality of the fixed corpus from which memory is retrieved. In this paper, by exploring the duality of the primal problem: better generation also prompts better memory, we propose a novel framework, selfmem, which addresses this limitation by iteratively employing a retrieval-augmented generator to create an unbounded memory pool and using a memory selector to choose one output as memory for the subsequent generation round. This enables the model to leverage its own output, referred to as self-memory, for improved generation. We evaluate the effectiveness of selfmem on three distinct text generation tasks: neural machine translation, abstractive text summarization, and dialogue generation, under two generation paradigms: fine-tuned small model and few-shot LLM. Our approach achieves state-of-the-art results in four directions in JRC-Acquis translation dataset, 50.3 ROUGE-1 in XSum, and 62.9 ROUGE-1 in BigPatent, demonstrating the potential of self-memory in enhancing retrieval-augmented generation models. Furthermore, we conduct thorough analyses of each component in the selfmem framework to identify current system bottlenecks and provide insights for future research.

----

## [0] Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge

**Authors**: *Abhin Shah, Karthikeyan Shanmugam, Murat Kocaoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/887932131fddf943e8fe3310b62c0147-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/887932131fddf943e8fe3310b62c0147-Abstract-Conference.html)

**Abstract**:

Causal effect estimation from data typically requires assumptions about the cause-effect relations either explicitly in the form of a causal graph structure within the Pearlian framework, or implicitly in terms of (conditional) independence statements between counterfactual variables within the potential outcomes framework. When the treatment variable and the outcome variable are confounded, front-door adjustment is an important special case where, given the graph, causal effect of the treatment on the target can be estimated using post-treatment variables. However, the exact formula for front-door adjustment depends on the structure of the graph, which is difficult to learn in practice. In this work, we provide testable conditional independence statements to compute the causal effect using front-door-like adjustment without knowing the graph under limited structural side information. We show that our method is applicable in scenarios where knowing the Markov equivalence class is not sufficient for causal effect estimation. We demonstrate the effectiveness of our method on a class of random graphs as well as real causal fairness benchmarks.

----



[Go to the previous page](NIPS-2023-list1800.md)

[Go to the next page](NIPS-2023-list1802.md)

[Go to the catalog section](README.md)