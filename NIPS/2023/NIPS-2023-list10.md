## [1800] NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks

        **Authors**: *Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81e1cdaa570954321d8b06be386cc3d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81e1cdaa570954321d8b06be386cc3d4-Abstract-Conference.html)

        **Abstract**:

        While multi-exit neural networks are regarded as a promising solution for making efficient inference via early exits, combating adversarial attacks remains a challenging problem. In multi-exit networks, due to the high dependency among different submodels, an adversarial example targeting a specific exit not only degrades the performance of the target exit but also reduces the performance of all other exits concurrently. This makes multi-exit networks highly vulnerable to simple adversarial attacks. In this paper, we propose NEO-KD, a knowledge-distillation-based adversarial training strategy that tackles this fundamental challenge based on two key contributions. NEO-KD first resorts to neighbor knowledge distillation to guide the output of the adversarial examples to tend to the ensemble outputs of neighbor exits of clean data. NEO-KD also employs exit-wise orthogonal knowledge distillation for reducing adversarial transferability across different submodels. The result is a significantly improved robustness against adversarial attacks. Experimental results on various datasets/models show that our method achieves the best adversarial accuracy with reduced computation budgets, compared to the baselines relying on existing adversarial training or knowledge distillation techniques for multi-exit networks.

        ----

        ## [1801] Self-Evaluation Guided Beam Search for Reasoning

        **Authors**: *Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, Michael Qizhe Xie*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b-Abstract-Conference.html)

        **Abstract**:

        Breaking down a problem into intermediate steps has demonstrated impressive performance in Large Language Model (LLM) reasoning. However, the growth of the reasoning chain introduces uncertainty and error accumulation, making it challenging to elicit accurate final results. To tackle this challenge of uncertainty in multi-step reasoning, we introduce a stepwise self-evaluation mechanism to guide and calibrate the reasoning process of LLMs. We propose a decoding algorithm integrating the self-evaluation guidance via stochastic beam search. The self-evaluation guidance serves as a better-calibrated automatic criterion, facilitating an efficient search in the reasoning space and resulting in superior prediction quality. Stochastic beam search balances exploitation and exploration of the search space with temperature-controlled randomness. Our approach surpasses the corresponding Codex-backboned baselines in few-shot accuracy by $6.34$%, $9.56$%, and $5.46$% on the GSM8K, AQuA, and StrategyQA benchmarks, respectively. Experiment results with Llama-2 on arithmetic reasoning demonstrate the efficiency of our method in outperforming the baseline methods with comparable computational budgets. Further analysis in multi-step reasoning finds our self-evaluation guidance pinpoints logic failures and leads to higher consistency and robustness. Our code is publicly available at [https://guideddecoding.github.io/](https://guideddecoding.github.io/).

        ----

        ## [1802] ViSt3D: Video Stylization with 3D CNN

        **Authors**: *Ayush Pande, Gaurav Sharma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8203a5156918d467328d5a90147ab307-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8203a5156918d467328d5a90147ab307-Abstract-Conference.html)

        **Abstract**:

        Visual stylization has been a very popular research area in recent times. While image stylization has seen a rapid advancement in the recent past, video stylization, while being more challenging, is relatively less explored. The immediate method of stylizing videos by stylizing each frame independently has been tried with some success. To the best of our knowledge, we present the first approach to video stylization using 3D CNN directly, building upon insights from 2D image stylization. Stylizing video is highly challenging, as the appearance and video motion, which includes both camera and subject motions, are inherently entangled in the representations learnt by a 3D CNN. Hence, a naive extension of 2D CNN stylization methods to 3D CNN does not work. To perform stylization with 3D CNN, we propose to explicitly disentangle motion and appearance, stylize the appearance part, and then add back the motion component and decode the final stylized video. In addition, we propose a dataset, curated from existing datasets, to train video stylization networks. We also provide an independently collected test set to study the generalization of video stylization methods. We provide results on this test dataset comparing the proposed method with 2D stylization methods applied frame by frame. We show successful stylization with 3D CNN for the first time, and obtain better stylization in terms of texture cf.\ the existing 2D methods.

        ----

        ## [1803] Smoothed Online Learning for Prediction in Piecewise Affine Systems

        **Authors**: *Adam Block, Max Simchowitz, Russ Tedrake*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82096f4f6f897529ecd3eabea603e9cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82096f4f6f897529ecd3eabea603e9cc-Abstract-Conference.html)

        **Abstract**:

        The problem of piecewise affine (PWA) regression and planning is of foundational importance to the study of online learning, control, and robotics, where it provides a theoretically and empirically tractable setting to study systems undergoing sharp changes in the dynamics.  Unfortunately, due to the discontinuities that arise when crossing into different ``pieces,'' learning in general sequential settings is impossible and practical algorithms are forced to resort to heuristic approaches.  This paper builds on the recently developed smoothed online learning framework and provides the first algorithms for prediction and simulation in PWA systems whose regret is polynomial in all relevant problem parameters under a weak smoothness assumption; moreover, our algorithms are efficient in the number of calls to an optimization oracle.  We further apply our results to the problems of one-step prediction and multi-step simulation regret in piecewise affine dynamical systems, where the learner is tasked with simulating trajectories and regret is measured in terms of the Wasserstein distance between simulated and true data.  Along the way, we develop several technical tools of more general interest.

        ----

        ## [1804] Adversarial Attacks on Online Learning to Rank with Click Feedback

        **Authors**: *Jinhang Zuo, Zhiyao Zhang, Zhiyong Wang, Shuai Li, Mohammad Hajiesmaili, Adam Wierman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/820e42f39773c6cbbd875553db45658f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/820e42f39773c6cbbd875553db45658f-Abstract-Conference.html)

        **Abstract**:

        Online learning to rank (OLTR) is a sequential decision-making problem where a learning agent selects an ordered list of items and receives feedback through user clicks. Although potential attacks against OLTR algorithms may cause serious losses in real-world applications, there is limited knowledge about adversarial attacks on OLTR. This paper studies attack strategies against multiple variants of OLTR. Our first result provides an attack strategy against the UCB algorithm on classical stochastic bandits with binary feedback, which solves the key issues caused by bounded and discrete feedback that previous works cannot handle. Building on this result, we design attack algorithms against UCB-based OLTR algorithms in position-based and cascade models. Finally, we propose a general attack strategy against any algorithm under the general click model. Each attack algorithm manipulates the learning agent into choosing the target attack item $T-o(T)$ times, incurring a cumulative cost of $o(T)$. Experiments on synthetic and real data further validate the effectiveness of our proposed attack algorithms.

        ----

        ## [1805] RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths

        **Authors**: *Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, Ping Luo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/821655c7dc4836838cd8524d07f9d6fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/821655c7dc4836838cd8524d07f9d6fd-Abstract-Conference.html)

        **Abstract**:

        Text-to-image generation has recently witnessed remarkable achievements. We introduce a text-conditional image diffusion model, termed RAPHAEL, to generate highly artistic images, which accurately portray the text prompts, encompassing multiple nouns, adjectives, and verbs. This is achieved by  stacking tens of mixture-of-experts (MoEs) layers, i.e., space-MoE  and time-MoE layers,  enabling billions of diffusion paths (routes) from the network input to the output. Each path intuitively functions as a "painter" for depicting a particular textual concept onto a specified image region at a diffusion timestep. Comprehensive experiments reveal that RAPHAEL outperforms recent cutting-edge models, such as Stable Diffusion, ERNIE-ViLG 2.0, DeepFloyd, and DALL-E 2, in terms of both image quality and aesthetic appeal.  Firstly, RAPHAEL exhibits superior performance in switching images across diverse styles, such as Japanese comics, realism, cyberpunk, and ink illustration. Secondly, a single model with three billion parameters, trained on 1,000 A100 GPUs for two months, achieves a state-of-the-art zero-shot FID score of 6.61 on the COCO dataset. Furthermore, RAPHAEL significantly surpasses its counterparts in human evaluation on the ViLG-300 benchmark. We believe that RAPHAEL holds the potential to propel the frontiers of image generation research in both academia and industry, paving the way for future breakthroughs in this rapidly evolving field. More details can be found on a webpage: https://raphael-painter.github.io/.

        ----

        ## [1806] Towards Evaluating Transfer-based Attacks Systematically, Practically, and Fairly

        **Authors**: *Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/823e43f5537d8c1894afd1f6ab00a927-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/823e43f5537d8c1894afd1f6ab00a927-Abstract-Conference.html)

        **Abstract**:

        The adversarial vulnerability of deep neural networks (DNNs) has drawn great attention due to the security risk of applying these models in real-world applications. Based on transferability of adversarial examples, an increasing number of transfer-based methods have been developed to fool black-box DNN models whose architecture and parameters are inaccessible. Although tremendous effort has been exerted, there still lacks a standardized benchmark that could be taken advantage of to compare these methods systematically, fairly, and practically. Our investigation shows that the evaluation of some methods needs to be more reasonable and more thorough to verify their effectiveness, to avoid, for example, unfair comparison and insufficient consideration of possible substitute/victim models. Therefore, we establish a transfer-based attack benchmark (TA-Bench) which implements 30+ methods. In this paper, we evaluate and compare them comprehensively on 10 popular substitute/victim models on ImageNet. New insights about the effectiveness of these methods are gained and guidelines for future evaluations are provided.

        ----

        ## [1807] Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger

        **Authors**: *Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, George Karypis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8249b30d877c91611fd8c7aa6ac2b5fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8249b30d877c91611fd8c7aa6ac2b5fe-Abstract-Conference.html)

        **Abstract**:

        Per-example gradient clipping is a key algorithmic step that enables practical differential private (DP) training for deep learning models. The choice of clipping threshold $R$, however, is vital for achieving high accuracy under DP. We propose an easy-to-use replacement, called automatic clipping, that eliminates the need to tune $R$ for any DP optimizers, including DP-SGD, DP-Adam, DP-LAMB and many others.The automatic variants are as private and computationally efficient as existing DP optimizers, but require no DP-specific hyperparameters and thus make DP training as amenable as the standard non-private training. We give a rigorous convergence analysis of automatic DP-SGD in the non-convex setting, showing that it can enjoy an asymptotic convergence rate that matches the standard SGD, under a symmetric gradient noise assumption of the per-sample gradients (commonly used in the non-DP literature). We demonstrate on various language and vision tasks that automatic clipping outperforms or matches the state-of-the-art, and can be easily employed with minimal changes to existing codebases.

        ----

        ## [1808] Error Discovery By Clustering Influence Embeddings

        **Authors**: *Fulton Wang, Julius Adebayo, Sarah Tan, Diego García-Olano, Narine Kokhlikyan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8278a2e5f9db8489cd908d20c43f1f87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8278a2e5f9db8489cd908d20c43f1f87-Abstract-Conference.html)

        **Abstract**:

        We present a method for identifying groups of test examples---slices---on which a model under-performs, a task now known as slice discovery. We formalize coherence---a requirement that erroneous predictions, within a slice, should be wrong for the same reason---as a key property that any slice discovery method should satisfy.  We then use influence functions to derive a new slice discovery method, InfEmbed, which satisfies coherence by returning slices whose examples are influenced similarly by the training data.  InfEmbed is simple, and consists of applying K-Means clustering to a novel representation we deem influence embeddings. We show InfEmbed outperforms current state-of-the-art methods on 2 benchmarks, and is effective for model debugging across several case studies.

        ----

        ## [1809] BadTrack: A Poison-Only Backdoor Attack on Visual Object Tracking

        **Authors**: *Bin Huang, Jiaqian Yu, Yiwei Chen, Siyang Pan, Qiang Wang, Zhi Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/828bb8f42d4ab15322b9315151959c61-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/828bb8f42d4ab15322b9315151959c61-Abstract-Conference.html)

        **Abstract**:

        Visual object tracking (VOT) is one of the most fundamental tasks in computer vision community. State-of-the-art VOT trackers extract positive and negative examples that are used to guide the tracker to distinguish the object from the background. In this paper, we show that this characteristic can be exploited to introduce new threats and hence propose a simple yet effective poison-only backdoor attack. To be specific, we poison a small part of the training data by attaching a predefined trigger pattern to the background region of each video frame, so that the trigger appears almost exclusively in the extracted negative examples. To the best of our knowledge, this is the first work that reveals the threat of poison-only backdoor attack on VOT trackers. We experimentally show that our backdoor attack can significantly degrade the performance of both two-stream Siamese and one-stream Transformer trackers on the poisoned data while gaining comparable performance with the benign trackers on the clean data.

        ----

        ## [1810] Spiking PointNet: Spiking Neural Networks for Point Clouds

        **Authors**: *Dayong Ren, Zhe Ma, Yuanpei Chen, Weihang Peng, Xiaode Liu, Yuhan Zhang, Yufei Guo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8296d5800a8e68e58ad0472b393be80e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8296d5800a8e68e58ad0472b393be80e-Abstract-Conference.html)

        **Abstract**:

        Recently, Spiking Neural Networks (SNNs), enjoying extreme energy efficiency, have drawn much research attention on 2D visual recognition and shown gradually increasing application potential. However, it still remains underexplored whether SNNs can be generalized to 3D recognition. To this end, we present Spiking PointNet in the paper, the first spiking neural model for efficient deep learning on point clouds. We discover that the two huge obstacles limiting the application of SNNs in point clouds are: the intrinsic optimization obstacle of SNNs that impedes the training of a big spiking model with large time steps, and the expensive memory and computation cost of PointNet that makes training a big spiking point model unrealistic. To solve the problems simultaneously, we present a trained-less but learning-more paradigm for Spiking PointNet with theoretical justifications and in-depth experimental analysis. In specific, our Spiking PointNet is trained with only a single time step but can obtain better performance with multiple time steps inference, compared to the one trained directly with multiple time steps. We conduct various experiments on ModelNet10, ModelNet40 to demonstrate the effectiveness of Sipiking PointNet. Notably, our Spiking PointNet even can outperform its ANN counterpart, which is rare in the SNN field thus providing a potential research direction for the following work. Moreover, Spiking PointNet shows impressive speedup and storage saving in the training phase. Our code is open-sourced at https://github.com/DayongRen/Spiking-PointNet.

        ----

        ## [1811] A Sublinear-Time Spectral Clustering Oracle with Improved Preprocessing Time

        **Authors**: *Ranran Shen, Pan Peng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82aec8518602748540a42b783468c94d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82aec8518602748540a42b783468c94d-Abstract-Conference.html)

        **Abstract**:

        We address the problem of designing a sublinear-time spectral clustering oracle for graphs that exhibit strong clusterability. Such graphs contain $k$ latent clusters, each characterized by a large inner conductance (at least $\varphi$) and a small outer conductance (at most $\varepsilon$). Our aim is to preprocess the graph to enable clustering membership queries, with the key requirement that both preprocessing and query answering should be performed in sublinear time, and the resulting partition should be consistent with a $k$-partition that is close to the ground-truth clustering. Previous oracles have relied on either a $\textrm{poly}(k)\log n$ gap between inner and outer conductances or exponential (in $k/\varepsilon$) preprocessing time. Our algorithm relaxes these assumptions, albeit at the cost of a slightly higher misclassification ratio. We also show that our clustering oracle is robust against a few random edge deletions. To validate our theoretical bounds, we conducted experiments on synthetic networks.

        ----

        ## [1812] Boosting with Tempered Exponential Measures

        **Authors**: *Richard Nock, Ehsan Amid, Manfred K. Warmuth*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82d3258eb58ceac31744a88005b7ddef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82d3258eb58ceac31744a88005b7ddef-Abstract-Conference.html)

        **Abstract**:

        One of the most popular ML algorithms, AdaBoost, can bederived from the dual of a relative entropyminimization problem subject to the fact that the positive weightson the examples sum to one. Essentially, harder examples receive higher probabilities. We generalize this setup to the recently introduced *temperedexponential measure*s (TEMs) where normalization is enforced on a specific power of the measure and not the measure itself.TEMs are indexed by a parameter $t$ and generalize exponential families ($t=1$). Our algorithm, $t$-AdaBoost, recovers AdaBoost as a special case ($t=1$). We show that $t$-AdaBoost retains AdaBoost's celebrated exponential convergence rate when $t\in [0,1)$ while allowing a slight improvement of the rate's hidden constant compared to $t=1$. $t$-AdaBoost partially computes on a generalization of classical arithmetic over the reals and brings notable properties like guaranteed bounded leveraging coefficients for $t\in [0,1)$. From the loss that $t$-AdaBoost minimizes (a generalization of the exponential loss), we show how to derive a new family of *tempered* losses for the induction of domain-partitioning classifiers like decision trees. Crucially, strict properness is ensured for all while their boosting rates span the full known spectrum. Experiments using $t$-AdaBoost+trees display that significant leverage can be achieved by tuning $t$.

        ----

        ## [1813] DP-HyPO: An Adaptive Private Framework for Hyperparameter Optimization

        **Authors**: *Hua Wang, Sheng Gao, Huanyu Zhang, Weijie J. Su, Milan Shen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82d7d58cba24731c0ca952dff1de46ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82d7d58cba24731c0ca952dff1de46ae-Abstract-Conference.html)

        **Abstract**:

        Hyperparameter optimization, also known as hyperparameter tuning, is a widely recognized technique for improving model performance. Regrettably, when training private ML models, many practitioners often overlook the privacy risks associated with hyperparameter optimization, which could potentially expose sensitive information about the underlying dataset.Currently, the sole existing approach to allow privacy-preserving hyperparameter optimization is to uniformly and randomly select hyperparameters for a number of runs, subsequently reporting the best-performing hyperparameter.In contrast, in non-private settings, practitioners commonly utilize "adaptive" hyperparameter optimization methods such as Gaussian Process-based optimization, which select the next candidate based on information gathered from previous outputs.This substantial contrast between private and non-private hyperparameter optimization underscores a critical concern. In our paper, we introduce DP-HyPO, a pioneering framework for "adaptive" private hyperparameter optimization, aiming to bridge the gap between private and non-private hyperparameter optimization. To accomplish this, we provide a comprehensive differential privacy analysis of our framework. Furthermore, we empirically demonstrate the effectiveness of DP-HyPO on a diverse set of real-world datasets.

        ----

        ## [1814] Active Learning-Based Species Range Estimation

        **Authors**: *Christian Lange, Elijah Cole, Grant Van Horn, Oisin Mac Aodha*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82eec786fdfbbfa53450c5feb7d1ac92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82eec786fdfbbfa53450c5feb7d1ac92-Abstract-Conference.html)

        **Abstract**:

        We propose a new active learning approach for efficiently estimating the geographic range of a species from a limited number of on the ground observations. We model the range of an unmapped species of interest as the weighted combination of estimated ranges obtained from a set of different species. We show that it is possible to generate this candidate set of ranges by using models that have been trained on large weakly supervised community collected observation data. From this, we develop a new active querying approach that sequentially selects geographic locations to visit that best reduce our uncertainty over an unmapped speciesâ€™ range. We conduct a detailed evaluation of our approach and compare it to existing active learning methods using an evaluation dataset containing expert-derived ranges for one thousand species. Our results demonstrate that our method outperforms alternative active learning methods and approaches the performance of end-to-end trained models, even when only using a fraction of the data. This highlights the utility of active learning via transfer learned spatial representations for species range estimation. It also emphasizes the value of leveraging emerging large-scale crowdsourced datasets, not only for modeling a species' range, but also for actively discovering them.

        ----

        ## [1815] One-Step Diffusion Distillation via Deep Equilibrium Models

        **Authors**: *Zhengyang Geng, Ashwini Pokle, J. Zico Kolter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82f05a105c928c10706213952bf0c8b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82f05a105c928c10706213952bf0c8b7-Abstract-Conference.html)

        **Abstract**:

        Diffusion models excel at producing high-quality samples but naively require hundreds of iterations, prompting multiple attempts to distill the generation process into a faster network. However, many existing approaches suffer from a variety of challenges: the process for distillation training can be complex, often requiring multiple training stages, and the resulting models perform poorly when utilized in single-step generative applications. In this paper, we introduce a simple yet effective means of distilling diffusion models *directly* from the initial noise to the resulting image. Of particular importance to our approach is to leverage a new Deep Equilibrium (DEQ) model as the distilled architecture: the Generative Equilibrium Transformer (GET). Our method enables fully offline training with just noise/image pairs from the diffusion model while achieving superior performance compared to existing one-step methods on comparable training budgets. We demonstrate that the DEQ architecture is crucial to this capability, as GET matches a $5\times$ larger ViT in terms of FID scores while striking a critical balance of computational cost and image quality. Code, checkpoints, and datasets are available [here](https://github.com/locuslab/get).

        ----

        ## [1816] Discrete-Smoothness in Online Algorithms with Predictions

        **Authors**: *Yossi Azar, Debmalya Panigrahi, Noam Touitou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82f0dae85424eb743017c90380e7ab9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/82f0dae85424eb743017c90380e7ab9b-Abstract-Conference.html)

        **Abstract**:

        In recent years, there has been an increasing focus on designing online algorithms with (machine-learned) predictions. The ideal learning-augmented algorithm is comparable to the optimum when given perfect predictions (consistency), to the best online approximation for arbitrary predictions (robustness), and should interpolate between these extremes as a smooth function of the prediction error. In this paper, we quantify these guarantees in terms of a general property that we call discrete-smoothness, and achieve discrete-smooth algorithms for online covering, specifically the facility location and set cover problems. For set cover, our work improves the results of Bamas, Maggiori, and Svensson (2020) by augmenting consistency and robustness with smoothness guarantees. For facility location, our work improves on prior work by Almanza et al. (2021) by generalizing to nonuniform costs and also providing smoothness guarantees by augmenting consistency and robustness.

        ----

        ## [1817] A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning

        **Authors**: *Valeriia Cherepanova, Roman Levin, Gowthami Somepalli, Jonas Geiping, C. Bayan Bruss, Andrew Gordon Wilson, Tom Goldstein, Micah Goldblum*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/82f39c7409155b74d15d73b048f06771-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/82f39c7409155b74d15d73b048f06771-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Academic tabular benchmarks often contain small sets of curated features. In contrast, data scientists typically collect as many features as possible into their datasets, and even engineer new features from existing ones. To prevent over-fitting in subsequent downstream modeling, practitioners commonly use automated feature selection methods that identify a reduced subset of informative features. Existing benchmarks for tabular feature selection consider classical downstream models, toy synthetic datasets, or do not evaluate feature selectors on the basis of downstream performance. We construct a challenging feature selection benchmark evaluated on downstream neural networks including transformers, using real datasets and multiple methods for generating extraneous features. We also propose an input-gradient-based analogue of LASSO for neural networks that outperforms classical feature selection methods on challenging problems such as selecting from corrupted or second-order features.

        ----

        ## [1818] Riemannian Projection-free Online Learning

        **Authors**: *Zihao Hu, Guanghui Wang, Jacob D. Abernethy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8305a0049227f7dd2bb91e11090f8cfa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8305a0049227f7dd2bb91e11090f8cfa-Abstract-Conference.html)

        **Abstract**:

        The projection operation is a critical component in a wide range of optimization algorithms, such as online gradient descent (OGD), for enforcing constraints and achieving optimal regret bounds. However, it suffers from computational complexity limitations in high-dimensional settings or when dealing with ill-conditioned constraint sets. Projection-free algorithms address this issue by replacing the projection oracle with more efficient optimization subroutines. But to date, these methods have been developed primarily in the Euclidean setting, and while there has been growing interest in optimization on  Riemannian manifolds, there has been essentially no work in trying to utilize projection-free tools here. An apparent issue is that non-trivial affine functions  are generally non-convex in such domains. In this paper, we present methods for obtaining sub-linear regret guarantees in online geodesically convex optimization  on curved spaces for two scenarios: when we have access to (a) a separation oracle or (b) a linear optimization oracle. For geodesically convex losses, and  when a separation oracle is available, our algorithms achieve $O(T^{\frac{1}{2}})$, $O(T^{\frac{3}{4}})$ and $O(T^{\frac{1}{2}})$ adaptive regret guarantees in the full  information setting, the bandit setting with one-point feedback and the bandit setting with two-point feedback, respectively. When a linear optimization oracle is   available, we obtain regret rates of $O(T^{\frac{3}{4}})$ for geodesically convex losses and $O(T^{\frac{2}{3}}\log T)$ for strongly geodesically convex losses.

        ----

        ## [1819] SwiFT: Swin 4D fMRI Transformer

        **Authors**: *Peter Yongho Kim, Junbeom Kwon, Sunghwan Joo, Sangyoon Bae, Donggyu Lee, Yoonho Jung, Shinjae Yoo, Jiook Cha, Taesup Moon*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8313b1920ee9c78d846c5798c1ce48be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8313b1920ee9c78d846c5798c1ce48be-Abstract-Conference.html)

        **Abstract**:

        Modeling spatiotemporal brain dynamics from high-dimensional data, such as functional Magnetic Resonance Imaging (fMRI), is a formidable task in neuroscience. Existing approaches for fMRI analysis utilize hand-crafted features, but the process of feature extraction risks losing essential information in fMRI scans. To address this challenge, we present SwiFT (Swin 4D fMRI Transformer), a Swin Transformer architecture that can learn brain dynamics directly from fMRI volumes in a memory and computation-efficient manner. SwiFT achieves this by implementing a 4D window multi-head self-attention mechanism and absolute positional embeddings. We evaluate SwiFT using multiple large-scale resting-state fMRI datasets, including the Human Connectome Project (HCP), Adolescent Brain Cognitive Development (ABCD), and UK Biobank (UKB) datasets, to predict sex, age, and cognitive intelligence. Our experimental outcomes reveal that SwiFT consistently outperforms recent state-of-the-art models. Furthermore, by leveraging its end-to-end learning capability, we show that contrastive loss-based self-supervised pre-training of SwiFT can enhance performance on downstream tasks. Additionally, we employ an explainable AI method to identify the brain regions associated with sex classification. To our knowledge, SwiFT is the first Swin Transformer architecture to process dimensional spatiotemporal brain functional data in an end-to-end fashion. Our work holds substantial potential in facilitating scalable learning of functional brain imaging in neuroscience research by reducing the hurdles associated with applying Transformer models to high-dimensional fMRI.

        ----

        ## [1820] Consistent Diffusion Models: Mitigating Sampling Drift by Learning to be Consistent

        **Authors**: *Giannis Daras, Yuval Dagan, Alex Dimakis, Constantinos Daskalakis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/831406cfe7e4a0aed5ac5c8a8389d1f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/831406cfe7e4a0aed5ac5c8a8389d1f5-Abstract-Conference.html)

        **Abstract**:

        Imperfect score-matching leads to a shift between the training and the sampling distribution of diffusion models. Due to the recursive nature of the generation process, errors in previous steps yield sampling iterates that drift away from the training distribution. However, the standard training objective via Denoising Score Matching (DSM) is only designed to optimize over non-drifted data. To train on drifted data, we propose to enforce a \emph{Consistency} property (CP) which states that predictions of the model on its owngenerated data are consistent across time. Theoretically, we show that the differential equation that describes CP together with the one that describes a conservative vector field, have a unique solution given some initial condition. Consequently, if the score is learned well on non-drifted points via DSM (enforcing the true initial condition) then enforcing CP on drifted points propagates true score values. Empirically, we show that enforcing CP improves the generation quality for conditional and unconditional generation on CIFAR-10, and in AFHQ and FFHQ. We open-source our code and models: https://github.com/giannisdaras/cdm.

        ----

        ## [1821] A High-Resolution Dataset for Instance Detection with Multi-View Object Capture

        **Authors**: *Qianqian Shen, Yunhan Zhao, Nahyun Kwon, Jeeeun Kim, Yanan Li, Shu Kong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/832ea0ff01bd512aab28bf416db9489c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/832ea0ff01bd512aab28bf416db9489c-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Instance detection (InsDet) is a long-lasting problem in robotics and computer vision, aiming to detect object instances (predefined by some visual examples) in a cluttered scene. Despite its practical significance, its advancement is overshadowed by Object Detection, which aims to detect objects belonging to some predefined classes. One major reason is that current InsDet datasets are too small in scale by today's standards. For example, the popular InsDet dataset GMU (published in 2016) has only 23 instances, far less than  COCO (80 classes), a well-known object detection dataset published in 2014. We are motivated to introduce a new InsDet dataset and protocol. First, we define a realistic setup for InsDet: training data consists of multi-view instance captures, along with diverse scene images allowing synthesizing training images by pasting instance images on them with free box annotations. Second, we release a real-world database, which contains multi-view capture of 100 object instances, and high-resolution (6k$\times$8k) testing images. Third, we extensively study baseline methods for InsDet on our dataset, analyze their performance and suggest future work. Somewhat surprisingly, using the off-the-shelf  class-agnostic segmentation model (Segment Anything Model, SAM) and the self-supervised feature representation DINOv2 performs the best, achieving $>$10 AP better than end-to-end trained InsDet models that repurpose object detectors (e.g., FasterRCNN and RetinaNet).

        ----

        ## [1822] AVIDa-hIL6: A Large-Scale VHH Dataset Produced from an Immunized Alpaca for Predicting Antigen-Antibody Interactions

        **Authors**: *Hirofumi Tsuruta, Hiroyuki Yamazaki, Ryota Maeda, Ryotaro Tamura, Jennifer N. Wei, Zelda E. Mariet, Poomarin Phloyphisut, Hidetoshi Shimokawa, Joseph R. Ledsam, Lucy J. Colwell, Akihiro Imura*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8339dacd9df7ffe9623760f74169dd1e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8339dacd9df7ffe9623760f74169dd1e-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Antibodies have become an important class of therapeutic agents to treat human diseases.To accelerate therapeutic antibody discovery, computational methods, especially machine learning, have attracted considerable interest for predicting specific interactions between antibody candidates and target antigens such as viruses and bacteria.However, the publicly available datasets in existing works have notable limitations, such as small sizes and the lack of non-binding samples and exact amino acid sequences.To overcome these limitations, we have developed AVIDa-hIL6, a large-scale dataset for predicting antigen-antibody interactions in the variable domain of heavy chain of heavy chain antibodies (VHHs), produced from an alpaca immunized with the human interleukin-6 (IL-6) protein, as antigens.By leveraging the simple structure of VHHs, which facilitates identification of full-length amino acid sequences by DNA sequencing technology, AVIDa-hIL6 contains 573,891 antigen-VHH pairs with amino acid sequences.All the antigen-VHH pairs have reliable labels for binding or non-binding, as generated by a novel labeling method.Furthermore, via introduction of artificial mutations, AVIDa-hIL6 contains 30 different mutants in addition to wild-type IL-6 protein.This characteristic provides opportunities to develop machine learning models for predicting changes in antibody binding by antigen mutations.We report experimental benchmark results on AVIDa-hIL6 by using machine learning models.The results indicate that the existing models have potential, but further research is needed to generalize them to predict effective antibodies against unknown mutants.The dataset is available at https://avida-hil6.cognanous.com.

        ----

        ## [1823] Token-Scaled Logit Distillation for Ternary Weight Generative Language Models

        **Authors**: *Minsoo Kim, Sihwa Lee, Janghwan Lee, Sukjin Hong, Du-Seong Chang, Wonyong Sung, Jungwook Choi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8342218a4ec08b8c19661725e9cd6c0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8342218a4ec08b8c19661725e9cd6c0b-Abstract-Conference.html)

        **Abstract**:

        Generative Language Models (GLMs) have shown impressive performance in tasks such as text generation, understanding, and reasoning. However, the large model size poses challenges for practical deployment. To solve this problem, Quantization-Aware Training (QAT) has become increasingly popular. However, current QAT methods for generative models have resulted in a noticeable loss of accuracy. To counteract this issue, we propose a novel knowledge distillation method specifically designed for GLMs. Our method, called token-scaled logit distillation, prevents overfitting and provides superior learning from the teacher model and ground truth. This research marks the first evaluation of ternary weight quantization-aware training of large-scale GLMs with less than 1.0 degradation in perplexity and achieves enhanced accuracy in tasks like common-sense QA and arithmetic reasoning  as well as natural language understanding. Our code is available at https://github.com/aiha-lab/TSLD.

        ----

        ## [1824] Efficient Exploration in Continuous-time Model-based Reinforcement Learning

        **Authors**: *Lenart Treven, Jonas Hübotter, Bhavya Sukhija, Florian Dörfler, Andreas Krause*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/836012122f3de08aeeae67369b087964-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/836012122f3de08aeeae67369b087964-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning algorithms typically consider discrete-time dynamics, even though the underlying systems are often continuous in time. In this paper, we introduce a model-based reinforcement learning algorithm that represents continuous-time dynamics using nonlinear ordinary differential equations (ODEs). We capture epistemic uncertainty using well-calibrated probabilistic models, and use the optimistic principle for exploration. Our regret bounds surface the importance of the measurement selection strategy (MSS), since in continuous time we not only must decide how to explore, but also when to observe the underlying system. Our analysis demonstrates that the regret is sublinear when modeling ODEs with Gaussian Processes (GP) for common choices of MSS, such as equidistant sampling. Additionally, we propose an adaptive, data-dependent, practical MSS that, when combined with GP dynamics, also achieves sublinear regret with significantly fewer samples. We showcase the benefits of continuous-time modeling over its discrete-time counterpart, as well as our proposed adaptive MSS over standard baselines, on several applications.

        ----

        ## [1825] On Learning Necessary and Sufficient Causal Graphs

        **Authors**: *Hengrui Cai, Yixin Wang, Michael I. Jordan, Rui Song*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/837b396039248acb08c385bebb6291b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/837b396039248acb08c385bebb6291b4-Abstract-Conference.html)

        **Abstract**:

        The causal revolution has stimulated interest in understanding complex relationships in various fields. Most of the existing methods aim to discover causal relationships among all variables within a complex large-scale graph. However, in practice, only a small subset of variables in the graph are relevant to the outcomes of interest. Consequently, causal estimation with the full causal graph---particularly given limited data---could lead to numerous falsely discovered, spurious variables that exhibit high correlation with, but exert no causal impact on, the target outcome. In this paper, we propose learning a class of necessary and sufficient causal graphs (NSCG) that exclusively comprises causally relevant variables for an outcome of interest, which we term causal features. The key idea is to employ probabilities of causation to systematically evaluate the importance of features in the causal graph, allowing us to identify a subgraph relevant to the outcome of interest. To learn NSCG from data, we develop a necessary and sufficient causal structural learning (NSCSL) algorithm, by establishing theoretical properties and relationships between probabilities of causation and natural causal effects of features. Across empirical studies of simulated and real data, we demonstrate that NSCSL outperforms existing algorithms and can reveal crucial yeast genes for target heritable traits of interest.

        ----

        ## [1826] Renku: a platform for sustainable data science

        **Authors**: *Rok Roskar, Chandrasekhar Ramakrishnan, Michele Volpi, Fernando Pérez-Cruz, Lilian Gasser, Firat Ozdemir, Patrick Paitz, Mohammad Alisafaee, Philipp Fischer, Ralf Grubenmann, Eliza Harris, Tasko Olevski, Carl Remlinger, Luis Salamanca, Elisabet Capon Garcia, Lorenzo Cavazzi, Jakub Chrobasik, Darlin Cordoba Osnas, Alessandro Degano, Jimena Dupre, Wesley Johnson, Eike Kettner, Laura Kinkead, Sean D. Murphy, Flora Thiebaut, Olivier Verscheure*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/838694e9ab6b0a193b84daaafcac0eed-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/838694e9ab6b0a193b84daaafcac0eed-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Data and code working together is fundamental to machine learning (ML), but the context around datasets and interactions between datasets and code are in general captured only rudimentarily. Context such as how the dataset was prepared and created, what source data were used, what code was used in processing, how the dataset evolved, and where it has been used and reused can provide much insight, but this information is often poorly documented. That is unfortunate since it makes datasets into black-boxes with potentially hidden characteristics that have downstream consequences. We argue that making dataset preparation more accessible and dataset usage easier to record and document would have significant benefits for the ML community: it would allow for greater diversity in datasets by inviting modification to published sources, simplify use of alternative datasets and, in doing so, make results more transparent and robust, while allowing for all contributions to be adequately credited. We present a platform, Renku, designed to support and encourage such sustainable development and use of data, datasets, and code, and we demonstrate its benefits through a few illustrative projects which span the spectrum from dataset creation to dataset consumption and showcasing.

        ----

        ## [1827] Slimmed Asymmetrical Contrastive Learning and Cross Distillation for Lightweight Model Training

        **Authors**: *Jian Meng, Li Yang, Kyungmin Lee, Jinwoo Shin, Deliang Fan, Jae-sun Seo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8393d955a00c463a982cefe77d0404e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8393d955a00c463a982cefe77d0404e1-Abstract-Conference.html)

        **Abstract**:

        Contrastive learning (CL) has been widely investigated with various learning mechanisms and achieves strong capability in learning representations of data in a self-supervised manner using unlabeled data. A common fashion of contrastive learning on this line is employing mega-sized encoders to achieve comparable performance as the supervised learning counterpart. Despite the success of the labelless training, current contrastive learning algorithms *failed* to achieve good performance with lightweight (compact) models, e.g., MobileNet, while the requirements of the heavy encoders impede the energy-efficient computation, especially for resource-constrained AI applications. Motivated by this, we propose a new self-supervised CL scheme, named SACL-XD, consisting of two technical components, **S**limmed **A**symmetrical **C**ontrastive **L**earning (SACL) and **Cross**-**D**istillation (XD), which collectively enable efficient CL with compact models. While relevant prior works employed a strong pre-trained model as the teacher of unsupervised knowledge distillation to a lightweight encoder, our proposed method trains CL models from scratch and outperforms them even without such an expensive requirement. Compared to the SoTA lightweight CL training (distillation) algorithms, SACL-XD achieves 1.79% ImageNet-1K accuracy improvement on MobileNet-V3 with 64$\times$ training FLOPs reduction.

        ----

        ## [1828] Beyond Unimodal: Generalising Neural Processes for Multimodal Uncertainty Estimation

        **Authors**: *Myong Chol Jung, He Zhao, Joanna Dipnall, Lan Du*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/839e23e5b1c52cfd1268f4023a3af0d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/839e23e5b1c52cfd1268f4023a3af0d6-Abstract-Conference.html)

        **Abstract**:

        Uncertainty estimation is an important research area to make deep neural networks (DNNs) more trustworthy. While extensive research on uncertainty estimation has been conducted with unimodal data, uncertainty estimation for multimodal data remains a challenge. Neural processes (NPs) have been demonstrated to be an effective uncertainty estimation method for unimodal data by providing the reliability of Gaussian processes with efficient and powerful DNNs. While NPs hold significant potential for multimodal uncertainty estimation, the adaptation of NPs for multimodal data has not been carefully studied. To bridge this gap, we propose Multimodal Neural Processes (MNPs) by generalising NPs for multimodal uncertainty estimation. Based on the framework of NPs, MNPs consist of several novel and principled mechanisms tailored to the characteristics of multimodal data. In extensive empirical evaluation, our method achieves state-of-the-art multimodal uncertainty estimation performance, showing its appealing robustness against noisy samples and reliability in out-of-distribution detection with faster computation time compared to the current state-of-the-art multimodal uncertainty estimation method.

        ----

        ## [1829] Trans-Dimensional Generative Modeling via Jump Diffusion Models

        **Authors**: *Andrew Campbell, William Harvey, Christian Weilbach, Valentin De Bortoli, Thomas Rainforth, Arnaud Doucet*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83a10a480fbec91c88f6a9293b4d2b05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83a10a480fbec91c88f6a9293b4d2b05-Abstract-Conference.html)

        **Abstract**:

        We propose a new class of generative model that naturally handles data of varying dimensionality by jointly modeling the state and dimension of each datapoint. The generative process is formulated as a jump diffusion process that makes jumps between different dimensional spaces. We first define a dimension destroying forward noising process, before deriving the dimension creating time-reversed generative process along with a novel evidence lower bound training objective for learning to approximate it.Simulating our learned approximation to the time-reversed generative process then provides an effective way of sampling data of varying dimensionality by jointly generating state values and dimensions. We demonstrate our approach on molecular and video datasets of varying dimensionality, reporting better compatibility with test-time diffusion guidance imputation tasks and improved interpolation capabilities versus fixed dimensional models that generate state values and dimensions separately.

        ----

        ## [1830] Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication

        **Authors**: *Chaofei Fan, Nick Hahn, Foram Kamdar, Donald T. Avansino, Guy H. Wilson, Leigh R. Hochberg, Krishna V. Shenoy, Jaimie M. Henderson, Francis R. Willett*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83a14a36de4502bac5b580db36e81858-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83a14a36de4502bac5b580db36e81858-Abstract-Conference.html)

        **Abstract**:

        Intracortical brain-computer interfaces (iBCIs) have shown promise for restoring rapid communication to people with neurological disorders such as amyotrophic lateral sclerosis (ALS). However, to maintain high performance over time, iBCIs typically need frequent recalibration to combat changes in the neural recordings that accrue over days. This requires iBCI users to stop using the iBCI and engage in supervised data collection, making the iBCI system hard to use. In this paper, we propose a method that enables self-recalibration of communication iBCIs without interrupting the user. Our method leverages large language models (LMs) to automatically correct errors in iBCI outputs. The self-recalibration process uses these corrected outputs ("pseudo-labels") to continually update the iBCI decoder online. Over a period of more than one year (403 days), we evaluated our Continual Online Recalibration with Pseudo-labels (CORP) framework with one clinical trial participant. CORP achieved  a stable decoding accuracy of 93.84% in an online handwriting iBCI task, significantly outperforming other baseline methods. Notably, this is the longest-running iBCI stability demonstration involving a human participant. Our results provide the first  evidence for long-term stabilization of a plug-and-play, high-performance communication iBCI, addressing a major barrier for the clinical translation of iBCIs.

        ----

        ## [1831] Towards robust and generalizable representations of extracellular data using contrastive learning

        **Authors**: *Ankit Vishnubhotla, Charlotte Loh, Akash Srivastava, Liam Paninski, Cole L. Hurwitz*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83c637c3bc0ca88eda6cf4f5f45bdced-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83c637c3bc0ca88eda6cf4f5f45bdced-Abstract-Conference.html)

        **Abstract**:

        Contrastive learning is quickly becoming an essential tool in neuroscience for extracting robust and meaningful representations of neural activity. Despite numerous applications to neuronal population data, there has been little exploration of how these methods can be adapted to key primary data analysis tasks such as spike sorting or cell-type classification. In this work, we propose a novel contrastive learning framework, CEED (Contrastive Embeddings for Extracellular Data), for high-density extracellular recordings. We demonstrate that through careful design of the network architecture and data augmentations, it is possible to generically extract representations that far outperform current specialized approaches. We validate our method across multiple high-density extracellular recordings. All code used to run CEED can be found at https://github.com/ankitvishnu23/CEED.

        ----

        ## [1832] Rethinking Conditional Diffusion Sampling with Progressive Guidance

        **Authors**: *Anh-Dung Dinh, Daochang Liu, Chang Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83ca9e252329e7b0704ead93893e6b1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83ca9e252329e7b0704ead93893e6b1b-Abstract-Conference.html)

        **Abstract**:

        This paper tackles two critical challenges encountered in classifier guidance for diffusion generative models, i.e., the lack of diversity and the presence of adversarial effects. These issues often result in a scarcity of diverse samples or the generation of non-robust features. The underlying cause lies in the mechanism of classifier guidance, where discriminative gradients push samples to be recognized as conditions aggressively. This inadvertently suppresses information with common features among relevant classes, resulting in a limited pool of features with less diversity or the absence of robust features for image construction.We propose a generalized classifier guidance method called Progressive Guidance, which mitigates the problems by allowing relevant classes' gradients to contribute to shared information construction when the image is noisy in early sampling steps. In the later sampling stage, we progressively enhance gradients to refine the details in the image toward the primary condition. This helps to attain a high level of diversity and robustness compared to the vanilla classifier guidance. Experimental results demonstrate that our proposed method further improves the image quality while offering a significant level of diversity as well as robust features.

        ----

        ## [1833] State-Action Similarity-Based Representations for Off-Policy Evaluation

        **Authors**: *Brahma S. Pavse, Josiah Hanna*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83dc5747870ea454cab25e30bef4eb8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/83dc5747870ea454cab25e30bef4eb8a-Abstract-Conference.html)

        **Abstract**:

        In reinforcement learning, off-policy evaluation (OPE) is the problem of estimating the expected return of an evaluation policy given a fixed dataset that was collected by running one or more different policies. One of the more empirically successful algorithms for OPE has been the fitted q-evaluation (FQE) algorithm that uses temporal difference updates to learn an action-value function, which is then used to estimate the expected return of the evaluation policy. Typically, the original fixed dataset is fed directly into FQE to learn the action-value function of the evaluation policy. Instead, in this paper, we seek to enhance the data-efficiency of FQE by first transforming the fixed dataset using a learned encoder, and then feeding the transformed dataset into FQE. To learn such an encoder, we introduce an OPE-tailored state-action behavioral similarity metric, and use this metric and the fixed dataset to  learn an encoder that models this metric. Theoretically, we show that this metric allows us to bound the error in the resulting OPE estimate. Empirically, we show that other state-action similarity metrics lead to representations that cannot represent the action-value function of the evaluation policy, and that our state-action representation method boosts the data-efficiency of FQE and lowers OPE error relative to other OPE-based representation learning methods on challenging OPE tasks. We also empirically show that the learned representations significantly mitigate divergence of FQE under varying distribution shifts. Our code is available here: https://github.com/Badger-RL/ROPE.

        ----

        ## [1834] Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs

        **Authors**: *Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin Chen-Chuan Chang, Fei Huang, Reynold Cheng, Yongbin Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/83fc8fab1710363050bbd1d4b8cc0021-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/83fc8fab1710363050bbd1d4b8cc0021-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Text-to-SQL parsing, which aims at converting natural language instructions into executable SQLs, has gained increasing attention in recent years.  In particular, GPT-4 and Claude-2 have shown impressive results in this task. However, most of the prevalent benchmarks, i.e., Spider, and WikiSQL, focus on database schema with few rows of database contents leaving the gap between academic study and real-world applications. To mitigate this gap, we present BIRD, a BIg benchmark for laRge-scale Database grounded in text-to-SQL tasks, containing 12,751 pairs of text-to-SQL data and 95 databases with a total size of 33.4 GB, spanning 37 professional domains. Our emphasis on database values highlights the new challenges of dirty database contents, external knowledge between NL questions and database contents,  and SQL efficiency, particularly in the context of massive databases. To solve these problems, text-to-SQL models must feature database value comprehension in addition to semantic parsing. The experimental results demonstrate the significance of database values in generating accurate text-to-SQLs for big databases. Furthermore, even the most popular and effective text-to-SQL models, i.e. GPT-4, only achieve 54.89% in execution accuracy, which is still far from the human result of 92.96%, proving that challenges still stand. We also provide an efficiency analysis to offer insights into generating text-to-efficient-SQLs that are beneficial to industries. We believe that BIRD will contribute to advancing real-world applications of text-to-SQL research.The leaderboard and source code are available: https://bird-bench.github.io/.

        ----

        ## [1835] CARE-MI: Chinese Benchmark for Misinformation Evaluation in Maternity and Infant Care

        **Authors**: *Tong Xiang, Liangzhi Li, Wangyue Li, Mingbai Bai, Lu Wei, Bowen Wang, Noa Garcia*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84062fe53d23e0791c6dbb456783e4a9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/84062fe53d23e0791c6dbb456783e4a9-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The recent advances in natural language processing (NLP), have led to a new trend of applying large language models (LLMs) to real-world scenarios. While the latest LLMs are astonishingly fluent when interacting with humans, they suffer from the misinformation problem by unintentionally generating factually false statements. This can lead to harmful consequences, especially when produced within sensitive contexts, such as healthcare. Yet few previous works have focused on evaluating misinformation in the long-form (LF) generation of LLMs, especially for knowledge-intensive topics. Moreover, although LLMs have been shown to perform well in different languages, misinformation evaluation has been mostly conducted in English. To this end, we present a benchmark, CARE-MI, for evaluating LLM misinformation in: 1) a sensitive topic, specifically the maternity and infant care domain; and 2) a language other than English, namely Chinese. Most importantly, we provide an innovative paradigm for building LF generation evaluation benchmarks that can be transferred to other knowledge-intensive domains and low-resourced languages. Our proposed benchmark fills the gap between the extensive usage of LLMs and the lack of datasets for assessing the misinformation generated by these models. It contains 1,612 expert-checked questions, accompanied with human-selected references. Using our benchmark, we conduct extensive experiments and found that current Chinese LLMs are far from perfect in the topic of maternity and infant care. In an effort to minimize the reliance on human resources for performance evaluation, we offer off-the-shelf judgment models for automatically assessing the LF output of LLMs given benchmark questions. Moreover, we compare potential solutions for LF generation evaluation and provide insights for building better automated metrics.

        ----

        ## [1836] Explore In-Context Learning for 3D Point Cloud Understanding

        **Authors**: *Zhongbin Fang, Xiangtai Li, Xia Li, Joachim M. Buhmann, Chen Change Loy, Mengyuan Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8407d254b5baacf69ee977aa34f0e521-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8407d254b5baacf69ee977aa34f0e521-Abstract-Conference.html)

        **Abstract**:

        With the rise of large-scale models trained on broad data, in-context learning has become a new learning paradigm that has demonstrated significant potential in natural language processing and computer vision tasks. Meanwhile, in-context learning is still largely unexplored in the 3D point cloud domain. Although masked modeling has been successfully applied for in-context learning in 2D vision, directly extending it to 3D point clouds remains a formidable challenge. In the case of point clouds, the tokens themselves are the point cloud positions (coordinates) that are masked during inference. Moreover, position embedding in previous works may inadvertently introduce information leakage. To address these challenges, we introduce a novel framework, named Point-In-Context, designed especially for in-context learning in 3D point clouds, where both inputs and outputs are modeled as coordinates for each task. Additionally, we propose the Joint Sampling module, carefully designed to work in tandem with the general point sampling operator, effectively resolving the aforementioned technical issues. We conduct extensive experiments to validate the versatility and adaptability of our proposed methods in handling a wide range of tasks. Furthermore, with a more effective prompt selection strategy, our framework surpasses the results of individually trained models.

        ----

        ## [1837] Learning Re-sampling Methods with Parameter Attribution for Image Super-resolution

        **Authors**: *Xiaotong Luo, Yuan Xie, Yanyun Qu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8434e0db3227276c00ef2b18c7f01c65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8434e0db3227276c00ef2b18c7f01c65-Abstract-Conference.html)

        **Abstract**:

        Single image super-resolution (SISR) has made a significant breakthrough benefiting from the prevalent rise of deep neural networks and large-scale training samples. The mainstream deep SR models primarily focus on network architecture design as well as optimization schemes, while few pay attention to the training data. In fact, most of the existing SR methods train the model on uniformly sampled patch pairs from the whole image. However, the uneven image content makes the training data present an unbalanced distribution, i.e., the easily reconstructed region (smooth) occupies the majority of the data, while the hard reconstructed region (edge or texture) has rarely few samples. Based on this phenomenon, we consider rethinking the current paradigm of merely using uniform data sampling way for training SR models. In this paper, we propose a simple yet effective Bi-Sampling Parameter Attribution (BSPA) method for accurate image SR. Specifically, the bi-sampling consists of uniform sampling and inverse sampling, which is introduced to reconcile the unbalanced inherent data bias. The former aims to keep the intrinsic data distribution, and the latter is designed to enhance the feature extraction ability of the model on the hard samples. Moreover, integrated gradient is introduced to attribute the contribution of each parameter in the alternate models trained by both sampling data so as to filter the trivial parameters for further dynamic refinement. By progressively decoupling the allocation of parameters, the SR model can learn a more compact representation. Extensive experiments on publicly available datasets demonstrate that our proposal can effectively boost the performance of baseline methods from the data re-sampling view.

        ----

        ## [1838] Interactive Visual Reasoning under Uncertainty

        **Authors**: *Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/844f722dbbcb27933ff5baf58a1f00c8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/844f722dbbcb27933ff5baf58a1f00c8-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        One of the fundamental cognitive abilities of humans is to quickly resolve uncertainty by generating hypotheses and testing them via active trials. Encountering a novel phenomenon accompanied by ambiguous cause-effect relationships, humans make hypotheses against data, conduct inferences from observation, test their theory via experimentation, and correct the proposition if inconsistency arises. These iterative processes persist until the underlying mechanism becomes clear. In this work, we devise the  IVRE (pronounced as "ivory") environment for evaluating artificial agents' reasoning ability under uncertainty. IVRE is an interactive environment featuring rich scenarios centered around Blicket detection. Agents in IVRE are placed into environments with various ambiguous action-effect pairs and asked to determine each object's role. They are encouraged to propose effective and efficient experiments to validate their hypotheses based on observations and actively gather new information. The game ends when all uncertainties are resolved or the maximum number of trials is consumed. By evaluating modern artificial agents in IVRE, we notice a clear failure of today's learning methods compared to humans. Such inefficacy in interactive reasoning ability under uncertainty calls for future research in building human-like intelligence.

        ----

        ## [1839] Generative Modeling through the Semi-dual Formulation of Unbalanced Optimal Transport

        **Authors**: *Jaemoo Choi, Jaewoong Choi, Myungjoo Kang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84706cdfc192cd0351daf48f379847e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/84706cdfc192cd0351daf48f379847e6-Abstract-Conference.html)

        **Abstract**:

        Optimal Transport (OT) problem investigates a transport map that bridges two distributions while minimizing a given cost function. In this regard, OT between tractable prior distribution and data has been utilized for generative modeling tasks. However, OT-based methods are susceptible to outliers and face optimization challenges during training. In this paper, we propose a novel generative model based on the semi-dual formulation of Unbalanced Optimal Transport (UOT). Unlike OT, UOT relaxes the hard constraint on distribution matching. This approach provides better robustness against outliers, stability during training, and faster convergence. We validate these properties empirically through experiments. Moreover, we study the theoretical upper-bound of divergence between distributions in UOT. Our model outperforms existing OT-based generative models, achieving FID scores of 2.97 on CIFAR-10 and 6.36 on CelebA-HQ-256. The code is available at \url{https://github.com/Jae-Moo/UOTM}.

        ----

        ## [1840] Generalized Belief Transport

        **Authors**: *Junqi Wang, Pei Wang, Patrick Shafto*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/847bb9bb1351f557a52d2ecdacb7e2d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/847bb9bb1351f557a52d2ecdacb7e2d3-Abstract-Conference.html)

        **Abstract**:

        Human learners have ability to adopt appropriate learning approaches depending on constraints such as prior on the hypothesis, urgency of decision, and drift of the environment. However, existing learning models are typically considered individually rather than in relation to one and other. To build agents that have the ability to move between different modes of learning over time, it is important to understand how learning models are related as points in a broader space of possibilities. We introduce a mathematical framework, Generalized Belief Transport (GBT), that unifies and generalizes prior models, including Bayesian inference, cooperative communication and classification, as parameterizations of three learning constraints within Unbalanced Optimal Transport (UOT). We visualize the space of learning models encoded by GBT as a cube which includes classic learning models as special points. We derive critical properties of this parameterized space including proving continuity and differentiability which is the basis for model interpolation, and study limiting behavior of the parameters, which allows attaching learning models on the boundaries. Moreover, we investigate the long-run behavior of GBT, explore convergence properties of models in GBT mathematical and computationally, document the ability to learn in the presence of distribution drift, and formulate conjectures about general behavior. We conclude with open questions and implications for more unified models of learning.

        ----

        ## [1841] Lie Point Symmetry and Physics-Informed Networks

        **Authors**: *Tara Akhound-Sadegh, Laurence Perreault Levasseur, Johannes Brandstetter, Max Welling, Siamak Ravanbakhsh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8493c860bec41705f7743d5764301b94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8493c860bec41705f7743d5764301b94-Abstract-Conference.html)

        **Abstract**:

        Symmetries have been leveraged to improve the generalization of neural networks through different mechanisms from data augmentation to equivariant architectures. However, despite their potential, their integration into neural solvers for partial differential equations (PDEs) remains largely unexplored. We explore the integration of PDE symmetries, known as Lie point symmetries, in a major family of neural solvers known as physics-informed neural networks (PINNs). We propose a loss function that informs the network about Lie point symmetries in the same way that PINN models try to enforce the underlying PDE through a loss function. Intuitively, our symmetry loss ensures that the infinitesimal generators of the Lie group conserve the PDE solutions.. Effectively, this means that once the network learns a solution, it also learns the neighbouring solutions generated by Lie point symmetries.Empirical evaluations indicate that the inductive bias introduced by the Lie point symmetries of the PDEs greatly boosts the sample efficiency of PINNs.

        ----

        ## [1842] Norm-based Generalization Bounds for Sparse Neural Networks

        **Authors**: *Tomer Galanti, Mengjia Xu, Liane Galanti, Tomaso A. Poggio*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8493e190ff1bbe3837eca821190b61ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8493e190ff1bbe3837eca821190b61ff-Abstract-Conference.html)

        **Abstract**:

        In this paper, we derive norm-based generalization bounds for sparse ReLU neural networks, including convolutional neural networks. These bounds differ from previous ones because they consider the sparse structure of the neural network architecture and the norms of the convolutional filters, rather than the norms of the (Toeplitz) matrices associated with the convolutional layers. Theoretically, we demonstrate that these bounds are significantly tighter than standard norm-based generalization bounds. Empirically, they offer relatively tight estimations of generalization for various simple classification problems. Collectively, these findings suggest that the sparsity of the underlying target function and the model's architecture plays a crucial role in the success of deep learning.

        ----

        ## [1843] Intelligent Knee Sleeves: A Real-time Multimodal Dataset for 3D Lower Body Motion Estimation Using Smart Textile

        **Authors**: *Wenwen Zhang, Arvin Tashakori, Zenan Jiang, Amir Servati, Harishkumar Narayana, Saeid Soltanian, Rou Yi Yeap, Meng Han Ma, Lauren Toy, Peyman Servati*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84948f178cfd3f6a0ffecda8fdcb3488-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/84948f178cfd3f6a0ffecda8fdcb3488-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The kinematics of human movements and locomotion are closely linked to the activation and contractions of muscles. To investigate this, we present a multimodal dataset with benchmarks collected using a novel pair of Intelligent Knee Sleeves (Texavie MarsWear Knee Sleeves) for human pose estimation. Our system utilizes synchronized datasets that comprise time-series data from the Knee Sleeves and the corresponding ground truth labels from visualized motion capture camera system. We employ these to generate 3D human models solely based on the wearable data of individuals performing different activities. We demonstrate the effectiveness of this camera-free system and machine learning algorithms in the assessment of various movements and exercises, including extension to unseen exercises and individuals. The results show an average error of 7.21 degrees across all eight lower body joints when compared to the ground truth, indicating the effectiveness and reliability of the Knee Sleeve system for the prediction of different lower body joints beyond knees. The results enable human pose estimation in a seamless manner without being limited by visual occlusion or the field of view of cameras. Our results show the potential of multimodal wearable sensing in a variety of applications from home fitness to sports, healthcare, and physical rehabilitation focusing on pose and movement estimation.

        ----

        ## [1844] Neural Injective Functions for Multisets, Measures and Graphs via a Finite Witness Theorem

        **Authors**: *Tal Amir, Steven J. Gortler, Ilai Avni, Ravina Ravina, Nadav Dym*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84b686f7cc7b7751e9aaac0da74f755a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/84b686f7cc7b7751e9aaac0da74f755a-Abstract-Conference.html)

        **Abstract**:

        Injective multiset functions have a key role in the theoretical study of machine learning on multisets and graphs. Yet, there remains a gap between the provably injective multiset functions considered in theory, which typically rely on polynomial moments, and the multiset functions used in practice, which rely on $\textit{neural moments}$ â€” whose injectivity on multisets has not been studied to date.In this paper, we bridge this gap by showing that moments of neural networks do define injective multiset functions,  provided that an analytic non-polynomial activation is used. The number of moments required by our theory is optimal essentially up to a multiplicative factor of two. To prove this result, we state and prove a $\textit{finite witness theorem}$, which is of independent interest. As a corollary to our main theorem, we derive new approximation results for functions on multisets and measures, and new separation results for graph neural networks. We also provide two negative results: (1) moments of piecewise-linear neural networks cannot be injective multiset functions; and (2) even when moment-based multiset functions are injective, they can never be bi-Lipschitz.

        ----

        ## [1845] Online Ad Procurement in Non-stationary Autobidding Worlds

        **Authors**: *Jason Cheuk Nam Liang, Haihao Lu, Baoyu Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/84bad835faaf48f24d990072bb5b80ee-Abstract-Conference.html)

        **Abstract**:

        Today's online advertisers procure digital ad impressions through interacting with autobidding platforms: advertisers convey high level procurement goals via setting levers such as budget, target return-on-investment, max cost per click, etc.. Then ads platforms subsequently procure impressions on advertisers' behalf, and report final procurement conversions (e.g. click) to advertisers. In practice, advertisers may receive minimal information on platforms' procurement details, and procurement outcomes  are subject to non-stationary factors like seasonal patterns, occasional system corruptions, and market trends which make it difficult for advertisers to optimize lever decisions effectively. Motivated by this,  we present an online learning framework that helps advertisers dynamically optimize ad platform lever decisions while subject to general long-term constraints in a realistic bandit feedback environment with non-stationary procurement outcomes. In particular, we introduce a primal-dual algorithm for online decision making with multi-dimension decision variables, bandit feedback and long-term uncertain constraints. We show that our algorithm achieves low regret in many worlds when procurement outcomes are generated through procedures that are stochastic, adversarial, adversarially corrupted, periodic, and ergodic, respectively, without having to know which procedure is the ground truth. Finally, we emphasize that our proposed algorithm and theoretical results extend beyond the applications of online advertising.

        ----

        ## [1846] Precise asymptotic generalization for multiclass classification with overparameterized linear models

        **Authors**: *David X. Wu, Anant Sahai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/84f44b36ceb4fbc9bb269959f4796eed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/84f44b36ceb4fbc9bb269959f4796eed-Abstract-Conference.html)

        **Abstract**:

        We study the asymptotic generalization of an overparameterized linear model for multiclass classification under the Gaussian covariates bi-level model introduced in Subramanian et al. (NeurIPS'22), where the number of data points, features, and classes all grow together. We fully resolve the conjecture posed in Subramanian et al. '22, matching the predicted regimes for which the model does and does not generalize. Furthermore, our new lower bounds are akin to an information-theoretic strong converse: they establish that the misclassification rate goes to 0 or 1 asymptotically. One surprising consequence of our tight results is that the min-norm interpolating classifier can be asymptotically suboptimal relative to noninterpolating classifiers in the regime where the min-norm interpolating regressor is known to be optimal. The key to our tight analysis is a new variant of the Hanson-Wright inequality which is broadly useful for multiclass problems with sparse labels. As an application, we show that the same type of analysis can be used to analyze the related multi-label classification problem under the same bi-level ensemble.

        ----

        ## [1847] Break It Down: Evidence for Structural Compositionality in Neural Networks

        **Authors**: *Michael A. Lepori, Thomas Serre, Ellie Pavlick*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85069585133c4c168c865e65d72e9775-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85069585133c4c168c865e65d72e9775-Abstract-Conference.html)

        **Abstract**:

        Though modern neural networks have achieved impressive performance in both vision and language tasks, we know little about the functions that they implement. One possibility is that neural networks implicitly break down complex tasks into subroutines, implement modular solutions to these subroutines, and compose them into an overall solution to a task --- a property we term structural compositionality.  Another possibility is that they may simply learn to match new inputs to learned templates, eliding task decomposition entirely. Here, we leverage model pruning techniques to investigate this question in both vision and language across a variety of architectures, tasks, and pretraining regimens. Our results demonstrate that models oftentimes implement solutions to subroutines via modular subnetworks, which can be ablated while maintaining the functionality of other subnetworks. This suggests that neural networks may be able to learn compositionality, obviating the need for specialized symbolic mechanisms.

        ----

        ## [1848] Focused Transformer: Contrastive Training for Context Scaling

        **Authors**: *Szymon Tworkowski, Konrad Staniszewski, Mikolaj Pacek, Yuhuai Wu, Henryk Michalewski, Piotr Milos*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8511d06d5590f4bda24d42087802cc81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8511d06d5590f4bda24d42087802cc81-Abstract-Conference.html)

        **Abstract**:

        Large language models have an exceptional capability to incorporate new information in a contextual manner. However, the full potential of such an approach is often restrained due to a limitation in the effective context length. One solution to this issue is to endow an attention layer with access to an additional context, which comprises of (key, value) pairs. Yet, as the number of documents increases, the proportion of relevant keys to irrelevant ones decreases, leading the model to focus more on the irrelevant keys. We identify a significant challenge, dubbed the distraction issue, where keys linked to different semantic values might overlap, making them hard to distinguish. To tackle this problem, we introduce the Focused Transformer (FoT), a technique that employs a training process inspired by contrastive learning. This novel approach enhances the structure of the (key, value) space, enabling an extension of the context length. Our method allows for fine-tuning pre-existing, large-scale models to lengthen their effective context. This is demonstrated by our fine-tuning of $3 B$ and $7 B$ OpenLLaMA checkpoints. The resulting models, which we name LongLLaMA, exhibit advancements in tasks requiring a long context. We further illustrate that our LongLLaMA models adeptly manage a $256 k$ context length for passkey retrieval.

        ----

        ## [1849] Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone

        **Authors**: *Zeyinzi Jiang, Chaojie Mao, Ziyuan Huang, Ao Ma, Yiliang Lv, Yujun Shen, Deli Zhao, Jingren Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8514a5203b87cba5e440bd62ab18f2b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8514a5203b87cba5e440bd62ab18f2b4-Abstract-Conference.html)

        **Abstract**:

        Parameter-efficient tuning has become a trend in transferring large-scale foundation models to downstream applications. Existing methods typically embed some light-weight tuners into the backbone, where both the design and the learning of the tuners are highly dependent on the base model. This work offers a new tuning paradigm, dubbed Res-Tuning, which intentionally unbinds tuners from the backbone. With both theoretical and empirical evidence, we show that popular tuning approaches have their equivalent counterparts under our unbinding formulation, and hence can be integrated into our framework effortlessly. Thanks to the structural disentanglement, we manage to free the design of tuners from the network architecture, facilitating flexible combination of various tuning strategies. We further propose a memory-efficient variant of Res-Tuning, where the bypass i.e., formed by a sequence of tuners) is effectively detached from the main branch, such that the gradients are back-propagated only to the tuners but not to the backbone. Such a detachment also allows one-time backbone forward for multi-task inference. Extensive experiments on both discriminative and generative tasks demonstrate the superiority of our method over existing alternatives from the perspectives of efficacy and efficiency. Project page: https://res-tuning.github.io/.

        ----

        ## [1850] Learning Energy-Based Prior Model with Diffusion-Amortized MCMC

        **Authors**: *Peiyu Yu, Yaxuan Zhu, Sirui Xie, Xiaojian Ma, Ruiqi Gao, Song-Chun Zhu, Ying Nian Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85381f4549b5ddf1d48e2e287d7d3d15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85381f4549b5ddf1d48e2e287d7d3d15-Abstract-Conference.html)

        **Abstract**:

        Latent space EBMs, also known as energy-based priors, have drawn growing interests in the field of generative modeling due to its flexibility in the formulation and strong modeling power of the latent space. However, the common practice of learning latent space EBMs with non-convergent short-run MCMC for prior and posterior sampling is hindering the model from further progress; the degenerate MCMC sampling quality in practice often leads to degraded generation quality and instability in training, especially with highly multi-modal and/or high-dimensional target distributions. To remedy this sampling issue, in this paper we introduce a simple but effective diffusion-based amortization method for long-run MCMC sampling and develop a novel learning algorithm for the latent space EBM based on it. We provide theoretical evidence that the learned amortization of MCMC is a valid long-run MCMC sampler. Experiments on several image modeling benchmark datasets demonstrate the superior performance of our method compared with strong counterparts.

        ----

        ## [1851] Perception Test: A Diagnostic Benchmark for Multimodal Video Models

        **Authors**: *Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adrià Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alexandre Fréchette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, João Carreira*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8540fba4abdc7f9f7a7b1cc6cd60e409-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8540fba4abdc7f9f7a7b1cc6cd60e409-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We propose a novel multimodal video benchmark - the Perception Test - to evaluate the perception and reasoning skills of pre-trained multimodal models (e.g. Flamingo, BEiT-3, or GPT-4). Compared to existing benchmarks that focus on computational tasks (e.g. classification, detection or tracking), the Perception Test focuses on skills (Memory, Abstraction, Physics, Semantics) and types of reasoning (descriptive, explanatory, predictive, counterfactual) across video, audio, and text modalities, to provide a comprehensive and efficient evaluation tool. The benchmark probes pre-trained models for their transfer capabilities, in a zero-shot / few-shot or limited finetuning regime. For these purposes, the Perception Test introduces 11.6k real-world videos, 23s average length, designed to show perceptually interesting situations, filmed by around 100 participants worldwide. The videos are densely annotated with six types of labels (multiple-choice and grounded video question-answers, object and point tracks, temporal action and sound segments), enabling both language and non-language evaluations. The fine-tuning and validation splits of the benchmark are publicly available (CC-BY license), in addition to a challenge server with a held-out test split. Human baseline results compared to state-of-the-art video QA models show a significant gap in performance (91.4% vs 45.8%), suggesting that there is significant room for improvement in multimodal video understanding.Dataset, baselines code, and challenge server are available at https://github.com/deepmind/perception_test

        ----

        ## [1852] Directed Cyclic Graph for Causal Discovery from Multivariate Functional Data

        **Authors**: *Saptarshi Roy, Raymond K. W. Wong, Yang Ni*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/854a9ab0f323b841955e70ca383b27d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/854a9ab0f323b841955e70ca383b27d1-Abstract-Conference.html)

        **Abstract**:

        Discovering causal relationship using multivariate functional data has received a significant amount of attention very recently. In this article, we introduce a functional linear structural equation model for causal structure learning when the underlying graph involving the multivariate functions may have cycles. To enhance interpretability, our model involves a low-dimensional causal embedded space such that all the relevant causal information in the multivariate functional data is preserved in this lower-dimensional subspace. We prove that the proposed model is causally identifiable under standard assumptions that are often made in the causal discovery literature. To carry out inference of our model, we develop a fully Bayesian framework with suitable prior specifications and uncertainty quantification through posterior summaries. We illustrate the superior performance of our method over existing methods in terms of causal graph estimation through extensive simulation studies. We also demonstrate the proposed method using a brain EEG dataset.

        ----

        ## [1853] Do SSL Models Have Déjà Vu? A Case of Unintended Memorization in Self-supervised Learning

        **Authors**: *Casey Meehan, Florian Bordes, Pascal Vincent, Kamalika Chaudhuri, Chuan Guo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/854b6ec839294bf332db0d86e2f83c3f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/854b6ec839294bf332db0d86e2f83c3f-Abstract-Conference.html)

        **Abstract**:

        Self-supervised learning (SSL) algorithms can produce useful image representations by learning to associate different parts of natural images with one another. However, when taken to the extreme, SSL models can unintendedly memorize specific parts in individual training samples rather than learning semantically meaningful associations. In this work, we perform a systematic study of the unintended memorization of image-specific information in SSL models -- which we refer to as déjà vu memorization. Concretely, we show that given the trained model and a crop of a training image containing only the background (e.g., water, sky, grass), it is possible to infer the foreground object with high accuracy or even visually reconstruct it. Furthermore, we show that déjà vu memorization is common to different SSL algorithms, is exacerbated by certain design choices, and cannot be detected by conventional techniques for evaluating representation quality. Our study of déjà vu memorization reveals previously unknown privacy risks in SSL models, as well as suggests potential practical mitigation strategies.

        ----

        ## [1854] Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes

        **Authors**: *Ali Younis, Erik B. Sudderth*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85b2ff7574ef265f3a4800db9112ce14-Abstract-Conference.html)

        **Abstract**:

        Particle filters flexibly represent multiple posterior modes nonparametrically, via a collection of weighted samples, but have classically been applied to tracking problems with known dynamics and observation likelihoods.  Such generative models may be inaccurate or unavailable for high-dimensional observations like images.  We instead leverage training data to discriminatively learn particle-based representations of uncertainty in latent object states, conditioned on arbitrary observations via deep neural network encoders.   While prior discriminative particle filters have used heuristic relaxations of discrete particle resampling, or biased learning by truncating gradients at resampling steps, we achieve unbiased and low-variance gradient estimates by representing posteriors as continuous mixture densities.  Our theory and experiments expose dramatic failures of existing reparameterization-based estimators for mixture gradients, an issue we address via an importance-sampling gradient estimator. Unlike standard recurrent neural networks, our mixture density particle filter represents multimodal uncertainty in continuous latent states, improving accuracy and robustness.  On a range of challenging tracking and robot localization problems, our approach achieves dramatic improvements in accuracy, will also showing much greater stability across multiple training runs.

        ----

        ## [1855] Benchmarking Robustness to Adversarial Image Obfuscations

        **Authors**: *Florian Stimberg, Ayan Chakrabarti, Chun-Ta Lu, Hussein Hazimeh, Otilia Stretcu, Wei Qiao, Yintao Liu, Merve Kaya, Cyrus Rashtchian, Ariel Fuxman, Mehmet Tek, Sven Gowal*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85c123f6da0fa159eb249e6a2e171903-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/85c123f6da0fa159eb249e6a2e171903-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Automated content filtering and moderation is an important tool that allows online platforms to build striving user communities that facilitate cooperation and prevent abuse. Unfortunately, resourceful actors try to bypass automated filters in a bid to post content that violate platform policies and codes of conduct. To reach this goal, these malicious actors may obfuscate policy violating images (e.g., overlay harmful images by carefully selected benign images or visual patterns) to prevent machine learning models from reaching the correct decision. In this paper, we invite researchers to tackle this specific issue and present a new image benchmark. This benchmark, based on ImageNet, simulates the type of obfuscations created by malicious actors. It goes beyond Image-Net-C and ImageNet-C-bar by proposing general, drastic, adversarial modifications that preserve the original content intent. It aims to tackle a more common adversarial threat than the one considered by lp-norm bounded adversaries. We evaluate 33 pretrained models on the benchmark and train models with different augmentations, architectures and training methods on subsets of the obfuscations to measure generalization. Our hope is that this benchmark will encourage researchers to test their models and methods and try to find new approaches that are more robust to these obfuscations.

        ----

        ## [1856] Learning Curves for Deep Structured Gaussian Feature Models

        **Authors**: *Jacob A. Zavatone-Veth, Cengiz Pehlevan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85d456fd41f3eec83bd3b0c337037a0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85d456fd41f3eec83bd3b0c337037a0e-Abstract-Conference.html)

        **Abstract**:

        In recent years, significant attention in deep learning theory has been devoted to analyzing when models that interpolate their training data can still generalize well to unseen examples. Many insights have been gained from studying models with multiple layers of Gaussian random features, for which one can compute precise generalization asymptotics. However, few works have considered the effect of weight anisotropy; most assume that the random features are generated using independent and identically distributed Gaussian weights, and allow only for structure in the input data. Here, we use the replica trick from statistical physics to derive learning curves for models with many layers of structured Gaussian features. We show that allowing correlations between the rows of the first layer of features can aid generalization, while structure in later layers is generally detrimental. Our results shed light on how weight structure affects generalization in a simple class of solvable models.

        ----

        ## [1857] Mirror Diffusion Models for Constrained and Watermarked Generation

        **Authors**: *Guan-Horng Liu, Tianrong Chen, Evangelos A. Theodorou, Molei Tao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/85f5c7372625d1e0df0e3996f85062d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/85f5c7372625d1e0df0e3996f85062d6-Abstract-Conference.html)

        **Abstract**:

        Modern successes of diffusion models in learning complex, high-dimensional data distributions are attributed, in part, to their capability to construct diffusion processes with analytic transition kernels and score functions. The tractability results in a simulation-free framework with stable regression losses, from which reversed, generative processes can be learned at scale. However, when data is confined to a constrained set as opposed to a standard Euclidean space, these desirable characteristics appear to be lost based on prior attempts. In this work, we propose Mirror Diffusion Models (MDM), a new class of diffusion models that generate data on convex constrained sets without losing any tractability. This is achieved by learning diffusion processes in a dual space constructed from a mirror map, which, crucially, is a standard Euclidean space. We derive efficient computation of mirror maps for popular constrained sets, such as simplices and $\ell_2$-balls, showing significantly improved performance of MDM over existing methods. For safety and privacy purposes, we also explore constrained sets as a new mechanism to embed invisible but quantitative information (i.e., watermarks) in generated data, for which MDM serves as a compelling approach. Our work brings new algorithmic opportunities for learning tractable diffusion on complex domains.

        ----

        ## [1858] Training Transitive and Commutative Multimodal Transformers with LoReTTa

        **Authors**: *Manuel Tran, Yashin Dicente Cid, Amal Lahiani, Fabian J. Theis, Tingying Peng, Eldad Klaiman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/860a092bb4d9d81d3133a01c50c01578-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/860a092bb4d9d81d3133a01c50c01578-Abstract-Conference.html)

        **Abstract**:

        Training multimodal foundation models is challenging due to the limited availability of multimodal datasets. While many public datasets pair images with text, few combine images with audio or text with audio. Even rarer are datasets that align all three modalities at once. Critical domains such as healthcare, infrastructure, or transportation are particularly affected by missing modalities. This makes it difficult to integrate all modalities into a large pre-trained neural network that can be used out-of-the-box or fine-tuned for different downstream tasks. We introduce LoReTTa ($\textbf{L}$inking m$\textbf{O}$dalities with a t$\textbf{R}$ansitive and commutativ$\textbf{E}$ pre-$\textbf{T}$raining s$\textbf{T}$r$\textbf{A}$tegy) to address this understudied problem. Our self-supervised framework unifies causal modeling and masked modeling with the rules of commutativity and transitivity. This allows us to transition within and between modalities. As a result, our pre-trained models are better at exploring the true underlying joint probability distribution. Given a dataset containing only the disjoint combinations $(A, B)$ and $(B, C)$, LoReTTa can model the relation $A \leftrightarrow C$ with $A \leftrightarrow B \leftrightarrow C$. In particular, we show that a transformer pre-trained with LoReTTa can handle any mixture of modalities at inference time, including the never-seen pair $(A, C)$ and the triplet $(A, B, C)$. We extensively evaluate our approach on a synthetic, medical, and reinforcement learning dataset. Across different domains, our universal multimodal transformer consistently outperforms strong baselines such as GPT, BERT, and CLIP on tasks involving the missing modality tuple.

        ----

        ## [1859] SaVeNet: A Scalable Vector Network for Enhanced Molecular Representation Learning

        **Authors**: *Sarp Aykent, Tian Xia*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/860c1c657deafe09f64c013c2888bd7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/860c1c657deafe09f64c013c2888bd7b-Abstract-Conference.html)

        **Abstract**:

        Geometric representation learning of molecules is challenging yet essential for applications in multiple domains. Despite the impressive breakthroughs made by geometric deep learning in various molecular representation learning tasks, effectively capturing complicated geometric features across spatial dimensions is still underexplored due to the significant difficulties in modeling efficient geometric representations and learning the inherent correlation in 3D structural modeling. These include computational inefficiency, underutilization of vectorial embeddings, and limited generalizability to integrate various geometric properties. To address the raised concerns, we introduce an efficient and effective framework, Scalable Vector Network (SaVeNet), designed to accommodate a range of geometric requirements without depending on costly embeddings. In addition, the proposed framework scales effectively with introduced direction noise. Theoretically, we analyze the desired properties (i.e., invariance and equivariant) and framework efficiency of the SaVeNet. Empirically, we conduct a comprehensive series of experiments to evaluate the efficiency and expressiveness of the proposed model. Our efficiency-focused experiments underscore the model's empirical superiority over existing methods. Experimental results on synthetic and real-world datasets demonstrate the expressiveness of our model, which achieves state-of-the-art performance across various tasks within molecular representation learning.

        ----

        ## [1860] Beyond Pretrained Features: Noisy Image Modeling Provides Adversarial Defense

        **Authors**: *Zunzhi You, Daochang Liu, Bohyung Han, Chang Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8629b0fff229b8a27efb1422e990605f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8629b0fff229b8a27efb1422e990605f-Abstract-Conference.html)

        **Abstract**:

        Recent advancements in masked image modeling (MIM) have made it a prevailing framework for self-supervised visual representation learning. The MIM pretrained models, like most deep neural network methods, remain vulnerable to adversarial attacks, limiting their practical application, and this issue has received little research attention. In this paper, we investigate how this powerful self-supervised learning paradigm can provide adversarial robustness to downstream classifiers. During the exploration, we find that noisy image modeling (NIM), a simple variant of MIM that adopts denoising as the pre-text task, reconstructs noisy images surprisingly well despite severe corruption. Motivated by this observation, we propose an adversarial defense method, referred to as De^3, by exploiting the pretrained decoder for denoising. Through De^3, NIM is able to enhance adversarial robustness beyond providing pretrained features. Furthermore, we incorporate a simple modification, sampling the noise scale hyperparameter from random distributions, and enable the defense to achieve a better and tunable trade-off between accuracy and robustness. Experimental results demonstrate that, in terms of adversarial robustness, NIM is superior to MIM thanks to its effective denoising capability. Moreover, the defense provided by NIM achieves performance on par with adversarial training while offering the extra tunability advantage. Source code and models are available at https://github.com/youzunzhi/NIM-AdvDef.

        ----

        ## [1861] UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild

        **Authors**: *Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, Ran Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/862f45ccecb2275851bc8acebb8b4d65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/862f45ccecb2275851bc8acebb8b4d65-Abstract-Conference.html)

        **Abstract**:

        Achieving machine autonomy and human control often represent divergent objectives in the design of interactive AI systems. Visual generative foundation models such as Stable Diffusion show promise in navigating these goals, especially when prompted with arbitrary languages. However, they often fall short in generating images with spatial, structural, or geometric controls. The integration of such controls, which can accommodate various visual conditions in a single unified model, remains an unaddressed challenge. In response, we introduce UniControl, a new generative foundation model that consolidates a wide array of controllable condition-to-image (C2I) tasks within a singular framework, while still allowing for arbitrary language prompts. UniControl enables pixel-level-precise image generation, where visual conditions primarily influence the generated structures and language prompts guide the style and context. To equip UniControl with the capacity to handle diverse visual conditions, we augment pretrained text-to-image diffusion models and introduce a task-aware HyperNet to modulate the diffusion models, enabling the adaptation to different C2I tasks simultaneously. Trained on nine unique C2I tasks, UniControl demonstrates impressive zero-shot generation abilities with unseen visual conditions. Experimental results show that UniControl often surpasses the performance of single-task-controlled methods of comparable model sizes. This control versatility positions UniControl as a significant advancement in the realm of controllable visual generation.

        ----

        ## [1862] Unlocking Deterministic Robustness Certification on ImageNet

        **Authors**: *Kai Hu, Andy Zou, Zifan Wang, Klas Leino, Matt Fredrikson*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/863da9d40547f1d1b18859519ce2dee4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/863da9d40547f1d1b18859519ce2dee4-Abstract-Conference.html)

        **Abstract**:

        Despite the promise of Lipschitz-based methods for provably-robust deep learning with deterministic guarantees, current state-of-the-art results are limited to feed-forward Convolutional Networks (ConvNets) on low-dimensional data, such as CIFAR-10. This paper investigates strategies for expanding certifiably robust training to larger, deeper models.A key challenge in certifying deep networks is efficient calculation of the Lipschitz bound for residual blocks found in ResNet and ViT architectures.We show that fast ways of bounding the Lipschitz constant for conventional ResNets are loose, and show how to address this by designing a new residual block, leading to the *Linear ResNet* (LiResNet) architecture.We then introduce *Efficient Margin MAximization* (EMMA), a loss function that stabilizes robust training by penalizing worst-case adversarial examples from multiple classes simultaneously.Together, these contributions yield new *state-of-the-art* robust accuracy on CIFAR-10/100 and Tiny-ImageNet under $\ell_2$ perturbations.Moreover, for the first time, we are able to scale up fast deterministic robustness guarantees to ImageNet, demonstrating that this approach to robust learning can be applied to real-world applications.

        ----

        ## [1863] Bayesian Optimisation of Functions on Graphs

        **Authors**: *Xingchen Wan, Pierre Osselin, Henry Kenlay, Binxin Ru, Michael A. Osborne, Xiaowen Dong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86419aba4e5eafd2b1009a2e3c540bb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86419aba4e5eafd2b1009a2e3c540bb0-Abstract-Conference.html)

        **Abstract**:

        The increasing availability of graph-structured data motivates the task of optimising over functions defined on the node set of graphs. Traditional graph search algorithms can be applied in this case, but they may be sample-inefficient and do not make use of information about the function values; on the other hand, Bayesian optimisation is a class of promising black-box solvers with superior sample efficiency, but it has scarcely been applied to such novel setups. To fill this gap, we propose a novel Bayesian optimisation framework that optimises over functions defined on generic, large-scale and potentially unknown graphs. Through the learning of suitable kernels on graphs, our framework has the advantage of adapting to the behaviour of the target function. The local modelling approach further guarantees the efficiency of our method. Extensive experiments on both synthetic and real-world graphs demonstrate the effectiveness of the proposed optimisation framework.

        ----

        ## [1864] RIO: A Benchmark for Reasoning Intention-Oriented Objects in Open Environments

        **Authors**: *Mengxue Qu, Yu Wu, Wu Liu, Xiaodan Liang, Jingkuan Song, Yao Zhao, Yunchao Wei*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Intention-oriented object detection aims to detect desired objects based on specific intentions or requirements. For instance, when we desire to "lie down and rest", we instinctively seek out a suitable option such as a "bed" or a "sofa" that can fulfill our needs. Previous work in this area is limited either by the number of intention descriptions or by the affordance vocabulary available for intention objects. These limitations make it challenging to handle intentions in open environments effectively. To facilitate this research, we construct a comprehensive dataset called Reasoning Intention-Oriented Objects (RIO). In particular, RIO is specifically designed to incorporate diverse real-world scenarios and a wide range of object categories. It offers the following key features: 1) intention descriptions in RIO are represented as natural sentences rather than a mere word or verb phrase, making them more practical and meaningful; 2) the intention descriptions are contextually relevant to the scene, enabling a broader range of potential functionalities associated with the objects; 3) the dataset comprises a total of 40,214 images and 130,585 intention-object pairs. With the proposed RIO, we evaluate the ability of some existing models to reason intention-oriented objects in open environments.

        ----

        ## [1865] Supervised Pretraining Can Learn In-Context Reinforcement Learning

        **Authors**: *Jonathan Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, Emma Brunskill*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8644b61a9bc87bf7844750a015feb600-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8644b61a9bc87bf7844750a015feb600-Abstract-Conference.html)

        **Abstract**:

        Large transformer models trained on diverse datasets have shown a remarkable ability to learn in-context, achieving high few-shot performance on tasks they were not explicitly trained to solve. In this paper, we study the in-context learning capabilities of transformers in decision-making problems, i.e., reinforcement learning (RL) for bandits and Markov decision processes. To do so, we introduce and study the Decision-Pretrained Transformer (DPT), a supervised pretraining method where a transformer predicts an optimal action given a query state and an in-context dataset of interactions from a diverse set of tasks. While simple, this procedure produces a model with several surprising capabilities. We find that the trained transformer can solve a range of RL problems in-context, exhibiting both exploration online and conservatism offline, despite not being explicitly trained to do so. The model also generalizes beyond the pretraining distribution to new tasks and automatically adapts its decision-making strategies to unknown structure. Theoretically, we show DPT can be viewed as an efficient implementation of Bayesian posterior sampling, a provably sample-efficient RL algorithm. We further leverage this connection to provide guarantees on the regret of the in-context algorithm yielded by DPT, and prove that it can learn faster than algorithms used to generate the pretraining data. These results suggest a promising yet simple path towards instilling strong in-context decision-making abilities in transformers.

        ----

        ## [1866] L2T-DLN: Learning to Teach with Dynamic Loss Network

        **Authors**: *Zhaoyang Hai, Liyuan Pan, Xiabi Liu, Zhengzheng Liu, Mirna Yunita*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8667f264f88c7938a73a53ab01eb1327-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8667f264f88c7938a73a53ab01eb1327-Abstract-Conference.html)

        **Abstract**:

        With the concept of teaching being introduced to the machine learning community, a teacher model start using dynamic loss functions to teach the training of a student model. The dynamic intends to set adaptive loss functions to different phases of student model learning. In existing works, the teacher model 1) merely determines the loss function based on the present states of the student model, e.g., disregards the experience of the teacher; 2) only utilizes the states of the student model, e.g., training iteration number and loss/accuracy from training/validation sets, while ignoring the states of the loss function. In this paper, we first formulate the loss adjustment as a temporal task by designing a teacher model with memory units, and, therefore, enables the student learning to be guided by the experience of the teacher model. Then, with a Dynamic Loss Network, we can additionally use the states of the loss to assist the teacher learning in enhancing the interactions between the teacher and the student model.   Extensive experiments demonstrate our approach can enhance student learning and improve the performance of various deep models on real-world tasks, including classification, objective detection, and semantic segmentation scenario.

        ----

        ## [1867] Structured Federated Learning through Clustered Additive Modeling

        **Authors**: *Jie Ma, Tianyi Zhou, Guodong Long, Jing Jiang, Chengqi Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8668fdc7b2ddf55a0e235824c66f2eee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8668fdc7b2ddf55a0e235824c66f2eee-Abstract-Conference.html)

        **Abstract**:

        Heterogeneous federated learning without assuming any structure is challenging due to the conflicts among non-identical data distributions of clients. In practice, clients often comprise near-homogeneous clusters so training a server-side model per cluster mitigates the conflicts. However, FL with client clustering often suffers from “clustering collapse'', i.e., one cluster's model excels on increasing clients, and reduces to single-model FL. Moreover, cluster-wise models hinder knowledge sharing between clusters and each model depends on fewer clients. Furthermore, the static clustering assumption on data may not hold for dynamically changing models, which are sensitive to cluster imbalance/initialization or outliers. To address these challenges, we propose ''Clustered Additive Modeling  (CAM)'', which applies a globally shared model $\Theta_g$ on top of the cluster-wise models $\Theta_{1:K}$, i.e., $y=h(x;\Theta_g)+f(x;\Theta_k)$ for clients of cluster-$k$. The global model captures the features shared by all clusters so $\Theta_{1:K}$ are enforced to focus on the difference among clusters. To train CAM, we develop a novel Fed-CAM algorithm that alternates between client clustering and training global/cluster models to predict the residual of each other.  We can easily modify any existing clustered FL methods by CAM and significantly improve their performance without ‘’clustering collapse'' in different non-IID settings. We also provide a convergence analysis of Fed-CAM algorithm.

        ----

        ## [1868] An Inductive Bias for Tabular Deep Learning

        **Authors**: *Ege Beyazit, Jonathan Kozaczuk, Bo Li, Vanessa Wallace, Bilal Fadlallah*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8671b6dffc08b4fcf5b8ce26799b2bef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8671b6dffc08b4fcf5b8ce26799b2bef-Abstract-Conference.html)

        **Abstract**:

        Deep learning methods have achieved state-of-the-art performance in most modeling tasks involving images, text and audio, however, they typically underperform tree-based methods on tabular data. In this paper, we hypothesize that a significant contributor to this performance gap is the interaction between irregular target functions resulting from the heterogeneous nature of tabular feature spaces, and the well-known tendency of neural networks to learn smooth functions. Utilizing tools from spectral analysis, we show that functions described by tabular datasets often have high irregularity, and that they can be smoothed by transformations such as scaling and ranking in order to improve performance. However, because these transformations tend to lose information or negatively impact the loss landscape during optimization, they need to be rigorously fine-tuned for each feature to achieve performance gains. To address these problems, we propose introducing frequency reduction as an inductive bias. We realize this bias as a neural network layer that promotes learning low-frequency representations of the input features, allowing the network to operate in a space where the target function is more regular. Our proposed method introduces less computational complexity than a fully connected layer, while significantly improving neural network performance, and speeding up its convergence on 14 tabular datasets.

        ----

        ## [1869] Fairness-guided Few-shot Prompting for Large Language Models

        **Authors**: *Huan Ma, Changqing Zhang, Yatao Bian, Lemao Liu, Zhirui Zhang, Peilin Zhao, Shu Zhang, Huazhu Fu, Qinghua Hu, Bingzhe Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8678da90126aa58326b2fc0254b33a8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8678da90126aa58326b2fc0254b33a8c-Abstract-Conference.html)

        **Abstract**:

        Large language models have demonstrated surprising ability to perform in-context learning, i.e., these models can be directly applied to solve numerous downstream tasks by conditioning on a prompt constructed by a few input-output examples. However, prior research has shown that in-context learning can suffer from high instability due to variations in training examples, example order, and prompt formats. Therefore, the construction of an appropriate prompt is essential for improving the performance of in-context learning. In this paper,  we revisit this problem from the view of predictive bias. Specifically, we introduce a metric to evaluate the predictive bias of a fixed prompt against labels or a given attributes. Then we empirically show that prompts with higher bias always lead to unsatisfactory predictive quality. Based on this observation, we propose a novel search strategy based on the greedy search to identify the near-optimal prompt for improving the performance of in-context learning. We perform comprehensive experiments with state-of-the-art mainstream models such as GPT-3 on various downstream tasks. Our results indicate that our method can enhance the model's in-context learning performance in an effective and interpretable manner.

        ----

        ## [1870] Efficient Robust Bayesian Optimization for Arbitrary Uncertain inputs

        **Authors**: *Lin Yang, Junlong Lyu, Wenlong Lyu, Zhitang Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/868f2f9a9950f7b0538b3ce7eb4c8eb8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/868f2f9a9950f7b0538b3ce7eb4c8eb8-Abstract-Conference.html)

        **Abstract**:

        Bayesian Optimization (BO) is a sample-efficient optimization algorithm widely employed across various applications. In some challenging BO tasks, input uncertainty arises due to the inevitable randomness in the optimization process, such as machining errors, execution noise, or contextual variability. This uncertainty deviates the input from the intended value before evaluation, resulting in significant performance fluctuations in the final result. In this paper, we introduce a novel robust Bayesian Optimization algorithm, AIRBO, which can effectively identify a robust optimum that performs consistently well under arbitrary input uncertainty. Our method directly models the uncertain inputs of arbitrary distributions by empowering the Gaussian Process with the Maximum Mean Discrepancy (MMD) and further accelerates the posterior inference via Nystrom approximation. Rigorous theoretical regret bound is established under MMD estimation error and extensive experiments on synthetic functions and real problems demonstrate that our approach can handle various input uncertainties and achieve a state-of-the-art performance.

        ----

        ## [1871] HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution

        **Authors**: *Eric Nguyen, Michael Poli, Marjan Faizi, Armin W. Thomas, Michael Wornow, Callum Birch-Sykes, Stefano Massaroli, Aman Patel, Clayton M. Rabideau, Yoshua Bengio, Stefano Ermon, Christopher Ré, Stephen Baccus*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html)

        **Abstract**:

        Genomic (DNA) sequences encode an enormous amount of information for gene regulation and protein synthesis. Similar to natural language models, researchers have proposed foundation models in genomics to learn generalizable features from unlabeled genome data that can then be fine-tuned for downstream tasks such as identifying regulatory elements. Due to the quadratic scaling of attention, previous Transformer-based genomic models have used 512 to 4k tokens as context (<0.001% of the human genome), significantly limiting the modeling of long-range interactions in DNA. In addition, these methods rely on tokenizers or fixed k-mers to aggregate meaningful DNA units, losing single nucleotide resolution (i.e. DNA "characters") where subtle genetic variations can completely alter protein function via single nucleotide polymorphisms (SNPs). Recently, Hyena, a large language model based on implicit convolutions was shown to match attention in quality while allowing longer context lengths and lower time complexity. Leveraging Hyena’s new long-range capabilities, we present HyenaDNA, a genomic foundation model pretrained on the human reference genome with context lengths of up to 1 million tokens at the single nucleotide-level – an up to 500x increase over previous dense attention-based models. HyenaDNA scales sub-quadratically in sequence length (training up to 160x faster than Transformer), uses single nucleotide tokens, and has full global context at each layer. We explore what longer context enables - including the first use of in-context learning in genomics for simple adaptation to novel tasks without updating pretrained model weights. On fine-tuned benchmarks from the Nucleotide Transformer, HyenaDNA reaches state-of-the-art (SotA) on 12 of 18 datasets using a model with orders of magnitude less parameters and pretraining data.1 On the GenomicBenchmarks, HyenaDNA surpasses SotA on 7 of 8 datasets on average by +10 accuracy points. Code at https://github.com/HazyResearch/hyena-dna.

        ----

        ## [1872] Learning a 1-layer conditional generative model in total variation

        **Authors**: *Ajil Jalal, Justin Singh Kang, Ananya Uppal, Kannan Ramchandran, Eric Price*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86b8ad667206fb9a52ae575fbf1cd6be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86b8ad667206fb9a52ae575fbf1cd6be-Abstract-Conference.html)

        **Abstract**:

        A conditional generative model is a method for sampling from a conditional distribution $p(y \mid x)$.  For example, one may want to sample an image of a cat given the label ``cat''.  A feed-forward conditional generative model is a function $g(x, z)$ that takes the input $x$ and a random seed $z$, and outputs a sample $y$ from $p(y \mid x)$.  Ideally the distribution of outputs $(x, g(x, z))$ would be close in total variation to the ideal distribution $(x, y)$.Generalization bounds for other learning models require assumptions on the distribution of $x$, even in simple settings like linear regression with Gaussian noise. We show these assumptions are unnecessary in our model, for both linear regression and single-layer ReLU networks.  Given samples $(x, y)$, we show how to learn a 1-layer ReLU conditional generative model in total variation. As our result has no assumption on the distribution of inputs $x$, if we are given access to the internal activations of a deep generative model, we can compose our 1-layer guarantee to progressively learn the deep model using a near-linear number of samples.

        ----

        ## [1873] Model Shapley: Equitable Model Valuation with Black-box Access

        **Authors**: *Xinyi Xu, Thanh Lam, Chuan Sheng Foo, Bryan Kian Hsiang Low*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86bcae6da75c72e32f30a5553f094c06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86bcae6da75c72e32f30a5553f094c06-Abstract-Conference.html)

        **Abstract**:

        Valuation methods of data and machine learning (ML) models are essential to the establishment of AI marketplaces. Importantly, certain practical considerations (e.g., operational constraints, legal restrictions) favor the use of model valuation over data valuation. Also, existing marketplaces that involve trading of pre-trained ML models call for an equitable model valuation method to price them. In particular, we investigate the black-box access setting which allows querying a model (to observe predictions) without disclosing model-specific information (e.g., architecture and parameters). By exploiting a Dirichlet abstraction of a model’s predictions, we propose a novel and equitable model valuation method called model Shapley. We also leverage a Lipschitz continuity of model Shapley to design a learning approach for predicting the model Shapley values (MSVs) of many vendors’ models (e.g., 150) in a large-scale marketplace. We perform extensive empirical validation on the effectiveness of model Shapley using various real-world datasets and heterogeneous model types.

        ----

        ## [1874] Robust Concept Erasure via Kernelized Rate-Distortion Maximization

        **Authors**: *Somnath Basu Roy Chowdhury, Nicholas Monath, Kumar Avinava Dubey, Amr Ahmed, Snigdha Chaturvedi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86bd650f85480c595ecab29081a3774e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86bd650f85480c595ecab29081a3774e-Abstract-Conference.html)

        **Abstract**:

        Distributed representations provide a vector space that captures meaningful relationships between data instances. The distributed nature of these representations, however, entangles together multiple attributes or concepts of data instances (e.g., the topic or sentiment of a text, characteristics of the author (age, gender, etc), etc).  Recent work has proposed the task of concept erasure, in which rather than making a concept predictable, the goal is to remove an attribute from distributed representations while retaining other information from the original representation space as much as possible.  In this paper, we propose a new distance metric learning-based objective, the Kernelized Rate-Distortion Maximizer (KRaM), for performing concept erasure. KRaM fits a transformation of representations to match a specified distance measure (defined by a labeled concept to erase) using a modified rate-distortion function. Specifically, KRaM's objective function aims to make instances with similar concept labels dissimilar in the learned representation space while retaining other information.  We find that optimizing KRaM effectively erases various types of concepts—categorical, continuous, and vector-valued variables—from data representations across diverse domains. We also provide a theoretical analysis of several properties of KRaM's objective. To assess the quality of the learned representations, we propose an alignment score to evaluate their similarity with the original representation space. Additionally, we conduct experiments to showcase KRaM's efficacy in various settings, from erasing binary gender variables in word embeddings to vector-valued variables in GPT-3 representations.

        ----

        ## [1875] BiMatting: Efficient Video Matting via Binarization

        **Authors**: *Haotong Qin, Lei Ke, Xudong Ma, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, Xianglong Liu, Fisher Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86c070ce724102ee876d1935590e111a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86c070ce724102ee876d1935590e111a-Abstract-Conference.html)

        **Abstract**:

        Real-time video matting on edge devices faces significant computational resource constraints, limiting the widespread use of video matting in applications such as online conferences and short-form video production. Binarization is a powerful compression approach that greatly reduces computation and memory consumption by using 1-bit parameters and bitwise operations. However, binarization of the video matting model is not a straightforward process, and our empirical analysis has revealed two primary bottlenecks: severe representation degradation of the encoder and massive redundant computations of the decoder. To address these issues, we propose BiMatting, an accurate and efficient video matting model using binarization. Specifically, we construct shrinkable and dense topologies of the binarized encoder block to enhance the extracted representation. We sparsify the binarized units to reduce the low-information decoding computation. Through extensive experiments, we demonstrate that BiMatting outperforms other binarized video matting models, including state-of-the-art (SOTA) binarization methods, by a significant margin. Our approach even performs comparably to the full-precision counterpart in visual quality. Furthermore, BiMatting achieves remarkable savings of 12.4$\times$ and 21.6$\times$ in computation and storage, respectively, showcasing its potential and advantages in real-world resource-constrained scenarios. Our code and models are released at https://github.com/htqin/BiMatting .

        ----

        ## [1876] One Fits All: Power General Time Series Analysis by Pretrained LM

        **Authors**: *Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, Rong Jin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Abstract-Conference.html)

        **Abstract**:

        Although we have witnessed great success of pre-trained models in natural language processing (NLP) and computer vision (CV), limited progress has been made for general time series analysis. Unlike NLP and CV where a unified model can be used to perform different tasks, specially designed approach still dominates in each time series analysis task such as classification, anomaly detection, forecasting, and few-shot learning. The main challenge that blocks the development of pre-trained model for time series analysis is the lack of a large amount of data for training. In this work, we address this challenge by leveraging language or CV models, pre-trained from billions of tokens, for time series analysis. Specifically, we refrain from altering the self-attention and feedforward layers of the residual blocks in the pre-trained language or image model. This model, known as the Frozen Pretrained Transformer (FPT), is evaluated through fine-tuning on all major types of tasks involving time series. Our results demonstrate that pre-trained models on natural language or images can lead to a comparable or state-of-the-art performance in all main time series analysis tasks, as illustrated in Figure1. We also found both theoretically and empirically that the self-attention module behaviors similarly to principle component analysis (PCA), an observation that helps explains how transformer bridges the domain gap and a crucial step towards understanding the universality of a pre-trained transformer. The code is publicly available at https://anonymous.4open.science/r/Pretrained-LM-for-TSForcasting-C561.

        ----

        ## [1877] Parameterizing Non-Parametric Meta-Reinforcement Learning Tasks via Subtask Decomposition

        **Authors**: *Suyoung Lee, Myungsik Cho, Youngchul Sung*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86c1fd74fa25bd6be0072937803e0bd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86c1fd74fa25bd6be0072937803e0bd1-Abstract-Conference.html)

        **Abstract**:

        Meta-reinforcement learning (meta-RL) techniques have demonstrated remarkable success in generalizing deep reinforcement learning across a range of tasks. Nevertheless, these methods often struggle to generalize beyond tasks with parametric variations. To overcome this challenge, we propose Subtask Decomposition and Virtual Training (SDVT), a novel meta-RL approach that decomposes each non-parametric task into a collection of elementary subtasks and parameterizes the task based on its decomposition. We employ a  Gaussian mixture VAE to meta-learn the decomposition process, enabling the agent to reuse policies acquired from common subtasks. Additionally, we propose a virtual training procedure, specifically designed for non-parametric task variability, which generates hypothetical subtask compositions, thereby enhancing generalization to previously unseen subtask compositions. Our method significantly improves performance on the Meta-World ML-10 and ML-45 benchmarks, surpassing current state-of-the-art techniques.

        ----

        ## [1878] Near-Optimal Algorithms for Gaussians with Huber Contamination: Mean Estimation and Linear Regression

        **Authors**: *Ilias Diakonikolas, Daniel Kane, Ankit Pensia, Thanasis Pittas*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86c283920335ed1fec3edee227e05fbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86c283920335ed1fec3edee227e05fbf-Abstract-Conference.html)

        **Abstract**:

        We study the fundamental problems of Gaussian mean estimation and linear regression with Gaussian covariates in the presence of Huber contamination. Our main contribution is the design of the first sample near-optimal and almost linear-time algorithms with optimal error guarantees for both these problems. Specifically, for Gaussian robust mean estimation on $\mathbb R^d$ with contamination parameter $\epsilon \in (0, \epsilon_0)$ for a small absolute constant $\epsilon_0$, we give an algorithm with sample complexity $n = \tilde{O}(d/\epsilon^2)$ and almost linear runtime that approximates the target mean within $\ell_2$-error $O(\epsilon)$. This improves on prior work that achieved this error guarantee with polynomially suboptimal sample and time complexity. For robust linear regression, we give the first algorithm with sample complexity $n = \tilde{O}(d/\epsilon^2)$ and almost linear runtime that approximates the target regressor within $\ell_2$-error $O(\epsilon)$. This is the first polynomial sample and time algorithm achieving the optimal error guarantee, answering an open question in the literature. At the technical level, we develop a methodology that yields almost-linear time algorithms for multi-directional filtering that may be of broader interest.

        ----

        ## [1879] Causal Discovery from Subsampled Time Series with Proxy Variables

        **Authors**: *Mingzhou Liu, Xinwei Sun, Lingjing Hu, Yizhou Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/86cba2b31d17f237866a2e6c52c7878a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/86cba2b31d17f237866a2e6c52c7878a-Abstract-Conference.html)

        **Abstract**:

        Inferring causal structures from time series data is the central interest of many scientific inquiries. A major barrier to such inference is the problem of subsampling, i.e., the frequency of measurement is much lower than that of causal influence. To overcome this problem, numerous methods have been proposed, yet either was limited to the linear case or failed to achieve identifiability. In this paper, we propose a constraint-based algorithm that can identify the entire causal structure from subsampled time series, without any parametric constraint. Our observation is that the challenge of subsampling arises mainly from hidden variables at the unobserved time steps. Meanwhile, every hidden variable has an observed proxy, which is essentially itself at some observable time in the future, benefiting from the temporal structure. Based on these, we can leverage the proxies to remove the bias induced by the hidden variables and hence achieve identifiability. Following this intuition, we propose a proxy-based causal discovery algorithm. Our algorithm is nonparametric and can achieve full causal identification. Theoretical advantages are reflected in synthetic and real-world experiments.

        ----

        ## [1880] "Why Not Looking backward?" A Robust Two-Step Method to Automatically Terminate Bayesian Optimization

        **Authors**: *Shuang Li, Ke Li, Wei Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/870c1e0589822bf37590b84984c345c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/870c1e0589822bf37590b84984c345c4-Abstract-Conference.html)

        **Abstract**:

        Bayesian Optimization (BO) is a powerful method for tackling expensive black-box optimization problems. As a sequential model-based optimization strategy, BO iteratively explores promising solutions until a predetermined budget, either iterations or time, is exhausted. The decision on when to terminate BO significantly influences both the quality of solutions and its computational efficiency. In this paper, we propose a simple, yet theoretically grounded, two-step method for automatically terminating BO. Our core concept is to proactively identify if the search is within a convex region by examining previously observed samples. BO is halted once the local regret within this convex region falls below a predetermined threshold. To enhance numerical stability, we propose an approximation method for calculating the termination indicator by solving a bilevel optimization problem. We conduct extensive empirical studies on diverse benchmark problems, including synthetic functions, reinforcement learning, and hyperparameter optimization. Experimental results demonstrate that our proposed method saves up to $\approx 80\%$ computational budget yet is with an order of magnitude smaller performance degradation, comparing against the other peer methods. In addition, our proposed termination method is robust in terms of the setting of its termination criterion.

        ----

        ## [1881] Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models

        **Authors**: *Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Jianfeng Gao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) have achieved remarkable progress in solving various natural language processing tasks due to emergent reasoning abilities. However, LLMs have inherent limitations as they are incapable of accessing up-to-date information (stored on the Web or in task-specific knowledge bases), using external tools, and performing precise mathematical and logical reasoning. In this paper, we present Chameleon, an AI system that mitigates these limitations by augmenting LLMs with plug-and-play modules for compositional reasoning. Chameleon synthesizes programs by composing various tools (e.g., LLMs, off-the-shelf vision models, web search engines, Python functions, and heuristic-based modules) for accomplishing complex reasoning tasks. At the heart of Chameleon is an LLM-based planner that assembles a sequence of tools to execute to generate the final response. We showcase the effectiveness of Chameleon on two multi-modal knowledge-intensive reasoning tasks: ScienceQA and TabMWP. Chameleon, powered by GPT-4, achieves an 86.54% overall accuracy on ScienceQA, improving the best published few-shot result by 11.37%. On TabMWP, GPT-4-powered Chameleon improves the accuracy by 17.0%, lifting the state of the art to 98.78%. Our analysis also shows that the GPT-4-powered planner exhibits more consistent and rational tool selection via inferring potential constraints from instructions, compared to a ChatGPT-powered planner.

        ----

        ## [1882] Diffusion Models and Semi-Supervised Learners Benefit Mutually with Few Labels

        **Authors**: *Zebin You, Yong Zhong, Fan Bao, Jiacheng Sun, Chongxuan Li, Jun Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8735753cc18f6baa92d1f069fd8b14a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8735753cc18f6baa92d1f069fd8b14a0-Abstract-Conference.html)

        **Abstract**:

        In an effort to further advance semi-supervised generative and classification tasks, we propose a simple yet effective training strategy called *dual pseudo training* (DPT), built upon strong semi-supervised learners and diffusion models. DPT operates in three stages: training a classifier on partially labeled data to predict pseudo-labels; training a conditional generative model using these pseudo-labels to generate pseudo images; and retraining the classifier with a mix of real and pseudo images.  Empirically, DPT consistently achieves SOTA performance of semi-supervised generation and classification across various settings. In particular, with one or two labels per class, DPT achieves a Fr√©chet Inception Distance (FID) score of 3.08 or 2.52 on ImageNet $256\times256$. Besides, DPT outperforms competitive semi-supervised baselines substantially on  ImageNet classification tasks, *achieving top-1 accuracies of 59.0 (+2.8), 69.5 (+3.0), and 74.4 (+2.0)* with one, two, or five labels per class, respectively. Notably, our results demonstrate that diffusion can generate realistic images with only a few labels (e.g., $<0.1$%) and generative augmentation remains viable for semi-supervised classification. Our code is available at *https://github.com/ML-GSAI/DPT*.

        ----

        ## [1883] Pre-Training Protein Encoder via Siamese Sequence-Structure Diffusion Trajectory Prediction

        **Authors**: *Zuobai Zhang, Minghao Xu, Aurélie C. Lozano, Vijil Chenthamarakshan, Payel Das, Jian Tang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/873c86d9a979ab80d8e2919510d4446b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/873c86d9a979ab80d8e2919510d4446b-Abstract-Conference.html)

        **Abstract**:

        Self-supervised pre-training methods on proteins have recently gained attention, with most approaches focusing on either protein sequences or structures, neglecting the exploration of their joint distribution, which is crucial for a comprehensive understanding of protein functions by integrating co-evolutionary information and structural characteristics. In this work, inspired by the success of denoising diffusion models in generative tasks, we propose the DiffPreT approach to pre-train a protein encoder by sequence-structure joint diffusion modeling. DiffPreT guides the encoder to recover the native protein sequences and structures from the perturbed ones along the joint diffusion trajectory, which acquires the joint distribution of sequences and structures. Considering the essential protein conformational variations, we enhance DiffPreT by a method called Siamese Diffusion Trajectory Prediction (SiamDiff) to capture the correlation between different conformers of a protein. SiamDiff attains this goal by maximizing the mutual information between representations of diffusion trajectories of structurally-correlated conformers. We study the effectiveness of DiffPreT and SiamDiff on both atom- and residue-level structure-based protein understanding tasks. Experimental results show that the performance of DiffPreT is consistently competitive on all tasks, and SiamDiff achieves new state-of-the-art performance, considering the mean ranks on all tasks. Code will be released upon acceptance.

        ----

        ## [1884] Progressive Ensemble Distillation: Building Ensembles for Efficient Inference

        **Authors**: *Don Kurian Dennis, Abhishek Shetty, Anish Prasad Sevekari, Kazuhito Koishida, Virginia Smith*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87425754bcc35f2bc62ef4a421a772d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87425754bcc35f2bc62ef4a421a772d6-Abstract-Conference.html)

        **Abstract**:

        Knowledge distillation is commonly used to compress an ensemble of models into a single model. In this work we study the problem of progressive ensemble distillation: Given a large, pretrained teacher model , we seek to decompose the model into an ensemble of smaller, low-inference cost student models . The resulting ensemble allows for flexibly tuning accuracy vs. inference cost, which can be useful for a multitude of applications in efficient inference. Our method, B-DISTIL, uses a boosting procedure that allows function composition based aggregation rules to construct expressive ensembles with similar performance as using much smaller student models. We demonstrate the effectiveness of B-DISTIL by decomposing pretrained models across a variety of image, speech, and sensor datasets. Our method comes with strong theoretical guarantees in terms of convergence as well as generalization.

        ----

        ## [1885] Differentially Private Approximate Near Neighbor Counting in High Dimensions

        **Authors**: *Alexandr Andoni, Piotr Indyk, Sepideh Mahabadi, Shyam Narayanan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87571720167f7e88827c40e468e3101f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87571720167f7e88827c40e468e3101f-Abstract-Conference.html)

        **Abstract**:

        Range counting (e.g., counting the number of data points falling into a given query ball) under differential privacy has been studied extensively. However, the current algorithms for this problem are subject to the following dichotomy. One class of algorithms suffers from an additive error that is a fixed polynomial in the number of points. Another class of algorithms allows for polylogarithmic additive error, but the error grows exponentially in the dimension. To achieve the latter, the problem is relaxed to allow a “fuzzy” definition of the range boundary, e.g., a count of the points in a ball of radius $r$ might also include points in a ball of radius $cr$ for some $c>1$. In this paper we present an efficient algorithm that offers a sweet spot between these two classes. The algorithm has an additive error that is an arbitrary small power of the data set size, depending on how fuzzy the range boundary is, as well as a small ($1+o(1)$) multiplicative error. Crucially, the amount of noise added has no dependence on the dimension. Our algorithm introduces a variant of Locality-Sensitive Hashing, utilizing it in a novel manner.

        ----

        ## [1886] Reinforcement-Enhanced Autoregressive Feature Transformation: Gradient-steered Search in Continuous Space for Postfix Expressions

        **Authors**: *Dongjie Wang, Meng Xiao, Min Wu, Pengfei Wang, Yuanchun Zhou, Yanjie Fu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8797d13e5998acfab387d4bf0a5b9b00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8797d13e5998acfab387d4bf0a5b9b00-Abstract-Conference.html)

        **Abstract**:

        Feature transformation aims to generate new pattern-discriminative feature space from original features to improve downstream machine learning  (ML) task performances. However, the discrete search space for the optimal feature explosively grows on the basis of combinations of features and operations from low-order forms to high-order forms. Existing methods, such as exhaustive search, expansion reduction, evolutionary algorithms, reinforcement learning, and iterative greedy, suffer from large search space. Overly emphasizing efficiency in algorithm design usually sacrifice stability or robustness. To fundamentally fill this gap, we reformulate discrete feature transformation as a continuous space optimization task and develop an embedding-optimization-reconstruction framework. This framework includes four steps: 1) reinforcement-enhanced data preparation, aiming to prepare high-quality transformation-accuracy training data; 2) feature transformation operation sequence embedding, intending to encapsulate the knowledge of prepared training data within a continuous space; 3) gradient-steered optimal embedding search, dedicating to uncover potentially superior embeddings within the learned space; 4) transformation operation sequence reconstruction, striving to reproduce the feature transformation solution to pinpoint the optimal feature space. Finally, extensive experiments and case studies are performed to demonstrate the effectiveness and robustness of the proposed method. The code and data are publicly accessible https://www.dropbox.com/sh/imh8ckui7va3k5u/AACulQegVx0MuywYyoCqSdVPa?dl=0.

        ----

        ## [1887] FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning

        **Authors**: *Kun Song, Huimin Ma, Bochao Zou, Huishuai Zhang, Weiran Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87cf37e2085655bad7bad0a014e0edad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87cf37e2085655bad7bad0a014e0edad-Abstract-Conference.html)

        **Abstract**:

        Due to the limited availability of data, existing few-shot learning methods trained from scratch fail to achieve satisfactory performance. In contrast, large-scale pre-trained models such as CLIP demonstrate remarkable few-shot and zero-shot capabilities. To enhance the performance of pre-trained models for downstream tasks, fine-tuning the model on downstream data is frequently necessary. However, fine-tuning the pre-trained model leads to a decrease in its generalizability in the presence of distribution shift, while the limited number of samples in few-shot learning makes the model highly susceptible to overfitting. Consequently, existing methods for fine-tuning few-shot learning primarily focus on fine-tuning the model's classification head or introducing additional structure. In this paper, we introduce a fine-tuning approach termed Feature Discrimination Alignment (FD-Align). Our method aims to bolster the model's generalizability by preserving the consistency of spurious features across the fine-tuning process. Extensive experimental results validate the efficacy of our approach for both ID and OOD tasks. Once fine-tuned, the model can seamlessly integrate with existing methods, leading to performance improvements. Our code can be found in https://github.com/skingorz/FD-Align.

        ----

        ## [1888] A Step Towards Worldwide Biodiversity Assessment: The BIOSCAN-1M Insect Dataset

        **Authors**: *Zahra Gharaee, ZeMing Gong, Nicholas Pellegrino, Iuliia Zarubiieva, Joakim Bruslund Haurum, Scott C. Lowe, Jaclyn T. A. McKeown, Chris C. Y. Ho, Joschka McLeod, Yi-Yun C. Wei, Jireh Agda, Sujeevan Ratnasingham, Dirk Steinke, Angel X. Chang, Graham W. Taylor, Paul W. Fieguth*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87dbbdc3a685a97ad28489a1d57c45c1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/87dbbdc3a685a97ad28489a1d57c45c1-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In an effort to catalog insect biodiversity, we propose a new large dataset of hand-labelled insect images, the BIOSCAN-1M Insect Dataset. Each record is taxonomically classified by an expert, and also has associated genetic information including raw nucleotide barcode sequences and assigned barcode index numbers, which are genetic-based proxies for species classification. This paper presents a curated million-image dataset, primarily to train computer-vision models capable of providing image-based taxonomic assessment, however, the dataset also presents compelling characteristics, the study of which would be of interest to the broader machine learning community. Driven by the biological nature inherent to the dataset, a characteristic long-tailed class-imbalance distribution is exhibited. Furthermore, taxonomic labelling is a hierarchical classification scheme, presenting a highly fine-grained classification problem at lower levels. Beyond spurring interest in biodiversity research within the machine learning community, progress on creating an image-based taxonomic classifier will also further the ultimate goal of all BIOSCAN research: to lay the foundation for a comprehensive survey of global biodiversity. This paper introduces the dataset and explores the classification task through the implementation and analysis of a baseline classifier. The code repository of the BIOSCAN-1M-Insect dataset is available at https://github.com/zahrag/BIOSCAN-1M

        ----

        ## [1889] D-Separation for Causal Self-Explanation

        **Authors**: *Wei Liu, Jun Wang, Haozhao Wang, Ruixuan Li, Zhiying Deng, Yuankai Zhang, Yang Qiu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87e82678c0d6e5b729398426f82e9af6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87e82678c0d6e5b729398426f82e9af6-Abstract-Conference.html)

        **Abstract**:

        Rationalization aims to strengthen the interpretability of NLP models by extracting a subset of human-intelligible pieces of their inputting texts. Conventional works generally employ the maximum mutual information (MMI) criterion to find the rationale that is most indicative of the target label. However, this criterion can be influenced by spurious features that correlate with the causal rationale or the target label. Instead of attempting to rectify the issues of the MMI criterion, we propose a novel criterion to uncover the causal rationale, termed the Minimum Conditional Dependence (MCD) criterion, which is grounded on our finding that the non-causal features and the target label are \emph{d-separated} by the causal rationale. By minimizing the dependence between the non-selected parts of the input and the target label conditioned on the selected rationale candidate, all the causes of the label are compelled to be selected. In this study, we employ a simple and practical measure for dependence, specifically the KL-divergence, to validate our proposed MCD criterion.  Empirically, we demonstrate that MCD improves the F1 score by up to 13.7% compared to previous state-of-the-art MMI-based methods.Our code is in an anonymous repository: https://anonymous.4open.science/r/MCD-CE88.

        ----

        ## [1890] History Filtering in Imperfect Information Games: Algorithms and Complexity

        **Authors**: *Christopher Solinas, Douglas Rebstock, Nathan R. Sturtevant, Michael Buro*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/87ee1bbac4635e7c948f3eea83c1f262-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/87ee1bbac4635e7c948f3eea83c1f262-Abstract-Conference.html)

        **Abstract**:

        Historically applied exclusively to perfect information games, depth-limited search with value functions has been key to recent advances in AI for imperfect information games. Most prominent approaches with strong theoretical guarantees require subgame decomposition - a process in which a subgame is computed from public information and player beliefs. However, subgame decomposition can itself require non-trivial computations, and its tractability depends on the existence of efficient algorithms for either full enumeration or generation of the histories that form the root of the subgame. Despite this, no formal analysis of the tractability of such computations has been established in prior work, and application domains have often consisted of games, such as poker, for which enumeration is trivial on modern hardware.Applying these ideas to more complex domains requires understanding their cost. In this work, we introduce and analyze the computational aspects and tractability of filtering histories for subgame decomposition. We show that constructing a single history from the root of the subgame is generally intractable, and then provide a necessary and sufficient condition for efficient enumeration. We also introduce a novel Markov Chain Monte Carlo-based generation algorithm for trick-taking card games - a domain where enumeration is often prohibitively expensive. Our experiments demonstrate its improved scalability in the trick-taking card game Oh Hell.These contributions clarify when and how depth-limited search via subgame decomposition can be an effective tool for sequential decision-making in imperfect information settings.

        ----

        ## [1891] Constant Approximation for Individual Preference Stable Clustering

        **Authors**: *Anders Aamand, Justin Y. Chen, Allen Liu, Sandeep Silwal, Pattara Sukprasert, Ali Vakilian, Fred Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/881259965dacb9f42967aae84a157283-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/881259965dacb9f42967aae84a157283-Abstract-Conference.html)

        **Abstract**:

        Individual preference (IP) stability, introduced by Ahmadi et al. (ICML 2022), is a natural clustering objective inspired by stability and fairness constraints. A clustering is $\alpha$-IP stable if the average distance of every data point to its own cluster is at most $\alpha$ times the average distance to any other cluster. Unfortunately, determining if a dataset admits a $1$-IP stable clustering is NP-Hard. Moreover, before this work, it was unknown if an $o(n)$-IP stable clustering always exists, as the prior state of the art only guaranteed an $O(n)$-IP stable clustering. We close this gap in understanding and show that an $O(1)$-IP stable clustering always exists for general metrics, and we give an efficient algorithm which outputs such a clustering.  We also introduce generalizations of IP stability beyond average distance and give efficient near optimal algorithms in the cases where we consider the maximum and minimum distances within and between clusters.

        ----

        ## [1892] Intervention Generalization: A View from Factor Graph Models

        **Authors**: *Gecia Bravo Hermsdorff, David S. Watson, Jialin Yu, Jakob Zeitler, Ricardo Silva*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88139fdcc82fc597090620d77b023282-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88139fdcc82fc597090620d77b023282-Abstract-Conference.html)

        **Abstract**:

        One of the goals of causal inference is to generalize from past experiments and observational data to novel conditions. While it is in principle possible to eventually learn a mapping from a novel experimental condition to an outcome of interest, provided a sufficient variety of experiments is available in the training data, coping with a large combinatorial space of possible interventions is hard. Under a typical sparse experimental design, this mapping is ill-posed without relying on heavy regularization or prior distributions. Such assumptions may or may not be reliable, and can be hard to defend or test. In this paper, we take a close look at how to warrant a leap from past experiments to novel conditions based on minimal assumptions about the factorization of the distribution of the manipulated system, communicated in the well-understood language of factor graph models. A postulated interventional factor model (IFM) may not always be informative, but it conveniently abstracts away a need for explicitly modeling unmeasured confounding and feedback mechanisms, leading to directly testable claims. Given an IFM and datasets from a collection of experimental regimes, we derive conditions for identifiability of the expected outcomes of new regimes never observed in these training data. We implement our framework using several efficient algorithms, and apply them on a range of semi-synthetic experiments.

        ----

        ## [1893] Decision Tree for Locally Private Estimation with Public Data

        **Authors**: *Yuheng Ma, Han Zhang, Yuchao Cai, Hanfang Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88237ac4e9941b1be5c6d3c1ad408184-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88237ac4e9941b1be5c6d3c1ad408184-Abstract-Conference.html)

        **Abstract**:

        We propose conducting locally differentially private (LDP) estimation with the aid of a small amount of public data to enhance the performance of private estimation. Specifically, we introduce an efficient algorithm called Locally differentially Private Decision Tree (LPDT) for LDP regression. We first use the public data to grow a decision tree partition and then fit an estimator according to the partition privately.  From a theoretical perspective, we show that LPDT is $\varepsilon$-LDP and has a mini-max optimal convergence rate under a mild assumption of similarity between public and private data, whereas the lower bound of the convergence rate of LPDT without public data is strictly slower, which implies that the public data helps to improve the convergence rates of LDP estimation. We conduct experiments on both synthetic and real-world data to demonstrate the superior performance of LPDT compared with other state-of-the-art LDP regression methods. Moreover, we show that LPDT remains effective despite considerable disparities between public and private data.

        ----

        ## [1894] DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization

        **Authors**: *Haoran Ye, Jiarui Wang, Zhiguang Cao, Helan Liang, Yong Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html)

        **Abstract**:

        Ant Colony Optimization (ACO) is a meta-heuristic algorithm that has been successfully applied to various Combinatorial Optimization Problems (COPs). Traditionally, customizing ACO for a specific problem requires the expert design of knowledge-driven heuristics. In this paper, we propose DeepACO, a generic framework that leverages deep reinforcement learning to automate heuristic designs. DeepACO serves to strengthen the heuristic measures of existing ACO algorithms and dispense with laborious manual design in future ACO applications. As a neural-enhanced meta-heuristic, DeepACO consistently outperforms its ACO counterparts on eight COPs using a single neural model and a single set of hyperparameters. As a Neural Combinatorial Optimization method, DeepACO performs better than or on par with problem-specific methods on canonical routing problems. Our code is publicly available at https://github.com/henry-yeh/DeepACO.

        ----

        ## [1895] Offline Imitation Learning with Variational Counterfactual Reasoning

        **Authors**: *Zexu Sun, Bowei He, Jinxin Liu, Xu Chen, Chen Ma, Shuai Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8833c8aa10542d24d693bbaf6a4598f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8833c8aa10542d24d693bbaf6a4598f5-Abstract-Conference.html)

        **Abstract**:

        In offline imitation learning (IL), an agent aims to learn an optimal expert behavior policy without additional online environment interactions. However, in many real-world scenarios, such as robotics manipulation, the offline dataset is collected from suboptimal behaviors without rewards. Due to the scarce expert data, the agents usually suffer from simply memorizing poor trajectories and are vulnerable to the variations in the environments, lacking the capability of generalizing to new environments.To automatically generate high-quality expert data and improve the generalization ability of the agent, we propose a framework named \underline{O}ffline \underline{I}mitation \underline{L}earning with \underline{C}ounterfactual data \underline{A}ugmentation (OILCA) by doing counterfactual inference. In particular, we leverage identifiable variational autoencoder to generate \textit{counterfactual} samples for expert data augmentation. We theoretically analyze the influence of the generated expert data and the improvement of generalization. Moreover, we conduct extensive experiments to demonstrate that our approach significantly outperforms various baselines on both \textsc{DeepMind Control Suite} benchmark for in-distribution performance and \textsc{CausalWorld} benchmark for out-of-distribution generalization.

        ----

        ## [1896] LART: Neural Correspondence Learning with Latent Regularization Transformer for 3D Motion Transfer

        **Authors**: *Haoyu Chen, Hao Tang, Radu Timofte, Luc Van Gool, Guoying Zhao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88593e16b09104fb6010d370c081d7bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88593e16b09104fb6010d370c081d7bc-Abstract-Conference.html)

        **Abstract**:

        3D motion transfer aims at transferring the motion from a dynamic input sequence to a static 3D object and outputs an identical motion of the target with high-fidelity and realistic visual effects. In this work, we propose a novel 3D Transformer framework called LART for 3D motion transfer. With carefully-designed architectures, LART is able to implicitly learn the correspondence via a flexible geometry perception. Thus, unlike other existing methods, LART does not require any key point annotations or pre-defined correspondence between the motion source and target meshes and can also handle large-size full-detailed unseen 3D targets. Besides, we introduce a novel latent metric regularization on the Transformer for better motion generation. Our rationale lies in the observation that the decoded motions can be approximately expressed as linearly geometric distortion at the frame level. The metric preservation of motions could be translated to the formation of linear paths in the underlying latent space as a rigorous constraint to control the synthetic motions occurring in the construction of the latent space. The proposed LART shows a high learning efficiency with the need for a few samples from the AMASS dataset to generate motions with plausible visual effects. The experimental results verify the potential of our generative model in applications of motion transfer, content generation, temporal interpolation, and motion denoising. The code is made available: https://github.com/mikecheninoulu/LART.

        ----

        ## [1897] Neural Relation Graph: A Unified Framework for Identifying Label Noise and Outlier Data

        **Authors**: *Jang-Hyun Kim, Sangdoo Yun, Hyun Oh Song*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/886ed40d7882c9f891824e42a452c228-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/886ed40d7882c9f891824e42a452c228-Abstract-Conference.html)

        **Abstract**:

        Diagnosing and cleaning data is a crucial step for building robust machine learning systems. However, identifying problems within large-scale datasets with real-world distributions is challenging due to the presence of complex issues such as label errors, under-representation, and outliers. In this paper, we propose a unified approach for identifying the problematic data by utilizing a largely ignored source of information: a relational structure of data in the feature-embedded space. To this end, we present scalable and effective algorithms for detecting label errors and outlier data based on the relational graph structure of data. We further introduce a visualization tool that provides contextual information of a data point in the feature-embedded space, serving as an effective tool for interactively diagnosing data. We evaluate the label error and outlier/out-of-distribution (OOD) detection performances of our approach on the large-scale image, speech, and language domain tasks, including ImageNet, ESC-50, and SST2. Our approach achieves state-of-the-art detection performance on all tasks considered and demonstrates its effectiveness in debugging large-scale real-world datasets across various domains. We release codes at https://github.com/snu-mllab/Neural-Relation-Graph.

        ----

        ## [1898] Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory

        **Authors**: *Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, Rui Yan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/887262aeb3eafb01ef0fd0e3a87a8831-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/887262aeb3eafb01ef0fd0e3a87a8831-Abstract-Conference.html)

        **Abstract**:

        With direct access to human-written reference as memory, retrieval-augmented generation has achieved much progress in a wide range of text generation tasks. Since better memory would typically prompt better generation (we define this as primal problem). The traditional approach for memory retrieval involves selecting memory that exhibits the highest similarity to the input. However, this method is constrained by the quality of the fixed corpus from which memory is retrieved. In this paper, by exploring the duality of the primal problem: better generation also prompts better memory, we propose a novel framework, selfmem, which addresses this limitation by iteratively employing a retrieval-augmented generator to create an unbounded memory pool and using a memory selector to choose one output as memory for the subsequent generation round. This enables the model to leverage its own output, referred to as self-memory, for improved generation. We evaluate the effectiveness of selfmem on three distinct text generation tasks: neural machine translation, abstractive text summarization, and dialogue generation, under two generation paradigms: fine-tuned small model and few-shot LLM. Our approach achieves state-of-the-art results in four directions in JRC-Acquis translation dataset, 50.3 ROUGE-1 in XSum, and 62.9 ROUGE-1 in BigPatent, demonstrating the potential of self-memory in enhancing retrieval-augmented generation models. Furthermore, we conduct thorough analyses of each component in the selfmem framework to identify current system bottlenecks and provide insights for future research.

        ----

        ## [1899] Front-door Adjustment Beyond Markov Equivalence with Limited Graph Knowledge

        **Authors**: *Abhin Shah, Karthikeyan Shanmugam, Murat Kocaoglu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/887932131fddf943e8fe3310b62c0147-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/887932131fddf943e8fe3310b62c0147-Abstract-Conference.html)

        **Abstract**:

        Causal effect estimation from data typically requires assumptions about the cause-effect relations either explicitly in the form of a causal graph structure within the Pearlian framework, or implicitly in terms of (conditional) independence statements between counterfactual variables within the potential outcomes framework. When the treatment variable and the outcome variable are confounded, front-door adjustment is an important special case where, given the graph, causal effect of the treatment on the target can be estimated using post-treatment variables. However, the exact formula for front-door adjustment depends on the structure of the graph, which is difficult to learn in practice. In this work, we provide testable conditional independence statements to compute the causal effect using front-door-like adjustment without knowing the graph under limited structural side information. We show that our method is applicable in scenarios where knowing the Markov equivalence class is not sufficient for causal effect estimation. We demonstrate the effectiveness of our method on a class of random graphs as well as real causal fairness benchmarks.

        ----

        ## [1900] Training Energy-Based Normalizing Flow with Score-Matching Objectives

        **Authors**: *Chen-Hao Chao, Wei-Fang Sun, Yen-Chang Hsu, Zsolt Kira, Chun-Yi Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8882d370cdafec9885b918a8cfac642e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8882d370cdafec9885b918a8cfac642e-Abstract-Conference.html)

        **Abstract**:

        In this paper, we establish a connection between the parameterization of flow-based and energy-based generative models, and present a new flow-based modeling approach called energy-based normalizing flow (EBFlow). We demonstrate that by optimizing EBFlow with score-matching objectives, the computation of Jacobian determinants for linear transformations can be entirely bypassed. This feature enables the use of arbitrary linear layers in the construction of flow-based models without increasing the computational time complexity of each training iteration from $\mathcal{O}(D^2L)$ to $\mathcal{O}(D^3L)$ for an $L$-layered model that accepts $D$-dimensional inputs. This makes the training of EBFlow more efficient than the commonly-adopted maximum likelihood training method. In addition to the reduction in runtime, we enhance the training stability and empirical performance of EBFlow through a number of techniques developed based on our analysis of the score-matching methods. The experimental results demonstrate that our approach achieves a significant speedup compared to maximum likelihood estimation while outperforming prior methods with a noticeable margin in terms of negative log-likelihood (NLL).

        ----

        ## [1901] LayoutPrompter: Awaken the Design Ability of Large Language Models

        **Authors**: *Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian-Guang Lou, Dongmei Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88a129e44f25a571ae8b838057c46855-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88a129e44f25a571ae8b838057c46855-Abstract-Conference.html)

        **Abstract**:

        Conditional graphic layout generation, which automatically maps user constraints to high-quality layouts, has attracted widespread attention today. Although recent works have achieved promising performance, the lack of versatility and data efficiency hinders their practical applications. In this work, we propose LayoutPrompter, which leverages large language models (LLMs) to address the above problems through in-context learning. LayoutPrompter is made up of three key components, namely input-output serialization, dynamic exemplar selection and layout ranking. Specifically, the input-output serialization component meticulously designs the input and output formats for each layout generation task. Dynamic exemplar selection is responsible for selecting the most helpful prompting exemplars for a given input. And a layout ranker is used to pick the highest quality layout from multiple outputs of LLMs. We conduct experiments on all existing layout generation tasks using four public datasets. Despite the simplicity of our approach, experimental results show that LayoutPrompter can compete with or even outperform state-of-the-art approaches on these tasks without any model training or fine-tuning. This demonstrates the effectiveness of this versatile and training-free approach. In addition, the ablation studies show that LayoutPrompter is significantly superior to the training-based baseline in a low-data regime, further indicating the data efficiency of LayoutPrompter. Our project is available at https://github.com/microsoft/LayoutGeneration/tree/main/LayoutPrompter.

        ----

        ## [1902] Classification of Heavy-tailed Features in High Dimensions: a Superstatistical Approach

        **Authors**: *Urte Adomaityte, Gabriele Sicuro, Pierpaolo Vivo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html)

        **Abstract**:

        We characterise the learning of a mixture of two clouds of data points with generic centroids via empirical risk minimisation in the high dimensional regime, under the assumptions of generic convex loss and convex regularisation. Each cloud of data points is obtained via a double-stochastic process, where the sample is obtained from a Gaussian distribution whose variance is itself a random parameter sampled from a scalar distribution $\varrho$. As a result, our analysis covers a large family of data distributions, including the case of power-law-tailed distributions with no covariance, and allows us to test recent ''Gaussian universality'' claims. We study the generalisation performance of the obtained estimator, we analyse the role of regularisation, and we analytically characterise the separability transition.

        ----

        ## [1903] CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra

        **Authors**: *Andres Potapczynski, Marc Finzi, Geoff Pleiss, Andrew Gordon Wilson*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html)

        **Abstract**:

        Many areas of machine learning and science involve large linear algebra problems, such as eigendecompositions, solving linear systems, computing matrix exponentials, and trace estimation. The matrices involved often have Kronecker, convolutional, block diagonal, sum, or product structure. In this paper, we propose a simple but general framework for large-scale linear algebra problems in machine learning, named CoLA (Compositional Linear Algebra). By combining a linear operator abstraction with compositional dispatch rules, CoLA automatically constructs memory and runtime efficient numerical algorithms. Moreover, CoLA provides memory efficient automatic differentiation, low precision computation, and GPU acceleration in both JAX and PyTorch, while also accommodating new objects, operations, and rules in downstream packages via multiple dispatch. CoLA can accelerate many algebraic operations, while making it easy to prototype matrix structures and algorithms, providing an appealing drop-in tool for virtually any computational effort that requires linear algebra. We showcase its efficacy across a broad range of applications, including partial differential equations, Gaussian processes, equivariant model construction, and unsupervised learning.

        ----

        ## [1904] Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language

        **Authors**: *Eghbal A. Hosseini, Evelina Fedorenko*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88dddaf430b5bc38ab8228902bb61821-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88dddaf430b5bc38ab8228902bb61821-Abstract-Conference.html)

        **Abstract**:

        Predicting upcoming events is critical to our ability to effectively interact with ourenvironment and conspecifics. In natural language processing, transformer models,which are trained on next-word prediction, appear to construct a general-purposerepresentation of language that can support diverse downstream tasks. However, westill lack an understanding of how a predictive objective shapes such representations.Inspired by recent work in vision neuroscience Hénaff et al. (2019), here we test ahypothesis about predictive representations of autoregressive transformer models.In particular, we test whether the neural trajectory of a sequence of words in asentence becomes progressively more straight as it passes through the layers of thenetwork. The key insight behind this hypothesis is that straighter trajectories shouldfacilitate prediction via linear extrapolation. We quantify straightness using a 1-dimensional curvature metric, and present four findings in support of the trajectorystraightening hypothesis: i) In trained models, the curvature progressively decreasesfrom the first to the middle layers of the network. ii) Models that perform better onthe next-word prediction objective, including larger models and models trained onlarger datasets, exhibit greater decreases in curvature, suggesting that this improvedability to straighten sentence neural trajectories may be the underlying driver ofbetter language modeling performance. iii) Given the same linguistic context, thesequences that are generated by the model have lower curvature than the groundtruth (the actual continuations observed in a language corpus), suggesting thatthe model favors straighter trajectories for making predictions. iv) A consistentrelationship holds between the average curvature and the average surprisal ofsentences in the middle layers of models, such that sentences with straighter neuraltrajectories also have lower surprisal. Importantly, untrained models don’t exhibitthese behaviors. In tandem, these results support the trajectory straighteninghypothesis and provide a possible mechanism for how the geometry of the internalrepresentations of autoregressive models supports next word prediction.

        ----

        ## [1905] Weakly Coupled Deep Q-Networks

        **Authors**: *Ibrahim El Shar, Daniel Jiang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8912b4892064a4f08a0c04f92913c134-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8912b4892064a4f08a0c04f92913c134-Abstract-Conference.html)

        **Abstract**:

        We propose weakly coupled deep Q-networks (WCDQN), a novel deep reinforcement learning algorithm that enhances performance in a class of structured problems called weakly coupled Markov decision processes (WCMDP). WCMDPs consist of multiple independent subproblems connected by an action space constraint, which is a structural property that frequently emerges in practice. Despite this appealing structure, WCMDPs quickly become intractable as the number of subproblems grows. WCDQN employs a single network to train multiple DQN ``subagents,'' one for each subproblem, and then combine their solutions to establish an upper bound on the optimal action value. This guides the main DQN agent towards optimality. We show that the tabular version, weakly coupled Q-learning (WCQL), converges almost surely to the optimal action value. Numerical experiments show faster convergence compared to DQN and related techniques in settings with as many as 10 subproblems, $3^{10}$ total actions, and a continuous state space.

        ----

        ## [1906] Provably Fast Convergence of Independent Natural Policy Gradient for Markov Potential Games

        **Authors**: *Youbang Sun, Tao Liu, Ruida Zhou, P. R. Kumar, Shahin Shahrampour*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html)

        **Abstract**:

        This work studies an independent natural policy gradient (NPG) algorithm for the multi-agent reinforcement learning problem in Markov potential games. It is shown that, under mild technical assumptions and the introduction of the \textit{suboptimality gap}, the independent NPG method with an oracle providing exact policy evaluation asymptotically reaches an $\epsilon$-Nash Equilibrium (NE) within $\mathcal{O}(1/\epsilon)$ iterations. This improves upon the previous best result of $\mathcal{O}(1/\epsilon^2)$ iterations and is of the same order, $\mathcal{O}(1/\epsilon)$, that is achievable for the single-agent case. Empirical results for a synthetic potential game and a congestion game are presented to verify the theoretical bounds.

        ----

        ## [1907] MMGP: a Mesh Morphing Gaussian Process-based machine learning method for regression of physical problems under nonparametrized geometrical variability

        **Authors**: *Fabien Casenave, Brian Staber, Xavier Roynard*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html)

        **Abstract**:

        When learning simulations for modeling physical phenomena in industrial designs, geometrical variabilities are of prime interest. While classical regression techniques prove effective for parameterized geometries, practical scenarios often involve the absence of shape parametrization during the inference stage, leaving us with only mesh discretizations as available data. Learning simulations from such mesh-based representations poses significant challenges, with recent advances relying heavily on deep graph neural networks to overcome the limitations of conventional machine learning approaches. Despite their promising results, graph neural networks exhibit certain drawbacks, including their dependency on extensive datasets and limitations in providing built-in predictive uncertainties or handling large meshes. In this work, we propose a machine learning method that do not rely on graph neural networks. Complex geometrical shapes and variations with fixed topology are dealt with using well-known mesh morphing onto a common support, combined with classical dimensionality reduction techniques and Gaussian processes. The proposed methodology can easily deal with large meshes without the need for explicit shape parameterization and provides crucial predictive uncertainties, which are essential for informed decision-making. In the considered numerical experiments, the proposed method is competitive with respect to existing graph neural networks, regarding training efficiency and accuracy of the predictions.

        ----

        ## [1908] Adaptive Online Replanning with Diffusion Models

        **Authors**: *Siyuan Zhou, Yilun Du, Shun Zhang, Mengdi Xu, Yikang Shen, Wei Xiao, Dit-Yan Yeung, Chuang Gan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/893a5db6100028ec814cfd99fe92c31b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/893a5db6100028ec814cfd99fe92c31b-Abstract-Conference.html)

        **Abstract**:

        Diffusion models have risen a promising approach to data-driven planning, and have demonstrated impressive robotic control, reinforcement learning, and video planning performance. Given an effective planner, an important question to consider is replanning -- when given plans should be regenerated due to both action execution error and external environment changes.  Direct plan execution, without replanning, is problematic as errors from individual actions rapidly accumulate and environments are partially observable and stochastic. Simultaneously, replanning at each timestep incurs a substantial computational cost, and may prevent successful task execution, as different generated plans prevent consistent progress to any particular goal. In this paper, we explore how we may effectively replan with diffusion models. We propose a principled approach to determine when to replan, based on the diffusion model's estimated likelihood of existing generated plans. We further present an approach to replan existing trajectories to ensure that new plans follow the same goal state as the original trajectory, which may efficiently bootstrap off previously generated plans.  We illustrate how a combination of our proposed additions significantly improves the performance of diffusion planners leading to 38\% gains over past diffusion planning approaches on Maze2D and further enables handling of stochastic and long-horizon robotic control tasks.

        ----

        ## [1909] SODA: Robust Training of Test-Time Data Adaptors

        **Authors**: *Zige Wang, Yonggang Zhang, Zhen Fang, Long Lan, Wenjing Yang, Bo Han*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/893ca2e5ff5bb258da30e0a82f4c8de9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/893ca2e5ff5bb258da30e0a82f4c8de9-Abstract-Conference.html)

        **Abstract**:

        Adapting models deployed to test distributions can mitigate the performance degradation caused by distribution shifts. However, privacy concerns may render model parameters inaccessible. One promising approach involves utilizing zeroth-order optimization (ZOO) to train a data adaptor to adapt the test data to fit the deployed models. Nevertheless, the data adaptor trained with ZOO typically brings restricted improvements due to the potential corruption of data features caused by the data adaptor. To address this issue, we revisit ZOO in the context of test-time data adaptation. We find that the issue directly stems from the unreliable estimation of the gradients used to optimize the data adaptor, which is inherently due to the unreliable nature of the pseudo-labels assigned to the test data. Based on this observation, we propose pseudo-label-robust data adaptation (SODA) to improve the performance of data adaptation. Specifically, SODA leverages high-confidence predicted labels as reliable labels to optimize the data adaptor with ZOO for label prediction. For data with low-confidence predictions, SODA encourages the adaptor to preserve data information to mitigate data corruption. Empirical results indicate that SODA can significantly enhance the performance of deployed models in the presence of distribution shifts without requiring access to model parameters.

        ----

        ## [1910] Training Neural Networks is NP-Hard in Fixed Dimension

        **Authors**: *Vincent Froese, Christoph Hertrich*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8948a8d039ed52d1031db6c7c2373378-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8948a8d039ed52d1031db6c7c2373378-Abstract-Conference.html)

        **Abstract**:

        We study the parameterized complexity of training two-layer neural networks with respect to the dimension of the input data and the number of hidden neurons, considering ReLU and linear threshold activation functions. Albeit the computational complexity of these problems has been studied numerous times in recent years, several questions are still open. We answer questions by Arora et al. (ICLR 2018) and Khalife and Basu (IPCO 2022) showing that both problems are NP-hard for two dimensions, which excludes any polynomial-time algorithm for constant dimension. We also answer a question by Froese et al. (JAIR 2022) proving W[1]-hardness for four ReLUs (or two linear threshold neurons) with zero training error. Finally, in the ReLU case, we show fixed-parameter tractability for the combined parameter number of dimensions and number of ReLUs if the network is assumed to compute a convex map. Our results settle the complexity status regarding these parameters almost completely.

        ----

        ## [1911] GlyphControl: Glyph Conditional Control for Visual Text Generation

        **Authors**: *Yukang Yang, Dongnan Gui, Yuhui Yuan, Weicong Liang, Haisong Ding, Han Hu, Kai Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8951bbdcf234132bcce680825e7cb354-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8951bbdcf234132bcce680825e7cb354-Abstract-Conference.html)

        **Abstract**:

        Recently, there has been an increasing interest in developing diffusion-based text-to-image generative models capable of generating coherent and well-formed visual text. In this paper, we propose a novel and efficient approach called GlyphControl to address this task. Unlike existing methods that rely on character-aware text encoders like ByT5 and require retraining of text-to-image models, our approach leverages additional glyph conditional information to enhance the performance of the off-the-shelf Stable-Diffusion model in generating accurate visual text. By incorporating glyph instructions, users can customize the content, location, and size of the generated text according to their specific requirements. To facilitate further research in visual text generation, we construct a training benchmark dataset called LAION-Glyph. We evaluate the effectiveness of our approach by measuring OCR-based metrics, CLIP score, and FID of the generated visual text. Our empirical evaluations demonstrate that GlyphControl outperforms the recent DeepFloyd IF approach in terms of OCR accuracy, CLIP score, and FID, highlighting the efficacy of our method.

        ----

        ## [1912] Domain Adaptive Imitation Learning with Visual Observation

        **Authors**: *Sungho Choi, Seungyul Han, Woojun Kim, Jongseong Chae, Whiyoung Jung, Youngchul Sung*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/899511e37a8e01e1bd6f6f1d377cc250-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/899511e37a8e01e1bd6f6f1d377cc250-Abstract-Conference.html)

        **Abstract**:

        In this paper, we consider domain-adaptive imitation learning with visual observation, where an agent in a target domain learns to perform a task by observing expert demonstrations in a source domain. Domain adaptive imitation learning arises in practical scenarios where a robot, receiving visual sensory data, needs to mimic movements by visually observing other robots from different angles or observing robots of different shapes. To overcome the domain shift in cross-domain imitation learning with visual observation, we propose a novel framework for extracting domain-independent behavioral features from input observations that can be used to train the learner, based on dual feature extraction and image reconstruction. Empirical results demonstrate that our approach outperforms previous algorithms for imitation learning from visual observation with domain shift.

        ----

        ## [1913] Discriminative Feature Attributions: Bridging Post Hoc Explainability and Inherent Interpretability

        **Authors**: *Usha Bhalla, Suraj Srinivas, Himabindu Lakkaraju*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/89beb2a345269f3f9afe48cee35403aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/89beb2a345269f3f9afe48cee35403aa-Abstract-Conference.html)

        **Abstract**:

        With the increased deployment of machine learning models in various real-world applications, researchers and practitioners alike have emphasized the need for explanations of model behaviour. To this end, two broad strategies have been outlined in prior literature to explain models. Post hoc explanation methods explain the behaviour of complex black-box models by identifying features critical to model predictions; however, prior work has shown that these explanations may not be faithful, in that they incorrectly attribute high importance to features that are unimportant or non-discriminative for the underlying task. Inherently interpretable models, on the other hand, circumvent these issues by explicitly encoding explanations into model architecture, meaning their explanations are naturally faithful, but they often exhibit poor predictive performance due to their limited expressive power. In this work, we identify a key reason for the lack of faithfulness of feature attributions: the lack of robustness of the underlying black-box models, especially the erasure of unimportant distractor features in the input. To address this issue, we propose Distractor Erasure Tuning (DiET), a method that adapts black-box models to be robust to distractor erasure, thus providing discriminative and faithful attributions. This strategy naturally combines the ease-of-use of post hoc explanations with the faithfulness of inherently interpretable models. We perform extensive experiments on semi-synthetic and real-world datasets, and show that DiET produces models that (1) closely approximate the original black-box models they are intended to explain, and (2) yield explanations that match approximate ground truths available by construction.

        ----

        ## [1914] LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models

        **Authors**: *Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya K, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John J. Nay, Jonathan H. Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael A. Livermore, Nikon Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, Zehua Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The advent of large language models (LLMs) and their adoption by the legal community has given rise to the question: what types of legal reasoning can LLMs perform? To enable greater study of this question, we present LegalBench: a collaboratively constructed legal reasoning benchmark consisting of 162 tasks covering six different types of legal reasoning. LegalBench was built through an interdisciplinary process, in which we collected tasks designed and hand-crafted by legal professionals. Because these subject matter experts took a leading role in construction, tasks either measure legal reasoning capabilities that are practically useful, or measure reasoning skills that lawyers find interesting. To enable cross-disciplinary conversations about LLMs in the law, we additionally show how popular legal frameworks for describing legal reasoning—which distinguish between its many forms—correspond to LegalBench tasks, thus giving lawyers and LLM developers a common vocabulary. This paper describes LegalBench, presents an empirical evaluation of 20 open-source and commercial LLMs, and illustrates the types of research explorations LegalBench enables.

        ----

        ## [1915] Detecting hidden confounding in observational data using multiple environments

        **Authors**: *Rickard Karlsson, Jesse H. Krijthe*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/89e541b817ea043a971840a926e12b37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/89e541b817ea043a971840a926e12b37-Abstract-Conference.html)

        **Abstract**:

        A common assumption in causal inference from observational data is that there is no hidden confounding. Yet it is, in general, impossible to verify the presence of hidden confounding factors from a single dataset. Under the assumption of independent causal mechanisms underlying the data-generating process, we demonstrate a way to detect unobserved confounders when having multiple observational datasets coming from different environments. We present a theory for testable conditional independencies that are only absent when there is hidden confounding and examine cases where we violate its assumptions: degenerate & dependent mechanisms, and faithfulness violations. Additionally, we propose a procedure to test these independencies and study its empirical finite-sample behavior using simulation studies and semi-synthetic data based on a real-world dataset. In most cases, the proposed procedure correctly predicts the presence of hidden confounding, particularly when the confounding bias is large.

        ----

        ## [1916] Information Geometry of the Retinal Representation Manifold

        **Authors**: *Xuehao Ding, Dongsoo Lee, Joshua Melander, George Sivulka, Surya Ganguli, Stephen Baccus*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8a267516a7a697965c6ae4f48b908605-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8a267516a7a697965c6ae4f48b908605-Abstract-Conference.html)

        **Abstract**:

        The ability for the brain to discriminate among visual stimuli is constrained by their retinal representations. Previous studies of visual discriminability have been limited to either low-dimensional artificial stimuli or pure theoretical considerations without a realistic encoding model. Here we propose a novel framework for understanding stimulus discriminability achieved by retinal representations of naturalistic stimuli with the method of information geometry. To model the joint probability distribution of neural responses conditioned on the stimulus, we created a stochastic encoding model of a population of salamander retinal ganglion cells based on a three-layer convolutional neural network model. This model not only accurately captured the mean response to natural scenes but also a variety of second-order statistics. With the model and the proposed theory, we computed the Fisher information metric over stimuli to study the most discriminable stimulus directions. We found that the most discriminable stimulus varied substantially across stimuli, allowing an examination of the relationship between the most discriminable stimulus and the current stimulus. By examining responses generated by the most discriminable stimuli we further found that the most discriminative response mode is often aligned with the most stochastic mode. This finding carries the important implication that under natural scenes, retinal noise correlations are information-limiting rather than increasing information transmission as has been previously speculated. We additionally observed that sensitivity saturates less in the population than for single cells and that as a function of firing rate, Fisher information varies less than sensitivity. We conclude that under natural scenes, population coding benefits from complementary coding and helps to equalize the information carried by different firing rates, which may facilitate decoding of the stimulus under principles of information maximization.

        ----

        ## [1917] RoboHive: A Unified Framework for Robot Learning

        **Authors**: *Vikash Kumar, Rutav M. Shah, Gaoyue Zhou, Vincent Moens, Vittorio Caggiano, Abhishek Gupta, Aravind Rajeswaran*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8a84a4341c375b8441b36836bb343d4e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8a84a4341c375b8441b36836bb343d4e-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We present RoboHive, a comprehensive software platform and ecosystem for research in the field of Robot Learning and Embodied Artificial Intelligence. Our platform encompasses a diverse range of pre-existing and novel environments, including dexterous manipulation with the Shadow Hand, whole-arm manipulation tasks with Franka and Fetch robots, quadruped locomotion, among others. Included environments are organized within and cover multiple domains such as hand manipulation, locomotion, multi-task, multi-agent, muscles, etc. In comparison to prior works, RoboHive offers a streamlined and unified task interface taking dependency on only a minimal set of well-maintained packages, features tasks with high physics fidelity and rich visual diversity, and supports common hardware drivers for real-world deployment. The unified interface of RoboHive offers a convenient and accessible abstraction for algorithmic research in imitation, reinforcement, multi-task, and hierarchical learning. Furthermore, RoboHive includes expert demonstrations and baseline results for most environments, providing a standard for benchmarking and comparisons. Details: https://sites.google.com/view/robohive

        ----

        ## [1918] Sequential Memory with Temporal Predictive Coding

        **Authors**: *Mufeng Tang, Helen Barron, Rafal Bogacz*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8a8b9c7f979e8819a7986b3ef825c08a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8a8b9c7f979e8819a7986b3ef825c08a-Abstract-Conference.html)

        **Abstract**:

        Forming accurate memory of sequential stimuli is a fundamental function of biological agents. However, the computational mechanism underlying sequential memory in the brain remains unclear. Inspired by neuroscience theories and recent successes in applying predictive coding (PC) to \emph{static} memory tasks, in this work we propose a novel PC-based model for \emph{sequential} memory, called \emph{temporal predictive coding} (tPC). We show that our tPC models can memorize and retrieve sequential inputs accurately with a biologically plausible neural implementation. Importantly, our analytical study reveals that tPC can be viewed as a classical Asymmetric Hopfield Network (AHN) with an implicit statistical whitening process, which leads to more stable performance in sequential memory tasks of structured inputs. Moreover, we find that tPC exhibits properties consistent with behavioral observations and theories in neuroscience, thereby strengthening its biological relevance. Our work establishes a possible computational mechanism underlying sequential memory in the brain that can also be theoretically interpreted using existing memory model frameworks.

        ----

        ## [1919] Transportability for Bandits with Data from Different Environments

        **Authors**: *Alexis Bellot, Alan Malek, Silvia Chiappa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8a8ce53beb3775522305e0a6033d4455-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8a8ce53beb3775522305e0a6033d4455-Abstract-Conference.html)

        **Abstract**:

        A unifying theme in the design of intelligent agents is to efficiently optimize a policy based on what prior knowledge of the problem is available and what actions can be taken to learn more about it. Bandits are a canonical instance of this task that has been intensely studied in the literature. Most methods, however, typically rely solely on an agent's experimentation in a single environment (or multiple closely related environments). In this paper, we relax this assumption and consider the design of bandit algorithms from a combination of batch data and qualitative assumptions about the relatedness across different environments, represented in the form of causal models. In particular, we show that it is possible to exploit invariances across environments, wherever they may occur in the underlying causal model, to consistently improve learning. The resulting bandit algorithm has a sub-linear regret bound with an explicit dependency on a term that captures how informative related environments are for the task at hand; and may have substantially lower regret than experimentation-only bandit instances.

        ----

        ## [1920] Students Parrot Their Teachers: Membership Inference on Model Distillation

        **Authors**: *Matthew Jagielski, Milad Nasr, Katherine Lee, Christopher A. Choquette-Choo, Nicholas Carlini, Florian Tramèr*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b07d224a643b02e7571e083578a86d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b07d224a643b02e7571e083578a86d2-Abstract-Conference.html)

        **Abstract**:

        Model distillation is frequently proposed as a technique to reduce the privacy leakage of machine learning. These empirical privacy defenses rely on the intuition that distilled student'' models protect the privacy of training data, as they only interact with this data indirectly through ateacher'' model. In this work, we design membership inference attacks to systematically study the privacy provided by knowledge distillation to both the teacher and student training sets. Our new attacks show that distillation alone provides only limited privacy across a number of domains. We explain the success of our attacks on distillation by showing that membership inference attacks on a private dataset can succeed even if the target model is never queried on any actual training points, but only on inputs whose predictions are highly influenced by training data. Finally, we show that our attacks are strongest when student and teacher sets are similar, or when the attacker can poison the teacher set.

        ----

        ## [1921] DiffuseBot: Breeding Soft Robots With Physics-Augmented Generative Diffusion Models

        **Authors**: *Tsun-Hsuan Johnson Wang, Juntian Zheng, Pingchuan Ma, Yilun Du, Byungchul Kim, Andrew Spielberg, Josh B. Tenenbaum, Chuang Gan, Daniela Rus*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b1008098947ad59144c18a78337f937-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b1008098947ad59144c18a78337f937-Abstract-Conference.html)

        **Abstract**:

        Nature evolves creatures with a high complexity of morphological and behavioral intelligence, meanwhile computational methods lag in approaching that diversity and efficacy.  Co-optimization of artificial creatures' morphology and control in silico shows promise for applications in physical soft robotics and virtual character creation; such approaches, however, require developing new learning algorithms that can reason about function atop pure structure. In this paper, we present DiffuseBot, a physics-augmented diffusion model that generates soft robot morphologies capable of excelling in a wide spectrum of tasks. \name bridges the gap between virtually generated content and physical utility by (i) augmenting the diffusion process with a physical dynamical simulation which provides a certificate of performance, and (ii) introducing a co-design procedure that jointly optimizes physical design and control by leveraging information about physical sensitivities from differentiable simulation.  We showcase a range of simulated and fabricated robots along with their capabilities. Check our website: https://diffusebot.github.io/

        ----

        ## [1922] Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks

        **Authors**: *Jimmy Z. Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html)

        **Abstract**:

        We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected. In particular, we consider clean-label targeted attacks (in which the goal is to cause the model to misclassify a specific test point) on datasets including CIFAR-10, Imagenette, and Imagewoof. This attack is realized by constructing camouflage datapoints that mask the effect of a poisoned dataset. We demonstrate efficacy of our attack when unlearning is performed via retraining from scratch, the idealized setting of machine unlearning which other efficient methods attempt to emulate, as well as against the approximate unlearning approach of Graves et al. (2021).

        ----

        ## [1923] Thought Cloning: Learning to Think while Acting by Imitating Human Thinking

        **Authors**: *Shengran Hu, Jeff Clune*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b4ba64e549c410185c4d3eac3a81726-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b4ba64e549c410185c4d3eac3a81726-Abstract-Conference.html)

        **Abstract**:

        Language is often considered a key aspect of human thinking, providing us with exceptional abilities to generalize, explore, plan, replan, and adapt to new situations. However, Reinforcement Learning (RL) agents are far from human-level performance in any of these abilities. We hypothesize one reason for such cognitive deficiencies is that they lack the benefits of thinking in language and that we can improve AI agents by training them to $\textit{think like humans do}$. We introduce a novel Imitation Learning framework, Thought Cloning, where the idea is to not just clone the behaviors of human demonstrators, $\textit{but also the thoughts humans have as they perform these behaviors}$. While we expect Thought Cloning to truly shine at scale on internet-sized datasets (e.g. online videos with transcripts), here we conduct experiments in a domain where the thinking and action data are synthetically generated. Results reveal that Thought Cloning learns much faster than Behavioral Cloning and its performance advantage grows the further out of distribution test tasks are, highlighting its ability to better handle novel situations. Thought Cloning also provides important benefits for AI Safety and Interpretability, and makes it easier to debug and improve AI. Because we can observe the agentâ€™s thoughts, we can (1) more easily diagnose why things are going wrong, making it easier to fix the problem, (2) steer the agent by correcting its thinking, or (3) prevent it from doing unsafe things it plans to do. Overall, by training agents $\textit{how to think}$ as well as behave, Thought Cloning creates safer, more powerful agents.

        ----

        ## [1924] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities

        **Authors**: *Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b54ecd9823fff6d37e61ece8f87e534-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b54ecd9823fff6d37e61ece8f87e534-Abstract-Conference.html)

        **Abstract**:

        Many approaches in machine learning rely on a weighted graph to encode thesimilarities between samples in a dataset. Entropic affinities (EAs), which are notably used in the popular Dimensionality Reduction (DR) algorithm t-SNE, are particular instances of such graphs. To ensure robustness to heterogeneous sampling densities, EAs assign a kernel bandwidth parameter to every sample in such a way that the entropy of each row in the affinity matrix is kept constant at a specific value, whose exponential is known as perplexity. EAs are inherently asymmetric and row-wise stochastic, but they are used in DR approaches after undergoing heuristic symmetrization methods that violate both the row-wise constant entropy and stochasticity properties. In this work, we uncover a novel characterization of EA as an optimal transport problem, allowing a natural symmetrization that can be computed efficiently using dual ascent. The corresponding novel affinity matrix derives advantages from symmetric doubly stochastic normalization in terms of clustering performance, while also effectively controlling the entropy of each row thus making it particularly robust to varying noise levels. Following, we present a new DR algorithm, SNEkhorn, that leverages this new affinity matrix. We show its clear superiority to state-of-the-art approaches with several indicators on both synthetic and real-world datasets.

        ----

        ## [1925] Hyperbolic Graph Neural Networks at Scale: A Meta Learning Approach

        **Authors**: *Nurendra Choudhary, Nikhil Rao, Chandan K. Reddy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b6a8b010e9a266aad40a024c5976d5c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b6a8b010e9a266aad40a024c5976d5c-Abstract-Conference.html)

        **Abstract**:

        The progress in hyperbolic neural networks (HNNs) research is hindered by their absence of inductive bias mechanisms, which are essential for generalizing to new tasks and facilitating scalable learning over large datasets. In this paper, we aim to alleviate these issues by learning generalizable inductive biases from the nodesâ€™ local subgraph and transfer them for faster learning over new subgraphs with a disjoint set of nodes, edges, and labels in a few-shot setting. We introduce a novel method, Hyperbolic GRAph Meta Learner (H-GRAM), that, for the tasks of node classification and link prediction, learns transferable information from a set of support local subgraphs in the form of hyperbolic meta gradients and label hyperbolic protonets to enable faster learning over a query set of new tasks dealing with disjoint subgraphs. Furthermore, we show that an extension of our meta-learning framework also mitigates the scalability challenges seen in HNNs faced by existing approaches. Our comparative analysis shows that H-GRAM effectively learns and transfers information in multiple challenging few-shot settings compared to other state-of-the-art baselines. Additionally, we demonstrate that, unlike standard HNNs, our approach is able to scale over large graph datasets and improve performance over its Euclidean counterparts.

        ----

        ## [1926] FELM: Benchmarking Factuality Evaluation of Large Language Models

        **Authors**: *Shiqi Chen, Yiran Zhao, Jinghan Zhang, I-Chun Chern, Siyang Gao, Pengfei Liu, Junxian He*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b8a7960d343e023a6a0afe37eee6022-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b8a7960d343e023a6a0afe37eee6022-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Assessing factuality of text generated by large language models (LLMs) is an emerging yet crucial research area, aimed at alerting users to potential errors and guiding the development of more reliable LLMs. Nonetheless, the evaluators assessing factuality necessitate suitable evaluation themselves to gauge progress and foster advancements. This direction remains under-explored, resulting in substantial impediments to the progress of factuality evaluators. To mitigate this issue, we introduce a benchmark for Factuality Evaluation of large Language Models, referred to as FELM. In this benchmark, we collect responses generated from LLMs and annotate factuality labels in a fine-grained manner. Contrary to previous studies that primarily concentrate on the factuality of world knowledge (e.g. information from Wikipedia), FELM focuses on factuality across diverse domains, spanning from world knowledge to math and reasoning. Our annotation is based on text segments, which can help pinpoint specific factual errors. The factuality annotations are further supplemented by predefined error types and reference links that either support or contradict the statement. In our experiments, we investigate the performance of several LLM-based factuality evaluators on FELM, including both vanilla LLMs and those augmented with retrieval mechanisms and chain-of-thought processes. Our findings reveal that while retrieval aids factuality evaluation, current LLMs are far from satisfactory to faithfully detect factual errors.

        ----

        ## [1927] AircraftVerse: A Large-Scale Multimodal Dataset of Aerial Vehicle Designs

        **Authors**: *Adam B. Cobb, Anirban Roy, Daniel Elenius, F. Michael Heim, Brian Swenson, Sydney Whittington, James D. Walker, Theodore Bapty, Joseph Hite, Karthik Ramani, Christopher McComb, Susmit Jha*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b94879b177d9780c17f5a78f62a6a8a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b94879b177d9780c17f5a78f62a6a8a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We present AircraftVerse, a publicly available aerial vehicle design dataset. Aircraft design encompasses different physics domains and, hence, multiple modalities of representation. The evaluation of these designs requires the use of scientific analytical and simulation models ranging from computer-aided design tools for structural and manufacturing analysis, computational fluid dynamics tools for drag and lift computation, battery models for energy estimation, and simulation models for flight control and dynamics. AircraftVerse contains $27{,}714$  diverse air vehicle designs - the largest corpus of designs with this level of complexity. Each design comprises the following artifacts: a symbolic design tree describing topology, propulsion subsystem, battery subsystem, and other design details; a STandard for the Exchange of Product (STEP) model data; a 3D CAD design using a stereolithography (STL) file format; a 3D point cloud for the shape of the design; and evaluation results from high fidelity state-of-the-art physics models that characterize performance metrics such as maximum flight distance and hover-time. We also present baseline surrogate models that use different modalities of design representation to predict design performance metrics, which we provide as part of our dataset release. Finally, we discuss the potential impact of this dataset on the use of learning in aircraft design, and more generally, in the emerging field of deep learning for scientific design. AircraftVerse is accompanied by a datasheet as suggested in the recent literature, and it is released under Creative Commons Attribution-ShareAlike (CC BY-SA) license. The dataset with baseline models are hosted at http://doi.org/10.5281/zenodo.6525446, code at https://github.com/SRI-CSL/AircraftVerse, and the dataset description at  https://uavdesignverse.onrender.com/.

        ----

        ## [1928] On Separate Normalization in Self-supervised Transformers

        **Authors**: *Xiaohui Chen, Yinkai Wang, Yuanqi Du, Soha Hassoun, Liping Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ba80c47b9d3dced79ee835b7d3bf72a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ba80c47b9d3dced79ee835b7d3bf72a-Abstract-Conference.html)

        **Abstract**:

        Self-supervised training methods for transformers have demonstrated remarkable performance across various domains. Previous transformer-based models, such as masked autoencoders (MAE), typically utilize a single normalization layer for both the [CLS] symbol and the tokens. We propose in this paper a simple modification that employs separate normalization layers for the tokens and the [CLS] symbol to better capture their distinct characteristics and enhance downstream task performance. Our method aims to alleviate the potential negative effects of using the same normalization statistics for both token types, which may not be optimally aligned with their individual roles. We empirically show that by utilizing a separate normalization layer, the [CLS] embeddings can better encode the global contextual information and are distributed more uniformly in its anisotropic space. When replacing the conventional normalization layer with the two separate layers, we observe an average  2.7% performance improvement over the image, natural language, and graph domains.

        ----

        ## [1929] PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection

        **Authors**: *Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue Zhou, Shanghang Zhang, Hao Zhao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8bc5aef775aacc1650a9790f1428bcea-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8bc5aef775aacc1650a9790f1428bcea-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Object anomaly detection is an important problem in the field of machine vision and has seen remarkable progress recently. However, two significant challenges hinder its research and application. First, existing datasets lack comprehensive visual information from various pose angles. They usually have an unrealistic assumption that the anomaly-free training dataset is pose-aligned, and the testing samples have the same pose as the training data. However, in practice, anomaly may exist in any regions on a object, the training and query samples may have different poses, calling for the study on pose-agnostic anomaly detection. Second, the absence of a consensus on experimental protocols for pose-agnostic anomaly detection leads to unfair comparisons of different methods, hindering the research on pose-agnostic anomaly detection. To address these issues, we develop Multi-pose Anomaly Detection (MAD) dataset and Pose-agnostic Anomaly Detection (PAD) benchmark, which takes the first step to address the pose-agnostic anomaly detection problem. Specifically, we build MAD using 20 complex-shaped LEGO toys including 4K views with various poses, and high-quality and diverse 3D anomalies in both simulated and real environments. Additionally, we propose a novel method OmniposeAD, trained using MAD, specifically designed for pose-agnostic anomaly detection. Through comprehensive evaluations, we demonstrate the relevance of our dataset and method. Furthermore, we provide an open-source benchmark library, including dataset and baseline methods that cover 8 anomaly detection paradigms, to facilitate future research and application in this domain. Code, data, and models are publicly available at https://github.com/EricLee0224/PAD.

        ----

        ## [1930] Modulated Neural ODEs

        **Authors**: *Ilze Amanda Auzina, Çagatay Yildiz, Sara Magliacane, Matthias Bethge, Efstratios Gavves*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8bc74514d554a90c996576f6c373f5f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8bc74514d554a90c996576f6c373f5f3-Abstract-Conference.html)

        **Abstract**:

        Neural ordinary differential equations (NODEs) have been proven useful for learning non-linear dynamics of arbitrary trajectories. However, current NODE methods capture variations across trajectories only via the initial state value or by auto-regressive encoder updates. In this work, we introduce Modulated Neural ODEs (MoNODEs), a novel framework that sets apart dynamics states from underlying static factors of variation and improves the existing NODE methods.  In particular, we introduce *time-invariant modulator variables* that are learned from the data. We incorporate our proposed framework into four existing NODE variants. We test MoNODE on oscillating systems, videos and human walking trajectories, where each trajectory has trajectory-specific modulation. Our framework consistently improves the existing model ability to generalize to new dynamic parameterizations and to perform far-horizon forecasting. In addition, we verify that the proposed modulator variables are informative of the true unknown factors of variation as measured by $R^2$ scores.

        ----

        ## [1931] DrugCLIP: Contrasive Protein-Molecule Representation Learning for Virtual Screening

        **Authors**: *Bowen Gao, Bo Qiang, Haichuan Tan, Yinjun Jia, Minsi Ren, Minsi Lu, Jingjing Liu, Wei-Ying Ma, Yanyan Lan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html)

        **Abstract**:

        Virtual screening, which identifies potential drugs from vast compound databases to bind with a particular protein pocket, is a critical step in AI-assisted drug discovery. Traditional docking methods are highly time-consuming, and can only work with a restricted search library in real-life applications. Recent supervised learning approaches using scoring functions for binding-affinity prediction, although promising, have not yet surpassed docking methods due to their strong dependency on limited data with reliable binding-affinity labels. In this paper, we propose a novel contrastive learning framework, DrugCLIP, by reformulating virtual screening as a dense retrieval task and employing contrastive learning to align representations of binding protein pockets and molecules from a large quantity of pairwise data without explicit binding-affinity scores. We also introduce a biological-knowledge inspired data augmentation strategy to learn better protein-molecule representations. Extensive experiments show that DrugCLIP significantly outperforms traditional docking and supervised learning methods on diverse virtual screening benchmarks with highly reduced computation time, especially in zero-shot setting.

        ----

        ## [1932] On the Convergence of Black-Box Variational Inference

        **Authors**: *Kyurae Kim, Jisu Oh, Kaiwen Wu, Yian Ma, Jacob R. Gardner*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8bea36ac39e11ebe49e9eddbd4b8bd3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8bea36ac39e11ebe49e9eddbd4b8bd3a-Abstract-Conference.html)

        **Abstract**:

        We provide the first convergence guarantee for black-box variational inference (BBVI) with the reparameterization gradient.  While preliminary investigations worked on simplified versions of BBVI (e.g., bounded domain, bounded support, only optimizing for the scale, and such), our setup does not need any such algorithmic modifications.  Our results hold for log-smooth posterior densities with and without strong log-concavity and the location-scale variational family.  Notably, our analysis reveals that certain algorithm design choices commonly employed in practice, such as nonlinear parameterizations of the scale matrix, can result in suboptimal convergence rates.  Fortunately, running BBVI with proximal stochastic gradient descent fixes these limitations and thus achieves the strongest known convergence guarantees.  We evaluate this theoretical insight by comparing proximal SGD against other standard implementations of BBVI on large-scale Bayesian inference problems.

        ----

        ## [1933] DRAUC: An Instance-wise Distributionally Robust AUC Optimization Framework

        **Authors**: *Siran Dai, Qianqian Xu, Zhiyong Yang, Xiaochun Cao, Qingming Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c086821724b99f4c756648bb0f165db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c086821724b99f4c756648bb0f165db-Abstract-Conference.html)

        **Abstract**:

        The Area Under the ROC Curve (AUC) is a widely employed metric in long-tailed classification scenarios. Nevertheless, most existing methods primarily assume that training and testing examples are drawn i.i.d. from the same distribution, which is often unachievable in practice. Distributionally Robust Optimization (DRO) enhances model performance by optimizing it for the local worst-case scenario, but directly integrating AUC optimization with DRO results in an intractable optimization problem. To tackle this challenge, methodically we propose an instance-wise surrogate loss of Distributionally Robust AUC (DRAUC) and build our optimization framework on top of it. Moreover, we highlight that conventional DRAUC may induce label bias, hence introducing distribution-aware DRAUC as a more suitable metric for robust AUC learning. Theoretically, we affirm that the generalization gap between the training loss and testing error diminishes if the training set is sufficiently large. Empirically, experiments on corrupted benchmark datasets demonstrate the effectiveness of our proposed method. Code is available at: https://github.com/EldercatSAM/DRAUC.

        ----

        ## [1934] iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models

        **Authors**: *Tianyu Chen, Kevin Bello, Bryon Aragam, Pradeep Ravikumar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c1d92835eb4e601f396c97ec60439fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c1d92835eb4e601f396c97ec60439fe-Abstract-Conference.html)

        **Abstract**:

        Structural causal models (SCMs) are widely used in various disciplines to represent causal relationships among variables in complex systems.Unfortunately, the underlying causal structure is often unknown, and estimating it from data remains a challenging task. In many situations, however, the end goal is to localize the changes (shifts) in the causal mechanisms between related datasets instead of learning the full causal structure of the individual datasets. Some applications include root cause analysis, analyzing gene regulatory network structure changes between healthy and cancerous individuals, or explaining distribution shifts. This paper focuses on identifying the causal mechanism shifts in two or more related datasets over the same set of variables---without estimating the entire DAG structure of each SCM.Prior work under this setting assumed linear models with Gaussian noises; instead, in this work we assume that each SCM belongs to the more general class of nonlinear additive noise models (ANMs).A key technical contribution of this work is to show that the Jacobian of the score function for the mixture distribution allows for the identification of shifts under general non-parametric functional mechanisms.Once the shifted variables are identified, we leverage recent work to estimate the structural differences, if any, for the shifted variables.Experiments on synthetic and real-world data are provided to showcase the applicability of this approach.Code implementing the proposed method is open-source and publicly available at  https://github.com/kevinsbello/iSCAN.

        ----

        ## [1935] Optimal Learners for Realizable Regression: PAC Learning and Online Learning

        **Authors**: *Idan Attias, Steve Hanneke, Alkis Kalavasis, Amin Karbasi, Grigoris Velegkas*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c22e5e918198702765ecff4b20d0a90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c22e5e918198702765ecff4b20d0a90-Abstract-Conference.html)

        **Abstract**:

        In this work, we aim to characterize the statistical complexity of realizable regression both in the PAC learning setting and the online learning setting. Previous work had established the sufficiency of finiteness of the fat shattering dimension for PAC learnability and the necessity of finiteness of the scaled Natarajan dimension, but little progress had been made towards a more complete characterization since  the work of Simon 1997 (SICOMP '97). To this end,  we first introduce a minimax instance optimal learner for realizable regression and propose a novel dimension that both qualitatively and quantitatively characterizes which classes of real-valued predictors are learnable.  We then identify a combinatorial dimension related to the graph dimension that characterizes ERM learnability in the realizable setting. Finally, we establish a necessary condition for learnability based on a combinatorial dimension related to the DS dimension, and conjecture that it may also be sufficient in this context. Additionally, in the context of online learning we provide a dimension that characterizes the minimax instance optimal cumulative loss up to a constant factor and design an optimal online learner for realizable regression, thus resolving an open question raised by Daskalakis and Golowich in STOC '22.

        ----

        ## [1936] Sounding Bodies: Modeling 3D Spatial Sound of Humans Using Body Pose and Audio

        **Authors**: *Xudong Xu, Dejan Markovic, Jacob Sandakly, Todd Keebler, Steven Krenn, Alexander Richard*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c234d9c7e738a793947e0282c36eb95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c234d9c7e738a793947e0282c36eb95-Abstract-Conference.html)

        **Abstract**:

        While 3D human body modeling has received much attention in computer vision, modeling the acoustic equivalent, i.e. modeling 3D spatial audio produced by body motion and speech, has fallen short in the community. To close this gap, we present a model that can generate accurate 3D spatial audio for full human bodies. The system consumes, as input, audio signals from headset microphones and body pose, and produces, as output, a 3D sound field surrounding the transmitter's body, from which spatial audio can be rendered at any arbitrary position in the 3D space. We collect a first-of-its-kind multimodal dataset of human bodies, recorded with multiple cameras and a spherical array of 345 microphones. In an empirical evaluation, we demonstrate that our model can produce accurate body-induced sound fields when trained with a suitable loss. Dataset and code are available online.

        ----

        ## [1937] Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering

        **Authors**: *Noah Hollmann, Samuel Müller, Frank Hutter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c2df4c35cdbee764ebb9e9d0acd5197-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c2df4c35cdbee764ebb9e9d0acd5197-Abstract-Conference.html)

        **Abstract**:

        As the field of automated machine learning (AutoML) advances, it becomes increasingly important to incorporate domain knowledge into these systems.We present an approach for doing so by harnessing the power of large language models (LLMs). Specifically, we introduce Context-Aware Automated Feature Engineering (CAAFE), a feature engineering method for tabular datasets that utilizes an LLM to iteratively generate additional semantically meaningful features for tabular datasets based on the description of the dataset. The method produces both Python code for creating new features and explanations for the utility of the generated features.Despite being methodologically simple, CAAFE improves performance on 11 out of 14 datasets -- boosting mean ROC AUC performance from 0.798 to 0.822 across all dataset - similar to the improvement achieved by using a random forest instead of logistic regression on our datasets. Furthermore, CAAFE is interpretable by providing a textual explanation for each generated feature.CAAFE paves the way for more extensive semi-automation in data science tasks and emphasizes the significance of context-aware solutions that can extend the scope of AutoML systems to semantic AutoML. We release our code, a simple demo and a python package.

        ----

        ## [1938] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning

        **Authors**: *Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Lifelong learning offers a promising paradigm of building a generalist agent that learns and adapts over its lifespan. Unlike traditional lifelong learning problems in image and text domains, which primarily involve the transfer of declarative knowledge of entities and concepts, lifelong learning in decision-making (LLDM) also necessitates the transfer of procedural knowledge, such as actions and behaviors. To advance research in LLDM, we introduce LIBERO, a novel benchmark of lifelong learning for robot manipulation. Specifically, LIBERO highlights five key research topics in LLDM: 1) how to efficiently transfer declarative knowledge, procedural knowledge, or the mixture of both; 2) how to design effective policy architectures and 3) effective algorithms for LLDM; 4) the robustness of a lifelong learner with respect to task ordering; and 5) the effect of model pretraining for LLDM. We develop an extendible procedural generation pipeline that can in principle generate infinitely many tasks. For benchmarking purpose, we create four task suites (130 tasks in total) that we use to investigate the above-mentioned research topics. To support sample-efficient learning, we provide high-quality human-teleoperated demonstration data for all tasks. Our extensive experiments present several insightful or even unexpected discoveries: sequential finetuning outperforms existing lifelong learning methods in forward transfer, no single visual encoder architecture excels at all types of knowledge transfer, and naive supervised pretraining can hinder agents' performance in the subsequent LLDM.

        ----

        ## [1939] On quantum backpropagation, information reuse, and cheating measurement collapse

        **Authors**: *Amira Abbas, Robbie King, Hsin-Yuan Huang, William J. Huggins, Ramis Movassagh, Dar Gilboa, Jarrod R. McClean*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c3caae2f725c8e2a55ecd600563d172-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c3caae2f725c8e2a55ecd600563d172-Abstract-Conference.html)

        **Abstract**:

        The success of modern deep learning hinges on the ability to train neural networks at scale. Through clever reuse of intermediate information, backpropagation facilitates training through gradient computation at a total cost roughly proportional to running the function, rather than incurring an additional factor proportional to the number of parameters -- which can now be in the trillions.  Naively, one expects that quantum measurement collapse entirely rules out the reuse of quantum information as in backpropagation. But recent developments in shadow tomography, which assumes access to multiple copies of a quantum state, have challenged that notion.  Here, we investigate whether parameterized quantum models can train as efficiently as classical neural networks. We show that achieving backpropagation scaling is impossible without access to multiple copies of a state.  With this added ability, we introduce an algorithm with foundations in shadow tomography that matches backpropagation scaling in quantum resources while reducing classical auxiliary computational costs to open problems in shadow tomography. These results highlight the nuance of reusing quantum information for practical purposes and clarify the unique difficulties in training large quantum models, which could alter the course of quantum machine learning.

        ----

        ## [1940] First Order Methods with Markovian Noise: from Acceleration to Variational Inequalities

        **Authors**: *Aleksandr Beznosikov, Sergey Samsonov, Marina Sheshukova, Alexander V. Gasnikov, Alexey Naumov, Eric Moulines*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c3e38ce55a0fa44bc325bc6fdb7f4e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c3e38ce55a0fa44bc325bc6fdb7f4e5-Abstract-Conference.html)

        **Abstract**:

        This paper delves into stochastic optimization problems that involve Markovian noise. We present a unified approach for the theoretical analysis of first-order gradient methods for stochastic optimization and variational inequalities. Our approach covers scenarios for both non-convex and strongly convex minimization problems. To achieve an optimal (linear) dependence on the mixing time of the underlying noise sequence, we use the randomized batching scheme, which is based on the multilevel Monte Carlo method. Moreover, our technique allows us to eliminate the limiting assumptions of previous research on Markov noise, such as the need for a bounded domain and uniformly bounded stochastic gradients. Our extension to variational inequalities under Markovian noise is original. Additionally, we provide lower bounds that match the oracle complexity of our method in the case of strongly convex optimization problems.

        ----

        ## [1941] Fair, Polylog-Approximate Low-Cost Hierarchical Clustering

        **Authors**: *Marina Knittel, Max Springer, John P. Dickerson, MohammadTaghi Hajiaghayi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c533f53f76c4df9a7f08e7cb676d132-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c533f53f76c4df9a7f08e7cb676d132-Abstract-Conference.html)

        **Abstract**:

        Research in fair machine learning, and particularly clustering, has been crucial in recent years given the many ethical controversies that modern intelligent systems have posed. Ahmadian et al. [2020] established the study of fairness in hierarchical clustering, a stronger, more structured variant of its well-known flat counterpart, though their proposed algorithm that optimizes for Dasgupta's [2016] famous cost function was highly theoretical. Knittel et al. [2023] then proposed the first practical fair approximation for cost, however they were unable to break the polynomial-approximate barrier they posed as a hurdle of interest. We break this barrier, proposing the first truly polylogarithmic-approximate low-cost fair hierarchical clustering, thus greatly bridging the gap between the best fair and vanilla hierarchical clustering approximations.

        ----

        ## [1942] A Novel Approach for Effective Multi-View Clustering with Information-Theoretic Perspective

        **Authors**: *Chenhang Cui, Yazhou Ren, Jingyu Pu, Jiawei Li, Xiaorong Pu, Tianyi Wu, Yutao Shi, Lifang He*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract-Conference.html)

        **Abstract**:

        Multi-view clustering (MVC) is a popular technique for improving clustering performance using various data sources. However, existing methods primarily focus on acquiring consistent information while often neglecting the issue of redundancy across multiple views.This study presents a new approach called Sufficient Multi-View Clustering (SUMVC) that examines the multi-view clustering framework from an information-theoretic standpoint. Our proposed method consists of two parts. Firstly, we develop a simple and reliable multi-view clustering method SCMVC (simple consistent multi-view clustering) that employs variational analysis to generate consistent information. Secondly, we propose a sufficient representation lower bound to enhance consistent information and minimise unnecessary information among views. The proposed SUMVC method offers a promising solution to the problem of multi-view clustering and provides a new perspective for analyzing multi-view data.  To verify the effectiveness of our model, we conducted a theoretical analysis based on the Bayes Error Rate, and experiments on multiple multi-view datasets demonstrate the superior performance of SUMVC.

        ----

        ## [1943] OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding

        **Authors**: *Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, Hao Su*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c7304e77c832ddc70075dfee081ca6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c7304e77c832ddc70075dfee081ca6c-Abstract-Conference.html)

        **Abstract**:

        We introduce OpenShape, a method for learning multi-modal joint representations of text, image, and point clouds. We adopt the commonly used multi-modal contrastive learning framework for representation alignment, but with a specific focus on scaling up 3D representations to enable open-world 3D shape understanding. To achieve this, we scale up training data by ensembling multiple 3D datasets and propose several strategies to automatically filter and enrich noisy text descriptions. We also explore and compare strategies for scaling 3D backbone networks and introduce a novel hard negative mining module for more efficient training. We evaluate OpenShape on zero-shot 3D classification benchmarks and demonstrate its superior capabilities for open-world recognition. Specifically, OpenShape achieves a zero-shot accuracy of 46.8% on the 1,156-category Objaverse-LVIS benchmark, compared to less than 10% for existing methods. OpenShape also achieves an accuracy of 85.3% on ModelNet40, outperforming previous zero-shot baseline methods by 20% and performing on par with some fully-supervised methods. Furthermore, we show that our learned embeddings encode a wide range of visual and semantic concepts (e.g., subcategories, color, shape, style) and facilitate fine-grained text-3D and image-3D interactions. Due to their alignment with CLIP embeddings, our learned shape representations can also be integrated with off-the-shelf CLIP-based models for various applications, such as point cloud captioning and point cloud-conditioned image generation.

        ----

        ## [1944] KuaiSim: A Comprehensive Simulator for Recommender Systems

        **Authors**: *Kesen Zhao, Shuchang Liu, Qingpeng Cai, Xiangyu Zhao, Ziru Liu, Dong Zheng, Peng Jiang, Kun Gai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c7f8f98f9a8f5650922dd4545254f28-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c7f8f98f9a8f5650922dd4545254f28-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Reinforcement Learning (RL)-based recommender systems (RSs) have garnered considerable attention due to their ability to learn optimal recommendation policies and maximize long-term user rewards. However, deploying RL models directly in online environments and generating authentic data through A/B tests can pose challenges and require substantial resources. Simulators offer an alternative approach by providing training and evaluation environments for RS models, reducing reliance on real-world data. Existing simulators have shown promising results but also have limitations such as simplified user feedback, lacking consistency with real-world data, the challenge of simulator evaluation, and difficulties in migration and expansion across RSs.To address these challenges, we propose KuaiSim, a comprehensive user environment that provides user feedback with multi-behavior and cross-session responses.The resulting simulator can support three levels of recommendation problems: the request level list-wise recommendation task, the whole-session level sequential recommendation task, and the cross-session level retention optimization task. For each task, KuaiSim also provides evaluation protocols and baseline recommendation algorithms that further serve as benchmarks for future research. We also restructure existing competitive simulators on the Kuairand Dataset and compare them against KuaiSim to future assess their performance and behavioral differences. Furthermore, to showcase KuaiSim's flexibility in accommodating different datasets, we demonstrate its versatility and robustness when deploying it on the ML-1m dataset. The implementation code is available online to ease reproducibility \footnote{https://github.com/Applied-Machine-Learning-Lab/KuaiSim}.

        ----

        ## [1945] Optimizing over trained GNNs via symmetry breaking

        **Authors**: *Shiqiang Zhang, Juan S. Campos, Christian Feldmann, David Walz, Frederik Sandfort, Miriam Mathea, Calvin Tsay, Ruth Misener*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c8cd1b78cdae751265c88efc136e5bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c8cd1b78cdae751265c88efc136e5bd-Abstract-Conference.html)

        **Abstract**:

        Optimization over trained machine learning models has applications including: verification, minimizing neural acquisition functions, and integrating a trained surrogate into a larger decision-making problem. This paper formulates and solves optimization problems constrained by trained graph neural networks (GNNs). To circumvent the symmetry issue caused by graph isomorphism, we propose two types of symmetry-breaking constraints: one indexing a node 0 and one indexing the remaining nodes by lexicographically ordering their neighbor sets. To guarantee that adding these constraints will not remove all symmetric solutions, we construct a graph indexing algorithm and prove that the resulting graph indexing satisfies the proposed symmetry-breaking constraints. For the classical GNN architectures considered in this paper, optimizing over a GNN with a fixed graph is equivalent to optimizing over a dense neural network. Thus, we study the case where the input graph is not fixed, implying that each edge is a decision variable, and develop two mixed-integer optimization formulations. To test our symmetry-breaking strategies and optimization formulations, we consider an application in molecular design.

        ----

        ## [1946] REx: Data-Free Residual Quantization Error Expansion

        **Authors**: *Edouard Yvinec, Arnaud Dapogny, Matthieu Cord, Kevin Bailly*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c96b559340daa7bb29f56ccfbbc9c2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c96b559340daa7bb29f56ccfbbc9c2f-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) are ubiquitous in computer vision and natural language processing, but suffer from high inference cost. This problem can be addressed by quantization, which consists in converting floating point operations into a lower bit-width format. With the growing concerns on privacy rights, we focus our efforts on data-free methods. However, such techniques suffer from their lack of adaptability to the target devices, as a hardware typically only supports specific bit widths. Thus, to adapt to a variety of devices, a quantization method shall be flexible enough to find good accuracy v.s. speed trade-offs for every bit width and target device. To achieve this, we propose REx, a quantization method that leverages residual error expansion, along with group sparsity. We show experimentally that REx enables better trade-offs (in terms of accuracy given any target bit-width) on both convnets and transformers for computer vision, as well as NLP models. In particular, when applied to large language models, we show that REx elegantly solves the outlier problem that hinders state-of-the-art quantization methods.In addition, REx is backed off by strong theoretical guarantees on the preservation of the predictive function of the original model. Lastly, we show that REx is agnostic to the quantization operator and can be used in combination with previous quantization work.

        ----

        ## [1947] A Unified, Scalable Framework for Neural Population Decoding

        **Authors**: *Mehdi Azabou, Vinam Arora, Venkataramana Ganesh, Ximeng Mao, Santosh Nachimuthu, Michael Mendelson, Blake A. Richards, Matthew G. Perich, Guillaume Lajoie, Eva L. Dyer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html)

        **Abstract**:

        Our ability to use deep learning approaches to decipher neural activity would likely benefit from greater scale, in terms of both the model size and the datasets. However, the integration of many neural recordings into one unified model is challenging, as each recording contains the activity of different neurons from different individual animals. In this paper, we introduce a training framework and architecture designed to model the population dynamics of neural activity across diverse, large-scale neural recordings. Our method first tokenizes individual spikes within the dataset to build an efficient representation of neural events that captures the fine temporal structure of neural activity. We then employ cross-attention and a PerceiverIO backbone to further construct a latent tokenization of neural population activities. Utilizing this architecture and training framework, we construct a large-scale multi-session model trained on large datasets from seven nonhuman primates, spanning over 158 different sessions of recording from over 27,373 neural units and over 100 hours of recordings. In a number of different tasks, we demonstrate that our pretrained model can be rapidly adapted to new, unseen sessions with unspecified neuron correspondence, enabling few-shot performance with minimal labels. This work presents a powerful new approach for building deep learning tools to analyze neural data and stakes out a clear path to training at scale for neural decoding models.

        ----

        ## [1948] Species196: A One-Million Semi-supervised Dataset for Fine-grained Species Recognition

        **Authors**: *Wei He, Kai Han, Ying Nie, Chengcheng Wang, Yunhe Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8cb92f326d01fd7f4371283ee2fa6386-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8cb92f326d01fd7f4371283ee2fa6386-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The development of foundation vision models has pushed the general visual recognition to a high level, but cannot well address the fine-grained recognition in specialized domain such as invasive species classification. Identifying and managing invasive species has strong social and ecological value. Currently, most invasive species datasets are limited in scale and cover a narrow range of species, which restricts the development of deep-learning based invasion biometrics systems. To fill the gap of this area, we introduced Species196, a large-scale semi-supervised dataset of 196-category invasive species. It collects over 19K images with expert-level accurate annotations (Species196-L), and 1.2M unlabeled images of invasive species (Species196-U). The dataset provides four experimental settings for benchmarking the existing models and algorithms, namely, supervised learning, semi-supervised learning and self-supervised pretraining. To facilitate future research on these four learning paradigms, we conduct an empirical study of the representative methods on the introduced dataset. The dataset will be made publicly available at https://species-dataset.github.io/.

        ----

        ## [1949] PTADisc: A Cross-Course Dataset Supporting Personalized Learning in Cold-Start Scenarios

        **Authors**: *Liya Hu, Zhiang Dong, Jingyuan Chen, Guifeng Wang, Zhihua Wang, Zhou Zhao, Fei Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8cf04c64d1734e5f7e63418a2a4d49de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8cf04c64d1734e5f7e63418a2a4d49de-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The focus of our work is on diagnostic tasks in personalized learning, such as cognitive diagnosis and knowledge tracing. The goal of these tasks is to assess students' latent proficiency on knowledge concepts through analyzing their historical learning records. However, existing research has been limited to single-course scenarios; cross-course studies have not been explored due to a lack of dataset. We address this issue by constructing PTADisc, a Diverse, Immense, Student-centered dataset that emphasizes its sufficient Cross-course information for personalized learning. PTADisc includes 74 courses, 1,530,100 students, 4,054 concepts, 225,615 problems, and over 680 million student response logs. Based on PTADisc, we developed a model-agnostic Cross-Course Learner Modeling Framework (CCLMF) which utilizes relationships between students' proficiency across courses to alleviate the difficulty of diagnosing student knowledge state in cold-start scenarios. CCLMF uses a meta network to generate personalized mapping functions between courses. The experimental results on PTADisc verify the effectiveness of CCLMF with an average improvement of 4.2% on AUC. We also report the performance of baseline models for cognitive diagnosis and knowledge tracing over PTADisc, demonstrating that our dataset supports a wide scope of research in personalized learning. Additionally, PTADisc contains valuable programming logs and student-group information that are worth exploring in the future.

        ----

        ## [1950] Adaptive Contextual Perception: How To Generalize To New Backgrounds and Ambiguous Objects

        **Authors**: *Zhuofan Ying, Peter Hase, Mohit Bansal*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8d2c36836fb0e7d78fe68762ff8b5f1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8d2c36836fb0e7d78fe68762ff8b5f1e-Abstract-Conference.html)

        **Abstract**:

        Biological vision systems make adaptive use of context to recognize objects in new settings with novel contexts as well as occluded or blurry objects in familiar settings. In this paper, we investigate how vision models adaptively use context for out-of-distribution (OOD) generalization and leverage our analysis results to improve model OOD generalization. First, we formulate two distinct OOD settings where the contexts are either beneficial Object-Disambiguation or irrelevant Background-Invariance, reflecting the diverse contextual challenges faced in biological vision. We then analyze model performance in these two different OOD settings and demonstrate that models that excel in one setting tend to struggle in the other. Notably, prior works on learning causal features improve on one setting but hurt on the other. This underscores the importance of generalizing across both OOD settings, as this ability is crucial for both human cognition and robust AI systems. Next, to better understand the model properties contributing to OOD generalization, we use representational geometry analysis and our own probing methods to examine a population of models, and we discover that those with more factorized representations and appropriate feature weighting are more successful in handling Object-Disambiguation and Background-Invariance tests. We further validate these findings through causal intervention, manipulating representation factorization and feature weighting to demonstrate their causal effect on performance. Motivated by our analysis results, we propose new augmentation methods aimed at enhancing model generalization. The proposed methods outperform strong baselines, yielding improvements in both in-distribution and OOD tests. We conclude that, in order to replicate the generalization abilities of biological vision, computer vision models must have factorized object vs. background representations and appropriately weigh both kinds of features.

        ----

        ## [1951] PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning

        **Authors**: *Florian Bordes, Shashank Shekhar, Mark Ibrahim, Diane Bouchacourt, Pascal Vincent, Ari Morcos*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8d352fd0f07fde4a74f9476603b3773b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8d352fd0f07fde4a74f9476603b3773b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Synthetic image datasets offer unmatched advantages for designing and evaluating deep neural networks: they make it possible to (i) render as many data samples as needed, (ii) precisely control each scene and yield granular ground truth labels (and captions), (iii) precisely control distribution shifts between training and testing to isolate variables of interest for sound experimentation.Despite such promise, the use of synthetic image data is still limited -- and often played down -- mainly due to their lack of realism. Most works therefore rely on datasets of real images, which have often been scraped from public images on the internet, and may have issues with regards to privacy, bias, and copyright, while offering little control over how objects precisely appear.In this work, we present a path to democratize the use of photorealistic synthetic data: we develop a new generation of interactive environments for representation learning research, that offer both controllability and realism. We use the Unreal Engine, a powerful game engine well known in the entertainment industry, to produce PUG (Photorealistic Unreal Graphics) environments and datasets for representation learning. Using PUG for evaluation and fine-tuning, we demonstrate the potential of PUG to both enable more rigorous evaluations and to improve model training.

        ----

        ## [1952] On the Gini-impurity Preservation For Privacy Random Forests

        **Authors**: *Xinran Xie, Man-Jie Yuan, Xuetong Bai, Wei Gao, Zhi-Hua Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8d6b1d775014eff18256abeb207202ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8d6b1d775014eff18256abeb207202ad-Abstract-Conference.html)

        **Abstract**:

        Random forests have been one successful ensemble algorithms in machine learning. Various techniques have been utilized to preserve the privacy of random forests from anonymization, differential privacy, homomorphic encryption, etc., whereas it rarely takes into account some crucial ingredients of learning algorithm. This work presents a new encryption to preserve data's Gini impurity, which plays a crucial role during the construction of random forests. Our basic idea is to modify the structure of binary search tree to store several examples in each node,  and encrypt data features by incorporating label and order information. Theoretically, we prove that our scheme preserves the minimum Gini impurity in ciphertexts without decrypting, and present the security guarantee for encryption. For random forests, we encrypt data features based on our Gini-impurity-preserving scheme, and take the homomorphic encryption scheme CKKS to encrypt data labels  due to their importance and privacy. We  conduct extensive experiments to show the effectiveness, efficiency and security of our proposed method.

        ----

        ## [1953] Debiasing Pretrained Generative Models by Uniformly Sampling Semantic Attributes

        **Authors**: *Walter Gerych, Kevin Hickey, Luke Buquicchio, Kavin Chandrasekaran, Abdulaziz Alajaji, Elke A. Rundensteiner, Emmanuel Agu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8d7060b2ee6ff728692398783e3d59d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8d7060b2ee6ff728692398783e3d59d1-Abstract-Conference.html)

        **Abstract**:

        Generative models are being increasingly used in science and industry applications. Unfortunately,  they often perpetuate the biases present in their training sets, such as societal biases causing certain groups to be underrepresented in the data. For instance, image generators may overwhelmingly produce images of white people due to few non-white samples in their training data. It is imperative to debias generative models so they synthesize an equal number of instances for each group, while not requiring retraining of the model to avoid  prohibitive expense. We thus propose a distribution mapping module that produces samples from a fair noise distribution, such that the  pretrained generative model produces semantically uniform outputs -  an equal number of instances for each group - when conditioned on these samples. This does not involve retraining the generator, nor does it require any real training data. Experiments on debiasing generators trained on popular real-world  datasets show that our method outperforms existing approaches.

        ----

        ## [1954] Improved Algorithms for Stochastic Linear Bandits Using Tail Bounds for Martingale Mixtures

        **Authors**: *Hamish Flynn, David Reeb, Melih Kandemir, Jan R. Peters*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8db0d67d22e0ec08c95b810be3a66907-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8db0d67d22e0ec08c95b810be3a66907-Abstract-Conference.html)

        **Abstract**:

        We present improved algorithms with worst-case regret guarantees for the stochastic linear bandit problem. The widely used "optimism in the face of uncertainty" principle reduces a stochastic bandit problem to the construction of a confidence sequence for the unknown reward function. The performance of the resulting bandit algorithm depends on the size of the confidence sequence, with smaller confidence sets yielding better empirical performance and stronger regret guarantees. In this work, we use a novel tail bound for adaptive martingale mixtures to construct confidence sequences which are suitable for stochastic bandits. These confidence sequences allow for efficient action selection via convex programming. We prove that a linear bandit algorithm based on our confidence sequences is guaranteed to achieve competitive worst-case regret. We show that our confidence sequences are tighter than competitors, both empirically and theoretically. Finally, we demonstrate that our tighter confidence sequences give improved performance in several hyperparameter tuning tasks.

        ----

        ## [1955] Tame a Wild Camera: In-the-Wild Monocular Camera Calibration

        **Authors**: *Shengjie Zhu, Abhinav Kumar, Masa Hu, Xiaoming Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8db9279f593652ee9bb2223b4a2c43fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8db9279f593652ee9bb2223b4a2c43fa-Abstract-Conference.html)

        **Abstract**:

        3D sensing for monocular in-the-wild images, e.g., depth estimation and 3D object detection, has become increasingly important.However, the unknown intrinsic parameter hinders their development and deployment.Previous methods for the monocular camera calibration rely on specific 3D objects or strong geometry prior, such as using a checkerboard or imposing a Manhattan World assumption.This work instead calibrates intrinsic via exploiting the monocular 3D prior.Given an undistorted image as input, our method calibrates the complete 4 Degree-of-Freedom (DoF) intrinsic parameters.First, we show intrinsic is determined by the two well-studied monocular priors: monocular depthmap and surface normal map.However, this solution necessitates a low-bias and low-variance depth estimation.Alternatively, we introduce the incidence field, defined as the incidence rays between points in 3D space and pixels in the 2D imaging plane.We show that: 1) The incidence field is a pixel-wise parametrization of the intrinsic invariant to image cropping and resizing.2) The incidence field is a learnable monocular 3D prior, determined pixel-wisely by up-to-sacle monocular depthmap and surface normal.With the estimated incidence field, a robust RANSAC algorithm recovers intrinsic.We show the effectiveness of our method through superior performance on synthetic and zero-shot testing datasets.Beyond calibration, we demonstrate downstream applications in image manipulation detection \& restoration, uncalibrated two-view pose estimation, and 3D sensing.

        ----

        ## [1956] ATTA: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation

        **Authors**: *Zhitong Gao, Shipeng Yan, Xuming He*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8dcc306a2522c60a78f047ab8739e631-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8dcc306a2522c60a78f047ab8739e631-Abstract-Conference.html)

        **Abstract**:

        Recent advancements in dense out-of-distribution (OOD) detection have primarily focused on scenarios where the training and testing datasets share a similar domain, with the assumption that no domain shift exists between them. However, in real-world situations, domain shift often exits and significantly affects the accuracy of existing out-of-distribution (OOD) detection models. In this work, we propose a dual-level OOD detection framework to handle domain shift and semantic shift jointly. The first level distinguishes whether domain shift exists in the image by leveraging global low-level features, while the second level identifies pixels with semantic shift by utilizing dense high-level feature maps. In this way, we can selectively adapt the model to unseen domains as well as enhance model's capacity in detecting novel classes. We validate the efficacy of our proposed method on several OOD segmentation benchmarks, including those with significant domain shifts and those without, observing consistent performance improvements across various baseline models. Code is available at https://github.com/gaozhitong/ATTA.

        ----

        ## [1957] Regression with Cost-based Rejection

        **Authors**: *Xin Cheng, Yuzhou Cao, Haobo Wang, Hongxin Wei, Bo An, Lei Feng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ddcba644f602835a52b962d9a119eea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ddcba644f602835a52b962d9a119eea-Abstract-Conference.html)

        **Abstract**:

        Learning with rejection is an important framework that can refrain from making predictions to avoid critical mispredictions by balancing between prediction and rejection. Previous studies on cost-based rejection only focused on the classification setting, which cannot handle the continuous and infinite target space in the regression setting. In this paper, we investigate a novel regression problem called regression with cost-based rejection, where the model can reject to make predictions on some examples given certain rejection costs. To solve this problem, we first formulate the expected risk for this problem and then derive the Bayes optimal solution, which shows that the optimal model should reject to make predictions on the examples whose variance is larger than the rejection cost when the mean squared error is used as the evaluation metric. Furthermore, we propose to train the model by a surrogate loss function that considers rejection as binary classification and we provide conditions for the model consistency, which implies that the Bayes optimal solution can be recovered by our proposed surrogate loss. Extensive experiments demonstrate the effectiveness of our proposed method.

        ----

        ## [1958] A State Representation for Diminishing Rewards

        **Authors**: *Ted Moskovitz, Samo Hromadka, Ahmed Touati, Diana Borsa, Maneesh Sahani*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8df0fe2bba0f14208a10c1cb22e71552-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8df0fe2bba0f14208a10c1cb22e71552-Abstract-Conference.html)

        **Abstract**:

        A common setting in multitask reinforcement learning (RL) demands that an agent rapidly adapt to various stationary reward functions randomly sampled from a fixed distribution. In such situations, the successor representation (SR) is a popular framework which supports rapid policy evaluation by decoupling a policy's expected discounted, cumulative state occupancies from a specific reward function. However, in the natural world, sequential tasks are rarely independent, and instead reflect shifting priorities based on the availability and subjective perception of rewarding stimuli. Reflecting this disjunction, in this paper we study the phenomenon of diminishing marginal utility and introduce a novel state representation, the $\lambda$ representation ($\lambda$R) which, surprisingly, is required for policy evaluation in this setting and which generalizes the SR as well as several other state representations from the literature. We establish the $\lambda$R's formal properties and examine its normative advantages in the context of machine learning, as well as its usefulness for studying natural behaviors, particularly foraging.

        ----

        ## [1959] Unified Segment-to-Segment Framework for Simultaneous Sequence Generation

        **Authors**: *Shaolei Zhang, Yang Feng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8df705957a5262de3cb37ba9f1fb96f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8df705957a5262de3cb37ba9f1fb96f3-Abstract-Conference.html)

        **Abstract**:

        Simultaneous sequence generation is a pivotal task for real-time scenarios, such as streaming speech recognition, simultaneous machine translation and simultaneous speech translation, where the target sequence is generated while receiving the source sequence. The crux of achieving high-quality generation with low latency lies in identifying the optimal moments for generating, accomplished by learning a mapping between the source and target sequences. However, existing methods often rely on task-specific heuristics for different sequence types, limiting the modelâ€™s capacity to adaptively learn the source-target mapping and hindering the exploration of multi-task learning for various simultaneous tasks. In this paper, we propose a unified segment-to-segment framework (Seg2Seg) for simultaneous sequence generation, which learns the mapping in an adaptive and unified manner. During the process of simultaneous generation, the model alternates between waiting for a source segment and generating a target segment, making the segment serve as the natural bridge between the source and target. To accomplish this, Seg2Seg introduces a latent segment as the pivot between source to target and explores all potential source-target mappings via the proposed expectation training, thereby learning the optimal moments for generating. Experiments on multiple simultaneous generation tasks demonstrate that Seg2Seg achieves state-of-the-art performance and exhibits better generality across various tasks.

        ----

        ## [1960] DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting

        **Authors**: *Salva Rühling Cachay, Bo Zhao, Hailey Joren, Rose Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8df90a1440ce782d1f5607b7a38f2531-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8df90a1440ce782d1f5607b7a38f2531-Abstract-Conference.html)

        **Abstract**:

        While diffusion models can successfully generate data and make predictions, they are predominantly designed for static images. We propose an approach for training diffusion models for dynamics forecasting that leverages the temporal dynamics encoded in the data, directly coupling it with the diffusion steps in the network. We train a stochastic, time-conditioned interpolator and a backbone forecaster networkthat mimic the forward and reverse processes of conventional diffusion models, respectively. This design choice naturally encodes multi-step and long-range forecasting capabilities, allowing for highly flexible, continuous-time sampling trajectories and the ability to trade-off performance with accelerated sampling at inference time. In addition, the dynamics-informed diffusion process imposes a strong inductive bias, allowing for improved computational efficiency compared to traditional Gaussian noise-based diffusion models. Our approach performs competitively on probabilistic skill score metrics in complex dynamics forecasting of sea surface temperatures, Navier-Stokes flows, and spring mesh systems.

        ----

        ## [1961] Towards a Comprehensive Benchmark for High-Level Synthesis Targeted to FPGAs

        **Authors**: *Yunsheng Bai, Atefeh Sohrabizadeh, Zongyue Qin, Ziniu Hu, Yizhou Sun, Jason Cong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8dfc3a2720a4112243a285b98e0d4415-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8dfc3a2720a4112243a285b98e0d4415-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        High-level synthesis (HLS) aims to raise the abstraction layer in hardware design, enabling the design of domain-specific accelerators (DSAs) like field-programmable gate arrays (FPGAs) using C/C++ instead of hardware description languages (HDLs). Compiler directives in the form of pragmas play a crucial role in modifying the microarchitecture within the HLS framework. However, the space of possible microarchitectures grows exponentially with the number of pragmas. Moreover, the evaluation of each candidate design using the HLS tool consumes significant time, ranging from minutes to hours, leading to a time-consuming optimization process. To accelerate this process, machine learning models have been used to predict design quality in milliseconds. However, existing open-source datasets for training such models are limited in terms of design complexity and available optimizations. In this paper, we present HLSyn, the first benchmark that addresses these limitations. It contains more complex programs with a wider range of optimization pragmas, making it a comprehensive dataset for training and evaluating design quality prediction models. The HLSyn benchmark consists of 42 unique programs/kernels, resulting in over 42,000 labeled designs. We conduct an extensive comparison of state-of-the-art baselines to assess their effectiveness in predicting design quality. As an ongoing project, we anticipate expanding the HLSyn benchmark in terms of both quantity and variety of programs to further support the development of this field.

        ----

        ## [1962] Energy Discrepancies: A Score-Independent Loss for Energy-Based Models

        **Authors**: *Tobias Schröder, Zijing Ou, Jen Lim, Yingzhen Li, Sebastian J. Vollmer, Andrew Duncan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e176ef071f00f1b233461c5ad5e1b24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e176ef071f00f1b233461c5ad5e1b24-Abstract-Conference.html)

        **Abstract**:

        Energy-based models are a simple yet powerful class of probabilistic models, but their widespread adoption has been limited by the computational burden of training them. We propose a novel loss function called Energy Discrepancy (ED) which does not rely on the computation of scores or expensive Markov chain Monte Carlo. We show that energy discrepancy approaches the explicit score matching and negative log-likelihood loss under different limits, effectively interpolating between both. Consequently, minimum energy discrepancy estimation overcomes the problem of nearsightedness encountered in score-based estimation methods, while also enjoying theoretical guarantees. Through numerical experiments, we demonstrate that ED learns low-dimensional data distributions faster and more accurately than explicit score matching or contrastive divergence. For high-dimensional image data, we describe how the manifold hypothesis puts limitations on our approach and demonstrate the effectiveness of energy discrepancy by training the energy-based model as a prior of a variational decoder model.

        ----

        ## [1963] Learning to Group Auxiliary Datasets for Molecule

        **Authors**: *Tinglin Huang, Ziniu Hu, Rex Ying*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e2571d13f432b301d4c5e3cc70227a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e2571d13f432b301d4c5e3cc70227a6-Abstract-Conference.html)

        **Abstract**:

        The limited availability of annotations in small molecule datasets presents a challenge to machine learning models. To address this, one common strategy is to collaborate with additional auxiliary datasets. However, having more data does not always guarantee improvements. Negative transfer can occur when the knowledge in the target dataset differs or contradicts that of the auxiliary molecule datasets. In light of this, identifying the auxiliary molecule datasets that can benefit the target dataset when jointly trained remains a critical and unresolved problem. Through an empirical analysis, we observe that combining graph structure similarity and task similarity can serve as a more reliable indicator for identifying high-affinity auxiliary datasets. Motivated by this insight, we propose MolGroup, which separates the dataset affinity into task and structure affinity to predict the potential benefits of each auxiliary molecule dataset. MolGroup achieves this by utilizing a routing mechanism optimized through a bi-level optimization framework. Empowered by the meta gradient, the routing mechanism is optimized toward maximizing the target dataset's performance and quantifies the affinity as the gating score. As a result, MolGroup is capable of predicting the optimal combination of auxiliary datasets for each target dataset. Our extensive experiments demonstrate the efficiency and effectiveness of MolGroup, showing an average improvement of 4.41%/3.47% for GIN/Graphormer trained with the group of molecule datasets selected by MolGroup on 11 target molecule datasets.

        ----

        ## [1964] Equivariant Spatio-Temporal Attentive Graph Networks to Simulate Physical Dynamics

        **Authors**: *Liming Wu, Zhichao Hou, Jirui Yuan, Yu Rong, Wenbing Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e2a75e0c7b579a6cf176dc0858cde55-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e2a75e0c7b579a6cf176dc0858cde55-Abstract-Conference.html)

        **Abstract**:

        Learning to represent and simulate the dynamics of physical systems is a crucial yet challenging task. Existing equivariant Graph Neural Network (GNN) based methods have encapsulated the symmetry of physics, \emph{e.g.}, translations, rotations, etc, leading to better generalization ability. Nevertheless, their frame-to-frame formulation of the task overlooks the non-Markov property mainly incurred by unobserved dynamics in the environment. In this paper, we reformulate dynamics simulation as a spatio-temporal prediction task, by employing the trajectory in the past period to recover the Non-Markovian interactions. We propose Equivariant Spatio-Temporal Attentive Graph Networks (ESTAG), an equivariant version of spatio-temporal GNNs, to fulfil our purpose. At its core, we design a novel Equivariant Discrete Fourier Transform (EDFT) to extract periodic patterns from the history frames, and then construct an Equivariant Spatial Module (ESM) to accomplish spatial message passing, and an Equivariant Temporal Module (ETM) with the forward attention and equivariant pooling mechanisms to aggregate temporal message. We evaluate our model on three real datasets corresponding to the molecular-, protein- and macro-level. Experimental results verify the effectiveness of ESTAG compared to typical spatio-temporal GNNs and equivariant GNNs.

        ----

        ## [1965] Differentially Private Decoupled Graph Convolutions for Multigranular Topology Protection

        **Authors**: *Eli Chien, Wei-Ning Chen, Chao Pan, Pan Li, Ayfer Özgür, Olgica Milenkovic*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e3db2040672d85fd12e6313945594fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e3db2040672d85fd12e6313945594fe-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) have proven to be highly effective in solving real-world learning problems that involve graph-structured data. However, GNNs can also inadvertently expose sensitive user information and interactions through their model predictions. To address these privacy concerns, Differential Privacy (DP) protocols are employed to control the trade-off between provable privacy protection and model utility. Applying standard DP approaches to GNNs directly is not advisable due to two main reasons. First, the prediction of node labels, which relies on neighboring node attributes through graph convolutions, can lead to privacy leakage. Second, in practical applications, the privacy requirements for node attributes and graph topology may differ. In the latter setting, existing DP-GNN models fail to provide multigranular trade-offs between graph topology privacy, node attribute privacy, and GNN utility. To address both limitations, we propose a new framework termed Graph Differential Privacy (GDP), specifically tailored to graph learning. GDP ensures both provably private model parameters as well as private predictions. Additionally, we describe a novel unified notion of graph dataset adjacency to analyze the properties of GDP for different levels of graph topology privacy. Our findings reveal that DP-GNNs, which rely on graph convolutions, not only fail to meet the requirements for multigranular graph topology privacy but also necessitate the injection of DP noise that scales at least linearly with the maximum node degree. In contrast, our proposed Differentially Private Decoupled Graph Convolutions (DPDGCs) represent a more flexible and efficient alternative to graph convolutions that still provides the necessary guarantees of GDP. To validate our approach, we conducted extensive experiments on seven node classification benchmarking and illustrative synthetic datasets. The results demonstrate that DPDGCs significantly outperform existing DP-GNNs in terms of privacy-utility trade-offs.

        ----

        ## [1966] Team-PSRO for Learning Approximate TMECor in Large Team Games via Cooperative Reinforcement Learning

        **Authors**: *Stephen McAleer, Gabriele Farina, Gaoyue Zhou, Mingzhi Wang, Yaodong Yang, Tuomas Sandholm*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e4ccc9ca6ae2225c4cbb7782ab48daf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e4ccc9ca6ae2225c4cbb7782ab48daf-Abstract-Conference.html)

        **Abstract**:

        Recent algorithms have achieved superhuman performance at a number of two-player zero-sum games such as poker and go. However, many real-world situations are multi-player games. Zero-sum two-team games, such as bridge and football, involve two teams where each member of the team shares the same reward with every other member of that team, and each team has the negative of the reward of the other team. A popular solution concept in this setting, called TMECor, assumes that teams can jointly correlate their strategies before play, but are not able to communicate during play. This setting is harder than two-player zero-sum games because each player on a team has different information and must use their public actions to signal to other members of the team. Prior works either have game-theoretic guarantees but only work in very small games, or are able to scale to large games but do not have game-theoretic guarantees. In this paper we introduce two algorithms: Team-PSRO, an extension of PSRO from two-player games to team games, and Team-PSRO Mix-and-Match which improves upon Team PSRO by better using population policies. In Team-PSRO, in every iteration both teams learn a joint best response to the opponent's meta-strategy via reinforcement learning. As the reinforcement learning joint best response approaches the optimal best response, Team-PSRO is guaranteed to converge to a TMECor. In experiments on Kuhn poker and Liar's Dice, we show that a tabular version of Team-PSRO converges to TMECor, and a version of Team PSRO using deep cooperative reinforcement learning beats self-play reinforcement learning in the large game of Google Research Football.

        ----

        ## [1967] Learning Linear Causal Representations from Interventions under General Nonlinear Mixing

        **Authors**: *Simon Buchholz, Goutham Rajendran, Elan Rosenfeld, Bryon Aragam, Bernhard Schölkopf, Pradeep Ravikumar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e5de4cb639ef718f44060dc257cb04f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e5de4cb639ef718f44060dc257cb04f-Abstract-Conference.html)

        **Abstract**:

        We study the problem of learning causal representations from unknown, latent interventions in a general setting, where the latent distribution is Gaussian but the mixing function is completely general. We prove strong identifiability results given unknown single-node interventions, i.e., without having access to the intervention targets. This generalizes prior works which have focused on weaker classes, such as linear maps or paired counterfactual data. This is also the first instance of identifiability from non-paired interventions for deep neural network embeddings and general causal structures. Our proof relies on carefully uncovering the high-dimensional geometric structure present in the data distribution after a non-linear density transformation, which we capture by analyzing quadratic forms of precision matrices of the latent distributions. Finally, we propose a contrastive algorithm to identify the latent variables in practice and evaluate its performance on various tasks.

        ----

        ## [1968] Disentanglement via Latent Quantization

        **Authors**: *Kyle Hsu, William Dorrell, James C. R. Whittington, Jiajun Wu, Chelsea Finn*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html)

        **Abstract**:

        In disentangled representation learning, a model is asked to tease apart a dataset's underlying sources of variation and represent them independently of one another. Since the model is provided with no ground truth information about these sources, inductive biases take a paramount role in enabling disentanglement. In this work, we construct an inductive bias towards encoding to and decoding from an organized latent space. Concretely, we do this by (i) quantizing the latent space into discrete code vectors with a separate learnable scalar codebook per dimension and (ii) applying strong model regularization via an unusually high weight decay. Intuitively, the latent space design forces the encoder to combinatorially construct codes from a small number of distinct scalar values, which in turn enables the decoder to assign a consistent meaning to each value. Regularization then serves to drive the model towards this parsimonious strategy. We demonstrate the broad applicability of this approach by adding it to both basic data-reconstructing (vanilla autoencoder) and latent-reconstructing (InfoGAN) generative models. For reliable evaluation, we also propose InfoMEC, a new set of metrics for disentanglement that is cohesively grounded in information theory and fixes well-established shortcomings in previous metrics. Together with regularization, latent quantization dramatically improves the modularity and explicitness of learned representations on a representative suite of benchmark datasets. In particular, our quantized-latent autoencoder (QLAE) consistently outperforms strong methods from prior work in these key disentanglement properties without compromising data reconstruction.

        ----

        ## [1969] Variance-Reduced Gradient Estimation via Noise-Reuse in Online Evolution Strategies

        **Authors**: *Oscar Li, James Harrison, Jascha Sohl-Dickstein, Virginia Smith, Luke Metz*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e69a97cbdd91ac0808603fa589d6c17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e69a97cbdd91ac0808603fa589d6c17-Abstract-Conference.html)

        **Abstract**:

        Unrolled computation graphs are prevalent throughout machine learning but present challenges to automatic differentiation (AD) gradient estimation methods when their loss functions exhibit extreme local sensitivtiy, discontinuity, or blackbox characteristics. In such scenarios, online evolution strategies methods are a more capable alternative, while being more parallelizable than vanilla evolution strategies (ES) by interleaving partial unrolls and gradient updates. In this work, we propose a general class of unbiased online evolution strategies methods. We analytically and empirically characterize the variance of this class of gradient estimators and identify the one with the least variance, which we term Noise-Reuse Evolution Strategies (NRES). Experimentally, we show NRES results in faster convergence than existing AD and ES methods in terms of wall-clock time and number of unroll steps across a variety of applications, including learning dynamical systems, meta-training learned optimizers, and reinforcement learning.

        ----

        ## [1970] Beyond Black-Box Advice: Learning-Augmented Algorithms for MDPs with Q-Value Predictions

        **Authors**: *Tongxin Li, Yiheng Lin, Shaolei Ren, Adam Wierman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e806d3c56ed5f1dab85d601e13cbe38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e806d3c56ed5f1dab85d601e13cbe38-Abstract-Conference.html)

        **Abstract**:

        We study the tradeoff between consistency and robustness in the context of a single-trajectory time-varying Markov Decision Process (MDP) with untrusted machine-learned advice. Our work departs from the typical approach of treating advice as coming from black-box sources by instead considering a setting where additional information about  how the advice is generated is available. We prove a first-of-its-kind consistency and robustness tradeoff given Q-value advice under a general MDP model that includes both continuous and discrete state/action spaces. Our results highlight that utilizing Q-value advice enables dynamic pursuit of the better of machine-learned advice and a robust baseline, thus result in near-optimal performance guarantees, which provably improves what can be obtained solely with black-box advice.

        ----

        ## [1971] Graph Contrastive Learning with Stable and Scalable Spectral Encoding

        **Authors**: *Deyu Bo, Yuan Fang, Yang Liu, Chuan Shi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e9a6582caa59fda0302349702965171-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e9a6582caa59fda0302349702965171-Abstract-Conference.html)

        **Abstract**:

        Graph contrastive learning (GCL) aims to learn representations by capturing the agreements between different graph views. Traditional GCL methods generate views in the spatial domain, but it has been recently discovered that the spectral domain also plays a vital role in complementing spatial views. However, existing spectral-based graph views either ignore the eigenvectors that encode valuable positional information or suffer from high complexity when trying to address the instability of spectral features. To tackle these challenges, we first design an informative, stable, and scalable spectral encoder, termed EigenMLP, to learn effective representations from the spectral features. Theoretically, EigenMLP is invariant to the rotation and reflection transformations on eigenvectors and robust against perturbations. Then, we propose a spatial-spectral contrastive framework (Sp$^{2}$GCL) to capture the consistency between the spatial information encoded by graph neural networks and the spectral information learned by EigenMLP, thus effectively fusing these two graph views. Experiments on the node- and graph-level datasets show that our method not only learns effective graph representations but also achieves a 2--10x speedup over other spectral-based methods.

        ----

        ## [1972] A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence

        **Authors**: *Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, Ming-Hsuan Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e9bdc23f169a05ea9b72ccef4574551-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e9bdc23f169a05ea9b72ccef4574551-Abstract-Conference.html)

        **Abstract**:

        Text-to-image diffusion models have made significant advances in generating and editing high-quality images.  As a result, numerous approaches have explored the ability of diffusion model features to understand and process single images for downstream tasks, e.g., classification, semantic segmentation, and stylization. However, significantly less is known about what these features reveal across multiple, different images and objects.  In this work, we exploit Stable Diffusion (SD) features for semantic and dense correspondence and discover that with simple post-processing, SD features can perform quantitatively similar to SOTA representations. Interestingly, the qualitative analysis reveals that SD features have very different properties compared to existing representation learning features, such as the recently released DINOv2: while DINOv2 provides sparse but accurate matches, SD features provide high-quality spatial information but sometimes inaccurate semantic matches. We demonstrate that a simple fusion of these two features works surprisingly well, and a zero-shot evaluation using nearest neighbors on these fused features provides a significant performance gain over state-of-the-art methods on benchmark datasets, e.g., SPair-71k, PF-Pascal, and TSS.  We also show that these correspondences can enable interesting applications such as instance swapping in two images. Project page: https://sd-complements-dino.github.io/.

        ----

        ## [1973] SatLM: Satisfiability-Aided Language Models Using Declarative Prompting

        **Authors**: *Xi Ye, Qiaochu Chen, Isil Dillig, Greg Durrett*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e9c7d4a48bdac81a58f983a64aaf42b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e9c7d4a48bdac81a58f983a64aaf42b-Abstract-Conference.html)

        **Abstract**:

        Prior work has combined chain-of-thought prompting in large language models (LLMs) with programmatic representations to perform effective and transparent reasoning. While such an approach works well for tasks that only require forward reasoning (e.g., straightforward arithmetic), it is less effective for constraint solving problems that require more sophisticated planning and search. In this paper, we propose a new satisfiability-aided language modeling (SatLM) approach for improving the reasoning capabilities of LLMs. We use an LLM to generate a declarative task specification rather than an imperative program and leverage an off-the-shelf automated theorem prover to derive the final answer. This approach has two key advantages. The declarative specification is closer to the problem description than the reasoning steps are, so the LLM can parse it out of the description more accurately. Furthermore, by offloading the actual reasoning task to an automated theorem prover, our approach can guarantee the correctness of the answer with respect to the parsed specification and avoid planning errors in the solving process. We evaluate SATLM on 8 different datasets and show that it consistently outperforms program-aided LMs in the imperative paradigm. In particular, SATLM outperforms program-aided LMs by 23% on a challenging subset of the GSM arithmetic reasoning dataset; SATLM also achieves a new SoTA on LSAT and BoardgameQA, surpassing previous models that are trained on the respective training sets.

        ----

        ## [1974] A normative theory of social conflict

        **Authors**: *Sergey Shuvaev, Evgeny Amelchenko, Dmitry A. Smagin, Natalia N. Kudryavtseva, Grigori Enikolopov, Alexei A. Koulakov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ec61d4084443d29c9e47ac60f9aea31-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ec61d4084443d29c9e47ac60f9aea31-Abstract-Conference.html)

        **Abstract**:

        Social conflict is a survival mechanism yielding both normal and pathological behaviors. To understand its underlying principles, we collected behavioral and whole-brain neural data from mice advancing through stages of social conflict. We modeled the animals’ interactions as a normal-form game using Bayesian inference to account for the partial observability of animals’ strengths. We find that our behavioral and neural data are consistent with the first-level Theory of Mind (1-ToM) model where mice form “primary” beliefs about the strengths of all mice involved and “secondary” beliefs that estimate the beliefs of their opponents. Our model identifies the brain regions that carry the information about these beliefs and offers a framework for studies of social behaviors in partially observable settings.

        ----

        ## [1975] Learning Invariant Representations of Graph Neural Networks via Cluster Generalization

        **Authors**: *Donglin Xia, Xiao Wang, Nian Liu, Chuan Shi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ed2293e714b7692b63117e330e551e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ed2293e714b7692b63117e330e551e8-Abstract-Conference.html)

        **Abstract**:

        Graph neural networks (GNNs) have become increasingly popular in modeling graph-structured data due to their ability to learn node representations by aggregating local structure information. However, it is widely acknowledged that the test graph structure may differ from the training graph structure, resulting in a structure shift. In this paper, we experimentally  find that the performance of GNNs drops significantly when the structure shift happens, suggesting that the learned models may be biased towards specific structure patterns. To address this challenge, we propose the Cluster Information Transfer (\textbf{CIT}) mechanism, which can learn invariant representations for GNNs, thereby improving their generalization ability to various and unknown test graphs with structure shift. The CIT mechanism achieves this by combining different cluster information with the nodes while preserving their cluster-independent information. By generating nodes across different clusters, the mechanism significantly enhances the diversity of the nodes and helps GNNs learn the invariant representations. We provide a theoretical analysis of the CIT mechanism, showing that the impact of changing clusters during structure shift can be mitigated after transfer. Additionally, the proposed mechanism is a plug-in that can be easily used to improve existing GNNs. We comprehensively evaluate our proposed method on three typical structure shift scenarios, demonstrating its effectiveness in enhancing GNNs' performance.

        ----

        ## [1976] Transformers learn to implement preconditioned gradient descent for in-context learning

        **Authors**: *Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, Suvrit Sra*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ed3d610ea4b68e7afb30ea7d01422c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ed3d610ea4b68e7afb30ea7d01422c6-Abstract-Conference.html)

        **Abstract**:

        Several recent works demonstrate that transformers can implement algorithms like gradient descent. By a careful construction of weights, these works show that multiple layers of transformers are expressive enough to simulate iterations of gradient descent. Going beyond the question of expressivity, we ask: \emph{Can transformers learn to implement such algorithms by training over random problem instances?} To our knowledge, we make the first theoretical progress on this question via an analysis of the loss landscape for linear transformers trained over random instances of linear regression. For a single attention layer, we prove the global minimum of the training objective implements a single iteration of preconditioned gradient descent. Notably, the preconditioning matrix not only adapts to the input distribution but also to the variance induced by data inadequacy.  For a transformer with $L$ attention layers, we prove certain critical points of the training objective implement $L$ iterations of preconditioned gradient descent. Our results call for future theoretical studies on learning algorithms by training transformers.

        ----

        ## [1977] Linear Time Algorithms for k-means with Multi-Swap Local Search

        **Authors**: *Junyu Huang, Qilong Feng, Ziyun Huang, Jinhui Xu, Jianxin Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8eec8d7bcecf034304174e6b57dbc19a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8eec8d7bcecf034304174e6b57dbc19a-Abstract-Conference.html)

        **Abstract**:

        The local search methods have been widely used to solve the clustering problems. In practice, local search algorithms for clustering problems mainly adapt the single-swap strategy, which enables them to handle large-scale datasets and achieve linear running time in the data size. However, compared with multi-swap local search algorithms, there is a considerable gap on the approximation ratios of the single-swap local search algorithms. Although the current multi-swap local search algorithms provide small constant approximation, the proposed algorithms tend to have large polynomial running time, which cannot be used to handle large-scale datasets. In this paper, we propose a multi-swap local search algorithm for the $k$-means problem with linear running time in the data size. Given a swap size $t$, our proposed algorithm can achieve a $(50(1+\frac{1}{t})+\epsilon)$-approximation, which improves the current best result 509 (ICML 2019) with linear running time in the data size. Our proposed method, compared with previous multi-swap local search algorithms, is the first one to achieve linear running time in the data size. To obtain a more practical algorithm for the problem with better clustering quality and running time, we propose a sampling-based method which accelerates the process of clustering cost update during swaps. Besides, a recombination mechanism is proposed to find potentially better solutions. Empirical experiments show that our proposed algorithms achieve better performances compared with branch and bound solver (NeurIPS 2022) and other existing state-of-the-art local search algorithms on both small and large datasets.

        ----

        ## [1978] VaRT: Variational Regression Trees

        **Authors**: *Sebastian Salazar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8eff4196f50c43eda7bcf0f0cf87a0d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8eff4196f50c43eda7bcf0f0cf87a0d0-Abstract-Conference.html)

        **Abstract**:

        Decision trees are a well-established tool in machine learning for classification and regression tasks. In this paper, we introduce a novel non-parametric Bayesian model that uses variational inference to approximate a posterior distribution over the space of stochastic decision trees. We evaluate the model's performance on 18 datasets and demonstrate its competitiveness with other state-of-the-art methods in regression tasks. We also explore its application to causal inference problems. We provide a fully vectorized implementation of our algorithm in PyTorch.

        ----

        ## [1979] STREAMER: Streaming Representation Learning and Event Segmentation in a Hierarchical Manner

        **Authors**: *Ramy Mounir, Sujal Vijayaraghavan, Sudeep Sarkar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f0d446441a938d9de420a8ab8d7fd36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f0d446441a938d9de420a8ab8d7fd36-Abstract-Conference.html)

        **Abstract**:

        We present a novel self-supervised approach for hierarchical representation learning and segmentation of perceptual inputs in a streaming fashion. Our research addresses how to semantically group streaming inputs into chunks at various levels of a hierarchy while simultaneously learning, for each chunk, robust global representations throughout the domain. To achieve this, we propose STREAMER, an architecture that is trained layer-by-layer, adapting to the complexity of the input domain. In our approach, each layer is trained with two primary objectives: making accurate predictions into the future and providing necessary information to other levels for achieving the same objective. The event hierarchy is constructed by detecting prediction error peaks at different levels, where a detected boundary triggers a bottom-up information flow. At an event boundary, the encoded representation of inputs at one layer becomes the input to a higher-level layer. Additionally, we design a communication module that facilitates top-down and bottom-up exchange of information during the prediction process. Notably, our model is fully self-supervised and trained in a streaming manner, enabling a single pass on the training data. This means that the model encounters each input only once and does not store the data. We evaluate the performance of our model on the egocentric EPIC-KITCHENS dataset, specifically focusing on temporal event segmentation. Furthermore, we conduct event retrieval experiments using the learned representations to demonstrate the high quality of our video event representations. Illustration videos and code are available on our project page: https://ramymounir.com/publications/streamer

        ----

        ## [1980] Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning

        **Authors**: *Sotetsu Koyamada, Shinri Okano, Soichiro Nishimori, Yu Murata, Keigo Habara, Haruka Kita, Shin Ishii*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f153093758af93861a74a1305dfdc18-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f153093758af93861a74a1305dfdc18-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We propose Pgx, a suite of board game reinforcement learning (RL) environments written in JAX and optimized for GPU/TPU accelerators. By leveraging JAX's auto-vectorization and parallelization over accelerators, Pgx can efficiently scale to thousands of simultaneous simulations over accelerators. In our experiments on a DGX-A100 workstation, we discovered that Pgx can simulate RL environments 10-100x faster than existing implementations available in Python. Pgx includes RL environments commonly used as benchmarks in RL research, such as backgammon, chess, shogi, and Go. Additionally, Pgx offers miniature game sets and baseline models to facilitate rapid research cycles. We demonstrate the efficient training of the Gumbel AlphaZero algorithm with Pgx environments. Overall, Pgx provides high-performance environment simulators for researchers to accelerate their RL experiments. Pgx is available at https://github.com/sotetsuk/pgx.

        ----

        ## [1981] Robust Distributed Learning: Tight Error Bounds and Breakdown Point under Data Heterogeneity

        **Authors**: *Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, Geovani Rizk*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f182e220092f7f1fc44f3313023f5a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f182e220092f7f1fc44f3313023f5a0-Abstract-Conference.html)

        **Abstract**:

        The theory underlying robust distributed learning algorithms, designed to resist adversarial machines, matches empirical observations when data is homogeneous. Under data heterogeneity however, which is the norm in practical scenarios, established lower bounds on the learning error are essentially vacuous and greatly mismatch empirical observations. This is because the heterogeneity model considered is too restrictive and does not cover basic learning tasks such as least-squares regression. We consider in this paper a more realistic heterogeneity model, namely $(G,B)$-gradient dissimilarity, and show that it covers a larger class of learning problems than existing theory. Notably, we show that the breakdown point under heterogeneity is lower than the classical fraction $\frac{1}{2}$. We also prove a new lower bound on the learning error of any distributed learning algorithm. We derive a matching upper bound for a robust variant of distributed gradient descent, and empirically show that our analysis reduces the gap between theory and practice.

        ----

        ## [1982] Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers

        **Authors**: *Zixuan Jiang, Jiaqi Gu, Hanqing Zhu, David Z. Pan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f1bacee31caf990a4f08d84f0ccb322-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f1bacee31caf990a4f08d84f0ccb322-Abstract-Conference.html)

        **Abstract**:

        Transformers have achieved great success in machine learning applications.Normalization techniques, such as Layer Normalization (LayerNorm, LN) and Root Mean Square Normalization (RMSNorm), play a critical role in accelerating and stabilizing the training of Transformers.While LayerNorm recenters and rescales input vectors, RMSNorm only rescales the vectors by their RMS value.Despite being more computationally efficient, RMSNorm may compromise the representation ability of Transformers.There is currently no consensus regarding the preferred normalization technique, as some models employ LayerNorm while others utilize RMSNorm, especially in recent large language models.It is challenging to convert Transformers with one normalization to the other type.While there is an ongoing disagreement between the two normalization types,we propose a solution to unify two mainstream Transformer architectures, Pre-LN and Pre-RMSNorm Transformers.By removing the inherent redundant mean information in the main branch of Pre-LN Transformers, we can reduce LayerNorm to RMSNorm, achieving higher efficiency.We further propose the Compressed RMSNorm (CRMSNorm) and Pre-CRMSNorm Transformer based on a lossless compression of the zero-mean vectors.We formally establish the equivalence of Pre-LN, Pre-RMSNorm, and Pre-CRMSNorm Transformer variants in both training and inference.It implies that Pre-LN Transformers can be substituted with Pre-(C)RMSNorm counterparts at almost no cost, offering the same arithmetic functionality along with free efficiency improvement.Experiments demonstrate that we can reduce the training and inference time of Pre-LN Transformers by 1% - 10%.

        ----

        ## [1983] Multimodal Clinical Benchmark for Emergency Care (MC-BEC): A Comprehensive Benchmark for Evaluating Foundation Models in Emergency Medicine

        **Authors**: *Emma Chen, Aman Kansal, Julie Chen, Boyang Tom Jin, Julia Rachel Reisler, David A. Kim, Pranav Rajpurkar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f61049e8fe5b9ed714860b951066f1e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f61049e8fe5b9ed714860b951066f1e-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We propose the Multimodal Clinical Benchmark for Emergency Care (MC-BEC), a comprehensive benchmark for evaluating foundation models in Emergency Medicine using a dataset of 100K+ continuously monitored Emergency Department visits from 2020-2022. MC-BEC focuses on clinically relevant prediction tasks at timescales from minutes to days, including predicting patient decompensation, disposition, and emergency department (ED) revisit, and includes a standardized evaluation framework with train-test splits and evaluation metrics. The multimodal dataset includes a wide range of detailed clinical data, including triage information, prior diagnoses and medications, continuously measured vital signs, electrocardiogram and photoplethysmograph waveforms, orders placed and medications administered throughout the visit, free-text reports of imaging studies, and information on ED diagnosis, disposition, and subsequent revisits. We provide performance baselines for each prediction task to enable the evaluation of multimodal, multitask models. We believe that MC-BEC will encourage researchers to develop more effective, generalizable, and accessible foundation models for multimodal clinical data.

        ----

        ## [1984] AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix

        **Authors**: *Yun Yue, Zhiling Ye, Jiadi Jiang, Yongchao Liu, Ke Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f9d459c19b59b5400ce396e0f8c23e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f9d459c19b59b5400ce396e0f8c23e0-Abstract-Conference.html)

        **Abstract**:

        Adaptive optimizers, such as Adam, have achieved remarkable success in deep learning. A key component of these optimizers is the so-called preconditioning matrix, providing enhanced gradient information and regulating the step size of each gradient direction. In this paper, we propose a novel approach to designing the preconditioning matrix by utilizing the gradient difference between two successive steps as the diagonal elements. These diagonal elements are closely related to the Hessian and can be perceived as an approximation of the inner product between the Hessian row vectors and difference of the adjacent parameter vectors. Additionally, we introduce an auto-switching function that enables the preconditioning matrix to switch dynamically between Stochastic Gradient Descent (SGD) and the adaptive optimizer. Based on these two techniques, we develop a new optimizer named AGD that enhances the generalization performance. We evaluate AGD on public datasets of Natural Language Processing (NLP), Computer Vision (CV), and Recommendation Systems (RecSys). Our experimental results demonstrate that AGD outperforms the state-of-the-art (SOTA) optimizers, achieving highly competitive or significantly better predictive performance. Furthermore, we analyze how AGD is able to switch automatically between SGD and the adaptive optimizer and its actual effects on various scenarios. The code is available at https://github.com/intelligent-machine-learning/dlrover/tree/master/atorch/atorch/optimizers.

        ----

        ## [1985] PDP: Parameter-free Differentiable Pruning is All You Need

        **Authors**: *Minsik Cho, Saurabh Adya, Devang Naik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f9f4eb32b9081a90f2a0b2627eb2a24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f9f4eb32b9081a90f2a0b2627eb2a24-Abstract-Conference.html)

        **Abstract**:

        DNN pruning is a popular way to reduce the size of a model, improve the inferencelatency, and minimize the power consumption on DNN accelerators. However,existing approaches might be too complex, expensive or ineffective to apply toa variety of vision/language tasks, DNN architectures and to honor structuredpruning constraints. In this paper, we propose an efficient yet effective train-timepruning scheme, Parameter-free Differentiable Pruning (PDP), which offers state-of-the-art qualities in model size, accuracy, and training cost. PDP uses a dynamicfunction of weights during training to generate soft pruning masks for the weightsin a parameter-free manner for a given pruning target. While differentiable, thesimplicity and efficiency of PDP make it universal enough to deliver state-of-the-artrandom/structured/channel pruning results on various vision and natural languagetasks. For example, for MobileNet-v1, PDP can achieve 68.2% top-1 ImageNet1kaccuracy at 86.6% sparsity, which is 1.7% higher accuracy than those from thestate-of-the-art algorithms. Also, PDP yields over 83.1% accuracy on Multi-GenreNatural Language Inference with 90% sparsity for BERT, while the next best fromthe existing techniques shows 81.5% accuracy. In addition, PDP can be applied tostructured pruning, such as N:M pruning and channel pruning. For 1:4 structuredpruning of ResNet18, PDP improved the top-1 ImageNet1k accuracy by over 3.6%over the state-of-the-art. For channel pruning of ResNet50, PDP reduced the top-1ImageNet1k accuracy by 0.6% from the state-of-the-art.

        ----

        ## [1986] ExPT: Synthetic Pretraining for Few-Shot Experimental Design

        **Authors**: *Tung Nguyen, Sudhanshu Agrawal, Aditya Grover*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8fab4407e1fe9006b39180525c0d323c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8fab4407e1fe9006b39180525c0d323c-Abstract-Conference.html)

        **Abstract**:

        Experimental design is a fundamental problem in many science and engineering fields. In this problem, sample efficiency is crucial due to the time, money, and safety costs of real-world design evaluations. Existing approaches either rely on active data collection or access to large, labeled datasets of past experiments, making them impractical in many real-world scenarios. In this work, we address the more challenging yet realistic setting of few-shot experimental design, where only a few labeled data points of input designs and their corresponding values are available. We approach this problem as a conditional generation task, where a model conditions on a few labeled examples and the desired output to generate an optimal input design. To this end, we introduce Experiment Pretrained Transformers (ExPT), a foundation model for few-shot experimental design that employs a novel combination of synthetic pretraining with in-context learning. In ExPT, we only assume knowledge of a finite collection of unlabelled data points from the input domain and pretrain a transformer neural network to optimize diverse synthetic functions defined over this domain. Unsupervised pretraining allows ExPT to adapt to any design task at test time in an in-context fashion by conditioning on a few labeled data points from the target task and generating the candidate optima. We evaluate ExPT on few-shot experimental design in challenging domains and demonstrate its superior generality and performance compared to existing methods. The source code is available at https://github.com/tung-nd/ExPT.git.

        ----

        ## [1987] ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings

        **Authors**: *Shibo Hao, Tianyang Liu, Zhen Wang, Zhiting Hu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html)

        **Abstract**:

        Integrating large language models (LLMs) with various tools has led to increased attention in the field. Existing approaches either involve fine-tuning the LLM, which is both computationally costly and limited to a fixed set of tools, or prompting LLMs by in-context tool demonstrations. Although the latter method offers adaptability to new tools, it struggles with the inherent context length constraint of LLMs when many new tools are presented, and mastering a new set of tools with few-shot examples remains challenging, resulting in suboptimal performance. To address these limitations, we propose a novel solution, named ToolkenGPT, wherein LLMs effectively learn to master tools as predicting tokens through tool embeddings for solving complex tasks. In this framework, each tool is transformed into vector embeddings and plugged into the language model head. Once the function is triggered during text generation, the LLM enters a special function mode to execute the tool calls. Our experiments show that function embeddings effectively help LLMs understand tool use and improve on several tasks, including numerical reasoning, knowledge-based question answering and embodied decision-making.

        ----

        ## [1988] CLIP4HOI: Towards Adapting CLIP for Practical Zero-Shot HOI Detection

        **Authors**: *Yunyao Mao, Jiajun Deng, Wengang Zhou, Li Li, Yao Fang, Houqiang Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8fd5bc08e744fe0dfe798c61d1575a22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8fd5bc08e744fe0dfe798c61d1575a22-Abstract-Conference.html)

        **Abstract**:

        Zero-shot Human-Object Interaction (HOI) detection aims to identify both seen and unseen HOI categories. A strong zero-shot HOI detector is supposed to be not only capable of discriminating novel interactions but also robust to positional distribution discrepancy between seen and unseen categories when locating human-object pairs. However, top-performing zero-shot HOI detectors rely on seen and predefined unseen categories to distill knowledge from CLIP and jointly locate human-object pairs without considering the potential positional distribution discrepancy, leading to impaired transferability. In this paper, we introduce CLIP4HOI, a novel framework for zero-shot HOI detection. CLIP4HOI is developed on the vision-language model CLIP and ameliorates the above issues in the following two aspects. First, to avoid the model from overfitting to the joint positional distribution of seen human-object pairs, we seek to tackle the problem of zero-shot HOI detection in a disentangled two-stage paradigm. To be specific, humans and objects are independently identified and all feasible human-object pairs are processed by Human-Object interactor for pairwise proposal generation. Second, to facilitate better transferability, the CLIP model is elaborately adapted into a fine-grained HOI classifier for proposal discrimination, avoiding data-sensitive knowledge distillation. Finally, experiments on prevalent benchmarks show that our CLIP4HOI outperforms previous approaches on both rare and unseen categories, and sets a series of state-of-the-art records under a variety of zero-shot settings.

        ----

        ## [1989] Transformer-based Planning for Symbolic Regression

        **Authors**: *Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, Chandan K. Reddy*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html)

        **Abstract**:

        Symbolic regression (SR) is a challenging task in machine learning that involves finding a mathematical expression for a function based on its values. Recent advancements in SR have demonstrated the effectiveness of pre-trained transformer models in generating equations as sequences, leveraging large-scale pre-training on synthetic datasets and offering notable advantages in terms of inference time over classical Genetic Programming (GP) methods. However, these models primarily rely on supervised pre-training objectives borrowed from text generation and overlook equation discovery goals like accuracy and complexity. To address this, we propose TPSR, a Transformer-based Planning strategy for Symbolic Regression that incorporates Monte Carlo Tree Search planning algorithm into the transformer decoding process. Unlike conventional decoding strategies, TPSR enables the integration of non-differentiable equation verification feedback, such as fitting accuracy and complexity, as external sources of knowledge into the transformer equation generation process. Extensive experiments on various datasets show that our approach outperforms state-of-the-art methods, enhancing the model's fitting-complexity trade-off, extrapolation abilities, and robustness to noise.

        ----

        ## [1990] Exploring Geometry of Blind Spots in Vision models

        **Authors**: *Sriram Balasubramanian, Gaurang Sriramanan, Vinu Sankar Sadasivan, Soheil Feizi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90043ebd68500f9efe84fedf860a64f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90043ebd68500f9efe84fedf860a64f3-Abstract-Conference.html)

        **Abstract**:

        Despite the remarkable success of deep neural networks in a myriad of settings, several works have demonstrated their overwhelming sensitivity to near-imperceptible perturbations, known as adversarial attacks. On the other hand, prior works have also observed that deep networks can be under-sensitive, wherein large-magnitude perturbations in input space do not induce appreciable changes to network activations. In this work, we study in detail the phenomenon of under-sensitivity in vision models such as CNNs and Transformers, and present techniques to study the geometry and extent of “equi-confidence” level sets of such networks. We propose a Level Set Traversal algorithm that iteratively explores regions of high confidence with respect to the input space using orthogonal components of the local gradients. Given a source image, we use this algorithm to identify inputs that lie in the same equi-confidence level set as the source image despite being perceptually similar to arbitrary images from other classes. We further observe that the source image is linearly connected by a high-confidence path to these inputs, uncovering a star-like structure for level sets of deep networks. Furthermore, we attempt to identify and estimate the extent of these connected higher-dimensional regions over which the model maintains a high degree of confidence.

        ----

        ## [1991] Provable benefits of annealing for estimating normalizing constants: Importance Sampling, Noise-Contrastive Estimation, and beyond

        **Authors**: *Omar Chehab, Aapo Hyvärinen, Andrej Risteski*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90080022263cddafddd4a0726f1fb186-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90080022263cddafddd4a0726f1fb186-Abstract-Conference.html)

        **Abstract**:

        Recent research has developed several Monte Carlo methods for estimating the normalization constant (partition function) based on the idea of annealing. This means sampling successively from a path of distributions which interpolate between a tractable "proposal" distribution and the unnormalized "target" distribution. Prominent estimators in this family include annealed importance sampling and annealed noise-contrastive estimation (NCE). Such methods hinge on a number of design choices: which estimator to use, which path of distributions to use and whether to use a path at all; so far, there is no definitive theory on which choices are efficient. Here, we evaluate each design choice by the asymptotic estimation error it produces. First, we show that using NCE is more efficient than the importance sampling estimator, but in the limit of infinitesimal path steps, the difference vanishes. Second, we find that using the geometric path brings down the estimation error from an exponential to a polynomial function of the parameter distance between the target and proposal distributions. Third, we find that the arithmetic path, while rarely used, can offer optimality properties over the universally-used geometric path. In fact, in a particular limit, the optimal path is arithmetic. Based on this theory, we finally propose a two-step estimator to approximate the optimal path in an efficient way.

        ----

        ## [1992] Strategic Distribution Shift of Interacting Agents via Coupled Gradient Flows

        **Authors**: *Lauren E. Conger, Franca Hoffmann, Eric Mazumdar, Lillian J. Ratliff*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/902c462e821e5e639ac3422b48b65932-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/902c462e821e5e639ac3422b48b65932-Abstract-Conference.html)

        **Abstract**:

        We propose a novel framework for analyzing the dynamics of distribution shift in real-world systems that captures the feedback loop between learning algorithms and the distributions on which they are deployed. Prior work largely models feedback-induced distribution shift as adversarial or via an overly simplistic distribution-shift structure. In contrast, we propose a coupled partial differential equation model that captures fine-grained changes in the distribution over time by accounting for complex dynamics that arise due to strategic responses to algorithmic decision-making, non-local endogenous population interactions, and other exogenous sources of distribution shift. We consider two common settings in machine learning: cooperative settings with information asymmetries, and competitive settings where a learner faces strategic users. For both of these settings, when the algorithm retrains via gradient descent, we prove asymptotic convergence of the retraining procedure to a steady-state, both in finite and in infinite dimensions, obtaining explicit rates in terms of the model parameters. To do so we derive new results on the convergence of coupled PDEs that extends what is known on multi-species systems. Empirically, we show that our approach captures well-documented forms of distribution shifts like polarization and disparate impacts that simpler models cannot capture.

        ----

        ## [1993] Learning Time-Invariant Representations for Individual Neurons from Population Dynamics

        **Authors**: *Lu Mi, Trung Le, Tianxing He, Eli Shlizerman, Uygar Sümbül*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html)

        **Abstract**:

        Neurons can display highly variable dynamics. While such variability presumably supports the wide range of behaviors generated by the organism, their gene expressions are relatively stable in the adult brain. This suggests that neuronal activity is a combination of its time-invariant identity and the inputs the neuron receives from the rest of the circuit. Here, we propose a self-supervised learning based method to assign time-invariant representations to individual neurons based on permutation-, and population size-invariant summary of population recordings. We fit dynamical models to neuronal activity to learn a representation by considering the activity of both the individual and the neighboring population. Our self-supervised approach and use of implicit representations enable robust inference against imperfections such as partial overlap of neurons across sessions, trial-to-trial variability, and limited availability of molecular (transcriptomic) labels for downstream supervised tasks. We demonstrate our method on a public multimodal dataset of mouse cortical neuronal activity and transcriptomic labels. We report >35\% improvement in predicting the transcriptomic subclass identity and >20\% improvement in predicting class identity with respect to the state-of-the-art.

        ----

        ## [1994] GeoTMI: Predicting Quantum Chemical Property with Easy-to-Obtain Geometry via Positional Denoising

        **Authors**: *Hyeonsu Kim, Jeheon Woo, Seonghwan Kim, Seokhyun Moon, Jun Hyeong Kim, Woo Youn Kim*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/903c5eb12f2389c4847574df90503d63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/903c5eb12f2389c4847574df90503d63-Abstract-Conference.html)

        **Abstract**:

        As quantum chemical properties have a dependence on their geometries, graph neural networks (GNNs) using 3D geometric information have achieved high prediction accuracy in many tasks. However, they often require 3D geometries obtained from high-level quantum mechanical calculations, which are practically infeasible, limiting their applicability to real-world problems. To tackle this, we propose a new training framework, GeoTMI, that employs denoising process to predict properties accurately using easy-to-obtain geometries (corrupted versions of correct geometries, such as those obtained from low-level calculations). Our starting point was the idea that the correct geometry is the best description of the target property. Hence, to incorporate information of the correct, GeoTMI aims to maximize mutual information between three variables: the correct and the corrupted geometries and the property. GeoTMI also explicitly updates the corrupted input to approach the correct geometry as it passes through the GNN layers, contributing to more effective denoising. We investigated the performance of the proposed method using 3D GNNs for three prediction tasks: molecular properties, a chemical reaction property, and relaxed energy in a heterogeneous catalytic system. Our results showed consistent improvements in accuracy across various tasks, demonstrating the effectiveness and robustness of GeoTMI.

        ----

        ## [1995] PRED: Pre-training via Semantic Rendering on LiDAR Point Clouds

        **Authors**: *Hao Yang, Haiyang Wang, Di Dai, Liwei Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/903f778fe1341e5351b5b63e0e6b197f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/903f778fe1341e5351b5b63e0e6b197f-Abstract-Conference.html)

        **Abstract**:

        Pre-training is crucial in 3D-related fields such as autonomous driving where point cloud annotation is costly and challenging. Many recent studies on point cloud pre-training, however, have overlooked the issue of incompleteness, where only a fraction of the points are captured by LiDAR, leading to ambiguity during the training phase. On the other hand, images offer more comprehensive information and richer semantics that can bolster point cloud encoders in addressing the incompleteness issue inherent in point clouds. Yet, incorporating images into point cloud pre-training presents its own challenges due to occlusions, potentially causing misalignments between points and pixels. In this work, we propose PRED, a novel image-assisted pre-training framework for outdoor point clouds in an occlusion-aware manner. The main ingredient of our framework is a Birds-Eye-View (BEV) feature map conditioned semantic rendering, leveraging the semantics of images for supervision through neural rendering. We further enhance our model's performance by incorporating point-wise masking with a high mask ratio (95%). Extensive experiments demonstrate PRED's superiority over prior point cloud pre-training methods, providing significant improvements on various large-scale datasets for 3D perception tasks. Codes will be available at https://github.com/PRED4pc/PRED.

        ----

        ## [1996] Active Observing in Continuous-time Control

        **Authors**: *Samuel Holt, Alihan Hüyük, Mihaela van der Schaar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9050e8d5b5de08d16e65dc79ad5c0146-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9050e8d5b5de08d16e65dc79ad5c0146-Abstract-Conference.html)

        **Abstract**:

        The control of continuous-time environments while actively deciding when to take costly observations in time is a crucial yet unexplored problem, particularly relevant to real-world scenarios such as medicine, low-power systems, and resource management. Existing approaches either rely on continuous-time control methods that take regular, expensive observations in time or discrete-time control with costly observation methods, which are inapplicable to continuous-time settings due to the compounding discretization errors introduced by time discretization. In this work, we are the first to formalize the continuous-time control problem with costly observations. Our key theoretical contribution shows that observing at regular time intervals is not optimal in certain environments, while irregular observation policies yield higher expected utility.  This perspective paves the way for the development of novel methods that can take irregular observations in continuous-time control with costly observations. We empirically validate our theoretical findings in various continuous-time environments, including a cancer simulation, by constructing a simple initial method to solve this new problem, with a heuristic threshold on the variance of reward rollouts in an offline continuous-time model-based model predictive control (MPC) planner. Although determining the optimal method remains an open problem, our work offers valuable insights and understanding of this unique problem, laying the foundation for future research in this area.

        ----

        ## [1997] Principled Weight Initialisation for Input-Convex Neural Networks

        **Authors**: *Pieter-Jan Hoedt, Günter Klambauer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9062b7d6e522dadf4f7d85d49b60d81e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9062b7d6e522dadf4f7d85d49b60d81e-Abstract-Conference.html)

        **Abstract**:

        Input-Convex Neural Networks (ICNNs) are networks that guarantee convexity in their input-output mapping. These networks have been successfully applied for energy-based modelling, optimal transport problems and learning invariances.The convexity of ICNNs is achieved by using non-decreasing convex activation functions and non-negative weights. Because of these peculiarities, previous initialisation strategies, which implicitly assume centred weights, are not effective for ICNNs. By studying signal propagation through layers with non-negative weights, we are able to derive a principled weight initialisation for ICNNs. Concretely, we generalise signal propagation theory by removing the assumption that weights are sampled from a centred distribution. In a set of experiments, we demonstrate that our principled initialisation effectively accelerates learning in ICNNs and leads to better generalisation. Moreover, we find that, in contrast to common belief, ICNNs can be trained without skip-connections when initialised correctly. Finally, we apply ICNNs to a real-world drug discovery task and show that they allow for more effective molecular latent space exploration.

        ----

        ## [1998] Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning

        **Authors**: *Yifan Zang, Jinmin He, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/906c860f1b7515a8ffec02dcdac74048-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/906c860f1b7515a8ffec02dcdac74048-Abstract-Conference.html)

        **Abstract**:

        Grouping is ubiquitous in natural systems and is essential for promoting efficiency in team coordination. This paper proposes a novel formulation of Group-oriented Multi-Agent Reinforcement Learning (GoMARL), which learns automatic grouping without domain knowledge for efficient cooperation. In contrast to existing approaches that attempt to directly learn the complex relationship between the joint action-values and individual utilities, we empower subgroups as a bridge to model the connection between small sets of agents and encourage cooperation among them, thereby improving the learning efficiency of the whole team. In particular, we factorize the joint action-values as a combination of group-wise values, which guide agents to improve their policies in a fine-grained fashion. We present an automatic grouping mechanism to generate dynamic groups and group action-values. We further introduce a hierarchical control for policy learning that drives the agents in the same group to specialize in similar policies and possess diverse strategies for various groups. Experiments on the StarCraft II micromanagement tasks and Google Research Football scenarios verify our method's effectiveness. Extensive component studies show how grouping works and enhances performance.

        ----

        ## [1999] On the Minimax Regret for Online Learning with Feedback Graphs

        **Authors**: *Khaled Eldowa, Emmanuel Esposito, Tommaso Cesari, Nicolò Cesa-Bianchi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/908f03779b5b063413fbf0247a46a403-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/908f03779b5b063413fbf0247a46a403-Abstract-Conference.html)

        **Abstract**:

        In this work, we improve on the upper and lower bounds for the regret of online learning with strongly observable undirected feedback graphs. The best known upper bound for this problem is $\mathcal{O}\bigl(\sqrt{\alpha T\ln K}\bigr)$, where $K$ is the number of actions, $\alpha$ is the independence number of the graph, and $T$ is the time horizon. The $\sqrt{\ln K}$ factor is known to be necessary when $\alpha = 1$ (the experts case). On the other hand, when $\alpha = K$ (the bandits case), the minimax rate is known to be $\Theta\bigl(\sqrt{KT}\bigr)$, and a lower bound $\Omega\bigl(\sqrt{\alpha T}\bigr)$ is known to hold for any $\alpha$. Our improved upper bound $\mathcal{O}\bigl(\sqrt{\alpha T(1+\ln(K/\alpha))}\bigr)$ holds for any $\alpha$ and matches the lower bounds for bandits and experts, while interpolating intermediate cases. To prove this result, we use FTRL with $q$-Tsallis entropy for a carefully chosen value of $q \in [1/2, 1)$ that varies with $\alpha$. The analysis of this algorithm requires a new bound on the variance term in the regret. We also show how to extend our techniques to time-varying graphs, without requiring prior knowledge of their independence numbers. Our upper bound is complemented by an improved $\Omega\bigl(\sqrt{\alpha T(\ln K)/(\ln\alpha)}\bigr)$ lower bound for all $\alpha > 1$, whose analysis relies on a novel reduction to multitask learning. This shows that a logarithmic factor is necessary as soon as $\alpha < K$.

        ----

        

[Go to the previous page](NIPS-2023-list09.md)

[Go to the next page](NIPS-2023-list11.md)

[Go to the catalog section](README.md)