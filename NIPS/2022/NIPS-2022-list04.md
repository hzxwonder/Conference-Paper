## [600] Ensemble of Averages: Improving Model Selection and Boosting Performance in Domain Generalization

        **Authors**: *Devansh Arpit, Huan Wang, Yingbo Zhou, Caiming Xiong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/372cb7805eaccb2b7eed641271a30eec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/372cb7805eaccb2b7eed641271a30eec-Abstract-Conference.html)

        **Abstract**:

        In Domain Generalization (DG) settings, models trained independently on a given set of training domains have notoriously chaotic performance on distribution shifted test domains, and stochasticity in optimization (e.g. seed) plays a big role. This makes deep learning models unreliable in real world settings. We first show that this chaotic behavior exists even along the training optimization trajectory of a single model, and propose a simple model averaging protocol that both significantly boosts domain generalization and diminishes the impact of stochasticity by improving the rank correlation between the in-domain validation accuracy and out-domain test accuracy, which is crucial for reliable early stopping. Taking advantage of our observation, we show that instead of ensembling unaveraged models (that is typical in practice), ensembling moving average models (EoA) from independent runs further boosts performance. We theoretically explain the boost in performance of ensembling and model averaging by adapting the well known Bias-Variance trade-off to the domain generalization setting. On the DomainBed benchmark, when using a pre-trained ResNet-50, this ensemble of averages achieves an average of $68.0\%$, beating vanilla ERM (w/o averaging/ensembling) by $\sim 4\%$, and when using a pre-trained RegNetY-16GF, achieves an average of $76.6\%$, beating vanilla ERM by $\sim 6\%$.

        ----

        ## [601] Is $L^2$ Physics Informed Loss Always Suitable for Training Physics Informed Neural Network?

        **Authors**: *Chuwei Wang, Shanda Li, Di He, Liwei Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html)

        **Abstract**:

        The Physics-Informed Neural Network (PINN) approach is a new and promising way to solve partial differential equations using deep learning. The $L^2$ Physics-Informed Loss is the de-facto standard in training Physics-Informed Neural Networks. In this paper, we challenge this common practice by investigating the relationship between the loss function and the approximation quality of the learned solution. In particular, we leverage the concept of stability in the literature of partial differential equation to study the asymptotic behavior of the learned solution as the loss approaches zero. With this concept, we study an important class of high-dimensional non-linear PDEs in optimal control, the Hamilton-Jacobi-Bellman (HJB) Equation, and prove that for general $L^p$ Physics-Informed Loss, a wide class of HJB equation is stable only if $p$ is sufficiently large. Therefore, the commonly used $L^2$ loss is not suitable for training PINN on those equations, while $L^{\infty}$ loss is a better choice. Based on the theoretical insight, we develop a novel PINN training algorithm to minimize the $L^{\infty}$ loss for HJB equations which is in a similar spirit to adversarial training. The effectiveness of the proposed algorithm is empirically demonstrated through experiments.  Our code is released at https://github.com/LithiumDA/L_inf-PINN.

        ----

        ## [602] Vision GNN: An Image is Worth Graph of Nodes

        **Authors**: *Kai Han, Yunhe Wang, Jianyuan Guo, Yehui Tang, Enhua Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3743e69c8e47eb2e6d3afaea80e439fb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3743e69c8e47eb2e6d3afaea80e439fb-Abstract-Conference.html)

        **Abstract**:

        Network architecture plays a key role in the deep learning-based computer vision system. The widely-used convolutional neural network and transformer treat the image as a grid or sequence structure, which is not flexible to capture irregular and complex objects. In this paper, we propose to represent the image as a graph structure and introduce a new \emph{Vision GNN} (ViG) architecture to extract graph-level feature for visual tasks. We first split the image to a number of patches which are viewed as nodes, and construct a graph by connecting the nearest neighbors. Based on the graph representation of images, we build our ViG model to transform and exchange information among all the nodes. ViG consists of two basic modules: Grapher module with graph convolution for aggregating and updating graph information, and FFN module with two linear layers for node feature transformation. Both isotropic and pyramid architectures of ViG are built with different model sizes. Extensive experiments on image recognition and object detection tasks demonstrate the superiority of our ViG architecture. We hope this pioneering study of GNN on general visual tasks will provide useful inspiration and experience for future research.  The PyTorch code is available at \url{https://github.com/huawei-noah/Efficient-AI-Backbones} and the MindSpore code is available at \url{https://gitee.com/mindspore/models}.

        ----

        ## [603] ELIGN: Expectation Alignment as a Multi-Agent Intrinsic Reward

        **Authors**: *Zixian Ma, Rose Wang, Fei-Fei Li, Michael S. Bernstein, Ranjay Krishna*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3753163b089e405ef10302698cd9a7fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3753163b089e405ef10302698cd9a7fc-Abstract-Conference.html)

        **Abstract**:

        Modern multi-agent reinforcement learning frameworks rely on centralized training and reward shaping to perform well. However, centralized training and dense rewards are not readily available in the real world. Current multi-agent algorithms struggle to learn in the alternative setup of decentralized training or sparse rewards. To address these issues, we propose a self-supervised intrinsic reward  \textit{ELIGN - expectation alignment - } inspired by the self-organization principle in Zoology. Similar to how animals collaborate in a decentralized manner with those in their vicinity, agents trained with expectation alignment learn behaviors that match their neighbors' expectations. This allows the agents to learn collaborative behaviors without any external reward or centralized training. We demonstrate the efficacy of our approach across 6 tasks in the multi-agent particle and the complex Google Research football environments, comparing ELIGN to sparse and curiosity-based intrinsic rewards. When the number of agents increases, ELIGN scales well in all multi-agent tasks except for one where agents have different capabilities. We show that agent coordination improves through expectation alignment because agents learn to divide tasks amongst themselves, break coordination symmetries, and confuse adversaries. These results identify tasks where expectation alignment is a more useful strategy than curiosity-driven exploration for multi-agent coordination, enabling agents to do zero-shot coordination.

        ----

        ## [604] Factored DRO: Factored Distributionally Robust Policies for Contextual Bandits

        **Authors**: *Tong Mu, Yash Chandak, Tatsunori B. Hashimoto, Emma Brunskill*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/375cc00f0a2297af84304d0c0cd3b7ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/375cc00f0a2297af84304d0c0cd3b7ed-Abstract-Conference.html)

        **Abstract**:

        While there has been extensive work on learning from offline data for contextual multi-armed bandit settings, existing methods typically assume there is no environment shift: that the learned policy will operate in the same environmental process as that of data collection. However, this assumption may limit the use of these methods for many practical situations where there may be distribution shifts. In this work we propose Factored Distributionally Robust Optimization (Factored-DRO), which is able to separately handle distribution shifts in the context distribution and shifts in the reward generating process. Prior work that either ignores potential shifts in the context, or considers them jointly, can lead to performance that is too conservative, especially under certain forms of reward feedback. Our Factored-DRO objective mitigates this by considering the shifts separately, and our proposed estimators are consistent and converge asymptotically. We also introduce a practical algorithm and demonstrate promising empirical results in environments based on real-world datasets, such as voting outcomes and scene classification.

        ----

        ## [605] Could Giant Pre-trained Image Models Extract Universal Representations?

        **Authors**: *Yutong Lin, Ze Liu, Zheng Zhang, Han Hu, Nanning Zheng, Stephen Lin, Yue Cao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3776558654d8db1bfcb9ebde0e01184e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3776558654d8db1bfcb9ebde0e01184e-Abstract-Conference.html)

        **Abstract**:

        Frozen pretrained models have become a viable alternative to the pretraining-then-finetuning paradigm for transfer learning. However, with frozen models there are relatively few parameters available for adapting to downstream tasks, which is problematic in computer vision where tasks vary significantly in input/output format and the type of information that is of value. In this paper, we present a study of frozen pretrained models when applied to diverse and representative computer vision tasks, including object detection, semantic segmentation and video action recognition. From this empirical analysis, our work answers the questions of what pretraining task fits best with this frozen setting, how to make the frozen setting more flexible to various downstream tasks, and the effect of larger model sizes. We additionally examine the upper bound of performance using a giant frozen pretrained model with 3 billion parameters (SwinV2-G) and find that it reaches competitive performance on a varied set of major benchmarks with only one shared frozen base network: 60.0 box mAP and 52.2 mask mAP on COCO object detection test-dev, 57.6 val mIoU on ADE20K semantic segmentation, and 81.7 top-1 accuracy on Kinetics-400 action recognition. With this work, we hope to bring greater attention to this promising path of freezing pretrained image models.

        ----

        ## [606] A Scalable Deterministic Global Optimization Algorithm for Training Optimal Decision Tree

        **Authors**: *Kaixun Hua, Jiayang Ren, Yankai Cao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/37771cc0be272368102a37f202bb88d8-Abstract-Conference.html)

        **Abstract**:

        The training of optimal decision tree via mixed-integer programming (MIP) has attracted much attention in recent literature. However, for large datasets, state-of-the-art approaches struggle to solve the optimal decision tree training problems to a provable global optimal solution within a reasonable time. In this paper, we reformulate the optimal decision tree training problem as a two-stage optimization problem and propose a tailored reduced-space branch and bound algorithm to train optimal decision tree for the classification tasks with continuous features. We present several structure-exploiting lower and upper bounding methods. The computation of bounds can be decomposed into the solution of many small-scale subproblems and can be naturally parallelized. With these bounding methods, we prove that our algorithm can converge by branching only on variables representing the optimal decision tree structure, which is invariant to the size of datasets. Moreover, we propose a novel sample reduction method that can predetermine the cost of part of samples at each BB node. Combining the sample reduction method with the parallelized bounding strategies, our algorithm can be extremely scalable. Our algorithm can find global optimal solutions on dataset with over 245,000 samples (1000 cores, less than 1% optimality gap, within 2 hours). We test 21 real-world datasets from UCI Repository. The results reveal that for datasets with over 7,000 samples, our algorithm can, on average, improve the training accuracy by 3.6% and testing accuracy by 2.8%, compared to the current state-of-the-art.

        ----

        ## [607] Thor: Wielding Hammers to Integrate Language Models and Automated Theorem Provers

        **Authors**: *Albert Qiaochu Jiang, Wenda Li, Szymon Tworkowski, Konrad Czechowski, Tomasz Odrzygózdz, Piotr Milos, Yuhuai Wu, Mateja Jamnik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/377c25312668e48f2e531e2f2c422483-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/377c25312668e48f2e531e2f2c422483-Abstract-Conference.html)

        **Abstract**:

        In theorem proving, the task of selecting useful premises from a large library to unlock the proof of a given conjecture is crucially important. This presents a challenge for all theorem provers, especially the ones based on language models, due to their relative inability to reason over huge volumes of premises in text form. This paper introduces Thor, a framework integrating language models and automated theorem provers to overcome this difficulty. In Thor, a class of methods called hammers that leverage the power of automated theorem provers are used for premise selection, while all other tasks are designated to language models. Thor increases a language model's success rate on the PISA dataset from $39\%$ to $57\%$, while solving $8.2\%$ of problems neither language models nor automated theorem provers are able to solve on their own. Furthermore, with a significantly smaller computational budget, Thor can achieve a success rate on the MiniF2F dataset that is on par with the best existing methods. Thor can be instantiated for the majority of popular interactive theorem provers via a straightforward protocol we provide.

        ----

        ## [608] Rethinking Knowledge Graph Evaluation Under the Open-World Assumption

        **Authors**: *Haotong Yang, Zhouchen Lin, Muhan Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/378226e5df7eded3e401de5c9493143c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/378226e5df7eded3e401de5c9493143c-Abstract-Conference.html)

        **Abstract**:

        Most knowledge graphs (KGs) are incomplete, which motivates one important research topic on automatically complementing knowledge graphs. However, evaluation of knowledge graph completion (KGC) models often ignores the incompleteness---facts in the test set are ranked against all unknown triplets which may contain a large number of missing facts not included in the KG yet. Treating all unknown triplets as false is called the closed-world assumption. This closed-world assumption might negatively affect the fairness and consistency of the evaluation metrics. In this paper, we study KGC evaluation under a more realistic setting, namely the open-world assumption, where unknown triplets are considered to include many missing facts not included in the training or test sets. For the currently most used metrics such as mean reciprocal rank (MRR) and Hits@K, we point out that their behavior may be unexpected under the open-world assumption. Specifically, with not many missing facts, their numbers show a logarithmic trend with respect to the true strength of the model, and thus, the metric increase could be insignificant in terms of reflecting the true model improvement. Further, considering the variance, we show that the degradation in the reported numbers may result in incorrect comparisons between different models, where stronger models may have lower metric numbers. We validate the phenomenon both theoretically and experimentally. Finally, we suggest possible causes and solutions for this problem. Our code and data are available at https://github.com/GraphPKU/Open-World-KG .

        ----

        ## [609] KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation

        **Authors**: *Ta-Chung Chi, Ting-Han Fan, Peter J. Ramadge, Alexander Rudnicky*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/37a413841a614b5414b333585e7613b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/37a413841a614b5414b333585e7613b8-Abstract-Conference.html)

        **Abstract**:

        Relative positional embeddings (RPE) have received considerable attention since RPEs effectively model the relative distance among tokens and enable length extrapolation. We propose KERPLE, a framework that generalizes relative position embedding for extrapolation by kernelizing positional differences. We achieve this goal using conditionally positive definite (CPD) kernels, a class of functions known for generalizing distance metrics. To maintain the inner product interpretation of self-attention, we show that a CPD kernel can be transformed into a PD kernel by adding a constant offset. This offset is implicitly absorbed in the Softmax normalization during self-attention. The diversity of CPD kernels allows us to derive various RPEs that enable length extrapolation in a principled way. Experiments demonstrate that the logarithmic variant achieves excellent extrapolation performance on three large language modeling datasets. Our implementation and pretrained checkpoints are released at~\url{https://github.com/chijames/KERPLE.git}.

        ----

        ## [610] Zonotope Domains for Lagrangian Neural Network Verification

        **Authors**: *Matt Jordan, Jonathan Hayase, Alex Dimakis, Sewoong Oh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/37c5e5e5e44950fe20f9e6452d0373ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/37c5e5e5e44950fe20f9e6452d0373ec-Abstract-Conference.html)

        **Abstract**:

        Neural network verification aims to provide provable bounds for the output of a neural network for a given input range. Notable prior works in this domain have either generated bounds using abstract domains, which preserve some dependency between intermediate neurons in the network; or framed verification as an optimization problem and solved a relaxation using Lagrangian methods. A key drawback of the latter technique is that each neuron is treated independently, thereby ignoring important neuron interactions. We provide an approach that merges these two threads and uses zonotopes within a Lagrangian decomposition. Crucially, we can decompose the problem of verifying a deep neural network into the verification of many 2-layer neural networks. While each of these problems is provably hard, we provide efficient relaxation methods that are amenable to efficient dual ascent procedures. Our technique yields bounds that improve upon both linear programming and Lagrangian-based verification techniques in both time and bound tightness.

        ----

        ## [611] Neural Basis Models for Interpretability

        **Authors**: *Filip Radenovic, Abhimanyu Dubey, Dhruv Mahajan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/37da88965c016dca016514df0e420c72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/37da88965c016dca016514df0e420c72-Abstract-Conference.html)

        **Abstract**:

        Due to the widespread use of complex machine learning models in real-world applications, it is becoming critical to explain model predictions. However, these models are typically black-box deep neural networks, explained post-hoc via methods with known faithfulness limitations. Generalized Additive Models (GAMs) are an inherently interpretable class of models that address this limitation by learning a non-linear shape function for each feature separately, followed by a linear model on top. However, these models are typically difficult to train, require numerous parameters, and are difficult to scale.     We propose an entirely new subfamily of GAMs that utilizes basis decomposition of shape functions. A small number of basis functions are shared among all features, and are learned jointly for a given task, thus making our model scale much better to large-scale data with high-dimensional features, especially when features are sparse. We propose an architecture denoted as the Neural Basis Model (NBM) which uses a single neural network to learn these bases. On a variety of tabular and image datasets, we demonstrate that for interpretable machine learning, NBMs are the state-of-the-art in accuracy, model size, and, throughput and can easily model all higher-order feature interactions.    Source code is available at \href{https://github.com/facebookresearch/nbm-spam}{\ttfamily github.com/facebookresearch/nbm-spam}.

        ----

        ## [612] RecursiveMix: Mixed Learning with History

        **Authors**: *Lingfeng Yang, Xiang Li, Borui Zhao, Renjie Song, Jian Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/37e44c4b5321605735be9761f9b758fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/37e44c4b5321605735be9761f9b758fc-Abstract-Conference.html)

        **Abstract**:

        Mix-based augmentation has been proven fundamental to the generalization of deep vision models. However, current augmentations only mix samples from the current data batch during training, which ignores the possible knowledge accumulated in the learning history. In this paper, we propose a recursive mixed-sample learning paradigm, termed ``RecursiveMix'' (RM), by exploring a novel training strategy that leverages the historical input-prediction-label triplets. More specifically, we iteratively resize the input image batch from the previous iteration and paste it into the current batch while their labels are fused proportionally to the area of the operated patches. Furthermore, a consistency loss is introduced to align the identical image semantics across the iterations, which helps the learning of scale-invariant feature representations. Based on ResNet-50, RM largely improves classification accuracy by $\sim$3.2% on CIFAR-100 and $\sim$2.8% on ImageNet with negligible extra computation/storage costs. In the downstream object detection task, the RM-pretrained model outperforms the baseline by 2.1 AP points and surpasses CutMix by 1.4 AP points under the ATSS detector on COCO. In semantic segmentation, RM also surpasses the baseline and CutMix by 1.9 and 1.1 mIoU points under UperNet on ADE20K, respectively. Codes and pretrained models are available at https://github.com/implus/RecursiveMix.

        ----

        ## [613] GAR: Generalized Autoregression for Multi-Fidelity Fusion

        **Authors**: *Yuxin Wang, Zheng Xing, Wei W. Xing*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html)

        **Abstract**:

        In many scientiﬁc research and engineering applications, where repeated simulations of complex systems are conducted, a surrogate is commonly adopted to quickly estimate the whole system. To reduce the expensive cost of generating training examples, it has become a promising approach to combine the results of low-ﬁdelity (fast but inaccurate) and high-ﬁdelity (slow but accurate) simulations. Despite the fast developments of multi-ﬁdelity fusion techniques, most existing methods require particular data structures and do not scale well to high-dimensional output. To resolve these issues, we generalize the classic autoregression (AR), which is wildly used due to its simplicity, robustness, accuracy, and tractability, and propose generalized autoregression (GAR) using tensor formulation and latent features. GAR can deal with arbitrary dimensional outputs and arbitrary multiﬁdelity data structure to satisfy the demand of multi-ﬁdelity fusion for complex problems; it admits a fully tractable likelihood and posterior requiring no approximate inference and scales well to high-dimensional problems. Furthermore, we prove the autokrigeability theorem based on GAR in the multi-ﬁdelity case and develop CIGAR, a simpliﬁed GAR with the same predictive mean accuracy but requires signiﬁcantly less computation. In experiments of canonical PDEs and scientiﬁc computational examples, the proposed method consistently outperforms the SOTA methods with a large margin (up to 6x improvement in RMSE) with only a few high-ﬁdelity training samples.

        ----

        ## [614] Anonymized Histograms in Intermediate Privacy Models

        **Authors**: *Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/380afe1a245a3b2134010620eae88865-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/380afe1a245a3b2134010620eae88865-Abstract-Conference.html)

        **Abstract**:

        We study the problem of  privately computing the $\mbox{\it anonymized histogram}$ (a.k.a. $\mbox{\it unattributed histogram}$), which is defined as the histogram without item labels. Previous works have provided algorithms with $\ell_1$- and $\ell_2^2$-errors of $O_\varepsilon(\sqrt{n})$ in the central model of differential privacy (DP).In this work, we provide an algorithm with a nearly matching error guarantee of $\widetilde{O}_\varepsilon(\sqrt{n})$ in the shuffle DP and pan-private models. Our algorithm is very simple: it just post-processes the discrete Laplace-noised histogram!  Using this algorithm as a subroutine, we show applications in privately estimating symmetric properties of distributions such as entropy, support coverage, and support size.

        ----

        ## [615] Truly Deterministic Policy Optimization

        **Authors**: *Ehsan Saleh, Saba Ghaffari, Timothy Bretl, Matthew West*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3819dd04c2c87bf0d1deea1740ef0ad5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3819dd04c2c87bf0d1deea1740ef0ad5-Abstract-Conference.html)

        **Abstract**:

        In this paper, we present a policy gradient method that avoids exploratory noise injection and performs policy search over the deterministic landscape, with the goal of improving learning with long horizons and non-local rewards. By avoiding noise injection all sources of estimation variance can be eliminated in systems with deterministic dynamics (up to the initial state distribution). Since deterministic policy regularization is impossible using traditional non-metric measures such as the KL divergence, we derive a Wasserstein-based quadratic model for our purposes. We state conditions on the system model under which it is possible to establish a monotonic policy improvement guarantee, propose a surrogate function for policy gradient estimation, and show that it is possible to compute exact advantage estimates if both the state transition model and the policy are deterministic. Finally, we describe two novel robotic control environments---one with non-local rewards in the frequency domain and the other with a long horizon (8000 time-steps)---for which our policy gradient method (TDPO) significantly outperforms existing methods (PPO, TRPO, DDPG, and TD3). Our implementation with all the experimental settings and a video of the physical hardware test is available at https://github.com/ehsansaleh/tdpo .

        ----

        ## [616] Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners

        **Authors**: *Zhenhailong Wang, Manling Li, Ruochen Xu, Luowei Zhou, Jie Lei, Xudong Lin, Shuohang Wang, Ziyi Yang, Chenguang Zhu, Derek Hoiem, Shih-Fu Chang, Mohit Bansal, Heng Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/381ceeae4a1feb1abc59c773f7e61839-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/381ceeae4a1feb1abc59c773f7e61839-Abstract-Conference.html)

        **Abstract**:

        The goal of this work is to build flexible video-language models that can generalize to various video-to-text tasks from few examples. Existing few-shot video-language learners focus exclusively on the encoder, resulting in the absence of a video-to-text decoder to handle generative tasks. Video captioners have been pretrained on large-scale video-language datasets, but they rely heavily on finetuning and lack the ability to generate text for unseen tasks in a few-shot setting. We propose VidIL, a few-shot Video-language Learner via Image and Language models, which demonstrates strong performance on few-shot video-to-text tasks without the necessity of pretraining or finetuning on any video datasets. We use image-language models to translate the video content into frame captions, object, attribute, and event phrases, and compose them into a temporal-aware template.  We then instruct a language model, with a prompt containing a few in-context examples, to generate a target output from the composed content. The flexibility of prompting allows the model to capture any form of text input, such as automatic speech recognition (ASR) transcripts. Our experiments demonstrate the power of language models in understanding videos on a wide variety of video-language tasks, including video captioning, video question answering, video caption retrieval, and video future event prediction. Especially, on video future event prediction, our few-shot model significantly outperforms state-of-the-art supervised models trained on large-scale video datasets.Code and processed data are publicly available for research purposes at https://github.com/MikeWangWZHL/VidIL.

        ----

        ## [617] 3DB: A Framework for Debugging Computer Vision Models

        **Authors**: *Guillaume Leclerc, Hadi Salman, Andrew Ilyas, Sai Vemprala, Logan Engstrom, Vibhav Vineet, Kai Yuanqing Xiao, Pengchuan Zhang, Shibani Santurkar, Greg Yang, Ashish Kapoor, Aleksander Madry*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3848bc3112429079af85dedb7d369ef4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3848bc3112429079af85dedb7d369ef4-Abstract-Conference.html)

        **Abstract**:

        We introduce 3DB: an extendable, unified framework for testing and debugging vision models using photorealistic simulation.  We demonstrate, through a wide range of use cases, that 3DB allows users to discover vulnerabilities in computer vision systems and gain insights into how models make decisions. 3DB captures and generalizes many robustness analyses from prior work, and enables one to study their interplay. Finally, we find that the insights generated by the system transfer to the physical world. 3DB will be released as a library alongside a set of examples and documentation. We attach 3DB to the submission.

        ----

        ## [618] Empirical Gateaux Derivatives for Causal Inference

        **Authors**: *Michael I. Jordan, Yixin Wang, Angela Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3848fef259495bfd04d60cdc5c1b4db7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3848fef259495bfd04d60cdc5c1b4db7-Abstract-Conference.html)

        **Abstract**:

        We study a constructive procedure that approximates Gateaux derivatives for statistical functionals by finite-differencing, with attention to causal inference functionals. We focus on the case where probability distributions are not known a priori but need also to be estimated from data, leading to empirical Gateaux derivatives, and study relationships between empirical, numerical, and analytical Gateaux derivatives. Starting with a case study of counterfactual mean estimation, we verify the exact relationship between finite-differences and the analytical Gateaux derivative. We then derive requirements on the rates of numerical approximation in perturbation and smoothing that preserve statistical benefits. We study more complicated functionals such as dynamic treatment regimes and the linear-programming formulation for policy optimization infinite-horizon Markov decision processes. In the case of the latter, this approach can be used to approximate bias adjustments in the presence of arbitrary constraints, illustrating the usefulness of constructive approaches for Gateaux derivatives. We find that, omitting unfavorable dimension dependence of smoothing, although rate-double robustness permits for coarser rates of perturbation size than implied by generic approximation analysis of finite-differences for the case of the counterfactual mean, this is not the case for the infinite-horizon MDP policy value.

        ----

        ## [619] Calibrated Data-Dependent Constraints with Exact Satisfaction Guarantees

        **Authors**: *Songkai Xue, Yuekai Sun, Mikhail Yurochkin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/38c5feed4b72c96f6cf925ccc9832ecf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/38c5feed4b72c96f6cf925ccc9832ecf-Abstract-Conference.html)

        **Abstract**:

        We consider the task of training machine learning models with data-dependent constraints. Such constraints often arise as empirical versions of expected value constraints that enforce fairness or stability goals. We reformulate data-dependent constraints so that they are calibrated: enforcing the reformulated constraints guarantees that their expected value counterparts are satisfied with a user-prescribed probability. The resulting optimization problem is amendable to standard stochastic optimization algorithms, and we demonstrate the efficacy of our method on a fairness-sensitive classification task where we wish to guarantee the classifier's fairness (at test time).

        ----

        ## [620] Towards Consistency in Adversarial Classification

        **Authors**: *Laurent Meunier, Raphael Ettedgui, Rafael Pinot, Yann Chevaleyre, Jamal Atif*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/38d6af46cca4ce1f7d699bf11078cb84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/38d6af46cca4ce1f7d699bf11078cb84-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the problem of consistency in the context of adversarial examples. Specifically, we tackle the following question: can surrogate losses still be used as a proxy for minimizing the $0/1$ loss in the presence of an adversary that alters the inputs at test-time? Different from the standard classification task, this question cannot be reduced to a point-wise minimization problem, and calibration needs not to be sufficient to ensure consistency. In this paper, we expose some pathological behaviors specific to the adversarial problem, and show that no convex surrogate loss can be consistent or calibrated in this context. It is therefore necessary to design another class of surrogate functions that can be used to solve the adversarial consistency issue. As a first step towards designing such a class, we identify sufficient and necessary conditions for a surrogate loss to be calibrated in both the adversarial and standard settings. Finally, we give some directions for building a class of losses that could be consistent in the adversarial framework.

        ----

        ## [621] Residual Multiplicative Filter Networks for Multiscale Reconstruction

        **Authors**: *Shayan Shekarforoush, David B. Lindell, David J. Fleet, Marcus A. Brubaker*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/38e491559eb9e4cf31b8cd3a4e222436-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/38e491559eb9e4cf31b8cd3a4e222436-Abstract-Conference.html)

        **Abstract**:

        Coordinate networks like Multiplicative Filter Networks (MFNs) and BACON offer some control over the frequency spectrum used to represent continuous signals such as images or 3D volumes. Yet, they are not readily applicable to problems for which coarse-to-fine estimation is required, including various inverse problems in which coarse-to-fine optimization plays a key role in avoiding poor local minima. We introduce a new coordinate network architecture and training scheme that enables coarse-to-fine optimization with fine-grained control over the frequency support of learned reconstructions. This is achieved with two key innovations. First, we incorporate skip connections so that structure at one scale is preserved when fitting finer-scale structure. Second, we propose a novel initialization scheme to provide control over the model frequency spectrum at each stage of optimization. We demonstrate how these modifications enable multiscale optimization for coarse-to-fine fitting to natural images. We then evaluate our model on synthetically generated datasets for the the problem of single-particle cryo-EM reconstruction. We learn high resolution multiscale structures, on par with the state-of-the art. Project webpage: https://shekshaa.github.io/ResidualMFN/.

        ----

        ## [622] WT-MVSNet: Window-based Transformers for Multi-view Stereo

        **Authors**: *Jinli Liao, Yikang Ding, Yoli Shavit, Dihe Huang, Shihao Ren, Jia Guo, Wensen Feng, Kai Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/38e511a690709603d4cc3a1c52b4a9fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/38e511a690709603d4cc3a1c52b4a9fd-Abstract-Conference.html)

        **Abstract**:

        Recently, Transformers have been shown to enhance the performance of multi-view stereo by enabling long-range feature interaction. In this work, we propose Window-based Transformers (WT) for local feature matching and global feature aggregation in multi-view stereo. We introduce a Window-based Epipolar Transformer (WET) which reduces matching redundancy by using epipolar constraints. Since point-to-line matching is sensitive to erroneous camera pose and calibration, we match windows near the epipolar lines. A second Shifted WT is employed for aggregating global information within cost volume. We present a novel Cost Transformer (CT) to replace 3D convolutions for cost volume regularization. In order to better constrain the estimated depth maps from multiple views, we further design a novel geometric consistency loss (Geo Loss) which punishes unreliable areas where multi-view consistency is not satisfied. Our WT multi-view stereo method (WT-MVSNet) achieves state-of-the-art performance across multiple datasets and ranks $1^{st}$ on Tanks and Temples benchmark. Code will be available upon acceptance.

        ----

        ## [623] Private Isotonic Regression

        **Authors**: *Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/38fd51cf36f28566230a93a5fbeaabbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/38fd51cf36f28566230a93a5fbeaabbf-Abstract-Conference.html)

        **Abstract**:

        In this paper, we consider the problem of differentially private (DP) algorithms for isotonic regression.  For the most general problem of isotonic regression over a partially ordered set (poset)  $\mathcal{X}$ and for any Lipschitz loss function, we obtain a pure-DP algorithm that, given $n$ input points, has an expected excess empirical risk of roughly $\mathrm{width}(\mathcal{X}) \cdot \log|\mathcal{X}| / n$, where $\mathrm{width}(\mathcal{X})$ is the width of the poset.  In contrast, we also obtain a near-matching lower bound of roughly $(\mathrm{width}(\mathcal{X}) + \log |\mathcal{X}|) / n$, that holds even for approximate-DP algorithms. Moreover, we show that the above bounds are essentially the best that can be obtained without utilizing any further structure of the poset.In the special case of a totally ordered set and for $\ell_1$ and $\ell_2^2$ losses, our algorithm can be implemented in near-linear running time; we also provide extensions of this algorithm to the problem of private isotonic regression with additional structural constraints on the output function.

        ----

        ## [624] A Closer Look at Offline RL Agents

        **Authors**: *Yuwei Fu, Di Wu, Benoit Boulet*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3908cadfcc99db12001eafb1207353e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3908cadfcc99db12001eafb1207353e9-Abstract-Conference.html)

        **Abstract**:

        Despite recent advances in the field of Offline Reinforcement Learning (RL), less attention has been paid to understanding the behaviors of learned RL agents. As a result, there remain some gaps in our understandings, i.e., why is one offline RL agent more performant than another? In this work, we first introduce a set of experiments to evaluate offline RL agents, focusing on three fundamental aspects: representations, value functions and policies. Counterintuitively, we show that a more performant offline RL agent can learn relatively low-quality representations and inaccurate value functions. Furthermore, we showcase that the proposed experiment setups can be effectively used to diagnose the bottleneck of offline RL agents. Inspired by the evaluation results, a novel offline RL algorithm is proposed by a simple modification of IQL and achieves SOTA performance. Finally, we investigate when a learned dynamics model is helpful to model-free offline RL agents, and introduce an uncertainty-based sample selection method to mitigate the problem of model noises. Code is available at: https://github.com/fuyw/RIQL.

        ----

        ## [625] Generalization Bounds for Estimating Causal Effects of Continuous Treatments

        **Authors**: *Xin Wang, Shengfei Lyu, Xingyu Wu, Tianhao Wu, Huanhuan Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/390bb66a088d37f62ee9fb779c5953c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/390bb66a088d37f62ee9fb779c5953c2-Abstract-Conference.html)

        **Abstract**:

        We focus on estimating causal effects of continuous treatments (e.g., dosage in medicine), also known as dose-response function. Existing methods in causal inference for continuous treatments using neural networks are effective and to some extent reduce selection bias, which is introduced by non-randomized treatments among individuals and might lead to covariate imbalance and thus unreliable inference. To theoretically support the alleviation of selection bias in the setting of continuous treatments, we exploit the re-weighting schema and the Integral Probability Metric (IPM) distance to derive an upper bound on the counterfactual loss of estimating the average dose-response function (ADRF), and herein the IPM distance builds a bridge from a source (factual) domain to an infinite number of target (counterfactual) domains. We provide a discretized approximation of the IPM distance with a theoretical guarantee in the practical implementation. Based on the theoretical analyses, we also propose a novel algorithm, called Average Dose- response estiMatIon via re-weighTing schema (ADMIT). ADMIT simultaneously learns a re-weighting network, which aims to alleviate the selection bias, and an inference network, which makes factual and counterfactual estimations. In addition, the effectiveness of ADMIT is empirically demonstrated in both synthetic and semi-synthetic experiments by outperforming the existing benchmarks.

        ----

        ## [626] Better Uncertainty Calibration via Proper Scores for Classification and Beyond

        **Authors**: *Sebastian G. Gruber, Florian Buettner*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3915a87ddac8e8c2f23dbabbcee6eec9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3915a87ddac8e8c2f23dbabbcee6eec9-Abstract-Conference.html)

        **Abstract**:

        With model trustworthiness being crucial for sensitive real-world applications, practitioners are putting more and more focus on improving the uncertainty calibration of deep neural networks.Calibration errors are designed to quantify the reliability of probabilistic predictions but their estimators are usually biased and inconsistent.In this work, we introduce the framework of \textit{proper calibration errors}, which relates every calibration error to a proper score and provides a respective upper bound with optimal estimation properties.This relationship can be used to reliably quantify the model calibration improvement.We theoretically and empirically demonstrate the shortcomings of commonly used estimators compared to our approach.Due to the wide applicability of proper scores, this gives a natural extension of recalibration beyond classification.

        ----

        ## [627] Video Diffusion Models

        **Authors**: *Jonathan Ho, Tim Salimans, Alexey A. Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html)

        **Abstract**:

        Generating temporally coherent high fidelity video is an important milestone in generative modeling research. We make progress towards this milestone by proposing a diffusion model for video generation that shows very promising initial results. Our model is a natural extension of the standard image diffusion architecture, and it enables jointly training from image and video data, which we find to reduce the variance of minibatch gradients and speed up optimization. To generate long and higher resolution videos we introduce a new conditional sampling technique for spatial and temporal video extension that performs better than previously proposed methods. We present the first results on a large text-conditioned video generation task, as well as state-of-the-art results on established benchmarks for video prediction and unconditional video generation. Supplementary material is available at https://video-diffusion.github.io/.

        ----

        ## [628] Formulating Robustness Against Unforeseen Attacks

        **Authors**: *Sihui Dai, Saeed Mahloujifar, Prateek Mittal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/392ac56724c133c37d5ea746e52f921f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/392ac56724c133c37d5ea746e52f921f-Abstract-Conference.html)

        **Abstract**:

        Existing defenses against adversarial examples such as adversarial training typically assume that the adversary will conform to a specific or known threat model, such as $\ell_p$ perturbations within a fixed budget. In this paper, we focus on the scenario where there is a mismatch in the threat model assumed by the defense during training, and the actual capabilities of the adversary at test time. We ask the question: if the learner trains against a specific ``source" threat model, when can we expect robustness to generalize to a stronger unknown ``target" threat model during test-time? Our key contribution is to formally define the problem of learning and generalization with an unforeseen adversary, which helps us reason about the increase in adversarial risk from the conventional perspective of a known adversary. Applying our framework, we derive a generalization bound which relates the generalization gap between source and target threat models to variation of the feature extractor, which measures the expected maximum difference between extracted features across a given threat model. Based on our generalization bound, we propose variation regularization (VR) which reduces variation of the feature extractor across the source threat model during training. We empirically demonstrate that using VR can lead to improved generalization to unforeseen attacks during test-time, and combining VR with perceptual adversarial training (Laidlaw et al., 2021) achieves state-of-the-art robustness on unforeseen attacks. Our code is publicly available at https://github.com/inspire-group/variation-regularization.

        ----

        ## [629] Single Model Uncertainty Estimation via Stochastic Data Centering

        **Authors**: *Jayaraman J. Thiagarajan, Rushil Anirudh, Vivek Sivaraman Narayanaswamy, Timo Bremer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html)

        **Abstract**:

        We are interested in estimating the uncertainties of deep neural networks, which play an important role in many scientific and engineering problems. In this paper, we present a striking new finding that an ensemble of neural networks with the same weight initialization, trained on datasets that are shifted by a constant bias gives rise to slightly inconsistent trained models, where the differences in predictions are a strong indicator of epistemic uncertainties. Using the neural tangent kernel (NTK), we demonstrate that this phenomena occurs in part because the NTK is not shift-invariant. Since this is achieved via a trivial input transformation, we show that this behavior can therefore be approximated by training a single neural network -- using a technique that we call $\Delta-$UQ -- that estimates uncertainty around prediction by marginalizing out the effect of the biases during inference. We show that $\Delta-$UQ's uncertainty estimates are superior to many of the current methods on a variety of benchmarks-- outlier rejection, calibration under distribution shift, and sequential design optimization of black box functions. Code for $\Delta-$UQ can be accessed at github.com/LLNL/DeltaUQ

        ----

        ## [630] Autoinverse: Uncertainty Aware Inversion of Neural Networks

        **Authors**: *Navid Ansari, Hans-Peter Seidel, Nima Vahidi Ferdowsi, Vahid Babaei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3942d2f7fc0a5af1bf70dcbdcf4c56aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3942d2f7fc0a5af1bf70dcbdcf4c56aa-Abstract-Conference.html)

        **Abstract**:

        Neural networks are powerful surrogates for numerous forward processes.The inversion of such surrogates is extremely valuable in science and engineering. The most important property of a successful neural inverse method is the performance of its solutions when deployed in the real world, i.e., on the native forward process (and not only the learned surrogate). We propose Autoinverse, a highly automated approach for inverting neural network surrogates. Our main insight is to seek inverse solutions in the vicinity of reliable data which have been sampled form the forward process and used for training the surrogate model. Autoinverse finds such solutions by taking into account the predictive uncertainty of the surrogate and minimizing it during the inversion. Apart from high accuracy, Autoinverse enforces the feasibility of solutions, comes with embedded regularization, and is initialization free. We verify our proposed method through addressing a set of real-world problems in control, fabrication, and design.

        ----

        ## [631] FedPop: A Bayesian Approach for Personalised Federated Learning

        **Authors**: *Nikita Kotelevskii, Maxime Vono, Alain Durmus, Eric Moulines*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/395409679270591fd2a70abc694cf5a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/395409679270591fd2a70abc694cf5a1-Abstract-Conference.html)

        **Abstract**:

        Personalised federated learning (FL) aims at collaboratively learning a machine learning model tailored for each client. Albeit promising advances have been made in this direction, most of the existing approaches do not allow for uncertainty quantification which is crucial in many applications. In addition, personalisation in the cross-silo and cross-device setting still involves important issues, especially for new clients or those having a small number of observations. This paper aims at filling these gaps. To this end, we propose a novel methodology coined FedPop by recasting personalised FL into the population modeling paradigm where clientsâ€™ models involve fixed common population parameters and random effects, aiming at explaining data heterogeneity. To derive convergence guarantees for our scheme, we introduce a new class of federated stochastic optimisation algorithms that relies on Markov chain Monte Carlo methods. Compared to existing personalised FL methods, the proposed methodology has important benefits: it is robust to client drift, practical for inference on new clients, and above all, enables uncertainty quantification under mild computational and memory overheads. We provide nonasymptotic convergence guarantees for the proposed algorithms and illustrate their performances on various personalised federated learning tasks.

        ----

        ## [632] Semantic Diffusion Network for Semantic Segmentation

        **Authors**: *Haoru Tan, Sitong Wu, Jimin Pi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/396446770f5e8496ca1feb02079d4fb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/396446770f5e8496ca1feb02079d4fb7-Abstract-Conference.html)

        **Abstract**:

        Precise and accurate predictions over boundary areas are essential for semantic segmentation. However, the commonly used convolutional operators tend to smooth and blur local detail cues, making it difficult for deep models to generate accurate boundary predictions. In this paper, we introduce an operator-level approach to enhance semantic boundary awareness, so as to improve the prediction of the deep semantic segmentation model. Specifically, we formulate the boundary feature enhancement process as an anisotropic diffusion process. We propose a novel learnable approach called semantic diffusion network (SDN) for approximating the diffusion process, which contains a parameterized semantic difference convolution operator followed by a feature fusion module and constructs a differentiable mapping from original backbone features to advanced boundary-aware features. The proposed SDN is an efficient and flexible module that can be plugged into existing encoder-decoder segmentation models. Extensive experiments show that our approach can achieve consistent improvements over several typical state-of-the-art segmentation baseline models on challenging public benchmarks.

        ----

        ## [633] Iterative Structural Inference of Directed Graphs

        **Authors**: *Aoran Wang, Jun Pang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39717429762da92201a750dd03386920-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39717429762da92201a750dd03386920-Abstract-Conference.html)

        **Abstract**:

        In this paper, we propose a variational model, iterative Structural Inference of Directed Graphs (iSIDG), to infer the existence of directed interactions from observational agentsâ€™ features over a time period in a dynamical system. First, the iterative process in our model feeds the learned interactions back to encourage our model to eliminate indirect interactions and to emphasize directional representation during learning. Second, we show that extra regularization terms in the objective function for smoothness, connectiveness, and sparsity prompt our model to infer a more realistic structure and to further eliminate indirect interactions. We evaluate iSIDG on various datasets including biological networks, simulated fMRI data, and physical simulations to demonstrate that our model is able to precisely infer the existence of interactions, and is significantly superior to baseline models.

        ----

        ## [634] An efficient graph generative model for navigating ultra-large combinatorial synthesis libraries

        **Authors**: *Aryan Pedawi, Pawel Gniewek, Chaoyi Chang, Brandon M. Anderson, Henry van den Bedem*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39781da4b5d05bc2908ce08e43bc6404-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39781da4b5d05bc2908ce08e43bc6404-Abstract-Conference.html)

        **Abstract**:

        Virtual, make-on-demand chemical libraries have transformed early-stage drug discovery by unlocking vast, synthetically accessible regions of chemical space. Recent years have witnessed rapid growth in these libraries from millions to trillions of compounds, hiding undiscovered, potent hits for a variety of therapeutic targets. However, they are quickly approaching a size beyond that which permits explicit enumeration, presenting new challenges for virtual screening. To overcome these challenges, we propose the Combinatorial Synthesis Library Variational Auto-Encoder (CSLVAE). The proposed generative model represents such libraries as a differentiable, hierarchically-organized database. Given a compound from the library, the molecular encoder constructs a query for retrieval, which is utilized by the molecular decoder to reconstruct the compound by first decoding its chemical reaction and subsequently decoding its reactants. Our design minimizes autoregression in the decoder, facilitating the generation of large, valid molecular graphs. Our method performs fast and parallel batch inference for ultra-large synthesis libraries, enabling a number of important applications in early-stage drug discovery. Compounds proposed by our method are guaranteed to be in the library, and thus synthetically and cost-effectively accessible. Importantly, CSLVAE can encode out-of-library compounds and search for in-library analogues. In experiments, we demonstrate the capabilities of the proposed method in the navigation of massive combinatorial synthesis libraries.

        ----

        ## [635] Learning to Discover and Detect Objects

        **Authors**: *Vladimir Fomenko, Ismail Elezi, Deva Ramanan, Laura Leal-Taixé, Aljosa Osep*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39a636ecfecd89220aaf0fc79230e1f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39a636ecfecd89220aaf0fc79230e1f5-Abstract-Conference.html)

        **Abstract**:

        We tackle the problem of novel class discovery and localization (NCDL). In this setting, we assume a source dataset with supervision for only some object classes. Instances of other classes need to be discovered, classified, and localized automatically based on visual similarity without any human supervision. To tackle NCDL, we propose a two-stage object detection network Region-based NCDL (RNCDL) that uses a region proposal network to localize regions of interest (RoIs). We then train our network to learn to classify each RoI, either as one of the known classes, seen in the source dataset, or one of the novel classes, with a long-tail distribution constraint on the class assignments, reflecting the natural frequency of classes in the real world. By training our detection network with this objective in an end-to-end manner, it learns to classify all region proposals for a large variety of classes, including those not part of the labeled object class vocabulary. Our experiments conducted using COCO and LVIS datasets reveal that our method is significantly more effective than multi-stage pipelines that rely on traditional clustering algorithms. Furthermore, we demonstrate the generality of our approach by applying our method to a large-scale Visual Genome dataset, where our network successfully learns to detect various semantic classes without direct supervision.

        ----

        ## [636] Simulation-guided Beam Search for Neural Combinatorial Optimization

        **Authors**: *Jinho Choo, Yeong-Dae Kwon, Jihoon Kim, Jeongwoo Jae, André Hottung, Kevin Tierney, Youngjune Gwon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39b9b60f0d149eabd1fff2d7c7d5afc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39b9b60f0d149eabd1fff2d7c7d5afc4-Abstract-Conference.html)

        **Abstract**:

        Neural approaches for combinatorial optimization (CO) equip a learning mechanism to discover powerful heuristics for solving complex real-world problems. While neural approaches capable of high-quality solutions in a single shot are emerging, state-of-the-art approaches are often unable to take full advantage of the solving time available to them. In contrast, hand-crafted heuristics perform highly effective search well and exploit the computation time given to them, but contain heuristics that are difficult to adapt to a dataset being solved. With the goal of providing a powerful search procedure to neural CO approaches, we propose simulation-guided beam search (SGBS), which examines candidate solutions within a fixed-width tree search that both a neural net-learned policy and a simulation (rollout) identify as promising. We further hybridize SGBS with efficient active search (EAS), where SGBS enhances the quality of solutions backpropagated in EAS, and EAS improves the quality of the policy used in SGBS. We evaluate our methods on well-known CO benchmarks and show that SGBS significantly improves the quality of the solutions found under reasonable runtime assumptions.

        ----

        ## [637] FlowHMM: Flow-based continuous hidden Markov models

        **Authors**: *Pawel Lorek, Rafal Nowak, Tomasz Trzcinski, Maciej Zieba*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39c5871aa13be86ab978cba7069cbcec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39c5871aa13be86ab978cba7069cbcec-Abstract-Conference.html)

        **Abstract**:

        Continuous hidden Markov models (HMMs) assume that observations are generated from a mixture of Gaussian densities, limiting their ability to model more complex distributions. In this work, we address this shortcoming and propose  novel continuous HMM models, dubbed FlowHMMs, that enable learning general continuous observation densities without constraining them to follow a Gaussian distribution or their mixtures. To that end, we leverage deep flow-based architectures that model complex, non-Gaussian functions and propose two variants of training a FlowHMM model. The first one, based on gradient-based technique, can be applied directly to continuous multidimensional data, yet its application to larger data sequences remains computationally expensive. Therefore, we also present a second approach to training our FlowHMM that relies on the co-occurrence matrix of discretized observations and considers the joint distribution of pairs of co-observed values, hence rendering the training time independent of the training sequence length. As a result, we obtain a model that can be flexibly adapted to the characteristics and dimensionality of the data. We perform a variety of experiments in which we compare both training strategies with a baseline of Gaussian mixture models. We show, that in terms of quality of the recovered probability distribution, accuracy of prediction of hidden states, and likelihood of unseen data, our approach outperforms the standard Gaussian methods.

        ----

        ## [638] Near Instance-Optimal PAC Reinforcement Learning for Deterministic MDPs

        **Authors**: *Andrea Tirinzoni, Aymen Al Marjani, Emilie Kaufmann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39c60dda48ebf0a2e5dda52ce08eb5c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39c60dda48ebf0a2e5dda52ce08eb5c8-Abstract-Conference.html)

        **Abstract**:

        In probably approximately correct (PAC) reinforcement learning (RL), an agent is required to identify an $\epsilon$-optimal policy with probability $1-\delta$. While minimax optimal algorithms exist for this problem, its instance-dependent complexity remains elusive in episodic Markov decision processes (MDPs). In this paper, we propose the first nearly matching (up to a horizon squared factor and logarithmic terms) upper and lower bounds on the sample complexity of PAC RL in deterministic episodic MDPs with finite state and action spaces. In particular, our bounds feature a new notion of sub-optimality gap for state-action pairs that we call the deterministic return gap. While our instance-dependent lower bound is written as a linear program, our algorithms are very simple and do not require solving such an optimization problem during learning. Their design and analyses employ novel ideas, including graph-theoretical concepts (minimum flows) and a new maximum-coverage exploration strategy.

        ----

        ## [639] VICRegL: Self-Supervised Learning of Local Visual Features

        **Authors**: *Adrien Bardes, Jean Ponce, Yann LeCun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39cee562b91611c16ac0b100f0bc1ea1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39cee562b91611c16ac0b100f0bc1ea1-Abstract-Conference.html)

        **Abstract**:

        Most recent self-supervised methods for learning image representations focus on either producing a global feature with invariance properties, or producing a set of local features. The former works best for classification tasks while the latter is best for detection and segmentation tasks. This paper explores the fundamental trade-off between learning local and global features. A new method called VICRegL is proposed that learns good global and local features simultaneously, yielding excellent performance on detection and segmentation tasks while maintaining good performance on classification tasks. Concretely, two identical branches of a standard convolutional net architecture are fed two differently distorted versions of the same image. The VICReg criterion is applied to pairs of global feature vectors. Simultaneously, the VICReg criterion is applied to pairs of local feature vectors occurring before the last pooling layer. Two local feature vectors are attracted to each other if their l2-distance is below a threshold or if their relative locations are consistent with a known geometric transformation between the two input images. We demonstrate strong performance on linear classification and segmentation transfer tasks. Code and pretrained models are publicly available at: https://github.com/facebookresearch/VICRegL

        ----

        ## [640] Alleviating Adversarial Attacks on Variational Autoencoders with MCMC

        **Authors**: *Anna Kuzina, Max Welling, Jakub M. Tomczak*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)

        **Abstract**:

        Variational autoencoders (VAEs) are latent variable models that can generate complex objects and provide meaningful latent representations. Moreover, they could be further used in downstream tasks such as classification. As previous work has shown, one can easily fool VAEs to produce unexpected latent representations and reconstructions for a visually slightly modified input. Here, we examine several objective functions for adversarial attacks construction proposed previously and present a solution to alleviate the effect of these attacks. Our method utilizes the Markov Chain Monte Carlo (MCMC) technique in the inference step that we motivate with a theoretical analysis. Thus, we do not incorporate any extra costs during training and the performance on non-attacked inputs is not decreased. We validate our approach on a variety of datasets (MNIST, Fashion MNIST, Color MNIST, CelebA) and VAE configurations ($\beta$-VAE, NVAE, $\beta$-TCVAE), and show that our approach consistently improves the model robustness to adversarial attacks.

        ----

        ## [641] Policy Gradient With Serial Markov Chain Reasoning

        **Authors**: *Edoardo Cetin, Oya Çeliktutan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/39fac857b4467e3ef4f358186bb07d81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/39fac857b4467e3ef4f358186bb07d81-Abstract-Conference.html)

        **Abstract**:

        We introduce a new framework that performs decision-making in reinforcement learning (RL) as an iterative reasoning process. We model agent behavior as the steady-state distribution of a parameterized reasoning Markov chain (RMC), optimized with a new tractable estimate of the policy gradient. We perform action selection by simulating the RMC for enough reasoning steps to approach its steady-state distribution. We show our framework has several useful properties that are inherently missing from traditional RL. For instance, it allows agent behavior to approximate any continuous distribution over actions by parameterizing the RMC with a simple Gaussian transition function. Moreover, the number of reasoning steps to reach convergence can scale adaptively with the difficulty of each action selection decision and can be accelerated by re-using past solutions. Our resulting algorithm achieves state-of-the-art performance in popular Mujoco and DeepMind Control benchmarks, both for proprioceptive and pixel-based tasks.

        ----

        ## [642] DTG-SSOD: Dense Teacher Guidance for Semi-Supervised Object Detection

        **Authors**: *Gang Li, Xiang Li, Yujie Wang, Yichao Wu, Ding Liang, Shanshan Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3a02b6df276223b68c69ca572cb3c4a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3a02b6df276223b68c69ca572cb3c4a8-Abstract-Conference.html)

        **Abstract**:

        The Mean-Teacher (MT) scheme is widely adopted in semi-supervised object detection (SSOD). In MT, sparse pseudo labels, offered by the final predictions of the teacher (e.g., after Non Maximum Suppression (NMS) post-processing), are adopted for the dense supervision for the student via hand-crafted label assignment. However, the "sparse-to-dense'' paradigm complicates the pipeline of SSOD, and simultaneously neglects the powerful direct, dense teacher supervision. In this paper, we attempt to directly leverage the dense guidance of teacher to supervise student training, i.e., the "dense-to-dense'' paradigm. Specifically, we propose the Inverse NMS Clustering (INC) and Rank Matching (RM) to instantiate the dense supervision, without the widely used, conventional sparse pseudo labels. INC leads the student to group candidate boxes into clusters in NMS as the teacher does, which is implemented by learning grouping information revealed in NMS procedure of the teacher. After obtaining the same grouping scheme as the teacher via INC, the student further imitates the rank distribution of the teacher over clustered candidates through Rank Matching. With the proposed INC and RM, we integrate Dense Teacher Guidance into Semi-Supervised Object Detection (termed "DTG-SSOD''), successfully abandoning sparse pseudo labels and enabling more informative learning on unlabeled data. On COCO benchmark, our DTG-SSOD achieves state-of-the-art performance under various labelling ratios. For example, under 10% labelling ratio, DTG-SSOD improves the supervised baseline from 26.9 to 35.9 mAP, outperforming the previous best method Soft Teacher by 1.9 points.

        ----

        ## [643] Human-AI Shared Control via Policy Dissection

        **Authors**: *Quanyi Li, Zhenghao Peng, Haibin Wu, Lan Feng, Bolei Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3a02da3fdfd592d9a5273101c3546611-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3a02da3fdfd592d9a5273101c3546611-Abstract-Conference.html)

        **Abstract**:

        Human-AI shared control allows human to interact and collaborate with autonomous agents to accomplish control tasks in complex environments. Previous Reinforcement Learning (RL) methods attempted goal-conditioned designs to achieve human-controllable policies at the cost of redesigning the reward function and training paradigm. Inspired by the neuroscience approach to investigate the motor cortex in primates, we develop a simple yet effective frequency-based approach called Policy Dissection to align the intermediate representation of the learned neural controller with the kinematic attributes of the agent behavior. Without modifying the neural controller or retraining the model, the proposed approach can convert a given RL-trained policy into a human-controllable policy. We evaluate the proposed approach on many RL tasks such as autonomous driving and locomotion. The experiments show that human-AI shared control system achieved by Policy Dissection in driving task can substantially improve the performance and safety in unseen traffic scenes. With human in the inference loop, the locomotion robots also exhibit versatile controllable motion skills even though they are only trained to move forward. Our results suggest the promising direction of implementing human-AI shared autonomy through interpreting the learned representation of the autonomous agents. Code and demo videos are available at https://metadriverse.github.io/policydissect

        ----

        ## [644] Improving Generative Adversarial Networks via Adversarial Learning in Latent Space

        **Authors**: *Yang Li, Yichuan Mo, Liangliang Shi, Junchi Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3a1fc7b8e200a45110872c56f0569f61-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3a1fc7b8e200a45110872c56f0569f61-Abstract-Conference.html)

        **Abstract**:

        For Generative Adversarial Networks which map a latent distribution to the target distribution, in this paper, we study how the sampling in latent space can affect the generation performance, especially for images. We observe that, as the neural generator is a continuous function, two close samples in latent space would be mapped into two nearby images, while their quality can differ much as the quality generally does not exhibit a continuous nature in pixel space. From such a continuous mapping function perspective, it is also possible that two distant latent samples can be mapped into two close images (if not exactly the same). In particular, if the latent samples are mapped in aggregation into a single mode, mode collapse occurs. Accordingly, we propose adding an implicit latent transform before the mapping function to improve latent $z$ from its initial distribution, e.g., Gaussian. This is achieved using well-developed adversarial sample mining techniques, e.g. iterative fast gradient sign method (I-FGSM). We further propose new GAN training pipelines to obtain better generative mappings w.r.t quality and diversity by introducing targeted latent transforms into the bi-level optimization of GAN. Experimental results on visual data show that our method can effectively achieve improvement in both quality and diversity.

        ----

        ## [645] ShapeCrafter: A Recursive Text-Conditioned 3D Shape Generation Model

        **Authors**: *Rao Fu, Xiao Zhan, Yiwen Chen, Daniel Ritchie, Srinath Sridhar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3a33ae4d634b49b0866b4142a1f82a2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3a33ae4d634b49b0866b4142a1f82a2f-Abstract-Conference.html)

        **Abstract**:

        We present ShapeCrafter, a neural network for recursive text-conditioned 3D shape generation. Existing methods to generate text-conditioned 3D shapes consume an entire text prompt to generate a 3D shape in a single step. However, humans tend to describe shapes recursively---we may start with an initial description and progressively add details based on intermediate results. To capture this recursive process, we introduce a method to generate a 3D shape distribution, conditioned on an initial phrase, that gradually evolves as more phrases are added. Since existing datasets are insufficient for training this approach, we present Text2Shape++, a large dataset of 369K shape--text pairs that supports recursive shape generation. To capture local details that are often used to refine shape descriptions, we build on top of vector-quantized deep implicit functions that generate a distribution of high-quality shapes. Results show that our method can generate shapes consistent with text descriptions, and shapes evolve gradually as more phrases are added. Our method supports shape editing, extrapolation, and can enable new applications in human--machine collaboration for creative design.

        ----

        ## [646] SoundSpaces 20: A Simulation Platform for Visual-Acoustic Learning

        **Authors**: *Changan Chen, Carl Schissler, Sanchit Garg, Philip Kobernik, Alexander Clegg, Paul Calamia, Dhruv Batra, Philip W. Robinson, Kristen Grauman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3a48b0eaba26ba862220a307a9edb0bb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/3a48b0eaba26ba862220a307a9edb0bb-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce SoundSpaces 2.0, a platform for on-the-fly geometry-based audio rendering for 3D environments. Given a 3D mesh of a real-world environment, SoundSpaces can generate highly realistic acoustics for arbitrary sounds captured from arbitrary microphone locations. Together with existing 3D visual assets, it supports an array of audio-visual research tasks, such as audio-visual navigation, mapping, source localization and separation, and acoustic matching. Compared to existing resources, SoundSpaces 2.0 has the advantages of allowing continuous spatial sampling, generalization to novel environments, and configurable microphone and material properties. To our knowledge, this is the first geometry-based acoustic simulation that offers high fidelity and realism while also being fast enough to use for embodied learning. We showcase the simulator's properties and  benchmark its performance against real-world audio measurements. In addition, we demonstrate two downstream tasks---embodied navigation and far-field automatic speech recognition---and highlight sim2real performance for the latter. SoundSpaces 2.0 is publicly available to facilitate wider research for perceptual systems that can both see and hear.

        ----

        ## [647] AutoWS-Bench-101: Benchmarking Automated Weak Supervision with 100 Labels

        **Authors**: *Nicholas Roberts, Xintong Li, Tzu-Heng Huang, Dyah Adila, Spencer Schoenberg, Cheng-Yu Liu, Lauren Pick, Haotian Ma, Aws Albarghouthi, Frederic Sala*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3a54969b29a793de4e6b6d5a6062e494-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/3a54969b29a793de4e6b6d5a6062e494-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Weak supervision (WS) is a powerful method to build labeled datasets for training supervised models in the face of little-to-no labeled data. It replaces hand-labeling data with aggregating multiple noisy-but-cheap label estimates expressed by labeling functions (LFs). While it has been used successfully in many domains, weak supervision's application scope is limited by the difficulty of constructing labeling functions for domains with complex or high-dimensional features. To address this, a handful of methods have proposed automating the LF design process using a small set of ground truth labels. In this work, we introduce AutoWS-Bench-101: a framework for evaluating automated WS (AutoWS) techniques in challenging WS settings---a set of diverse application domains on which it has been previously difficult or impossible to apply traditional WS techniques. While AutoWS is a promising direction toward expanding the application-scope of WS, the emergence of powerful methods such as zero-shot foundation models reveal the need to understand how AutoWS techniques compare or cooperate with modern zero-shot or few-shot learners. This informs the central question of AutoWS-Bench-101: given an initial set of 100 labels for each task, we ask whether a practitioner should use an AutoWS method to generate additional labels or use some simpler baseline, such as zero-shot predictions from a foundation model or supervised learning. We observe that it is necessary for AutoWS methods to incorporate signal from foundation models if they are to outperform simple few-shot baselines, and AutoWS-Bench-101 promotes future research in this direction. We conclude with a thorough ablation study of AutoWS methods.

        ----

        ## [648] Towards Disentangling Information Paths with Coded ResNeXt

        **Authors**: *Apostolos Avranas, Marios Kountouris*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3a819a53408e20b75d1954bf617ccc0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3a819a53408e20b75d1954bf617ccc0a-Abstract-Conference.html)

        **Abstract**:

        The conventional, widely used treatment of deep learning models as black boxes provides limited or no insights into the mechanisms that guide neural network decisions. Significant research effort has been dedicated to building interpretable models to address this issue. Most efforts either focus on the high-level features associated with the last layers, or attempt to interpret the output of a single layer. In this paper, we take a novel approach to enhance the transparency of the function of the whole network. We propose a neural network architecture for classification, in which the information that is relevant to each class flows through specific paths. These paths are designed in advance before training leveraging coding theory and without depending on the semantic similarities between classes. A key property is that each path can be used as an autonomous single-purpose model. This enables us to obtain, without any additional training and for any class, a lightweight binary classifier that has at least $60\%$ fewer parameters than the original network. Furthermore, our coding theory based approach allows the neural network to make early predictions at intermediate layers during inference, without requiring its full evaluation. Remarkably, the proposed architecture provides all the aforementioned properties while improving the overall accuracy. We demonstrate these properties on a slightly modified ResNeXt model tested on CIFAR-10/100 and ImageNet-1k.

        ----

        ## [649] Time-Conditioned Dances with Simplicial Complexes: Zigzag Filtration Curve based Supra-Hodge Convolution Networks for Time-series Forecasting

        **Authors**: *Yuzhou Chen, Yulia R. Gel, H. Vincent Poor*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3a899fa79bc4110bca1eaa6649e9a8fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3a899fa79bc4110bca1eaa6649e9a8fa-Abstract-Conference.html)

        **Abstract**:

        Graph neural networks (GNNs) offer a new powerful alternative for multivariate time series forecasting, demonstrating remarkable success in a variety of spatio-temporal applications, from urban flow monitoring systems to health care informatics to financial analytics. Yet, such GNN models pre-dominantly capture only lower order interactions, that is, pairwise relations among nodes, and also largely ignore intrinsic time-conditioned information on the underlying topology of multivariate time series. To address these limitations, we propose a new time-aware GNN architecture which amplifies the power of the recently emerged simplicial neural networks with a time-conditioned topological knowledge representation in a form of zigzag persistence. That is, our new approach, Zigzag Filtration Curve based Supra-Hodge Convolution Networks (ZFC-SHCN) is built upon the two main components: (i) a new highly computationally efficientzigzag persistence curve which allows us to systematically encode time-conditioned topological information, and (ii) a new temporal multiplex graph representation module for learning higher-order network interactions. We discuss theoretical properties of the proposed time-conditioned topological knowledge representation and extensively validate the new time-aware ZFC-SHCN model in conjunction with time series forecasting on a broad range of synthetic and real-world datasets: traffic flows, COVID-19 biosurveillance, Ethereum blockchain, surface air temperature, wind energy, and vector autoregressions. Our experiments demonstrate that the ZFC-SHCN achieves the state-of-the-art performance with lower requirements on computational costs.

        ----

        ## [650] Are Defenses for Graph Neural Networks Robust?

        **Authors**: *Felix Mujkanovic, Simon Geisler, Stephan Günnemann, Aleksandar Bojchevski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ac904a31f9141444009777abef2ed8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ac904a31f9141444009777abef2ed8e-Abstract-Conference.html)

        **Abstract**:

        A cursory reading of the literature suggests that we have made a lot of progress in designing effective adversarial defenses for Graph Neural Networks (GNNs). Yet, the standard methodology has a serious flaw – virtually all of the defenses are evaluated against non-adaptive attacks leading to overly optimistic robustness estimates. We perform a thorough robustness analysis of 7 of the most popular defenses spanning the entire spectrum of strategies, i.e., aimed at improving the graph, the architecture, or the training. The results are sobering – most defenses show no or only marginal improvement compared to an undefended baseline. We advocate using custom adaptive attacks as a gold standard and we outline the lessons we learned from successfully designing such attacks. Moreover, our diverse collection of perturbed graphs forms a (black-box) unit test offering a first glance at a model's robustness.

        ----

        ## [651] GraB: Finding Provably Better Data Permutations than Random Reshuffling

        **Authors**: *Yucheng Lu, Wentao Guo, Christopher De Sa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3acb49252187efa352a1ae0e4b066ced-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3acb49252187efa352a1ae0e4b066ced-Abstract-Conference.html)

        **Abstract**:

        Random reshuffling, which randomly permutes the dataset each epoch, is widely adopted in model training because it yields faster convergence than with-replacement sampling. Recent studies indicate greedily chosen data orderings can further speed up convergence empirically, at the cost of using more computation and memory. However, greedy ordering lacks theoretical justification and has limited utility due to its non-trivial memory and computation overhead. In this paper, we first formulate an example-ordering framework named \emph{herding} and answer affirmatively that SGD with herding converges at the rate $O(T^{-2/3})$ on smooth, non-convex objectives, faster than the $O(n^{1/3}T^{-2/3})$ obtained by random reshuffling, where $n$ denotes the number of data points and $T$ denotes the total number of iterations. To reduce the memory overhead, we leverage discrepancy minimization theory to propose an online Gradient Balancing algorithm (GraB) that enjoys the same rate as herding, while reducing the memory usage from $O(nd)$ to just $O(d)$ and computation from $O(n^2)$ to $O(n)$, where $d$ denotes the model dimension. We show empirically on applications including MNIST, CIFAR10, WikiText and GLUE that GraB can outperform random reshuffling in terms of both training and validation performance, and even outperform state-of-the-art greedy ordering while reducing memory usage over $100\times$.

        ----

        ## [652] Amortized Proximal Optimization

        **Authors**: *Juhan Bae, Paul Vicol, Jeff Z. HaoChen, Roger B. Grosse*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3af25aa3de8b7b02ddbd1b6be5031be8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3af25aa3de8b7b02ddbd1b6be5031be8-Abstract-Conference.html)

        **Abstract**:

        We propose a framework for online meta-optimization of parameters that govern optimization, called Amortized Proximal Optimization (APO). We first interpret various existing neural network optimizers as approximate stochastic proximal point methods which trade off the current-batch loss with proximity terms in both function space and weight space. The idea behind APO is to amortize the minimization of the proximal point objective by meta-learning the parameters of an update rule. We show how APO can be used to adapt a learning rate or a structured preconditioning matrix. Under appropriate assumptions, APO can recover existing optimizers such as natural gradient descent and KFAC. It enjoys low computational overhead and avoids expensive and numerically sensitive operations required by some second-order optimizers, such as matrix inverses. We empirically test APO for online adaptation of learning rates and structured preconditioning matrices for regression, image reconstruction, image classification, and natural language translation tasks. Empirically, the learning rate schedules found by APO generally outperform optimal fixed learning rates and are competitive with manually tuned decay schedules. Using APO to adapt a structured preconditioning matrix generally results in optimization performance competitive with second-order methods. Moreover, the absence of matrix inversion provides numerical stability, making it effective for low-precision training.

        ----

        ## [653] A Statistical Online Inference Approach in Averaged Stochastic Approximation

        **Authors**: *Chuhan Xie, Zhihua Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3afaa2102fb8ea44cbadc13e45bba718-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3afaa2102fb8ea44cbadc13e45bba718-Abstract-Conference.html)

        **Abstract**:

        In this paper we propose a general framework to perform statistical online inference in a class of constant step size stochastic approximation (SA) problems, including the well-known stochastic gradient descent (SGD) and Q-learning. Regarding a constant step size SA procedure as a time-homogeneous Markov chain, we establish a functional central limit theorem (FCLT) for it under weaker conditions, and then construct confidence intervals for parameters via random scaling. To leverage the FCLT results in the Markov chain setting, an alternative condition that is more applicable for SA problems is established. We conduct experiments to perform inference with both random scaling and other traditional inference methods, and finds that the former has a more accurate and robust performance.

        ----

        ## [654] SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization

        **Authors**: *Chuanyang Zheng, Zheyang Li, Kai Zhang, Zhi Yang, Wenming Tan, Jun Xiao, Ye Ren, Shiliang Pu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3b11c5cc84b6da2838db348b37dbd1a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3b11c5cc84b6da2838db348b37dbd1a2-Abstract-Conference.html)

        **Abstract**:

        Vision Transformers (ViTs) yield impressive performance across various vision tasks. However, heavy computation and memory footprint make them inaccessible for edge devices. Previous works apply importance criteria determined independently by each individual component to prune ViTs. Considering that heterogeneous components in ViTs play distinct roles, these approaches lead to suboptimal performance. In this paper, we introduce joint importance, which integrates essential structural-aware interactions between components for the first time, to perform collaborative pruning. Based on the theoretical analysis, we construct a Taylor-based approximation to evaluate the joint importance. This guides pruning toward a more balanced reduction across all components. To further reduce the algorithm complexity, we incorporate the interactions into the optimization function under some mild assumptions. Moreover, the proposed method can be seamlessly applied to various tasks including object detection. Extensive experiments demonstrate the effectiveness of our method. Notably, the proposed approach outperforms the existing state-of-the-art approaches on ImageNet, increasing accuracy by 0.7% over the DeiT-Base baseline while saving 50% FLOPs. On COCO, we are the first to show that 70% FLOPs of FasterRCNN with ViT backbone can be removed with only 0.3% mAP drop. The code is available at https://github.com/hikvision-research/SAViT.

        ----

        ## [655] Deciding What to Model: Value-Equivalent Sampling for Reinforcement Learning

        **Authors**: *Dilip Arumugam, Benjamin Van Roy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3b18d368150474ac6fc9bb665d3eb3da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3b18d368150474ac6fc9bb665d3eb3da-Abstract-Conference.html)

        **Abstract**:

        The quintessential model-based reinforcement-learning agent iteratively refines its estimates or prior beliefs about the true underlying model of the environment. Recent empirical successes in model-based reinforcement learning with function approximation, however, eschew the true model in favor of a surrogate that, while ignoring various facets of the environment, still facilitates effective planning over behaviors. Recently formalized as the value equivalence principle, this algorithmic technique is perhaps unavoidable as real-world reinforcement learning demands consideration of a simple, computationally-bounded agent interacting with an overwhelmingly complex environment, whose underlying dynamics likely exceed the agent's capacity for representation. In this work, we consider the scenario where agent limitations may entirely preclude identifying an exactly value-equivalent model, immediately giving rise to a trade-off between identifying a model that is simple enough to learn while only incurring bounded sub-optimality. To address this problem, we introduce an algorithm that, using rate-distortion theory, iteratively computes an approximately-value-equivalent, lossy compression of the environment which an agent may feasibly target in lieu of the true model. We prove an information-theoretic, Bayesian regret bound for our algorithm that holds for any finite-horizon, episodic sequential decision-making problem. Crucially, our regret bound can be expressed in one of two possible forms, providing a performance guarantee for finding either the simplest model that achieves a desired sub-optimality gap or, alternatively, the best model given a limit on agent capacity.

        ----

        ## [656] Giga-scale Kernel Matrix-Vector Multiplication on GPU

        **Authors**: *Robert Hu, Siu Lun Chau, Dino Sejdinovic, Joan Glaunès*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3b1f32693e9fe15c949a0742bf226803-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3b1f32693e9fe15c949a0742bf226803-Abstract-Conference.html)

        **Abstract**:

        Kernel matrix-vector multiplication (KMVM) is a foundational operation in machine learning and scientific computing. However, as KMVM tends to scale quadratically in both memory and time, applications are often limited by these computational constraints. In this paper, we propose a novel approximation procedure coined \textit{Faster-Fast and Free Memory Method} ($\text{F}^3$M) to address these scaling issues of KMVM for tall~($10^8\sim 10^9$) and skinny~($D\leq7$) data. Extensive experiments demonstrate that $\text{F}^3$M has empirical \emph{linear time and memory} complexity with a relative error of order $10^{-3}$ and can compute a full KMVM for a billion points \emph{in under a minute} on a high-end GPU, leading to a significant speed-up in comparison to existing CPU methods. We demonstrate the utility of our procedure by applying it as a drop-in for the state-of-the-art GPU-based linear solver FALKON, \emph{improving speed 1.5-5.5 times} at the cost of $<1\%$ drop in accuracy. We further demonstrate competitive results on \emph{Gaussian Process regression} coupled with significant speedups on a variety of real-world datasets.

        ----

        ## [657] MultiScan: Scalable RGBD scanning for 3D environments with articulated objects

        **Authors**: *Yongsen Mao, Yiming Zhang, Hanxiao Jiang, Angel X. Chang, Manolis Savva*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html)

        **Abstract**:

        We introduce MultiScan, a scalable RGBD dataset construction pipeline leveraging commodity mobile devices to scan indoor scenes with articulated objects and web-based semantic annotation interfaces to efficiently annotate object and part semantics and part mobility parameters. We use this pipeline to collect 273 scans of 117 indoor scenes containing 10957 objects and 5129 parts. The resulting MultiScan dataset provides RGBD streams with per-frame camera poses, textured 3D surface meshes, richly annotated part-level and object-level semantic labels, and part mobility parameters. We validate our dataset on instance segmentation and part mobility estimation tasks and benchmark methods for these tasks from prior work. Our experiments show that part segmentation and mobility estimation in real 3D scenes remain challenging despite recent progress in 3D object segmentation.

        ----

        ## [658] Are GANs overkill for NLP?

        **Authors**: *David Alvarez-Melis, Vikas K. Garg, Adam Kalai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3b54ff26ae928fb2f111198c75f6a7e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3b54ff26ae928fb2f111198c75f6a7e3-Abstract-Conference.html)

        **Abstract**:

        This work offers a novel theoretical perspective on why, despite numerous attempts, adversarial approaches to generative modeling (e.g., GANs) have not been as successful for certain generation tasks, particularly sequential tasks such as Natural Language Generation, as they have in others, such as Computer Vision. In particular, on sequential data such as text, maximum-likelihood approaches are significantly more utilized than GANs. We show that,  while it may seem that maximizing likelihood is inherently different than minimizing distinguishability, this distinction is largely an artifact of the limited representational capacity of the model family, for a wide class of adversarial objectives. We give a theoretical model in which minimizing KL-divergence (i.e., maximizing likelihood) is a more efficient approach to effectively minimizing the same distinguishability criteria that adversarial models seek to optimize. Reductions show that minimizing distinguishability can be seen as simply boosting likelihood for certain families of models including n-gram models and neural networks with a softmax output layer. To achieve a full polynomial-time reduction, a novel next-token distinguishability model is considered. Some preliminary empirical evidence is also provided to substantiate our theoretical analyses.

        ----

        ## [659] "Why Not Other Classes?": Towards Class-Contrastive Back-Propagation Explanations

        **Authors**: *Yipei Wang, Xiaoqian Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3b7a66b2d1258e892c89f485b8f896e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3b7a66b2d1258e892c89f485b8f896e0-Abstract-Conference.html)

        **Abstract**:

        Numerous methods have been developed to explain the inner mechanism of deep neural network (DNN) based classifiers. Existing explanation methods are often limited to explaining predictions of a pre-specified class, which answers the question “why is the input classified into this class?” However, such explanations with respect to a single class are inherently insufficient because they do not capture features with class-discriminative power. That is, features that are important for predicting one class may also be important for other classes. To capture features with true class-discriminative power, we should instead ask “why is the input classified into this class, but not others?” To answer this question, we propose a weighted contrastive framework for explaining DNNs. Our framework can easily convert any existing back-propagation explanation methods to build class-contrastive explanations. We theoretically validate our weighted contrast explanation in general back-propagation explanations, and show that our framework enables class-contrastive explanations with significant improvements in both qualitative and quantitative experiments. Based on the results, we point out an important blind spot in the current explainable artificial intelligence (XAI) study, where explanations towards the predicted logits and the probabilities are obfuscated. We suggest that these two aspects should be distinguished explicitly any time explanation methods are applied.

        ----

        ## [660] Neural Stochastic Control

        **Authors**: *Jingdong Zhang, Qunxi Zhu, Wei Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3b91129cf07287aac3de7b8adba2196f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3b91129cf07287aac3de7b8adba2196f-Abstract-Conference.html)

        **Abstract**:

        Control problems are always challenging since they arise from the real-world systems where stochasticity and randomness are of ubiquitous presence.  This naturally and urgently calls for developing efficient neural control policies for stabilizing not only the deterministic equations but the stochastic systems as well.  Here, in order to meet this paramount call, we propose two types of controllers, viz., the exponential stabilizer (ES) based on the stochastic Lyapunov theory and the asymptotic stabilizer (AS) based on the stochastic asymptotic stability theory.  The ES can render the controlled systems exponentially convergent but it requires a long computational time; conversely, the AS makes the training much faster but it can only assure the asymptotic (not the exponential) attractiveness of the control targets. These two stochastic controllers thus are complementary in applications. We also investigate rigorously the linear control in both convergence time and energy cost and numerically compare it with the proposed controllers in these terms.  More significantly, we use several representative physical systems to illustrate the usefulness of the proposed controllers in stabilization of dynamical systems.

        ----

        ## [661] Constrained Update Projection Approach to Safe Policy Optimization

        **Authors**: *Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li, Yaodong Yang, Gang Pan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html)

        **Abstract**:

        Safe reinforcement learning (RL) studies problems where an intelligent agent has to not only maximize reward but also avoid exploring unsafe areas. In this study, we propose CUP, a novel policy optimization method based on Constrained Update Projection framework that enjoys rigorous safety guarantee. Central to our CUP development is the newly proposed surrogate functions along with the performance bound. Compared to previous safe reinforcement learning meth- ods, CUP enjoys the benefits of 1) CUP generalizes the surrogate functions to generalized advantage estimator (GAE), leading to strong empirical performance. 2) CUP unifies performance bounds, providing a better understanding and in- terpretability for some existing algorithms; 3) CUP provides a non-convex im- plementation via only first-order optimizers, which does not require any strong approximation on the convexity of the objectives. To validate our CUP method, we compared CUP against a comprehensive list of safe RL baselines on a wide range of tasks. Experiments show the effectiveness of CUP both in terms of reward and safety constraint satisfaction. We have opened the source code of CUP at https://github.com/zmsn-2077/CUP-safe-rl.

        ----

        ## [662] DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection

        **Authors**: *Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing Xu, Hang Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ba960559212691be13fa81d9e5e0047-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ba960559212691be13fa81d9e5e0047-Abstract-Conference.html)

        **Abstract**:

        Open-world object detection, as a more general and challenging goal, aims to recognize and localize objects described by arbitrary category names. The recent work GLIP formulates this problem as a grounding problem by concatenating all category names of detection datasets into sentences, which leads to inefficient interaction between category names. This paper presents DetCLIP, a paralleled visual-concept pre-training method for open-world detection by resorting to knowledge enrichment from a designed concept dictionary. To achieve better learning efficiency, we propose a novel paralleled concept formulation that extracts concepts separately to better utilize heterogeneous datasets (i.e., detection, grounding, and image-text pairs) for training. We further design a concept dictionary (with descriptions) from various online sources and detection datasets to provide prior knowledge for each concept. By enriching the concepts with their descriptions,we explicitly build the relationships among various concepts to facilitate the open-domain learning. The proposed concept dictionary is further used to provide sufficient negative concepts for the construction of the word-region alignment loss, and to complete labels for objects with missing descriptions in captions of image-text pair data. The proposed framework demonstrates strong zero-shot detection performances, e.g., on the LVIS dataset, our DetCLIP-T outperforms GLIP-T by 9.9% mAP and obtains a 13.5% improvement on rare categories compared to the fully-supervised model with the same backbone as ours.

        ----

        ## [663] The Sample Complexity of One-Hidden-Layer Neural Networks

        **Authors**: *Gal Vardi, Ohad Shamir, Nati Srebro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3baf4eeffad860ca9c54aeab632716b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3baf4eeffad860ca9c54aeab632716b4-Abstract-Conference.html)

        **Abstract**:

        We study norm-based uniform convergence bounds for neural networks, aiming at a tight understanding of how these are affected by the architecture and type of norm constraint, for the simple class of scalar-valued one-hidden-layer networks, and inputs bounded in Euclidean norm. We begin by proving that in general, controlling the spectral norm of the hidden layer weight matrix is insufficient to get uniform convergence guarantees (independent of the network width), while a stronger Frobenius norm control is sufficient, extending and improving on previous work. Motivated by the proof constructions, we identify and analyze two important settings where (perhaps surprisingly) a mere spectral norm control turns out to be sufficient: First, when the network's activation functions are sufficiently smooth (with the result extending to deeper networks); and second, for certain types of convolutional networks. In the latter setting, we study how the sample complexity is additionally affected by parameters such as the amount of overlap between patches and the overall number of patches.

        ----

        ## [664] Learning from Few Samples: Transformation-Invariant SVMs with Composition and Locality at Multiple Scales

        **Authors**: *Tao Liu, P. R. Kumar, Ruida Zhou, Xi Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3bb52493b3fb77a3340401d5f7a1b6b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3bb52493b3fb77a3340401d5f7a1b6b6-Abstract-Conference.html)

        **Abstract**:

        Motivated by the problem of learning with small sample sizes, this paper shows how to incorporate into support-vector machines (SVMs) those properties that have made convolutional neural networks (CNNs) successful. Particularly important is the ability to incorporate domain knowledge of invariances, e.g., translational invariance of images. Kernels based on the \textit{maximum} similarity over a group of transformations are not generally positive definite. Perhaps it is for this reason that they have not been studied theoretically. We address this lacuna and show that positive definiteness indeed holds \textit{with high probability} for kernels based on the maximum similarity in the small training sample set regime of interest, and that they do yield the best results in that regime. We also show how additional properties such as their ability to incorporate local features at multiple spatial scales, e.g., as done in CNNs through max pooling, and to provide the benefits of composition through the architecture of multiple layers, can also be embedded into SVMs. We verify through experiments on widely available image sets that the resulting SVMs do provide superior accuracy in comparison to well-established deep neural network benchmarks for small sample sizes.

        ----

        ## [665] Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation

        **Authors**: *Zhiwei Hao, Jianyuan Guo, Ding Jia, Kai Han, Yehui Tang, Chao Zhang, Han Hu, Yunhe Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3bd2d73b4e96b0ac5a319be58a96016c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3bd2d73b4e96b0ac5a319be58a96016c-Abstract-Conference.html)

        **Abstract**:

        In the past few years, transformers have achieved promising performance on various computer vision tasks. Unfortunately, the immense inference overhead of most existing vision transformers withholds them from being deployed on edge devices such as cell phones and smart watches. Knowledge distillation is a widely used paradigm for compressing cumbersome architectures into compact students via transferring information. However, most of them are designed for convolutional neural networks (CNNs), which do not fully investigate the character of vision transformers. In this paper, we fully utilize the patch-level information and propose a fine-grained manifold distillation method for transformer-based networks. Specifically, we train a tiny student model to match a pre-trained teacher model in the patch-level manifold space. Then, we decouple the manifold matching loss into three terms with careful design to further reduce the computational costs for the patch relationship. Equipped with the proposed method, a DeiT-Tiny model containing 5M parameters achieves 76.5\% top-1 accuracy on ImageNet-1k, which is +2.0\% higher than previous distillation approaches. Transfer learning results on other classification benchmarks and downstream vision tasks also demonstrate the superiority of our method over the state-of-the-art algorithms.

        ----

        ## [666] Equivariant Graph Hierarchy-Based Neural Networks

        **Authors**: *Jiaqi Han, Wenbing Huang, Tingyang Xu, Yu Rong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3bdeb28a531f7af94b56bcdf8ee88f17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3bdeb28a531f7af94b56bcdf8ee88f17-Abstract-Conference.html)

        **Abstract**:

        Equivariant Graph neural Networks (EGNs) are powerful in characterizing the dynamics of multi-body physical systems. Existing EGNs conduct flat message passing, which, yet, is unable to capture the spatial/dynamical hierarchy for complex systems particularly, limiting substructure discovery and global information fusion. In this paper, we propose Equivariant Hierarchy-based Graph Networks (EGHNs) which consist of the three key components: generalized Equivariant Matrix Message Passing (EMMP) , E-Pool and E-UnPool. In particular, EMMP is able to improve the expressivity of conventional equivariant message passing, E-Pool assigns the quantities of the low-level nodes into high-level clusters, while E-UnPool leverages the high-level information to update the dynamics of the low-level nodes. As their names imply, both E-Pool and E-UnPool are guaranteed to be equivariant to meet physic symmetry. Considerable experimental evaluations verify the effectiveness of our EGHN on several applications including multi-object dynamics simulation, motion capture, and protein dynamics modeling.

        ----

        ## [667] Learning interacting dynamical systems with latent Gaussian process ODEs

        **Authors**: *Çagatay Yildiz, Melih Kandemir, Barbara Rakitsch*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3be14af22f0b311325664277f48111f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3be14af22f0b311325664277f48111f4-Abstract-Conference.html)

        **Abstract**:

        We study uncertainty-aware modeling of continuous-time dynamics of interacting objects. We introduce a new model that decomposes independent dynamics of single objects accurately from their interactions. By employing latent Gaussian process ordinary differential equations, our model infers both independent dynamics and their interactions with reliable uncertainty estimates. In our formulation, each object is represented as a graph node and interactions are modeled by accumulating the messages coming from neighboring objects. We show that efficient inference of such a complex network of variables is possible with modern variational sparse Gaussian process inference techniques. We empirically demonstrate that our model improves the reliability of long-term predictions over neural network based alternatives and it successfully handles missing dynamic or static information. Furthermore, we observe that only our model can successfully encapsulate independent dynamics and interaction information in distinct functions and show the benefit from this disentanglement in extrapolation scenarios.

        ----

        ## [668] OLIVES Dataset: Ophthalmic Labels for Investigating Visual Eye Semantics

        **Authors**: *Mohit Prabhushankar, Kiran Kokilepersaud, Yash-Yee Logan, Stephanie Trejo Corona, Ghassan AlRegib, Charles C. Wykoff*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3be60b4a739b95a07a944a1a2c41e05e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/3be60b4a739b95a07a944a1a2c41e05e-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Clinical diagnosis of the eye is performed over multifarious data modalities including scalar clinical labels, vectorized biomarkers, two-dimensional fundus images, and three-dimensional Optical Coherence Tomography (OCT) scans. Clinical practitioners use all available data modalities for diagnosing and treating eye diseases like Diabetic Retinopathy (DR) or Diabetic Macular Edema (DME). Enabling usage of machine learning algorithms within the ophthalmic medical domain requires research into the relationships and interactions between all relevant data over a treatment period. Existing datasets are limited in that they neither provide data nor consider the explicit relationship modeling between the data modalities. In this paper, we introduce the Ophthalmic Labels for Investigating Visual Eye Semantics (OLIVES) dataset that addresses the above limitation. This is the first OCT and near-IR fundus dataset that includes clinical labels, biomarker labels, disease labels, and time-series patient treatment information from associated clinical trials. The dataset consists of 1268 near-IR fundus images each with at least 49 OCT scans, and 16 biomarkers, along with 4 clinical labels and a disease diagnosis of DR or DME. In total, there are 96 eyes' data averaged over a period of at least two years with each eye treated for an average of 66 weeks and 7 injections. We benchmark the utility of OLIVES dataset for ophthalmic data as well as provide benchmarks and concrete research directions for core and emerging machine learning paradigms within medical image analysis.

        ----

        ## [669] Off-Policy Evaluation for Action-Dependent Non-stationary Environments

        **Authors**: *Yash Chandak, Shiv Shankar, Nathaniel D. Bastian, Bruno C. da Silva, Emma Brunskill, Philip S. Thomas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3bf80b34f731313b8292f4578e820c90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3bf80b34f731313b8292f4578e820c90-Abstract-Conference.html)

        **Abstract**:

        Methods for sequential decision-making are often built upon a foundational assumption that the underlying decision process is stationary. This limits the application of such methods because real-world problems are often subject to changes due to external factors (\textit{passive} non-stationarity), changes induced by interactions with the system itself (\textit{active} non-stationarity), or both (\textit{hybrid} non-stationarity). In this work, we take the first steps towards the fundamental challenge of on-policy and off-policy evaluation amidst structured changes due to active, passive, or hybrid non-stationarity. Towards this goal, we make a \textit{higher-order stationarity} assumption such that non-stationarity results in changes over time, but the way changes happen is fixed. We propose, OPEN, an algorithm that uses a double application of counterfactual reasoning and a novel importance-weighted instrument-variable regression to obtain both a lower bias and a lower variance estimate of the structure in the changes of a policy's past performances. Finally, we show promising results on how OPEN can be used to predict future performances for several domains inspired by real-world applications that exhibit non-stationarity.

        ----

        ## [670] Fast Mixing of Stochastic Gradient Descent with Normalization and Weight Decay

        **Authors**: *Zhiyuan Li, Tianhao Wang, Dingli Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3c215225324f9988858602dc92219615-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3c215225324f9988858602dc92219615-Abstract-Conference.html)

        **Abstract**:

        We prove the Fast Equilibrium Conjecture proposed by Li et al., (2020), i.e., stochastic gradient descent (SGD) on a scale-invariant loss (e.g., using networks with various normalization schemes) with learning rate $\eta$ and weight decay factor $\lambda$ mixes in function space in $\mathcal{\tilde{O}}(\frac{1}{\lambda\eta})$ steps,  under two standard assumptions: (1) the noise covariance matrix is non-degenerate and (2) the minimizers of the loss form a connected, compact and analytic manifold. The analysis uses the framework of Li et al., (2021) and shows that for every $T>0$, the iterates of SGD with learning rate $\eta$ and weight decay factor $\lambda$ on the scale-invariant loss converge in distribution in $\Theta\left(\eta^{-1}\lambda^{-1}(T+\ln(\lambda/\eta))\right)$  iterations as $\eta\lambda\to 0$ while satisfying $\eta \le O(\lambda)\le O(1)$.  Moreover, the evolution of the limiting distribution can be described by a stochastic differential equation that mixes to the same equilibrium distribution for every initialization around the manifold of minimizers as $T\to\infty$.

        ----

        ## [671] OPEN: Orthogonal Propagation with Ego-Network Modeling

        **Authors**: *Liang Yang, Lina Kang, Qiuliang Zhang, Mengzhe Li, Bingxin Niu, Dongxiao He, Zhen Wang, Chuan Wang, Xiaochun Cao, Yuanfang Guo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3c2b60a3f269c404e9329ee119f2d34a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3c2b60a3f269c404e9329ee119f2d34a-Abstract-Conference.html)

        **Abstract**:

        To alleviate the unfavorable effect of noisy topology in Graph Neural networks (GNNs), some efforts perform the local topology refinement through the pairwise propagation weight learning and the multi-channel extension. Unfortunately, most of them suffer a common and fatal drawback: irrelevant propagation to one node and in multi-channels. These two kinds of irrelevances make propagation weights in multi-channels free to be determined by the labeled data, and thus the GNNs are exposed to overfitting. To tackle this issue, a novel Orthogonal Propagation with Ego-Network modeling (OPEN) is proposed by modeling relevances between propagations. Specifically, the relevance between propagations to one node is modeled by whole ego-network modeling, while the relevance between propagations in multi-channels is modeled via diversity requirement. By interpreting the propagations to one node from the perspective of dimension reduction, propagation weights are inferred from principal components of the ego-network, which are orthogonal to each other. Theoretical analysis and experimental evaluations reveal four attractive characteristics of OPEN as modeling high-order relationships beyond pairwise one, preventing overfitting, robustness, and high efficiency.

        ----

        ## [672] Beyond the Best: Distribution Functional Estimation in Infinite-Armed Bandits

        **Authors**: *Yifei Wang, Tavor Z. Baharav, Yanjun Han, Jiantao Jiao, David Tse*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3c2fd72a28fb98facf98546727320249-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3c2fd72a28fb98facf98546727320249-Abstract-Conference.html)

        **Abstract**:

        In the infinite-armed bandit problem, each arm's average reward is sampled from an unknown distribution, and each arm can be sampled further to obtain noisy estimates of the average reward of that arm. Prior work focuses on the best arm, i.e. estimating the maximum of the average reward distribution. We consider a general class of distribution functionals beyond the maximum and obtain optimal sample complexities in both offline and online settings. We show that online estimation, where the learner can sequentially choose whether to sample a new or existing arm, offers no advantage over the offline setting for estimating the mean functional, but significantly reduces the sample complexity for other functionals such as the median, maximum, and trimmed mean. We propose unified meta algorithms for the online and offline settings and derive matching lower bounds using different Wasserstein distances. For the special case of median estimation, we identify a curious thresholding phenomenon on the indistinguishability between Gaussian convolutions with respect to the noise level, which may be of independent interest.

        ----

        ## [673] Adversarial training for high-stakes reliability

        **Authors**: *Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3c44405d619a6920384a45bce876b41e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3c44405d619a6920384a45bce876b41e-Abstract-Conference.html)

        **Abstract**:

        In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.In this work, we used a safe language generation task (``avoid injuries'') as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques---including a tool that assists human adversaries---to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs.  We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on--- tripling the time to find adversarial examples without tools and doubling the time with our tool (from 13 to 26 minutes)---without affecting in-distribution performance.  We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.

        ----

        ## [674] ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models

        **Authors**: *Chunyuan Li, Haotian Liu, Liunian Harold Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, Jianfeng Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3c4688b6a76f25f2311daa0d75a58f1a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/3c4688b6a76f25f2311daa0d75a58f1a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Learning visual representations from natural language supervision has recently shown great promise in a number of pioneering works. In general, these language-augmented visual models demonstrate strong transferability to a variety of datasets/tasks. However, it remains challenging to evaluate the transferablity of these foundation models due to the lack of easy-to-use toolkits for fair benchmarking. To tackle this, we build ELEVATER (Evaluation of Language-augmented Visual Task-level Transfer), the first benchmark to compare and evaluate pre-trained language-augmented visual models. Several highlights include: (i) Datasets. As downstream evaluation suites, it consists of 20 image classification datasets and 35 object detection datasets, each of which is augmented with external knowledge. (ii) Toolkit. An automatic hyper-parameter tuning toolkit is developed to ensure the fairness in model adaption. To leverage the full power of language-augmented visual models, novel language-aware initialization methods are proposed to significantly improve the adaption performance. (iii) Metrics. A variety of evaluation metrics are used, including sample-efficiency (zero-shot and few-shot) and parameter-efficiency (linear probing and full model fine-tuning). We will publicly release ELEVATER.

        ----

        ## [675] Augmented RBMLE-UCB Approach for Adaptive Control of Linear Quadratic Systems

        **Authors**: *Akshay Mete, Rahul Singh, P. R. Kumar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3c601cd5866099648c6dc783e7f39858-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3c601cd5866099648c6dc783e7f39858-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of controlling an unknown stochastic linear system with quadratic costs -- called the adaptive LQ control problem. We re-examine an approach called ``Reward-Biased Maximum Likelihood Estimate'' (RBMLE) that was proposed more than forty years ago, and which predates the ``Upper Confidence Bound'' (UCB) method, as well as the definition of ``regret'' for bandit problems. It simply added a term favoring parameters with larger rewards to the criterion for parameter estimation.  We show how the RBMLE and UCB methods can be reconciled, and thereby propose an Augmented RBMLE-UCB algorithm that combines the penalty of the RBMLE method with the constraints of the UCB method, uniting the two approaches to optimism in the face of uncertainty. We establish that theoretically, this method retains ${\mathcal{O}}(\sqrt{T})$ regret, the best known so far. We further compare the empirical performance of the proposed Augmented RBMLE-UCB and the standard RBMLE (without the augmentation) with UCB, Thompson Sampling, Input Perturbation, Randomized Certainty Equivalence and StabL on many real-world examples including flight control of Boeing 747 and Unmanned Aerial Vehicle. We perform extensive simulation studies showing that the Augmented RBMLE consistently outperforms UCB, Thompson Sampling and StabL by a huge margin, while it is marginally better than Input Perturbation and moderately better than Randomized Certainty Equivalence.

        ----

        ## [676] Stochastic Window Transformer for Image Restoration

        **Authors**: *Jie Xiao, Xueyang Fu, Feng Wu, Zheng-Jun Zha*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ca6d336ddaa316a6ae953a20b9477cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ca6d336ddaa316a6ae953a20b9477cf-Abstract-Conference.html)

        **Abstract**:

        Thanks to the powerful representation capabilities, transformers have made impressive progress in image restoration. However, existing transformers-based methods do not carefully consider the particularities of image restoration. In general, image restoration requires that an ideal approach should be translation-invariant to the degradation, i.e., the undesirable degradation should be removed irrespective of its position within the image. Furthermore, the local relationships also play a vital role, which should be faithfully exploited for recovering clean images. Nevertheless, most transformers either adopt local attention with the fixed local window strategy or global attention, which unfortunately breaks the translation invariance and causes huge loss of local relationships. To address these issues, we propose an elegant stochastic window strategy for transformers. Specifically, we first introduce the window partition with stochastic shift to replace the original fixed window partition for training. Then, we design a new layer expectation propagation algorithm to efficiently approximate the expectation of the induced stochastic transformer for testing. Our stochastic window transformer not only enjoys powerful representation but also maintains the desired property of translation invariance and locality. Experiments validate the stochastic window strategy consistently improves performance on various image restoration tasks (deraining, denoising and deblurring) by significant margins. The code is available at https://github.com/jiexiaou/Stoformer.

        ----

        ## [677] Phase Transition from Clean Training to Adversarial Training

        **Authors**: *Yue Xing, Qifan Song, Guang Cheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html)

        **Abstract**:

        Adversarial training is one important algorithm to achieve robust machine learning models. However, numerous empirical results show a great performance degradation from clean training to adversarial training (e.g., 90+\% vs 67\% testing accuracy on CIFAR-10 dataset), which does not match the theoretical guarantee delivered by the existing studies. Such a gap inspires us to explore the existence of an (asymptotic) phase transition phenomenon with respect to the attack strength: adversarial training is as well behaved as clean training in the small-attack regime, but there is a sharp transition from clean training to adversarial training in the large-attack regime. We validate this conjecture in linear regression models, and conduct comprehensive experiments in deep neural networks.

        ----

        ## [678] pFL-Bench: A Comprehensive Benchmark for Personalized Federated Learning

        **Authors**: *Daoyuan Chen, Dawei Gao, Weirui Kuang, Yaliang Li, Bolin Ding*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3cc03e19fed71a2b9347d83921ca2e7d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/3cc03e19fed71a2b9347d83921ca2e7d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Personalized Federated Learning (pFL), which utilizes and deploys distinct local models, has gained increasing attention in recent years due to its success in handling the statistical heterogeneity of FL clients. However, standardized evaluation and systematical analysis of diverse pFL methods remain a challenge. Firstly, the highly varied datasets, FL simulation settings and pFL implementations prevent easy and fair comparisons of pFL methods. Secondly, the current pFL literature diverges in the adopted evaluation and ablation protocols. Finally, the effectiveness and robustness of pFL methods are under-explored in various practical scenarios, such as the generalization to new clients and the participation of resource-limited clients. To tackle these challenges, we propose the first comprehensive pFL benchmark, pFL-Bench, for facilitating rapid, reproducible, standardized and thorough pFL evaluation. The proposed benchmark contains more than 10 dataset variants in various application domains with a unified data partition and realistic heterogeneous settings; a modularized and easy-to-extend pFL codebase with more than 20 competitive pFL method implementations; and systematic evaluations under containerized environments in terms of generalization, fairness, system overhead, and convergence. We highlight the benefits and potential of state-of-the-art pFL methods and hope pFL-Bench enables further pFL research and broad applications that would otherwise be difficult owing to the absence of a dedicated benchmark. The code is released at https://github.com/alibaba/FederatedScope/tree/master/benchmark/pFL-Bench.

        ----

        ## [679] Squeezeformer: An Efficient Transformer for Automatic Speech Recognition

        **Authors**: *Sehoon Kim, Amir Gholami, Albert E. Shaw, Nicholas Lee, Karttikeya Mangalam, Jitendra Malik, Michael W. Mahoney, Kurt Keutzer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ccf6da39eeb8fefc8bbb1b0124adbd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ccf6da39eeb8fefc8bbb1b0124adbd1-Abstract-Conference.html)

        **Abstract**:

        The recently proposed Conformer model has become the de facto backbone model for various downstream speech tasks based on its hybrid attention-convolution architecture that captures both local and global features. However, through a series of systematic studies, we find that the Conformer architectureâ€™s design choices are not optimal. After re-examining the design choices for both the macro and micro-architecture of Conformer, we propose Squeezeformer which consistently outperforms the state-of-the-art ASR models under the same training schemes. In particular, for the macro-architecture, Squeezeformer incorporates (i) the Temporal U-Net structure which reduces the cost of the multi-head attention modules on long sequences, and (ii) a simpler block structure of multi-head attention or convolution modules followed up by feed-forward module instead of the Macaron structure proposed in Conformer. Furthermore, for the micro-architecture, Squeezeformer (i) simplifies the activations in the convolutional block, (ii) removes redundant Layer Normalization operations, and (iii) incorporates an efficient depthwise down-sampling layer to efficiently sub-sample the input signal. Squeezeformer achieves state-of-the-art results of 7.5%, 6.5%, and 6.0% word-error-rate (WER) on LibriSpeech test-other without external language models, which are 3.1%, 1.4%, and 0.6% better than Conformer-CTC with the same number of FLOPs. Our code is open-sourced and available online.

        ----

        ## [680] Deep Generalized Schrödinger Bridge

        **Authors**: *Guan-Horng Liu, Tianrong Chen, Oswin So, Evangelos A. Theodorou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d17b7f7d52c83ab6e97e2dc0bda2e71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d17b7f7d52c83ab6e97e2dc0bda2e71-Abstract-Conference.html)

        **Abstract**:

        Mean-Field Game (MFG) serves as a crucial mathematical framework in modeling the collective behavior of individual agents interacting stochastically with a large population. In this work, we aim at solving a challenging class of MFGs in which the differentiability of these interacting preferences may not be available to the solver, and the population is urged to converge exactly to some desired distribution. These setups are, despite being well-motivated for practical purposes, complicated enough to paralyze most (deep) numerical solvers. Nevertheless, we show that Schrödinger Bridge — as an entropy-regularized optimal transport model — can be generalized to accepting mean-field structures, hence solving these MFGs. This is achieved via the application of Forward-Backward Stochastic Differential Equations theory, which, intriguingly, leads to a computational framework with a similar structure to Temporal Difference learning. As such, it opens up novel algorithmic connections to Deep Reinforcement Learning that we leverage to facilitate practical training. We show that our proposed objective function provides necessary and sufficient conditions to the mean-field problem. Our method, named Deep Generalized Schrödinger Bridge (DeepGSB), not only outperforms prior methods in solving classical population navigation MFGs, but is also capable of solving 1000-dimensional opinion depolarization, setting a new state-of-the-art numerical solver for high-dimensional MFGs. Our code will be made available at https://github.com/ghliu/DeepGSB.

        ----

        ## [681] Characterizing the Ventral Visual Stream with Response-Optimized Neural Encoding Models

        **Authors**: *Meenakshi Khosla, Keith Jamison, Amy Kuceyeski, Mert R. Sabuncu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d278eebf6e555e6efd050817d774586-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d278eebf6e555e6efd050817d774586-Abstract-Conference.html)

        **Abstract**:

        Decades of experimental research based on simple, abstract stimuli has revealed the coding principles of the ventral visual processing hierarchy, from the presence of edge detectors in the primary visual cortex to the selectivity for complex visual categories in the anterior ventral stream. However, these studies are, by construction, constrained by their $\textit{a priori}$ hypotheses. Furthermore, beyond the early stages, precise neuronal tuning properties and representational transformations along the ventral visual pathway remain poorly understood. In this work, we propose to employ response-optimized encoding models trained solely to predict the functional MRI activation, in order to gain insights into the tuning properties and representational transformations in the series of areas along the ventral visual pathway. We demonstrate the strong generalization abilities of these models on artificial stimuli and novel datasets. Intriguingly, we find that response-optimized models trained towards the ventral-occipital and lateral-occipital areas, but not early visual areas, can recapitulate complex visual behaviors like object categorization and perceived image-similarity in humans. We further probe the trained networks to reveal representational biases in different visual areas and generate experimentally testable hypotheses. Our analyses suggest a shape-based processing along the ventral visual stream and provide a unified picture of multiple neural phenomena characterized over the last decades with controlled fMRI studies.

        ----

        ## [682] Learning sparse features can lead to overfitting in neural networks

        **Authors**: *Leonardo Petrini, Francesco Cagnetta, Eric Vanden-Eijnden, Matthieu Wyart*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d3a9e085540c65dd3e5731361f9320e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d3a9e085540c65dd3e5731361f9320e-Abstract-Conference.html)

        **Abstract**:

        It is widely believed that the success of deep networks lies in their ability to learn a meaningful representation of the features of the data. Yet, understanding when and how this feature learning improves performance remains a challenge: for example, it is beneficial for modern architectures trained to classify images, whereas it is detrimental for fully-connected networks trained for the same task on the same data. Here we propose an explanation for this puzzle, by showing that feature learning can perform worse than lazy training (via random feature kernel or the NTK) as the former can lead to a sparser neural representation. Although sparsity is known to be essential for learning anisotropic data, it is detrimental when the target function is constant or smooth along certain directions of input space. We illustrate this phenomenon in two settings: (i) regression of Gaussian random functions on the $d$-dimensional unit sphere and  (ii) classification of benchmark datasets of images. For (i), we compute the scaling of the generalization error with number of training points, and show that methods that do not learn features generalize better, even when the dimension of the input space is large. For (ii), we show empirically that learning features can indeed lead to sparse and thereby less smooth representations of the image predictors. This fact is plausibly responsible for deteriorating the performance, which is known to be correlated with smoothness along diffeomorphisms.

        ----

        ## [683] Public Wisdom Matters! Discourse-Aware Hyperbolic Fourier Co-Attention for Social Text Classification

        **Authors**: *Karish Grover, S. M. Phaneendra Angara, Md. Shad Akhtar, Tanmoy Chakraborty*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d57795f0e263aa69577f1bbceade46b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d57795f0e263aa69577f1bbceade46b-Abstract-Conference.html)

        **Abstract**:

        Social media has become the fulcrum of all forms of communication. Classifying social texts such as fake news, rumour, sarcasm, etc. has gained significant attention. The surface-level signals expressed by a social-text itself may not be adequate for such tasks; therefore, recent methods attempted to incorporate other intrinsic signals such as user behavior and the underlying graph structure. Oftentimes, the public wisdom expressed through the comments/replies to a social-text acts as a surrogate of crowd-sourced view and may provide us with complementary signals. State-of-the-art methods on social-text classification tend to ignore such a rich hierarchical signal. Here, we propose Hyphen, a discourse-aware hyperbolic spectral co-attention network. Hyphen is a fusion of hyperbolic graph representation learning with a novel Fourier co-attention mechanism in an attempt to generalise the social-text classification tasks by incorporating public discourse. We parse public discourse as an Abstract Meaning Representation (AMR) graph and use the powerful hyperbolic geometric representation to model graphs with hierarchical structure. Finally, we equip it with a novel Fourier co-attention mechanism to capture the correlation between the source post and public discourse. Extensive experiments on four different social-text classification tasks, namely detecting fake news, hate speech, rumour, and sarcasm, show that Hyphen generalises well, and achieves state-of-the-art results on ten benchmark datasets. We also employ a sentence-level fact-checked and annotated dataset to evaluate how Hyphen is capable of producing explanations as analogous evidence to the final prediction.

        ----

        ## [684] Harmonizing the object recognition strategies of deep neural networks with humans

        **Authors**: *Thomas Fel, Ivan F. Rodriguez Rodriguez, Drew Linsley, Thomas Serre*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d681cc4487b97c08e5aa67224dd74f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d681cc4487b97c08e5aa67224dd74f2-Abstract-Conference.html)

        **Abstract**:

        The many successes of deep neural networks (DNNs) over the past decade have largely been driven by computational scale rather than insights from biological intelligence. Here, we explore if these trends have also carried concomitant improvements in explaining the visual strategies humans rely on for object recognition. We do this by comparing two related but distinct properties of visual strategies in humans and DNNs: where they believe important visual features are in images and how they use those features to categorize objects. Across 84 different DNNs trained on ImageNet and three independent datasets measuring the where and the how of human visual strategies for object recognition on those images, we find a systematic trade-off between DNN categorization accuracy and alignment with human visual strategies for object recognition. \textit{State-of-the-art DNNs are progressively becoming less aligned with humans as their accuracy improves}. We rectify this growing issue with our neural harmonizer: a general-purpose training routine that both aligns DNN and human visual strategies and improves categorization accuracy. Our work represents the first demonstration that the scaling laws that are guiding the design of DNNs today have also produced worse models of human vision. We release our code and data at https://serre-lab.github.io/Harmonization to help the field build more human-like DNNs.

        ----

        ## [685] Learning Mixed Multinomial Logits with Provable Guarantees

        **Authors**: *Yiqun Hu, David Simchi-Levi, Zhenzhen Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d6d1bdb10e7c4855721bc44e992585c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d6d1bdb10e7c4855721bc44e992585c-Abstract-Conference.html)

        **Abstract**:

        A mixture of multinomial logits (MMNL) generalizes the single logit model, which is commonly used in predicting the probabilities of different outcomes. While extensive algorithms have been developed in the literature to learn MMNL models, theoretical results are limited. Built on the Frank-Wolfe (FW) method, we propose a new algorithm that learns both mixture weights and component-specific logit parameters with provable convergence guarantees for an arbitrary number of mixtures. Our algorithm utilizes historical choice data to generate a set of candidate choice probability vectors, each being close to the ground truth with a high probability. We further provide a sample complexity analysis to show that only a polynomial number of samples is required to secure the performance guarantee of our algorithm. Finally, we conduct simulation studies to evaluate the performance and demonstrate how to apply our algorithm to real-world applications.

        ----

        ## [686] Defining and Characterizing Reward Gaming

        **Authors**: *Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, David Krueger*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html)

        **Abstract**:

        We provide the first formal definition of \textbf{reward hacking}, a phenomenon where optimizing an imperfect proxy reward function, $\mathcal{\tilde{R}}$, leads to poor performance according to the true reward function, $\mathcal{R}$.  We say that a proxy is \textbf{unhackable} if increasing the expected proxy return can never decrease the expected true return.Intuitively, it might be possible to create an unhackable proxy by leaving some terms out of the reward function (making it ``narrower'') or overlooking fine-grained distinctions between roughly equivalent outcomes, but we show this is usually not the case.A key insight is that the linearity of reward (in state-action visit counts) makes unhackability a very strong condition. In particular, for the set of all stochastic policies, two reward functions can only be unhackable if one of them is constant.We thus turn our attention to deterministic policies and finite sets of stochastic policies, where non-trivial unhackable pairs always exist, and establish necessary and sufficient conditions for the existence of simplifications, an important special case of unhackability.Our results reveal a tension between using reward functions to specify narrow tasks and aligning AI systems with human values.

        ----

        ## [687] Learning Distinct and Representative Modes for Image Captioning

        **Authors**: *Qi Chen, Chaorui Deng, Qi Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html)

        **Abstract**:

        Over the years, state-of-the-art (SoTA) image captioning methods have achieved promising results on some evaluation metrics (e.g., CIDEr). However, recent findings show that the captions generated by these methods tend to be biased toward the "average" caption that only captures the most general mode (a.k.a, language pattern) in the training corpus, i.e., the so-called mode collapse problem. Affected by it, the generated captions are limited in diversity and usually less informative than natural image descriptions made by humans. In this paper, we seek to avoid this problem by proposing a Discrete Mode Learning (DML) paradigm for image captioning. Our innovative idea is to explore the rich modes in the training caption corpus to learn a set of "mode embeddings", and further use them to control the mode of the generated captions for existing image captioning models. Specifically, the proposed DML optimizes a dual architecture that consists of an image-conditioned discrete variational autoencoder (CdVAE) branch and a mode-conditioned image captioning (MIC) branch. The CdVAE branch maps each image caption to one of the mode embeddings stored in a learned codebook, and is trained with a pure non-autoregressive generation objective to make the modes distinct and representative. The MIC branch can be simply modified from an existing image captioning model, where the mode embedding is added to the original word embeddings as the control signal. In the experiments, we apply the proposed DML to two widely used image captioning models, Transformer and AoANet. The results show that the learned mode embedding successfully facilitates these models to generate high-quality image captions with different modes, further leading to better performance for both diversity and quality on the MS COCO dataset.

        ----

        ## [688] Lifting the Information Ratio: An Information-Theoretic Analysis of Thompson Sampling for Contextual Bandits

        **Authors**: *Gergely Neu, Julia Olkhovskaya, Matteo Papini, Ludovic Schwartz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3d84d9b523e6e82916d496e58761002e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3d84d9b523e6e82916d496e58761002e-Abstract-Conference.html)

        **Abstract**:

        We study the Bayesian regret of the renowned Thompson Sampling algorithm in contextual bandits with binary losses and adversarially-selected contexts. We adapt the information-theoretic perspective of Russo and Van Roy [2016] to the contextual setting by considering a lifted version of the information ratio defined in terms of the unknown model parameter instead of the optimal action or optimal policy as done in previous works on the same setting. This allows us to bound the regret in terms of the entropy of the prior distribution through a remarkably simple proof, and with no structural assumptions on the likelihood or the prior. The extension to priors with infinite entropy only requires a Lipschitz assumption on the log-likelihood. An interesting special case is that of logistic bandits with $d$-dimensional parameters, $K$ actions, and Lipschitz logits, for which we provide a $\tilde{O}(\sqrt{dKT})$ regret upper-bound that does not depend on the smallest slope of the sigmoid link function.

        ----

        ## [689] Is this the Right Neighborhood? Accurate and Query Efficient Model Agnostic Explanations

        **Authors**: *Amit Dhurandhar, Karthikeyan Natesan Ramamurthy, Karthikeyan Shanmugam*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3daf673aa4a06eec9b343686d88333c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3daf673aa4a06eec9b343686d88333c7-Abstract-Conference.html)

        **Abstract**:

        There have been multiple works that try to ascertain explanations for decisions of black box models on particular inputs by perturbing the input or by sampling around it, creating a neighborhood and then fitting a sparse (linear) model (e.g. LIME). Many of these methods are unstable and so more recent work tries to find stable or robust alternatives. However, stable solutions may not accurately represent the behavior of the model around the input. Thus, the question we ask in this paper is are we approximating the local boundary around the input accurately? In particular, are we sampling the right neighborhood so that a linear approximation of the black box is faithful to its true behavior around that input given that the black box can be highly non-linear (viz. deep relu network with many linear pieces). It is difficult to know the correct neighborhood width (or radius) as too small a width can lead to a bad condition number of the inverse covariance matrix of function fitting procedures resulting in unstable predictions, while too large a width may lead to accounting for multiple linear pieces and consequently a poor local approximation. We in this paper propose a simple approach that is robust across neighborhood widths in recovering faithful local explanations. In addition to a naive implementation of our approach which can still be accurate, we propose a novel adaptive neighborhood sampling scheme (ANS) that we formally show can be much more sample and query efficient. We then empirically evaluate our approach on  real data where our explanations are significantly more sample and query efficient than the competitors, while also being faithful and stable across different widths.

        ----

        ## [690] Object Scene Representation Transformer

        **Authors**: *Mehdi S. M. Sajjadi, Daniel Duckworth, Aravindh Mahendran, Sjoerd van Steenkiste, Filip Pavetic, Mario Lucic, Leonidas J. Guibas, Klaus Greff, Thomas Kipf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3dc83fcfa4d13e30070bd4b230c38cfe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3dc83fcfa4d13e30070bd4b230c38cfe-Abstract-Conference.html)

        **Abstract**:

        A compositional understanding of the world in terms of objects and their geometry in 3D space is considered a cornerstone of human cognition. Facilitating the learning of such a representation in neural networks holds promise for substantially improving labeled data efficiency. As a key step in this direction, we make progress on the problem of learning 3D-consistent decompositions of complex scenes into individual objects in an unsupervised fashion. We introduce Object Scene Representation Transformer (OSRT), a 3D-centric model in which individual object representations naturally emerge through novel view synthesis. OSRT scales to significantly more complex scenes with larger diversity of objects and backgrounds than existing methods. At the same time, it is multiple orders of magnitude faster at compositional rendering thanks to its light field parametrization and the novel Slot Mixer decoder. We believe this work will not only accelerate future architecture exploration and scaling efforts, but it will also serve as a useful tool for both object-centric as well as neural scene representation learning communities.

        ----

        ## [691] Data-Driven Conditional Robust Optimization

        **Authors**: *Abhilash Reddy Chenreddy, Nymisha Bandi, Erick Delage*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study a novel approach for data-driven decision-making under uncertainty in the presence of contextual information. Specifically, we solve this problem from a Conditional Robust Optimization (CRO) point of view. We propose an integrated framework that designs the conditional uncertainty set by jointly learning the partitions in the covariate data space and simultaneously constructing partition specific deep uncertainty sets for the random vector that perturbs the CRO problem. We also provide  theoretical guarantees for the coverage of the uncertainty sets and value at risk performances obtained using the proposed CRO approach. Finally, we use the simulated and real world data to show the implementation of our approach and compare it against two non-contextual benchmark approaches to demonstrate the value of exploiting contextual information in robust optimization.

        ----

        ## [692] COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics

        **Authors**: *Lianhui Qin, Sean Welleck, Daniel Khashabi, Yejin Choi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3e25d1aff47964c8409fd5c8dc0438d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3e25d1aff47964c8409fd5c8dc0438d7-Abstract-Conference.html)

        **Abstract**:

        Many applications of text generation require incorporating different constraints to control the semantics or style of generated text. These constraints can be hard (e.g., ensuring certain keywords are included in the output) and soft (e.g., contextualizing the output with the left- or right-hand context). In this paper, we present Energy-based Constrained Decoding with Langevin Dynamics (COLD), a decoding framework which unifies constrained generation as specifying constraints through an energy function, then performing efficient differentiable reasoning over the constraints through gradient-based sampling. COLD decoding is a flexible framework that can be applied directly to off-the-shelf left-to-right language models without the need for any task-specific fine-tuning, as demonstrated through three challenging text generation applications: lexically-constrained generation, abductive reasoning, and counterfactual reasoning. Our experiments on these constrained generation tasks point to the effectiveness of our approach, both in terms of automatic and human evaluation.

        ----

        ## [693] Matching in Multi-arm Bandit with Collision

        **Authors**: *Yirui Zhang, Siwei Wang, Zhixuan Fang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3e36cbffea708197676fa794ad57dc0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3e36cbffea708197676fa794ad57dc0a-Abstract-Conference.html)

        **Abstract**:

        In this paper, we consider the matching of multi-agent multi-armed bandit problem, i.e., while agents prefer arms with higher expected reward, arms also have preferences on agents. In such case, agents pulling the same arm may encounter collisions, which leads to a reward of zero.For this problem, we design a specific communication protocol which uses deliberate collision to transmit information among agents, and propose a layer-based algorithm that helps establish optimal stable matching between agents and arms. With this subtle communication protocol, our algorithm achieves a state-of-the-art $O(\log T)$ regret in the decentralized matching market, and outperforms existing baselines in experimental results.

        ----

        ## [694] Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts

        **Authors**: *Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, Neil Houlsby*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3e67e84abf900bb2c7cbd5759bfce62d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3e67e84abf900bb2c7cbd5759bfce62d-Abstract-Conference.html)

        **Abstract**:

        Large sparsely-activated models have obtained excellent performance in multiple domains.However, such models are typically trained on a single modality at a time.We present the Language-Image MoE, LIMoE, a sparse mixture of experts model capable of multimodal learning.LIMoE accepts both images and text simultaneously, while being trained using a contrastive loss.MoEs are a natural fit for a multimodal backbone, since expert layers can learn an appropriate partitioning of modalities.However, new challenges arise; in particular, training stability and balanced expert utilization, for which we propose an entropy-based regularization scheme.Across multiple scales, we demonstrate performance improvement over dense models of equivalent computational cost.LIMoE-L/16 trained comparably to CLIP-L/14 achieves 77.9% zero-shot ImageNet accuracy (vs. 76.2%), and when further scaled to H/14 (with additional data) it achieves 83.8%, approaching state-of-the-art methods which use custom per-modality backbones and pre-training schemes.We analyse the quantitative and qualitative behavior of LIMoE, and demonstrate phenomena such as differing treatment of the modalities and the emergence of modality-specific experts.

        ----

        ## [695] Few-shot Learning for Feature Selection with Hilbert-Schmidt Independence Criterion

        **Authors**: *Atsutoshi Kumagai, Tomoharu Iwata, Yasutoshi Ida, Yasuhiro Fujiwara*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3e683480be2766922a1b738d48ebd436-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3e683480be2766922a1b738d48ebd436-Abstract-Conference.html)

        **Abstract**:

        We propose a few-shot learning method for feature selection that can select relevant features given a small number of labeled instances. Existing methods require many labeled instances for accurate feature selection. However, sufficient instances are often unavailable. We use labeled instances in multiple related tasks to alleviate the lack of labeled instances in a target task. To measure the dependency between each feature and label, we use the Hilbert-Schmidt Independence Criterion, which is a kernel-based independence measure. By modeling the kernel functions with neural networks that take a few labeled instances in a task as input, we can encode the task-specific information to the kernels such that the kernels are appropriate for the task. Feature selection with such kernels is performed by using iterative optimization methods, in which each update step is obtained as a closed-form. This formulation enables us to directly and efficiently minimize the expected test error on features selected by a small number of labeled instances. We experimentally demonstrate that the proposed method outperforms existing feature selection methods.

        ----

        ## [696] Statistical Learning and Inverse Problems: A Stochastic Gradient Approach

        **Authors**: *Yuri R. Fonseca, Yuri F. Saporito*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3e8b1835833ef809059efa74b9df6805-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3e8b1835833ef809059efa74b9df6805-Abstract-Conference.html)

        **Abstract**:

        Inverse problems are paramount in Science and Engineering. In this paper, we consider the setup of Statistical Inverse Problem (SIP) and demonstrate how Stochastic Gradient Descent (SGD) algorithms can be used to solve linear SIP. We provide consistency and finite sample bounds for the excess risk. We also propose a modification for the SGD algorithm where we leverage machine learning methods to smooth the stochastic gradients and improve empirical performance. We exemplify the algorithm in a setting of great interest nowadays: the Functional Linear Regression model. In this case we consider a synthetic data example and a classification problem for predicting the main activity of bitcoin addresses based on their balances.

        ----

        ## [697] Hyperparameter Sensitivity in Deep Outlier Detection: Analysis and a Scalable Hyper-Ensemble Solution

        **Authors**: *Xueying Ding, Lingxiao Zhao, Leman Akoglu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3e9113e2bc2e700baa7d765470f140e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3e9113e2bc2e700baa7d765470f140e1-Abstract-Conference.html)

        **Abstract**:

        Outlier detection (OD) literature exhibits numerous algorithms as it applies to diverse domains. However, given a new detection task, it is unclear how to choose an algorithm to use, nor how to set its hyperparameter(s) (HPs) in unsupervised settings. HP tuning is an ever-growing problem with the arrival of many new detectors based on deep learning, which usually come with a long list of HPs. Surprisingly, the issue of model selection in the outlier mining literature has been “the elephant in the room”; a significant factor in unlocking the utmost potential of deep methods, yet little said or done to systematically tackle the issue. In the first part of this paper, we conduct the first large-scale analysis on the HP sensitivity of deep OD methods, and through more than 35,000 trained models, quantitatively demonstrate that model selection is inevitable. Next, we design a HP-robust and scalable deep hyper-ensemble model called ROBOD that assembles models with varying HP configurations, bypassing the choice paralysis. Importantly, we introduce novel strategies to speed up ensemble training, such as parameter sharing, batch/simultaneous training, and data subsampling, that allow us to train fewer models with fewer parameters. Extensive experiments on both image and tabular datasets show that ROBOD achieves and retains robust, state-of-the-art detection performance as compared to its modern counterparts, while taking only 2-10% of the time by the naïve hyper-ensemble with independent training.

        ----

        ## [698] TVLT: Textless Vision-Language Transformer

        **Authors**: *Zineng Tang, Jaemin Cho, Yixin Nie, Mohit Bansal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ea3134345f2e6228a29f35b86bce24d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ea3134345f2e6228a29f35b86bce24d-Abstract-Conference.html)

        **Abstract**:

        In this work, we present the Textless Vision-Language Transformer (TVLT), where homogeneous transformer blocks take raw visual and audio inputs for vision-and-language representation learning with minimal modality-specific design, and do not use text-specific modules such as tokenization or automatic speech recognition (ASR). TVLT is trained by reconstructing masked patches of continuous video frames and audio spectrograms (masked autoencoding) and contrastive modeling to align video and audio. TVLT attains performance comparable to its text-based counterpart on various multimodal tasks, such as visual question answering, image retrieval, video retrieval, and multimodal sentiment analysis, with 28x faster inference speed and only 1/3 of the parameters. Our findings suggest the possibility of learning compact and efficient visual-linguistic representations from low-level visual and audio signals without assuming the prior existence of text. Our code and checkpoints are available at: https://github.com/zinengtang/TVLT

        ----

        ## [699] A Deep Reinforcement Learning Framework for Column Generation

        **Authors**: *Cheng Chi, Amine Mohamed Aboussalah, Elias B. Khalil, Juyoung Wang, Zoha Sherkat-Masoumi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ecfe5c632afb7d96a2337b18ff99b1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ecfe5c632afb7d96a2337b18ff99b1f-Abstract-Conference.html)

        **Abstract**:

        Column Generation (CG) is an iterative algorithm for solving linear programs (LPs) with an extremely large number of variables (columns). CG is the workhorse for tackling large-scale integer linear programs, which rely on CG to solve LP relaxations within a branch and bound algorithm. Two canonical applications are the Cutting Stock Problem (CSP) and Vehicle Routing Problem with Time Windows (VRPTW). In VRPTW, for example, each binary variable represents the decision to include or exclude a route, of which there are exponentially many; CG incrementally grows the subset of columns being used, ultimately converging to an optimal solution.  We propose RLCG, the first Reinforcement Learning (RL) approach for CG. Unlike typical column selection rules which myopically select a column based on local information at each iteration, we treat CG as a sequential decision-making problem, as the column selected in an iteration affects subsequent iterations of the algorithm. This perspective lends itself to a Deep Reinforcement Learning approach that uses Graph Neural Networks (GNNs) to represent the variable-constraint structure in the LP of interest. We perform an extensive set of experiments using the publicly available BPPLIB benchmark for CSP and Solomon benchmark for VRPTW. RLCG converges faster and reduces the number of CG iterations by 22.4% for CSP and 40.9% for VRPTW on average compared to a commonly used greedy policy.

        ----

        ## [700] Learning to Drop Out: An Adversarial Approach to Training Sequence VAEs

        **Authors**: *Djordje Miladinovic, Kumar Shridhar, Kushal Jain, Max B. Paulus, Joachim M. Buhmann, Carl Allen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ed57b293db0aab7cc30c44f45262348-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ed57b293db0aab7cc30c44f45262348-Abstract-Conference.html)

        **Abstract**:

        In principle, applying variational autoencoders (VAEs) to sequential data offers a method for controlled sequence generation, manipulation, and structured representation learning. However, training sequence VAEs is challenging: autoregressive decoders can often explain the data without utilizing the latent space, known as posterior collapse. To mitigate this, state-of-the-art models weaken' thepowerful decoder' by applying uniformly random dropout to the decoder input.We show theoretically that this removes pointwise mutual information provided by the decoder input, which is compensated for by utilizing the latent space. We then propose an adversarial training strategy to achieve information-based stochastic dropout. Compared to uniform dropout on standard text benchmark datasets, our targeted approach increases both sequence modeling performance and the information captured in the latent space.

        ----

        ## [701] On the Adversarial Robustness of Mixture of Experts

        **Authors**: *Joan Puigcerver, Rodolphe Jenatton, Carlos Riquelme, Pranjal Awasthi, Srinadh Bhojanapalli*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3effb91593c4fb42b1da1528328eff49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3effb91593c4fb42b1da1528328eff49-Abstract-Conference.html)

        **Abstract**:

        Adversarial robustness is a key desirable property of neural networks. It has been empirically shown to be affected by their sizes, with larger networks being typically more robust. Recently, \citet{bubeck2021universal} proved a lower bound on the Lipschitz constant of functions that fit the training data in terms of their number of parameters. This raises an interesting open question, do---and can---functions with more parameters, but not necessarily more computational cost, have better robustness? We study this question for sparse Mixture of Expert models (MoEs), that make it possible to scale up the model size for a roughly constant computational cost. We theoretically show that under certain conditions on the routing and the structure of the data, MoEs can have significantly smaller Lipschitz constants than their dense counterparts. The robustness of MoEs can suffer when the highest weighted experts for an input implement sufficiently different functions. We next empirically evaluate the robustness of MoEs on ImageNet using adversarial attacks and show they are indeed more robust than dense models with the same computational cost. We make key observations showing the robustness of MoEs to the choice of experts, highlighting the redundancy of experts in models trained in practice.

        ----

        ## [702] Safety Guarantees for Neural Network Dynamic Systems via Stochastic Barrier Functions

        **Authors**: *Rayan Mazouz, Karan Muvvala, Akash Ratheesh, Luca Laurenti, Morteza Lahijanian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3f1f3e38d1ce5653afb81505d3e26618-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3f1f3e38d1ce5653afb81505d3e26618-Abstract-Conference.html)

        **Abstract**:

        Neural Networks (NNs) have been successfully employed to represent the state evolution of complex dynamical systems.  Such models, referred to as NN dynamic models (NNDMs), use iterative noisy predictions of NN to estimate a distribution of system trajectories over time. Despite their accuracy, safety analysis of NNDMs is known to be a challenging problem and remains largely unexplored.  To address this issue, in this paper, we introduce a method of providing safety guarantees for NNDMs.  Our approach is based on stochastic barrier functions, whose relation with safety are analogous to that of Lyapunov functions with stability.  We first show a method of synthesizing stochastic barrier functions for NNDMs via a convex optimization problem, which in turn provides a lower bound on the system's safety probability.  A key step in our method is the employment of the recent convex approximation results for NNs to find piece-wise linear bounds, which allow the formulation of the barrier function synthesis problem as a sum-of-squares optimization program.  If the obtained safety probability is above the desired threshold, the system is certified.  Otherwise, we introduce a method of generating controls for the system that robustly minimize the unsafety probability in a minimally-invasive manner.  We exploit the convexity property of the barrier function to formulate the optimal control synthesis problem as a linear program.  Experimental results illustrate the efficacy of the method. Namely, they show that the method can scale to multi-dimensional NNDMs with multiple layers and hundreds of neurons per layer, and that the controller can significantly improve the safety probability.

        ----

        ## [703] Simple and Optimal Greedy Online Contention Resolution Schemes

        **Authors**: *Vasilis Livanos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3f45a1768ed452c1706e0a3a272e3bbc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3f45a1768ed452c1706e0a3a272e3bbc-Abstract-Conference.html)

        **Abstract**:

        Matching based markets, like ad auctions, ride-sharing, and eBay, are inherently online and combinatorial, and therefore have been extensively studied under the lens of online stochastic combinatorial optimization models. The general framework that has emerged uses Contention Resolution Schemes (CRSs) introduced by Chekuri, Vondr√°k, and Zenklusen for combinatorial problems, where one first obtains a fractional solution to a (continuous) relaxation of the objective, and then proceeds to round it. When the order of rounding is controlled by an adversary, it is called an Online Contention Resolution Scheme (OCRSs), which has been successfully applied in online settings such as posted-price mechanisms, prophet inequalities and stochastic probing.The study of greedy OCRSs against an almighty adversary has emerged as one of the most interesting problems since it gives a simple-to-implement scheme against the worst possible scenario. Intuitively, a greedy OCRS has to make all its decisions before the online process starts. We present simple $1/e$ - selectable greedy OCRSs for the single-item setting, partition matroids, and transversal matroids. This improves upon the previous state-of-the-art greedy OCRSs of [FSZ16] that achieves $1/4$ for these constraints. Finally, we show that no better competitive ratio than $1/e$ is possible, making our greedy OCRSs the best possible.

        ----

        ## [704] Composition Theorems for Interactive Differential Privacy

        **Authors**: *Xin Lyu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3f52b555967a95ee850fcecbd29ee52d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3f52b555967a95ee850fcecbd29ee52d-Abstract-Conference.html)

        **Abstract**:

        An interactive mechanism is an algorithm that stores a data set and answers adaptively chosen queries to it. The mechanism is called differentially private, if any adversary cannot distinguish whether a specific individual is in the data set by interacting with the mechanism. We study composition properties of differential privacy in concurrent compositions. In this setting, an adversary interacts with $k$ interactive mechanisms in parallel and can interleave its queries to the mechanisms arbitrarily. Previously, Vadhan and Wang [2021] proved an optimal concurrent composition theorem for pure-differential privacy. We significantly generalize and extend their results. Namely, we prove optimal parallel composition properties for several major notions of differential privacy in the literature, including approximate DP, Renyi DP, and zero-concentrated DP. Our results demonstrate that the adversary gains no advantage by interleaving its queries to independently running mechanisms. Hence, interactivity is a feature that differential privacy grants us for free.Concurrently and independently of our work, Vadhan and Zhang [2022] proved an optimal concurrent composition theorem for f-DP [Dong et al., 2022], which implies our result for the approximate DP case.

        ----

        ## [705] On Margins and Generalisation for Voting Classifiers

        **Authors**: *Felix Biggs, Valentina Zantedeschi, Benjamin Guedj*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3f8675af3da6da231c9e75b889b7f047-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3f8675af3da6da231c9e75b889b7f047-Abstract-Conference.html)

        **Abstract**:

        We study the generalisation properties of majority voting on finite ensembles of classifiers, proving margin-based generalisation bounds via the PAC-Bayes theory. These provide state-of-the-art guarantees on a number of classification tasks. Our central results leverage the Dirichlet posteriors studied recently by Zantedeschi et al. (2021) for training voting classifiers; in contrast to that work our bounds apply to non-randomised votes via the use of margins. Our contributions add perspective to the debate on the ``margins theory'' proposed by Schapire et al. (1998) for the generalisation of ensemble classifiers.

        ----

        ## [706] Effective Backdoor Defense by Exploiting Sensitivity of Poisoned Samples

        **Authors**: *Weixin Chen, Baoyuan Wu, Haoqian Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3f9bbf77fbd858e5b6e39d39fe84ed2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3f9bbf77fbd858e5b6e39d39fe84ed2e-Abstract-Conference.html)

        **Abstract**:

        Poisoning-based backdoor attacks are serious threat for training deep models on data from untrustworthy sources. Given a backdoored model, we observe that the feature representations of poisoned samples with trigger are more sensitive to transformations than those of clean samples. It inspires us to design a simple sensitivity metric, called feature consistency towards transformations (FCT), to distinguish poisoned samples from clean samples in the untrustworthy training set. Moreover, we propose two effective backdoor defense methods. Built upon a sample-distinguishment module utilizing the FCT metric, the first method trains a secure model from scratch using a two-stage secure training module. And the second method removes backdoor from a backdoored model with a backdoor removal module which alternatively unlearns the distinguished poisoned samples and relearns the distinguished clean samples. Extensive results on three benchmark datasets demonstrate the superior defense performance against eight types of backdoor attacks, to state-of-the-art backdoor defenses. Codes are available at: https://github.com/SCLBD/Effectivebackdoordefense.

        ----

        ## [707] Rethinking the Reverse-engineering of Trojan Triggers

        **Authors**: *Zhenting Wang, Kai Mei, Hailun Ding, Juan Zhai, Shiqing Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3f9bf45ea04c98ad7cb857f951f499e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3f9bf45ea04c98ad7cb857f951f499e2-Abstract-Conference.html)

        **Abstract**:

        Deep Neural Networks are vulnerable to Trojan (or backdoor) attacks. Reverse-engineering methods can reconstruct the trigger and thus identify affected models. Existing reverse-engineering methods only consider input space constraints, e.g., trigger size in the input space.Expressly, they assume the triggers are static patterns in the input space and fail to detect models with feature space triggers such as image style transformations. We observe that both input-space and feature-space Trojans are associated with feature space hyperplanes.Based on this observation, we design a novel reverse-engineering method that exploits the feature space constraint to reverse-engineer Trojan triggers. Results on four datasets and seven different attacks demonstrate that our solution effectively defends both input-space and feature-space Trojans. It outperforms state-of-the-art reverse-engineering methods and other types of defenses in both Trojaned model detection and mitigation tasks. On average, the detection accuracy of our method is 93%. For Trojan mitigation, our method can reduce the ASR (attack success rate) to only 0.26% with the BA (benign accuracy) remaining nearly unchanged. Our code can be found at https://github.com/RU-System-Software-and-Security/FeatureRE.

        ----

        ## [708] Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures

        **Authors**: *Shitong Luo, Yufeng Su, Xingang Peng, Sheng Wang, Jian Peng, Jianzhu Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3fa7d76a0dc1179f1e98d1bc62403756-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3fa7d76a0dc1179f1e98d1bc62403756-Abstract-Conference.html)

        **Abstract**:

        Antibodies are immune system proteins that protect the host by binding to specific antigens such as viruses and bacteria. The binding between antibodies and antigens is mainly determined by the complementarity-determining regions (CDR) of the antibodies. In this work, we develop a deep generative model that jointly models sequences and structures of CDRs based on diffusion probabilistic models and equivariant neural networks. Our method is the first deep learning-based method that generates antibodies explicitly targeting specific antigen structures and is one of the earliest diffusion probabilistic models for protein structures. The model is a "Swiss Army Knife" capable of sequence-structure co-design, sequence design for given backbone structures, and antibody optimization. We conduct extensive experiments to evaluate the quality of both sequences and structures of designed antibodies. We find that our model could yield competitive results in binding affinity measured by biophysical energy functions and other protein design metrics.

        ----

        ## [709] Learning single-index models with shallow neural networks

        **Authors**: *Alberto Bietti, Joan Bruna, Clayton Sanford, Min Jae Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3fb6c52aeb11e09053c16eabee74dd7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3fb6c52aeb11e09053c16eabee74dd7b-Abstract-Conference.html)

        **Abstract**:

        Single-index models are a class of functions given by an unknown univariate ``link'' function applied to an unknown one-dimensional projection of the input. These models are particularly relevant in high dimension, when the data might present low-dimensional structure that learning algorithms should adapt to. While several statistical aspects of this model, such as the sample complexity of recovering the relevant (one-dimensional) subspace, are well-understood, they rely on tailored algorithms that exploit the specific structure of the target function. In this work, we introduce a natural class of shallow neural networks and study its ability to learn single-index models via gradient flow. More precisely, we consider shallow networks in which biases of the neurons are frozen at random initialization. We show that the corresponding optimization landscape is benign, which in turn leads to generalization guarantees that match the near-optimal sample complexity of dedicated semi-parametric methods.

        ----

        ## [710] Information bottleneck theory of high-dimensional regression: relevancy, efficiency and optimality

        **Authors**: *Vudtiwat Ngampruetikorn, David J. Schwab*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3fbcfbc2b4009ae8dfa17a562532d123-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3fbcfbc2b4009ae8dfa17a562532d123-Abstract-Conference.html)

        **Abstract**:

        Avoiding overfitting is a central challenge in machine learning, yet many large neural networks readily achieve zero training loss. This puzzling contradiction necessitates new approaches to the study of overfitting. Here we quantify overfitting via residual information, defined as the bits in fitted models that encode noise in training data. Information efficient learning algorithms minimize residual information while maximizing the relevant bits, which are predictive of the unknown generative models. We solve this optimization to obtain the information content of optimal algorithms for a linear regression problem and compare it to that of randomized ridge regression. Our results demonstrate the fundamental trade-off between residual and relevant information and characterize the relative information efficiency of randomized regression with respect to optimal algorithms. Finally, using results from random matrix theory, we reveal the information complexity of learning a linear map in high dimensions and unveil information-theoretic analogs of double and multiple descent phenomena.

        ----

        ## [711] RainNet: A Large-Scale Imagery Dataset and Benchmark for Spatial Precipitation Downscaling

        **Authors**: *Xuanhong Chen, Kairui Feng, Naiyuan Liu, Bingbing Ni, Yifan Lu, Zhengyan Tong, Ziang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html)

        **Abstract**:

        AI-for-science approaches have been applied to solve scientific problems (e.g., nuclear fusion, ecology, genomics, meteorology) and have achieved highly promising results. Spatial precipitation downscaling is one of the most important meteorological problem and urgently requires the participation of AI. However, the lack of a well-organized and annotated large-scale dataset hinders the training and verification of more effective and advancing deep-learning models for precipitation downscaling. To alleviate these obstacles, we present the first large-scale spatial precipitation downscaling dataset named RainNet, which contains more than 62,400 pairs of high-quality low/high-resolution precipitation maps for over 17 years, ready to help the evolution of deep learning models in precipitation downscaling. Specifically, the precipitation maps carefully collected in RainNet cover various meteorological phenomena (e.g., hurricane, squall), which is of great help to improve the model generalization ability. In addition, the map pairs in RainNet are organized in the form of image sequences (720 maps per month or 1 map/hour), showing complex physical properties, e.g., temporal misalignment, temporal sparse, and fluid properties. Furthermore, two deep-learning-oriented metrics are specifically introduced to evaluate or verify the comprehensive performance of the trained model (e.g., prediction maps reconstruction accuracy). To illustrate the applications of RainNet, 14 state-of-the-art models, including deep models and traditional approaches, are evaluated. To fully explore potential downscaling solutions, we propose an implicit physical estimation benchmark framework to learn the above characteristics. Extensive experiments demonstrate the value of RainNet in training and evaluating downscaling models. Our dataset is available at https://neuralchen.github.io/RainNet/.

        ----

        ## [712] Dataset Distillation using Neural Feature Regression

        **Authors**: *Yongchao Zhou, Ehsan Nezhadarya, Jimmy Ba*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3fe2a777282299ecb4f9e7ebb531f0ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3fe2a777282299ecb4f9e7ebb531f0ab-Abstract-Conference.html)

        **Abstract**:

        Dataset distillation aims to learn a small synthetic dataset that preserves most of the information from the original dataset. Dataset distillation can be formulated as a bi-level meta-learning problem where the outer loop optimizes the meta-dataset and the inner loop trains a model on the distilled data. Meta-gradient computation is one of the key challenges in this formulation, as differentiating through the inner loop learning procedure introduces significant computation and memory costs. In this paper, we address these challenges using neural Feature Regression with Pooling (FRePo), achieving the state-of-the-art performance with an order of magnitude less memory requirement and two orders of magnitude faster training than previous methods. The proposed algorithm is analogous to truncated backpropagation through time with a pool of models to alleviate various types of overfitting in dataset distillation. FRePo significantly outperforms the previous methods on CIFAR100, Tiny ImageNet, and ImageNet-1K. Furthermore, we show that high-quality distilled data can greatly improve various downstream applications, such as continual learning and membership inference defense. Please check out our webpage at https://sites.google.com/view/frepo.

        ----

        ## [713] ZeroC: A Neuro-Symbolic Model for Zero-shot Concept Recognition and Acquisition at Inference Time

        **Authors**: *Tailin Wu, Megan Tjandrasuwita, Zhengxuan Wu, Xuelin Yang, Kevin Liu, Rok Sosic, Jure Leskovec*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3ff48dde82306fe8f26f3e51dd1054d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3ff48dde82306fe8f26f3e51dd1054d7-Abstract-Conference.html)

        **Abstract**:

        Humans have the remarkable ability to recognize and acquire novel visual concepts in a zero-shot manner. Given a high-level, symbolic description of a novel concept in terms of previously learned visual concepts and their relations, humans can recognize novel concepts without seeing any examples. Moreover, they can acquire new concepts by parsing and communicating symbolic structures using learned visual concepts and relations. Endowing these capabilities in machines is pivotal in improving their generalization capability at inference time. In this work, we introduce Zero-shot Concept Recognition and Acquisition (ZeroC), a neuro-symbolic architecture that can recognize and acquire novel concepts in a zero-shot way.  ZeroC represents concepts as graphs of constituent concept models (as nodes) and their relations (as edges). To allow inference time composition, we employ energy-based models (EBMs) to model concepts and relations. We design ZeroC architecture so that it allows a one-to-one mapping between a symbolic graph structure of a concept and its corresponding EBM, which for the first time, allows acquiring new concepts, communicating its graph structure, and applying it to classification and detection tasks (even across domains) at inference time. We introduce algorithms for learning and inference with ZeroC. We evaluate ZeroC on a challenging grid-world dataset which is designed to probe zero-shot concept recognition and acquisition, and demonstrate its capability.

        ----

        ## [714] Theoretically Better and Numerically Faster Distributed Optimization with Smoothness-Aware Quantization Techniques

        **Authors**: *Bokun Wang, Mher Safaryan, Peter Richtárik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/401de6a666d7672757bdadfc53c3c123-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/401de6a666d7672757bdadfc53c3c123-Abstract-Conference.html)

        **Abstract**:

        To address the high communication costs of distributed machine learning, a large body of work has been devoted in recent years to designing various compression strategies, such as sparsification and quantization, and optimization algorithms capable of using them. Recently, Safaryan et al. (2021) pioneered a dramatically different compression design approach: they first use the local training data to form local smoothness matrices and then propose to design a compressor capable of exploiting the smoothness information contained therein. While this novel approach leads to substantial savings in communication, it is limited to sparsification as it crucially depends on the linearity of the compression operator. In this work, we generalize their smoothness-aware compression strategy to arbitrary unbiased compression operators, which also include sparsification. Specializing our results to stochastic quantization, we guarantee significant savings in communication complexity compared to standard quantization. In particular, we prove that block quantization with $n$ blocks theoretically outperforms single block quantization, leading to a reduction in communication complexity by an $\mathcal{O}(n)$ factor, where $n$ is the number of nodes in the distributed system. Finally, we provide extensive numerical evidence with convex optimization problems that our smoothness-aware quantization strategies outperform existing quantization schemes as well as the aforementioned smoothness-aware sparsification strategies with respect to three evaluation metrics: the number of iterations, the total amount of bits communicated, and wall-clock time.

        ----

        ## [715] Sparse Structure Search for Delta Tuning

        **Authors**: *Shengding Hu, Zhen Zhang, Ning Ding, Yadao Wang, Yasheng Wang, Zhiyuan Liu, Maosong Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4027fc4573ec9114182d1fef37a6321a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4027fc4573ec9114182d1fef37a6321a-Abstract-Conference.html)

        **Abstract**:

        Adapting large pre-trained models (PTMs) through fine-tuning imposes prohibitive computational and storage burdens. Recent studies of delta tuning (DT), i.e., parameter-efficient tuning,  find that only optimizing a small portion of parameters conditioned on PTMs could yield on-par performance compared to conventional fine-tuning. Generally, DT methods exquisitely design delta modules (DT modules) which could be applied to arbitrary fine-grained positions inside PTMs. However, the effectiveness of these fine-grained positions largely relies on sophisticated manual designation, thereby usually producing sub-optimal results. In contrast to the manual designation, we explore constructing DT modules in an automatic manner. We automatically \textbf{S}earch for the \textbf{S}parse \textbf{S}tructure of \textbf{Delta} Tuning (S$^3$Delta).   Based on a unified framework of various DT methods, S$^3$Delta conducts the differentiable DT structure search through bi-level optimization and proposes shifted global sigmoid method to explicitly control the number of trainable parameters.  Extensive experiments show that S$^3$Delta surpasses manual and random structures with less trainable parameters. The searched structures preserve more than 99\% fine-tuning performance with 0.01\% trainable parameters. Moreover, the advantage of S$^3$Delta is amplified with extremely low trainable parameters budgets (0.0009\%$\sim$0.01\%). The searched structures are transferable and explainable, providing suggestions and guidance for the future design of DT methods. Our codes are publicly available at \url{https://github.com/thunlp/S3Delta}.

        ----

        ## [716] On the Safety of Interpretable Machine Learning: A Maximum Deviation Approach

        **Authors**: *Dennis Wei, Rahul Nair, Amit Dhurandhar, Kush R. Varshney, Elizabeth Daly, Moninder Singh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/402e12102d6ec3ea3df40ce1b23d423a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/402e12102d6ec3ea3df40ce1b23d423a-Abstract-Conference.html)

        **Abstract**:

        Interpretable and explainable machine learning has seen a recent surge of interest. We focus on safety as a key motivation behind the surge and make the relationship between interpretability and safety more quantitative. Toward assessing safety, we introduce the concept of maximum deviation via an optimization problem to find the largest deviation of a supervised learning model from a reference model regarded as safe. We then show how interpretability facilitates this safety assessment. For models including decision trees, generalized linear and additive models, the maximum deviation can be computed exactly and efficiently. For tree ensembles, which are not regarded as interpretable, discrete optimization techniques can still provide informative bounds. For a broader class of piecewise Lipschitz functions, we leverage the multi-armed bandit literature to show that interpretability produces tighter (regret) bounds on the maximum deviation. We present case studies, including one on mortgage approval, to illustrate our methods and the insights about models that may be obtained from deviation maximization.

        ----

        ## [717] Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting

        **Authors**: *Yong Liu, Haixu Wu, Jianmin Wang, Mingsheng Long*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html)

        **Abstract**:

        Transformers have shown great power in time series forecasting due to their global-range modeling ability. However, their performance can degenerate terribly on non-stationary real-world data in which the joint distribution changes over time. Previous studies primarily adopt stationarization to attenuate the non-stationarity of original series for better predictability. But the stationarized series deprived of inherent non-stationarity can be less instructive for real-world bursty events forecasting. This problem, termed over-stationarization in this paper, leads Transformers to generate indistinguishable temporal attentions for different series and impedes the predictive capability of deep models. To tackle the dilemma between series predictability and model capability, we propose Non-stationary Transformers as a generic framework with two interdependent modules: Series Stationarization and De-stationary Attention. Concretely, Series Stationarization unifies the statistics of each input and converts the output with restored statistics for better predictability. To address the over-stationarization problem, De-stationary Attention is devised to recover the intrinsic non-stationary information into temporal dependencies by approximating distinguishable attentions learned from raw series. Our Non-stationary Transformers framework consistently boosts mainstream Transformers by a large margin, which reduces MSE by 49.43% on Transformer, 47.34% on Informer, and 46.89% on Reformer, making them the state-of-the-art in time series forecasting. Code is available at this repository: https://github.com/thuml/Nonstationary_Transformers.

        ----

        ## [718] Risk-Driven Design of Perception Systems

        **Authors**: *Anthony Corso, Sydney M. Katz, Craig Innes, Xin Du, Subramanian Ramamoorthy, Mykel J. Kochenderfer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/40739b3bb584c117b3e2f418d17f63a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/40739b3bb584c117b3e2f418d17f63a1-Abstract-Conference.html)

        **Abstract**:

        Modern autonomous systems rely on perception modules to process complex sensor measurements into state estimates. These estimates are then passed to a controller, which uses them to make safety-critical decisions. It is therefore important that we design perception systems to minimize errors that reduce the overall safety of the system. We develop a risk-driven approach to designing perception systems that accounts for the effect of perceptual errors on the performance of the fully-integrated, closed-loop system. We formulate a risk function to quantify the effect of a given perceptual error on overall safety, and show how we can use it to design safer perception systems by including a risk-dependent term in the loss function and generating training data in risk-sensitive regions. We evaluate our techniques on a realistic vision-based aircraft detect and avoid application and show that risk-driven design reduces collision risk by 37% over a baseline system.

        ----

        ## [719] A Simple Approach to Automated Spectral Clustering

        **Authors**: *Jicong Fan, Yiheng Tu, Zhao Zhang, Mingbo Zhao, Haijun Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/407fb8c5f3fda374c57d1bb18313ea5d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/407fb8c5f3fda374c57d1bb18313ea5d-Abstract-Conference.html)

        **Abstract**:

        The performance of spectral clustering heavily relies on the quality of affinity matrix. A variety of affinity-matrix-construction (AMC) methods have been proposed but they have hyperparameters to determine beforehand, which requires strong experience and leads to difficulty in real applications, especially when the inter-cluster similarity is high and/or the dataset is large.  In addition, we often need to choose different AMC methods for different datasets, which still depends on experience. To solve these two challenging problems,  in this paper, we present a simple yet effective method for automated spectral clustering. First, we propose to find the most reliable affinity matrix via grid search or Bayesian optimization among a set of candidates given by different AMC methods with different hyperparameters, where the reliability is quantified by the \textit{relative-eigen-gap} of graph Laplacian introduced in this paper. Second, we propose a fast and accurate AMC method based on least squares representation and thresholding and prove its effectiveness theoretically.  Finally, we provide a large-scale extension for the automated spectral clustering method, of which the time complexity is linear with the number of data points. Extensive experiments of natural image clustering show that our method is more versatile, accurate, and efficient than baseline methods.

        ----

        ## [720] Joint Entropy Search for Multi-Objective Bayesian Optimization

        **Authors**: *Ben Tu, Axel Gandy, Nikolas Kantas, Behrang Shafei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4086fe59dc3584708468fba0e459f6a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4086fe59dc3584708468fba0e459f6a7-Abstract-Conference.html)

        **Abstract**:

        Many real-world problems can be phrased as a multi-objective optimization problem, where the goal is to identify the best set of compromises between the competing objectives. Multi-objective Bayesian optimization (BO) is a sample efficient strategy that can be deployed to solve these vector-valued optimization problems where access is limited to a number of noisy objective function evaluations. In this paper, we propose a novel information-theoretic acquisition function for BO called Joint Entropy Search (JES), which considers the joint information gain for the optimal set of inputs and outputs. We present several analytical approximations to the JES acquisition function and also introduce an extension to the batch setting. We showcase the effectiveness of this new approach on a range of synthetic and real-world problems in terms of the hypervolume and its weighted variants.

        ----

        ## [721] Subgroup Robustness Grows On Trees: An Empirical Baseline Investigation

        **Authors**: *Josh Gardner, Zoran Popovic, Ludwig Schmidt*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/408cf1a1d9ff35d5fea7075565dbf434-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/408cf1a1d9ff35d5fea7075565dbf434-Abstract-Conference.html)

        **Abstract**:

        Researchers have proposed many methods for fair and robust machine learning, but comprehensive empirical evaluation of their subgroup robustness is lacking. In this work, we address this gap in the context of tabular data, where sensitive subgroups are clearly-defined, real-world fairness problems abound, and prior works often do not compare to state-of-the-art tree-based models as baselines. We conduct an empirical comparison of several previously-proposed methods for fair and robust learning  alongside state-of-the-art tree-based methods  and other baselines. Via experiments with more than $340{,}000$ model configurations on eight datasets, we show that tree-based methods have strong subgroup robustness, even when compared to robustness- and fairness-enhancing methods. Moreover, the best tree-based models tend to show good performance over a range of metrics, while robust or group-fair models can show brittleness, with significant performance differences across different metrics for a fixed model. We also demonstrate that tree-based models show less sensitivity to hyperparameter configurations, and are less costly to train. Our work suggests that tree-based ensemble models make an effective baseline for tabular data, and are a sensible default when subgroup robustness is desired. See https://github.com/jpgard/subgroup-robustness-grows-on-trees for code to reproduce our experiments and detailed experimental results.

        ----

        ## [722] Robustness to Unbounded Smoothness of Generalized SignSGD

        **Authors**: *Michael Crawshaw, Mingrui Liu, Francesco Orabona, Wei Zhang, Zhenxun Zhuang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/40924475a9bf768bdac3725e67745283-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/40924475a9bf768bdac3725e67745283-Abstract-Conference.html)

        **Abstract**:

        Traditional analyses in non-convex optimization typically rely on the smoothness assumption, namely requiring the gradients to be Lipschitz. However, recent evidence shows that this smoothness condition does not capture the properties of some deep learning objective functions, including the ones involving Recurrent Neural Networks and LSTMs. Instead, they satisfy a much more relaxed condition, with potentially unbounded smoothness. Under this relaxed assumption, it has been theoretically and empirically shown that the gradient-clipped SGD has an advantage over the vanilla one. In this paper, we show that clipping is not indispensable for Adam-type algorithms in tackling such scenarios: we theoretically prove that a generalized SignSGD algorithm can obtain similar convergence rates as SGD with clipping but does not need explicit clipping at all. This family of algorithms on one end recovers SignSGD and on the other end closely resembles the popular Adam algorithm. Our analysis underlines the critical role that momentum plays in analyzing SignSGD-type and Adam-type algorithms: it not only reduces the effects of noise, thus removing the need for large mini-batch in previous analyses of SignSGD-type algorithms, but it also substantially reduces the effects of unbounded smoothness and gradient norms. To the best of our knowledge, this work is the first one showing the benefit of Adam-type algorithms compared with non-adaptive gradient algorithms such as gradient descent in the unbounded smoothness setting. We also compare these algorithms with popular optimizers on a set of deep learning tasks, observing that we can match the performance of Adam while beating others.

        ----

        ## [723] GhostNetV2: Enhance Cheap Operation with Long-Range Attention

        **Authors**: *Yehui Tang, Kai Han, Jianyuan Guo, Chang Xu, Chao Xu, Yunhe Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/40b60852a4abdaa696b5a1a78da34635-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/40b60852a4abdaa696b5a1a78da34635-Abstract-Conference.html)

        **Abstract**:

        Light-weight convolutional neural networks (CNNs) are specially designed for applications on mobile devices with faster inference speed. The convolutional operation can only capture local information in a window region, which prevents  performance from being further improved. Introducing self-attention into convolution can capture global information well, but it will largely encumber the actual speed. In this paper, we propose a hardware-friendly attention mechanism (dubbed DFC attention) and then present a new GhostNetV2 architecture for mobile applications. The proposed DFC attention is constructed based on fully-connected layers, which can not only execute fast on common hardware but also capture the dependence between long-range pixels. We further revisit the expressiveness bottleneck in previous GhostNet and propose to enhance expanded features produced by cheap operations with DFC attention, so that a GhostNetV2 block can aggregate local and long-range information simultaneously. Extensive experiments demonstrate the superiority of GhostNetV2 over existing architectures. For example, it achieves 75.3% top-1 accuracy on ImageNet with 167M FLOPs, significantly suppressing GhostNetV1 (74.5%) with a similar computational cost. The source code will be available at https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnetv2_pytorch and https://gitee.com/mindspore/models/tree/master/research/cv/ghostnetv2.

        ----

        ## [724] Analyzing Sharpness along GD Trajectory: Progressive Sharpening and Edge of Stability

        **Authors**: *Zixuan Wang, Zhouzi Li, Jian Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/40bb79c081828bebdc39d65a82367246-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/40bb79c081828bebdc39d65a82367246-Abstract-Conference.html)

        **Abstract**:

        Recent findings demonstrate that modern neural networks trained by full-batch gradient descent typically enter a regime called Edge of Stability (EOS). In this regime, the sharpness, i.e., the maximum Hessian eigenvalue, first increases to the value 2/(step size) (the progressive sharpening phase) and then oscillates around this value (the EOS phase). This paper aims to analyze the GD dynamics and the sharpness along the optimization trajectory.Our analysis naturally divides the GD trajectory into four phases depending on the change in the sharpness value. We empirically identify the norm of output layer weight as an interesting indicator of the sharpness dynamics. Based on this empirical observation, we attempt to theoretically and empirically explain the dynamics of various key quantities that lead to the change of the sharpness in each phase of EOS. Moreover, based on certain assumptions, we provide a theoretical proof of the sharpness behavior in the EOS regime in two-layer fully-connected linear neural networks. We also discuss some other empirical findings and the limitation of our theoretical results.

        ----

        ## [725] You Never Stop Dancing: Non-freezing Dance Generation via Bank-constrained Manifold Projection

        **Authors**: *Jiangxin Sun, Chunyu Wang, Huang Hu, Hanjiang Lai, Zhi Jin, Jian-Fang Hu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/40bfe6177e8aed33c982264cf9e6e62c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/40bfe6177e8aed33c982264cf9e6e62c-Abstract-Conference.html)

        **Abstract**:

        One of the most overlooked challenges in dance generation is that the auto-regressive frameworks are prone to freezing motions due to noise accumulation. In this paper, we present two modules that can be plugged into the existing models to enable them to generate non-freezing and high fidelity dances. Since the high-dimensional motion data are easily swamped by noise, we propose to learn a low-dimensional manifold representation by an auto-encoder with a bank of latent codes, which can be used to reduce the noise in the predicted motions, thus preventing from freezing. We further extend the bank to provide explicit priors about the future motions to disambiguate motion prediction, which helps the predictors to generate motions with larger magnitude and higher fidelity than possible before. Extensive experiments on AIST++, a public large-scale 3D dance motion benchmark, demonstrate that our method notably outperforms the baselines in terms of quality, diversity and time length.

        ----

        ## [726] Nonnegative Tensor Completion via Integer Optimization

        **Authors**: *Caleb Bugg, Chen Chen, Anil Aswani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/40cf260d0c2df13e4286cc0fc0972618-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/40cf260d0c2df13e4286cc0fc0972618-Abstract-Conference.html)

        **Abstract**:

        Unlike matrix completion, tensor completion does not have an algorithm that is known to achieve the information-theoretic sample complexity rate. This paper develops a new algorithm for the special case of completion for nonnegative tensors. We prove that our algorithm converges in a linear (in numerical tolerance) number of oracle steps, while achieving the information-theoretic rate. Our approach is to define a new norm for nonnegative tensors using the gauge of a particular 0-1 polytope; integer linear programming can, in turn, be used to solve linear separation problems over this polytope. We combine this insight with a variant of the Frank-Wolfe algorithm to construct our numerical algorithm, and we demonstrate its effectiveness and scalability through computational experiments using a laptop on tensors with up to one-hundred million entries.

        ----

        ## [727] LION: Latent Point Diffusion Models for 3D Shape Generation

        **Authors**: *Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/40e56dabe12095a5fc44a6e4c3835948-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/40e56dabe12095a5fc44a6e4c3835948-Abstract-Conference.html)

        **Abstract**:

        Denoising diffusion models (DDMs) have shown promising results in 3D point cloud synthesis. To advance 3D DDMs and make them useful for digital artists, we require (i) high generation quality, (ii) flexibility for manipulation and applications such as conditional synthesis and shape interpolation, and (iii) the ability to output smooth surfaces or meshes. To this end, we introduce the hierarchical Latent Point Diffusion Model (LION) for 3D shape generation. LION is set up as a variational autoencoder (VAE) with a hierarchical latent space that combines a global shape latent representation with a point-structured latent space. For generation, we train two hierarchical DDMs in these latent spaces. The hierarchical VAE approach boosts performance compared to DDMs that operate on point clouds directly, while the point-structured latents are still ideally suited for DDM-based modeling. Experimentally, LION achieves state-of-the-art generation performance on multiple ShapeNet benchmarks. Furthermore, our VAE framework allows us to easily use LION for different relevant tasks: LION excels at multimodal shape denoising and voxel-conditioned synthesis, and it can be adapted for text- and image-driven 3D generation. We also demonstrate shape autoencoding and latent shape interpolation, and we augment LION with modern surface reconstruction techniques to generate smooth 3D meshes. We hope that LION provides a powerful tool for artists working with 3D shapes due to its high-quality generation, flexibility, and surface reconstruction. Project page and code: https://nv-tlabs.github.io/LION.

        ----

        ## [728] Distribution-Informed Neural Networks for Domain Adaptation Regression

        **Authors**: *Jun Wu, Jingrui He, Sheng Wang, Kaiyu Guan, Elizabeth A. Ainsworth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/410bbba8388369d8bb5875544d1d4428-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/410bbba8388369d8bb5875544d1d4428-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the problem of domain adaptation regression, which learns a regressor for a target domain by leveraging the knowledge from a relevant source domain. We start by proposing a distribution-informed neural network, which aims to build distribution-aware relationship of inputs and outputs from different domains. This allows us to develop a simple domain adaptation regression framework, which subsumes popular domain adaptation approaches based on domain invariant representation learning, reweighting, and adaptive Gaussian process. The resulting findings not only explain the connections of existing domain adaptation approaches, but also motivate the efficient training of domain adaptation approaches with overparameterized neural networks. We also analyze the convergence and generalization error bound of our framework based on the distribution-informed neural network. Specifically, our generalization bound focuses explicitly on the maximum mean discrepancy in the RKHS induced by the neural tangent kernel of distribution-informed neural network. This is in sharp contrast to the existing work which relies on domain discrepancy in the latent feature space heuristically formed by one or several hidden neural layers. The efficacy of our framework is also empirically verified on a variety of domain adaptation regression benchmarks.

        ----

        ## [729] Momentum Adversarial Distillation: Handling Large Distribution Shifts in Data-Free Knowledge Distillation

        **Authors**: *Kien Do, Hung Le, Dung Nguyen, Dang Nguyen, Haripriya Harikumar, Truyen Tran, Santu Rana, Svetha Venkatesh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/41128e5b3a7622da5b17588757599077-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/41128e5b3a7622da5b17588757599077-Abstract-Conference.html)

        **Abstract**:

        Data-free Knowledge Distillation (DFKD) has attracted attention recently thanks to its appealing capability of transferring knowledge from a teacher network to a student network without using training data. The main idea is to use a generator to synthesize data for training the student. As the generator gets updated, the distribution of synthetic data will change. Such distribution shift could be large if the generator and the student are trained adversarially, causing the student to forget the knowledge it acquired at the previous steps. To alleviate this problem, we propose a simple yet effective method called Momentum Adversarial Distillation (MAD) which maintains an exponential moving average (EMA) copy of the generator and uses synthetic samples from both the generator and the EMA generator to train the student. Since the EMA generator can be considered as an ensemble of the generator's old versions and often undergoes a smaller change in updates compared to the generator, training on its synthetic samples can help the student recall the past knowledge and prevent the student from adapting too quickly to the new updates of the generator. Our experiments on six benchmark datasets including big datasets like ImageNet and Places365 demonstrate the superior performance of MAD over competing methods for handling the large distribution shift problem. Our method also compares favorably to existing DFKD methods and even achieves state-of-the-art results in some cases.

        ----

        ## [730] Hard ImageNet: Segmentations for Objects with Strong Spurious Cues

        **Authors**: *Mazda Moayeri, Sahil Singla, Soheil Feizi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4120362f930b0b683cd30b71af56fad1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4120362f930b0b683cd30b71af56fad1-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Deep classifiers are known to rely on spurious features, leading to reduced generalization. The severity of this problem varies significantly by class. We identify $15$ classes in ImageNet with very strong spurious cues, and collect segmentation masks for these challenging objects to form \emph{Hard ImageNet}. Leveraging noise, saliency, and ablation based metrics, we demonstrate that models rely on spurious features in Hard ImageNet far more than in RIVAL10, an ImageNet analog to CIFAR10. We observe Hard ImageNet objects are less centered and occupy much less space in their images than RIVAL10 objects, leading to greater spurious feature reliance. Further, we use robust neural features to automatically rank our images based on the degree of spurious cues present. Comparing images with high and low rankings within a class reveals the exact spurious features models rely upon, and shows reduced performance when spurious features are absent. With Hard ImageNet's image rankings, object segmentations, and our extensive evaluation suite, the community can begin to address the problem of learning to detect challenging objects \emph{for the right reasons}, despite the presence of strong spurious cues.

        ----

        ## [731] VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

        **Authors**: *Zhan Tong, Yibing Song, Jue Wang, Limin Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html)

        **Abstract**:

        Pre-training video transformers on extra large-scale datasets is generally required to achieve premier performance on relatively small datasets. In this paper, we show that video masked autoencoders (VideoMAE) are data-efficient learners for self-supervised video pre-training (SSVP). We are inspired by the recent ImageMAE and propose customized video tube masking with an extremely high ratio. This simple design makes video reconstruction a more challenging and meaningful self-supervision task, thus encouraging extracting more effective video representations during the pre-training process. We obtain three important findings with VideoMAE: (1) An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance for VideoMAE. The temporally redundant video content enables higher masking ratio than that of images. (2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data. This is partially ascribed to the challenging task of video reconstruction to enforce high-level structure learning. (3) VideoMAE shows that data quality is more important than data quantity for SSVP. Domain shift between pre-training and target datasets is an important factor. Notably, our VideoMAE with the vanilla ViT backbone can achieve 87.4% on Kinects-400, 75.4% on Something-Something V2, 91.3% on UCF101, and 62.6% on HMDB51, without using any extra data. Code is available at https://github.com/MCG-NJU/VideoMAE.

        ----

        ## [732] Scalable design of Error-Correcting Output Codes using Discrete Optimization with Graph Coloring

        **Authors**: *Samarth Gupta, Saurabh Amin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/41792f041a3a0774418791993cf887fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/41792f041a3a0774418791993cf887fe-Abstract-Conference.html)

        **Abstract**:

        We study the problem of scalable design of Error-Correcting Output Codes (ECOC) for multi-class classification. Prior works on ECOC-based classifiers are limited to codebooks with small number of rows (classes) or columns, and do not provide optimality guarantees for the codebook design problem. We address these limitations by developing a codebook design approach based on a Mixed-Integer Quadratically Constrained Program (MIQCP). This discrete formulation is naturally suited for maximizing the error-correction capability of ECOC-based classifiers and incorporates various design criteria in a flexible manner. Our solution approach is tractable in that it incrementally increases the codebook size by adding columns to maximize the gain in error-correcting capability. In particular, we show that the maximal gain in error-correction can be upper bounded by solving a graph-coloring problem.  As a result, we can efficiently generate near-optimal codebooks for very large problem instances. These codebooks provide competitive multi-class classification performance on small class datasets such as MNIST and CIFAR10. Moreover, by leveraging transfer-learned binary classifiers, we achieve better classification performance over transfer-learned multi-class CNNs on large class datasets such as CIFAR100, Caltech-101/256. Our results highlight the advantages of simple and modular ECOC-based classifiers in improving classification accuracy without the risk of overfitting.

        ----

        ## [733] A New Family of Generalization Bounds Using Samplewise Evaluated CMI

        **Authors**: *Fredrik Hellström, Giuseppe Durisi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/41b6674c28a9b93ec8d22a53ca25bc3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/41b6674c28a9b93ec8d22a53ca25bc3b-Abstract-Conference.html)

        **Abstract**:

        We present a new family of information-theoretic generalization bounds, in which the training loss and the population loss are compared through a jointly convex function. This function is upper-bounded in terms of the disintegrated, samplewise, evaluated conditional mutual information (CMI), an information measure that depends on the losses incurred by the selected hypothesis, rather than on the hypothesis itself, as is common in probably approximately correct (PAC)-Bayesian results. We demonstrate the generality of this framework by recovering and extending previously known information-theoretic bounds. Furthermore, using the evaluated CMI, we derive a samplewise, average version of Seeger's PAC-Bayesian bound, where the convex function is the binary KL divergence. In some scenarios, this novel bound results in a tighter characterization of the population loss of deep neural networks than previous bounds. Finally, we derive high-probability versions of some of these average bounds. We demonstrate the unifying nature of the evaluated CMI bounds by using them to recover average and high-probability generalization bounds for multiclass classification with finite Natarajan dimension.

        ----

        ## [734] Maximum-Likelihood Inverse Reinforcement Learning with Finite-Time Guarantees

        **Authors**: *Siliang Zeng, Chenliang Li, Alfredo Garcia, Mingyi Hong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/41bd71e7bf7f9fe68f1c936940fd06bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/41bd71e7bf7f9fe68f1c936940fd06bd-Abstract-Conference.html)

        **Abstract**:

        Inverse reinforcement learning (IRL) aims to recover the reward function and the associated optimal policy that best fits observed sequences of states and actions implemented by an expert. Many algorithms for IRL have an inherent nested structure: the inner loop finds the optimal policy given parametrized rewards while the outer loop updates the estimates towards optimizing a measure of fit. For high dimensional environments such nested-loop structure entails a significant computational burden. To reduce the computational burden of a nested loop, novel methods such as SQIL \cite{reddy2019sqil} and IQ-Learn \cite{garg2021iq} emphasize policy estimation at the expense of reward estimation accuracy. However, without accurate estimated rewards, it is not possible to do counterfactual analysis such as predicting the optimal policy under different environment dynamics and/or learning new tasks. In this paper we develop a novel {\em single-loop} algorithm for IRL that does not compromise reward estimation accuracy. In the proposed algorithm, each policy improvement step is followed by a stochastic gradient step for likelihood maximization. We show that the proposed algorithm provably converges to a stationary solution with a finite-time guarantee. If the reward is parameterized linearly we show the identified solution corresponds to the solution of the maximum entropy IRL problem. Finally, by using robotics control problems in Mujoco and their transfer settings, we show that the proposed algorithm achieves superior performance compared with other IRL and imitation learning benchmarks.

        ----

        ## [735] Lost in Latent Space: Examining failures of disentangled models at combinatorial generalisation

        **Authors**: *Milton Llera Montero, Jeffrey S. Bowers, Rui Ponte Costa, Casimir J. H. Ludwig, Gaurav Malhotra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/41ca8a0eb2bc4927a499b910934b9b81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/41ca8a0eb2bc4927a499b910934b9b81-Abstract-Conference.html)

        **Abstract**:

        Recent research has shown that generative models with highly disentangled representations fail to generalise to unseen combination of generative factor values. These findings contradict earlier research which showed improved performance in out-of-training distribution settings when compared to entangled representations. Additionally, it is not clear if the reported failures are due to (a) encoders failing to map novel combinations to the proper regions of the latent space, or (b) novel combinations being mapped correctly but the decoder is unable to render the correct output for the unseen combinations. We investigate these alternatives by testing several models on a range of datasets and training settings. We find that (i) when models fail, their encoders also fail to map unseen combinations to correct regions of the latent space and (ii) when models succeed, it is either because the test conditions do not exclude enough examples, or because excluded cases involve combinations of object properties with it's shape. We argue that to generalise properly, models not only need to capture factors of variation, but also understand how to invert the process that causes the visual stimulus.

        ----

        ## [736] MultiGuard: Provably Robust Multi-label Classification against Adversarial Examples

        **Authors**: *Jinyuan Jia, Wenjie Qu, Neil Zhenqiang Gong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/41fb2ecb5b7d1b505bca787de0a603dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/41fb2ecb5b7d1b505bca787de0a603dc-Abstract-Conference.html)

        **Abstract**:

        Multi-label classification, which predicts a set of labels for an input, has many applications.  However, multiple recent studies showed that multi-label classification is vulnerable to adversarial examples. In particular, an attacker can manipulate the labels predicted by a multi-label classifier for an input via adding carefully crafted, human-imperceptible perturbation to it. Existing provable defenses for multi-class classification achieve sub-optimal provable robustness guarantees when generalized to multi-label classification. In this work, we propose MultiGuard, the first provably robust defense against adversarial examples to multi-label classification. Our MultiGuard leverages randomized smoothing, which is the state-of-the-art technique to build provably robust classifiers. Specifically, given an arbitrary multi-label classifier, our MultiGuard builds a smoothed multi-label classifier via adding random noise to the input. We consider isotropic Gaussian noise in this work. Our major theoretical contribution is that we show a certain number of ground truth labels of an input are provably in the set of labels predicted by our MultiGuard when the $\ell_2$-norm of the adversarial perturbation added to the input is bounded. Moreover, we design an algorithm to compute our provable robustness guarantees. Empirically, we evaluate our MultiGuard on VOC 2007, MS-COCO, and NUS-WIDE benchmark datasets. Our code is available at: https://github.com/quwenjie/MultiGuard

        ----

        ## [737] On Measuring Excess Capacity in Neural Networks

        **Authors**: *Florian Graf, Sebastian Zeng, Bastian Rieck, Marc Niethammer, Roland Kwitt*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/420492060687ca7448398c4c3fa10366-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/420492060687ca7448398c4c3fa10366-Abstract-Conference.html)

        **Abstract**:

        We study the excess capacity of deep networks in the context of supervised classification. That is, given a capacity measure of the underlying hypothesis class - in our case, empirical Rademacher complexity - to what extent can we (a priori) constrain this class while retaining an empirical error on a par with the unconstrained regime? To assess excess capacity in modern architectures (such as residual networks), we extend and unify prior Rademacher complexity bounds to accommodate function composition and addition, as well as the structure of convolutions. The capacity-driving terms in our bounds are the Lipschitz constants of the layers and a (2,1) group norm distance to the initializations of the convolution weights. Experiments on benchmark datasets of varying task difficulty indicate that (1) there is a substantial amount of excess capacity per task, and (2) capacity can be kept at a surprisingly similar level across tasks. Overall, this suggests a notion of compressibility with respect to weight norms, complementary to classic compression via weight pruning. Source code is available at https://github.com/rkwitt/excess_capacity.

        ----

        ## [738] On Leave-One-Out Conditional Mutual Information For Generalization

        **Authors**: *Mohamad Rida Rammal, Alessandro Achille, Aditya Golatkar, Suhas N. Diggavi, Stefano Soatto*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/421fa4f5e0bef2f044f1f4616fd17343-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/421fa4f5e0bef2f044f1f4616fd17343-Abstract-Conference.html)

        **Abstract**:

        We derive information theoretic generalization bounds for supervised learning algorithms based on a new measure of leave-one-out conditional mutual information (loo-CMI). In contrast to other CMI bounds, which may be hard to evaluate in practice, our loo-CMI bounds are easier to compute and can be interpreted in connection to other notions such as classical leave-one-out cross-validation, stability of the optimization algorithm, and the geometry of the loss-landscape. It applies both to the output of training algorithms as well as their predictions.  We empirically validate the quality of the bound by evaluating its predicted generalization gap in scenarios for deep learning. In particular, our bounds are non-vacuous on image-classification tasks.

        ----

        ## [739] Near-Isometric Properties of Kronecker-Structured Random Tensor Embeddings

        **Authors**: *Qijia Jiang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/423295ba2cea13cda44dc0a7d48ae244-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/423295ba2cea13cda44dc0a7d48ae244-Abstract-Conference.html)

        **Abstract**:

        We give uniform concentration inequality for random tensors acting on rank-1 Kronecker structured signals, which parallels a Gordon-type inequality for this class of tensor structured data. Two variants of the random embedding are considered, where the embedding dimension depends on explicit quantities characterizing the complexity of the signal. As applications of the tools developed herein, we illustrate with examples from signal recovery and optimization.

        ----

        ## [740] MExMI: Pool-based Active Model Extraction Crossover Membership Inference

        **Authors**: *Yaxin Xiao, Qingqing Ye, Haibo Hu, Huadi Zheng, Chengfang Fang, Jie Shi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4241c27d3161c7a7064bfc1a6e539563-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4241c27d3161c7a7064bfc1a6e539563-Abstract-Conference.html)

        **Abstract**:

        With increasing popularity of Machine Learning as a Service (MLaaS), ML models trained from public and proprietary data are deployed in the cloud and deliver prediction services to users. However, as the prediction API becomes a new attack surface, growing concerns have arisen on the confidentiality of ML models. Existing literatures show their vulnerability under model extraction (ME) attacks, while their private training data is vulnerable to another type of attacks, namely, membership inference (MI). In this paper, we show that ME and MI can reinforce each other through a chained and iterative reaction, which can significantly boost ME attack accuracy and improve MI by saving the query cost. As such, we build a framework MExMI for pool-based active model extraction (PAME) to exploit MI through three modules: “MI Pre-Filter”, “MI Post-Filter”, and “semi-supervised boosting”. Experimental results show that MExMI can improve up to 11.14% from the best known PAME attack and reach 94.07% fidelity with only 16k queries. Furthermore, the precision and recall of the MI attack in MExMI are on par with state-of-the-art MI attack which needs 150k queries.

        ----

        ## [741] Parameter-Efficient Masking Networks

        **Authors**: *Yue Bai, Huan Wang, Xu Ma, Yitian Zhang, Zhiqiang Tao, Yun Fu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/427048354ac2db22d43149c51346bafd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/427048354ac2db22d43149c51346bafd-Abstract-Conference.html)

        **Abstract**:

        A deeper network structure generally handles more complicated non-linearity and performs more competitively. Nowadays, advanced network designs often contain a large number of repetitive structures (e.g., Transformer). They empower the network capacity to a new level but also increase the model size inevitably, which is unfriendly to either model restoring or transferring. In this study, we are the first to investigate the representative potential of fixed random weights with limited unique values by learning diverse masks and introduce the Parameter-Efficient Masking Networks (PEMN). It also naturally leads to a new paradigm for model compression to diminish the model size. Concretely, motivated by the repetitive structures in modern neural networks, we utilize one random initialized layer, accompanied with different masks, to convey different feature mappings and represent repetitive network modules. Therefore, the model can be expressed as \textit{one-layer} with a bunch of masks, which significantly reduce the model storage cost. Furthermore, we enhance our strategy by learning masks for a model filled by padding a given random weights vector. In this way, our method can further lower the space complexity, especially for models without many repetitive architectures. We validate the potential of PEMN learning masks on random weights with limited unique values and test its effectiveness for a new compression paradigm based on different network architectures.Code is available at \href{https://github.com/yueb17/PEMN}{\textcolor{magenta}{https://github.com/yueb17/PEMN}}.

        ----

        ## [742] How Sampling Impacts the Robustness of Stochastic Neural Networks

        **Authors**: *Sina Däubener, Asja Fischer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/429d69979c22b06d6baa65caf3ab1e10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/429d69979c22b06d6baa65caf3ab1e10-Abstract-Conference.html)

        **Abstract**:

        Stochastic neural networks (SNNs) are random functions whose predictions are gained by averaging over multiple realizations. Consequently, a gradient-based adversarial example is calculated based on one set of samples and its classification on another set. In this paper, we derive a sufficient condition for such a stochastic prediction to be robust against a given sample-based attack. This allows us to identify the factors that lead to an increased robustness of SNNs and gives theoretical explanations for: (i) the well known observation, that increasing the amount of samples drawn for the estimation of adversarial examples increases the attack's strength,(ii) why increasing the number of samples during an attack can not fully reduce the effect of stochasticity, (iii) why the sample size during inference does not influence the robustness, and(iv) why a higher gradient variance and a shorter expected value of the gradient relates to a higher robustness. Our theoretical findings give a unified view on the mechanisms underlying previously proposed approaches for increasing attack strengths or model robustness and are verified by an extensive empirical analysis.

        ----

        ## [743] Toward Robust Spiking Neural Network Against Adversarial Perturbation

        **Authors**: *Ling Liang, Kaidi Xu, Xing Hu, Lei Deng, Yuan Xie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/42bc612558891859b1b8717051f2c7b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/42bc612558891859b1b8717051f2c7b0-Abstract-Conference.html)

        **Abstract**:

        As spiking neural networks (SNNs) are deployed increasingly in real-world efficiency critical applications,  the security concerns in SNNs attract more attention.Currently, researchers have already demonstrated an SNN can be attacked with adversarial examples. How to build a robust SNN becomes an urgent issue.Recently, many studies apply certified training in artificial neural networks (ANNs), which can improve the robustness of an NN model promisely. However, existing certifications cannot transfer to SNNs directly because of the distinct neuron behavior and input formats for SNNs. In this work, we first design S-IBP and S-CROWN that tackle the non-linear functions in SNNs' neuron modeling. Then, we formalize the boundaries for both digital and spike inputs. Finally, we demonstrate the efficiency of our proposed robust training method in different datasets and model architectures. Based on our experiment, we can achieve a maximum $37.7\%$ attack error reduction with $3.7\%$ original accuracy loss. To the best of our knowledge, this is the first analysis on robust training of SNNs.

        ----

        ## [744] GRASP: Navigating Retrosynthetic Planning with Goal-driven Policy

        **Authors**: *Yemin Yu, Ying Wei, Kun Kuang, Zhengxing Huang, Huaxiu Yao, Fei Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/42beaab8aa8da1c77581609a61eced93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/42beaab8aa8da1c77581609a61eced93-Abstract-Conference.html)

        **Abstract**:

        Retrosynthetic planning occupies a crucial position in synthetic chemistry and, accordingly, drug discovery, which aims to find synthetic pathways of a target molecule through a sequential decision-making process on a set of feasible reactions. While the majority of recent works focus on the prediction of feasible reactions at each step, there have been limited attempts toward improving the sequential decision-making policy. Existing strategies rely on either the expensive and high-variance value estimation by online rollout, or a settled value estimation neural network pre-trained with simulated pathways of limited diversity and no negative feedback. Besides, how to return multiple candidate pathways that are not only diverse but also desirable for chemists (e.g., affordable building block materials) remains an open challenge. To this end, we propose a Goal-dRiven Actor-critic retroSynthetic Planning (GRASP) framework, where we identify the policy that performs goal-driven retrosynthesis navigation toward a user-demand objective. Our experiments on the benchmark Pistachio dataset and a chemists-designed dataset demonstrate that the framework outperforms state-of-the-art approaches by up to 32.2% on search efficiency and 5.6% on quality. Remarkably, our user studies show that GRASP successfully plans pathways that accomplish the goal prescribed with a designated goal (building block materials).

        ----

        ## [745] End-to-end Symbolic Regression with Transformers

        **Authors**: *Pierre-Alexandre Kamienny, Stéphane d'Ascoli, Guillaume Lample, François Charton*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html)

        **Abstract**:

        Symbolic regression, the task of predicting the mathematical expression of a function from the observation of its values, is a difficult task which usually involves a two-step procedure: predicting the "skeleton" of the expression up to the choice of numerical constants, then fitting the constants by optimizing a non-convex loss function. The dominant approach is genetic programming, which evolves candidates by iterating this subroutine a large number of times. Neural networks have recently been tasked to predict the correct skeleton in a single try, but remain much less powerful.In this paper, we challenge this two-step procedure, and task a Transformer to directly predict the full mathematical expression, constants included. One can subsequently refine the predicted constants by feeding them to the non-convex optimizer as an informed initialization. We present ablations to show that this end-to-end approach yields better results, sometimes even without the refinement step. We evaluate our model on problems from the SRBench benchmark and show that our model approaches the performance of state-of-the-art genetic programming with several orders of magnitude faster inference.

        ----

        ## [746] Gold-standard solutions to the Schrödinger equation using deep learning: How much physics do we need?

        **Authors**: *Leon Gerard, Michael Scherbela, Philipp Marquetand, Philipp Grohs*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/430894999584d0bd358611e2ecf00b15-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/430894999584d0bd358611e2ecf00b15-Abstract-Conference.html)

        **Abstract**:

        Finding accurate solutions to the Schrödinger equation is the key unsolved challenge of computational chemistry. Given its importance for the development of new chemical compounds, decades of research have been dedicated to this problem, but due to the large dimensionality even the best available methods do not yet reach the desired accuracy.Recently the combination of deep learning with Monte Carlo methods has emerged as a promising way to obtain highly accurate energies and moderate scaling of computational cost. In this paper we significantly contribute towards this goal by introducing a novel deep-learning architecture that achieves 40-70% lower energy error at 6x lower computational cost compared to previous approaches. Using our method we establish a new benchmark by calculating the most accurate variational ground state energies ever published for a number of different atoms and molecules.We systematically break down and measure our improvements, focusing in particular on the effect of increasing physical prior knowledge.We surprisingly find that increasing the prior knowledge given to the architecture can actually decrease accuracy.

        ----

        ## [747] EcoFormer: Energy-Saving Attention with Linear Complexity

        **Authors**: *Jing Liu, Zizheng Pan, Haoyu He, Jianfei Cai, Bohan Zhuang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4310ae054ce265e56d8ea897971149b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4310ae054ce265e56d8ea897971149b5-Abstract-Conference.html)

        **Abstract**:

        Transformer is a transformative framework for deep learning which models sequential data and has achieved remarkable performance on a wide range of tasks, but with high computational and energy cost. To improve its efficiency, a popular choice is to compress the models via binarization which constrains the floating-point values into binary ones to save resource consumption owing to cheap bitwise operations significantly. However, existing binarization methods only aim at minimizing the information loss for the input distribution statistically, while ignoring the pairwise similarity modeling at the core of the attention mechanism. To this end, we propose a new binarization paradigm customized to high-dimensional softmax attention via kernelized hashing, called EcoFormer, to map the original queries and keys into low-dimensional binary codes in Hamming space. The kernelized hash functions are learned to match the ground-truth similarity relations extracted from the attention map in a self-supervised way. Based on the equivalence between the inner product of binary codes and the Hamming distance as well as the associative property of matrix multiplication, we can approximate the attention in linear complexity by expressing it as a dot-product of binary codes. Moreover, the compact binary representations of queries and keys in EcoFormer enable us to replace most of the expensive multiply-accumulate operations in attention with simple accumulations to save considerable on-chip energy footprint on edge devices. Extensive experiments on both vision and language tasks show that EcoFormer consistently achieves comparable performance with standard attentions while consuming much fewer resources. For example, based on PVTv2-B0 and ImageNet-1K, EcoFormer achieves a 73% reduction in on-chip energy footprint with only a slight performance drop of 0.33% compared to the standard attention. Code is available at https://github.com/ziplab/EcoFormer.

        ----

        ## [748] Wild-Time: A Benchmark of in-the-Wild Distribution Shift over Time

        **Authors**: *Huaxiu Yao, Caroline Choi, Bochuan Cao, Yoonho Lee, Pang Wei Koh, Chelsea Finn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/43119db5d59f07cc08fca7ba6820179a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Distribution shifts occur when the test distribution differs from the training distribution, and can considerably degrade performance of machine learning models deployed in the real world. While recent works have studied robustness to distribution shifts, distribution shifts arising from the passage of time have the additional structure of timestamp metadata. Real-world examples of such shifts are underexplored, and it is unclear whether existing models can leverage trends in past distribution shifts to reliably extrapolate into the future. To address this gap, we curate Wild-Time, a benchmark of 5 datasets that reflect temporal distribution shifts arising in a variety of real-world applications, including drug discovery, patient prognosis, and news classification. On these datasets, we systematically benchmark 13 approaches with various inductive biases. We evaluate methods in domain-generalization, continual learning, self-supervised learning, and ensemble learning, which leverage timestamps to extract the common structure of the distribution shifts. We extend several domain-generalization methods to the temporal distribution shift setting by treating windows of time as different domains. Finally, we propose two evaluation strategies to evaluate model performance under temporal distribution shifts---evaluation with a fixed time split (Eval-Fix) and evaluation with a data stream (Eval-Stream). Eval-Fix, our primary evaluation strategy, aims to provide a simple evaluation protocol for the broader machine learning community, while Eval-Stream serves as a complementary benchmark for continual learning approaches. Our experiments demonstrate that existing methods are limited in tackling temporal distribution shift: across all settings, we observe an average performance drop of 20% from in-distribution to out-of-distribution data.

        ----

        ## [749] Sound and Complete Causal Identification with Latent Variables Given Local Background Knowledge

        **Authors**: *Tian-Zuo Wang, Tian Qin, Zhi-Hua Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4334031bd9d518af18a6aba32ad70c8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4334031bd9d518af18a6aba32ad70c8e-Abstract-Conference.html)

        **Abstract**:

        Great efforts have been devoted to causal discovery from observational data, and it is well known that introducing some background knowledge attained from experiments or human expertise can be very helpful. However, it remains unknown that \emph{what causal relations are identifiable given background knowledge in the presence of latent confounders}. In this paper, we solve the problem with sound and complete orientation rules when the background knowledge is given in a \emph{local} form. Furthermore, based on the solution to the problem, this paper proposes a general active learning framework for causal discovery in the presence of latent confounders, with its effectiveness and efficiency validated by experiments.

        ----

        ## [750] A Unified Diversity Measure for Multiagent Reinforcement Learning

        **Authors**: *Zongkai Liu, Chao Yu, Yaodong Yang, Peng Sun, Zifan Wu, Yuan Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/435cce71b4007699041dfffa4f034079-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/435cce71b4007699041dfffa4f034079-Abstract-Conference.html)

        **Abstract**:

        Promoting behavioural diversity is of critical importance in multi-agent reinforcement learning, since it helps the agent population maintain robust performance when encountering unfamiliar opponents at test time, or,  when the game is highly non-transitive in the strategy space (e.g., Rock-Paper-Scissor). While a myriad of diversity metrics have been proposed, there are no widely accepted or unified  definitions in the literature, making the consequent diversity-aware learning algorithms difficult to evaluate and the insights elusive. In this work, we propose a novel  metric called the Unified Diversity Measure (UDM) that offers a unified view for existing diversity metrics. Based on UDM, we design the UDM-Fictitious Play (UDM-FP) and UDM-Policy Space Response Oracle (UDM-PSRO) algorithms as efficient solvers for  normal-form games and open-ended games. In theory, we prove that UDM-based methods can enlarge the gamescape by increasing the response capacity of the strategy pool, and have convergence guarantee to two-player Nash equilibrium. We validate our  algorithms on games that show strong non-transitivity, and empirical results show that our algorithms achieve better performances than strong PSRO baselines in terms of the exploitability and population effectivity.

        ----

        ## [751] HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions

        **Authors**: *Yongming Rao, Wenliang Zhao, Yansong Tang, Jie Zhou, Ser-Nam Lim, Jiwen Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/436d042b2dd81214d23ae43eb196b146-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/436d042b2dd81214d23ae43eb196b146-Abstract-Conference.html)

        **Abstract**:

        Recent progress in vision Transformers exhibits great success in various tasks driven by the new spatial modeling mechanism based on dot-product self-attention. In this paper, we show that the key ingredients behind the vision Transformers, namely input-adaptive, long-range and high-order spatial interactions, can also be efficiently implemented with a convolution-based framework. We present the Recursive Gated Convolution ($\textit{g}^\textit{n}$Conv) that performs high-order spatial interactions with gated convolutions and recursive designs. The new operation is highly flexible and customizable, which is compatible with various variants of convolution and extends the two-order interactions in self-attention to arbitrary orders without introducing significant extra computation. $\textit{g}^\textit{n}$Conv can serve as a plug-and-play module to improve various vision Transformers and convolution-based models. Based on the operation, we construct a new family of generic vision backbones named HorNet. Extensive experiments on ImageNet classification, COCO object detection and ADE20K semantic segmentation show HorNet outperform Swin Transformers and ConvNeXt by a significant margin with similar overall architecture and training configurations. HorNet also shows favorable scalability to more training data and larger model sizes. Apart from the effectiveness in visual encoders, we also show $\textit{g}^\textit{n}$Conv can be applied to task-specific decoders and consistently improve dense prediction performance with less computation. Our results demonstrate that $\textit{g}^\textit{n}$Conv can be a new basic module for visual modeling that effectively combines the merits of both vision Transformers and CNNs. Code is available at https://github.com/raoyongming/HorNet.

        ----

        ## [752] Learning Tractable Probabilistic Models from Inconsistent Local Estimates

        **Authors**: *Shasha Jin, Vasundhara Komaragiri, Tahrima Rahman, Vibhav Gogate*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/437d9bde2999f6e3e854e09f250261a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/437d9bde2999f6e3e854e09f250261a5-Abstract-Conference.html)

        **Abstract**:

        Tractable probabilistic models such as cutset networks which admit exact linear time posterior marginal inference are often preferred in practice over intractable models such as Bayesian and Markov networks. This is because although tractable models, when learned from data, are slightly inferior to the intractable ones in terms of goodness-of-fit measures such as log-likelihood, they do not use approximate inference at prediction time and as a result exhibit superior predictive performance. In this paper, we consider the problem of improving a tractable model using a large number of local probability estimates, each defined over a small subset of variables that are either available from experts or via an external process. Given a model learned from fully-observed, but small amount of possibly noisy data, the key idea in our approach is to update the parameters of the model via a gradient descent procedure that seeks to minimize a convex combination of two quantities: one that enforces closeness via KL divergence to the local estimates and another that enforces closeness to the given model. We show that although the gradients are NP-hard to compute on arbitrary graphical models, they can be efficiently computed over tractable models. We show via experiments that our approach yields tractable models that are significantly superior to the ones learned from small amount of possibly noisy data, even when the local estimates are inconsistent.

        ----

        ## [753] Decomposed Knowledge Distillation for Class-Incremental Semantic Segmentation

        **Authors**: *Donghyeon Baek, Youngmin Oh, Sanghoon Lee, Junghyup Lee, Bumsub Ham*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/439bf902de1807088d8b731ca20b0777-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/439bf902de1807088d8b731ca20b0777-Abstract-Conference.html)

        **Abstract**:

        Class-incremental semantic segmentation (CISS) labels each pixel of an image with a corresponding object/stuff class continually. To this end, it is crucial to learn novel classes incrementally without forgetting previously learned knowledge. Current CISS methods typically use a knowledge distillation (KD) technique for preserving classifier logits, or freeze a feature extractor, to avoid the forgetting problem. The strong constraints, however, prevent learning discriminative features for novel classes. We introduce a CISS framework that alleviates the forgetting problem and facilitates learning novel classes effectively. We have found that a logit can be decomposed into two terms. They quantify how likely an input belongs to a particular class or not, providing a clue for a reasoning process of a model. The KD technique, in this context, preserves the sum of two terms ($\textit{i.e.}$, a class logit), suggesting that each could be changed and thus the KD does not imitate the reasoning process. To impose constraints on each term explicitly, we propose a new decomposed knowledge distillation (DKD) technique, improving the rigidity of a model and addressing the forgetting problem more effectively. We also introduce a novel initialization method to train new classifiers for novel classes. In CISS, the number of negative training samples for novel classes is not sufficient to discriminate old classes. To mitigate this, we propose to transfer knowledge of negatives to the classifiers successively using an auxiliary classifier, boosting the performance significantly. Experimental results on standard CISS benchmarks demonstrate the effectiveness of our framework.

        ----

        ## [754] Minimax Optimal Algorithms for Fixed-Budget Best Arm Identification

        **Authors**: *Junpei Komiyama, Taira Tsuchiya, Junya Honda*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/43c18853329c7504996b255252b6cb1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/43c18853329c7504996b255252b6cb1f-Abstract-Conference.html)

        **Abstract**:

        We consider the fixed-budget best arm identification problem where the goal is to find the arm of the largest mean with a fixed number of samples. It is known that the probability of misidentifying the best arm is exponentially small to the number of rounds. However, limited characterizations have been discussed on the rate (exponent) of this value. In this paper, we characterize the minimax optimal rate as a result of an optimization over all possible parameters. We introduce two rates, $R^{\mathrm{go}}$ and $R^{\mathrm{go}}_{\infty}$, corresponding to lower bounds on the probability of misidentification, each of which is associated with a proposed algorithm. The rate $R^{\mathrm{go}}$ is associated with $R^{\mathrm{go}}$-tracking, which can be efficiently implemented by a neural network and is shown to outperform existing algorithms. However, this rate requires a nontrivial condition to be achievable. To address this issue, we introduce the second rate $R^{\mathrm{go}}_\infty$. We show that this rate is indeed achievable by introducing a conceptual algorithm called delayed optimal tracking (DOT).

        ----

        ## [755] Reduced Representation of Deformation Fields for Effective Non-rigid Shape Matching

        **Authors**: *Ramana Sundararaman, Riccardo Marin, Emanuele Rodolà, Maks Ovsjanikov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/43d1d3bdd92204c96fa4ac3c578f6a33-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/43d1d3bdd92204c96fa4ac3c578f6a33-Abstract-Conference.html)

        **Abstract**:

        In this work we present a novel approach for computing correspondences between non-rigid objects, by exploiting a reduced representation of deformation fields. Different from existing works that represent deformation fields by training a general-purpose neural network, we advocate for an approximation based on mesh-free methods. By letting the network learn deformation parameters at a sparse set of positions in space (nodes), we reconstruct the continuous deformation field in a closed-form with guaranteed smoothness. With this reduction in degrees of freedom, we show significant improvement in terms of data-efficiency thus enabling limited supervision. Furthermore, our approximation provides direct access to first-order derivatives of deformation fields, which facilitates enforcing desirable regularization effectively. Our resulting model has high expressive power and is able to capture complex deformations. We illustrate its effectiveness through state-of-the-art results across multiple deformable shape matching benchmarks. Our code and data are publicly available at: https://github.com/Sentient07/DeformationBasis.

        ----

        ## [756] BEVFusion: A Simple and Robust LiDAR-Camera Fusion Framework

        **Authors**: *Tingting Liang, Hongwei Xie, Kaicheng Yu, Zhongyu Xia, Zhiwei Lin, Yongtao Wang, Tao Tang, Bing Wang, Zhi Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/43d2b7fbee8431f7cef0d0afed51c691-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/43d2b7fbee8431f7cef0d0afed51c691-Abstract-Conference.html)

        **Abstract**:

        Fusing the camera and LiDAR information has become a de-facto standard for 3D object detection tasks. Current methods rely on point clouds from the LiDAR sensor as queries to leverage the feature from the image space. However, people discovered that this underlying assumption makes the current fusion framework infeasible to produce any prediction when there is a LiDAR malfunction, regardless of minor or major. This fundamentally limits the deployment capability to realistic autonomous driving scenarios. In contrast, we propose a surprisingly simple yet novel fusion framework, dubbed BEVFusion, whose camera stream does not depend on the input of LiDAR data, thus addressing the downside of previous methods. We empirically show that our framework surpasses the state-of-the-art methods under the normal training settings. Under the robustness training settings that simulate various LiDAR malfunctions, our framework significantly surpasses the state-of-the-art methods by 15.7% to 28.9% mAP. To the best of our knowledge, we are the first to handle realistic LiDAR malfunction and can be deployed to realistic scenarios without any post-processing procedure.

        ----

        ## [757] Automatic Differentiation of Programs with Discrete Randomness

        **Authors**: *Gaurav Arya, Moritz Schauer, Frank Schäfer, Christopher Rackauckas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/43d8e5fc816c692f342493331d5e98fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/43d8e5fc816c692f342493331d5e98fc-Abstract-Conference.html)

        **Abstract**:

        Automatic differentiation (AD), a technique for constructing new programs which compute the derivative of an original program, has become ubiquitous throughout scientific computing and deep learning due to the improved performance afforded by gradient-based optimization. However, AD systems have been restricted to the subset of programs that have a continuous dependence on parameters. Programs that have discrete stochastic behaviors governed by distribution parameters, such as flipping a coin with probability $p$ of being heads, pose a challenge to these systems because the connection between the result (heads vs tails) and the parameters ($p$) is fundamentally discrete. In this paper we develop a new reparameterization-based methodology that allows for generating programs whose expectation is the derivative of the expectation of the original program. We showcase how this method gives an unbiased and low-variance estimator which is as automated as traditional AD mechanisms. We demonstrate unbiased forward-mode AD of discrete-time Markov chains, agent-based models such as Conway's Game of Life, and unbiased reverse-mode AD of a particle filter. Our code package is available at https://github.com/gaurav-arya/StochasticAD.jl.

        ----

        ## [758] A Closer Look at the Adversarial Robustness of Deep Equilibrium Models

        **Authors**: *Zonghan Yang, Tianyu Pang, Yang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/43da8cca8f14139774bcbd935d51e0f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/43da8cca8f14139774bcbd935d51e0f2-Abstract-Conference.html)

        **Abstract**:

        Deep equilibrium models (DEQs) refrain from the traditional layer-stacking paradigm and turn to find the fixed point of a single layer. DEQs have achieved promising performance on different applications with featured memory efficiency. At the same time, the adversarial vulnerability of DEQs raises concerns. Several works propose to certify robustness for monotone DEQs. However, limited efforts are devoted to studying empirical robustness for general DEQs. To this end, we observe that an adversarially trained DEQ requires more forward steps to arrive at the equilibrium state, or even violates its fixed-point structure. Besides, the forward and backward tracks of DEQs are misaligned due to the black-box solvers. These facts cause gradient obfuscation when applying the ready-made attacks to evaluate or adversarially train DEQs. Given this, we develop approaches to estimate the intermediate gradients of DEQs and integrate them into the attacking pipelines. Our approaches facilitate fully white-box evaluations and lead to effective adversarial defense for DEQs. Extensive experiments on CIFAR-10 validate the adversarial robustness of DEQs competitive with deep networks of similar sizes.

        ----

        ## [759] Near-Optimal Private and Scalable $k$-Clustering

        **Authors**: *Vincent Cohen-Addad, Alessandro Epasto, Vahab Mirrokni, Shyam Narayanan, Peilin Zhong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/43f55776896a2e33239c2954519f605e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/43f55776896a2e33239c2954519f605e-Abstract-Conference.html)

        **Abstract**:

        We study the differentially private (DP) $k$-means and $k$-median clustering problems of $n$ points in $d$-dimensional Euclidean space in the massively parallel computation (MPC) model. We provide two near-optimal algorithms where the near-optimality is in three aspects: they both achieve (1). $O(1)$ parallel computation rounds, (2). near-linear in $n$ and polynomial in $k$ total computational work (i.e., near-linear running time when $n$ is a sufficient polynomial in $k$), (3). $O(1)$ relative approximation and $\text{poly}(k, d)$ additive error. Note that $\Omega(1)$ relative approximation is provably necessary even for any polynomial-time non-private algorithm, and $\Omega(k)$ additive error is a provable lower bound for any polynomial-time DP $k$-means/median algorithm. Our two algorithms provide a tradeoff between the relative approximation and the additive error: the first has $O(1)$ relative approximation and $\sim (k^{2.5} + k^{1.01} \sqrt{d})$ additive error, and the second one achieves $(1+\gamma)$ relative approximation to the optimal non-private algorithm for an arbitrary small constant $\gamma>0$ and with $\text{poly}(k, d)$ additive error for a larger polynomial dependence on $k$ and $d$.    To achieve our result, we develop a general framework which partitions the data and reduces the DP clustering problem for the entire dataset to the DP clustering problem for each part. To control the blow-up of the additive error introduced by each part, we develop a novel charging argument which might be of independent interest.

        ----

        ## [760] NS3: Neuro-symbolic Semantic Code Search

        **Authors**: *Shushan Arakelyan, Anna Hakhverdyan, Miltiadis Allamanis, Luis Garcia, Christophe Hauser, Xiang Ren*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/43f5f6c5cb333115914c8448b8506411-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/43f5f6c5cb333115914c8448b8506411-Abstract-Conference.html)

        **Abstract**:

        Semantic code search is the task of retrieving a code snippet given a textual description of its functionality. Recent work has been focused on using similarity metrics between neural embeddings of text and code. However, current language models are known to struggle with longer, compositional sentences, and multi-step reasoning. To overcome this limitation, we propose supplementing the query sentence with a layout of its semantic structure. The semantic layout is used to break down the final reasoning decision into a series of lower-level decisions. We use a Neural Module Network architecture to implement this idea. We compare our model - $NS^3$  (Neuro-Symbolic Semantic Search) - to a number of baselines, including state-of-the-art semantic code retrieval methods, such as CodeBERT, CuBERT and GraphCodeBERT, and evaluate on two datasets - Code Search Net (CSN) and Code Search and Question Answering (CoSQA). On these datasets, we demonstrate that our approach results in higher performance. We also perform additional studies to show the effectiveness of our modular design when handling compositional queries.

        ----

        ## [761] Revisiting Sparse Convolutional Model for Visual Recognition

        **Authors**: *Xili Dai, Mingyang Li, Pengyuan Zhai, Shengbang Tong, Xingjian Gao, Shao-Lun Huang, Zhihui Zhu, Chong You, Yi Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4418f6a54f4314202688d77956e731ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4418f6a54f4314202688d77956e731ce-Abstract-Conference.html)

        **Abstract**:

        Despite strong empirical performance for image classification, deep neural networks are often regarded as ``black boxes'' and they are difficult to interpret. On the other hand, sparse convolutional models, which assume that a signal can be expressed by a linear combination of a few elements from a convolutional dictionary, are powerful tools for analyzing natural images with good theoretical interpretability and biological plausibility. However, such principled models have not demonstrated competitive performance when compared with empirically designed deep networks. This paper revisits the sparse convolutional modeling for image classification and bridges the gap between good empirical performance (of deep learning) and good interpretability (of sparse convolutional models). Our method uses differentiable optimization layers that are defined from convolutional sparse coding as drop-in replacements of standard convolutional layers in conventional deep neural networks. We show that such models have equally strong empirical performance on CIFAR-10, CIFAR-100 and ImageNet datasets when compared to conventional neural networks. By leveraging stable recovery property of sparse modeling, we further show that such models can be much more robust to input corruptions as well as adversarial perturbations in testing through a simple proper trade-off between sparse regularization and data reconstruction terms.

        ----

        ## [762] Temporal Latent Bottleneck: Synthesis of Fast and Slow Processing Mechanisms in Sequence Learning

        **Authors**: *Aniket Didolkar, Kshitij Gupta, Anirudh Goyal, Nitesh B. Gundavarapu, Alex Lamb, Nan Rosemary Ke, Yoshua Bengio*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4446b84fdf15b17c091582e4b86f8a05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4446b84fdf15b17c091582e4b86f8a05-Abstract-Conference.html)

        **Abstract**:

        Recurrent neural networks have a strong inductive bias towards learning temporally compressed representations, as the entire history of a sequence is represented by a single vector.  By contrast, Transformers have little inductive bias towards learning temporally compressed representations, as they allow for attention over all previously computed elements in a sequence.  Having a more compressed representation of a sequence may be beneficial for generalization, as a high-level representation may be more easily re-used and re-purposed and will contain fewer irrelevant details. At the same time, excessive compression of representations comes at the cost of expressiveness.  We propose a solution which divides computation into two streams.  A slow stream that is recurrent in nature aims to learn a specialized and compressed representation, by forcing chunks of $K$ time steps into a single representation which is divided into multiple vectors.  At the same time, a fast stream is parameterized as a Transformer to process chunks consisting of $K$ time-steps conditioned on the information in the slow-stream.  In the proposed approach we hope to gain the expressiveness of the Transformer, while encouraging better compression and structuring of representations in the slow stream. We show the benefits of the proposed method in terms of improved sample efficiency and generalization performance as compared to various competitive baselines for visual perception and sequential decision making tasks.

        ----

        ## [763] A Near-Optimal Primal-Dual Method for Off-Policy Learning in CMDP

        **Authors**: *Fan Chen, Junyu Zhang, Zaiwen Wen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/444d69470b24ded080183c907b711bbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/444d69470b24ded080183c907b711bbf-Abstract-Conference.html)

        **Abstract**:

        As an important framework for safe Reinforcement Learning, the Constrained Markov Decision Process (CMDP) has been extensively studied in the recent literature. However, despite the rich results under various on-policy learning settings, there still lacks some essential understanding of the offline CMDP problems, in terms of both the algorithm design and the information theoretic sample complexity lower bound. In this paper, we focus on solving the CMDP problems where only offline data are available. By adopting the concept of the single-policy concentrability coefficient $C^*$, we establish an $\Omega\left(\frac{\min\left\{|\mathcal{S}||\mathcal{A}|,|\mathcal{S}|+I\right\} C^*}{(1-\gamma)^3\epsilon^2}\right)$ sample complexity lower bound for the offline CMDP problem, where $I$ stands for the number of constraints. By introducing a simple but novel deviation control mechanism, we propose a near-optimal primal-dual learning algorithm called DPDL. This algorithm provably guarantees zero constraint violation and its sample complexity matches the above lower bound except for an $\tilde{\mathcal{O}}((1-\gamma)^{-1})$ factor. Comprehensive discussion on how to deal with the unknown constant $C^*$ and the potential asynchronous structure on the offline dataset are also included.

        ----

        ## [764] DReS-FL: Dropout-Resilient Secure Federated Learning for Non-IID Clients via Secret Data Sharing

        **Authors**: *Jiawei Shao, Yuchang Sun, Songze Li, Jun Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/448fc91f669c15d10364ee01d512cc10-Abstract-Conference.html)

        **Abstract**:

        Federated learning (FL) strives to enable collaborative training of machine learning models without centrally collecting clients' private data. Different from centralized training, the local datasets across clients in FL are non-independent and identically distributed (non-IID). In addition, the data-owning clients may drop out of the training process arbitrarily. These characteristics will significantly degrade the training performance. This paper proposes a Dropout-Resilient Secure Federated Learning (DReS-FL) framework based on Lagrange coded computing (LCC) to tackle both the non-IID and dropout problems. The key idea is to utilize Lagrange coding to secretly share the private datasets among clients so that each client receives an encoded version of the global dataset, and the local gradient computation over this dataset is unbiased. To correctly decode the gradient at the server, the gradient function has to be a polynomial in a finite field, and thus we construct polynomial integer neural networks (PINNs) to enable our framework. Theoretical analysis shows that DReS-FL is resilient to client dropouts and provides privacy protection for the local datasets. Furthermore, we experimentally demonstrate that DReS-FL consistently leads to significant performance gains over baseline methods.

        ----

        ## [765] BackdoorBench: A Comprehensive Benchmark of Backdoor Learning

        **Authors**: *Baoyuan Wu, Hongrui Chen, Mingda Zhang, Zihao Zhu, Shaokui Wei, Danni Yuan, Chao Shen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4491ea1c91aa2b22c373e5f1dfce234f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/4491ea1c91aa2b22c373e5f1dfce234f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Backdoor learning is an emerging and vital topic for studying deep neural networks' vulnerability (DNNs). Many pioneering backdoor attack and defense methods are being proposed, successively or concurrently, in the status of a rapid arms race. However, we find that the evaluations of new methods are often unthorough to verify their claims and accurate performance, mainly due to the rapid development, diverse settings, and the difficulties of implementation and reproducibility.  Without thorough evaluations and comparisons, it is not easy to track the current progress and design the future development roadmap of the literature. To alleviate this dilemma, we build a comprehensive benchmark of backdoor learning called BackdoorBench. It consists of an extensible modular-based codebase (currently including implementations of 8 state-of-the-art (SOTA) attacks and 9 SOTA defense algorithms) and a standardized protocol of complete backdoor learning. We also provide comprehensive evaluations of every pair of 8 attacks against 9 defenses, with 5 poisoning ratios, based on 5 models and 4 datasets, thus 8,000 pairs of evaluations in total. We present abundant analysis from different perspectives about these 8,000 evaluations, studying the effects of different factors in backdoor learning.  All codes and evaluations of BackdoorBench are publicly available at https://backdoorbench.github.io.

        ----

        ## [766] REVIVE: Regional Visual Representation Matters in Knowledge-Based Visual Question Answering

        **Authors**: *Yuanze Lin, Yujia Xie, Dongdong Chen, Yichong Xu, Chenguang Zhu, Lu Yuan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/44956951349095f74492a5471128a7e0-Abstract-Conference.html)

        **Abstract**:

        This paper revisits visual representation in knowledge-based visual question answering (VQA) and demonstrates that using regional information in a better way can significantly improve the performance. While visual representation is extensively studied in  traditional VQA, it is under-explored in knowledge-based VQA even though these two tasks share the common spirit, i.e., rely on visual input to answer the question. Specifically, we observe in most state-of-the-art knowledge-based VQA methods: 1) visual features are  extracted either from the whole image or in a sliding window manner for retrieving knowledge, and the important relationship within/among object regions is neglected; 2) visual features are not well utilized in the final answering model, which is counter-intuitive to some extent. Based on these observations, we propose a new knowledge-based VQA method REVIVE, which tries to utilize the explicit information of object regions not only in the knowledge retrieval stage but also in the answering model. The key motivation is that object regions and inherent relationship are important for knowledge-based VQA. We perform extensive experiments on the standard OK-VQA dataset and achieve new state-of the-art performance, i.e., 58.0 accuracy, surpassing previous state-of-the-art method by a large margin (+3.6%). We also conduct detailed analysis and show the necessity of regional information in different framework components for knowledge-based VQA. Code is publicly available at https://github.com/yzleroy/REVIVE.

        ----

        ## [767] FedAvg with Fine Tuning: Local Updates Lead to Representation Learning

        **Authors**: *Liam Collins, Hamed Hassani, Aryan Mokhtari, Sanjay Shakkottai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/449590dfd5789cc7043f85f8bb7afa47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/449590dfd5789cc7043f85f8bb7afa47-Abstract-Conference.html)

        **Abstract**:

        The Federated Averaging (FedAvg) algorithm, which consists of alternating between a few local stochastic gradient updates at client nodes, followed by a model averaging update at the server, is perhaps the most commonly used method in Federated Learning. Notwithstanding its simplicity, several empirical studies have illustrated that the model output by FedAvg leads to a model that generalizes well to new unseen tasks after a few fine-tuning steps. This surprising performance of such a simple method, however, is not fully understood from a theoretical point of view. In this paper, we formally investigate this phenomenon in the multi-task linear regression setting. We show that the reason behind the generalizability of the FedAvg output is FedAvg’s power in learning the common data representation among the clients’ tasks, by leveraging the diversity among client data distributions via multiple local updates between communication rounds. We formally establish the iteration complexity required by the clients for proving such result in the setting where the underlying shared representation is a linear map. To the best of our knowledge, this is the first result showing that FedAvg learns an expressive representation in any setting. Moreover, we show that multiple local updates between communication rounds are necessary for representation learning, as distributed gradient methods that make only one local update between rounds provably cannot recover the ground-truth representation in the linear setting, and empirically yield neural network representations that generalize drastically worse to new clients than those learned by FedAvg trained on heterogeneous image classification datasets.

        ----

        ## [768] Modeling Transitivity and Cyclicity in Directed Graphs via Binary Code Box Embeddings

        **Authors**: *Dongxu Zhang, Michael Boratko, Cameron Musco, Andrew McCallum*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/44a1f18afd6d5cc34d7e5c3d8a80f63b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/44a1f18afd6d5cc34d7e5c3d8a80f63b-Abstract-Conference.html)

        **Abstract**:

        Modeling directed graphs with differentiable representations is a fundamental requirement for performing machine learning on graph-structured data. Geometric embedding models (e.g. hyperbolic, cone, and box embeddings) excel at this task, exhibiting useful inductive biases for directed graphs. However, modeling directed graphs that both contain cycles and some element of transitivity, two properties common in real-world settings, is challenging. Box embeddings, which can be thought of as representing the graph as an intersection over some learned super-graphs, have a natural inductive bias toward modeling transitivity, but (as we prove) cannot model cycles. To this end, we propose binary code box embeddings, where a learned binary code selects a subset of graphs for intersection. We explore several variants, including global binary codes (amounting to a union over intersections) and per-vertex binary codes (allowing greater flexibility) as well as methods of regularization. Theoretical and empirical results show that the proposed models not only preserve a useful inductive bias of transitivity but also have sufficient representational capacity to model arbitrary graphs, including graphs with cycles.

        ----

        ## [769] Generalization Bounds for Gradient Methods via Discrete and Continuous Prior

        **Authors**: *Xuanyuan Luo, Bei Luo, Jian Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/44cdeb5ab7da31d9b5cd88fd44e3da84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/44cdeb5ab7da31d9b5cd88fd44e3da84-Abstract-Conference.html)

        **Abstract**:

        Proving algorithm-dependent generalization error bounds for gradient-type optimization methods has attracted significant attention recently in learning theory. However, most existing trajectory-based analyses require either restrictive assumptions on the learning rate (e.g., fast decreasing learning rate), or continuous injected noise (such as the Gaussian noise in Langevin dynamics). In this paper, we introduce a new discrete data-dependent prior to the PAC-Bayesian framework, and prove a high probability generalization bound of order $O(\frac{1}{n}\cdot \sum_{t=1}^T(\gamma_t/\varepsilon_t)^2\left\|{\mathrm{g}_t}\right\|^2)$  for Floored GD (i.e. a version of gradient descent with precision level $\varepsilon_t$), where $n$ is the number of training samples, $\gamma_t$ is the learning rate at step $t$, $\mathrm{g}_t$ is roughly the difference of the gradient computed using all samples and that using only prior samples. $\left\|{\mathrm{g}_t}\right\|$ is upper bounded by and and typical much smaller than the gradient norm $\left\|{\nabla f(W_t)}\right\|$. We remark that our bound holds for nonconvex and nonsmooth scenarios. Moreover, our theoretical results provide numerically favorable upper bounds of testing errors (e.g., $0.037$ on MNIST). Using similar technique, we can also obtain new generalization bounds for a certain variant of SGD. Furthermore, we study the generalization bounds for gradient Langevin Dynamics (GLD). Using the same framework with a carefully constructed continuous prior, we show a new high probability generalization bound of order $O(\frac{1}{n} + \frac{L^2}{n^2}\sum_{t=1}^T(\gamma_t/\sigma_t)^2)$ for GLD. The new $1/n^2$ rate is due to the concentration of the difference between the gradient of training samples and that of the prior.

        ----

        ## [770] On Deep Generative Models for Approximation and Estimation of Distributions on Manifolds

        **Authors**: *Biraj Dahal, Alexander Havrilla, Minshuo Chen, Tuo Zhao, Wenjing Liao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/44d20d542f3f4e3b7097e5e3f78f99f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/44d20d542f3f4e3b7097e5e3f78f99f1-Abstract-Conference.html)

        **Abstract**:

        Deep generative models have experienced great empirical successes in distribution learning. Many existing experiments have demonstrated that deep generative networks can efficiently generate high-dimensional complex data from a low-dimensional easy-to-sample distribution. However, this phenomenon can not be justified by existing theories. The widely held manifold hypothesis speculates that real-world data sets, such as natural images and signals, exhibit low-dimensional geometric structures. In this paper, we take such low-dimensional data structures into consideration by assuming that data distributions are supported on a low-dimensional manifold. We prove approximation and estimation theories of deep generative networks for estimating distributions on a low-dimensional manifold under the Wasserstein-1 loss. We show that the Wasserstein-1 loss converges to zero at a fast rate depending on the intrinsic dimension instead of the ambient data dimension. Our theory leverages the low-dimensional geometric structures in data sets and justifies the practical power of deep generative models. We require no smoothness assumptions on the data distribution which is desirable in practice.

        ----

        ## [771] Memory Efficient Continual Learning with Transformers

        **Authors**: *Beyza Ermis, Giovanni Zappella, Martin Wistuba, Aditya Rawal, Cédric Archambeau*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html)

        **Abstract**:

        In many real-world scenarios, data to train machine learning models becomes available over time. Unfortunately, these models struggle to continually learn new concepts without forgetting what has been learnt in the past. This phenomenon is known as catastrophic forgetting and it is difficult to prevent due to practical constraints. For instance, the amount of data that can be stored or the computational resources that can be used might be limited. Moreover, applications increasingly rely on large pre-trained neural networks, such as pre-trained Transformers, since compute or data might not be available in sufficiently large quantities to practitioners to train from scratch. In this paper, we devise a method to incrementally train a model on a sequence of tasks using pre-trained Transformers and extending them with Adapters. Different than the existing approaches, our method is able to scale to a large number of tasks without significant overhead and allows sharing information across tasks. On both image and text classification tasks, we empirically demonstrate that our method maintains a good predictive performance without retraining the model or increasing the number of model parameters over time. The resulting model is also significantly faster at inference time compared to Adapter-based state-of-the-art methods.

        ----

        ## [772] Counterfactual Neural Temporal Point Process for Estimating Causal Influence of Misinformation on Social Media

        **Authors**: *Yizhou Zhang, Defu Cao, Yan Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45542d647974ca6af58441c4817c9b5b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45542d647974ca6af58441c4817c9b5b-Abstract-Conference.html)

        **Abstract**:

        Recent years have witnessed the rise of misinformation campaigns that spread specific narratives on social media to manipulate public opinions on different areas, such as politics and healthcare. Consequently, an effective and efficient automatic methodology to estimate the influence of the misinformation on user beliefs and activities is needed. However, existing works on misinformation impact estimation either rely on small-scale psychological experiments or can only discover the correlation between user behaviour and misinformation. To address these issues, in this paper, we build up a causal framework that model the causal effect of misinformation from the perspective of temporal point process. To adapt the large-scale data, we design an efficient yet precise way to estimate the \textbf{Individual Treatment Effect} (ITE) via neural temporal point process and gaussian mixture models. Extensive experiments on synthetic dataset verify the effectiveness and efficiency of our model. We further apply our model on a real-world dataset of social media posts and engagements about COVID-19 vaccines. The experimental results indicate that our model recognized identifiable causal effect of misinformation that hurts people's subjective emotions toward the vaccines.

        ----

        ## [773] Curriculum Reinforcement Learning using Optimal Transport via Gradual Domain Adaptation

        **Authors**: *Peide Huang, Mengdi Xu, Jiacheng Zhu, Laixi Shi, Fei Fang, Ding Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4556f5398bd2c61bd7500e306b4e560a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4556f5398bd2c61bd7500e306b4e560a-Abstract-Conference.html)

        **Abstract**:

        Curriculum Reinforcement Learning (CRL) aims to create a sequence of tasks, starting from easy ones and gradually learning towards difficult tasks. In this work, we focus on the idea of framing CRL as interpolations between a source (auxiliary) and a target task distribution. Although existing studies have shown the great potential of this idea, it remains unclear how to formally quantify and generate the movement between task distributions. Inspired by the insights from gradual domain adaptation in semi-supervised learning, we create a natural curriculum by breaking down the potentially large task distributional shift in CRL into smaller shifts. We propose GRADIENT which formulates CRL as an optimal transport problem with a tailored distance metric between tasks. Specifically, we generate a sequence of task distributions as a geodesic interpolation between the source and target distributions, which are actually the Wasserstein barycenter. Different from many existing methods, our algorithm considers a task-dependent contextual distance metric and is capable of handling nonparametric distributions in both continuous and discrete context settings. In addition, we theoretically show that GRADIENT enables smooth transfer between subsequent stages in the curriculum under certain conditions. We conduct extensive experiments in locomotion and manipulation tasks and show that our proposed GRADIENT achieves higher performance than baselines in terms of learning efficiency and asymptotic performance.

        ----

        ## [774] Temporally-Consistent Survival Analysis

        **Authors**: *Lucas Maystre, Daniel Russo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/455e1e30edf721bd7fa334fffabdcad8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/455e1e30edf721bd7fa334fffabdcad8-Abstract-Conference.html)

        **Abstract**:

        We study survival analysis in the dynamic setting: We seek to model the time to an event of interest given sequences of states. Taking inspiration from temporal-difference learning, a central idea in reinforcement learning, we develop algorithms that estimate a discrete-time survival model by exploiting a temporal-consistency condition. Intuitively, this condition captures the fact that the survival distribution at consecutive states should be similar, accounting for the delay between states. Our method can be combined with any parametric survival model and naturally accommodates right-censored observations. We demonstrate empirically that it achieves better sample-efficiency and predictive performance compared to approaches that directly regress the observed survival outcome.

        ----

        ## [775] Symbolic Distillation for Learned TCP Congestion Control

        **Authors**: *S. P. Sharan, Wenqing Zheng, Kuo-Feng Hsu, Jiarong Xing, Ang Chen, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4574ac9854d4defe3bf119d07b817084-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4574ac9854d4defe3bf119d07b817084-Abstract-Conference.html)

        **Abstract**:

        Recent advances in TCP congestion control (CC) have achieved tremendous success with deep reinforcement learning (RL) approaches, which use feedforward neural networks (NN) to learn complex environment conditions and make better decisions. However, such ``black-box'' policies lack interpretability and reliability, and often, they need to operate outside the traditional TCP datapath due to the use of complex NNs. This paper proposes a novel two-stage solution to achieve the best of both worlds: first to train a deep RL agent, then distill its (over-)parameterized NN policy into white-box, light-weight rules in the form of symbolic expressions that are much easier to understand and to implement in constrained environments. At the core of our proposal is a novel symbolic branching algorithm that enables the rule to be aware of the context in terms of various network conditions, eventually converting the NN policy into a symbolic tree. The distilled symbolic rules preserve and often improve performance over state-of-the-art NN policies while being faster and simpler than a standard neural network. We validate the performance of our distilled symbolic rules on both simulation and emulation environments. Our code is available at https://github.com/VITA-Group/SymbolicPCC.

        ----

        ## [776] A Probabilistic Graph Coupling View of Dimension Reduction

        **Authors**: *Hugues Van Assel, Thibault Espinasse, Julien Chiquet, Franck Picard*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45994782a61bb51cad5c2bae36834265-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45994782a61bb51cad5c2bae36834265-Abstract-Conference.html)

        **Abstract**:

        Most popular dimension reduction (DR) methods like t-SNE and UMAP are based on minimizing a cost between input and latent pairwise similarities. Though widely used, these approaches lack clear probabilistic foundations to enable a full understanding of their properties and limitations. To that extent, we introduce a unifying statistical framework based on the coupling of hidden graphs using cross entropy. These graphs induce a Markov random field dependency structure among the observations in both input and latent spaces. We show that existing pairwise similarity DR methods can be retrieved from our framework with particular choices of priors for the graphs. Moreover this reveals that these methods relying on shift-invariant kernels suffer from a statistical degeneracy that explains poor performances in conserving coarse-grain dependencies. New links are drawn with PCA which appears as a non-degenerate graph coupling model.

        ----

        ## [777] Hardness of Noise-Free Learning for Two-Hidden-Layer Neural Networks

        **Authors**: *Sitan Chen, Aravind Gollakota, Adam R. Klivans, Raghu Meka*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45a7ca247462d9e465ee88c8a302ca70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45a7ca247462d9e465ee88c8a302ca70-Abstract-Conference.html)

        **Abstract**:

        We give superpolynomial statistical query (SQ) lower bounds for learning two-hidden-layer ReLU networks with respect to Gaussian inputs in the standard (noise-free) model. No general SQ lower bounds were known for learning ReLU networks of any depth in this setting: previous SQ lower bounds held only for adversarial noise models (agnostic learning) (Kothari and Klivans 2014, Goel et al. 2020a, Diakonikolas et al. 2020a) or restricted models such as correlational SQ (Goel et al. 2020b, Diakonikolas et al. 2020b). Prior work hinted at the impossibility of our result: Vempala and Wilmes (2019) showed that general SQ lower bounds cannot apply to any real-valued family of functions that satisfies a simple non-degeneracy condition. To circumvent their result, we refine a lifting procedure due to Daniely and Vardi (2021) that reduces Boolean PAC learning problems to Gaussian ones. We show how to extend their technique to other learning models and, in many well-studied cases, obtain a more efficient reduction. As such, we also prove new cryptographic hardness results for PAC learning two-hidden-layer ReLU networks, as well as new lower bounds for learning constant-depth ReLU networks from membership queries.

        ----

        ## [778] Convergence beyond the over-parameterized regime using Rayleigh quotients

        **Authors**: *David A. R. Robin, Kevin Scaman, Marc Lelarge*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45d74e190008c7bff2845ffc8e3facd3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45d74e190008c7bff2845ffc8e3facd3-Abstract-Conference.html)

        **Abstract**:

        In this paper, we present a new strategy to prove the convergence of Deep Learning architectures to a zero training (or even testing) loss by gradient flow. Our analysis is centered on the notion of Rayleigh quotients in order to prove Kurdyka-Lojasiewicz inequalities for a broader set of neural network architectures and loss functions. We show that Rayleigh quotients provide a unified view for several convergence analysis techniques in the literature. Our strategy produces a proof of convergence for various examples of parametric learning. In particular, our analysis does not require the number of parameters to tend to infinity, nor the number of samples to be finite, thus extending to test loss minimization and beyond the over-parameterized regime.

        ----

        ## [779] Optimistic Posterior Sampling for Reinforcement Learning with Few Samples and Tight Guarantees

        **Authors**: *Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Rémi Munos, Alexey Naumov, Mark Rowland, Michal Valko, Pierre Ménard*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45e15bae91a6f213d45e203b8a29be48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45e15bae91a6f213d45e203b8a29be48-Abstract-Conference.html)

        **Abstract**:

        We consider reinforcement learning in an environment modeled by an episodic, tabular, step-dependent Markov decision process of horizon $H$ with $S$ states, and $A$ actions.  The performance of an agent is measured by the regret after interacting with the environment for $T$ episodes. We propose an optimistic posterior sampling algorithm for reinforcement learning (OPSRL), a simple variant of posterior sampling that only needs a number of posterior samples logarithmic in $H$, $S$, $A$, and $T$ per state-action pair. For OPSRL we guarantee a high-probability regret bound of order at most $O(\sqrt{H^3SAT})$ ignoring $\text{poly}\log(HSAT)$ terms. The key novel technical ingredient is a new sharp anti-concentration inequality for linear forms of a Dirichlet random vector which may be of independent interest. Specifically, we extend the normal approximation-based lower bound for Beta distributions by Alfers and Dinges (1984) to Dirichlet distributions. Our bound matches the lower bound of order $\Omega(\sqrt{H^3SAT})$, thereby answering the open problems raised by Agrawal and Jia (2017) for the episodic setting.

        ----

        ## [780] On Convergence of FedProx: Local Dissimilarity Invariant Bounds, Non-smoothness and Beyond

        **Authors**: *Xiaotong Yuan, Ping Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45ecdd6cf1f507d378a3442ed89e580b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45ecdd6cf1f507d378a3442ed89e580b-Abstract-Conference.html)

        **Abstract**:

        The \FedProx~algorithm is a simple yet powerful distributed proximal point optimization method widely used for federated learning (FL) over heterogeneous data. Despite its popularity and remarkable success witnessed in practice, the theoretical understanding of FedProx is largely underinvestigated: the appealing convergence behavior of \FedProx~is so far characterized under certain non-standard and unrealistic dissimilarity assumptions of local functions, and the results are limited to smooth optimization problems. In order to remedy these deficiencies, we develop a novel local dissimilarity invariant convergence theory for \FedProx~and its minibatch stochastic extension through the lens of algorithmic stability. As a result, we contribute to derive several new and deeper insights into \FedProx~for non-convex federated optimization including: 1) convergence guarantees invariant to certain stringent local dissimilarity conditions; 2) convergence guarantees for non-smooth FL problems; and 3) linear speedup with respect to size of minibatch and number of sampled devices. Our theory for the first time reveals that local dissimilarity and smoothness are not must-have for \FedProx~to get favorable complexity bounds.

        ----

        ## [781] Can Push-forward Generative Models Fit Multimodal Distributions?

        **Authors**: *Antoine Salmona, Valentin De Bortoli, Julie Delon, Agnès Desolneux*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45f0d179ef7e10eb7366550cd4e574ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45f0d179ef7e10eb7366550cd4e574ae-Abstract-Conference.html)

        **Abstract**:

        Many generative models synthesize data by transforming a standard Gaussian random variable using a deterministic neural network. Among these models are the Variational Autoencoders and the Generative Adversarial Networks. In this work, we call them "push-forward" models and study their expressivity. We formally demonstrate that the Lipschitz constant of these generative networks has to be large in order to fit multimodal distributions. More precisely, we show that the total variation distance and the Kullback-Leibler divergence between the generated and the data distribution are bounded from below by a constant depending on the mode separation and the Lipschitz constant. Since constraining the Lipschitz constants of neural networks is a common way to stabilize generative models, there is a provable trade-off between the ability of push-forward models to approximate multimodal distributions and the stability of their training. We validate our findings on one-dimensional and image datasets and empirically show that the recently introduced diffusion models do not suffer of such limitation.

        ----

        ## [782] Improved Imaging by Invex Regularizers with Global Optima Guarantees

        **Authors**: *Samuel Pinilla, Tingting Mu, Neil Bourne, Jeyan Thiyagalingam*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45f7927942098d14e473fc5d000031e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45f7927942098d14e473fc5d000031e2-Abstract-Conference.html)

        **Abstract**:

        Image reconstruction enhanced by regularizers, e.g., to enforce sparsity, low rank or smoothness priors on images, has many successful applications in vision tasks such as computer photography, biomedical and spectral imaging. It has been well accepted that non-convex regularizers normally perform better than convex ones in terms of the reconstruction quality. But their convergence analysis is only established to a critical point, rather than the global optima. To mitigate the loss of guarantees for global optima, we propose to apply the concept of invexity and provide the first list of proved invex regularizers for improving image reconstruction. Moreover, we establish convergence guarantees to global optima for various advanced image reconstruction techniques after being improved by such invex regularization. To the best of our knowledge, this is the first practical work applying invex regularization to improve imaging with global optima guarantees. To demonstrate the effectiveness of invex regularization, numerical experiments are conducted for various imaging tasks using benchmark datasets.

        ----

        ## [783] The Neural Covariance SDE: Shaped Infinite Depth-and-Width Networks at Initialization

        **Authors**: *Mufan (Bill) Li, Mihai Nica, Daniel M. Roy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/45fc4a0da7e7f6fbabaabe2d20a441d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/45fc4a0da7e7f6fbabaabe2d20a441d1-Abstract-Conference.html)

        **Abstract**:

        The logit outputs of a feedforward neural network at initialization are conditionally Gaussian, given a random covariance matrix defined by the penultimate layer. In this work, we study the distribution of this random matrix. Recent work has shown that shaping the activation function as network depth grows large is necessary for this covariance matrix to be non-degenerate. However, the current infinite-width-style understanding of this shaping method is unsatisfactory for large depth: infinite-width analyses ignore the microscopic fluctuations from layer to layer, but these fluctuations accumulate over many layers. To overcome this shortcoming, we study the random covariance matrix in the shaped infinite-depth-and-width limit. We identify the precise scaling of the activation function necessary to arrive at a non-trivial limit, and show that the random covariance matrix is governed by a stochastic differential equation (SDE) that we call the Neural Covariance SDE. Using simulations, we show that the SDE closely matches the distribution of the random covariance matrix of finite networks. Additionally, we recover an if-and-only-if condition for exploding and vanishing norms of large shaped networks based on the activation function.

        ----

        ## [784] Rethinking and Scaling Up Graph Contrastive Learning: An Extremely Efficient Approach with Group Discrimination

        **Authors**: *Yizhen Zheng, Shirui Pan, Vincent C. S. Lee, Yu Zheng, Philip S. Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/46027e3de0db3617a911f1a647def3bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/46027e3de0db3617a911f1a647def3bf-Abstract-Conference.html)

        **Abstract**:

        Graph contrastive learning (GCL) alleviates the heavy reliance on label information for graph representation learning (GRL) via self-supervised learning schemes. The core idea is to learn by maximising mutual information for similar instances, which requires similarity computation between two node instances. However, GCL is inefficient in both time and memory consumption. In addition, GCL normally requires a large number of training epochs to be well-trained on large-scale datasets. Inspired by an observation of a technical defect (i.e., inappropriate usage of Sigmoid function) commonly used in two representative GCL works, DGI and MVGRL, we revisit GCL and introduce a new learning paradigm for self-supervised graph representation learning, namely, Group Discrimination (GD), and propose a novel GD-based method called Graph Group Discrimination (GGD). Instead of similarity computation, GGD  directly discriminates two groups of node samples with a very simple binary cross-entropy loss. In addition, GGD requires much fewer training epochs to obtain competitive performance compared with GCL methods on large-scale datasets. These two advantages endow GGD with very efficient property. Extensive experiments show that GGD  outperforms state-of-the-art self-supervised methods on eight datasets. In particular, GGD can be trained in 0.18 seconds (6.44 seconds including data preprocessing) on ogbn-arxiv, which is orders of magnitude (10,000+) faster than GCL baselines while consuming much less memory. Trained with 9 hours on ogbn-papers100M with billion edges, GGD outperforms its GCL counterparts in both accuracy and efficiency.

        ----

        ## [785] Diverse Weight Averaging for Out-of-Distribution Generalization

        **Authors**: *Alexandre Ramé, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, Matthieu Cord*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/46108d807b50ad4144eb353b5d0e8851-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/46108d807b50ad4144eb353b5d0e8851-Abstract-Conference.html)

        **Abstract**:

        Standard neural networks struggle to generalize under distribution shifts in computer vision. Fortunately, combining multiple networks can consistently improve out-of-distribution generalization. In particular, weight averaging (WA) strategies were shown to perform best on the competitive DomainBed benchmark; they directly average the weights of multiple networks despite their nonlinearities. In this paper, we propose Diverse Weight Averaging (DiWA), a new WA strategy whose main motivation is to increase the functional diversity across averaged models. To this end, DiWA averages weights obtained from several independent training runs: indeed, models obtained from different runs are more diverse than those collected along a single run thanks to differences in hyperparameters and training procedures. We motivate the need for diversity by a new bias-variance-covariance-locality decomposition of the expected error, exploiting similarities between WA and standard functional ensembling. Moreover, this decomposition highlights that WA succeeds when the variance term dominates, which we show occurs when the marginal distribution changes at test time. Experimentally, DiWA consistently improves the state of the art on DomainBed without inference overhead.

        ----

        ## [786] Bivariate Causal Discovery for Categorical Data via Classification with Optimal Label Permutation

        **Authors**: *Yang Ni*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4620a66570e554a3ff0e39dc59bcb07a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4620a66570e554a3ff0e39dc59bcb07a-Abstract-Conference.html)

        **Abstract**:

        Causal discovery for quantitative data has been extensively studied but less is known for categorical data. We propose a novel causal model for categorical data based on a new classification model, termed classification with optimal label permutation (COLP). By design, COLP is a parsimonious classifier, which gives rise to a provably identifiable causal model. A simple learning algorithm via comparing likelihood functions of causal and anti-causal models suffices to learn the causal direction. Through experiments with synthetic and real data, we demonstrate the favorable performance of the proposed COLP-based causal model compared to state-of-the-art methods. We also make available an accompanying R package COLP, which contains the proposed causal discovery algorithm and a benchmark dataset of categorical cause-effect pairs.

        ----

        ## [787] Phase transitions in when feedback is useful

        **Authors**: *Lokesh Boominathan, Xaq Pitkow*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4627150df8d491015cfb67c48f99c97a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4627150df8d491015cfb67c48f99c97a-Abstract-Conference.html)

        **Abstract**:

        Sensory observations about the world are invariably ambiguous. Inference about the world's latent variables is thus an important computation for the brain. However, computational constraints limit the performance of these computations. These constraints include energetic costs for neural activity and noise on every channel. Efficient coding is one prominent theory that describes how such limited resources can best be used. In one incarnation, this leads to a theory of predictive coding, where predictions are subtracted from signals, reducing the cost of sending something that is already known. This theory does not, however, account for the costs or noise associated with those predictions. Here we offer a theory that accounts for both feedforward and feedback costs, and noise in all computations. We formulate this inference problem as message-passing on a graph whereby feedback serves as an internal control signal aiming to maximize how well an inference tracks a target state while minimizing the costs of computation. We apply this novel formulation of inference as control to the canonical problem of inferring the hidden scalar state of a linear dynamical system with Gaussian variability. The best solution depends on architectural constraints, such as Dale's law, the ubiquitous law that each neuron makes solely excitatory or inhibitory postsynaptic connections. This biological structure can create asymmetric costs for feedforward and feedback channels. Under such conditions, our theory predicts the gain of optimal predictive feedback and how it is incorporated into the inference computation. We show that there is a non-monotonic dependence of optimal feedback gain as a function of both the computational parameters and the world dynamics, leading to phase transitions in whether feedback provides any utility in optimal inference under computational constraints.

        ----

        ## [788] Stochastic Second-Order Methods Improve Best-Known Sample Complexity of SGD for Gradient-Dominated Functions

        **Authors**: *Saeed Masiha, Saber Salehkaleybar, Niao He, Negar Kiyavash, Patrick Thiran*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/46323351ebc2afa42b30a6122815cb95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/46323351ebc2afa42b30a6122815cb95-Abstract-Conference.html)

        **Abstract**:

        We study the performance of Stochastic Cubic Regularized Newton (SCRN) on a class of functions satisfying gradient dominance property with $1\le\alpha\le2$ which holds in a wide range of applications in machine learning and signal processing. This condition ensures that any first-order stationary point is a global optimum. We prove that the total sample complexity of SCRN in achieving $\epsilon$-global optimum is $\mathcal{O}(\epsilon^{-7/(2\alpha)+1})$ for $1\le\alpha< 3/2$ and $\mathcal{\tilde{O}}(\epsilon^{-2/(\alpha)})$ for $3/2\le\alpha\le 2$. SCRN improves the best-known sample complexity of stochastic gradient descent. Even under a weak version of gradient dominance property, which is applicable to policy-based reinforcement learning (RL), SCRN  achieves the same improvement over stochastic policy gradient methods. Additionally, we show that the average sample complexity of SCRN can be reduced to ${\mathcal{O}}(\epsilon^{-2})$ for $\alpha=1$ using a variance reduction method with time-varying batch sizes. Experimental results in various RL settings showcase the remarkable performance of SCRN compared to first-order methods.

        ----

        ## [789] Posterior and Computational Uncertainty in Gaussian Processes

        **Authors**: *Jonathan Wenger, Geoff Pleiss, Marvin Pförtner, Philipp Hennig, John P. Cunningham*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4683beb6bab325650db13afd05d1a14a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4683beb6bab325650db13afd05d1a14a-Abstract-Conference.html)

        **Abstract**:

        Gaussian processes scale prohibitively with the size of the dataset. In response, many approximation methods have been developed, which inevitably introduce approximation error. This additional source of uncertainty, due to limited computation, is entirely ignored when using the approximate posterior. Therefore in practice, GP models are often as much about the approximation method as they are about the data. Here, we develop a new class of methods that provides consistent estimation of the combined uncertainty arising from both the finite number of data observed and the finite amount of computation expended. The most common GP approximations map to an instance in this class, such as methods based on the Cholesky factorization, conjugate gradients, and inducing points. For any method in this class, we prove (i) convergence of its posterior mean in the associated RKHS, (ii) decomposability of its combined posterior covariance into mathematical and computational covariances, and (iii) that the combined variance is a tight worst-case bound for the squared error between the method's posterior mean and the latent function. Finally, we empirically demonstrate the consequences of ignoring computational uncertainty and show how implicitly modeling it improves generalization performance on benchmark datasets.

        ----

        ## [790] Understanding the Evolution of Linear Regions in Deep Reinforcement Learning

        **Authors**: *Setareh Cohan, Nam Hee Kim, David Rolnick, Michiel van de Panne*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4685275b9a6a2c55d78135563dfd50bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4685275b9a6a2c55d78135563dfd50bb-Abstract-Conference.html)

        **Abstract**:

        Policies produced by deep reinforcement learning are typically characterised by their learning curves, but they remain poorly understood in many other respects. ReLU-based policies result in a partitioning of the input space into piecewise linear regions. We seek to understand how observed region counts and their densities evolve during deep reinforcement learning using empirical results that span a range of continuous control tasks and policy network dimensions. Intuitively, we may expect that during training, the region density increases in the areas that are frequently visited by the policy, thereby affording fine-grained control. We use recent theoretical and empirical results for the linear regions induced by neural networks in supervised learning settings for grounding and comparison of our results. Empirically, we find that the region density increases only moderately throughout training, as measured along fixed trajectories coming from the final policy. However, the trajectories themselves also increase in length during training, and thus the region densities decrease as seen from the perspective of the current trajectory. Our findings suggest that the complexity of deep reinforcement learning policies does not principally emerge from a significant growth in the complexity of functions observed on-and-around trajectories of the policy.

        ----

        ## [791] Causal Discovery in Heterogeneous Environments Under the Sparse Mechanism Shift Hypothesis

        **Authors**: *Ronan Perry, Julius von Kügelgen, Bernhard Schölkopf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/46a126492ea6fb87410e55a58df2e189-Abstract-Conference.html)

        **Abstract**:

        Machine learning approaches commonly rely on the assumption of independent and identically distributed (i.i.d.) data. In reality, however, this assumption is almost always violated due to distribution shifts between environments. Although valuable learning signals can be provided by heterogeneous data from changing distributions, it is also known that learning under arbitrary (adversarial) changes is impossible. Causality provides a useful framework for modeling distribution shifts, since causal models encode both observational and interventional distributions. In this work, we explore the sparse mechanism shift hypothesis which posits that distribution shifts occur due to a small number of changing causal conditionals. Motivated by this idea, we apply it to learning causal structure from heterogeneous environments, where i.i.d. data only allows for learning an equivalence class of graphs without restrictive assumptions. We propose the Mechanism Shift Score (MSS), a score-based approach amenable to various empirical estimators, which provably identifies the entire causal structure with high probability if the sparse mechanism shifts hypothesis holds. Empirically, we verify behavior predicted by the theory and compare multiple estimators and score functions to identify the best approaches in practice. Compared to other methods, we show how MSS bridges a gap by both being nonparametric as well as explicitly leveraging sparse changes.

        ----

        ## [792] Towards Practical Control of Singular Values of Convolutional Layers

        **Authors**: *Alexandra Senderovich, Ekaterina Bulatova, Anton Obukhov, Maxim V. Rakhuba*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/46b1be2b90c6addc84efdf5d7e90eebc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/46b1be2b90c6addc84efdf5d7e90eebc-Abstract-Conference.html)

        **Abstract**:

        In general, convolutional neural networks (CNNs) are easy to train, but their essential properties, such as generalization error and adversarial robustness, are hard to control. Recent research demonstrated that singular values of convolutional layers significantly affect such elusive properties and offered several methods for controlling them. Nevertheless, these methods present an intractable computational challenge or resort to coarse approximations. In this paper, we offer a principled approach to alleviating constraints of the prior art at the expense of an insignificant reduction in layer expressivity. Our method is based on the tensor-train decomposition; it retains control over the actual singular values of convolutional mappings while providing structurally sparse and hardware-friendly representation. We demonstrate the improved properties of modern CNNs with our method and analyze its impact on the model performance, calibration, and adversarial robustness. The source code is available at: https://github.com/WhiteTeaDragon/practicalsvdconv

        ----

        ## [793] Deep Multi-Modal Structural Equations For Causal Effect Estimation With Unstructured Proxies

        **Authors**: *Shachi Deshpande, Kaiwen Wang, Dhruv Sreenivas, Zheng Li, Volodymyr Kuleshov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/46e654963ca9f2b9ff05d1bbfce2420c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/46e654963ca9f2b9ff05d1bbfce2420c-Abstract-Conference.html)

        **Abstract**:

        Estimating the effect of intervention from observational data while accounting for confounding variables is a key task in causal inference. Oftentimes, the confounders are unobserved, but we have access to large amounts of additional unstructured data (images, text) that contain valuable proxy signal about the missing confounders. This paper argues that leveraging this unstructured data can greatly improve the accuracy of causal effect estimation. Specifically, we introduce deep multi-modal structural equations, a generative model for causal effect estimation in which confounders are latent variables and unstructured data are proxy variables. This model supports multiple multimodal proxies (images, text) as well as missing data. We empirically demonstrate that our approach outperforms existing methods based on propensity scores and corrects for confounding using unstructured inputs on tasks in genomics and healthcare. Our methods can potentially support the use of large amounts of data that were previously not used in causal inference

        ----

        ## [794] Generic bounds on the approximation error for physics-informed (and) operator learning

        **Authors**: *Tim De Ryck, Siddhartha Mishra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/46f0114c06524debc60ef2a72769f7a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/46f0114c06524debc60ef2a72769f7a9-Abstract-Conference.html)

        **Abstract**:

        We propose a very general framework for deriving rigorous bounds on the approximation error for physics-informed neural networks (PINNs) and operator learning architectures such as DeepONets and FNOs as well as for physics-informed operator learning. These bounds guarantee that PINNs and (physics-informed) DeepONets or FNOs will efficiently approximate the underlying solution or solution-operator of generic partial differential equations (PDEs). Our framework utilizes existing neural network approximation results to obtain bounds on more-involved learning architectures for PDEs. We illustrate the general framework by deriving the first rigorous bounds on the approximation error of physics-informed operator learning and by showing that PINNs (and physics-informed DeepONets and FNOs) mitigate the curse of dimensionality in approximating nonlinear parabolic PDEs.

        ----

        ## [795] Learning Distributions Generated by Single-Layer ReLU Networks in the Presence of Arbitrary Outliers

        **Authors**: *Saikiran Bulusu, Geethu Joseph, Mustafa Cenk Gursoy, Pramod K. Varshney*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/470e23d14e330ab0daa5387916b95f9c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/470e23d14e330ab0daa5387916b95f9c-Abstract-Conference.html)

        **Abstract**:

        We consider a set of data samples such that a fraction of the samples are arbitrary outliers, and the rest are the output samples of a single-layer neural network with rectified linear unit (ReLU) activation. Our goal is to estimate the parameters (weight matrix and bias vector) of the neural network, assuming the bias vector to be non-negative. We estimate the network parameters using the gradient descent algorithm combined with either the median- or trimmed mean-based filters to mitigate the effect of the arbitrary outliers. We then prove that $\tilde{O}\left( \frac{1}{p^2}+\frac{1}{\epsilon^2p}\right)$ samples and $\tilde{O}\left(  \frac{d^2}{p^2}+ \frac{d^2}{\epsilon^2p}\right)$ time are sufficient for our algorithm to estimate the neural network parameters within an error of $\epsilon$ when the outlier probability is $1-p$, where $2/3

        ----

        ## [796] GenerSpeech: Towards Style Transfer for Generalizable Out-Of-Domain Text-to-Speech

        **Authors**: *Rongjie Huang, Yi Ren, Jinglin Liu, Chenye Cui, Zhou Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/4730d10b22261faa9a95ebf7497bc556-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/4730d10b22261faa9a95ebf7497bc556-Abstract-Conference.html)

        **Abstract**:

        Style transfer for out-of-domain (OOD) speech synthesis aims to generate speech samples with unseen style (e.g., speaker identity, emotion, and prosody) derived from an acoustic reference, while facing the following challenges: 1) The highly dynamic style features in expressive voice are difficult to model and transfer; and 2) the TTS models should be robust enough to handle diverse OOD conditions that differ from the source data. This paper proposes GenerSpeech, a text-to-speech model towards high-fidelity zero-shot style transfer of OOD custom voice. GenerSpeech decomposes the speech variation into the style-agnostic and style-specific parts by introducing two components: 1) a multi-level style adaptor to efficiently model a large range of style conditions, including global speaker and emotion characteristics, and the local (utterance, phoneme, and word-level) fine-grained prosodic representations; and 2) a generalizable content adaptor with Mix-Style Layer Normalization to eliminate style information in the linguistic content representation and thus improve model generalization. Our evaluations on zero-shot style transfer demonstrate that GenerSpeech surpasses the state-of-the-art models in terms of audio quality and style similarity. The extension studies to adaptive style transfer further show that GenerSpeech performs robustly in the few-shot data setting. Audio samples are available at \url{https://GenerSpeech.github.io/}.

        ----

        ## [797] On Non-Linear operators for Geometric Deep Learning

        **Authors**: *Grégoire Sergeant-Perthuis, Jakob Maier, Joan Bruna, Edouard Oyallon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/474815daf1d4096ff78b7e4fdd2086a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/474815daf1d4096ff78b7e4fdd2086a5-Abstract-Conference.html)

        **Abstract**:

        This work studies operators mapping vector and scalar fields defined over a manifold $\mathcal{M}$, and which commute with its group of diffeomorphisms $\text{Diff}(\mathcal{M})$. We prove that in the case of scalar fields $L^p_\omega(\mathcal{M,\mathbb{R}})$, those operators correspond to point-wise non-linearities, recovering and extending known results on $\mathbb{R}^d$. In the context of Neural Networks defined over $\mathcal{M}$, it indicates that point-wise non-linear operators are the only universal family that commutes with any group of symmetries, and justifies their systematic use in combination with dedicated linear operators commuting with specific symmetries. In the case of vector fields $L^p_\omega(\mathcal{M},T\mathcal{M})$, we show that those operators are solely the scalar multiplication. It indicates that $\text{Diff}(\mathcal{M})$ is too rich and that there is no universal class of non-linear operators to motivate the design of Neural Networks over the symmetries of $\mathcal{M}$.

        ----

        ## [798] Momentum Aggregation for Private Non-convex ERM

        **Authors**: *Hoang Tran, Ashok Cutkosky*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/47547ee84e3fbbcbbbbad7c1fd9a973b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/47547ee84e3fbbcbbbbad7c1fd9a973b-Abstract-Conference.html)

        **Abstract**:

        We introduce new algorithms and convergence guarantees for privacy-preserving non-convex Empirical Risk Minimization (ERM) on smooth $d$-dimensional objectives. We develop an improved sensitivity analysis of stochastic gradient descent on smooth objectives that exploits the recurrence of examples in different epochs. By combining this new approach with recent analysis of momentum with private aggregation techniques, we provide an $(\epsilon,\delta)$-differential private algorithm that finds a gradient of norm $O\left(\frac{d^{1/3}}{(\epsilon N)^{2/3}}\right)$ in $O\left(\frac{N^{7/3}\epsilon^{4/3}}{d^{2/3}}\right)$ gradient evaluations, improving the previous best gradient bound of $\tilde O\left(\frac{d^{1/4}}{\sqrt{\epsilon N}}\right)$.

        ----

        ## [799] Learning in Congestion Games with Bandit Feedback

        **Authors**: *Qiwen Cui, Zhihan Xiong, Maryam Fazel, Simon S. Du*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/47561f5e1dc53c7d119185e217b523d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/47561f5e1dc53c7d119185e217b523d0-Abstract-Conference.html)

        **Abstract**:

        In this paper, we investigate Nash-regret minimization in congestion games, a class of games with benign theoretical structure and broad real-world applications. We first propose a centralized algorithm based on the optimism in the face of uncertainty principle for congestion games with (semi-)bandit feedback, and obtain finite-sample guarantees. Then we propose a decentralized algorithm via a novel combination of the Frank-Wolfe method and G-optimal design. By exploiting the structure of the congestion game, we show the sample complexity of both algorithms depends only polynomially on the number of players and the number of facilities, but not the size of the action set, which can be exponentially large in terms of the number of facilities. We further define a new problem class, Markov congestion games, which allows us to model the non-stationarity in congestion games. We propose a centralized algorithm for Markov congestion games, whose sample complexity again has only polynomial dependence on all relevant problem parameters, but not the size of the action set.

        ----

        

[Go to the previous page](NIPS-2022-list03.md)

[Go to the next page](NIPS-2022-list05.md)

[Go to the catalog section](README.md)