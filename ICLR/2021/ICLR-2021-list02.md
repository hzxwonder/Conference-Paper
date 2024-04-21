## [200] Deberta: decoding-Enhanced Bert with Disentangled Attention

**Authors**: *Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XPZIaotutsD](https://openreview.net/forum?id=XPZIaotutsD)

**Abstract**:

Recent progress in pre-trained neural language models has significantly improved the performance of many natural language processing (NLP) tasks. In this paper we propose a new model architecture DeBERTa (Decoding-enhanced BERT with disentangled attention) that improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions, respectively. Second, an enhanced mask decoder is used to incorporate absolute positions in the decoding layer to predict the masked tokens in model pre-training. In addition, a new virtual adversarial training method is used for fine-tuning to improve models’ 
 generalization. We show that these techniques significantly improve the efficiency of model pre-training and the performance of both natural language understand(NLU) and natural langauge generation (NLG) downstream tasks. Compared to RoBERTa-Large, a DeBERTa model trained on half of the training data performs consistently better on a wide range of NLP tasks, achieving improvements on MNLI by +0.9% (90.2% vs. 91.1%), on SQuAD v2.0 by +2.3% (88.4% vs. 90.7%) and RACE by +3.6% (83.2% vs. 86.8%). Notably, we scale up DeBERTa by training a larger version that consists of 48 Transform layers with 1.5 billion parameters. The significant performance boost makes the single DeBERTa model surpass the human performance on the SuperGLUE benchmark (Wang et al., 2019a) for the first time in terms of macro-average score (89.9 versus 89.8), and the ensemble DeBERTa model sits atop the SuperGLUE leaderboard as of January 6, 2021, outperforming the human baseline by a decent margin (90.3 versus
89.8). The pre-trained DeBERTa models and the source code were released at: https://github.com/microsoft/DeBERTa.

----

## [201] Optimism in Reinforcement Learning with Generalized Linear Function Approximation

**Authors**: *Yining Wang, Ruosong Wang, Simon Shaolei Du, Akshay Krishnamurthy*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=CBmJwzneppz](https://openreview.net/forum?id=CBmJwzneppz)

**Abstract**:

We design a new provably efficient algorithm for episodic reinforcement learning with generalized linear function approximation. We analyze the algorithm under a new expressivity assumption that we call ``optimistic closure,'' which is strictly weaker than assumptions from prior analyses for the linear setting. With optimistic closure, we prove that our algorithm enjoys a regret bound of $\widetilde{O}\left(H\sqrt{d^3 T}\right)$ where $H$ is the horizon, $d$ is the dimensionality of the state-action features and $T$ is the number of episodes. This is the first statistically and computationally efficient algorithm for reinforcement learning with generalized linear functions.

----

## [202] Graph Traversal with Tensor Functionals: A Meta-Algorithm for Scalable Learning

**Authors**: *Elan Sopher Markowitz, Keshav Balasubramanian, Mehrnoosh Mirtaheri, Sami Abu-El-Haija, Bryan Perozzi, Greg Ver Steeg, Aram Galstyan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6DOZ8XNNfGN](https://openreview.net/forum?id=6DOZ8XNNfGN)

**Abstract**:

Graph Representation Learning (GRL) methods have impacted fields from chemistry to social science. However, their algorithmic implementations are specialized to specific use-cases e.g. "message passing" methods are run differently from "node embedding" ones. Despite their apparent differences, all these methods utilize the graph structure,  and therefore, their learning can be approximated with stochastic graph traversals.  We propose Graph Traversal via Tensor Functionals (GTTF), a unifying meta-algorithm framework for easing the implementation of diverse graph algorithms and enabling transparent and efficient scaling to large graphs.  GTTF is founded upon a data structure (stored as a sparse tensor) and a stochastic graph traversal algorithm (described using tensor operations). The algorithm is a functional that accept two functions, and can be specialized to obtain a variety of GRL models and objectives, simply by changing those two functions. We show for a wide class of methods, our algorithm learns in an unbiased fashion and, in expectation, approximates the learning as if the specialized implementations were run directly.
With these capabilities, we scale otherwise non-scalable methods to set state-of-the-art on large graph datasets while being more efficient than existing GRL libraries -- with only a handful of lines of code for each method specialization.

----

## [203] Diverse Video Generation using a Gaussian Process Trigger

**Authors**: *Gaurav Shrivastava, Abhinav Shrivastava*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Qm7R_SdqTpT](https://openreview.net/forum?id=Qm7R_SdqTpT)

**Abstract**:

Generating future frames given a few context (or past) frames is a challenging task. It requires modeling the temporal coherence of videos as well as multi-modality in terms of diversity in the potential future states. Current variational approaches for video generation tend to marginalize over multi-modal future outcomes. Instead, we propose to explicitly model the multi-modality in the future outcomes and leverage it to sample diverse futures. Our approach, Diverse Video Generator, uses a GP to learn priors on future states given the past and maintains a probability distribution over possible futures given a particular sample. We leverage the changes in this distribution over time to control the sampling of diverse future states by estimating the end of on-going sequences. In particular, we use the variance of GP over the output function space to trigger a change in the action sequence. We achieve state-of-the-art results on diverse future frame generation in terms of reconstruction quality and diversity of the generated sequences.

----

## [204] Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU

**Authors**: *Patrick Kidger, Terry J. Lyons*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lqU2cs3Zca](https://openreview.net/forum?id=lqU2cs3Zca)

**Abstract**:

Signatory is a library for calculating and performing functionality related to the signature and logsignature transforms. The focus is on machine learning, and as such includes features such as CPU parallelism, GPU support, and backpropagation. To our knowledge it is the first GPU-capable library for these operations. Signatory implements new features not available in previous libraries, such as efficient precomputation strategies. Furthermore, several novel algorithmic improvements are introduced, producing substantial real-world speedups even on the CPU without parallelism. The library operates as a Python wrapper around C++, and is compatible with the PyTorch ecosystem. It may be installed directly via \texttt{pip}. Source code, documentation, examples, benchmarks and tests may be found at \texttt{\url{https://github.com/patrick-kidger/signatory}}. The license is Apache-2.0.

----

## [205] MoPro: Webly Supervised Learning with Momentum Prototypes

**Authors**: *Junnan Li, Caiming Xiong, Steven C. H. Hoi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0-EYBhgw80y](https://openreview.net/forum?id=0-EYBhgw80y)

**Abstract**:

We propose a webly-supervised representation learning method that does not suffer from the annotation unscalability of supervised learning, nor the computation unscalability of self-supervised learning. Most existing works on webly-supervised representation learning adopt a vanilla supervised learning method without accounting for the prevalent noise in the training data, whereas most prior methods in learning with label noise are less effective for real-world large-scale noisy data. We propose momentum prototypes (MoPro), a simple contrastive learning method that achieves online label noise correction, out-of-distribution sample removal, and representation learning. MoPro achieves state-of-the-art performance on WebVision, a weakly-labeled noisy dataset. MoPro also shows superior performance when the pretrained model is transferred to down-stream image classification and detection tasks. It outperforms the ImageNet supervised pretrained model by +10.5 on 1-shot classification on VOC, and outperforms the best self-supervised pretrained model by +17.3 when finetuned on 1% of ImageNet labeled samples. Furthermore, MoPro is more robust to distribution shifts. Code and pretrained models are available at https://github.com/salesforce/MoPro.

----

## [206] A Universal Representation Transformer Layer for Few-Shot Image Classification

**Authors**: *Lu Liu, William L. Hamilton, Guodong Long, Jing Jiang, Hugo Larochelle*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=04cII6MumYV](https://openreview.net/forum?id=04cII6MumYV)

**Abstract**:

Few-shot classification aims to recognize unseen classes when presented with only a small number of samples. We consider the problem of multi-domain few-shot image classification, where unseen classes and examples come from diverse data sources. This problem has seen growing interest and has inspired the development of benchmarks such as Meta-Dataset. A key challenge in this multi-domain setting is to effectively integrate the feature representations from the diverse set of training domains. Here, we propose a Universal Representation Transformer (URT) layer, that meta-learns to leverage universal features for few-shot classification by dynamically re-weighting and composing the most appropriate domain-specific representations. In experiments, we show that URT sets a new state-of-the-art result on Meta-Dataset. Specifically, it achieves top-performance on the highest number of data sources compared to competing methods. We analyze variants of URT and present a visualization of the attention score heatmaps that sheds light on how the model performs cross-domain generalization.

----

## [207] Primal Wasserstein Imitation Learning

**Authors**: *Robert Dadashi, Léonard Hussenot, Matthieu Geist, Olivier Pietquin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TtYSU29zgR](https://openreview.net/forum?id=TtYSU29zgR)

**Abstract**:

Imitation Learning (IL) methods seek to match the behavior of an agent with that of an expert. In the present work, we propose a new IL method based on a conceptually simple algorithm: Primal Wasserstein Imitation Learning (PWIL), which ties to the primal form of the Wasserstein distance between the expert and the agent state-action distributions. We present a reward function which is derived offline, as opposed to recent adversarial IL algorithms that learn a reward function through interactions with the environment, and which requires little fine-tuning. We show that we can recover expert behavior on a variety of continuous control tasks of the MuJoCo domain in a sample efficient manner in terms of agent interactions and of expert interactions with the environment. Finally, we show that the behavior of the agent we train matches the behavior of the expert with the Wasserstein distance, rather than the commonly used proxy of performance.

----

## [208] Learning perturbation sets for robust machine learning

**Authors**: *Eric Wong, J. Zico Kolter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MIDckA56aD](https://openreview.net/forum?id=MIDckA56aD)

**Abstract**:

Although much progress has been made towards robust deep learning, a significant gap in robustness remains between real-world perturbations and more narrowly defined sets typically studied in adversarial defenses. In this paper, we aim to bridge this gap by learning perturbation sets from data, in order to characterize real-world effects for robust training and evaluation. Specifically, we use a conditional generator that defines the perturbation set over a constrained region of the latent space. We formulate desirable properties that measure the quality of a learned perturbation set, and theoretically prove that a conditional variational autoencoder naturally satisfies these criteria. Using this framework, our approach can generate a variety of perturbations at different complexities and scales, ranging from baseline spatial transformations, through common image corruptions, to lighting variations. We measure the quality of our learned perturbation sets both quantitatively and qualitatively, finding that our models are capable of producing a diverse set of meaningful perturbations beyond the limited data seen during training. Finally, we leverage our learned perturbation sets to train models which are empirically and certifiably robust to adversarial image corruptions and adversarial lighting variations, while improving generalization on non-adversarial data. All code and configuration files for reproducing the experiments as well as pretrained model weights can be found at https://github.com/locuslab/perturbation_learning.

----

## [209] CopulaGNN: Towards Integrating Representational and Correlational Roles of Graphs in Graph Neural Networks

**Authors**: *Jiaqi Ma, Bo Chang, Xuefei Zhang, Qiaozhu Mei*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XI-OJ5yyse](https://openreview.net/forum?id=XI-OJ5yyse)

**Abstract**:

Graph-structured data are ubiquitous. However, graphs encode diverse types of information and thus play different roles in data representation. In this paper, we distinguish the \textit{representational} and the \textit{correlational} roles played by the graphs in node-level prediction tasks, and we investigate how Graph Neural Network (GNN) models can effectively leverage both types of information. Conceptually, the representational information provides guidance for the model to construct better node features; while the correlational information indicates the correlation between node outcomes conditional on node features. Through a simulation study, we find that many popular GNN models are incapable of effectively utilizing the correlational information. By leveraging the idea of the copula, a principled way to describe the dependence among multivariate random variables, we offer a general solution. The proposed Copula Graph Neural Network (CopulaGNN) can take a wide range of GNN models as base models and utilize both representational and correlational information stored in the graphs. Experimental results on two types of regression tasks verify the effectiveness of the proposed method.

----

## [210] On the Critical Role of Conventions in Adaptive Human-AI Collaboration

**Authors**: *Andy Shih, Arjun Sawhney, Jovana Kondic, Stefano Ermon, Dorsa Sadigh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8Ln-Bq0mZcy](https://openreview.net/forum?id=8Ln-Bq0mZcy)

**Abstract**:

Humans can quickly adapt to new partners in collaborative tasks (e.g. playing basketball), because they understand which fundamental skills of the task (e.g. how to dribble, how to shoot) carry over across new partners. Humans can also quickly adapt to similar tasks with the same partners by carrying over conventions that they have developed (e.g. raising hand signals pass the ball), without learning to coordinate from scratch. To collaborate seamlessly with humans, AI agents should adapt quickly to new partners and new tasks as well. However, current approaches have not attempted to distinguish between the complexities intrinsic to a task and the conventions used by a partner, and more generally there has been little focus on leveraging conventions for adapting to new settings. In this work, we propose a learning framework that teases apart rule-dependent representation from convention-dependent representation in a principled way. We show that, under some assumptions, our rule-dependent representation is a sufficient statistic of the distribution over best-response strategies across partners. Using this separation of representations, our agents are able to adapt quickly to new partners, and to coordinate with old partners on new tasks in a zero-shot manner. We experimentally validate our approach on three collaborative tasks varying in complexity: a contextual multi-armed bandit, a block placing task, and the card game Hanabi.

----

## [211] On the Bottleneck of Graph Neural Networks and its Practical Implications

**Authors**: *Uri Alon, Eran Yahav*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=i80OPhOCVH2](https://openreview.net/forum?id=i80OPhOCVH2)

**Abstract**:

Since the proposal of the graph neural network (GNN) by Gori et al. (2005) and Scarselli et al. (2008), one of the major problems in training GNNs was their struggle to propagate information between distant nodes in the graph.
We propose a new explanation for this problem: GNNs are susceptible to a bottleneck when aggregating messages across a long path. This bottleneck causes the over-squashing of exponentially growing information into fixed-size vectors.
As a result, GNNs fail to propagate messages originating from distant nodes and perform poorly when the prediction task depends on long-range interaction.
In this paper, we highlight the inherent problem of over-squashing in GNNs:
we demonstrate that the bottleneck hinders popular GNNs from fitting long-range signals in the training data;
we further show that GNNs that absorb incoming edges equally, such as GCN and GIN, are more susceptible to over-squashing than GAT and GGNN;
finally, we show that prior work, which extensively tuned GNN models of long-range problems, suffers from over-squashing, and that breaking the bottleneck improves their state-of-the-art results without any tuning or additional weights. 
Our code is available at https://github.com/tech-srl/bottleneck/ .

----

## [212] The geometry of integration in text classification RNNs

**Authors**: *Kyle Aitken, Vinay Venkatesh Ramasesh, Ankush Garg, Yuan Cao, David Sussillo, Niru Maheswaranathan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=42kiJ7n_8xO](https://openreview.net/forum?id=42kiJ7n_8xO)

**Abstract**:

Despite the widespread application of recurrent neural networks (RNNs), a unified understanding of how RNNs solve particular tasks remains elusive.  In particular, it is unclear what dynamical patterns arise in trained RNNs, and how those pat-terns depend on the training dataset or task.  This work addresses these questions in the context of text classification, building on earlier work studying the dynamics of binary sentiment-classification networks (Maheswaranathan et al., 2019).  We study text-classification tasks beyond the binary case, exploring the dynamics ofRNNs trained on both natural and synthetic datasets.  These dynamics, which we find to be both interpretable and low-dimensional, share a common mechanism across architectures and datasets:  specifically, these text-classification networks use low-dimensional attractor manifolds to accumulate evidence for each class as they process the text.  The dimensionality and geometry of the attractor manifold are determined by the structure of the training dataset, with the dimensionality reflecting the number of scalar quantities the network remembers in order to classify.In categorical classification, for example, we show that this dimensionality is one less than the number of classes. Correlations in the dataset, such as those induced by ordering, can further reduce the dimensionality of the attractor manifold; we show how to predict this reduction using simple word-count statistics computed on the training dataset. To the degree that integration of evidence towards a decision is a common computational primitive, this work continues to lay the foundation for using dynamical systems techniques to study the inner workings of RNNs.

----

## [213] Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability

**Authors**: *Jeremy Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, Ameet Talwalkar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jh-rTtvkGeM](https://openreview.net/forum?id=jh-rTtvkGeM)

**Abstract**:

We empirically demonstrate that full-batch gradient descent on neural network training objectives typically operates in a regime we call the Edge of Stability. In this regime, the maximum eigenvalue of the training loss Hessian hovers just above the value $2 / \text{(step size)}$, and the training loss behaves non-monotonically over short timescales, yet consistently decreases over long timescales. Since this behavior is inconsistent with several widespread presumptions in the field of optimization, our findings raise questions as to whether these presumptions are relevant to neural network training. We hope that our findings will inspire future efforts aimed at rigorously understanding optimization at the Edge of Stability.

----

## [214] CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and Transfer Learning

**Authors**: *Ossama Ahmed, Frederik Träuble, Anirudh Goyal, Alexander Neitz, Manuel Wuthrich, Yoshua Bengio, Bernhard Schölkopf, Stefan Bauer*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=SK7A5pdrgov](https://openreview.net/forum?id=SK7A5pdrgov)

**Abstract**:

Despite recent successes of reinforcement learning (RL), it remains a challenge for agents to transfer learned skills to related environments. To facilitate research addressing this problem, we proposeCausalWorld, a benchmark for causal structure and transfer learning in a robotic manipulation environment. The environment is a simulation of an open-source robotic platform, hence offering the possibility of sim-to-real transfer. Tasks consist of constructing 3D shapes from a set of blocks - inspired by how children learn to build complex structures. The key strength of CausalWorld is that it provides a combinatorial family of such tasks with common causal structure and underlying factors (including, e.g., robot and object masses, colors, sizes). The user (or the agent) may intervene on all causal variables, which allows  for  fine-grained  control  over  how  similar  different  tasks  (or  task  distributions) are. One can thus easily define training and evaluation distributions of a desired difficulty level,  targeting a specific form of generalization (e.g.,  only changes in appearance or object mass). Further, this common parametrization facilitates defining curricula by interpolating between an initial and a target task. While users may define their own task distributions, we present eight meaningful distributions as concrete benchmarks, ranging from simple to very challenging, all of which require long-horizon planning as well as precise low-level motor control. Finally, we provide baseline results for a subset of these tasks on distinct training curricula and corresponding evaluation protocols, verifying the feasibility of the tasks in this benchmark.

----

## [215] Empirical or Invariant Risk Minimization? A Sample Complexity Perspective

**Authors**: *Kartik Ahuja, Jun Wang, Amit Dhurandhar, Karthikeyan Shanmugam, Kush R. Varshney*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jrA5GAccy_](https://openreview.net/forum?id=jrA5GAccy_)

**Abstract**:

Recently, invariant risk minimization (IRM) was proposed as a promising solution to address out-of-distribution (OOD) generalization. However, it is unclear when IRM should be preferred over the widely-employed empirical risk minimization (ERM) framework. In this work, we analyze both these frameworks from the perspective of sample complexity, thus taking a firm step towards answering this important question. We find that depending on the type of data generation mechanism, the two approaches might have very different finite sample and asymptotic behavior. For example, in the covariate shift setting we see that the two approaches not only arrive at the same asymptotic solution, but also have similar finite sample behavior with no clear winner. For other distribution shifts such as those involving confounders or anti-causal variables, however, the two approaches arrive at different asymptotic solutions where IRM is guaranteed to be close to the desired OOD solutions in the finite sample regime, while ERM is biased even asymptotically.  We further investigate how different factors --- the number of environments, complexity of the model, and IRM penalty weight ---  impact the sample complexity of IRM in relation to its distance from the OOD solutions.

----

## [216] Scaling Symbolic Methods using Gradients for Neural Model Explanation

**Authors**: *Subham Sekhar Sahoo, Subhashini Venugopalan, Li Li, Rishabh Singh, Patrick Riley*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=V5j-jdoDDP](https://openreview.net/forum?id=V5j-jdoDDP)

**Abstract**:

Symbolic techniques based on Satisfiability Modulo Theory (SMT) solvers have been proposed for analyzing and verifying neural network properties, but their usage has been fairly limited owing to their poor scalability with larger networks. In this work, we propose a technique for combining gradient-based methods with symbolic techniques to scale such analyses and demonstrate its application for model explanation. In particular, we apply this technique to identify minimal regions in an input that are most relevant for a neural network's prediction. Our approach uses gradient information (based on Integrated Gradients) to focus on a subset of neurons in the first layer, which allows our technique to scale to large networks. The corresponding SMT constraints encode the minimal input mask discovery problem such that after masking the input, the activations of the selected neurons are still above a threshold. After solving for the minimal masks, our approach scores the mask regions to generate a relative ordering of the features within the mask. This produces a saliency map which explains" where a model is looking" when making a prediction. We evaluate our technique on three datasets-MNIST, ImageNet, and Beer Reviews, and demonstrate both quantitatively and qualitatively that the regions generated by our approach are sparser and achieve higher saliency scores compared to the gradient-based methods alone. Code and examples are at - https://github.com/google-research/google-research/tree/master/smug_saliency

----

## [217] Control-Aware Representations for Model-based Reinforcement Learning

**Authors**: *Brandon Cui, Yinlam Chow, Mohammad Ghavamzadeh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dgd4EJqsbW5](https://openreview.net/forum?id=dgd4EJqsbW5)

**Abstract**:

A major challenge in modern reinforcement learning (RL) is efficient control of dynamical systems from high-dimensional sensory observations.   Learning controllable embedding (LCE) is a promising approach that addresses this challenge by embedding the observations into a lower-dimensional latent space, estimating the latent dynamics, and utilizing it to perform control in the latent space.  Two important questions in this area are how to learn a representation that is amenable to the control problem at hand, and how to achieve an end-to-end framework for representation learning and control.  In this paper, we take a few steps towards addressing these questions. We first formulate a LCE model to learn representations that are suitable to be used by a policy iteration style algorithm in the latent space.We call this model control-aware representation learning(CARL). We derive a loss function and three implementations for CARL. In the offline implementation, we replace the locally-linear control algorithm (e.g., iLQR) used by the existing LCE methods with a RL algorithm, namely model-based soft actor-critic, and show that it results in significant improvement. In online CARL, we interleave representation learning and control, and demonstrate further gain in performance.  Finally, we propose value-guided CARL, a variation in which we optimize a weighted version of the CARL loss function, where the weights depend on the TD-error of the current policy. We evaluate the proposed algorithms by extensive experiments on benchmark tasks and compare them with several LCE baselines.

----

## [218] C-Learning: Learning to Achieve Goals via Recursive Classification

**Authors**: *Benjamin Eysenbach, Ruslan Salakhutdinov, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tc5qisoB-C](https://openreview.net/forum?id=tc5qisoB-C)

**Abstract**:

We study the problem of predicting and controlling the future state distribution of an autonomous agent. This problem, which can be viewed as a reframing of goal-conditioned reinforcement learning (RL), is centered around learning a conditional probability density function over future states. Instead of directly estimating this density function, we indirectly estimate this density function by training a classifier to predict whether an observation comes from the future. Via Bayes' rule, predictions from our classifier can be transformed into predictions over future states. Importantly, an off-policy variant of our algorithm allows us to predict the future state distribution of a new policy, without collecting new experience. This variant allows us to optimize functionals of a policy's future state distribution, such as the density of reaching a particular goal state. While conceptually similar to Q-learning, our work lays a principled foundation for goal-conditioned RL as density estimation, providing justification for goal-conditioned methods used in prior work. This foundation makes hypotheses about Q-learning, including the optimal goal-sampling ratio, which we confirm experimentally. Moreover, our proposed method is competitive with prior goal-conditioned RL methods.

----

## [219] The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers

**Authors**: *Preetum Nakkiran, Behnam Neyshabur, Hanie Sedghi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=guetrIHLFGI](https://openreview.net/forum?id=guetrIHLFGI)

**Abstract**:

We propose a new framework for reasoning about generalization in deep learning. 
The core idea is to couple the Real World, where optimizers take stochastic gradient steps on the empirical loss, to an Ideal World, where optimizers take steps on the population loss. This leads to an alternate decomposition of test error into: (1) the Ideal World test error plus (2) the gap between the two worlds. If the gap (2) is universally small, this reduces the problem of generalization in offline learning to the problem of optimization in online learning.
We then give empirical evidence that this gap between worlds can be small in realistic deep learning settings, in particular supervised image classification. For example, CNNs generalize better than MLPs on image distributions in the Real World, but this is "because" they optimize faster on the population loss in the Ideal World. This suggests our framework is a useful tool for understanding generalization in deep learning, and lays the foundation for future research in this direction.

----

## [220] Improving VAEs' Robustness to Adversarial Attack

**Authors**: *Matthew Willetts, Alexander Camuto, Tom Rainforth, Stephen J. Roberts, Christopher C. Holmes*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-Hs_otp2RB](https://openreview.net/forum?id=-Hs_otp2RB)

**Abstract**:

Variational autoencoders (VAEs) have recently been shown to be vulnerable to adversarial attacks, wherein they are fooled into reconstructing a chosen target image. However, how to defend against such attacks remains an open problem. We make significant advances in addressing this issue by introducing methods for producing adversarially robust VAEs. Namely, we first demonstrate that methods proposed to obtain disentangled latent representations produce VAEs that are more robust to these attacks. However, this robustness comes at the cost of reducing the quality of the reconstructions. We ameliorate this by applying disentangling methods to hierarchical VAEs. The resulting models produce high--fidelity autoencoders that are also adversarially robust. We confirm their capabilities on several different datasets and with current state-of-the-art VAE adversarial attacks, and also show that they increase the robustness of downstream tasks to attack.

----

## [221] What Can You Learn From Your Muscles? Learning Visual Representation from Human Interactions

**Authors**: *Kiana Ehsani, Daniel Gordon, Thomas Hai Dang Nguyen, Roozbeh Mottaghi, Ali Farhadi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Qm8UNVCFdh](https://openreview.net/forum?id=Qm8UNVCFdh)

**Abstract**:

Learning effective representations of visual data that generalize to a variety of downstream tasks has been a long quest for computer vision. Most representation learning approaches rely solely on visual data such as images or videos. In this paper, we explore a novel approach, where we use human interaction and attention cues to investigate whether we can learn better representations compared to visual-only representations. For this study, we collect a dataset of human interactions capturing body part movements and gaze in their daily lives. Our experiments show that our ``"muscly-supervised" representation that encodes interaction and attention cues outperforms a visual-only state-of-the-art method MoCo (He et al.,2020), on a variety of target tasks: scene classification (semantic), action recognition (temporal), depth estimation (geometric), dynamics prediction (physics) and walkable surface estimation (affordance). Our code and dataset are available at: https://github.com/ehsanik/muscleTorch.

----

## [222] EEC: Learning to Encode and Regenerate Images for Continual Learning

**Authors**: *Ali Ayub, Alan R. Wagner*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lWaz5a9lcFU](https://openreview.net/forum?id=lWaz5a9lcFU)

**Abstract**:

The two main impediments to continual learning are catastrophic forgetting and memory limitations on the storage of data. To cope with these challenges, we propose a novel, cognitively-inspired approach which trains autoencoders with Neural Style Transfer to encode and store images. Reconstructed images from encoded episodes are replayed when training the classifier model on a new task to avoid catastrophic forgetting. The loss function for the reconstructed images is weighted to reduce its effect during classifier training to cope with image degradation. When the system runs out of memory the encoded episodes are converted into centroids and covariance matrices, which are used to generate pseudo-images during classifier training, keeping classifier performance stable with less memory. Our approach increases classification accuracy by 13-17% over state-of-the-art methods on benchmark datasets, while requiring 78% less storage space.

----

## [223] Impact of Representation Learning in Linear Bandits

**Authors**: *Jiaqi Yang, Wei Hu, Jason D. Lee, Simon Shaolei Du*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=edJ_HipawCa](https://openreview.net/forum?id=edJ_HipawCa)

**Abstract**:

We study how representation learning can improve the efficiency of bandit problems. We study the setting where we play $T$ linear bandits with dimension $d$ concurrently, and these $T$ bandit tasks share a common $k (\ll d)$ dimensional linear representation. For the finite-action setting, we present a new algorithm which achieves $\widetilde{O}(T\sqrt{kN} + \sqrt{dkNT})$ regret, where $N$ is the number of rounds we play for each bandit. When $T$ is sufficiently large, our algorithm significantly outperforms the naive algorithm (playing $T$ bandits independently) that achieves $\widetilde{O}(T\sqrt{d N})$ regret. We also provide an $\Omega(T\sqrt{kN} + \sqrt{dkNT})$ regret lower bound, showing that our algorithm is minimax-optimal up to poly-logarithmic factors.  Furthermore, we extend our algorithm to the infinite-action setting and obtain a corresponding regret bound which demonstrates the benefit of representation learning in certain regimes. We also present experiments on synthetic and real-world data to illustrate our theoretical findings and demonstrate the effectiveness of our proposed algorithms.

----

## [224] MODALS: Modality-agnostic Automated Data Augmentation in the Latent Space

**Authors**: *Tsz-Him Cheung, Dit-Yan Yeung*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XjYgR6gbCEc](https://openreview.net/forum?id=XjYgR6gbCEc)

**Abstract**:

Data augmentation is an efficient way to expand a training dataset by creating additional artificial data. While data augmentation is found to be effective in improving the generalization capabilities of models for various machine learning tasks, the underlying augmentation methods are usually manually designed and carefully evaluated for each data modality separately, like image processing functions for image data and word-replacing rules for text data. In this work, we propose an automated data augmentation approach called MODALS (Modality-agnostic Automated Data Augmentation in the Latent Space) to augment data for any modality in a generic way. MODALS exploits automated data augmentation to fine-tune four universal data transformation operations in the latent space to adapt the transform to data of different modalities. Through comprehensive experiments, we demonstrate the effectiveness of MODALS on multiple datasets for text, tabular, time-series and image modalities.

----

## [225] The Recurrent Neural Tangent Kernel

**Authors**: *Sina Alemohammad, Zichao Wang, Randall Balestriero, Richard G. Baraniuk*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3T9iFICe0Y9](https://openreview.net/forum?id=3T9iFICe0Y9)

**Abstract**:

The study of deep neural networks (DNNs) in the infinite-width limit, via the so-called neural tangent kernel (NTK) approach, has provided new insights into the dynamics of learning, generalization, and the impact of initialization. One key DNN architecture remains to be kernelized, namely, the recurrent neural network (RNN).  In this paper we introduce and study the Recurrent Neural Tangent Kernel (RNTK), which provides new insights into the behavior of  overparametrized RNNs. A key property of the RNTK should greatly benefit practitioners is its ability to compare inputs of different length. To this end, we characterize how the RNTK weights different time steps to form its output under different initialization parameters and nonlinearity choices. A synthetic and 56 real-world data experiments demonstrate that the RNTK offers significant performance gains over other kernels, including standard NTKs, across a wide array of data sets.

----

## [226] Projected Latent Markov Chain Monte Carlo: Conditional Sampling of Normalizing Flows

**Authors**: *Chris Cannella, Mohammadreza Soltani, Vahid Tarokh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MBpHUFrcG2x](https://openreview.net/forum?id=MBpHUFrcG2x)

**Abstract**:

We introduce Projected Latent Markov Chain Monte Carlo (PL-MCMC), a technique for sampling from the exact conditional distributions learned by normalizing flows. As a conditional sampling method, PL-MCMC enables Monte Carlo Expectation Maximization (MC-EM) training of normalizing flows from incomplete data. Through experimental tests applying normalizing flows to missing data tasks for a variety of data sets, we demonstrate the efficacy of PL-MCMC for conditional sampling from normalizing flows.

----

## [227] Learning the Pareto Front with Hypernetworks

**Authors**: *Aviv Navon, Aviv Shamsian, Ethan Fetaya, Gal Chechik*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NjF772F4ZZR](https://openreview.net/forum?id=NjF772F4ZZR)

**Abstract**:

Multi-objective optimization (MOO) problems are prevalent in machine learning. These problems have a set of optimal solutions, called the Pareto front, where each point on the front represents a different trade-off between possibly conflicting objectives. Recent MOO methods can target a specific desired ray in loss space however, most approaches still face two grave limitations: (i) A separate model has to be trained for each point on the front; and (ii) The exact trade-off must be known before the optimization process. Here, we tackle the problem of learning the entire Pareto front, with the capability of selecting a desired operating point on the front after training. We call this new setup Pareto-Front Learning (PFL).

We describe an approach to PFL implemented using HyperNetworks, which we term Pareto HyperNetworks (PHNs). PHN learns the entire Pareto front simultaneously using a single hypernetwork, which receives as input a desired preference vector and returns a Pareto-optimal model whose loss vector is in the desired ray. The unified model is runtime efficient compared to training multiple models and generalizes to new operating points not used during training. We evaluate our method on a wide set of problems, from multi-task regression and classification to fairness. PHNs learn the entire Pareto front at roughly the same time as learning a single point on the front and at the same time reach a better solution set. PFL opens the door to new applications where models are selected based on preferences that are only available at run time.

----

## [228] Estimating and Evaluating Regression Predictive Uncertainty in Deep Object Detectors

**Authors**: *Ali Harakeh, Steven L. Waslander*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YLewtnvKgR7](https://openreview.net/forum?id=YLewtnvKgR7)

**Abstract**:

Predictive uncertainty estimation is an essential next step for the reliable deployment of deep object detectors in safety-critical tasks. In this work, we focus on estimating predictive distributions for bounding box regression output with variance networks. We show that in the context of object detection, training variance networks with negative log likelihood (NLL) can lead to high entropy predictive distributions regardless of the correctness of the output mean. We propose to use the energy score as a non-local proper scoring rule and find that when used for training, the energy score leads to better calibrated and lower entropy predictive distributions than NLL. We also address the widespread use of non-proper scoring metrics for evaluating predictive distributions from deep object detectors by proposing an alternate evaluation approach founded on proper scoring rules. Using the proposed evaluation tools, we show that although variance networks can be used to produce high quality predictive distributions, ad-hoc approaches used by seminal object detectors for choosing regression targets during training do not provide wide enough data support for reliable variance learning. We hope that our work helps shift evaluation in probabilistic object detection to better align with predictive uncertainty evaluation in other machine learning domains. Code for all models, evaluation, and datasets is available at: https://github.com/asharakeh/probdet.git.

----

## [229] Predicting Classification Accuracy When Adding New Unobserved Classes

**Authors**: *Yuli Slavutsky, Yuval Benjamini*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Y9McSeEaqUh](https://openreview.net/forum?id=Y9McSeEaqUh)

**Abstract**:

Multiclass classifiers are often designed and evaluated only on a sample from the classes on which they will eventually be applied. Hence, their final accuracy remains unknown. In this work we study how a classifier’s performance over the initial class sample can be used to extrapolate its expected accuracy on a larger, unobserved set of classes. For this, we define a measure of separation between correct and incorrect classes that is independent of the number of classes: the "reversed ROC" (rROC), which is obtained by replacing the roles of classes and data-points in the common ROC. We show that the classification accuracy is a function of the rROC in multiclass classifiers, for which the learned representation of data from the initial class sample remains unchanged when new classes are added. Using these results we formulate a robust neural-network-based algorithm, "CleaneX", which learns to estimate the accuracy of such classifiers on arbitrarily large sets of classes. Unlike previous methods, our method uses both the observed accuracies of the classifier and densities of classification scores, and therefore achieves remarkably better predictions than current state-of-the-art methods on both simulations and real datasets of object detection, face recognition, and brain decoding.

----

## [230] BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction

**Authors**: *Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, Shi Gu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=POWv6hDd9XH](https://openreview.net/forum?id=POWv6hDd9XH)

**Abstract**:

We study the challenging task of neural network quantization without end-to-end retraining, called Post-training Quantization (PTQ). PTQ usually requires a small subset of training data but produces less powerful quantized models than Quantization-Aware Training (QAT). In this work, we propose a novel PTQ framework, dubbed BRECQ, which pushes the limits of bitwidth in PTQ down to INT2 for the first time. BRECQ leverages the basic building blocks in neural networks and reconstructs them one-by-one. In a comprehensive theoretical study of the second-order error, we show that BRECQ achieves a good balance between cross-layer dependency and generalization error. To further employ the power of quantization, the mixed precision technique is incorporated in our framework by approximating the inter-layer and intra-layer sensitivity. Extensive experiments on various handcrafted and searched neural architectures are conducted for both image classification and object detection tasks. And for the first time we prove that, without bells and whistles, PTQ can attain 4-bit ResNet and MobileNetV2 comparable with QAT and enjoy 240 times faster production of quantized models.  Codes are available at https://github.com/yhhhli/BRECQ.

----

## [231] No MCMC for me: Amortized sampling for fast and stable training of energy-based models

**Authors**: *Will Sussman Grathwohl, Jacob Jin Kelly, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, David Duvenaud*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ixpSxO9flk3](https://openreview.net/forum?id=ixpSxO9flk3)

**Abstract**:

Energy-Based Models (EBMs) present a flexible and appealing way to represent uncertainty. Despite recent advances, training EBMs on high-dimensional data remains a challenging problem as the state-of-the-art approaches are costly, unstable, and require considerable tuning and domain expertise to apply successfully. In this work, we present a simple method for training EBMs at scale which uses an entropy-regularized generator to amortize the MCMC sampling typically used in EBM training. We improve upon prior MCMC-based entropy regularization methods with a fast variational approximation. We demonstrate the effectiveness of our approach by using it to train tractable likelihood models. Next, we apply our estimator to the recently proposed Joint Energy Model (JEM), where we match the original performance with faster and stable training. This allows us to extend JEM models to semi-supervised classification on tabular data from a variety of continuous domains.

----

## [232] GraphCodeBERT: Pre-training Code Representations with Data Flow

**Authors**: *Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, Michele Tufano, Shao Kun Deng, Colin B. Clement, Dawn Drain, Neel Sundaresan, Jian Yin, Daxin Jiang, Ming Zhou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jLoC4ez43PZ](https://openreview.net/forum?id=jLoC4ez43PZ)

**Abstract**:

Pre-trained models for programming language have achieved dramatic empirical improvements on a variety of code-related tasks such as code search, code completion, code summarization, etc. However, existing pre-trained models regard a code snippet as a sequence of tokens, while ignoring the inherent structure of code, which provides crucial code semantics and would enhance the code understanding process. We present GraphCodeBERT, a pre-trained model for programming language that considers the inherent structure of code. Instead of taking syntactic-level structure of code like abstract syntax tree (AST), we use data flow in the pre-training stage, which is a semantic-level structure of code that encodes the relation of "where-the-value-comes-from" between variables. Such a semantic-level structure is neat and does not bring an unnecessarily deep hierarchy of AST, the property of which makes the model more efficient. We develop GraphCodeBERT based on Transformer. In addition to using the task of masked language modeling, we introduce two structure-aware pre-training tasks. One is to predict code structure edges, and the other is to align representations between source code and code structure. We implement the model in an efficient way with a graph-guided masked attention function to incorporate the code structure. We evaluate our model on four tasks, including code search, clone detection, code translation, and code refinement. Results show that code structure and newly introduced pre-training tasks can improve GraphCodeBERT and achieves state-of-the-art performance on the four downstream tasks. We further show that the model prefers structure-level attentions over token-level attentions in the task of code search.

----

## [233] Conservative Safety Critics for Exploration

**Authors**: *Homanga Bharadhwaj, Aviral Kumar, Nicholas Rhinehart, Sergey Levine, Florian Shkurti, Animesh Garg*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=iaO86DUuKi](https://openreview.net/forum?id=iaO86DUuKi)

**Abstract**:

Safe exploration presents a major challenge in reinforcement learning (RL): when active data collection requires deploying partially trained policies, we must ensure that these policies avoid catastrophically unsafe regions, while still enabling trial and error learning. In this paper, we target the problem of safe exploration in RL, by learning a conservative safety estimate of environment states through a critic, and provably upper bound the likelihood of catastrophic failures at every training iteration. We theoretically characterize the tradeoff between safety and policy improvement, show that the safety constraints are satisfied with high probability during training, derive provable convergence guarantees for our approach which is no worse asymptotically then standard RL, and empirically demonstrate the efficacy of the proposed approach on a suite of challenging navigation, manipulation, and locomotion tasks. Our results demonstrate that the proposed approach can achieve competitive task performance, while incurring significantly lower catastrophic failure rates during training as compared to prior methods. Videos are at this URL https://sites.google.com/view/conservative-safety-critics/

----

## [234] Improve Object Detection with Feature-based Knowledge Distillation: Towards Accurate and Efficient Detectors

**Authors**: *Linfeng Zhang, Kaisheng Ma*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uKhGRvM8QNH](https://openreview.net/forum?id=uKhGRvM8QNH)

**Abstract**:

Knowledge distillation, in which a student model is trained to mimic a teacher model, has been proved as an effective technique for model compression and model accuracy boosting. However, most knowledge distillation methods, designed for image classification, have failed on more challenging tasks, such as object detection. In this paper, we suggest that the failure of knowledge distillation on object detection is mainly caused by two reasons: (1) the imbalance between pixels of foreground and background and (2) lack of distillation on the relation between different pixels. Observing the above reasons, we propose attention-guided distillation and non-local distillation to address the two problems, respectively.  Attention-guided distillation is proposed to find the crucial pixels of foreground objects with attention mechanism and then make the students take more effort to learn their features. Non-local distillation is proposed to enable students to learn not only the feature of an individual pixel but also the relation between different pixels captured by non-local modules. Experiments show that our methods achieve excellent AP improvements on both one-stage and two-stage, both anchor-based and anchor-free detectors. For example, Faster RCNN (ResNet101 backbone) with our distillation achieves 43.9 AP on COCO2017, which is 4.1 higher than the baseline. Codes have been released on Github.

----

## [235] A Temporal Kernel Approach for Deep Learning with Continuous-time Information

**Authors**: *Da Xu, Chuanwei Ruan, Evren Körpeoglu, Sushant Kumar, Kannan Achan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=whE31dn74cL](https://openreview.net/forum?id=whE31dn74cL)

**Abstract**:

Sequential deep learning models such as RNN, causal CNN and attention mechanism do not readily consume continuous-time information. Discretizing the temporal data, as we show, causes inconsistency even for simple continuous-time processes. Current approaches often handle time in a heuristic manner to be consistent with the existing deep learning architectures and implementations. In this paper, we provide a principled way to characterize continuous-time systems using deep learning tools. Notably, the proposed approach applies to all the major deep learning architectures and requires little modifications to the implementation. The critical insight is to represent the continuous-time system by composing neural networks with a temporal kernel, where we gain our intuition from the recent advancements in understanding deep learning with Gaussian process and neural tangent kernel. To represent the temporal kernel, we introduce the random feature approach and convert the kernel learning problem to spectral density estimation under reparameterization. We further prove the convergence and consistency results even when the temporal kernel is non-stationary, and the spectral density is misspecified. The simulations and real-data experiments demonstrate the empirical effectiveness of our temporal kernel approach in a broad range of settings.

----

## [236] For self-supervised learning, Rationality implies generalization, provably

**Authors**: *Yamini Bansal, Gal Kaplun, Boaz Barak*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Srmggo3b3X6](https://openreview.net/forum?id=Srmggo3b3X6)

**Abstract**:

We prove a new upper bound on the generalization gap of classifiers that are obtained by first using self-supervision to learn a representation $r$ of the training~data, and then fitting a simple (e.g., linear) classifier $g$ to the labels. Specifically, we show that (under the assumptions described below) the generalization gap of such classifiers tends to zero if $\mathsf{C}(g) \ll n$, where $\mathsf{C}(g)$ is an appropriately-defined measure of the simple classifier $g$'s complexity, and $n$ is the number of training samples. We stress that our bound is independent of the complexity of the representation $r$. 
We do not make any structural or conditional-independence  assumptions on the representation-learning task, which can use the same training dataset that is later used for classification. Rather, we assume that the training procedure satisfies certain natural noise-robustness (adding small amount of label noise causes small degradation in performance) and rationality  (getting the wrong label is not better than getting no label at all) conditions that widely hold across many standard architectures.
We also conduct an extensive empirical study of the generalization gap and the quantities used in our assumptions for a variety of self-supervision based algorithms, including SimCLR, AMDIM and BigBiGAN,  on the CIFAR-10 and ImageNet datasets. We show that, unlike standard supervised classifiers, these algorithms display small generalization gap, and the bounds we prove on this gap are often non vacuous.

----

## [237] How to Find Your Friendly Neighborhood: Graph Attention Design with Self-Supervision

**Authors**: *Dongkwan Kim, Alice Oh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Wi5KUNlqWty](https://openreview.net/forum?id=Wi5KUNlqWty)

**Abstract**:

Attention mechanism in graph neural networks is designed to assign larger weights to important neighbor nodes for better representation. However, what graph attention learns is not understood well, particularly when graphs are noisy. In this paper, we propose a self-supervised graph attention network (SuperGAT), an improved graph attention model for noisy graphs. Specifically, we exploit two attention forms compatible with a self-supervised task to predict edges, whose presence and absence contain the inherent information about the importance of the relationships between nodes. By encoding edges, SuperGAT learns more expressive attention in distinguishing mislinked neighbors. We find two graph characteristics influence the effectiveness of attention forms and self-supervision: homophily and average degree. Thus, our recipe provides guidance on which attention design to use when those two graph characteristics are known. Our experiment on 17 real-world datasets demonstrates that our recipe generalizes across 15 datasets of them, and our models designed by recipe show improved performance over baselines.

----

## [238] Interpretable Models for Granger Causality Using Self-explaining Neural Networks

**Authors**: *Ricards Marcinkevics, Julia E. Vogt*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=DEa4JdMWRHp](https://openreview.net/forum?id=DEa4JdMWRHp)

**Abstract**:

Exploratory analysis of time series data can yield a better understanding of complex dynamical systems. Granger causality is a practical framework for analysing interactions in sequential data, applied in a wide range of domains. In this paper, we propose a novel framework for inferring multivariate Granger causality under nonlinear dynamics based on an extension of self-explaining neural networks. This framework is more interpretable than other neural-network-based techniques for inferring Granger causality, since in addition to relational inference, it also allows detecting signs of Granger-causal effects and inspecting their variability over time. In comprehensive experiments on simulated data, we show that our framework performs on par with several powerful baseline methods at inferring Granger causality and that it achieves better performance at inferring interaction signs. The results suggest that our framework is a viable and more interpretable alternative to sparse-input neural networks for inferring Granger causality.

----

## [239] Meta-learning Symmetries by Reparameterization

**Authors**: *Allan Zhou, Tom Knowles, Chelsea Finn*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-QxT4mJdijq](https://openreview.net/forum?id=-QxT4mJdijq)

**Abstract**:

Many successful deep learning architectures are equivariant to certain transformations in order to conserve parameters and improve generalization: most famously, convolution layers are equivariant to shifts of the input. This approach only works when practitioners know the symmetries of the task and can manually construct an architecture with the corresponding equivariances. Our goal is an approach for learning equivariances from data, without needing to design custom task-specific architectures. We present a method for learning and encoding equivariances into networks by learning corresponding parameter sharing patterns from data. Our method can provably represent equivariance-inducing parameter sharing for any finite group of symmetry transformations. Our experiments suggest that it can automatically learn to encode equivariances to common transformations used in image processing tasks.

----

## [240] Removing Undesirable Feature Contributions Using Out-of-Distribution Data

**Authors**: *Saehyung Lee, Changhwa Park, Hyungyu Lee, Jihun Yi, Jonghyun Lee, Sungroh Yoon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eIHYL6fpbkA](https://openreview.net/forum?id=eIHYL6fpbkA)

**Abstract**:

Several data augmentation methods deploy unlabeled-in-distribution (UID) data to bridge the gap between the training and inference of neural networks. However, these methods have clear limitations in terms of availability of UID data and dependence of algorithms on pseudo-labels. Herein, we propose a data augmentation method to improve generalization in both adversarial and standard learning by using out-of-distribution (OOD) data that are devoid of the abovementioned issues. We show how to improve generalization theoretically using OOD data in each learning scenario and complement our theoretical analysis with experiments on CIFAR-10, CIFAR-100, and a subset of ImageNet. The results indicate that undesirable features are shared even among image data that seem to have little correlation from a human point of view. We also present the advantages of the proposed method through comparison with other data augmentation methods, which can be used in the absence of UID data. Furthermore, we demonstrate that the proposed method can further improve the existing state-of-the-art adversarial training.

----

## [241] Mind the Gap when Conditioning Amortised Inference in Sequential Latent-Variable Models

**Authors**: *Justin Bayer, Maximilian Soelch, Atanas Mirchev, Baris Kayalibay, Patrick van der Smagt*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=a2gqxKDvYys](https://openreview.net/forum?id=a2gqxKDvYys)

**Abstract**:

Amortised inference enables scalable learning of sequential latent-variable models (LVMs) with the evidence lower bound (ELBO). In this setting, variational posteriors are often only partially conditioned. While the true posteriors depend, e.g., on the entire sequence of observations, approximate posteriors are only informed by past observations. This mimics the Bayesian filter---a mixture of smoothing posteriors. Yet, we show that the ELBO objective forces partially-conditioned amortised posteriors to approximate products of smoothing posteriors instead. Consequently, the learned generative model is compromised. We demonstrate these theoretical findings in three scenarios: traffic flow, handwritten digits, and aerial vehicle dynamics. Using fully-conditioned approximate posteriors, performance improves in terms of generative modelling and multi-step prediction.

----

## [242] On the Universality of the Double Descent Peak in Ridgeless Regression

**Authors**: *David Holzmüller*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0IO5VdnSAaH](https://openreview.net/forum?id=0IO5VdnSAaH)

**Abstract**:

We prove a non-asymptotic distribution-independent lower bound for the expected mean squared generalization error caused by label noise in ridgeless linear regression. Our lower bound generalizes a similar known result to the overparameterized (interpolating) regime. In contrast to most previous works, our analysis applies to a broad class of input distributions with almost surely full-rank feature matrices, which allows us to cover various types of deterministic or random feature maps. Our lower bound is asymptotically sharp and implies that in the presence of label noise, ridgeless linear regression does not perform well around the interpolation threshold for any of these feature maps. We analyze the imposed assumptions in detail and provide a theory for analytic (random) feature maps. Using this theory, we can show that our assumptions are satisfied for input distributions with a (Lebesgue) density and feature maps given by random deep neural networks with analytic activation functions like sigmoid, tanh, softplus or GELU. As further examples, we show that feature maps from random Fourier features and polynomial kernels also satisfy our assumptions. We complement our theory with further experimental and analytic results.

----

## [243] Fair Mixup: Fairness via Interpolation

**Authors**: *Ching-Yao Chuang, Youssef Mroueh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=DNl5s5BXeBn](https://openreview.net/forum?id=DNl5s5BXeBn)

**Abstract**:

Training classifiers under fairness constraints such as group fairness, regularizes the disparities of predictions between the groups. Nevertheless, even though the constraints are satisfied during training, they might not generalize at evaluation time. To improve the generalizability of fair classifiers, we propose fair mixup, a new data augmentation strategy for imposing the fairness constraint. In particular, we show that fairness can be achieved by regularizing the models on paths of interpolated samples  between the groups. We use mixup, a powerful data augmentation strategy  to generate these interpolates. We analyze fair mixup and empirically show that it ensures a better generalization for both accuracy and fairness measurement in tabular, vision, and language benchmarks.

----

## [244] Self-supervised Learning from a Multi-view Perspective

**Authors**: *Yao-Hung Hubert Tsai, Yue Wu, Ruslan Salakhutdinov, Louis-Philippe Morency*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-bdp_8Itjwp](https://openreview.net/forum?id=-bdp_8Itjwp)

**Abstract**:

As a subset of unsupervised representation learning, self-supervised representation learning adopts self-defined signals as supervision and uses the learned representation for downstream tasks, such as object detection and image captioning. Many proposed approaches for self-supervised learning follow naturally a multi-view perspective, where the input (e.g., original images) and the self-supervised signals (e.g., augmented images) can be seen as two redundant views of the data. Building from this multi-view perspective, this paper provides an information-theoretical framework to better understand the properties that encourage successful self-supervised learning. Specifically, we demonstrate that self-supervised learned representations can extract task-relevant information and discard task-irrelevant information. Our theoretical framework paves the way to a larger space of self-supervised learning objective design. In particular, we propose a composite objective that bridges the gap between prior contrastive and predictive learning objectives, and introduce an additional objective term to discard task-irrelevant information. To verify our analysis, we conduct controlled experiments to evaluate the impact of the composite objectives. We also explore our framework's empirical generalization beyond the multi-view perspective, where the cross-view redundancy may not be clearly observed.

----

## [245] Integrating Categorical Semantics into Unsupervised Domain Translation

**Authors**: *Samuel Lavoie-Marchildon, Faruk Ahmed, Aaron C. Courville*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=IMPA6MndSXU](https://openreview.net/forum?id=IMPA6MndSXU)

**Abstract**:

While unsupervised domain translation (UDT) has seen a lot of success recently, we argue that mediating its translation via categorical semantic features could broaden its applicability. In particular, we demonstrate that categorical semantics improves the translation between perceptually different domains sharing multiple object categories. We propose a method to learn, in an unsupervised manner, categorical semantic features (such as object labels) that are invariant of the source and target domains. We show that conditioning the style encoder of unsupervised domain translation methods on the learned categorical semantics leads to a translation preserving the digits on MNIST$\leftrightarrow$SVHN and to a more realistic stylization on Sketches$\to$Reals.

----

## [246] The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods

**Authors**: *Louis Thiry, Michael Arbel, Eugene Belilovsky, Edouard Oyallon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=aYuZO9DIdnn](https://openreview.net/forum?id=aYuZO9DIdnn)

**Abstract**:

A recent line of work showed that  various forms of convolutional  kernel methods can be competitive with standard supervised deep convolutional networks on datasets like CIFAR-10, obtaining accuracies in the range of 87-90% while being more amenable to theoretical analysis. In this work, we highlight the importance of a data-dependent feature extraction step that is key to the obtain good performance in convolutional kernel methods. This step typically corresponds to a whitened dictionary of patches, and gives rise to a data-driven convolutional kernel methods.We extensively study its effect, demonstrating it is the key ingredient for high performance of these methods. Specifically, we show that one of the simplest instances of such kernel methods, based on a single layer of  image patches followed by a linear classifier is already obtaining classification accuracies on CIFAR-10 in the same range as previous more sophisticated convolutional kernel methods. We scale this method to the challenging ImageNet dataset, showing such a simple approach can exceed all existing non-learned representation methods. This is a new baseline for object recognition without representation learning methods, that  initiates the investigation of  convolutional kernel models  on ImageNet. We conduct experiments to analyze the dictionary that we used, our ablations showing they exhibit low-dimensional properties.

----

## [247] Open Question Answering over Tables and Text

**Authors**: *Wenhu Chen, Ming-Wei Chang, Eva Schlinger, William Yang Wang, William W. Cohen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MmCRswl1UYl](https://openreview.net/forum?id=MmCRswl1UYl)

**Abstract**:

In open question answering (QA), the answer to a question is produced by retrieving and then analyzing documents that might contain answers to the question.  Most open QA systems have considered only retrieving information from unstructured text.  Here we consider for the first time open QA over {\em both} tabular and textual data and present a new large-scale dataset \emph{Open Table-and-Text Question Answering} (OTT-QA) to evaluate performance on this task. Most questions in OTT-QA require multi-hop inference across tabular data and unstructured text, and the evidence required to answer a question can be distributed in different ways over these two types of input, making evidence retrieval challenging---our baseline model using an iterative retriever and BERT-based reader achieves an exact match score less than 10\%. We then propose two novel techniques to address the challenge of retrieving and aggregating evidence for OTT-QA. The first technique is to use ``early fusion'' to group multiple highly relevant tabular and textual units into a fused block, which provides more context for the retriever to search for.  The second technique is to use a cross-block reader to model the cross-dependency between multiple retrieved evidence with global-local sparse attention. Combining these two techniques improves the score significantly, to above 27\%.

----

## [248] Evaluation of Similarity-based Explanations

**Authors**: *Kazuaki Hanawa, Sho Yokoi, Satoshi Hara, Kentaro Inui*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9uvhpyQwzM_](https://openreview.net/forum?id=9uvhpyQwzM_)

**Abstract**:

Explaining the predictions made by complex machine learning models helps users to understand and accept the predicted outputs with confidence. One promising way is to use similarity-based explanation that provides similar instances as evidence to support model predictions. Several relevance metrics are used for this purpose. In this study, we investigated relevance metrics that can provide reasonable explanations to users. Specifically, we adopted three tests to evaluate whether the relevance metrics satisfy the minimal requirements for similarity-based explanation. Our experiments revealed that the cosine similarity of the gradients of the loss performs best, which would be a recommended choice in practice. In addition, we showed that some metrics perform poorly in our tests and analyzed the reasons of their failure. We expect our insights to help practitioners in selecting appropriate relevance metrics and also aid further researches for designing better relevance metrics for explanations.

----

## [249] A Diffusion Theory For Deep Learning Dynamics: Stochastic Gradient Descent Exponentially Favors Flat Minima

**Authors**: *Zeke Xie, Issei Sato, Masashi Sugiyama*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=wXgk_iCiYGo](https://openreview.net/forum?id=wXgk_iCiYGo)

**Abstract**:

Stochastic Gradient Descent (SGD) and its variants are mainstream methods for training deep networks in practice. SGD is known to find a flat minimum that often generalizes well. However, it is mathematically unclear how deep learning can select a flat minimum among so many minima. To answer the question quantitatively, we develop a density diffusion theory to reveal how minima selection quantitatively depends on the minima sharpness and the hyperparameters. To the best of our knowledge, we are the first to theoretically and empirically prove that, benefited from the Hessian-dependent covariance of stochastic gradient noise, SGD favors flat minima exponentially more than sharp minima, while Gradient Descent (GD) with injected white noise favors flat minima only polynomially more than sharp minima. We also reveal that either a small learning rate or large-batch training requires exponentially many iterations to escape from minima in terms of the ratio of the batch size and learning rate. Thus, large-batch training cannot search flat minima efficiently in a realistic computational time.

----

## [250] How Much Over-parameterization Is Sufficient to Learn Deep ReLU Networks?

**Authors**: *Zixiang Chen, Yuan Cao, Difan Zou, Quanquan Gu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=fgd7we_uZa6](https://openreview.net/forum?id=fgd7we_uZa6)

**Abstract**:

A recent line of research on deep learning focuses on the extremely over-parameterized setting, and shows that when the network width is larger than a high degree polynomial of the training sample size $n$ and the inverse of the target error $\epsilon^{-1}$, deep neural networks learned by (stochastic) gradient descent enjoy nice optimization and generalization guarantees. Very recently, it is shown that under certain margin assumptions on the training data, a polylogarithmic width condition suffices for two-layer ReLU networks to converge and generalize (Ji and Telgarsky, 2020). However, whether deep neural networks can be learned with such a mild over-parameterization is still an open question. In this work, we answer this question affirmatively and establish sharper learning guarantees for deep ReLU networks trained by (stochastic) gradient descent. In specific, under certain assumptions made in previous work, our optimization and generalization guarantees hold with network width polylogarithmic in $n$ and $\epsilon^{-1}$. Our results push the study of over-parameterized deep neural networks towards more practical settings.

----

## [251] Auction Learning as a Two-Player Game

**Authors**: *Jad Rahme, Samy Jelassi, S. Matthew Weinberg*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YHdeAO61l6T](https://openreview.net/forum?id=YHdeAO61l6T)

**Abstract**:

Designing an incentive compatible auction that maximizes expected revenue is a central problem in Auction Design. While theoretical approaches to the problem have hit some limits, a recent research direction initiated by Duetting et al. (2019) consists in building neural network architectures to find optimal auctions.  We propose two conceptual deviations from their approach which result in enhanced performance. First, we use recent results in theoretical auction design to introduce a time-independent Lagrangian.  This not only circumvents the need for an expensive hyper-parameter search (as in prior work), but also provides a single metric to compare the performance of two auctions (absent from prior work). Second, the optimization procedure in previous work uses an inner maximization loop to compute optimal misreports. We amortize this process through the introduction of an additional neural network. We demonstrate the effectiveness of our approach by learning competitive or strictly improved auctions compared to prior work. Both results together further imply a novel formulation of Auction Design as a two-player game with stationary utility functions.

----

## [252] Robust Reinforcement Learning on State Observations with Learned Optimal Adversary

**Authors**: *Huan Zhang, Hongge Chen, Duane S. Boning, Cho-Jui Hsieh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=sCZbhBvqQaU](https://openreview.net/forum?id=sCZbhBvqQaU)

**Abstract**:

We study the robustness of reinforcement learning (RL) with adversarially perturbed state observations, which aligns with the setting of many adversarial attacks to deep reinforcement learning (DRL) and is also important for rolling out real-world RL agent under unpredictable sensing noise. With a fixed agent policy, we demonstrate that an optimal adversary to perturb state observations can be found, which is guaranteed to obtain the worst case agent reward. For DRL settings, this leads to a novel empirical adversarial attack to RL agents via a learned adversary that is much stronger than previous ones. To enhance the robustness of an agent, we propose a framework of alternating training with learned adversaries (ATLA), which trains an adversary online together with the agent using policy gradient following the optimal adversarial attack framework. Additionally, inspired by the analysis of state-adversarial Markov decision process (SA-MDP), we show that past states and actions (history) can be useful for learning a robust agent, and we empirically find a LSTM based policy can be more robust under adversaries. Empirical evaluations on a few continuous control environments show that ATLA achieves state-of-the-art performance under strong adversaries. Our code is available at https://github.com/huanzhang12/ATLA_robust_RL.

----

## [253] Optimizing Memory Placement using Evolutionary Graph Reinforcement Learning

**Authors**: *Shauharda Khadka, Estelle Aflalo, Mattias Marder, Avrech Ben-David, Santiago Miret, Shie Mannor, Tamir Hazan, Hanlin Tang, Somdeb Majumdar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-6vS_4Kfz0](https://openreview.net/forum?id=-6vS_4Kfz0)

**Abstract**:

For deep neural network accelerators, memory movement is both energetically expensive and can bound computation. Therefore, optimal mapping of tensors to memory hierarchies is critical to performance. The growing complexity of neural networks calls for automated memory mapping instead of manual heuristic approaches; yet the search space of neural network computational graphs have previously been prohibitively large. We introduce Evolutionary Graph Reinforcement Learning (EGRL), a method designed for large search spaces, that combines graph neural networks, reinforcement learning, and evolutionary search. A set of fast, stateless policies guide the evolutionary search to improve its sample-efficiency. We train and validate our approach directly on the Intel NNP-I chip for inference. EGRL outperforms policy-gradient, evolutionary search and dynamic programming baselines on BERT, ResNet-101 and ResNet-50. We additionally achieve 28-78% speed-up compared to the native NNP-I compiler on all three workloads.

----

## [254] Hierarchical Autoregressive Modeling for Neural Video Compression

**Authors**: *Ruihan Yang, Yibo Yang, Joseph Marino, Stephan Mandt*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TK_6nNb_C7q](https://openreview.net/forum?id=TK_6nNb_C7q)

**Abstract**:

Recent work by Marino et al. (2020) showed improved performance in sequential density estimation by combining masked autoregressive flows with hierarchical latent variable models. We draw a connection between such autoregressive generative models and the task of lossy video compression. Specifically, we view recent neural video compression methods (Lu et al., 2019; Yang et al., 2020b; Agustssonet al., 2020) as instances of a generalized stochastic temporal autoregressive transform, and propose avenues for enhancement based on this insight. Comprehensive evaluations on large-scale video data show improved rate-distortion performance over both state-of-the-art neural and conventional video compression methods.

----

## [255] Individually Fair Rankings

**Authors**: *Amanda Bower, Hamid Eftekhari, Mikhail Yurochkin, Yuekai Sun*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=71zCSP_HuBN](https://openreview.net/forum?id=71zCSP_HuBN)

**Abstract**:

We develop an algorithm to train individually fair learning-to-rank (LTR) models. The proposed approach ensures items from minority groups appear alongside similar items from majority groups. This notion of fair ranking is based on the definition of individual fairness from supervised learning and is more nuanced than prior fair LTR approaches that simply ensure the ranking model provides underrepresented items with a basic level of exposure. The crux of our method is an optimal transport-based regularizer that enforces individual fairness and an efficient algorithm for optimizing the regularizer. We show that our approach leads to certifiably individually fair LTR models and demonstrate the efficacy of our method on ranking tasks subject to demographic biases.

----

## [256] Learning Neural Generative Dynamics for Molecular Conformation Generation

**Authors**: *Minkai Xu, Shitong Luo, Yoshua Bengio, Jian Peng, Jian Tang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=pAbm1qfheGk](https://openreview.net/forum?id=pAbm1qfheGk)

**Abstract**:

We study how to generate molecule conformations (i.e., 3D structures) from a molecular graph. Traditional methods, such as molecular dynamics, sample conformations via computationally expensive simulations. Recently, machine learning methods have shown great potential by training on a large collection of conformation data. Challenges arise from the limited model capacity for capturing complex distributions of conformations and the difficulty in modeling long-range dependencies between atoms. Inspired by the recent progress in deep generative models, in this paper, we propose a novel probabilistic framework to generate valid and diverse conformations given a molecular graph. We propose a method combining the advantages of both flow-based and energy-based models, enjoying: (1) a high model capacity to estimate the multimodal conformation distribution; (2) explicitly capturing the complex long-range dependencies between atoms in the observation space. Extensive experiments demonstrate the superior performance of the proposed method on several benchmarks, including conformation generation and distance modeling tasks, with a significant improvement over existing generative models for molecular conformation sampling.

----

## [257] Efficient Certified Defenses Against Patch Attacks on Image Classifiers

**Authors**: *Jan Hendrik Metzen, Maksym Yatsura*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hr-3PMvDpil](https://openreview.net/forum?id=hr-3PMvDpil)

**Abstract**:

Adversarial patches pose a realistic threat model for physical world attacks on autonomous systems via their perception component. Autonomous systems in safety-critical domains such as automated driving should thus contain a fail-safe fallback component that combines certifiable robustness against patches with efficient inference while maintaining high performance on clean inputs. We propose BagCert, a novel combination of model architecture and certification procedure that allows efficient certification. We derive a loss that enables end-to-end optimization of certified robustness against patches of different sizes and locations. On CIFAR10, BagCert certifies 10.000 examples in 43 seconds on a single GPU and obtains 86% clean and 60% certified accuracy against 5x5 patches.

----

## [258] Convex Regularization behind Neural Reconstruction

**Authors**: *Arda Sahiner, Morteza Mardani, Batu Ozturkler, Mert Pilanci, John M. Pauly*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=VErQxgyrbfn](https://openreview.net/forum?id=VErQxgyrbfn)

**Abstract**:

Neural networks have shown tremendous potential for reconstructing high-resolution images in inverse problems. The non-convex and opaque nature of neural networks, however, hinders their utility in sensitive applications such as medical imaging. To cope with this challenge, this paper advocates a convex duality framework that makes a two-layer fully-convolutional ReLU denoising network amenable to convex optimization. The convex dual network not only offers the optimum training with convex solvers, but also facilitates interpreting training and prediction. In particular, it implies training neural networks with weight decay regularization induces path sparsity while the prediction is piecewise linear filtering. A range of experiments with MNIST and fastMRI datasets confirm the efficacy of the dual network optimization problem.

----

## [259] Targeted Attack against Deep Neural Networks via Flipping Limited Weight Bits

**Authors**: *Jiawang Bai, Baoyuan Wu, Yong Zhang, Yiming Li, Zhifeng Li, Shu-Tao Xia*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=iKQAk8a2kM0](https://openreview.net/forum?id=iKQAk8a2kM0)

**Abstract**:

To explore the vulnerability of deep neural networks (DNNs), many attack paradigms have been well studied, such as the poisoning-based backdoor attack in the training stage and the adversarial attack in the inference stage. In this paper, we study a novel attack paradigm, which modifies model parameters in the deployment stage for malicious purposes. Specifically, our goal is to misclassify a specific sample into a target class without any sample modification, while not significantly reduce the prediction accuracy of other samples to ensure the stealthiness. To this end, we formulate this problem as a binary integer programming (BIP), since the parameters are stored as binary bits ($i.e.$, 0 and 1) in the memory. By utilizing the latest technique in integer programming, we equivalently reformulate this BIP problem as a continuous optimization problem, which can be effectively and efficiently solved using the alternating direction method of multipliers (ADMM) method. Consequently, the flipped critical bits can be easily determined through optimization, rather than using a heuristic strategy. Extensive experiments demonstrate the superiority of our method in attacking DNNs.

----

## [260] Generalized Multimodal ELBO

**Authors**: *Thomas M. Sutter, Imant Daunhawer, Julia E. Vogt*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5Y21V0RDBV](https://openreview.net/forum?id=5Y21V0RDBV)

**Abstract**:

Multiple data types naturally co-occur when describing real-world phenomena and learning from them is a long-standing goal in machine learning research. However, existing self-supervised generative models approximating an ELBO are not able to fulfill all desired requirements of multimodal models: their posterior approximation functions lead to a trade-off between the semantic coherence and the ability to learn the joint data distribution. We propose a new, generalized ELBO formulation for multimodal data that overcomes these limitations. The new objective encompasses two previous methods as special cases and combines their benefits without compromises. In extensive experiments, we demonstrate the advantage of the proposed method compared to state-of-the-art models in self-supervised, generative learning tasks.

----

## [261] Large-width functional asymptotics for deep Gaussian neural networks

**Authors**: *Daniele Bracale, Stefano Favaro, Sandra Fortini, Stefano Peluchetti*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0aW6lYOYB7d](https://openreview.net/forum?id=0aW6lYOYB7d)

**Abstract**:

In this paper, we consider fully connected feed-forward deep neural networks where weights and biases are independent and identically distributed according to Gaussian distributions. Extending previous results (Matthews et al., 2018a;b;Yang, 2019)  we adopt a function-space perspective, i.e. we look at neural networks as infinite-dimensional random elements on the input space $\mathbb{R}^I$. Under suitable assumptions on the activation function we show that: i) a network defines a continuous Gaussian process on the input space $\mathbb{R}^I$; ii) a network with re-scaled weights converges weakly to a continuous Gaussian process in the large-width limit; iii) the limiting Gaussian process has almost surely locally $\gamma$-Hölder continuous paths, for $0 < \gamma <1$. Our results contribute to recent theoretical studies on the interplay between infinitely wide deep neural networks and Gaussian processes by establishing weak convergence in function-space with respect to a stronger metric.

----

## [262] Distributed Momentum for Byzantine-resilient Stochastic Gradient Descent

**Authors**: *El Mahdi El Mhamdi, Rachid Guerraoui, Sébastien Rouault*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=H8UHdhWG6A3](https://openreview.net/forum?id=H8UHdhWG6A3)

**Abstract**:

Byzantine-resilient Stochastic Gradient Descent (SGD) aims at shielding model training from Byzantine faults, be they ill-labeled training datapoints, exploited software/hardware vulnerabilities, or malicious worker nodes in a distributed setting.
Two recent attacks have been challenging state-of-the-art defenses though, often successfully precluding the model from even fitting the training set.
The main identified weakness in current defenses is their requirement of a sufficiently low variance-norm ratio for the stochastic gradients.
We propose a practical method which, despite increasing the variance, reduces the variance-norm ratio, mitigating the identified weakness.
We assess the effectiveness of our method over 736 different training configurations, comprising the 2 state-of-the-art attacks and 6 defenses.
For confidence and reproducibility purposes, each configuration is run 5 times with specified seeds (1 to 5), totalling 3680 runs.
In our experiments, when the attack is effective enough to decrease the highest observed top-1 cross-accuracy by at least 20% compared to the unattacked run, our technique systematically increases back the highest observed accuracy, and is able to recover at least 20% in more than 60% of the cases.

----

## [263] Fully Unsupervised Diversity Denoising with Convolutional Variational Autoencoders

**Authors**: *Mangal Prakash, Alexander Krull, Florian Jug*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=agHLCOBM5jP](https://openreview.net/forum?id=agHLCOBM5jP)

**Abstract**:

Deep Learning based methods have emerged as the indisputable leaders for virtually all image restoration tasks. Especially in the domain of microscopy images, various content-aware image restoration (CARE) approaches are now used to improve the interpretability of acquired data. Naturally, there are limitations to what can be restored in corrupted images, and like for all inverse problems, many potential solutions exist, and one of them must be chosen. Here, we propose DivNoising, a denoising approach based on fully convolutional variational autoencoders (VAEs), overcoming the problem of having to choose a single solution by predicting a whole distribution of denoised images. First we introduce a principled way of formulating the unsupervised denoising problem within the VAE framework by explicitly incorporating imaging noise models into the decoder. Our approach is fully unsupervised, only requiring noisy images and a suitable description of the imaging noise distribution. We show that such a noise model can either be measured, bootstrapped from noisy data, or co-learned during training. If desired, consensus predictions can be inferred from a set of DivNoising predictions, leading to competitive results with other unsupervised methods and, on occasion, even with the supervised state-of-the-art. DivNoising samples from the posterior enable a plethora of useful applications. We are (i) showing denoising results for 13 datasets, (ii) discussing how optical character recognition (OCR) applications can benefit from diverse predictions, and are (iii) demonstrating how instance cell segmentation improves when using diverse DivNoising predictions.

----

## [264] Auxiliary Learning by Implicit Differentiation

**Authors**: *Aviv Navon, Idan Achituve, Haggai Maron, Gal Chechik, Ethan Fetaya*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=n7wIfYPdVet](https://openreview.net/forum?id=n7wIfYPdVet)

**Abstract**:

Training neural networks with auxiliary tasks is a common practice for improving the performance on a main task of interest.
Two main challenges arise in this multi-task learning setting: (i) designing useful auxiliary tasks; and (ii) combining auxiliary tasks into a single coherent loss. Here, we propose a novel framework, AuxiLearn, that targets both challenges based on implicit differentiation. First, when useful auxiliaries are known, we propose learning a network that combines all losses into a single coherent objective function. This network can learn non-linear interactions between tasks. Second, when no useful auxiliary task is known, we describe how to learn a network that generates a meaningful, novel auxiliary task. We evaluate AuxiLearn in a series of tasks and domains, including image segmentation and learning with attributes in the low data regime, and find that it consistently outperforms competing methods.

----

## [265] Balancing Constraints and Rewards with Meta-Gradient D4PG

**Authors**: *Dan A. Calian, Daniel J. Mankowitz, Tom Zahavy, Zhongwen Xu, Junhyuk Oh, Nir Levine, Timothy A. Mann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TQt98Ya7UMP](https://openreview.net/forum?id=TQt98Ya7UMP)

**Abstract**:

Deploying Reinforcement Learning (RL) agents to solve real-world applications often requires satisfying complex system constraints. Often the constraint thresholds are incorrectly set due to the complex nature of a system or the inability to verify the thresholds offline (e.g, no simulator or reasonable offline evaluation procedure exists). This results in solutions where a task cannot be solved without violating the constraints. However, in many real-world cases, constraint violations are undesirable yet they are not catastrophic, motivating the need for soft-constrained RL approaches. We present two soft-constrained RL approaches that utilize meta-gradients to find a good trade-off between expected return and minimizing constraint violations. We demonstrate the effectiveness of these approaches by showing that they consistently outperform the baselines across four different Mujoco domains.

----

## [266] Adversarially Guided Actor-Critic

**Authors**: *Yannis Flet-Berliac, Johan Ferret, Olivier Pietquin, Philippe Preux, Matthieu Geist*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_mQp5cr_iNy](https://openreview.net/forum?id=_mQp5cr_iNy)

**Abstract**:

Despite definite success in deep reinforcement learning problems, actor-critic algorithms are still confronted with sample inefficiency in complex environments, particularly in tasks where efficient exploration is a bottleneck. These methods consider a policy (the actor) and a value function (the critic) whose respective losses are built using different motivations and approaches. This paper introduces a third protagonist: the adversary. While the adversary mimics the actor by minimizing the KL-divergence between their respective action distributions, the actor, in addition to learning to solve the task, tries to differentiate itself from the adversary predictions. This novel objective stimulates the actor to follow strategies that could not have been correctly predicted from previous trajectories, making its behavior innovative in tasks where the reward is extremely rare. Our experimental analysis shows that the resulting Adversarially Guided Actor-Critic (AGAC) algorithm leads to more exhaustive exploration. Notably, AGAC outperforms current state-of-the-art methods on a set of various hard-exploration and procedurally-generated tasks.

----

## [267] DARTS-: Robustly Stepping out of Performance Collapse Without Indicators

**Authors**: *Xiangxiang Chu, Xiaoxing Wang, Bo Zhang, Shun Lu, Xiaolin Wei, Junchi Yan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KLH36ELmwIB](https://openreview.net/forum?id=KLH36ELmwIB)

**Abstract**:

Despite the fast development of differentiable architecture search (DARTS), it suffers from a standing instability issue regarding searching performance, which extremely limits its application. Existing robustifying methods draw clues from the outcome instead of finding out the causing factor. Various indicators such as Hessian eigenvalues are proposed as a signal of performance collapse, and the searching should be stopped once an indicator reaches a preset threshold.
However, these methods tend to easily reject good architectures if thresholds are inappropriately set, let alone the searching is intrinsically noisy. In this paper, we undertake a more subtle and direct approach to resolve the collapse. 
We first demonstrate that skip connections with a learnable architectural coefficient can easily recover from a disadvantageous state and become dominant.  We conjecture that skip connections profit too much from this privilege, hence causing the collapse for the derived model. Therefore, we propose to factor out this benefit with an auxiliary skip connection, ensuring a fairer competition for all operations. Extensive experiments on various datasets verify that our approach can substantially improve the robustness of DARTS. Our code is available at https://github.com/Meituan-AutoML/DARTS-

----

## [268] Are wider nets better given the same number of parameters?

**Authors**: *Anna Golubeva, Guy Gur-Ari, Behnam Neyshabur*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_zx8Oka09eF](https://openreview.net/forum?id=_zx8Oka09eF)

**Abstract**:

Empirical studies demonstrate that the performance of neural networks improves with increasing number of parameters. In most of these studies, the number of parameters is increased by increasing the network width. This begs the question: Is the observed improvement due to the larger number of parameters, or is it due to the larger width itself? We compare different ways of increasing model width while keeping the number of parameters constant. We show that for models initialized with a random, static sparsity pattern in the weight tensors, network width is the determining factor for good performance, while the number of weights is secondary, as long as the model achieves high training accuarcy. As a step towards understanding this effect, we analyze these models in the framework of Gaussian Process kernels. We find that the distance between the sparse finite-width model kernel and the infinite-width kernel at initialization is indicative of model performance.

----

## [269] Optimal Conversion of Conventional Artificial Neural Networks to Spiking Neural Networks

**Authors**: *Shikuang Deng, Shi Gu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=FZ1oTwcXchK](https://openreview.net/forum?id=FZ1oTwcXchK)

**Abstract**:

Spiking neural networks (SNNs) are biology-inspired artificial neural networks (ANNs) that comprise of spiking neurons to process asynchronous discrete signals. While more efficient in power consumption and inference speed on the neuromorphic hardware, SNNs are usually difficult to train directly from scratch with spikes due to the discreteness. As an alternative, many efforts have been devoted to converting conventional ANNs into SNNs by copying the weights from ANNs and adjusting the spiking threshold potential of neurons in SNNs. Researchers have designed new SNN architectures and conversion algorithms to diminish the conversion error. However, an effective conversion should address the difference between the SNN and ANN architectures with an efficient approximation of the loss function, which is missing in the field. In this work, we analyze the conversion error by recursive reduction to layer-wise summation and propose a novel strategic pipeline that transfers the weights to the target SNN by combining threshold balance and soft-reset mechanisms. This pipeline enables almost no accuracy loss between the converted SNNs and conventional ANNs with only $\sim1/10$ of the typical SNN simulation time. Our method is promising to get implanted onto embedded platforms with better support of SNNs with limited energy and memory. Codes are available at https://github.com/Jackn0/snn_optimal_conversion_pipeline.

----

## [270] Deep Equals Shallow for ReLU Networks in Kernel Regimes

**Authors**: *Alberto Bietti, Francis R. Bach*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=aDjoksTpXOP](https://openreview.net/forum?id=aDjoksTpXOP)

**Abstract**:

Deep networks are often considered to be more expressive than shallow ones in terms of approximation. Indeed, certain functions can be approximated by deep networks provably more efficiently than by shallow ones, however, no tractable algorithms are known for learning such deep models. Separately, a recent line of work has shown that deep networks trained with gradient descent may behave like (tractable) kernel methods in a certain over-parameterized regime, where the kernel is determined by the architecture and initialization, and this paper focuses on approximation for such kernels. We show that for ReLU activations, the kernels derived from deep fully-connected networks have essentially the same approximation properties as their shallow two-layer counterpart, namely the same eigenvalue decay for the corresponding integral operator. This highlights the limitations of the kernel framework for understanding the benefits of such deep architectures. Our main theoretical result relies on characterizing such eigenvalue decays through differentiability properties of the kernel function, which also easily applies to the study of other kernels defined on the sphere.

----

## [271] Graph Coarsening with Neural Networks

**Authors**: *Chen Cai, Dingkang Wang, Yusu Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uxpzitPEooJ](https://openreview.net/forum?id=uxpzitPEooJ)

**Abstract**:

As large scale-graphs become increasingly more prevalent, it poses significant computational challenges to process, extract and analyze large graph data. Graph coarsening is one popular technique to reduce the size of a graph while maintaining essential properties. Despite rich graph coarsening literature, there is only limited exploration of data-driven method in the field. In this work, we leverage the recent progress of deep learning on graphs for graph coarsening. We first propose a framework for measuring the quality of coarsening algorithm and show that depending on the goal, we need to carefully choose the Laplace operator on the coarse graph and associated projection/lift operators. Motivated by the observation that the current choice of edge weight for the coarse graph may be sub-optimal, we parametrize the weight assignment map with graph neural networks and train it to improve the coarsening quality in an unsupervised way. Through extensive experiments on both synthetic and real networks, we demonstrate that our method significantly improves common graph coarsening methods under various metrics, reduction ratios, graph sizes, and graph types. It generalizes to graphs of larger size (more than $25\times$ of training graphs), adaptive to different losses (both differentiable and non-differentiable), and scales to much larger graphs than previous work.

----

## [272] Early Stopping in Deep Networks: Double Descent and How to Eliminate it

**Authors**: *Reinhard Heckel, Fatih Furkan Yilmaz*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tlV90jvZbw](https://openreview.net/forum?id=tlV90jvZbw)

**Abstract**:

Over-parameterized models, such as large deep networks, often exhibit a double descent phenomenon, whereas a function of model size, error first decreases, increases, and decreases at last. This intriguing double descent behavior also occurs as a function of training epochs and has been conjectured to arise because training epochs control the model complexity. In this paper, we show that such epoch-wise double descent occurs for a different reason: It is caused by a superposition of two or more bias-variance tradeoffs that arise because different parts of the network are learned at different epochs, and mitigating this by proper scaling of stepsizes can significantly improve the early stopping performance. We show this analytically for i) linear regression, where differently scaled features give rise to a superposition of bias-variance tradeoffs, and for ii) a wide two-layer neural network, where the first and second layers govern bias-variance tradeoffs. Inspired by this theory, we study two standard convolutional networks empirically and show that eliminating epoch-wise double descent through adjusting stepsizes of different layers improves the early stopping performance.

----

## [273] Efficient Inference of Flexible Interaction in Spiking-neuron Networks

**Authors**: *Feng Zhou, Yixuan Zhang, Jun Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=aGfU_xziEX8](https://openreview.net/forum?id=aGfU_xziEX8)

**Abstract**:

Hawkes process provides an effective statistical framework for analyzing the time-dependent interaction of neuronal spiking activities. Although utilized in many real applications, the classic Hawkes process is incapable of modelling inhibitory interactions among neurons. Instead, the nonlinear Hawkes process allows for a more flexible influence pattern with excitatory or inhibitory interactions. In this paper, three sets of auxiliary latent variables (Polya-Gamma variables, latent marked Poisson processes and sparsity variables) are augmented to make functional connection weights in a Gaussian form, which allows for a simple iterative algorithm with analytical updates. As a result, an efficient expectation-maximization (EM) algorithm is derived to obtain the maximum a posteriori (MAP) estimate. We demonstrate the accuracy and efficiency performance of our algorithm on synthetic and real data. For real neural recordings, we show our algorithm can estimate the temporal dynamics of interaction and reveal the interpretable functional connectivity underlying neural spike trains.

----

## [274] DICE: Diversity in Deep Ensembles via Conditional Redundancy Adversarial Estimation

**Authors**: *Alexandre Ramé, Matthieu Cord*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=R2ZlTVPx0Gk](https://openreview.net/forum?id=R2ZlTVPx0Gk)

**Abstract**:

Deep ensembles perform better than a single network thanks to the diversity among their members. Recent approaches regularize predictions to increase diversity; however, they also drastically decrease individual members’ performances. In this paper, we argue that learning strategies for deep ensembles need to tackle the trade-off between ensemble diversity and individual accuracies. Motivated by arguments from information theory and leveraging recent advances in neural estimation of conditional mutual information, we introduce a novel training criterion called DICE: it increases diversity by reducing spurious correlations among features. The main idea is that features extracted from pairs of members should only share information useful for target class prediction without being conditionally redundant. Therefore, besides the classification loss with information bottleneck, we adversarially prevent features from being conditionally predictable from each other. We manage to reduce simultaneous errors while protecting class information. We obtain state-of-the-art accuracy results on CIFAR-10/100: for example, an ensemble of 5 networks trained with DICE matches an ensemble of 7 networks trained independently. We further analyze the consequences on calibration, uncertainty estimation, out-of-distribution detection and online co-distillation.

----

## [275] Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks

**Authors**: *Yanbang Wang, Yen-Yu Chang, Yunyu Liu, Jure Leskovec, Pan Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KYPz4YsCPj](https://openreview.net/forum?id=KYPz4YsCPj)

**Abstract**:

Temporal networks serve as abstractions of many real-world dynamic systems. These networks typically evolve according to certain laws, such as the law of triadic closure, which is universal in social networks. Inductive representation learning of temporal networks should be able to capture such laws and further be applied to systems that follow the same laws but have not been unseen during the training stage. Previous works in this area depend on either network node identities or rich edge attributes and typically fail to extract these laws. Here, we propose {\em Causal Anonymous Walks (CAWs)} to inductively represent a temporal network. CAWs are extracted by temporal random walks and work as automatic retrieval of temporal network motifs to represent network dynamics while avoiding the time-consuming selection and counting of those motifs. CAWs adopt a novel anonymization strategy that replaces node identities with the hitting counts of the nodes based on a set of sampled walks to keep the method inductive, and simultaneously establish the correlation between motifs. We further propose a neural-network model CAW-N to encode CAWs, and pair it with a CAW sampling strategy with constant memory and time cost to support online training and inference. CAW-N is evaluated to predict links over 6 real temporal networks and uniformly outperforms previous SOTA methods by averaged 15\% AUC gain in the inductive setting. CAW-N also outperforms previous methods in 5 out of the 6 networks in the transductive setting.

----

## [276] FairBatch: Batch Selection for Model Fairness

**Authors**: *Yuji Roh, Kangwook Lee, Steven Euijong Whang, Changho Suh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YNnpaAKeCfx](https://openreview.net/forum?id=YNnpaAKeCfx)

**Abstract**:

Training a fair machine learning model is essential to prevent demographic disparity. Existing techniques for improving model fairness require broad changes in either data preprocessing or model training, rendering themselves difficult-to-adopt for potentially already complex machine learning systems. We address this problem via the lens of bilevel optimization. While keeping the standard training algorithm as an inner optimizer, we incorporate an outer optimizer so as to equip the inner problem with an additional functionality: Adaptively selecting minibatch sizes for the purpose of improving model fairness. Our batch selection algorithm, which we call FairBatch, implements this optimization and supports prominent fairness measures: equal opportunity, equalized odds, and demographic parity. FairBatch comes with a significant implementation benefit -- it does not require any modification to data preprocessing or model training. For instance, a single-line change of PyTorch code for replacing batch selection part of model training suffices to employ FairBatch. Our experiments conducted both on synthetic and benchmark real data demonstrate that FairBatch can provide such functionalities while achieving comparable (or even greater) performances against the state of the arts.  Furthermore, FairBatch can readily improve fairness of any pre-trained model simply via fine-tuning. It is also compatible with existing batch selection techniques intended for different purposes, such as faster convergence, thus gracefully achieving multiple purposes.

----

## [277] Representation Balancing Offline Model-based Reinforcement Learning

**Authors**: *Byung-Jun Lee, Jongmin Lee, Kee-Eung Kim*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QpNz8r_Ri2Y](https://openreview.net/forum?id=QpNz8r_Ri2Y)

**Abstract**:

One of the main challenges in offline and off-policy reinforcement learning is to cope with the distribution shift that arises from the mismatch between the target policy and the data collection policy. In this paper, we focus on a model-based approach, particularly on learning the representation for a robust model of the environment under the distribution shift, which has been first studied by Representation Balancing MDP (RepBM). Although this prior work has shown promising results, there are a number of shortcomings that still hinder its applicability to practical tasks. In particular, we address the curse of horizon exhibited by RepBM, rejecting most of the pre-collected data in long-term tasks. We present a new objective for model learning motivated by recent advances in the estimation of stationary distribution corrections. This effectively overcomes the aforementioned limitation of RepBM, as well as naturally extending to continuous action spaces and stochastic policies. We also present an offline model-based policy optimization using this new objective, yielding the state-of-the-art performance in a representative set of benchmark offline RL tasks.

----

## [278] Accelerating Convergence of Replica Exchange Stochastic Gradient MCMC via Variance Reduction

**Authors**: *Wei Deng, Qi Feng, Georgios Karagiannis, Guang Lin, Faming Liang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=iOnhIy-a-0n](https://openreview.net/forum?id=iOnhIy-a-0n)

**Abstract**:

Replica exchange stochastic gradient Langevin dynamics (reSGLD) has shown promise in accelerating the convergence in non-convex learning; however, an excessively large correction for avoiding biases from noisy energy estimators has limited the potential of the acceleration. To address this issue, we study the variance reduction for noisy energy estimators, which promotes much more effective swaps. Theoretically, we provide a non-asymptotic analysis on the exponential convergence for the underlying continuous-time Markov jump process; moreover, we consider a generalized Girsanov theorem which includes the change of Poisson measure to overcome the crude discretization based on the Gr\"{o}wall's inequality and yields a much tighter error in the 2-Wasserstein ($\mathcal{W}_2$) distance. Numerically, we conduct extensive experiments and obtain state-of-the-art results in optimization and uncertainty estimates for synthetic experiments and image data.

----

## [279] The Importance of Pessimism in Fixed-Dataset Policy Optimization

**Authors**: *Jacob Buckman, Carles Gelada, Marc G. Bellemare*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=E3Ys6a1NTGT](https://openreview.net/forum?id=E3Ys6a1NTGT)

**Abstract**:

We study worst-case guarantees on the expected return of fixed-dataset policy optimization algorithms. Our core contribution is a unified conceptual and mathematical framework for the study of algorithms in this regime. This analysis reveals that for naive approaches, the possibility of erroneous value overestimation leads to a difficult-to-satisfy requirement: in order to guarantee that we select a policy which is near-optimal, we may need the dataset to be informative of the value of every policy. To avoid this, algorithms can follow the pessimism principle, which states that we should choose the policy which acts optimally in the worst possible world. We show why pessimistic algorithms can achieve good performance even when the dataset is not informative of every policy, and derive families of algorithms which follow this principle. These theoretical findings are validated by experiments on a tabular gridworld, and deep learning experiments on four MinAtar environments.

----

## [280] Interpreting Knowledge Graph Relation Representation from Word Embeddings

**Authors**: *Carl Allen, Ivana Balazevic, Timothy M. Hospedales*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gLWj29369lW](https://openreview.net/forum?id=gLWj29369lW)

**Abstract**:

Many models learn representations of knowledge graph data by exploiting its low-rank latent structure, encoding known relations between entities and enabling unknown facts to be inferred. To predict whether a relation holds between entities, embeddings are typically compared in the latent space following a relation-specific mapping. Whilst their predictive performance has steadily improved, how such models capture the underlying latent structure of semantic information remains unexplained. Building on recent theoretical understanding of word embeddings, we categorise knowledge graph relations into three types and for each derive explicit requirements of their representations. We show that empirical properties of relation representations and the relative performance of leading knowledge graph representation methods are justified by our analysis.

----

## [281] Hopfield Networks is All You Need

**Authors**: *Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Lukas Gruber, Markus Holzleitner, Thomas Adler, David P. Kreil, Michael K. Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tL89RnzIiCd](https://openreview.net/forum?id=tL89RnzIiCd)

**Abstract**:

We introduce a modern Hopfield network with continuous states and a corresponding update rule. The new Hopfield network can store exponentially (with the dimension of the associative space) many patterns, retrieves the pattern with one update, and has exponentially small retrieval errors. It has three types of energy minima (fixed points of the update): (1) global fixed point averaging over all patterns, (2) metastable states averaging over a subset of patterns, and (3) fixed points which store a single pattern. The new update rule is equivalent to the attention mechanism used in transformers. This equivalence enables a characterization of the heads of transformer models. These heads perform in the first layers preferably global averaging and in higher layers partial averaging via metastable states. The new modern Hopfield network can be integrated  into deep learning architectures as layers to allow the storage of and access to raw input data, intermediate results, or learned prototypes.
These Hopfield layers enable new ways of deep learning, beyond fully-connected, convolutional, or recurrent networks, and provide pooling, memory, association, and attention mechanisms. We demonstrate the broad applicability of the Hopfield layers
across various domains. Hopfield layers improved state-of-the-art on three out of four considered multiple instance learning problems as well as on immune repertoire classification with several hundreds of thousands of instances. On the UCI benchmark collections of small classification tasks, where deep learning methods typically struggle, Hopfield layers yielded a new state-of-the-art when compared to different machine learning methods. Finally, Hopfield layers achieved state-of-the-art on two drug design datasets. The implementation is available at: \url{https://github.com/ml-jku/hopfield-layers}

----

## [282] Uncertainty Estimation and Calibration with Finite-State Probabilistic RNNs

**Authors**: *Cheng Wang, Carolin Lawrence, Mathias Niepert*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9EKHN1jOlA](https://openreview.net/forum?id=9EKHN1jOlA)

**Abstract**:

Uncertainty quantification is crucial for building reliable and trustable machine learning systems. We propose to estimate uncertainty in recurrent neural networks (RNNs) via stochastic discrete state transitions over recurrent timesteps. The uncertainty of the model can be quantified by running a prediction several times, each time sampling from the recurrent state transition distribution, leading to potentially different results if the model is uncertain. Alongside uncertainty quantification, our proposed method offers several advantages in different settings. The proposed method can (1) learn deterministic and probabilistic automata from data, (2) learn well-calibrated models on real-world classification tasks, (3) improve the performance of out-of-distribution detection, and (4) control the exploration-exploitation trade-off in reinforcement learning. An implementation is available.

----

## [283] Understanding the failure modes of out-of-distribution generalization

**Authors**: *Vaishnavh Nagarajan, Anders Andreassen, Behnam Neyshabur*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=fSTD6NFIW_b](https://openreview.net/forum?id=fSTD6NFIW_b)

**Abstract**:

Empirical studies suggest that machine learning models often rely on features, such as the background, that may be spuriously correlated with the label only during training time, resulting in poor accuracy during test-time. In this work, we identify the fundamental factors that give rise to this behavior, by explaining why models fail this way even in easy-to-learn tasks where one would expect these models to succeed. In particular, through a theoretical study of gradient-descent-trained linear classifiers on some easy-to-learn tasks, we uncover two complementary failure modes. These modes arise from how spurious correlations induce two kinds of skews in the data: one geometric in nature and another, statistical. Finally, we construct natural modifications of image classification datasets to understand when these failure modes can arise in practice. We also design experiments to isolate the two failure modes when training modern neural networks on these datasets.

----

## [284] Generative Language-Grounded Policy in Vision-and-Language Navigation with Bayes' Rule

**Authors**: *Shuhei Kurita, Kyunghyun Cho*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=45uOPa46Kh](https://openreview.net/forum?id=45uOPa46Kh)

**Abstract**:

Vision-and-language navigation (VLN) is a task in which an agent is embodied in a realistic 3D environment and follows an instruction to reach the goal node. While most of the previous studies have built and investigated a discriminative approach, we notice that there are in fact two possible approaches to building such a VLN agent: discriminative and generative. In this paper, we design and investigate a generative language-grounded policy which uses a language model to compute the distribution over all possible instructions i.e. all possible sequences of vocabulary tokens given action and the transition history. In experiments, we show that the proposed generative approach outperforms the discriminative approach in the Room-2-Room (R2R) and Room-4-Room (R4R) datasets, especially in the unseen environments. We further show that the combination of the generative and discriminative policies achieves close to the state-of-the art results in the R2R dataset, demonstrating that the generative and discriminative policies capture the different aspects of VLN.

----

## [285] Emergent Road Rules In Multi-Agent Driving Environments

**Authors**: *Avik Pal, Jonah Philion, Yuan-Hong Liao, Sanja Fidler*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=d8Q1mt2Ghw](https://openreview.net/forum?id=d8Q1mt2Ghw)

**Abstract**:

For autonomous vehicles to safely share the road with human drivers, autonomous vehicles must abide by specific "road rules" that human drivers have agreed to follow. "Road rules" include rules that drivers are required to follow by law – such as the requirement that vehicles stop at red lights – as well as more subtle social rules – such as the implicit designation of fast lanes on the highway. In this paper, we provide empirical evidence that suggests that – instead of hard-coding road rules into self-driving algorithms – a scalable alternative may be to design multi-agent environments in which road rules emerge as optimal solutions to the problem of maximizing traffic flow.  We analyze what ingredients in driving environments cause the emergence of these road rules and find that two crucial factors are noisy perception and agents’ spatial density.  We provide qualitative and quantitative evidence of the emergence of seven social driving behaviors, ranging from obeying traffic signals to following lanes, all of which emerge from training agents to drive quickly to destinations without colliding. Our results add empirical support for the social road rules that countries worldwide have agreed on for safe, efficient driving.

----

## [286] Wasserstein-2 Generative Networks

**Authors**: *Alexander Korotin, Vage Egiazarian, Arip Asadulaev, Alexander Safin, Evgeny Burnaev*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bEoxzW_EXsa](https://openreview.net/forum?id=bEoxzW_EXsa)

**Abstract**:

We propose a novel end-to-end non-minimax algorithm for training optimal transport mappings for the quadratic cost (Wasserstein-2 distance). The algorithm uses input convex neural networks and a cycle-consistency regularization to approximate Wasserstein-2 distance. In contrast to popular entropic and quadratic regularizers, cycle-consistency does not introduce bias and scales well to high dimensions. From the theoretical side, we estimate the properties of the generative mapping fitted by our algorithm. From the practical side, we evaluate our algorithm on a wide range of tasks: image-to-image color transfer, latent space optimal transport, image-to-image style transfer, and domain adaptation.

----

## [287] Vulnerability-Aware Poisoning Mechanism for Online RL with Unknown Dynamics

**Authors**: *Yanchao Sun, Da Huo, Furong Huang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9r30XCjf5Dt](https://openreview.net/forum?id=9r30XCjf5Dt)

**Abstract**:

Poisoning attacks on Reinforcement Learning (RL) systems could take advantage of RL algorithm’s vulnerabilities and cause failure of the learning. However, prior works on poisoning RL usually either unrealistically assume the attacker knows the underlying Markov Decision Process (MDP), or directly apply the poisoning methods in supervised learning to RL. In this work, we build a generic poisoning framework for online RL via a comprehensive investigation of heterogeneous poisoning models in RL. Without any prior knowledge of the MDP, we propose a strategic poisoning algorithm called Vulnerability-Aware Adversarial Critic Poison (VA2C-P), which works for on-policy deep RL agents, closing the gap that no poisoning method exists for policy-based RL agents. VA2C-P uses a novel metric, stability radius in RL, that measures the vulnerability of RL algorithms. Experiments on multiple deep RL agents and multiple environments show that our poisoning algorithm successfully prevents agents from learning a good policy or teaches the agents to converge to a target policy, with a limited attacking budget.

----

## [288] Tomographic Auto-Encoder: Unsupervised Bayesian Recovery of Corrupted Data

**Authors**: *Francesco Tonolini, Pablo Garcia Moreno, Andreas C. Damianou, Roderick Murray-Smith*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YtMG5ex0ou](https://openreview.net/forum?id=YtMG5ex0ou)

**Abstract**:

We propose a new probabilistic method for unsupervised recovery of corrupted data. Given a large ensemble of degraded samples, our method recovers accurate posteriors of clean values, allowing the exploration of the manifold of possible reconstructed data and hence characterising the underlying uncertainty. In this set-ting, direct application of classical variational methods often gives rise to collapsed densities that do not adequately explore the solution space.  Instead, we derive our novel reduced entropy condition approximate inference method that results in rich posteriors.  We test our model in a data recovery task under the common setting of missing values and noise, demonstrating superior performance to existing variational methods for imputation and de-noising with different real data sets. We further show higher classification accuracy after imputation, proving the advantage of propagating uncertainty to downstream tasks with our model.

----

## [289] Monotonic Kronecker-Factored Lattice

**Authors**: *William Taylor Bakst, Nobuyuki Morioka, Erez Louidor*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0pxiMpCyBtr](https://openreview.net/forum?id=0pxiMpCyBtr)

**Abstract**:

It is computationally challenging to learn flexible monotonic functions that guarantee model behavior and provide interpretability beyond a few input features, and in a time where minimizing resource use is increasingly important, we must be able to learn such models that are still efficient. In this paper we show how to effectively and efficiently learn such functions using Kronecker-Factored Lattice ($\mathrm{KFL}$), an efficient reparameterization of flexible monotonic lattice regression via Kronecker product. Both computational and storage costs scale linearly in the number of input features, which is a significant improvement over existing methods that grow exponentially. We also show that we can still properly enforce monotonicity and other shape constraints. The $\mathrm{KFL}$ function class consists of products of piecewise-linear functions, and the size of the function class can be further increased through ensembling. We prove that the function class of an ensemble of $M$ base $\mathrm{KFL}$ models strictly increases as $M$ increases up to a certain threshold. Beyond this threshold, every multilinear interpolated lattice function can be expressed. Our experimental results demonstrate that $\mathrm{KFL}$ trains faster with fewer parameters while still achieving accuracy and evaluation speeds comparable to or better than the baseline methods and preserving monotonicity guarantees on the learned model.

----

## [290] LEAF: A Learnable Frontend for Audio Classification

**Authors**: *Neil Zeghidour, Olivier Teboul, Félix de Chaumont Quitry, Marco Tagliasacchi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jM76BCb6F9m](https://openreview.net/forum?id=jM76BCb6F9m)

**Abstract**:

Mel-filterbanks are fixed, engineered audio features which emulate human perception and have been used through the history of audio understanding up to today. However, their undeniable qualities are counterbalanced by the fundamental limitations of handmade representations. In this work we show that we can train a single learnable frontend that outperforms mel-filterbanks on a wide range of audio signals, including speech, music, audio events and animal sounds, providing a general-purpose learned frontend for audio classification. To do so, we introduce a new principled, lightweight, fully learnable architecture that can be used as a drop-in replacement of mel-filterbanks. Our system learns all operations of audio features extraction, from filtering to pooling, compression and normalization, and can be integrated into any neural network at a negligible parameter cost. We perform multi-task training on eight diverse audio classification tasks, and show consistent improvements of our model over mel-filterbanks and previous learnable alternatives. Moreover, our system outperforms the current state-of-the-art learnable frontend on Audioset, with orders of magnitude fewer parameters.

----

## [291] Federated Learning via Posterior Averaging: A New Perspective and Practical Algorithms

**Authors**: *Maruan Al-Shedivat, Jennifer Gillenwater, Eric P. Xing, Afshin Rostamizadeh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=GFsU8a0sGB](https://openreview.net/forum?id=GFsU8a0sGB)

**Abstract**:

Federated learning is typically approached as an optimization problem, where the goal is to minimize a global loss function by distributing computation across client devices that possess local data and specify different parts of the global objective.  We present an alternative perspective and formulate federated learning as a posterior inference problem, where the goal is to infer a global posterior distribution by having client devices each infer the posterior of their local data.  While exact inference is often intractable, this perspective provides a principled way to search for global optima in federated settings.  Further, starting with the analysis of federated quadratic objectives, we develop a computation- and communication-efficient approximate posterior inference algorithm—federated posterior averaging (FedPA).  Our algorithm uses MCMC for approximate inference of local posteriors on the clients and efficiently communicates their statistics to the server, where the latter uses them to refine a global estimate of the posterior mode.  Finally, we show that FedPA generalizes federated averaging (FedAvg), can similarly benefit from adaptive optimizers, and yields state-of-the-art results on four realistic and challenging benchmarks, converging faster, to better optima.

----

## [292] Rank the Episodes: A Simple Approach for Exploration in Procedurally-Generated Environments

**Authors**: *Daochen Zha, Wenye Ma, Lei Yuan, Xia Hu, Ji Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MtEE0CktZht](https://openreview.net/forum?id=MtEE0CktZht)

**Abstract**:

Exploration under sparse reward is a long-standing challenge of model-free reinforcement learning. The state-of-the-art methods address this challenge by introducing intrinsic rewards to encourage exploration in novel states or uncertain environment dynamics. Unfortunately, methods based on intrinsic rewards often fall short in procedurally-generated environments, where a different environment is generated in each episode so that the agent is not likely to visit the same state more than once. Motivated by how humans distinguish good exploration behaviors by looking into the entire episode, we introduce RAPID, a simple yet effective episode-level exploration method for procedurally-generated environments. RAPID regards each episode as a whole and gives an episodic exploration score from both per-episode and long-term views. Those highly scored episodes are treated as good exploration behaviors and are stored in a small ranking buffer. The agent then imitates the episodes in the buffer to reproduce the past good exploration behaviors. We demonstrate our method on several procedurally-generated MiniGrid environments, a first-person-view 3D Maze navigation task from MiniWorld, and several sparse MuJoCo tasks. The results show that RAPID significantly outperforms the state-of-the-art intrinsic reward strategies in terms of sample efficiency and final performance. The code is available at https://github.com/daochenzha/rapid

----

## [293] Partitioned Learned Bloom Filters

**Authors**: *Kapil Vaidya, Eric Knorr, Michael Mitzenmacher, Tim Kraska*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6BRLOfrMhW](https://openreview.net/forum?id=6BRLOfrMhW)

**Abstract**:

Bloom filters are space-efficient probabilistic data structures that are used to test whether an element is a member of a set, and may return false positives.  Recently, variations referred to as learned Bloom filters were developed that can provide improved performance in terms of the rate of false positives, by using a learned model for the represented set.  However, previous methods for learned Bloom filters do not take full advantage of the learned model.  Here we show how to frame the problem of optimal model utilization as an optimization problem, and using our framework derive algorithms that can achieve near-optimal performance in many cases.

----

## [294] Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval

**Authors**: *Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, Arnold Overwijk*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=zeFrfgyZln](https://openreview.net/forum?id=zeFrfgyZln)

**Abstract**:

Conducting text retrieval in a learned dense representation space has many intriguing advantages. Yet dense retrieval (DR) often underperforms word-based sparse retrieval. In this paper, we first theoretically show the bottleneck of dense retrieval is the domination of uninformative negatives sampled in mini-batch training, which yield diminishing gradient norms, large gradient variances, and slow convergence. We then propose Approximate nearest neighbor Negative Contrastive Learning (ANCE), which selects hard training negatives globally from the entire corpus. Our experiments demonstrate the effectiveness of ANCE on web search, question answering, and in a commercial search engine, showing ANCE dot-product retrieval nearly matches the accuracy of BERT-based cascade IR pipeline. We also empirically validate our theory that negative sampling with ANCE better approximates the oracle importance sampling procedure and improves learning convergence.

----

## [295] Auxiliary Task Update Decomposition: the Good, the Bad and the neutral

**Authors**: *Lucio M. Dery, Yann N. Dauphin, David Grangier*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=1GTma8HwlYp](https://openreview.net/forum?id=1GTma8HwlYp)

**Abstract**:

While deep learning has been very beneficial in data-rich settings, tasks with smaller training set
often resort to pre-training or multitask learning to leverage data from other tasks. In this case,
careful consideration is needed to select tasks and model parameterizations such that updates from
the auxiliary tasks actually help the primary task. We seek to alleviate this burden by formulating a model-agnostic framework that performs fine-grained manipulation of the auxiliary task gradients. We propose to decompose auxiliary updates into directions which help, damage or leave the primary task loss unchanged. This allows weighting the update directions 
differently depending on their impact on the problem of interest. We present a novel and efficient algorithm for that
purpose and show its advantage in practice. Our method leverages efficient automatic differentiation 
procedures and randomized singular value decomposition for scalability. We show that our framework is 
generic and encompasses some prior work as particular cases. Our approach consistently outperforms strong and widely used baselines when leveraging out-of-distribution data for Text and Image classification tasks.

----

## [296] SSD: A Unified Framework for Self-Supervised Outlier Detection

**Authors**: *Vikash Sehwag, Mung Chiang, Prateek Mittal*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=v5gjXpmR8J](https://openreview.net/forum?id=v5gjXpmR8J)

**Abstract**:

We ask the following question: what training information is required to design an effective outlier/out-of-distribution (OOD) detector, i.e., detecting samples that lie far away from training distribution? Since unlabeled data is easily accessible for many applications, the most compelling approach is to develop detectors based on only unlabeled in-distribution data. However, we observe that most existing detectors based on unlabeled data perform poorly, often equivalent to a random prediction. In contrast, existing state-of-the-art OOD detectors achieve impressive performance but require access to fine-grained data labels for supervised training. We propose SSD, an outlier detector based on only unlabeled in-distribution data. We use self-supervised representation learning followed by a Mahalanobis distance based detection in the feature space. We demonstrate that SSD outperforms most existing detectors based on unlabeled data by a large margin. Additionally, SSD even achieves performance on par, and sometimes even better, with supervised training based detectors.  Finally, we expand our detection framework with two key extensions. First, we formulate few-shot OOD detection, in which the detector has access to only one to five samples from each class of the targeted OOD dataset. Second, we extend our framework to incorporate training data labels, if available. We find that our novel detection framework based on SSD displays enhanced performance with these extensions, and achieves state-of-the-art performance. Our code is publicly available at https://github.com/inspire-group/SSD.

----

## [297] Ask Your Humans: Using Human Instructions to Improve Generalization in Reinforcement Learning

**Authors**: *Valerie Chen, Abhinav Gupta, Kenneth Marino*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Y87Ri-GNHYu](https://openreview.net/forum?id=Y87Ri-GNHYu)

**Abstract**:

Complex, multi-task problems have proven to be difficult to solve efficiently in a sparse-reward reinforcement learning setting. In order to be sample efficient, multi-task learning requires reuse and sharing of low-level policies. To facilitate the automatic decomposition of hierarchical tasks, we propose the use of step-by-step human demonstrations in the form of natural language instructions and action trajectories. We introduce a dataset of such demonstrations in a crafting-based grid world. Our model consists of a high-level language generator and low-level policy, conditioned on language. We find that human demonstrations help solve the most complex tasks. We also find that incorporating natural language allows the model to generalize to unseen tasks in a zero-shot setting and to learn quickly from a few demonstrations. Generalization is not only reflected in the actions of the agent, but also in the generated natural language instructions in unseen tasks. Our approach also gives our trained agent interpretable behaviors because it is able to generate a sequence of high-level descriptions of its actions.

----

## [298] Revisiting Few-sample BERT Fine-tuning

**Authors**: *Tianyi Zhang, Felix Wu, Arzoo Katiyar, Kilian Q. Weinberger, Yoav Artzi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=cO1IH43yUF](https://openreview.net/forum?id=cO1IH43yUF)

**Abstract**:

This paper is a study of fine-tuning of BERT contextual representations, with focus on commonly observed instabilities in few-sample scenarios. We identify several factors that cause this instability: the common use of a non-standard optimization method with biased gradient estimation; the limited applicability of significant parts of the BERT network for down-stream tasks; and the prevalent practice of using a pre-determined, and small number of training iterations. We empirically test the impact of these factors, and identify alternative practices that resolve the commonly observed instability of the process. In light of these observations, we re-visit recently proposed methods to improve few-sample fine-tuning with BERT and re-evaluate their effectiveness. Generally, we observe the impact of these methods diminishes significantly with our modified process.

----

## [299] Tilted Empirical Risk Minimization

**Authors**: *Tian Li, Ahmad Beirami, Maziar Sanjabi, Virginia Smith*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=K5YasWXZT3O](https://openreview.net/forum?id=K5YasWXZT3O)

**Abstract**:

Empirical risk minimization (ERM) is typically designed to perform well on the average loss, which can result in estimators that are sensitive to outliers, generalize poorly, or treat subgroups unfairly. While many methods aim to address these problems individually, in this work, we explore them through a unified framework---tilted empirical risk minimization (TERM). In particular, we show that it is possible to flexibly tune the impact of individual losses through a straightforward extension to ERM using a hyperparameter called the tilt. We provide several interpretations of the resulting framework: We show that TERM can increase or decrease the influence of outliers, respectively, to enable fairness or robustness; has variance-reduction properties that can benefit generalization; and can be viewed as a smooth approximation to a superquantile method. We develop batch and stochastic first-order optimization methods for solving TERM, and show that the problem can be efficiently solved relative to common alternatives. Finally, we demonstrate that TERM can be used for a multitude of applications, such as enforcing fairness between subgroups, mitigating the effect of outliers, and handling class imbalance. TERM is not only competitive with existing solutions tailored to these individual problems, but can also enable entirely new applications, such as simultaneously addressing outliers and promoting fairness.

----

## [300] Deep Neural Tangent Kernel and Laplace Kernel Have the Same RKHS

**Authors**: *Lin Chen, Sheng Xu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vK9WrZ0QYQ](https://openreview.net/forum?id=vK9WrZ0QYQ)

**Abstract**:

We prove that the reproducing kernel Hilbert spaces (RKHS) of a deep neural tangent kernel and the Laplace kernel include the same set of functions, when both kernels are restricted to the sphere $\mathbb{S}^{d-1}$. Additionally, we prove that the exponential power kernel with a smaller power (making the kernel less smooth) leads to a larger RKHS, when it is restricted to the sphere $\mathbb{S}^{d-1}$ and when it is defined on the entire $\mathbb{R}^d$.

----

## [301] On the Transfer of Disentangled Representations in Realistic Settings

**Authors**: *Andrea Dittadi, Frederik Träuble, Francesco Locatello, Manuel Wuthrich, Vaibhav Agrawal, Ole Winther, Stefan Bauer, Bernhard Schölkopf*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8VXvj1QNRl1](https://openreview.net/forum?id=8VXvj1QNRl1)

**Abstract**:

Learning meaningful representations that disentangle the underlying structure of the data generating process is considered to be of key importance in machine learning. While disentangled representations were found to be useful for diverse tasks such as abstract reasoning and fair classification, their scalability and real-world impact remain questionable.
We introduce a new high-resolution dataset with 1M simulated images and over 1,800 annotated real-world images of the same setup. In contrast to previous work, this new dataset exhibits correlations, a complex underlying structure, and allows to evaluate transfer to unseen simulated and real-world settings where the encoder i) remains in distribution or ii) is out of distribution.
We propose new architectures in order to scale disentangled representation learning to realistic high-resolution settings and conduct a large-scale empirical study of disentangled representations on this dataset. We observe that disentanglement is a good predictor for out-of-distribution (OOD) task performance.

----

## [302] Calibration tests beyond classification

**Authors**: *David Widmann, Fredrik Lindsten, Dave Zachariah*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-bxf89v3Nx](https://openreview.net/forum?id=-bxf89v3Nx)

**Abstract**:

Most supervised machine learning tasks are subject to irreducible prediction errors. Probabilistic predictive models address this limitation by providing probability distributions that represent a belief over plausible targets, rather than point estimates. Such models can be a valuable tool in decision-making under uncertainty, provided that the model output is meaningful and interpretable. Calibrated models guarantee that the probabilistic predictions are neither over- nor under-confident. In the machine learning literature, different measures and statistical tests have been proposed and studied for evaluating the calibration of classification models. For regression problems, however, research has been focused on a weaker condition of calibration based on predicted quantiles for real-valued targets. In this paper, we propose the first framework that unifies calibration evaluation and tests for general probabilistic predictive models. It applies to any such model, including classification and regression models of arbitrary dimension. Furthermore, the framework generalizes existing measures and provides a more intuitive reformulation of a recently proposed framework for calibration in multi-class classification. In particular, we reformulate and generalize the kernel calibration error, its estimators, and hypothesis tests using scalar-valued kernels, and evaluate the calibration of real-valued regression
problems.

----

## [303] Overparameterisation and worst-case generalisation: friend or foe?

**Authors**: *Aditya Krishna Menon, Ankit Singh Rawat, Sanjiv Kumar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jphnJNOwe36](https://openreview.net/forum?id=jphnJNOwe36)

**Abstract**:

Overparameterised neural networks have demonstrated the remarkable ability to perfectly fit training samples, while still generalising to unseen test samples. However, several recent works have revealed that such models' good average performance does not always translate to good worst-case performance: in particular, they may perform poorly on subgroups that are under-represented in the training set. In this paper, we show that in certain settings, overparameterised models' performance on under-represented subgroups may be improved via post-hoc processing. Specifically, such models' bias can be restricted to their classification layers, and manifest as structured prediction shifts for rare subgroups. We detail two post-hoc correction techniques to mitigate this bias, which operate purely on the outputs of standard model training. We empirically verify that with such post-hoc correction, overparameterisation can improve average and worst-case performance.

----

## [304] You Only Need Adversarial Supervision for Semantic Image Synthesis

**Authors**: *Edgar Schönfeld, Vadim Sushko, Dan Zhang, Juergen Gall, Bernt Schiele, Anna Khoreva*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=yvQKLaqNE6M](https://openreview.net/forum?id=yvQKLaqNE6M)

**Abstract**:

Despite their recent successes, GAN models for semantic image synthesis still suffer from poor image quality when trained with only adversarial supervision. Historically, additionally employing the VGG-based perceptual loss has helped to overcome this issue, significantly improving the synthesis quality, but at the same time limiting the progress of GAN models for semantic image synthesis. In this work, we propose a novel, simplified GAN model, which needs only adversarial supervision to achieve high quality results. We re-design the discriminator as a semantic segmentation network, directly using the given semantic label maps as the ground truth for training. By providing stronger supervision to the discriminator as well as to the generator through spatially- and semantically-aware discriminator feedback, we are able to synthesize images of higher fidelity with better alignment to their input label maps, making the use of the perceptual loss superfluous. Moreover, we enable high-quality multi-modal image synthesis through global and local sampling of a 3D noise tensor injected into the generator, which allows complete or partial image change. We show that images synthesized by our model are more diverse and follow the color and texture distributions of real images more closely. We achieve an average improvement of $6$ FID and $5$ mIoU points over the state of the art across different datasets using only adversarial supervision.

----

## [305] Learning to Recombine and Resample Data For Compositional Generalization

**Authors**: *Ekin Akyürek, Afra Feyza Akyürek, Jacob Andreas*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PS3IMnScugk](https://openreview.net/forum?id=PS3IMnScugk)

**Abstract**:

Flexible neural sequence models outperform grammar- and automaton-based counterparts on a variety of tasks. However, neural models perform poorly in settings requiring compositional generalization beyond the training data—particularly to rare or unseen subsequences. Past work has found symbolic scaffolding (e.g. grammars or automata) essential in these settings. We describe R&R, a learned data augmentation scheme that enables a large category of compositional generalizations without appeal to latent symbolic structure. R&R has two components: recombination of original training examples via a prototype-based generative model and resampling of generated examples to encourage extrapolation. Training an ordinary neural sequence model on a dataset augmented with recombined and resampled examples significantly improves generalization in two language processing problems—instruction following (SCAN) and morphological analysis (SIGMORPHON 2018)—where R&R enables learning of new constructions and tenses from as few as eight initial examples.

----

## [306] A Critique of Self-Expressive Deep Subspace Clustering

**Authors**: *Benjamin David Haeffele, Chong You, René Vidal*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=FOyuZ26emy](https://openreview.net/forum?id=FOyuZ26emy)

**Abstract**:

Subspace clustering is an unsupervised clustering technique designed to cluster data that is supported on a union of linear subspaces, with each subspace defining a cluster with dimension lower than the ambient space. Many existing formulations for this problem are based on exploiting the self-expressive property of linear subspaces, where any point within a subspace can be represented as linear combination of other points within the subspace. To extend this approach to data supported on a union of non-linear manifolds, numerous studies have proposed learning an embedding of the original data using  a neural network which is regularized by a self-expressive loss function on the data in the embedded space to encourage a union of linear subspaces prior on the data in the embedded space. Here we show that there are a number of potential flaws with this approach which have not been adequately addressed in prior work. In particular, we show the model formulation is often ill-posed in that it can lead to a degenerate embedding of the data, which need not correspond to a union of subspaces at all and is poorly suited for clustering. We validate our theoretical results experimentally and also repeat prior experiments reported in the literature, where we conclude that a significant portion of the previously claimed performance benefits can be attributed to an ad-hoc post processing step rather than the deep subspace clustering model.

----

## [307] INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving

**Authors**: *Yuhuai Wu, Albert Q. Jiang, Jimmy Ba, Roger Baker Grosse*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=O6LPudowNQm](https://openreview.net/forum?id=O6LPudowNQm)

**Abstract**:

In learning-assisted theorem proving, one of the most critical challenges is to generalize to theorems unlike those seen at training time. In this paper, we introduce INT, an INequality Theorem proving benchmark designed to test agents’ generalization ability. INT is based on a theorem generator, which provides theoretically infinite data and allows us to measure 6 different types of generalization, each reflecting a distinct challenge, characteristic of automated theorem proving. In addition, provides a fast theorem proving environment with sequence-based and graph-based interfaces, conducive to performing learning-based research. We introduce base-lines with architectures including transformers and graph neural networks (GNNs)for INT. Using INT, we find that transformer-based agents achieve stronger test performance for most of the generalization tasks, despite having much larger out-of-distribution generalization gaps than GNNs. We further find that the addition of Monte Carlo Tree Search (MCTS) at test time helps to prove new theorems.

----

## [308] Improved Estimation of Concentration Under ℓp-Norm Distance Metrics Using Half Spaces

**Authors**: *Jack Prescott, Xiao Zhang, David E. Evans*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=BUlyHkzjgmA](https://openreview.net/forum?id=BUlyHkzjgmA)

**Abstract**:

Concentration of measure has been argued to be the fundamental cause of adversarial vulnerability. Mahloujifar et al. (2019) presented an empirical way to measure the concentration of a data distribution using samples, and employed it to find lower bounds on intrinsic robustness for several benchmark datasets. However, it remains unclear whether these lower bounds are tight enough to provide a useful approximation for the intrinsic robustness of a dataset. To gain a deeper understanding of the concentration of measure phenomenon, we first extend the Gaussian Isoperimetric Inequality to non-spherical Gaussian measures and arbitrary $\ell_p$-norms ($p \geq 2$). We leverage these theoretical insights to design a method that uses half-spaces to estimate the concentration of any empirical dataset under $\ell_p$-norm distance metrics. Our proposed algorithm is more efficient than Mahloujifar et al. (2019)'s, and experiments on synthetic datasets and image benchmarks demonstrate that it is able to find much tighter intrinsic robustness bounds. These tighter estimates provide further evidence that rules out intrinsic dataset concentration as a possible explanation for the adversarial vulnerability of state-of-the-art classifiers.

----

## [309] Adaptive Federated Optimization

**Authors**: *Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecný, Sanjiv Kumar, Hugh Brendan McMahan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LkFG3lB13U5](https://openreview.net/forum?id=LkFG3lB13U5)

**Abstract**:

Federated learning is a distributed machine learning paradigm in which a large number of clients coordinate with a central server to learn a model without sharing their own training data. Standard federated optimization methods such as Federated Averaging (FedAvg) are often difficult to tune and exhibit unfavorable convergence behavior. In non-federated settings, adaptive optimization methods have had notable success in combating such issues. In this work, we propose federated versions of adaptive optimizers, including Adagrad, Adam, and  Yogi, and analyze their convergence in the presence of heterogeneous data for general non-convex settings. Our results highlight the interplay between client heterogeneity and communication efficiency. We also perform extensive experiments on these methods and show that the use of adaptive optimizers can significantly improve the performance of federated learning.

----

## [310] On the Dynamics of Training Attention Models

**Authors**: *Haoye Lu, Yongyi Mao, Amiya Nayak*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=1OCTOShAmqB](https://openreview.net/forum?id=1OCTOShAmqB)

**Abstract**:

The attention mechanism has been widely used in deep neural networks as a model component. By now, it has become a critical building block in many state-of-the-art natural language models. Despite its great success established empirically, the working mechanism of attention has not been investigated at a sufficient theoretical depth to date. In this paper, we set up a simple text classification task and study the dynamics of training a simple attention-based classification model using gradient descent. In this setting, we show that, for the discriminative words that the model should attend to, a persisting identity exists relating its embedding and the inner product of its key and the query. This allows us to prove that training must converge to attending to the discriminative words when the attention output is classified by a linear classifier. Experiments are performed, which validate our theoretical analysis and provide further insights.

----

## [311] Linear Convergent Decentralized Optimization with Compression

**Authors**: *Xiaorui Liu, Yao Li, Rongrong Wang, Jiliang Tang, Ming Yan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=84gjULz1t5](https://openreview.net/forum?id=84gjULz1t5)

**Abstract**:

Communication compression has become a key strategy to speed up distributed optimization. However, existing decentralized algorithms with compression mainly focus on compressing DGD-type algorithms. They are unsatisfactory in terms of convergence rate, stability, and the capability to handle heterogeneous data. Motivated by primal-dual algorithms, this paper proposes the first \underline{L}in\underline{EA}r convergent \underline{D}ecentralized algorithm with compression, LEAD. Our theory describes the coupled dynamics of the inexact primal and dual update as well as compression error, and we provide the first consensus error bound in such settings without assuming bounded gradients. Experiments on convex problems validate our theoretical analysis, and empirical study on deep neural nets shows that LEAD is applicable to non-convex problems.

----

## [312] Efficient Transformers in Reinforcement Learning using Actor-Learner Distillation

**Authors**: *Emilio Parisotto, Ruslan Salakhutdinov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uR9LaO_QxF](https://openreview.net/forum?id=uR9LaO_QxF)

**Abstract**:

Many real-world applications such as robotics provide hard constraints on power and compute that limit the viable model complexity of Reinforcement Learning (RL) agents. Similarly, in many distributed RL settings, acting is done on un-accelerated hardware such as CPUs, which likewise restricts model size to prevent intractable experiment run times. These "actor-latency" constrained settings present a major obstruction to the scaling up of model complexity that has recently been extremely successful in supervised learning. To be able to utilize large model capacity while still operating within the limits imposed by the system during acting, we develop an "Actor-Learner Distillation" (ALD) procedure that leverages a continual form of distillation that transfers learning progress from a large capacity learner model to a small capacity actor model. As a case study, we develop this procedure in the context of partially-observable environments, where transformer models have had large improvements over LSTMs recently, at the cost of significantly higher computational complexity. With transformer models as the learner and LSTMs as the actor, we demonstrate in several challenging memory environments that using Actor-Learner Distillation largely recovers the clear sample-efficiency gains of the transformer learner model while maintaining the fast inference and reduced total training time of the LSTM actor model.

----

## [313] Large Associative Memory Problem in Neurobiology and Machine Learning

**Authors**: *Dmitry Krotov, John J. Hopfield*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=X4y_10OX-hX](https://openreview.net/forum?id=X4y_10OX-hX)

**Abstract**:

Dense Associative Memories or modern Hopfield networks permit storage and reliable  retrieval of an exponentially large (in the dimension of feature space) number of memories. At the same time, their naive implementation is non-biological, since it seemingly requires the existence of many-body synaptic junctions between the neurons.  We show that these models are effective descriptions of a more microscopic (written in terms of biological degrees of freedom) theory that has additional (hidden) neurons and only requires two-body interactions between them. For this reason our proposed microscopic theory is a valid model of large associative memory with a degree of biological plausibility. The dynamics of our network and its reduced dimensional equivalent both minimize energy (Lyapunov) functions. When certain dynamical variables (hidden neurons) are integrated out from our microscopic theory, one can recover many of the models that were previously discussed in the literature, e.g. the model presented in "Hopfield Networks is All You Need" paper. We also provide an alternative derivation of the energy function and the update rule proposed in the aforementioned paper and clarify the relationships between various models of this class.

----

## [314] Protecting DNNs from Theft using an Ensemble of Diverse Models

**Authors**: *Sanjay Kariyappa, Atul Prakash, Moinuddin K. Qureshi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LucJxySuJcE](https://openreview.net/forum?id=LucJxySuJcE)

**Abstract**:

Several recent works have demonstrated highly effective model stealing (MS) attacks on Deep Neural Networks (DNNs) in black-box settings, even when the training data is unavailable. These attacks typically use some form of Out of Distribution (OOD) data to query the target model and use the predictions obtained to train a clone model. Such a clone model learns to approximate the decision boundary of the target model, achieving high accuracy on in-distribution examples. We propose Ensemble of Diverse Models (EDM) to defend against such MS attacks. EDM is made up of models that are trained to produce dissimilar predictions for OOD inputs. By using a different member of the ensemble to service different queries, our defense produces predictions that are highly discontinuous in the input space for the adversary's OOD queries. Such discontinuities cause the clone model trained on these predictions to have poor generalization on in-distribution examples. Our evaluations on several image classification tasks demonstrate that EDM defense can severely degrade the accuracy of clone models (up to $39.7\%$). Our defense has minimal impact on the target accuracy, negligible computational costs during inference, and is compatible with existing defenses for MS attacks.

----

## [315] Proximal Gradient Descent-Ascent: Variable Convergence under KŁ Geometry

**Authors**: *Ziyi Chen, Yi Zhou, Tengyu Xu, Yingbin Liang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LVotkZmYyDi](https://openreview.net/forum?id=LVotkZmYyDi)

**Abstract**:

The gradient descent-ascent (GDA) algorithm has been widely applied to solve minimax optimization problems. In order to achieve convergent policy parameters for minimax optimization, it is important that GDA generates convergent variable sequences rather than convergent sequences of function value or gradient norm. However, the variable convergence of GDA has been proved only under convexity geometries, and it is lack of understanding in general nonconvex minimax optimization. This paper fills such a gap by studying the convergence of a more general proximal-GDA for regularized nonconvex-strongly-concave minimax optimization. Specifically, we show that proximal-GDA admits a novel Lyapunov function, which monotonically decreases in the minimax optimization process and drives the variable sequences to a critical point. By leveraging this Lyapunov function and the KL geometry that parameterizes the local geometries of general nonconvex functions, we formally establish the variable convergence of proximal-GDA to a certain critical point $x^*$, i.e., $x_t\to x^*, y_t\to y^*(x^*)$. Furthermore, over the full spectrum of the KL-parameterized geometry, we show that proximal-GDA achieves different types of convergence rates ranging from sublinear convergence up to finite-step convergence, depending on the geometry associated with the KL parameter. This is the first theoretical result on the variable convergence for nonconvex minimax optimization.

----

## [316] Contextual Dropout: An Efficient Sample-Dependent Dropout Module

**Authors**: *Xinjie Fan, Shujian Zhang, Korawat Tanwisuth, Xiaoning Qian, Mingyuan Zhou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ct8_a9h1M](https://openreview.net/forum?id=ct8_a9h1M)

**Abstract**:

Dropout has been demonstrated as a simple and effective module to not only regularize the training process of deep neural networks, but also provide the uncertainty estimation for prediction. However, the quality of uncertainty estimation is highly dependent on the dropout probabilities. Most current models use the same dropout distributions across all data samples due to its simplicity.  Despite the potential gains in the flexibility of modeling uncertainty, sample-dependent dropout, on the other hand, is less explored as it often encounters scalability issues or involves non-trivial model changes.  In this paper, we propose contextual dropout with an efficient structural design as a simple and scalable sample-dependent dropout module, which can be applied to a wide range of models at the expense of only slightly increased memory and computational cost. We learn the dropout probabilities with a variational objective, compatible with both Bernoulli dropout and Gaussian dropout. We apply the contextual dropout module to various models with applications to image classification and visual question answering and demonstrate the scalability of the method with large-scale datasets, such as ImageNet and VQA 2.0. Our experimental results show that the proposed method outperforms baseline methods in terms of both accuracy and quality of uncertainty estimation.

----

## [317] Mirostat: a Neural Text decoding Algorithm that directly controls perplexity

**Authors**: *Sourya Basu, Govardana Sachitanandam Ramachandran, Nitish Shirish Keskar, Lav R. Varshney*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=W1G1JZEIy5_](https://openreview.net/forum?id=W1G1JZEIy5_)

**Abstract**:

Neural text decoding algorithms strongly influence the quality of texts generated using language models, but popular algorithms like top-k, top-p (nucleus), and temperature-based sampling may yield texts that have objectionable repetition or incoherence. Although these methods generate high-quality text after ad hoc parameter tuning that depends on the language model and the length of generated text, not much is known about the control they provide over the statistics of the output. This is important, however, since recent reports show that humans prefer when perplexity is neither too much nor too little and since we experimentally show that cross-entropy (log of perplexity) has a near-linear relation with repetition. First, we provide a theoretical analysis of perplexity in top-k, top-p, and temperature sampling, under Zipfian statistics. Then, we use this analysis to design a feedback-based adaptive top-k text decoding algorithm called mirostat that generates text (of any length) with a predetermined target value of perplexity without any tuning. Experiments show that for low values of k and p, perplexity drops significantly with generated text length and leads to excessive repetitions (the boredom trap). Contrarily, for large values of k and p, perplexity increases with generated text length and leads to incoherence (confusion trap). Mirostat avoids both traps. Specifically, we show that setting target perplexity value beyond a threshold yields negligible sentence-level repetitions. Experiments with
human raters for fluency, coherence, and quality further verify our findings.

----

## [318] DialoGraph: Incorporating Interpretable Strategy-Graph Networks into Negotiation Dialogues

**Authors**: *Rishabh Joshi, Vidhisha Balachandran, Shikhar Vashishth, Alan W. Black, Yulia Tsvetkov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kDnal_bbb-E](https://openreview.net/forum?id=kDnal_bbb-E)

**Abstract**:

To successfully negotiate a deal, it is not enough to communicate fluently: pragmatic planning of persuasive negotiation strategies is essential. While modern dialogue agents excel at generating fluent sentences, they still lack pragmatic grounding and cannot reason strategically. We present DialoGraph, a negotiation system that incorporates pragmatic strategies in a negotiation dialogue using graph neural networks. DialoGraph explicitly incorporates dependencies between sequences of strategies to enable improved and interpretable prediction of next optimal strategies, given the dialogue context. Our graph-based method outperforms prior state-of-the-art negotiation models both in the accuracy of strategy/dialogue act prediction and in the quality of downstream dialogue response generation. We qualitatively show further benefits of learned strategy-graphs in providing explicit associations between effective negotiation strategies over the course of the dialogue, leading to interpretable and strategic dialogues.

----

## [319] Multi-Time Attention Networks for Irregularly Sampled Time Series

**Authors**: *Satya Narayan Shukla, Benjamin M. Marlin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=4c0J6lwQ4_](https://openreview.net/forum?id=4c0J6lwQ4_)

**Abstract**:

Irregular sampling occurs in many time series modeling applications where it presents a significant challenge to standard deep learning models. This work is motivated by the analysis of physiological time series data in electronic health records, which are sparse, irregularly sampled, and multivariate. In this paper, we propose a new deep learning framework for this setting that we call Multi-Time Attention Networks. Multi-Time Attention Networks learn an embedding of continuous time values and use an attention mechanism to produce a fixed-length representation of a time series containing a variable number of observations. We investigate the performance of this framework on interpolation and classification tasks using multiple datasets. Our results show that the proposed approach performs as well or better than a range of baseline and recently proposed models while offering significantly faster training times than current state-of-the-art methods.

----

## [320] Learning Energy-Based Generative Models via Coarse-to-Fine Expanding and Sampling

**Authors**: *Yang Zhao, Jianwen Xie, Ping Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=aD1_5zowqV](https://openreview.net/forum?id=aD1_5zowqV)

**Abstract**:

Energy-based models (EBMs) parameterized by neural networks can be trained by the Markov chain Monte Carlo (MCMC) sampling-based maximum likelihood estimation. Despite the recent significant success of EBMs in image generation, the current approaches to train EBMs are unstable and have difficulty synthesizing diverse and high-fidelity images. In this paper, we propose to train EBMs via a multistage coarse-to-fine expanding and sampling strategy, which starts with learning a coarse-level EBM from images at low resolution and then gradually transits to learn a finer-level EBM from images at higher resolution by expanding the energy function as the learning progresses. The proposed framework is computationally efficient with smooth learning and sampling. It achieves the best performance on image generation amongst all EBMs and is the first successful EBM to synthesize high-fidelity images at $512\times512$ resolution. It can also be useful for image restoration and out-of-distribution detection. Lastly, the proposed framework is further generalized to the one-sided unsupervised image-to-image translation and beats baseline methods in terms of model size and training budget. We also present a gradient-based generative saliency method to interpret the translation dynamics.

----

## [321] Unsupervised Audiovisual Synthesis via Exemplar Autoencoders

**Authors**: *Kangle Deng, Aayush Bansal, Deva Ramanan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=43VKWxg_Sqr](https://openreview.net/forum?id=43VKWxg_Sqr)

**Abstract**:

We present an unsupervised approach that converts the input speech of any individual into audiovisual streams of potentially-infinitely many output speakers. Our approach builds on simple autoencoders that project out-of-sample data onto the distribution of the training set. We use exemplar autoencoders to learn the voice, stylistic prosody, and visual appearance of a specific target exemplar speech. In contrast to existing methods, the proposed approach can be easily extended to an arbitrarily large number of speakers and styles using only 3 minutes of target audio-video data, without requiring any training data for the input speaker. To do so, we learn audiovisual bottleneck representations that capture the structured linguistic content of speech. We outperform prior approaches on both audio and video synthesis.

----

## [322] A Learning Theoretic Perspective on Local Explainability

**Authors**: *Jeffrey Li, Vaishnavh Nagarajan, Gregory Plumb, Ameet Talwalkar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7aL-OtQrBWD](https://openreview.net/forum?id=7aL-OtQrBWD)

**Abstract**:

In this paper, we explore connections between interpretable machine learning and learning theory through the lens of local approximation explanations. First, we tackle the traditional problem of performance generalization and bound the test-time predictive accuracy of a model using a notion of how locally explainable it is.  Second, we explore the novel problem of explanation generalization which is an important concern for a growing class of finite sample-based local approximation explanations. Finally, we validate our theoretical results empirically and show that they reflect what can be seen in practice.

----

## [323] SEED: Self-supervised Distillation For Visual Representation

**Authors**: *Zhiyuan Fang, Jianfeng Wang, Lijuan Wang, Lei Zhang, Yezhou Yang, Zicheng Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=AHm3dbp7D1D](https://openreview.net/forum?id=AHm3dbp7D1D)

**Abstract**:

This paper is concerned with self-supervised learning for small models. The problem is motivated by our empirical studies that while the widely used contrastive self-supervised learning method has shown great progress on large model training, it does not work well for small models. To address this problem, we propose a new learning paradigm, named $\textbf{SE}$lf-Sup$\textbf{E}$rvised $\textbf{D}$istillation (${\large S}$EED), where we leverage a larger network (as Teacher) to transfer its representational knowledge into a smaller architecture (as Student) in a self-supervised fashion. Instead of directly learning from unlabeled data, we train a student encoder to mimic the similarity score distribution inferred by a teacher over a set of instances. We show that ${\large S}$EED dramatically boosts the performance of small networks on downstream tasks. Compared with self-supervised baselines, ${\large S}$EED improves the top-1 accuracy from 42.2% to 67.6% on EfficientNet-B0 and from 36.3% to 68.2% on MobileNet-v3-Large on the ImageNet-1k dataset.

----

## [324] Isometric Propagation Network for Generalized Zero-shot Learning

**Authors**: *Lu Liu, Tianyi Zhou, Guodong Long, Jing Jiang, Xuanyi Dong, Chengqi Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-mWcQVLPSPy](https://openreview.net/forum?id=-mWcQVLPSPy)

**Abstract**:

Zero-shot learning (ZSL) aims to classify images of an unseen class only based on a few attributes describing that class but no access to any training sample. A popular strategy is to learn a mapping between the semantic space of class attributes and the visual space of images based on the seen classes and their data. Thus, an unseen class image can be ideally mapped to its corresponding class attributes. The key challenge is how to align the representations in the two spaces. For most ZSL settings, the attributes for each seen/unseen class are only represented by a vector while the seen-class data provide much more information. Thus, the imbalanced supervision from the semantic and the visual space can make the learned mapping easily overfitting to the seen classes. To resolve this problem, we propose Isometric Propagation Network (IPN), which learns to strengthen the relation between classes within each space and align the class dependency in the two spaces. Specifically, IPN learns to propagate the class representations on an auto-generated graph within each space. In contrast to only aligning the resulted static representation, we regularize the two dynamic propagation procedures to be isometric in terms of the two graphs' edge weights per step by minimizing a consistency loss between them. IPN achieves state-of-the-art performance on three popular ZSL benchmarks. To evaluate the generalization capability of IPN, we further build two larger benchmarks with more diverse unseen classes and demonstrate the advantages of IPN on them.

----

## [325] Effective and Efficient Vote Attack on Capsule Networks

**Authors**: *Jindong Gu, Baoyuan Wu, Volker Tresp*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=33rtZ4Sjwjn](https://openreview.net/forum?id=33rtZ4Sjwjn)

**Abstract**:

Standard Convolutional Neural Networks (CNNs) can be easily fooled by images with small quasi-imperceptible artificial perturbations. As alternatives to CNNs, the recently proposed Capsule Networks (CapsNets) are shown to be more robust to white-box attack than CNNs under popular attack protocols. Besides, the class-conditional reconstruction part of CapsNets is also used to detect adversarial examples. In this work, we investigate the adversarial robustness of CapsNets, especially how the inner workings of CapsNets change when the output capsules are attacked. The first observation is that adversarial examples misled CapsNets by manipulating the votes from primary capsules. Another observation is the high computational cost, when we directly apply multi-step attack methods designed for CNNs to attack CapsNets, due to the computationally expensive routing mechanism. Motivated by these two observations, we propose a novel vote attack where we attack votes of CapsNets directly. Our vote attack is not only effective, but also efficient by circumventing the routing process. Furthermore, we integrate our vote attack into the detection-aware attack paradigm, which can successfully bypass the class-conditional reconstruction based detection method. Extensive experiments demonstrate the superior attack performance of our vote attack on CapsNets.

----

## [326] Heteroskedastic and Imbalanced Deep Learning with Adaptive Regularization

**Authors**: *Kaidi Cao, Yining Chen, Junwei Lu, Nikos Aréchiga, Adrien Gaidon, Tengyu Ma*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=mEdwVCRJuX4](https://openreview.net/forum?id=mEdwVCRJuX4)

**Abstract**:

Real-world large-scale datasets are heteroskedastic and imbalanced --- labels have varying levels of uncertainty and label distributions are long-tailed. Heteroskedasticity and imbalance challenge deep learning algorithms due to the difficulty of distinguishing among mislabeled, ambiguous, and rare examples. Addressing heteroskedasticity and imbalance simultaneously is under-explored. We propose a data-dependent regularization technique for heteroskedastic datasets that regularizes different regions of the input space differently. Inspired by the theoretical derivation of the optimal regularization strength in a one-dimensional nonparametric classification setting, our approach adaptively regularizes the data points in higher-uncertainty, lower-density regions more heavily. We test our method on several benchmark tasks, including a real-world heteroskedastic and imbalanced dataset, WebVision. Our experiments corroborate our theory and demonstrate a significant improvement over other methods in noise-robust deep learning.

----

## [327] Continuous Wasserstein-2 Barycenter Estimation without Minimax Optimization

**Authors**: *Alexander Korotin, Lingxiao Li, Justin Solomon, Evgeny Burnaev*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3tFAs5E-Pe](https://openreview.net/forum?id=3tFAs5E-Pe)

**Abstract**:

Wasserstein barycenters provide a geometric notion of the weighted average of probability measures based on optimal transport. In this paper, we present a scalable algorithm to compute Wasserstein-2 barycenters given sample access to the input measures, which are not restricted to being discrete. While past approaches rely on entropic or quadratic regularization, we employ input convex neural networks and cycle-consistency regularization to avoid introducing bias. As a result, our approach does not resort to minimax optimization. We provide theoretical analysis on error bounds as well as empirical evidence of the effectiveness of the proposed approach in low-dimensional qualitative scenarios and high-dimensional quantitative experiments.

----

## [328] Neural Thompson Sampling

**Authors**: *Weitong Zhang, Dongruo Zhou, Lihong Li, Quanquan Gu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tkAtoZkcUnm](https://openreview.net/forum?id=tkAtoZkcUnm)

**Abstract**:

Thompson Sampling (TS) is one of the most effective algorithms for solving contextual multi-armed bandit problems. In this paper, we propose a new algorithm, called Neural Thompson Sampling, which adapts deep neural networks for both exploration and exploitation. At the core of our algorithm is a novel posterior distribution of the reward, where its mean is the neural network approximator, and its variance is built upon the neural tangent features of the corresponding neural network. We prove that, provided the underlying reward function is bounded, the proposed algorithm is guaranteed to achieve a cumulative regret of $O(T^{1/2})$, which matches the regret of other contextual bandit algorithms in terms of total round number $T$. Experimental comparisons with other benchmark bandit algorithms on various data sets corroborate our theory.

----

## [329] Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics

**Authors**: *Daniel Kunin, Javier Sagastuy-Breña, Surya Ganguli, Daniel L. K. Yamins, Hidenori Tanaka*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=q8qLAbQBupm](https://openreview.net/forum?id=q8qLAbQBupm)

**Abstract**:

Understanding the dynamics of neural network parameters during training is one of the key challenges in building a theoretical foundation for deep learning. A central obstacle is that the motion of a network in high-dimensional parameter space undergoes discrete finite steps along complex stochastic gradients derived from real-world datasets. We circumvent this obstacle through a unifying theoretical framework based on intrinsic symmetries embedded in a network's architecture that are present for any dataset. We show that any such symmetry imposes stringent geometric constraints on gradients and Hessians, leading to an associated conservation law in the continuous-time limit of stochastic gradient descent (SGD), akin to Noether's theorem in physics. We further show that finite learning rates used in practice can actually break these symmetry induced conservation laws. We apply tools from finite difference methods to derive modified gradient flow, a differential equation that better approximates the numerical trajectory taken by SGD at finite learning rates. We combine modified gradient flow with our framework of symmetries to derive exact integral expressions for the dynamics of certain parameter combinations. We empirically validate our analytic expressions for learning dynamics on VGG-16 trained on Tiny ImageNet. Overall, by exploiting symmetry, our work demonstrates that we can analytically describe the learning dynamics of various parameter combinations at finite learning rates and batch sizes for state of the art architectures trained on any dataset.

----

## [330] Neural gradients are near-lognormal: improved quantized and sparse training

**Authors**: *Brian Chmiel, Liad Ben-Uri, Moran Shkolnik, Elad Hoffer, Ron Banner, Daniel Soudry*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=EoFNy62JGd](https://openreview.net/forum?id=EoFNy62JGd)

**Abstract**:

While training can mostly be accelerated by reducing the time needed to propagate neural gradients (loss gradients with respect to the intermediate neural layer outputs) back throughout the model, most previous works focus on the quantization/pruning of weights and activations. These methods are often not applicable to neural gradients, which have very different statistical properties. Distinguished from weights and activations, we find that the distribution of neural gradients is approximately lognormal. Considering this, we suggest two closed-form analytical methods to reduce the computational and memory burdens of neural gradients. The first method optimizes the floating-point format and scale of the gradients. The second method accurately sets sparsity thresholds for gradient pruning.  Each method achieves state-of-the-art results on ImageNet. To the best of our knowledge, this paper is the first to (1) quantize the gradients to 6-bit floating-point formats, or (2) achieve up to 85% gradient sparsity --- in each case without accuracy degradation.
Reference implementation accompanies the paper in the supplementary material.

----

## [331] RODE: Learning Roles to Decompose Multi-Agent Tasks

**Authors**: *Tonghan Wang, Tarun Gupta, Anuj Mahajan, Bei Peng, Shimon Whiteson, Chongjie Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TTUVg6vkNjK](https://openreview.net/forum?id=TTUVg6vkNjK)

**Abstract**:

Role-based learning holds the promise of achieving scalable multi-agent learning by decomposing complex tasks using roles. However, it is largely unclear how to efficiently discover such a set of roles. To solve this problem, we propose to first decompose joint action spaces into restricted role action spaces by clustering actions according to their effects on the environment and other agents. Learning a role selector based on action effects makes role discovery much easier because it forms a bi-level learning hierarchy: the role selector searches in a smaller role space and at a lower temporal resolution, while role policies learn in significantly reduced primitive action-observation spaces. We further integrate information about action effects into the role policies to boost learning efficiency and policy generalization. By virtue of these advances, our method (1) outperforms the current state-of-the-art MARL algorithms on 9 of the 14 scenarios that comprise the challenging StarCraft II micromanagement benchmark and (2) achieves rapid transfer to new environments with three times the number of agents. Demonstrative videos can be viewed at https://sites.google.com/view/rode-marl.

----

## [332] Revisiting Hierarchical Approach for Persistent Long-Term Video Prediction

**Authors**: *Wonkwang Lee, Whie Jung, Han Zhang, Ting Chen, Jing Yu Koh, Thomas E. Huang, Hyungsuk Yoon, Honglak Lee, Seunghoon Hong*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3RLN4EPMdYd](https://openreview.net/forum?id=3RLN4EPMdYd)

**Abstract**:

Learning to predict the long-term future of video frames is notoriously challenging due to the inherent ambiguities in a distant future and dramatic amplification of prediction error over time. Despite the recent advances in the literature, existing approaches are limited to moderately short-term prediction (less than a few seconds), while extrapolating it to a longer future quickly leads to destruction in structure and content. In this work, we revisit the hierarchical models in video prediction. Our method generates future frames by first estimating a sequence of dense semantic structures and subsequently translating the estimated structures to pixels by video-to-video translation model. Despite the simplicity, we show that modeling structures and their dynamics in categorical structure space with stochastic sequential estimator leads to surprisingly successful long-term prediction. We evaluate our method on two challenging video prediction scenarios, \emph{car driving} and \emph{human dancing}, and demonstrate that it can generate complicated scene structures and motions over a very long time horizon (\ie~thousands frames), setting a new standard of video prediction with orders of magnitude longer prediction time than existing approaches. Video results are available at https://1konny.github.io/HVP/.

----

## [333] Physics-aware, probabilistic model order reduction with guaranteed stability

**Authors**: *Sebastian Kaltenbach, Phaedon-Stelios Koutsourelakis*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vyY0jnWG-tK](https://openreview.net/forum?id=vyY0jnWG-tK)

**Abstract**:

Given (small amounts of) time-series' data from  a high-dimensional, fine-grained, multiscale dynamical system, we propose a generative framework for learning an effective, lower-dimensional, coarse-grained dynamical model that is predictive of the fine-grained system's long-term evolution but also of its behavior under different initial conditions.
We target fine-grained models as they arise in physical applications (e.g. molecular dynamics, agent-based models), the dynamics  of which are strongly non-stationary but their transition to equilibrium is governed by unknown slow processes which are largely inaccessible by brute-force simulations.
Approaches based on domain knowledge heavily rely on physical insight in identifying temporally slow features and fail to enforce the long-term stability of the learned dynamics. On the other hand, purely statistical frameworks lack interpretability and rely on large amounts of expensive simulation data (long and multiple trajectories) as they cannot infuse domain knowledge. 
The generative framework proposed achieves  the aforementioned desiderata by  employing a flexible prior on the complex plane for the latent, slow processes, and  an intermediate layer of physics-motivated latent variables that reduces reliance on data and imbues inductive bias. In contrast to existing schemes, it does not require  the a priori definition of projection operators from the fine-grained description and addresses simultaneously the tasks of dimensionality reduction and model estimation.
We demonstrate its efficacy and accuracy in multiscale physical systems of particle dynamics where probabilistic, long-term predictions of phenomena not contained in the training data are produced.

----

## [334] Modelling Hierarchical Structure between Dialogue Policy and Natural Language Generator with Option Framework for Task-oriented Dialogue System

**Authors**: *Jianhong Wang, Yuan Zhang, Tae-Kyun Kim, Yunjie Gu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kLbhLJ8OT12](https://openreview.net/forum?id=kLbhLJ8OT12)

**Abstract**:

Designing task-oriented dialogue systems is a challenging research topic, since it needs not only to generate utterances fulfilling user requests but also to guarantee the comprehensibility. Many previous works trained end-to-end (E2E) models with supervised learning (SL), however, the bias in annotated system utterances remains as a bottleneck. Reinforcement learning (RL) deals with the problem through using non-differentiable evaluation metrics (e.g., the success rate) as rewards. Nonetheless, existing works with RL showed that the comprehensibility of generated system utterances could be corrupted when improving the performance on fulfilling user requests. In our work, we (1) propose modelling the hierarchical structure between dialogue policy and natural language generator (NLG) with the option framework, called HDNO, where the latent dialogue act is applied to avoid designing specific dialogue act representations; (2) train HDNO via hierarchical reinforcement learning (HRL), as well as suggest the asynchronous updates between dialogue policy and NLG during training to theoretically guarantee their convergence to a local maximizer; and (3) propose using a discriminator modelled with language models as an additional reward to further improve the comprehensibility. We test HDNO on MultiWoz 2.0 and MultiWoz 2.1, the datasets on multi-domain dialogues, in comparison with word-level E2E model trained with RL, LaRL and HDSA, showing improvements on the performance evaluated by automatic evaluation metrics and human evaluation. Finally, we demonstrate the semantic meanings of latent dialogue acts to show the explanability for HDNO.

----

## [335] Learning explanations that are hard to vary

**Authors**: *Giambattista Parascandolo, Alexander Neitz, Antonio Orvieto, Luigi Gresele, Bernhard Schölkopf*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hb1sDDSLbV](https://openreview.net/forum?id=hb1sDDSLbV)

**Abstract**:

In this paper, we investigate the principle that good explanations are hard to vary in the context of deep learning.
We show that averaging gradients across examples -- akin to a logical OR of patterns -- can favor memorization and `patchwork' solutions that sew together different strategies, instead of identifying invariances.
To inspect this, we first formalize a notion of consistency for minima of the loss surface, which measures to what extent a minimum appears only when examples are pooled.
We then propose and experimentally validate a simple alternative algorithm based on a logical AND, that focuses on invariances and prevents memorization in a set of real-world tasks. 
Finally, using a synthetic dataset with a clear distinction between invariant and spurious mechanisms, we dissect learning signals and compare this approach to well-established regularizers.

----

## [336] Efficient Generalized Spherical CNNs

**Authors**: *Oliver J. Cobb, Christopher G. R. Wallis, Augustine N. Mavor-Parker, Augustin Marignier, Matthew A. Price, Mayeul d'Avezac, Jason D. McEwen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rWZz3sJfCkm](https://openreview.net/forum?id=rWZz3sJfCkm)

**Abstract**:

Many problems across computer vision and the natural sciences require the analysis of spherical data, for which representations may be learned efficiently by encoding equivariance to rotational symmetries.  We present a generalized spherical CNN framework that encompasses various existing approaches and allows them to be leveraged alongside each other.  The only existing non-linear spherical CNN layer that is strictly equivariant has complexity $\mathcal{O}(C^2L^5)$, where $C$ is a measure of representational capacity and $L$ the spherical harmonic bandlimit.  Such a high computational cost often prohibits the use of strictly equivariant spherical CNNs.  We develop two new strictly equivariant layers with reduced complexity $\mathcal{O}(CL^4)$ and $\mathcal{O}(CL^3 \log L)$, making larger, more expressive models computationally feasible.  Moreover, we adopt efficient sampling theory to achieve further computational savings. We show that these developments allow the construction of more expressive hybrid models that achieve state-of-the-art accuracy and parameter efficiency on spherical benchmark problems.

----

## [337] Collective Robustness Certificates: Exploiting Interdependence in Graph Neural Networks

**Authors**: *Jan Schuchardt, Aleksandar Bojchevski, Johannes Klicpera, Stephan Günnemann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ULQdiUTHe3y](https://openreview.net/forum?id=ULQdiUTHe3y)

**Abstract**:

In tasks like node classification, image segmentation, and named-entity recognition we have a classifier that simultaneously outputs multiple predictions (a vector of labels) based on a single input, i.e. a single graph, image, or document respectively. Existing adversarial robustness certificates consider each prediction independently and are thus overly pessimistic for such tasks. They implicitly assume that an adversary can use different perturbed inputs to attack different predictions, ignoring the fact that we have a single shared input. We propose the first collective robustness certificate which computes the number of predictions that are simultaneously guaranteed to remain stable under perturbation, i.e. cannot be attacked. We focus on Graph Neural Networks and leverage their locality property - perturbations only affect the predictions in a close neighborhood - to fuse multiple single-node certificates into a drastically stronger collective certificate. For example, on the Citeseer dataset our collective certificate for node classification increases the average number of certifiable feature perturbations from $7$ to $351$.

----

## [338] Entropic gradient descent algorithms and wide flat minima

**Authors**: *Fabrizio Pittorino, Carlo Lucibello, Christoph Feinauer, Gabriele Perugini, Carlo Baldassi, Elizaveta Demyanenko, Riccardo Zecchina*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xjXg0bnoDmS](https://openreview.net/forum?id=xjXg0bnoDmS)

**Abstract**:

The properties of flat minima in the empirical risk landscape of neural networks have been debated for some time. Increasing evidence suggests they possess better generalization capabilities with respect to sharp ones. In this work we first discuss the relationship between alternative measures of flatness: The local entropy, which is useful for analysis and algorithm development, and the local energy, which is easier to compute and was shown empirically in extensive tests on state-of-the-art networks to be the best predictor of generalization capabilities. We show semi-analytically in simple controlled scenarios that these two measures correlate strongly with each other and with generalization. Then, we extend the analysis to the deep learning scenario by extensive numerical validations. We study two algorithms, Entropy-SGD and Replicated-SGD, that explicitly include the local entropy in the optimization objective. We devise a training schedule by which we consistently find flatter minima (using both flatness measures), and improve the generalization error for common architectures (e.g. ResNet, EfficientNet).

----

## [339] Achieving Linear Speedup with Partial Worker Participation in Non-IID Federated Learning

**Authors**: *Haibo Yang, Minghong Fang, Jia Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jDdzh5ul-d](https://openreview.net/forum?id=jDdzh5ul-d)

**Abstract**:

Federated learning (FL) is a distributed machine learning architecture that leverages a large number of workers to jointly learn a model with decentralized data.  FL has received increasing attention in recent years thanks to its data privacy protection, communication efficiency and a linear speedup for convergence in training (i.e., convergence performance increases linearly with respect to the number of workers). However, existing studies on linear speedup for convergence are only limited to the assumptions of i.i.d. datasets across workers and/or full worker participation, both of which rarely hold in practice. So far, it remains an open question whether or not the linear speedup for convergence is achievable under non-i.i.d. datasets with partial worker participation in FL.  In this paper, we show that the answer is affirmative.  Specifically, we show that the federated averaging (FedAvg) algorithm (with two-sided learning rates) on non-i.i.d. datasets in non-convex settings achieves a convergence rate $\mathcal{O}(\frac{1}{\sqrt{mKT}} + \frac{1}{T})$ for full worker participation and a convergence rate $\mathcal{O}(\frac{\sqrt{K}}{\sqrt{nT}} + \frac{1}{T})$ for partial worker participation, where $K$ is the number of local steps, $T$ is the number of total communication rounds, $m$ is the total worker number and $n$ is the worker number in one communication round if for partial worker participation. Our results also reveal that the local steps in FL could help the convergence and show that the maximum number of local steps can be improved to $T/m$ in full worker participation. We conduct extensive experiments on MNIST and CIFAR-10 to verify our theoretical results.

----

## [340] Categorical Normalizing Flows via Continuous Transformations

**Authors**: *Phillip Lippe, Efstratios Gavves*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-GLNZeVDuik](https://openreview.net/forum?id=-GLNZeVDuik)

**Abstract**:

Despite their popularity, to date, the application of normalizing flows on categorical data stays limited. The current practice of using dequantization to map discrete data to a continuous space is inapplicable as categorical data has no intrinsic order. Instead, categorical data have complex and latent relations that must be inferred, like the synonymy between words. In this paper, we investigate Categorical Normalizing Flows, that is normalizing flows for categorical data. By casting the encoding of categorical data in continuous space as a variational inference problem, we jointly optimize the continuous representation and the model likelihood. Using a factorized decoder, we introduce an inductive bias to model any interactions in the normalizing flow. As a consequence, we do not only simplify the optimization compared to having a joint decoder, but also make it possible to scale up to a large number of categories that is currently impossible with discrete normalizing flows. Based on Categorical Normalizing Flows, we propose GraphCNF a permutation-invariant generative model on graphs. GraphCNF implements a three step approach modeling the nodes, edges, and adjacency matrix stepwise to increase efficiency. On molecule generation, GraphCNF outperforms both one-shot and autoregressive flow-based state-of-the-art.

----

## [341] Learning to Represent Action Values as a Hypergraph on the Action Vertices

**Authors**: *Arash Tavakoli, Mehdi Fatemi, Petar Kormushev*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Xv_s64FiXTv](https://openreview.net/forum?id=Xv_s64FiXTv)

**Abstract**:

Action-value estimation is a critical component of many reinforcement learning (RL) methods whereby sample complexity relies heavily on how fast a good estimator for action value can be learned. By viewing this problem through the lens of representation learning, good representations of both state and action can facilitate action-value estimation. While advances in deep learning have seamlessly driven progress in learning state representations, given the specificity of the notion of agency to RL, little attention has been paid to learning action representations. We conjecture that leveraging the combinatorial structure of multi-dimensional action spaces is a key ingredient for learning good representations of action. To test this, we set forth the action hypergraph networks framework---a class of functions for learning action representations in multi-dimensional discrete action spaces with a structural inductive bias. Using this framework we realise an agent class based on a combination with deep Q-networks, which we dub hypergraph Q-networks. We show the effectiveness of our approach on a myriad of domains: illustrative prediction problems under minimal confounding effects, Atari 2600 games, and discretised physical control benchmarks.

----

## [342] Debiasing Concept-based Explanations with Causal Analysis

**Authors**: *Mohammad Taha Bahadori, David Heckerman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6puUoArESGp](https://openreview.net/forum?id=6puUoArESGp)

**Abstract**:

Concept-based explanation approach is a popular model interpertability tool because it expresses the reasons for a model's predictions in terms of concepts that are meaningful for the domain experts. In this work, we study the problem of the concepts being correlated with confounding information in the features. We propose a new causal prior graph for modeling the impacts of unobserved variables and a method to remove the impact of confounding information and noise using a two-stage regression technique borrowed from the instrumental variable literature. We also model the completeness of the concepts set and show that our debiasing method works when the concepts are not complete. Our synthetic and real-world experiments demonstrate the success of our method in removing biases and improving the ranking of the concepts in terms of their contribution to the explanation of the predictions.

----

## [343] Lifelong Learning of Compositional Structures

**Authors**: *Jorge A. Mendez, Eric Eaton*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ADWd4TJO13G](https://openreview.net/forum?id=ADWd4TJO13G)

**Abstract**:

A hallmark of human intelligence is the ability to construct self-contained chunks of knowledge and adequately reuse them in novel combinations for solving different yet structurally related problems. Learning such compositional structures has been a significant challenge for artificial systems, due to the combinatorial nature of the underlying search problem. To date, research into compositional learning has largely proceeded separately from work on lifelong or continual learning. We integrate these two lines of work to present a general-purpose framework for lifelong learning of compositional structures that can be used for solving a stream of related tasks. Our framework separates the learning process into two broad stages: learning how to best combine existing components in order to assimilate a novel problem, and learning how to adapt the set of existing components to accommodate the new problem. This separation explicitly handles the trade-off between the stability required to remember how to solve earlier tasks and the flexibility required to solve new tasks, as we show empirically in an extensive evaluation.

----

## [344] Rethinking Embedding Coupling in Pre-trained Language Models

**Authors**: *Hyung Won Chung, Thibault Févry, Henry Tsai, Melvin Johnson, Sebastian Ruder*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xpFFI_NtgpW](https://openreview.net/forum?id=xpFFI_NtgpW)

**Abstract**:

We re-evaluate the standard practice of sharing weights between input and output embeddings in state-of-the-art pre-trained language models. We show that decoupled embeddings provide increased modeling flexibility, allowing us to significantly improve the efficiency of parameter allocation in the input embedding of multilingual models. By reallocating the input embedding parameters in the Transformer layers, we achieve dramatically better performance on standard natural language understanding tasks with the same number of parameters during fine-tuning. We also show that allocating additional capacity to the output embedding provides benefits to the model that persist through the fine-tuning stage even though the output embedding is discarded after pre-training. Our analysis shows that larger output embeddings prevent the model's last layers from overspecializing to the pre-training task and encourage Transformer representations to be more general and more transferable to other tasks and languages. Harnessing these findings, we are able to train models that achieve strong performance on the XTREME benchmark without increasing the number of parameters at the fine-tuning stage.

----

## [345] Creative Sketch Generation

**Authors**: *Songwei Ge, Vedanuj Goswami, Larry Zitnick, Devi Parikh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gwnoVHIES05](https://openreview.net/forum?id=gwnoVHIES05)

**Abstract**:

Sketching or doodling is a popular creative activity that people engage in. However, most existing work in automatic sketch understanding or generation has focused on sketches that are quite mundane. In this work, we introduce two datasets of creative sketches -- Creative Birds and Creative Creatures -- containing 10k sketches each along with part annotations. We propose DoodlerGAN -- a part-based Generative Adversarial Network (GAN) -- to generate unseen compositions of novel part appearances. Quantitative evaluations as well as human studies demonstrate that sketches generated by our approach are more creative and of higher quality than existing approaches. In fact, in Creative Birds, subjects prefer sketches generated by DoodlerGAN over those drawn by humans!

----

## [346] Concept Learners for Few-Shot Learning

**Authors**: *Kaidi Cao, Maria Brbic, Jure Leskovec*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eJIJF3-LoZO](https://openreview.net/forum?id=eJIJF3-LoZO)

**Abstract**:

Developing algorithms that are able to generalize to a novel task given only a few labeled examples represents a fundamental challenge in closing the gap between machine- and human-level performance. The core of human cognition lies in the structured, reusable concepts that help us to rapidly adapt to new tasks and provide reasoning behind our decisions. However, existing meta-learning methods learn complex representations across prior labeled tasks without imposing any structure on the learned representations. Here we propose COMET, a meta-learning method that improves generalization ability by learning to learn along human-interpretable concept dimensions. Instead of learning a joint unstructured metric space, COMET learns mappings of high-level concepts into semi-structured metric spaces, and effectively combines the outputs of independent concept learners. We evaluate our model on few-shot tasks from diverse domains, including fine-grained image classification, document categorization  and cell type annotation on a novel dataset from a biological domain developed in our work. COMET significantly outperforms strong meta-learning baselines, achieving 6-15% relative improvement on the most challenging 1-shot learning tasks, while unlike existing methods providing interpretations behind the model's predictions.

----

## [347] Domain Generalization with MixStyle

**Authors**: *Kaiyang Zhou, Yongxin Yang, Yu Qiao, Tao Xiang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6xHJ37MVxxp](https://openreview.net/forum?id=6xHJ37MVxxp)

**Abstract**:

Though convolutional neural networks (CNNs) have demonstrated remarkable ability in learning discriminative features, they often generalize poorly to unseen domains. Domain generalization aims to address this problem by learning from a set of source domains a model that is generalizable to any unseen domain. In this paper, a novel approach is proposed based on probabilistically mixing instance-level feature statistics of training samples across source domains. Our method, termed MixStyle, is motivated by the observation that visual domain is closely related to image style (e.g., photo vs.~sketch images). Such style information is captured by the bottom layers of a CNN where our proposed style-mixing takes place. Mixing styles of training instances results in novel domains being synthesized implicitly, which increase the domain diversity of the source domains, and hence the generalizability of the trained model. MixStyle fits into mini-batch training perfectly and is extremely easy to implement. The effectiveness of MixStyle is demonstrated on a wide range of tasks including category classification, instance retrieval and reinforcement learning.

----

## [348] DeLighT: Deep and Light-weight Transformer

**Authors**: *Sachin Mehta, Marjan Ghazvininejad, Srinivasan Iyer, Luke Zettlemoyer, Hannaneh Hajishirzi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ujmgfuxSLrO](https://openreview.net/forum?id=ujmgfuxSLrO)

**Abstract**:

We introduce a deep and light-weight transformer, DeLighT, that delivers similar or better performance than standard transformer-based models with significantly fewer parameters. DeLighT more efficiently allocates parameters both (1) within each Transformer block using the DeLighT transformation, a deep and light-weight transformation and (2) across blocks using block-wise scaling, that allows for shallower and narrower DeLighT blocks near the input and wider and deeper DeLighT blocks near the output. Overall, DeLighT networks are 2.5 to 4 times deeper than standard transformer models and yet have fewer parameters and operations. Experiments on benchmark machine translation and language modeling tasks show that DeLighT matches or improves the performance of baseline Transformers with 2 to 3 times fewer parameters on average.

----

## [349] Single-Timescale Actor-Critic Provably Finds Globally Optimal Policy

**Authors**: *Zuyue Fu, Zhuoran Yang, Zhaoran Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=pqZV_srUVmK](https://openreview.net/forum?id=pqZV_srUVmK)

**Abstract**:

We study the global convergence and global optimality of actor-critic, one of the most popular families of reinforcement learning algorithms. While most existing works on actor-critic employ bi-level or two-timescale updates, we focus on the more practical single-timescale setting, where the actor and critic are updated simultaneously. Specifically, in each iteration, the critic update is obtained by applying the Bellman evaluation operator only once while the actor is updated in the policy gradient direction computed using the critic. Moreover, we consider two function approximation settings where both the actor and critic are represented by linear or deep neural networks. For both cases, we prove that the actor sequence converges to a globally optimal policy at a sublinear $O(K^{-1/2})$ rate, where $K$ is the number of iterations. To the best of our knowledge, we establish the rate of convergence and global optimality of single-timescale actor-critic with linear function approximation for the first time. Moreover, under the broader scope of policy optimization with nonlinear function approximation, we prove that actor-critic with deep neural network finds the globally optimal policy at a sublinear rate for the first time.

----

## [350] Mastering Atari with Discrete World Models

**Authors**: *Danijar Hafner, Timothy P. Lillicrap, Mohammad Norouzi, Jimmy Ba*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0oabwyZbOu](https://openreview.net/forum?id=0oabwyZbOu)

**Abstract**:

Intelligent agents need to generalize from past experience to achieve goals in complex environments. World models facilitate such generalization and allow learning behaviors from imagined outcomes to increase sample-efficiency. While learning world models from image inputs has recently become feasible for some tasks, modeling Atari games accurately enough to derive successful behaviors has remained an open challenge for many years. We introduce DreamerV2, a reinforcement learning agent that learns behaviors purely from predictions in the compact latent space of a powerful world model. The world model uses discrete representations and is trained separately from the policy. DreamerV2 constitutes the first agent that achieves human-level performance on the Atari benchmark of 55 tasks by learning behaviors inside a separately trained world model. With the same computational budget and wall-clock time, Dreamer V2 reaches 200M frames and surpasses the final performance of the top single-GPU agents IQN and Rainbow. DreamerV2 is also applicable to tasks with continuous actions, where it learns an accurate world model of a complex humanoid robot and solves stand-up and walking from only pixel inputs.

----

## [351] Learning Neural Event Functions for Ordinary Differential Equations

**Authors**: *Ricky T. Q. Chen, Brandon Amos, Maximilian Nickel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kW_zpEmMLdP](https://openreview.net/forum?id=kW_zpEmMLdP)

**Abstract**:

The existing Neural ODE formulation relies on an explicit knowledge of the termination time. We extend Neural ODEs to implicitly defined termination criteria modeled by neural event functions, which can be chained together and differentiated through. Neural Event ODEs are capable of modeling discrete and instantaneous changes in a continuous-time system, without prior knowledge of when these changes should occur or how many such changes should exist. We test our approach in modeling hybrid discrete- and continuous- systems such as switching dynamical systems and collision in multi-body systems, and we propose simulation-based training of point processes with applications in discrete control.

----

## [352] Contemplating Real-World Object Classification

**Authors**: *Ali Borji*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Q4EUywJIkqr](https://openreview.net/forum?id=Q4EUywJIkqr)

**Abstract**:

Deep object recognition models have been very successful over benchmark
datasets such as ImageNet. How accurate and robust are they to distribution
shifts arising from natural and synthetic variations in datasets? Prior research on
this problem has primarily focused on ImageNet variations (e.g., ImageNetV2,
ImageNet-A). To avoid potential inherited biases in these studies, we take a
different approach. Specifically, we reanalyze the ObjectNet dataset recently
proposed by Barbu et al. containing objects in daily life situations. They showed
a dramatic performance drop of the state of the art object recognition models on
this dataset. Due to the importance and implications of their results regarding
the generalization ability of deep models, we take a second look at their analysis.
We find that applying deep models to the isolated objects, rather than the entire
scene as is done in the original paper, results in around 20-30% performance
improvement. Relative to the numbers reported in Barbu et al., around 10-15%
of the performance loss is recovered, without any test time data augmentation.
Despite this gain, however, we conclude that deep models still suffer drastically
on the ObjectNet dataset. We also investigate the robustness of models against
synthetic image perturbations such as geometric transformations (e.g., scale,
rotation, translation), natural image distortions (e.g., impulse noise, blur) as well
as adversarial attacks (e.g., FGSM and PGD-5). Our results indicate that limiting
the object area as much as possible (i.e., from the entire image to the bounding
box to the segmentation mask) leads to consistent improvement in accuracy and
robustness. Finally, through a qualitative analysis of ObjectNet data, we find that
i) a large number of images in this dataset are hard to recognize even for humans,
and ii) easy (hard) samples for models match with easy (hard) samples for humans.
Overall, our analysis shows that ObjecNet is still a challenging test platform that
can be used to measure the generalization ability of models. The code and data
are available in [masked due to blind review].

----

## [353] Neural Spatio-Temporal Point Processes

**Authors**: *Ricky T. Q. Chen, Brandon Amos, Maximilian Nickel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XQQA6-So14](https://openreview.net/forum?id=XQQA6-So14)

**Abstract**:

We propose a new class of parameterizations for spatio-temporal point processes which leverage Neural ODEs as a computational method and enable flexible, high-fidelity models of discrete events that are localized in continuous time and space. Central to our approach is a combination of continuous-time neural networks with two novel neural architectures, \ie, Jump and Attentive Continuous-time Normalizing Flows. This approach allows us to learn complex distributions for both the spatial and temporal domain and to condition non-trivially on the observed event history. We validate our models on data sets from a wide variety of contexts such as seismology, epidemiology, urban mobility, and neuroscience.

----

## [354] Generative Time-series Modeling with Fourier Flows

**Authors**: *Ahmed M. Alaa, Alex James Chan, Mihaela van der Schaar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PpshD0AXfA](https://openreview.net/forum?id=PpshD0AXfA)

**Abstract**:

Generating synthetic time-series data is crucial in various application domains, such as medical prognosis, wherein research is hamstrung by the lack of access to data due to concerns over privacy. Most of the recently proposed methods for generating synthetic time-series rely on implicit likelihood modeling using generative adversarial networks (GANs)—but such models can be difficult to train, and may jeopardize privacy by “memorizing” temporal patterns in training data. In this paper, we propose an explicit likelihood model based on a novel class of normalizing flows that view time-series data in the frequency-domain rather than the time-domain. The proposed flow, dubbed a Fourier flow, uses a discrete Fourier transform (DFT) to convert variable-length time-series with arbitrary sampling periods into fixed-length spectral representations, then applies a (data-dependent) spectral filter to the frequency-transformed time-series. We show that, by virtue of the DFT analytic properties, the Jacobian determinants and inverse mapping for the Fourier flow can be computed efficiently in linearithmic time, without imposing explicit structural constraints as in existing flows such as NICE (Dinh et al. (2014)), RealNVP (Dinh et al. (2016)) and GLOW (Kingma & Dhariwal (2018)). Experiments show that Fourier flows perform competitively compared to state-of-the-art baselines.

----

## [355] DOP: Off-Policy Multi-Agent Decomposed Policy Gradients

**Authors**: *Yihan Wang, Beining Han, Tonghan Wang, Heng Dong, Chongjie Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6FqKiVAdI3Y](https://openreview.net/forum?id=6FqKiVAdI3Y)

**Abstract**:

Multi-agent policy gradient (MAPG) methods recently witness vigorous progress. However, there is a significant performance discrepancy between MAPG methods and state-of-the-art multi-agent value-based approaches. In this paper, we investigate causes that hinder the performance of MAPG algorithms and present a multi-agent decomposed policy gradient method (DOP). This method introduces the idea of value function decomposition into the multi-agent actor-critic framework. Based on this idea, DOP supports efficient off-policy learning and addresses the issue of centralized-decentralized mismatch and credit assignment in both discrete and continuous action spaces. We formally show that DOP critics have sufficient representational capability to guarantee convergence. In addition, empirical evaluations on the StarCraft II micromanagement benchmark and multi-agent particle environments demonstrate that DOP outperforms both state-of-the-art value-based and policy-based multi-agent reinforcement learning algorithms. Demonstrative videos are available at https://sites.google.com/view/dop-mapg/.

----

## [356] The Risks of Invariant Risk Minimization

**Authors**: *Elan Rosenfeld, Pradeep Kumar Ravikumar, Andrej Risteski*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=BbNIbVPJ-42](https://openreview.net/forum?id=BbNIbVPJ-42)

**Abstract**:

Invariant Causal Prediction (Peters et al., 2016) is a technique for out-of-distribution generalization which assumes that some aspects of the data distribution vary across the training set but that the underlying causal mechanisms remain constant. Recently, Arjovsky et al. (2019) proposed Invariant Risk Minimization (IRM), an objective based on this idea for learning deep, invariant features of data which are a complex function of latent variables; many alternatives have subsequently been suggested.  However, formal guarantees for all of these works are severely lacking.  In this paper,  we present the first analysis of classification under the IRM objective—as well as these recently proposed alternatives—under a fairly natural and general model. In the linear case, we show simple conditions under which the optimal solution succeeds or, more often, fails to recover the optimal invariant predictor. We furthermore present the very first results in the non-linear regime: we demonstrate that IRM can fail catastrophically unless the test data is sufficiently similar to the training distribution—this is precisely the issue that it was intended to solve. Thus, in this setting we find that IRM and its alternatives fundamentally do not improve over standard Empirical Risk Minimization.

----

## [357] DynaTune: Dynamic Tensor Program Optimization in Deep Neural Network Compilation

**Authors**: *Minjia Zhang, Menghao Li, Chi Wang, Mingqin Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=GTGb3M_KcUl](https://openreview.net/forum?id=GTGb3M_KcUl)

**Abstract**:

Recently, the DL compiler, together with Learning to Compile has proven to be a powerful technique for optimizing deep learning models. However, existing methods focus on accelerating the convergence speed of the individual tensor operator rather than the convergence speed of the entire model, which results in long optimization time to obtain a desired latency.

In this paper, we present a new method called DynaTune, which provides significantly faster convergence speed to optimize a DNN model. In particular, we consider a Multi-Armed Bandit (MAB) model for the tensor program optimization problem. We use UCB to handle the decision-making of time-slot-based optimization, and we devise a Bayesian belief model that allows predicting the potential performance gain of each operator with uncertainty quantification, which guides the optimization process. We evaluate and compare DynaTune with the state-of-the-art DL compiler. The experiment results show that DynaTune is 1.2--2.4 times faster to achieve the same optimization quality for a range of models across different hardware architectures.

----

## [358] Bag of Tricks for Adversarial Training

**Authors**: *Tianyu Pang, Xiao Yang, Yinpeng Dong, Hang Su, Jun Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Xb8xvrtB8Ce](https://openreview.net/forum?id=Xb8xvrtB8Ce)

**Abstract**:

Adversarial training (AT) is one of the most effective strategies for promoting model robustness. However, recent benchmarks show that most of the proposed improvements on AT are less effective than simply early stopping the training procedure. This counter-intuitive fact motivates us to investigate the implementation details of tens of AT methods. Surprisingly, we find that the basic settings (e.g., weight decay, training schedule, etc.) used in these methods are highly inconsistent. In this work, we provide comprehensive evaluations on CIFAR-10, focusing on the effects of mostly overlooked training tricks and hyperparameters for adversarially trained models. Our empirical observations suggest that adversarial robustness is much more sensitive to some basic training settings than we thought. For example, a slightly different value of weight decay can reduce the model robust accuracy by more than $7\%$, which is probable to override the potential promotion induced by the proposed methods. We conclude a baseline training setting and re-implement previous defenses to achieve new state-of-the-art results. These facts also appeal to more concerns on the overlooked confounders when benchmarking defenses.

----

## [359] Learning with Instance-Dependent Label Noise: A Sample Sieve Approach

**Authors**: *Hao Cheng, Zhaowei Zhu, Xingyu Li, Yifei Gong, Xing Sun, Yang Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=2VXyy9mIyU3](https://openreview.net/forum?id=2VXyy9mIyU3)

**Abstract**:

Human-annotated labels are often prone to noise, and the presence of such noise will degrade the performance of the resulting deep neural network (DNN) models. Much of the literature (with several recent exceptions) of learning with noisy labels focuses on the case when the label noise is independent of features. Practically, annotations errors tend to be instance-dependent and often depend on the difficulty levels of recognizing a certain task. Applying existing results from instance-independent settings would require a significant amount of estimation of noise rates. Therefore, providing theoretically rigorous solutions for learning with instance-dependent label noise remains a challenge. In this paper, we propose CORES$^{2}$ (COnfidence REgularized Sample Sieve), which progressively sieves out corrupted examples. The implementation of CORES$^{2}$  does not require specifying noise rates and yet we are able to provide theoretical guarantees of CORES$^{2}$ in filtering out the corrupted examples. This high-quality sample sieve allows us to treat clean examples and the corrupted ones separately in training a DNN solution, and such a separation is shown to be advantageous in the instance-dependent noise setting. We demonstrate the performance of CORES$^{2}$ on CIFAR10 and CIFAR100 datasets with synthetic instance-dependent label noise and Clothing1M with real-world human noise. As of independent interests, our sample sieve provides a generic machinery for anatomizing noisy datasets and provides a flexible interface for various robust training techniques to further improve the performance. Code is available at https://github.com/UCSC-REAL/cores.

----

## [360] Efficient Reinforcement Learning in Factored MDPs with Application to Constrained RL

**Authors**: *Xiaoyu Chen, Jiachen Hu, Lihong Li, Liwei Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=fmtSg8591Q](https://openreview.net/forum?id=fmtSg8591Q)

**Abstract**:

Reinforcement learning (RL) in episodic, factored Markov decision processes (FMDPs) is studied. We propose an algorithm called FMDP-BF, which leverages the factorization structure of FMDP.  The regret of FMDP-BF is shown to be exponentially smaller than that of optimal algorithms designed for non-factored MDPs, and improves on the best previous result for FMDPs~\citep{osband2014near} by a factor of $\sqrt{nH|\mathcal{S}_i|}$, where $|\mathcal{S}_i|$ is the cardinality of the factored state subspace, $H$ is the planning horizon and $n$ is the number of factored transition. To show the optimality of our bounds, we also provide a lower bound for FMDP, which indicates that our algorithm is near-optimal w.r.t. timestep $T$, horizon $H$ and factored state-action subspace cardinality. Finally, as an application, we study a new formulation of constrained RL, known as RL with knapsack constraints (RLwK), and provides the first sample-efficient algorithm based on FMDP-BF.

----

## [361] Unbiased Teacher for Semi-Supervised Object Detection

**Authors**: *Yen-Cheng Liu, Chih-Yao Ma, Zijian He, Chia-Wen Kuo, Kan Chen, Peizhao Zhang, Bichen Wu, Zsolt Kira, Peter Vajda*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MJIve1zgR_](https://openreview.net/forum?id=MJIve1zgR_)

**Abstract**:

Semi-supervised learning, i.e., training networks with both labeled and unlabeled data, has made significant progress recently. However, existing works have primarily focused on image classification tasks and neglected object detection which requires more annotation effort. In this work, we revisit the Semi-Supervised Object Detection (SS-OD) and identify the pseudo-labeling bias issue in SS-OD. To address this, we introduce Unbiased Teacher, a simple yet effective approach that jointly trains a student and a gradually progressing teacher in a mutually-beneficial manner. Together with a class-balance loss to downweight overly confident pseudo-labels, Unbiased Teacher consistently improved state-of-the-art methods by significant margins on COCO-standard, COCO-additional, and VOC datasets. Specifically, Unbiased Teacher achieves 6.8 absolute mAP improvements against state-of-the-art method when using 1% of labeled data on MS-COCO, achieves around 10 mAP improvements against the supervised baseline when using only 0.5, 1, 2% of labeled data on MS-COCO.

----

## [362] Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks

**Authors**: *Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, Xingjun Ma*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9l0K4OM-oXE](https://openreview.net/forum?id=9l0K4OM-oXE)

**Abstract**:

Deep neural networks (DNNs) are known vulnerable to backdoor attacks, a training time attack that injects a trigger pattern into a small proportion of training data so as to control the model's prediction at the test time. Backdoor attacks are notably dangerous since they do not affect the model's performance on clean examples, yet can fool the model to make the incorrect prediction whenever the trigger pattern appears during testing. In this paper, we propose a novel defense framework Neural Attention Distillation (NAD) to erase backdoor triggers from backdoored DNNs. NAD utilizes a teacher network to guide the finetuning of the backdoored student network on a small clean subset of data such that the intermediate-layer attention of the student network aligns with that of the teacher network. The teacher network can be obtained by an independent finetuning process on the same clean subset. We empirically show, against 6 state-of-the-art backdoor attacks,  NAD can effectively erase the backdoor triggers using only 5\% clean training data without causing obvious performance degradation on clean examples. Our code is available at https://github.com/bboylyg/NAD.

----

## [363] Contrastive Learning with Adversarial Perturbations for Conditional Text Generation

**Authors**: *Seanie Lee, Dong Bok Lee, Sung Ju Hwang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Wga_hrCa3P3](https://openreview.net/forum?id=Wga_hrCa3P3)

**Abstract**:

Recently, sequence-to-sequence (seq2seq) models with the Transformer architecture have achieved remarkable performance on various conditional text generation tasks, such as machine translation. However, most of them are trained with teacher forcing with the ground truth label given at each time step, without being exposed to incorrectly generated tokens during training, which hurts its generalization to unseen inputs, that is known as the "exposure bias" problem. In this work, we propose to solve the conditional text generation problem by contrasting positive pairs with negative pairs, such that the model is exposed to various valid or incorrect perturbations of the inputs, for improved generalization. However, training the model with naïve contrastive learning framework using random non-target sequences as negative examples is suboptimal, since they are easily distinguishable from the correct output, especially so with models pretrained with large text corpora. Also, generating positive examples requires domain-specific augmentation heuristics which may not generalize over diverse domains. To tackle this problem, we propose a principled method to generate positive and negative samples for contrastive learning of seq2seq models. Specifically, we generate negative examples by adding small perturbations to the input sequence to minimize its conditional likelihood, and positive examples by adding  large perturbations while enforcing it to have a high conditional likelihood. Such `"hard'' positive and negative pairs generated using our method guides the model to better distinguish correct outputs from incorrect ones. We empirically show that our proposed method significantly improves the generalization of the seq2seq on three text generation tasks --- machine translation, text summarization, and question generation.

----

## [364] When Optimizing f-Divergence is Robust with Label Noise

**Authors**: *Jiaheng Wei, Yang Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=WesiCoRVQ15](https://openreview.net/forum?id=WesiCoRVQ15)

**Abstract**:

We show when maximizing a properly defined $f$-divergence measure with respect to a classifier's predictions and the supervised labels is robust with label noise. Leveraging its variational form, we derive a nice decoupling property for a family of $f$-divergence measures when label noise presents, where the divergence is shown to be a linear combination of the variational difference defined on the clean distribution and a bias term introduced due to the noise. The above derivation helps us analyze the robustness of different $f$-divergence functions. With established robustness, this family of $f$-divergence functions arises as useful metrics for the problem of learning with noisy labels, which do not require the specification of the labels' noise rate. When they are possibly not robust, we propose fixes to make them so. In addition to the analytical results, we present thorough experimental evidence. Our code is available at https://github.com/UCSC-REAL/Robust-f-divergence-measures.

----

## [365] Conditional Generative Modeling via Learning the Latent Space

**Authors**: *Sameera Ramasinghe, Kanchana Nisal Ranasinghe, Salman H. Khan, Nick Barnes, Stephen Gould*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=VJnrYcnRc6](https://openreview.net/forum?id=VJnrYcnRc6)

**Abstract**:

Although deep learning has achieved appealing results on several machine learning tasks, most of the models are deterministic at inference, limiting their application to single-modal settings. We propose a novel general-purpose framework for conditional generation in multimodal spaces, that uses latent variables to model generalizable learning patterns while minimizing a family of regression cost functions. At inference, the latent variables are optimized to find solutions corresponding to multiple output modes.  Compared to existing generative solutions, our approach demonstrates faster and more stable convergence, and can learn better representations for downstream tasks. Importantly, it provides a simple generic model that can perform better than highly engineered pipelines tailored using domain expertise on a variety of tasks, while generating diverse outputs. Code available at https://github.com/samgregoost/cGML.

----

## [366] Text Generation by Learning from Demonstrations

**Authors**: *Richard Yuanzhe Pang, He He*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=RovX-uQ1Hua](https://openreview.net/forum?id=RovX-uQ1Hua)

**Abstract**:

Current approaches to text generation largely rely on autoregressive models and maximum likelihood estimation. This paradigm leads to (i) diverse but low-quality samples due to mismatched learning objective and evaluation metric (likelihood vs. quality) and (ii) exposure bias due to mismatched history distributions (gold vs. model-generated). To alleviate these problems, we frame text generation as an offline reinforcement learning (RL) problem with expert demonstrations (i.e., the reference), where the goal is to maximize quality given model-generated histories. We propose GOLD (generation by off-policy learning from demonstrations): an easy-to-optimize algorithm that learns from the demonstrations by importance weighting. Intuitively, GOLD upweights confident tokens and downweights unconfident ones in the reference during training, avoiding optimization issues faced by prior RL approaches that rely on online data collection. According to both automatic and human evaluation, models trained by GOLD outperform those trained by MLE and policy gradient on summarization, question generation, and machine translation. Further, our models are less sensitive to decoding algorithms and alleviate exposure bias.

----

## [367] Learning Long-term Visual Dynamics with Region Proposal Interaction Networks

**Authors**: *Haozhi Qi, Xiaolong Wang, Deepak Pathak, Yi Ma, Jitendra Malik*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_X_4Akcd8Re](https://openreview.net/forum?id=_X_4Akcd8Re)

**Abstract**:

Learning long-term dynamics models is the key to understanding physical common sense. Most existing approaches on learning dynamics from visual input sidestep long-term predictions by resorting to rapid re-planning with short-term models. This not only requires such models to be super accurate but also limits them only to tasks where an agent can continuously obtain feedback and take action at each step until completion. In this paper, we aim to leverage the ideas from success stories in visual recognition tasks to build object representations that can capture inter-object and object-environment interactions over a long range. To this end, we propose Region Proposal Interaction Networks (RPIN), which reason about each object's trajectory in a latent region-proposal feature space. Thanks to the simple yet effective object representation, our approach outperforms prior methods by a significant margin both in terms of prediction quality and their ability to plan for downstream tasks, and also generalize well to novel environments. Code, pre-trained models, and more visualization results are available at https://haozhi.io/RPIN.

----

## [368] ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations

**Authors**: *Rishabh Tiwari, Udbhav Bamba, Arnav Chavan, Deepak K. Gupta*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xCxXwTzx4L1](https://openreview.net/forum?id=xCxXwTzx4L1)

**Abstract**:

Structured pruning methods are among the effective strategies for extracting small resource-efficient convolutional neural networks from their dense counterparts with minimal loss in accuracy. However, most existing methods still suffer from one or more limitations, that include 1) the need for training the dense model from scratch with pruning-related parameters embedded in the architecture, 2) requiring model-specific hyperparameter settings, 3) inability to include budget-related constraint in the training process, and 4) instability under scenarios of extreme pruning. In this paper, we present ChipNet, a deterministic pruning strategy that employs continuous Heaviside function and a novel crispness loss to identify a highly sparse network out of an existing dense network. Our choice of continuous Heaviside function is inspired by the field of design optimization, where the material distribution task is posed as a continuous optimization problem, but only discrete values (0 or 1) are practically feasible and expected as final outcomes. Our approach's flexible design facilitates its use with different choices of budget constraints while maintaining stability for very low target budgets. Experimental results show that ChipNet outperforms state-of-the-art structured pruning methods by remarkable margins of up to 16.1% in terms of accuracy. Further, we show that the masks obtained with ChipNet are transferable across datasets. For certain cases, it was observed that masks transferred from a model trained on feature-rich teacher dataset provide better performance on the student dataset than those obtained by directly pruning on the student data itself.

----

## [369] Learning to Deceive Knowledge Graph Augmented Models via Targeted Perturbation

**Authors**: *Mrigank Raman, Aaron Chan, Siddhant Agarwal, Peifeng Wang, Hansen Wang, Sungchul Kim, Ryan A. Rossi, Handong Zhao, Nedim Lipka, Xiang Ren*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=b7g3_ZMHnT0](https://openreview.net/forum?id=b7g3_ZMHnT0)

**Abstract**:

Knowledge graphs (KGs) have helped neural models improve performance on various knowledge-intensive tasks, like question answering and item recommendation. By using attention over the KG, such KG-augmented models can also "explain" which KG information was most relevant for making a given prediction. In this paper, we question whether these models are really behaving as we expect. We show that, through a reinforcement learning policy (or even simple heuristics), one can produce deceptively perturbed KGs, which maintain the downstream performance of the original KG while significantly deviating from the original KG's semantics and structure. Our findings raise doubts about KG-augmented models' ability to reason about KG information and give sensible explanations.

----

## [370] IEPT: Instance-Level and Episode-Level Pretext Tasks for Few-Shot Learning

**Authors**: *Manli Zhang, Jianhong Zhang, Zhiwu Lu, Tao Xiang, Mingyu Ding, Songfang Huang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xzqLpqRzxLq](https://openreview.net/forum?id=xzqLpqRzxLq)

**Abstract**:

The need of collecting large quantities of labeled training data for each new task has limited the usefulness of deep neural networks. Given data from a set of source tasks, this limitation can be overcome using two transfer learning approaches: few-shot learning (FSL) and self-supervised learning (SSL). The former aims to learn `how to learn' by designing learning episodes using source tasks to simulate the challenge of solving the target new task with few labeled samples. In contrast, the latter exploits an annotation-free pretext task across all source tasks in order to learn generalizable feature representations. In this work, we propose a novel Instance-level and Episode-level Pretext Task (IEPT) framework that seamlessly integrates SSL into FSL. Specifically, given an FSL episode, we first apply geometric transformations to each instance to generate extended episodes. At the instance-level, transformation recognition is performed as per standard SSL. Importantly, at the episode-level, two SSL-FSL hybrid learning objectives are devised: (1) The consistency across the predictions of an FSL classifier from different extended episodes is maximized as an episode-level pretext task. (2) The features extracted from each instance across different episodes are integrated to construct a single FSL classifier for meta-learning. Extensive experiments show that our proposed model (i.e., FSL with IEPT) achieves the new state-of-the-art.

----

## [371] The Role of Momentum Parameters in the Optimal Convergence of Adaptive Polyak's Heavy-ball Methods

**Authors**: *Wei Tao, Sheng Long, Gaowei Wu, Qing Tao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=L7WD8ZdscQ5](https://openreview.net/forum?id=L7WD8ZdscQ5)

**Abstract**:

The adaptive stochastic gradient descent (SGD) with momentum has been widely adopted in deep learning as well as convex optimization. In practice, the last iterate is commonly used as the final solution. However, the available regret analysis and the setting of constant momentum parameters only guarantee the optimal convergence of the averaged solution. In this paper, we fill this theory-practice gap by investigating the convergence of the last iterate (referred to as {\it individual convergence}), which is a more difficult task than convergence analysis of the averaged solution. Specifically, in the constrained convex cases, we prove that the adaptive Polyak's Heavy-ball (HB) method, in which the step size is only updated using the exponential moving average strategy, attains an individual convergence rate of $O(\frac{1}{\sqrt{t}})$, as opposed to that of $O(\frac{\log t}{\sqrt {t}})$ of SGD, where $t$ is the number of iterations. Our new analysis not only shows how the HB momentum and its time-varying weight help us to achieve the acceleration in convex optimization but also gives valuable hints how the momentum parameters should be scheduled in deep learning. Empirical results validate the correctness of our convergence analysis in optimizing convex functions and demonstrate the improved performance of the adaptive HB methods in training deep networks.

----

## [372] Training with Quantization Noise for Extreme Model Compression

**Authors**: *Pierre Stock, Angela Fan, Benjamin Graham, Edouard Grave, Rémi Gribonval, Hervé Jégou, Armand Joulin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dV19Yyi1fS3](https://openreview.net/forum?id=dV19Yyi1fS3)

**Abstract**:

We tackle the problem of producing compact models, maximizing their accuracy for a given model size. A standard solution is to train networks with Quantization Aware Training, where the weights are quantized during training and the gradients approximated with the Straight-Through Estimator. In this paper, we extend this approach to work with extreme compression methods where the approximations introduced by STE are severe. Our proposal is to only quantize a different random subset of weights during each forward, allowing for unbiased gradients to flow through the other weights. Controlling the amount of noise and its form allows for extreme compression rates while maintaining the performance of the original model. As a result we establish new state-of-the-art compromises between accuracy and model size both in natural language processing and image classification. For example, applying our method to state-of-the-art Transformer and ConvNet architectures, we can achieve 82.5% accuracy on MNLI by compressing RoBERTa to 14 MB and 80.0% top-1 accuracy on ImageNet by compressing an EfficientNet-B3 to 3.3 MB.

----

## [373] Adaptive Extra-Gradient Methods for Min-Max Optimization and Games

**Authors**: *Kimon Antonakopoulos, Elena Veronica Belmega, Panayotis Mertikopoulos*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=R0a0kFI3dJx](https://openreview.net/forum?id=R0a0kFI3dJx)

**Abstract**:

We present a new family of min-max optimization algorithms that automatically exploit the geometry of the gradient data observed at earlier iterations to perform more informative extra-gradient steps in later ones.
Thanks to this adaptation mechanism, the proposed method automatically detects whether the problem is smooth or not, without requiring any prior tuning by the optimizer.
As a result, the algorithm simultaneously achieves order-optimal convergence rates, \ie it  converges to an $\varepsilon$-optimal solution within $\mathcal{O}(1/\varepsilon)$ iterations in smooth problems, and within $\mathcal{O}(1/\varepsilon^2)$ iterations in non-smooth ones. Importantly, these guarantees do not require any of the standard boundedness or Lipschitz continuity conditions that are typically assumed in the literature; in particular, they apply even to problems with singularities (such as resource allocation problems and the like). This adaptation is achieved through the use of a geometric apparatus based on Finsler metrics and a suitably chosen mirror-prox template that allows us to derive sharp convergence rates for the methods at hand.

----

## [374] Distilling Knowledge from Reader to Retriever for Question Answering

**Authors**: *Gautier Izacard, Edouard Grave*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NTEz-6wysdb](https://openreview.net/forum?id=NTEz-6wysdb)

**Abstract**:

The task of information retrieval is an important component of many natural language processing systems, such as open domain question answering. While traditional methods were based on hand-crafted features, continuous representations based on neural networks recently obtained competitive results. A challenge of using such methods is to obtain supervised data to train the retriever model, corresponding to pairs of query and support documents. In this paper, we propose a technique to learn retriever models for downstream tasks, inspired by knowledge distillation, and which does not require annotated pairs of query and documents. Our approach leverages attention scores of a reader model, used to solve the task based on retrieved documents, to obtain synthetic labels for the retriever. We evaluate our method on question answering, obtaining state-of-the-art results.

----

## [375] Discovering Diverse Multi-Agent Strategic Behavior via Reward Randomization

**Authors**: *Zhenggang Tang, Chao Yu, Boyuan Chen, Huazhe Xu, Xiaolong Wang, Fei Fang, Simon Shaolei Du, Yu Wang, Yi Wu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lvRTC669EY_](https://openreview.net/forum?id=lvRTC669EY_)

**Abstract**:

We propose a simple, general and effective technique, Reward Randomization for discovering diverse strategic policies in complex multi-agent games. Combining reward randomization and policy gradient, we derive a new algorithm, Reward-Randomized Policy Gradient (RPG). RPG is able to discover a set of multiple distinctive human-interpretable strategies in challenging temporal trust dilemmas, including grid-world games and a real-world game Agar.io, where multiple equilibria exist but standard multi-agent policy gradient algorithms always converge to a fixed one with a sub-optimal payoff for every player even using state-of-the-art exploration techniques. Furthermore, with the set of diverse strategies from RPG, we can (1) achieve higher payoffs by fine-tuning the best policy from the set; and (2) obtain an adaptive agent by using this set of strategies as its training opponents.

----

## [376] not-MIWAE: Deep Generative Modelling with Missing not at Random Data

**Authors**: *Niels Bruun Ipsen, Pierre-Alexandre Mattei, Jes Frellsen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tu29GQT0JFy](https://openreview.net/forum?id=tu29GQT0JFy)

**Abstract**:

When a missing process depends on the missing values themselves, it needs to be explicitly modelled and taken into account while doing likelihood-based inference. We present an approach for building and fitting deep latent variable models (DLVMs) in cases where the missing process is dependent on the missing data. Specifically, a deep neural network enables us to flexibly model the conditional distribution of the missingness pattern given the data. This allows for incorporating prior information about the type of missingness (e.g.~self-censoring) into the model. Our inference technique, based on importance-weighted variational inference, involves maximising a lower bound of the joint likelihood. Stochastic gradients of the bound are obtained by using the reparameterisation trick both in latent space and data space. We show on various kinds of data sets and missingness patterns that explicitly modelling the missing process can be invaluable.

----

## [377] IDF++: Analyzing and Improving Integer Discrete Flows for Lossless Compression

**Authors**: *Rianne van den Berg, Alexey A. Gritsenko, Mostafa Dehghani, Casper Kaae Sønderby, Tim Salimans*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MBOyiNnYthd](https://openreview.net/forum?id=MBOyiNnYthd)

**Abstract**:

In this paper we analyse and improve integer discrete flows for lossless compression. Integer discrete flows are a recently proposed class of models that learn invertible transformations for integer-valued random variables. Their discrete nature makes them particularly suitable for lossless compression with entropy coding schemes. We start by investigating a recent theoretical claim that states that invertible flows for discrete random variables are less flexible than their continuous counterparts. We demonstrate with a proof that this claim does not hold for integer discrete flows due to the embedding of data with finite support into the countably infinite integer lattice. Furthermore, we zoom in on the effect of gradient bias due to the straight-through estimator in integer discrete flows, and demonstrate that its influence is highly dependent on architecture choices and less prominent than previously thought. Finally, we show how different architecture modifications improve the performance of this model class for lossless compression, and that they also enable more efficient compression: a model with half the number of flow layers performs on par with or better than the original integer discrete flow model.

----

## [378] Explaining by Imitating: Understanding Decisions by Interpretable Policy Learning

**Authors**: *Alihan Hüyük, Daniel Jarrett, Cem Tekin, Mihaela van der Schaar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=unI5ucw_Jk](https://openreview.net/forum?id=unI5ucw_Jk)

**Abstract**:

Understanding human behavior from observed data is critical for transparency and accountability in decision-making. Consider real-world settings such as healthcare, in which modeling a decision-maker’s policy is challenging—with no access to underlying states, no knowledge of environment dynamics, and no allowance for live experimentation. We desire learning a data-driven representation of decision- making behavior that (1) inheres transparency by design, (2) accommodates partial observability, and (3) operates completely offline. To satisfy these key criteria, we propose a novel model-based Bayesian method for interpretable policy learning (“Interpole”) that jointly estimates an agent’s (possibly biased) belief-update process together with their (possibly suboptimal) belief-action mapping. Through experiments on both simulated and real-world data for the problem of Alzheimer’s disease diagnosis, we illustrate the potential of our approach as an investigative device for auditing, quantifying, and understanding human decision-making behavior.

----

## [379] Learning with AMIGo: Adversarially Motivated Intrinsic Goals

**Authors**: *Andres Campero, Roberta Raileanu, Heinrich Küttler, Joshua B. Tenenbaum, Tim Rocktäschel, Edward Grefenstette*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ETBc_MIMgoX](https://openreview.net/forum?id=ETBc_MIMgoX)

**Abstract**:

A key challenge for reinforcement learning (RL) consists of learning in environments with sparse extrinsic rewards. In contrast to current RL methods, humans are able to learn new skills with little or no reward by using various forms of intrinsic motivation. We propose AMIGo, a novel agent incorporating -- as form of meta-learning -- a goal-generating teacher that proposes Adversarially Motivated Intrinsic Goals to train a goal-conditioned "student" policy in the absence of (or alongside) environment reward. Specifically, through a simple but effective "constructively adversarial" objective, the teacher learns to propose increasingly challenging -- yet achievable -- goals that allow the student to learn general skills for acting in a new environment, independent of the task to be solved. We show that our method generates a natural curriculum of self-proposed goals which ultimately allows the agent to solve challenging procedurally-generated tasks where other forms of intrinsic motivation and state-of-the-art RL methods fail.

----

## [380] Incorporating Symmetry into Deep Dynamics Models for Improved Generalization

**Authors**: *Rui Wang, Robin Walters, Rose Yu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=wta_8Hx2KD](https://openreview.net/forum?id=wta_8Hx2KD)

**Abstract**:

Recent work has shown deep learning can accelerate the prediction of physical dynamics relative to numerical solvers. However, limited physical accuracy and an inability to generalize under distributional shift limit its applicability to the real world. We propose to improve accuracy and generalization by incorporating symmetries into convolutional neural networks. Specifically, we employ a variety of methods each tailored to enforce a different symmetry. Our models are both theoretically and experimentally robust to distributional shift by symmetry group transformations and enjoy favorable sample complexity. We demonstrate the advantage of our approach on a variety of physical dynamics including Rayleigh–Bénard convection and real-world ocean currents and temperatures. Compare with image or text applications, our work is a significant step towards applying equivariant neural networks to high-dimensional systems with complex dynamics.

----

## [381] CaPC Learning: Confidential and Private Collaborative Learning

**Authors**: *Christopher A. Choquette-Choo, Natalie Dullerud, Adam Dziedzic, Yunxiang Zhang, Somesh Jha, Nicolas Papernot, Xiao Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=h2EbJ4_wMVq](https://openreview.net/forum?id=h2EbJ4_wMVq)

**Abstract**:

Machine learning benefits from large training datasets, which may not always be possible to collect by any single entity, especially when using privacy-sensitive data. In many contexts, such as healthcare and finance, separate parties may wish to collaborate and learn from each other's data but are prevented from doing so due to privacy regulations. Some regulations prevent explicit sharing of data between parties by joining datasets in a central location (confidentiality). Others also limit implicit sharing of data, e.g., through model predictions (privacy). There is currently no method that enables machine learning in such a setting, where both confidentiality and privacy need to be preserved, to prevent both explicit and implicit sharing of data. Federated learning only provides confidentiality, not privacy, since gradients shared still contain private information. Differentially private learning assumes unreasonably large datasets. Furthermore, both of these learning paradigms produce a central model whose architecture was previously agreed upon by all parties rather than enabling collaborative learning where each party learns and improves their own local model. We introduce Confidential and Private Collaborative (CaPC) learning, the first method provably achieving both confidentiality and privacy in a collaborative setting. We leverage secure multi-party computation (MPC), homomorphic encryption (HE), and other techniques in combination with privately aggregated teacher models. We demonstrate how CaPC allows participants to collaborate without having to explicitly join their training sets or train a central model. Each party is able to improve the accuracy and fairness of their model, even in settings where each party has a model that performs well on their own dataset or when datasets are not IID and model architectures are heterogeneous across parties.

----

## [382] Heating up decision boundaries: isocapacitory saturation, adversarial scenarios and generalization bounds

**Authors**: *Bogdan Georgiev, Lukas Franken, Mayukh Mukherjee*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=UwGY2qjqoLD](https://openreview.net/forum?id=UwGY2qjqoLD)

**Abstract**:

In the present work we study classifiers' decision boundaries via Brownian motion processes in ambient data space and associated probabilistic techniques. Intuitively, our ideas correspond to placing a heat source at the decision boundary and observing how effectively the sample points warm up. We are largely motivated by the search for a soft measure that sheds further light on the decision boundary's geometry. En route, we  bridge aspects of potential theory and geometric analysis (Maz'ya 2011, Grigor'Yan and Saloff-Coste 2002) with active fields of ML research such as adversarial examples and generalization bounds. First, we focus on the geometric behavior of decision boundaries in the light of adversarial attack/defense mechanisms. Experimentally, we observe a certain capacitory trend over different adversarial defense strategies: decision boundaries locally become flatter as measured by isoperimetric inequalities (Ford et al 2019); however, our more sensitive heat-diffusion metrics  extend this analysis and further reveal that some non-trivial geometry invisible to plain distance-based methods is still preserved. Intuitively, we provide evidence that the decision boundaries nevertheless retain many persistent "wiggly and fuzzy" regions on a finer scale.
Second, we show how Brownian hitting probabilities translate to soft generalization bounds which are in turn connected to compression and noise stability (Arora et al 2018), and these bounds are significantly stronger if the decision boundary has controlled geometric features.

----

## [383] A PAC-Bayesian Approach to Generalization Bounds for Graph Neural Networks

**Authors**: *Renjie Liao, Raquel Urtasun, Richard S. Zemel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TR-Nj6nFx42](https://openreview.net/forum?id=TR-Nj6nFx42)

**Abstract**:

In this paper, we derive generalization bounds for two primary classes of graph neural networks (GNNs), namely graph convolutional networks (GCNs) and message passing GNNs (MPGNNs), via a PAC-Bayesian approach. Our result reveals that the maximum node degree and the spectral norm of the weights govern the generalization bounds of both models. We also show that our bound for GCNs is a natural generalization of the results developed in \citep{neyshabur2017pac} for fully-connected and convolutional neural networks. For MPGNNs, our PAC-Bayes bound improves over the Rademacher complexity based bound \citep{garg2020generalization}, showing a tighter dependency on the maximum node degree and the maximum hidden dimension. The key ingredients of our proofs are a perturbation analysis of GNNs and the generalization of PAC-Bayes analysis to non-homogeneous GNNs. We perform an empirical study on several synthetic and real-world graph datasets and verify that our PAC-Bayes bound is tighter than others.

----

## [384] Clairvoyance: A Pipeline Toolkit for Medical Time Series

**Authors**: *Daniel Jarrett, Jinsung Yoon, Ioana Bica, Zhaozhi Qian, Ari Ercole, Mihaela van der Schaar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xnC8YwKUE3k](https://openreview.net/forum?id=xnC8YwKUE3k)

**Abstract**:

Time-series learning is the bread and butter of data-driven *clinical decision support*, and the recent explosion in ML research has demonstrated great potential in various healthcare settings. At the same time, medical time-series problems in the wild are challenging due to their highly *composite* nature: They entail design choices and interactions among components that preprocess data, impute missing values, select features, issue predictions, estimate uncertainty, and interpret models. Despite exponential growth in electronic patient data, there is a remarkable gap between the potential and realized utilization of ML for clinical research and decision support. In particular, orchestrating a real-world project lifecycle poses challenges in engineering (i.e. hard to build), evaluation (i.e. hard to assess), and efficiency (i.e. hard to optimize). Designed to address these issues simultaneously, Clairvoyance proposes a unified, end-to-end, autoML-friendly pipeline that serves as a (i) software toolkit, (ii) empirical standard, and (iii) interface for optimization. Our ultimate goal lies in facilitating transparent and reproducible experimentation with complex inference workflows, providing integrated pathways for (1) personalized prediction, (2) treatment-effect estimation, and (3) information acquisition. Through illustrative examples on real-world data in outpatient, general wards, and intensive-care settings, we illustrate the applicability of the pipeline paradigm on core tasks in the healthcare journey. To the best of our knowledge, Clairvoyance is the first to demonstrate viability of a comprehensive and automatable pipeline for clinical time-series ML.

----

## [385] Self-supervised Representation Learning with Relative Predictive Coding

**Authors**: *Yao-Hung Hubert Tsai, Martin Q. Ma, Muqiao Yang, Han Zhao, Louis-Philippe Morency, Ruslan Salakhutdinov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=068E_JSq9O](https://openreview.net/forum?id=068E_JSq9O)

**Abstract**:

This paper introduces Relative Predictive Coding (RPC), a new contrastive representation learning objective that maintains a good balance among training stability, minibatch size sensitivity, and downstream task performance. The key to the success of RPC is two-fold. First, RPC introduces the relative parameters to regularize the objective for boundedness and low variance. Second, RPC contains no logarithm and exponential score functions, which are the main cause of training instability in prior contrastive objectives. We empirically verify the effectiveness of RPC on benchmark vision and speech self-supervised learning tasks. Lastly, we relate RPC with mutual information (MI) estimation, showing RPC can be used to estimate MI with low variance.

----

## [386] Offline Model-Based Optimization via Normalized Maximum Likelihood Estimation

**Authors**: *Justin Fu, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=FmMKSO4e8JK](https://openreview.net/forum?id=FmMKSO4e8JK)

**Abstract**:

In this work we consider data-driven optimization problems where one must maximize a function given only queries at a fixed set of points. This problem setting emerges in many domains where function evaluation is a complex and expensive process, such as in the design of materials, vehicles, or neural network architectures. Because the available data typically only covers a small manifold of the possible space of inputs, a principal challenge is to be able to construct algorithms that can reason about uncertainty and out-of-distribution values, since a naive optimizer can easily exploit an estimated model to return adversarial inputs. We propose to tackle the MBO problem by leveraging the normalized maximum-likelihood (NML) estimator, which provides a principled approach to handling uncertainty and out-of-distribution inputs. While in the standard formulation NML is intractable, we propose a tractable approximation that allows us to scale our method to high-capacity neural network models. We demonstrate that our method can effectively optimize high-dimensional design problems in a variety of disciplines such as chemistry, biology, and materials engineering.

----

## [387] On the Impossibility of Global Convergence in Multi-Loss Optimization

**Authors**: *Alistair Letcher*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NQbnPjPYaG6](https://openreview.net/forum?id=NQbnPjPYaG6)

**Abstract**:

Under mild regularity conditions, gradient-based methods converge globally to a critical point in the single-loss setting. This is known to break down for vanilla gradient descent when moving to multi-loss optimization, but can we hope to build some algorithm with global guarantees? We negatively resolve this open problem by proving that desirable convergence properties cannot simultaneously hold for any algorithm. Our result has more to do with the existence of games with no satisfactory outcomes, than with algorithms per se. More explicitly we construct a two-player game with zero-sum interactions whose losses are both coercive and analytic, but whose only simultaneous critical point is a strict maximum. Any 'reasonable' algorithm, defined to avoid strict maxima, will therefore fail to converge. This is fundamentally different from single losses, where coercivity implies existence of a global minimum.  Moreover, we prove that a wide range of existing gradient-based methods almost surely have bounded but non-convergent iterates in a constructed zero-sum game for suitably small learning rates. It nonetheless remains an open question whether such behavior can arise in high-dimensional games of interest to ML practitioners, such as GANs or multi-agent RL.

----

## [388] A Block Minifloat Representation for Training Deep Neural Networks

**Authors**: *Sean Fox, Seyedramin Rasoulinezhad, Julian Faraone, David Boland, Philip H. W. Leong*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6zaTwpNSsQ2](https://openreview.net/forum?id=6zaTwpNSsQ2)

**Abstract**:

Training Deep Neural Networks (DNN) with high efficiency can be difficult to achieve with native floating-point representations and commercially available hardware. Specialized arithmetic with custom acceleration offers perhaps the most promising alternative. Ongoing research is trending towards narrow floating-point representations, called minifloats, that pack more operations for a given silicon area and consume less power. In this paper, we introduce Block Minifloat (BM), a new spectrum of minifloat formats capable of training DNNs end-to-end with only 4-8 bit weight, activation and gradient tensors. While standard floating-point representations have two degrees of freedom, via the exponent and mantissa, BM exposes the exponent bias as an additional field for optimization. Crucially, this enables training with fewer exponent bits, yielding dense integer-like hardware for fused multiply-add (FMA) operations. For ResNet trained on ImageNet, 6-bit BM achieves almost no degradation in floating-point accuracy with FMA units that are $4.1\times(23.9\times)$ smaller and consume $2.3\times(16.1\times)$ less energy than FP8 (FP32). Furthermore, our 8-bit BM format matches floating-point accuracy while delivering a higher computational density and faster expected training times.

----

## [389] Selectivity considered harmful: evaluating the causal impact of class selectivity in DNNs

**Authors**: *Matthew L. Leavitt, Ari S. Morcos*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8nl0k08uMi](https://openreview.net/forum?id=8nl0k08uMi)

**Abstract**:

The properties of individual neurons are often analyzed in order to understand the biological and artificial neural networks in which they're embedded. Class selectivity—typically defined as how different a neuron's responses are across different classes of stimuli or data samples—is commonly used for this purpose. However, it remains an open question whether it is necessary and/or sufficient for deep neural networks (DNNs) to learn class selectivity in individual units. We investigated the causal impact of class selectivity on network function by directly regularizing for or against class selectivity. Using this regularizer to reduce class selectivity across units in convolutional neural networks increased test accuracy by over 2% in ResNet18 and 1% in ResNet50 trained on Tiny ImageNet. For ResNet20 trained on CIFAR10 we could reduce class selectivity by a factor of 2.5 with no impact on test accuracy, and reduce it nearly to zero with only a small (~2%) drop in test accuracy. In contrast, regularizing to increase class selectivity significantly decreased test accuracy across all models and datasets. These results indicate that class selectivity in individual units is neither sufficient nor strictly necessary, and can even impair DNN performance. They also encourage caution when focusing on the properties of single units as representative of the mechanisms by which DNNs function.

----

## [390] Discrete Graph Structure Learning for Forecasting Multiple Time Series

**Authors**: *Chao Shang, Jie Chen, Jinbo Bi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=WEHSlH5mOk](https://openreview.net/forum?id=WEHSlH5mOk)

**Abstract**:

Time series forecasting is an extensively studied subject in statistics, economics, and computer science. Exploration of the correlation and causation among the variables in a multivariate time series shows promise in enhancing the performance of a time series model. When using deep neural networks as forecasting models, we hypothesize that exploiting the pairwise information among multiple (multivariate) time series also improves their forecast. If an explicit graph structure is known, graph neural networks (GNNs) have been demonstrated as powerful tools to exploit the structure. In this work, we propose learning the structure simultaneously with the GNN if the graph is unknown. We cast the problem as learning a probabilistic graph model through optimizing the mean performance over the graph distribution. The distribution is parameterized by a neural network so that discrete graphs can be sampled differentiably through reparameterization. Empirical evaluations show that our method is simpler, more efficient, and better performing than a recently proposed bilevel learning approach for graph structure learning, as well as a broad array of forecasting models, either deep or non-deep learning based, and graph or non-graph based.

----

## [391] Contrastive Learning with Hard Negative Samples

**Authors**: *Joshua David Robinson, Ching-Yao Chuang, Suvrit Sra, Stefanie Jegelka*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=CR1XOQ0UTh-](https://openreview.net/forum?id=CR1XOQ0UTh-)

**Abstract**:

We consider the question: how can you sample good negative examples for contrastive learning? We argue that, as with metric learning, learning contrastive representations benefits from hard negative samples (i.e., points that are difficult to distinguish from an anchor point). The key challenge toward using hard negatives is that contrastive methods must remain unsupervised, making it infeasible to adopt existing negative sampling strategies that use label information. In response, we develop a new class of unsupervised methods for selecting hard negative samples where the user can control the amount of hardness. A limiting case of this sampling results in a representation that tightly clusters each class, and pushes different classes as far apart as possible. The proposed method improves downstream performance across multiple modalities, requires only few additional lines of code to implement, and introduces no computational overhead.

----

## [392] Intraclass clustering: an implicit learning ability that regularizes DNNs

**Authors**: *Simon Carbonnelle, Christophe De Vleeschouwer*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tqOvYpjPax2](https://openreview.net/forum?id=tqOvYpjPax2)

**Abstract**:

Several works have shown that the regularization mechanisms underlying deep neural networks' generalization performances are still poorly understood. In this paper, we hypothesize that deep neural networks are regularized through their ability to extract meaningful clusters among the samples of a class. This constitutes an implicit form of regularization, as no explicit training mechanisms or supervision target such behaviour. To support our hypothesis, we design four different measures of intraclass clustering, based on the neuron- and layer-level representations of the training data. We then show that these measures constitute accurate predictors of generalization performance across variations of a large set of hyperparameters (learning rate, batch size, optimizer, weight decay, dropout rate, data augmentation, network depth and width).

----

## [393] Sliced Kernelized Stein Discrepancy

**Authors**: *Wenbo Gong, Yingzhen Li, José Miguel Hernández-Lobato*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=t0TaKv0Gx6Z](https://openreview.net/forum?id=t0TaKv0Gx6Z)

**Abstract**:

Kernelized Stein discrepancy (KSD), though being extensively used in goodness-of-fit tests and model learning, suffers from the curse-of-dimensionality. We address this issue by proposing the sliced Stein discrepancy and its scalable and kernelized variants, which employs kernel-based test functions defined on the optimal one-dimensional projections. When applied to goodness-of-fit tests, extensive experiments show the proposed discrepancy significantly outperforms KSD and various baselines in high dimensions. For model learning, we show its advantages by training an independent component analysis when compared with existing Stein discrepancy baselines. We further propose a novel particle inference method called sliced Stein variational gradient descent (S-SVGD) which alleviates the mode-collapse issue of SVGD in training variational autoencoders.

----

## [394] Denoising Diffusion Implicit Models

**Authors**: *Jiaming Song, Chenlin Meng, Stefano Ermon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=St1giarCHLP](https://openreview.net/forum?id=St1giarCHLP)

**Abstract**:

Denoising diffusion probabilistic models (DDPMs) have achieved high quality image generation without adversarial training, yet they require simulating a Markov chain for many steps in order to produce a sample. To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same training procedure as DDPMs. In DDPMs, the generative process is defined as the reverse of a particular Markovian diffusion process. We generalize DDPMs via a class of non-Markovian diffusion processes that lead to the same training objective. These non-Markovian processes can correspond to generative processes that are deterministic, giving rise to implicit models that produce high quality samples much faster. We empirically demonstrate that DDIMs can produce high quality samples $10 \times$ to $50 \times$ faster in terms of wall-clock time compared to DDPMs, allow us to trade off computation for sample quality, perform semantically meaningful image interpolation directly in the latent space, and reconstruct observations with very low error.

----

## [395] Hierarchical Reinforcement Learning by Discovering Intrinsic Options

**Authors**: *Jesse Zhang, Haonan Yu, Wei Xu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=r-gPPHEjpmw](https://openreview.net/forum?id=r-gPPHEjpmw)

**Abstract**:

We propose a hierarchical reinforcement learning method, HIDIO, that can learn task-agnostic options in a self-supervised manner while jointly learning to utilize them to solve sparse-reward tasks. Unlike current hierarchical RL approaches that tend to formulate goal-reaching low-level tasks or pre-define ad hoc lower-level policies, HIDIO encourages lower-level option learning that is independent of the task at hand, requiring few assumptions or little knowledge about the task structure. These options are learned through an intrinsic entropy minimization objective conditioned on the option sub-trajectories. The learned options are diverse and task-agnostic. In experiments on sparse-reward robotic manipulation and navigation tasks, HIDIO achieves higher success rates with greater sample efficiency than regular RL baselines and two state-of-the-art hierarchical RL methods. Code at: https://github.com/jesbu1/hidio.

----

## [396] Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval

**Authors**: *Wenhan Xiong, Xiang Lorraine Li, Srini Iyer, Jingfei Du, Patrick S. H. Lewis, William Yang Wang, Yashar Mehdad, Scott Yih, Sebastian Riedel, Douwe Kiela, Barlas Oguz*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=EMHoBG0avc1](https://openreview.net/forum?id=EMHoBG0avc1)

**Abstract**:

We propose a simple and efficient multi-hop dense retrieval approach for answering complex open-domain questions, which achieves state-of-the-art performance on two multi-hop datasets, HotpotQA and multi-evidence FEVER. Contrary to previous work, our method does not require access to any corpus-specific information, such as inter-document hyperlinks or human-annotated entity markers, and can be applied to any unstructured text corpus. Our system also yields a much better efficiency-accuracy trade-off, matching the best published accuracy on HotpotQA while being 10 times faster at inference time.

----

## [397] Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective

**Authors**: *Helong Zhou, Liangchen Song, Jiajie Chen, Ye Zhou, Guoli Wang, Junsong Yuan, Qian Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gIHd-5X324](https://openreview.net/forum?id=gIHd-5X324)

**Abstract**:

Knowledge distillation is an effective approach to leverage a well-trained network or an ensemble of them, named as the teacher, to guide the training of a student network.  The outputs from the teacher network are used as soft labels for supervising the training of a new network.  Recent studies (M ̈uller et al., 2019; Yuan et al., 2020) revealed an intriguing property of the soft labels that making labels soft serves as a good regularization to the student network.   From the perspective of statistical learning,  regularization aims to reduce the variance,  however how bias and variance change is not clear for training with soft labels.   In this paper, we investigate the bias-variance tradeoff brought by distillation with soft labels.   Specifically,  we observe that during training the bias-variance tradeoff varies sample-wisely. Further, under the same distillation temperature setting, we observe that the distillation performance is negatively associated with the number of some specific samples, which are named as regularization samples since these samples lead to bias increasing and variance decreasing.  Nevertheless, we empirically find that completely filtering out regularization samples also deteriorates distillation performance.  Our discoveries inspired us to propose the novel weighted soft labels to help the network adaptively handle the sample-wise bias-variance tradeoff.  Experiments on standard evaluation benchmarks validate the effectiveness of our method. Our code is available in the supplementary.

----

## [398] A Design Space Study for LISTA and Beyond

**Authors**: *Tianjian Meng, Xiaohan Chen, Yifan Jiang, Zhangyang Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=GMgHyUPrXa](https://openreview.net/forum?id=GMgHyUPrXa)

**Abstract**:

In recent years, great success has been witnessed in building problem-specific deep networks from unrolling iterative algorithms, for solving inverse problems and beyond. Unrolling is believed to incorporate the model-based prior with the learning capacity of deep learning. This paper revisits \textit{the role of unrolling as a design approach for deep networks}: to what extent its resulting special architecture is superior, and can we find better? Using LISTA for sparse recovery as a representative example, we conduct the first thorough \textit{design space study} for the unrolled models.  Among all possible variations, we focus on extensively varying the connectivity patterns and neuron types, leading to a gigantic design space arising from LISTA. To efficiently explore this space and identify top performers, we leverage the emerging tool of neural architecture search (NAS). We carefully examine the searched top architectures in a number of settings, and are able to discover networks that consistently better than LISTA. We further present more visualization and analysis to ``open the black box", and find that the searched top architectures demonstrate highly consistent and potentially transferable patterns. We hope our study to spark more reflections and explorations on how to better mingle model-based optimization prior and data-driven learning.

----

## [399] What Should Not Be Contrastive in Contrastive Learning

**Authors**: *Tete Xiao, Xiaolong Wang, Alexei A. Efros, Trevor Darrell*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=CZ8Y3NzuVzO](https://openreview.net/forum?id=CZ8Y3NzuVzO)

**Abstract**:

Recent self-supervised contrastive methods have been able to produce impressive transferable visual representations by learning to be invariant to different data augmentations. However, these methods implicitly assume a particular set of representational invariances (e.g., invariance to color), and can perform poorly when a downstream task violates this assumption (e.g., distinguishing red vs. yellow cars). We introduce a contrastive learning framework which does not require prior knowledge of specific, task-dependent invariances. Our model learns to capture varying and invariant factors for visual representations by constructing separate embedding spaces, each of which is invariant to all but one augmentation. We use a multi-head network with a shared backbone which captures information across each augmentation and alone outperforms all baselines on downstream tasks. We further find that the concatenation of the invariant and varying spaces performs best across all tasks we investigate, including coarse-grained, fine-grained, and few-shot downstream classification tasks, and various data corruptions.

----



[Go to the previous page](ICLR-2021-list01.md)

[Go to the next page](ICLR-2021-list03.md)

[Go to the catalog section](README.md)