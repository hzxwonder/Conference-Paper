## [0] Learning Unseen Emotions from Gestures via Semantically-Conditioned Zero-Shot Perception with Adversarial Autoencoders

**Authors**: *Abhishek Banerjee, Uttaran Bhattacharya, Aniket Bera*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19873](https://doi.org/10.1609/aaai.v36i1.19873)

**Abstract**:

We present a novel generalized zero-shot algorithm to recognize perceived emotions from gestures. Our task is to map gestures to novel emotion categories not encountered in training. We introduce an adversarial autoencoder-based representation learning that correlates 3D motion-captured gesture sequences with the vectorized representation of the natural-language perceived emotion terms using word2vec embeddings. The language-semantic embedding provides a representation of the emotion label space, and we leverage this underlying distribution to map the gesture sequences to the appropriate categorical emotion labels. We train our method using a combination of gestures annotated with known emotion terms and gestures not annotated with any emotions. We evaluate our method on the MPI Emotional Body Expressions Database (EBEDB) and obtain an accuracy of 58.43%. We see an improvement in performance compared to current state-of-the-art algorithms for generalized zero-shot learning by an absolute 25-27%. We also demonstrate our approach on publicly available online videos and movie scenes, where the actors' pose has been extracted and map to their respective emotive states.

----

## [1] Optimized Potential Initialization for Low-Latency Spiking Neural Networks

**Authors**: *Tong Bu, Jianhao Ding, Zhaofei Yu, Tiejun Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19874](https://doi.org/10.1609/aaai.v36i1.19874)

**Abstract**:

Spiking Neural Networks (SNNs) have been attached great importance due to the distinctive properties of low power consumption, biological plausibility, and adversarial robustness. The most effective way to train deep SNNs is through ANN-to-SNN conversion, which have yielded the best performance in deep network structure and large-scale datasets. However, there is a trade-off between accuracy and latency. In order to achieve high precision as original ANNs, a long simulation time is needed to match the firing rate of a spiking neuron with the activation value of an analog neuron, which impedes the practical application of SNN. In this paper, we aim to achieve high-performance converted SNNs with extremely low latency (fewer than 32 time-steps). We start by theoretically analyzing ANN-to-SNN conversion and show that scaling the thresholds does play a similar role as weight normalization. Instead of introducing constraints that facilitate ANN-to-SNN conversion at the cost of model capacity, we applied a more direct way by optimizing the initial membrane potential to reduce the conversion loss in each layer. Besides, we demonstrate that optimal initialization of membrane potentials can implement expected error-free ANN-to-SNN conversion. We evaluate our algorithm on the CIFAR-10 dataset and CIFAR-100 dataset and achieve state-of-the-art accuracy, using fewer time-steps. For example, we reach top-1 accuracy of 93.38% on CIFAR-10 with 16 time-steps. Moreover, our method can be applied to other ANN-SNN conversion methodologies and remarkably promote performance when the time-steps is small.

----

## [2] Planning with Biological Neurons and Synapses

**Authors**: *Francesco D'Amore, Daniel Mitropolsky, Pierluigi Crescenzi, Emanuele Natale, Christos H. Papadimitriou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19875](https://doi.org/10.1609/aaai.v36i1.19875)

**Abstract**:

We revisit the planning problem in the blocks world, and we implement a known heuristic for this task. Importantly, our implementation is biologically plausible, in the sense that it is carried out exclusively through the spiking of neurons. Even though much has been accomplished in the blocks world over the past five decades, we believe that this is the first algorithm of its kind. The input is a sequence of symbols encoding an initial set of block stacks as well as a target set, and the output is a sequence of motion commands such as "put the top block in stack 1 on the table". The program is written in the Assembly Calculus, a recently proposed computational framework meant to model computation in the brain by bridging the gap between neural activity and cognitive function. Its elementary objects are assemblies of neurons (stable sets of neurons whose simultaneous firing signifies that the subject is thinking of an object, concept, word, etc.), its commands include project and merge, and its execution model is based on widely accepted tenets of neuroscience. A program in this framework essentially sets up a dynamical system of neurons and synapses that eventually, with high probability, accomplishes the task. The purpose of this work is to establish empirically that reasonably large programs in the Assembly Calculus can execute correctly and reliably; and that rather realistic --- if idealized --- higher cognitive functions, such as planning in the blocks world, can be implemented successfully by such programs.

----

## [3] Backprop-Free Reinforcement Learning with Active Neural Generative Coding

**Authors**: *Alexander G. Ororbia II, Ankur Arjun Mali*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19876](https://doi.org/10.1609/aaai.v36i1.19876)

**Abstract**:

In humans, perceptual awareness facilitates the fast recognition and extraction of information from sensory input. This awareness largely depends on how the human agent interacts with the environment. In this work, we propose active neural generative coding, a computational framework for learning action-driven generative models without backpropagation of errors (backprop) in dynamic environments. Specifically, we develop an intelligent agent that operates even with sparse rewards, drawing inspiration from the cognitive theory of planning as inference. We demonstrate on several simple control problems that our framework performs competitively with deep Q-learning. The robust performance of our agent offers promising evidence that a backprop-free approach for neural inference and learning can drive goal-directed behavior.

----

## [4] VECA: A New Benchmark and Toolkit for General Cognitive Development

**Authors**: *Kwanyoung Park, Hyunseok Oh, Youngki Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19877](https://doi.org/10.1609/aaai.v36i1.19877)

**Abstract**:

The developmental approach, simulating a cognitive development of a human, arises as a way to nurture a human-level commonsense and overcome the limitations of data-driven approaches. However, neither a virtual environment nor an evaluation platform exists for the overall development of core cognitive skills. We present the VECA(Virtual Environment for Cognitive Assessment), which consists of two main components: (i) a first benchmark to assess the overall cognitive development of an AI agent, and (ii) a novel toolkit to generate diverse and distinct cognitive tasks. VECA benchmark virtually implements the cognitive scale of Bayley Scales of Infant and Toddler Development-IV(Bayley-4), the gold-standard developmental assessment for human infants and toddlers. Our VECA toolkit provides a human toddler-like embodied agent with various human-like perceptual features crucial to human cognitive development, e.g., binocular vision, 3D-spatial audio, and tactile receptors. We compare several modern RL algorithms on our VECA benchmark and seek their limitations in modeling human-like cognitive development. We further analyze the validity of the VECA benchmark, as well as the effect of human-like sensory characteristics on cognitive skills.

----

## [5] Bridging between Cognitive Processing Signals and Linguistic Features via a Unified Attentional Network

**Authors**: *Yuqi Ren, Deyi Xiong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19878](https://doi.org/10.1609/aaai.v36i1.19878)

**Abstract**:

Cognitive processing signals can be used to improve natural language processing (NLP) tasks. However, it is not clear how these signals correlate with linguistic information. Bridging between human language processing and linguistic features has been widely studied in neurolinguistics, usually via single-variable controlled experiments with highly-controlled stimuli. Such methods not only compromises the authenticity of natural reading, but also are time-consuming and expensive. In this paper, we propose a data-driven method to investigate the relationship between cognitive processing signals and linguistic features. Specifically, we present a unified attentional framework that is composed of embedding, attention, encoding and predicting layers to selectively map cognitive processing signals to linguistic features. We define the mapping procedure as a bridging task and develop 12 bridging tasks for lexical, syntactic and semantic features. The proposed framework only requires cognitive processing signals recorded under natural reading as inputs, and can be used to detect a wide range of linguistic features with a single cognitive dataset. Observations from experiment results resonate with previous neuroscience findings. In addition to this, our experiments also reveal a number of interesting findings, such as the correlation between contextual eye-tracking features and tense of sentence.

----

## [6] Multi-Sacle Dynamic Coding Improved Spiking Actor Network for Reinforcement Learning

**Authors**: *Duzhen Zhang, Tielin Zhang, Shuncheng Jia, Bo Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19879](https://doi.org/10.1609/aaai.v36i1.19879)

**Abstract**:

With the help of deep neural networks (DNNs), deep reinforcement learning (DRL) has achieved great success on many complex tasks, from games to robotic control. Compared to DNNs with partial brain-inspired structures and functions, spiking neural networks (SNNs) consider more biological features, including spiking neurons with complex dynamics and learning paradigms with biologically plausible plasticity principles. Inspired by the efficient computation of cell assembly in the biological brain, whereby memory-based coding is much more complex than readout, we propose a multiscale dynamic coding improved spiking actor network (MDC-SAN) for reinforcement learning to achieve effective decision-making. The population coding at the network scale is integrated with the dynamic neurons coding (containing 2nd-order neuronal dynamics) at the neuron scale towards a powerful spatial-temporal state representation. Extensive experimental results show that our MDC-SAN performs better than its counterpart deep actor network (based on DNNs) on four continuous control tasks from OpenAI gym. We think this is a significant attempt to improve SNNs from the perspective of efficient coding towards effective decision-making, just like that in biological networks.

----

## [7] Joint Human Pose Estimation and Instance Segmentation with PosePlusSeg

**Authors**: *Niaz Ahmad, Jawad Khan, Jeremy Yuhyun Kim, Youngmoon Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19880](https://doi.org/10.1609/aaai.v36i1.19880)

**Abstract**:

Despite the advances in multi-person pose estimation, state-of-the-art techniques only deliver the human pose structure.Yet, they do not leverage the keypoints of human pose to deliver whole-body shape information for human instance segmentation. This paper presents PosePlusSeg, a joint model designed for both human pose estimation and instance segmentation. For pose estimation, PosePlusSeg first takes a bottom-up approach to detect the soft and hard keypoints of individuals by producing a strong keypoint heat map, then improves the keypoint detection confidence score by producing a body heat map. For instance segmentation, PosePlusSeg generates a mask offset where keypoint is defined as a centroid for the pixels in the embedding space, enabling instance-level segmentation for the human class. Finally, we propose a new pose and instance segmentation algorithm that enables PosePlusSeg to determine the joint structure of the human pose and instance segmentation. Experiments using the COCO challenging dataset demonstrate that PosePlusSeg copes better with challenging scenarios, like occlusions, en-tangled limbs, and overlapped people. PosePlusSeg outperforms state-of-the-art detection-based approaches achieving a 0.728 mAP for human pose estimation and a 0.445 mAP for instance segmentation. Code has been made available at:
 https://github.com/RaiseLab/PosePlusSeg.

----

## [8] Logic Rule Guided Attribution with Dynamic Ablation

**Authors**: *Jianqiao An, Yuandu Lai, Yahong Han*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19881](https://doi.org/10.1609/aaai.v36i1.19881)

**Abstract**:

With the increasing demands for understanding the internal behaviors of deep networks, Explainable AI (XAI) has been made remarkable progress in interpreting the model's decision. A family of attribution techniques has been proposed, highlighting whether the input pixels are responsible for the model's prediction. However, the existing attribution methods suffer from the lack of rule guidance and require further human interpretations. In this paper, we construct the 'if-then' logic rules that are sufficiently precise locally. Moreover, a novel rule-guided method, dynamic ablation (DA), is proposed to find a minimal bound sufficient in an input image to justify the network's prediction and aggregate iteratively to reach a complete attribution. Both qualitative and quantitative experiments are conducted to evaluate the proposed DA. We demonstrate the advantages of our method in providing clear and explicit explanations that are also easy for human experts to understand. Besides, through the attribution on a series of trained networks with different architectures, we show that more complex networks require less information to make a specific prediction.

----

## [9] Neural Marionette: Unsupervised Learning of Motion Skeleton and Latent Dynamics from Volumetric Video

**Authors**: *Jinseok Bae, Hojun Jang, Cheol-Hui Min, Hyungun Choi, Young Min Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19882](https://doi.org/10.1609/aaai.v36i1.19882)

**Abstract**:

We present Neural Marionette, an unsupervised approach that discovers the skeletal structure from a dynamic sequence and learns to generate diverse motions that are consistent with the observed motion dynamics. Given a video stream of point cloud observation of an articulated body under arbitrary motion, our approach discovers the unknown low-dimensional skeletal relationship that can effectively represent the movement. Then the discovered structure is utilized to encode the motion priors of dynamic sequences in a latent structure, which can be decoded to the relative joint rotations to represent the full skeletal motion. Our approach works without any prior knowledge of the underlying motion or skeletal structure, and we demonstrate that the discovered structure is even comparable to the hand-labeled ground truth skeleton in representing a 4D sequence of motion. The skeletal structure embeds the general semantics of possible motion space that can generate motions for diverse scenarios. We verify that the learned motion prior is generalizable to the multi-modal sequence generation, interpolation of two poses, and motion retargeting to a different skeletal structure.

----

## [10] Deformable Part Region Learning for Object Detection

**Authors**: *Seung-Hwan Bae*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19883](https://doi.org/10.1609/aaai.v36i1.19883)

**Abstract**:

In a convolutional object detector, the detection accuracy can be degraded often due to the low feature discriminability caused by geometric variation or transformation of an object. In this paper, we propose a deformable part region learning in order to allow decomposed part regions to be deformable according to geometric transformation of an object. To this end, we introduce trainable geometric parameters for the location of each part model. Because the ground truth of the part models is not available, we design classification and mask losses for part models, and learn the geometric parameters by minimizing an integral loss including those part losses. As a result, we can train a deformable part region network without extra super-vision and make each part model deformable according to object scale variation. Furthermore, for improving cascade object detection and instance segmentation, we present a Cascade deformable part region architecture which can refine whole and part detections iteratively in the cascade manner. Without bells and whistles, our implementation of a Cascade deformable part region detector achieves better detection and segmentation mAPs on COCO and VOC datasets, compared to the recent cascade and other state-of-the-art detectors.

----

## [11] Towards End-to-End Image Compression and Analysis with Transformers

**Authors**: *Yuanchao Bai, Xu Yang, Xianming Liu, Junjun Jiang, Yaowei Wang, Xiangyang Ji, Wen Gao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19884](https://doi.org/10.1609/aaai.v36i1.19884)

**Abstract**:

We propose an end-to-end image compression and analysis model with Transformers, targeting to the cloud-based image classification application. Instead of placing an existing Transformer-based image classification model directly after an image codec, we aim to redesign the Vision Transformer (ViT) model to perform image classification from the compressed features and facilitate image compression with the long-term information from the Transformer. Specifically, we first replace the patchify stem (i.e., image splitting and embedding) of the ViT model with a lightweight image encoder modelled by a convolutional neural network. The compressed features generated by the image encoder are injected convolutional inductive bias and are fed to the Transformer for image classification bypassing image reconstruction. Meanwhile, we propose a feature aggregation module to fuse the compressed features with the selected intermediate features of the Transformer, and feed the aggregated features to a deconvolutional neural network for image reconstruction. The aggregated features can obtain the long-term information from the self-attention mechanism of the Transformer and improve the compression performance. The rate-distortion-accuracy optimization problem is finally solved by a two-step training strategy. Experimental results demonstrate the effectiveness of the proposed model in both the image compression and the classification tasks.

----

## [12] Handwritten Mathematical Expression Recognition via Attention Aggregation Based Bi-directional Mutual Learning

**Authors**: *Xiaohang Bian, Bo Qin, Xiaozhe Xin, Jianwu Li, Xuefeng Su, Yanfeng Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19885](https://doi.org/10.1609/aaai.v36i1.19885)

**Abstract**:

Handwritten mathematical expression recognition aims to automatically generate LaTeX sequences from given images. Currently, attention-based encoder-decoder models are widely used in this task. They typically generate target sequences in a left-to-right (L2R) manner, leaving the right-to-left (R2L) contexts unexploited. In this paper, we propose an Attention aggregation based Bi-directional Mutual learning Network (ABM) which consists of one shared encoder and two parallel inverse decoders (L2R and R2L). The two decoders are enhanced via mutual distillation, which involves one-to-one knowledge transfer at each training step, making full use of the complementary information from two inverse directions. Moreover, in order to deal with mathematical symbols in diverse scales, an Attention Aggregation Module (AAM) is proposed to effectively integrate multi-scale coverage attentions. Notably, in the inference phase, given that the model already learns knowledge from two inverse directions, we only use the L2R branch for inference, keeping the original parameter size and inference speed. Extensive experiments demonstrate that our proposed approach achieves the recognition accuracy of 56.85 % on CROHME 2014, 52.92 % on CROHME 2016, and 53.96 % on CROHME 2019 without data augmentation and model ensembling, substantially outperforming the state-of-the-art methods. The source code is available in https://github.com/XH-B/ABM.

----

## [13] ADD: Frequency Attention and Multi-View Based Knowledge Distillation to Detect Low-Quality Compressed Deepfake Images

**Authors**: *Le Minh Binh, Simon S. Woo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19886](https://doi.org/10.1609/aaai.v36i1.19886)

**Abstract**:

Despite significant advancements of deep learning-based forgery detectors for distinguishing manipulated deepfake images, most detection approaches suffer from moderate to significant performance degradation with low-quality compressed deepfake images. Because of the limited information in low-quality images, detecting low-quality deepfake remains an important challenge. In this work, we apply frequency domain learning and optimal transport theory in knowledge distillation (KD) to specifically improve the detection of low-quality compressed deepfake images. We explore transfer learning capability in KD to enable a student network to learn discriminative features from low-quality images effectively. In particular, we propose the Attention-based Deepfake detection Distiller (ADD), which consists of two novel distillations: 1) frequency attention distillation that effectively retrieves the removed high-frequency components in the student network, and 2) multi-view attention distillation that creates multiple attention vectors by slicing the teacher’s and student’s tensors under different views to transfer the teacher tensor’s distribution to the student more efficiently. Our extensive experimental results demonstrate that our approach outperforms state-of-the-art baselines in detecting low-quality compressed deepfake images.

----

## [14] LUNA: Localizing Unfamiliarity Near Acquaintance for Open-Set Long-Tailed Recognition

**Authors**: *Jiarui Cai, Yizhou Wang, Hung-Min Hsu, Jenq-Neng Hwang, Kelsey Magrane, Craig S. Rose*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19887](https://doi.org/10.1609/aaai.v36i1.19887)

**Abstract**:

The predefined artificially-balanced training classes in object recognition have limited capability in modeling real-world scenarios where objects are imbalanced-distributed with unknown classes. In this paper, we discuss a promising solution to the Open-set Long-Tailed Recognition (OLTR) task utilizing metric learning. Firstly, we propose a distribution-sensitive loss, which weighs more on the tail classes to decrease the intra-class distance in the feature space. Building upon these concentrated feature clusters, a local-density-based metric is introduced, called Localizing Unfamiliarity Near Acquaintance (LUNA), to measure the novelty of a testing sample. LUNA is flexible with different cluster sizes and is reliable on the cluster boundary by considering neighbors of different properties. Moreover, contrary to most of the existing works that alleviate the open-set detection as a simple binary decision, LUNA is a quantitative measurement with interpretable meanings. Our proposed method exceeds the state-of-the-art algorithm by 4-6% in the closed-set recognition accuracy and 4% in F-measure under the open-set on the public benchmark datasets, including our own newly introduced fine-grained OLTR dataset about marine species (MS-LT), which is the first naturally-distributed OLTR dataset revealing the genuine genetic relationships of the classes.

----

## [15] Prior Gradient Mask Guided Pruning-Aware Fine-Tuning

**Authors**: *Linhang Cai, Zhulin An, Chuanguang Yang, Yangchun Yan, Yongjun Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19888](https://doi.org/10.1609/aaai.v36i1.19888)

**Abstract**:

We proposed a Prior Gradient Mask Guided Pruning-aware Fine-Tuning (PGMPF) framework to accelerate deep Convolutional Neural Networks (CNNs). In detail, the proposed PGMPF selectively suppresses the gradient of those ”unimportant” parameters via a prior gradient mask generated by the pruning criterion during fine-tuning. PGMPF has three charming characteristics over previous works: (1) Pruning-aware network fine-tuning. A typical pruning pipeline consists of training, pruning and fine-tuning, which are relatively independent, while PGMPF utilizes a variant of the pruning mask as a prior gradient mask to guide fine-tuning, without
 complicated pruning criteria. (2) An excellent tradeoff between large model capacity during fine-tuning and stable convergence speed to obtain the final compact model. Previous works preserve more training information of pruned parameters during fine-tuning to pursue better performance, which would incur catastrophic non-convergence of the pruned model for relatively large pruning rates, while our PGMPF greatly stabilizes the fine-tuning phase by gradually constraining the learning rate of those ”unimportant” parameters. (3) Channel-wise random dropout of the prior gradient mask to impose some gradient noise to fine-tuning to further improve the robustness of final compact model. Experimental results on three image classification benchmarks CIFAR10/ 100 and ILSVRC-2012 demonstrate the effectiveness of our method for various CNN architectures, datasets and pruning rates. Notably, on ILSVRC-2012, PGMPF reduces 53.5% FLOPs on ResNet-50 with only 0.90% top-1 accuracy drop and 0.52% top-5 accuracy drop, which has advanced the state-of-the-art with negligible extra computational cost.

----

## [16] Context-Aware Transfer Attacks for Object Detection

**Authors**: *Zikui Cai, Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19889](https://doi.org/10.1609/aaai.v36i1.19889)

**Abstract**:

Blackbox transfer attacks for image classifiers have been extensively studied in recent years. In contrast, little progress has been made on transfer attacks for object detectors. Object detectors take a holistic view of the image and the detection of one object (or lack thereof) often depends on other objects in the scene. This makes such detectors inherently context-aware and adversarial attacks in this space are more challenging than those targeting image classifiers. In this paper, we present a new approach to generate context-aware attacks for object detectors. We show that by using co-occurrence of objects and their relative locations and sizes as context information, we can successfully generate targeted mis-categorization attacks that achieve higher transfer success rates on blackbox object detectors than the state-of-the-art. We test our approach on a variety of object detectors with images from PASCAL VOC and MS COCO datasets and demonstrate up to 20 percentage points improvement in performance compared to the other state-of-the-art methods.

----

## [17] OoDHDR-Codec: Out-of-Distribution Generalization for HDR Image Compression

**Authors**: *Linfeng Cao, Aofan Jiang, Wei Li, Huaying Wu, Nanyang Ye*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19890](https://doi.org/10.1609/aaai.v36i1.19890)

**Abstract**:

Recently, deep learning has been proven to be a promising approach in standard dynamic range (SDR) image compression. However, due to the wide luminance distribution of high dynamic range (HDR) images and the lack of large standard datasets, developing a deep model for HDR image compression is much more challenging. To tackle this issue, we view HDR data as distributional shifts of SDR data and the HDR image compression can be modeled as an out-of-distribution generalization (OoD) problem. Herein, we propose a novel out-of-distribution (OoD) HDR image compression framework (OoDHDR-codec). It learns the general representation across HDR and SDR environments, and allows the model to be trained effectively using a large set of SDR datases supplemented with much fewer HDR samples. Specifically, OoDHDR-codec consists of two branches to process the data from two environments. The SDR branch is a standard blackbox network. For the HDR branch, we develop a hybrid system that models luminance masking and tone mapping with white-box modules and performs content compression with black-box neural networks. To improve the generalization from SDR training data on HDR data, we introduce an invariance regularization term to learn the common representation for both SDR and HDR compression. Extensive experimental results show that the OoDHDR codec achieves strong competitive in-distribution performance and state-of-the-art OoD performance. To the best of our knowledge, our proposed approach is the first work to model HDR compression as OoD generalization problems and our OoD generalization algorithmic framework can be applied to any deep compression model in addition to the network architectural choice demonstrated in the paper. Code available at https://github.com/caolinfeng/OoDHDR-codec.

----

## [18] Visual Consensus Modeling for Video-Text Retrieval

**Authors**: *Shuqiang Cao, Bairui Wang, Wei Zhang, Lin Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19891](https://doi.org/10.1609/aaai.v36i1.19891)

**Abstract**:

In this paper, we propose a novel method to mine the commonsense knowledge shared between the video and text modalities for video-text retrieval, namely visual consensus modeling. Different from the existing works, which learn the video and text representations and their complicated relationships solely based on the pairwise video-text data, we make the first attempt to model the visual consensus by mining the visual concepts from videos and exploiting their co-occurrence patterns within the video and text modalities with no reliance on any additional concept annotations. Specifically, we build a shareable and learnable graph as the visual consensus, where the nodes denoting the mined visual concepts and the edges connecting the nodes representing the co-occurrence relationships between the visual concepts. Extensive experimental results on the public benchmark datasets demonstrate that our proposed method, with the ability to effectively model the visual consensus,  achieves state-of-the-art performances on the bidirectional video-text retrieval task. Our code is available at https://github.com/sqiangcao99/VCM.

----

## [19] Proximal PanNet: A Model-Based Deep Network for Pansharpening

**Authors**: *Xiangyong Cao, Yang Chen, Wenfei Cao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19892](https://doi.org/10.1609/aaai.v36i1.19892)

**Abstract**:

Recently, deep learning techniques have been extensively studied for pansharpening, which aims to generate a high resolution multispectral (HRMS) image by fusing a low resolution multispectral (LRMS) image with a high resolution panchromatic (PAN) image. However, existing deep learning-based pansharpening methods directly learn the mapping from LRMS and PAN to HRMS. These network architectures always lack sufficient interpretability, which limits further performance improvements. To alleviate this issue, we propose a novel deep network for pansharpening by combining the model-based methodology with the deep learning method. Firstly, we build an observation model for pansharpening using the convolutional sparse coding (CSC) technique and design a proximal gradient algorithm to solve this model. Secondly, we unfold the iterative algorithm into a deep network, dubbed as Proximal PanNet, by learning the proximal operators using convolutional neural networks. Finally, all the learnable modules can be automatically learned in an end-to-end manner. Experimental results on some benchmark datasets show that our network performs better than other advanced methods both quantitatively and qualitatively.

----

## [20] CF-DETR: Coarse-to-Fine Transformers for End-to-End Object Detection

**Authors**: *Xipeng Cao, Peng Yuan, Bailan Feng, Kun Niu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19893](https://doi.org/10.1609/aaai.v36i1.19893)

**Abstract**:

The recently proposed DEtection TRansformer (DETR) achieves promising performance for end-to-end object detection. However, it has relatively lower detection performance on small objects and suffers from slow convergence. This paper observed that DETR performs surprisingly well even on small objects when measuring Average Precision (AP) at decreased Intersection-over-Union (IoU) thresholds. Motivated by this observation, we propose a simple way to improve DETR by refining the coarse features and predicted locations. Specifically, we propose a novel Coarse-to-Fine (CF) decoder layer constituted of a coarse layer and a carefully designed fine layer. Within each CF decoder layer, the extracted local information (region of interest feature) is introduced into the flow of global context information from the coarse layer to refine and enrich the object query features via the fine layer. In the fine layer, the multi-scale information can be fully explored and exploited via the Adaptive Scale Fusion(ASF) module and Local Cross-Attention (LCA) module. The multi-scale information can also be enhanced by another proposed Transformer Enhanced FPN (TEF) module to further improve the performance. With our proposed framework (named CF-DETR), the localization accuracy of objects (especially for small objects) can be largely improved. As a byproduct, the slow convergence issue of DETR can also be addressed. The effectiveness of CF-DETR is validated via extensive experiments on the coco benchmark. CF-DETR achieves state-of-the-art performance among end-to-end detectors, e.g., achieving 47.8 AP using ResNet-50 with 36 epochs in the standard 3x training schedule.

----

## [21] A Random CNN Sees Objects: One Inductive Bias of CNN and Its Applications

**Authors**: *Yun-Hao Cao, Jianxin Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19894](https://doi.org/10.1609/aaai.v36i1.19894)

**Abstract**:

This paper starts by revealing a surprising finding: without any learning, a randomly initialized CNN can localize objects surprisingly well. That is, a CNN has an inductive bias to naturally focus on objects, named as Tobias ("The object is at sight") in this paper. This empirical inductive bias is further analyzed and successfully applied to self-supervised learning (SSL). A CNN is encouraged to learn representations that focus on the foreground object, by transforming every image into various versions with different backgrounds, where the foreground and background separation is guided by Tobias. Experimental results show that the proposed Tobias significantly improves downstream tasks, especially for object detection. This paper also shows that Tobias has consistent improvements on training sets of different sizes, and is more resilient to changes in image augmentations.

----

## [22] Texture Generation Using Dual-Domain Feature Flow with Multi-View Hallucinations

**Authors**: *Seunggyu Chang, Jungchan Cho, Songhwai Oh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19895](https://doi.org/10.1609/aaai.v36i1.19895)

**Abstract**:

We propose a dual-domain generative model to estimate a texture map from a single image for colorizing a 3D human model. When estimating a texture map, a single image is insufficient as it reveals only one facet of a 3D object. To provide sufficient information for estimating a complete texture map, the proposed model simultaneously generates multi-view hallucinations in the image domain and an estimated texture map in the texture domain. During the generating process, each domain generator exchanges features to the other by a flow-based local attention mechanism. In this manner, the proposed model can estimate a texture map utilizing abundant multi-view image features from which multiview hallucinations are generated. As a result, the estimated texture map contains consistent colors and patterns over the entire region. Experiments show the superiority of our model for estimating a directly render-able texture map, which is applicable to 3D animation rendering. Furthermore, our model also improves an overall generation quality in the image domain for pose and viewpoint transfer tasks.

----

## [23] Resistance Training Using Prior Bias: Toward Unbiased Scene Graph Generation

**Authors**: *Chao Chen, Yibing Zhan, Baosheng Yu, Liu Liu, Yong Luo, Bo Du*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19896](https://doi.org/10.1609/aaai.v36i1.19896)

**Abstract**:

Scene Graph Generation (SGG) aims to build a structured representation of a scene using objects and pairwise relationships, which benefits downstream tasks. However, current SGG methods usually suffer from sub-optimal scene graph generation because of the long-tailed distribution of training data. To address this problem, we propose Resistance Training using Prior Bias (RTPB) for the scene graph generation. Specifically, RTPB uses a distributed-based prior bias to improve models' detecting ability on less frequent relationships during training, thus improving the model generalizability on tail categories. In addition, to further explore the contextual information of objects and relationships, we design a contextual encoding backbone network, termed as Dual Transformer (DTrans). We perform extensive experiments on a very popular benchmark, VG150, to demonstrate the effectiveness of our method for the unbiased scene graph generation. In specific, our RTPB achieves an improvement of over 10% under the mean recall when applied to current SGG methods. Furthermore, DTrans with RTPB outperforms nearly all state-of-the-art methods with a large margin. Code is available at https://github.com/ChCh1999/RTPB

----

## [24] SASA: Semantics-Augmented Set Abstraction for Point-Based 3D Object Detection

**Authors**: *Chen Chen, Zhe Chen, Jing Zhang, Dacheng Tao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19897](https://doi.org/10.1609/aaai.v36i1.19897)

**Abstract**:

Although point-based networks are demonstrated to be accurate for 3D point cloud modeling, they are still falling behind their voxel-based competitors in 3D detection. We observe that the prevailing set abstraction design for down-sampling points may maintain too much unimportant background information that can affect feature learning for detecting objects. To tackle this issue, we propose a novel set abstraction method named Semantics-Augmented Set Abstraction (SASA). Technically, we first add a binary segmentation module as the side output to help identify foreground points. Based on the estimated point-wise foreground scores, we then propose a semantics-guided point sampling algorithm to help retain more important foreground points during down-sampling. In practice, SASA shows to be effective in identifying valuable points related to foreground objects and improving feature learning for point-based 3D detection. Additionally, it is an easy-to-plug-in module and able to boost various point-based detectors, including single-stage and two-stage ones. Extensive experiments on the popular KITTI and nuScenes datasets validate the superiority of SASA, lifting point-based detection models to reach comparable performance to state-of-the-art voxel-based methods. Code is available at https://github.com/blakechen97/SASA.

----

## [25] Comprehensive Regularization in a Bi-directional Predictive Network for Video Anomaly Detection

**Authors**: *Chengwei Chen, Yuan Xie, Shaohui Lin, Angela Yao, Guannan Jiang, Wei Zhang, Yanyun Qu, Ruizhi Qiao, Bo Ren, Lizhuang Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19898](https://doi.org/10.1609/aaai.v36i1.19898)

**Abstract**:

Video anomaly detection aims to automatically identify unusual objects or behaviours by learning from normal videos. Previous methods tend to use simplistic reconstruction or prediction constraints, which leads to the insufficiency of learned representations for normal data. As such, we propose a novel bi-directional architecture with three consistency constraints to comprehensively regularize the prediction task from pixel-wise, cross-modal, and temporal-sequence levels. First, predictive consistency is proposed to consider the symmetry property of motion and appearance in forwards and backwards time, which ensures the highly realistic appearance and motion predictions at the pixel-wise level. Second, association consistency considers the relevance between different modalities and uses one modality to regularize the prediction of another one. Finally, temporal consistency utilizes the relationship of the video sequence and ensures that the predictive network generates temporally consistent frames. During inference, the pattern of abnormal frames is unpredictable and will therefore cause higher prediction errors. Experiments show that our method outperforms advanced anomaly detectors and achieves state-of-the-art results on UCSD Ped2, CUHK Avenue, and ShanghaiTech datasets.

----

## [26] Keypoint Message Passing for Video-Based Person Re-identification

**Authors**: *Di Chen, Andreas Doering, Shanshan Zhang, Jian Yang, Juergen Gall, Bernt Schiele*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19899](https://doi.org/10.1609/aaai.v36i1.19899)

**Abstract**:

Video-based person re-identification~(re-ID) is an important technique in visual surveillance systems which aims to match video snippets of people captured by different cameras. Existing methods are mostly based on convolutional neural networks~(CNNs), whose building blocks either process local neighbor pixels at a time, or, when 3D convolutions are used to model temporal information, suffer from the misalignment problem caused by person movement. In this paper, we propose to overcome the limitations of normal convolutions with a human-oriented graph method. Specifically, features located at person joint keypoints are extracted and connected as a spatial-temporal graph. These keypoint features are then updated by message passing from their connected nodes with a graph convolutional network~(GCN). During training, the GCN can be attached to any CNN-based person re-ID model to assist representation learning on feature maps, whilst it can be dropped after training for better inference speed. Our method brings significant improvements over the CNN-based baseline model on the MARS dataset with generated person keypoints and a newly annotated dataset: PoseTrackReID. It also defines a new state-of-the-art method in terms of top-1 accuracy and mean average precision in comparison to prior works.

----

## [27] DCAN: Improving Temporal Action Detection via Dual Context Aggregation

**Authors**: *Guo Chen, Yin-Dong Zheng, Limin Wang, Tong Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19900](https://doi.org/10.1609/aaai.v36i1.19900)

**Abstract**:

Temporal action detection aims to locate the boundaries of action in the video. The current method based on boundary matching enumerates and calculates all possible boundary matchings to generate proposals. However, these methods neglect the long-range context aggregation in boundary prediction. At the same time, due to the similar semantics of adjacent matchings, local semantic aggregation of densely-generated matchings cannot improve semantic richness and discrimination. In this paper, we propose the end-to-end proposal generation method named Dual Context Aggregation Network (DCAN) to aggregate context on two levels, namely, boundary level and proposal level, for generating high-quality action proposals, thereby improving the performance of temporal action detection. Specifically, we design the Multi-Path Temporal Context Aggregation (MTCA) to achieve smooth context aggregation on boundary level and precise evaluation of boundaries. For matching evaluation, Coarse-to-fine Matching (CFM) is designed to aggregate context on the proposal level and refine the matching map from coarse to fine. We conduct extensive experiments on ActivityNet v1.3 and THUMOS-14. DCAN obtains an average mAP of 35.39% on ActivityNet v1.3 and reaches mAP 54.14% at IoU@0.5 on THUMOS-14, which demonstrates DCAN can generate high-quality proposals and achieve state-of-the-art performance. We release the code at https://github.com/cg1177/DCAN.

----

## [28] Geometry-Contrastive Transformer for Generalized 3D Pose Transfer

**Authors**: *Haoyu Chen, Hao Tang, Zitong Yu, Nicu Sebe, Guoying Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19901](https://doi.org/10.1609/aaai.v36i1.19901)

**Abstract**:

We present a customized 3D mesh Transformer model for the pose transfer task. As the 3D pose transfer essentially is a deformation procedure dependent on the given meshes, the intuition of this work is to perceive the geometric inconsistency between the given meshes with the powerful self-attention mechanism. Specifically, we propose a novel geometry-contrastive Transformer that has an efficient 3D structured perceiving ability to the global geometric inconsistencies across the given meshes. Moreover, locally, a simple yet efficient central geodesic contrastive loss is further proposed to improve the regional geometric-inconsistency learning. At last, we present a latent isometric regularization module together with a novel semi-synthesized dataset for the cross-dataset 3D pose transfer task towards unknown spaces. The massive experimental results prove the efficacy of our approach by showing state-of-the-art quantitative performances on SMPL-NPT, FAUST and our new proposed dataset SMG-3D datasets, as well as promising qualitative results on MG-cloth and SMAL datasets. It's demonstrated that our method can achieve robust 3D pose transfer and be generalized to challenging meshes from unknown spaces on cross-dataset tasks. The code and dataset are made available. Code is available: https://github.com/mikecheninoulu/CGT.

----

## [29] Explore Inter-contrast between Videos via Composition for Weakly Supervised Temporal Sentence Grounding

**Authors**: *Jiaming Chen, Weixin Luo, Wei Zhang, Lin Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19902](https://doi.org/10.1609/aaai.v36i1.19902)

**Abstract**:

Weakly supervised temporal sentence grounding aims to temporally localize the target segment corresponding to a given natural language query, where it provides video-query pairs without temporal annotations during training. Most existing methods use the fused visual-linguistic feature to reconstruct the query, where the least reconstruction error determines the target segment. This work introduces a novel approach that explores the inter-contrast between videos in a composed video by selecting components from two different videos and fusing them into a single video. Such a straightforward yet effective composition strategy provides the temporal annotations at multiple composed positions, resulting in numerous videos with temporal ground-truths for training the temporal sentence grounding task. A transformer framework is introduced with multi-tasks training to learn a compact but efficient visual-linguistic space. The experimental results on the public Charades-STA and ActivityNet-Caption dataset demonstrate the effectiveness of the proposed method, where our approach achieves comparable performance over the state-of-the-art weakly-supervised baselines. The code is available at https://github.com/PPjmchen/Composition_WSTG.

----

## [30] Adaptive Image-to-Video Scene Graph Generation via Knowledge Reasoning and Adversarial Learning

**Authors**: *Jin Chen, Xiaofeng Ji, Xinxiao Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19903](https://doi.org/10.1609/aaai.v36i1.19903)

**Abstract**:

Scene graph in a video conveys a wealth of information about objects and their relationships in the scene, thus benefiting many downstream tasks such as video captioning and visual question answering. Existing methods of scene graph generation require large-scale training videos annotated with objects and relationships in each frame to learn a powerful model. However, such comprehensive annotation is time-consuming and labor-intensive. On the other hand, it is much easier and less cost to annotate images with scene graphs, so we investigate leveraging annotated images to facilitate training a scene graph generation model for unannotated videos, namely image-to-video scene graph generation. This task presents two challenges: 1) infer unseen dynamic relationships in videos from static relationships in images due to the absence of motion information in images; 2) adapt objects and static relationships from images to video frames due to the domain shift between them. To address the first challenge, we exploit external commonsense knowledge to infer the unseen dynamic relationship from the temporal evolution of static relationships. We tackle the second challenge by hierarchical adversarial learning to reduce the data distribution discrepancy between images and video frames. Extensive experiment results on two benchmark video datasets demonstrate the effectiveness of our method.

----

## [31] Text Gestalt: Stroke-Aware Scene Text Image Super-resolution

**Authors**: *Jingye Chen, Haiyang Yu, Jianqi Ma, Bin Li, Xiangyang Xue*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19904](https://doi.org/10.1609/aaai.v36i1.19904)

**Abstract**:

In the last decade, the blossom of deep learning has witnessed the rapid development of scene text recognition. However, the recognition of low-resolution scene text images remains a challenge. Even though some super-resolution methods have been proposed to tackle this problem, they usually treat text images as general images while ignoring the fact that the visual quality of strokes (the atomic unit of text) plays an essential role for text recognition. According to Gestalt Psychology, humans are capable of composing parts of details into the most similar objects guided by prior knowledge. Likewise, when humans observe a low-resolution text image, they will inherently use partial stroke-level details to recover the appearance of holistic characters. Inspired by Gestalt Psychology, we put forward a Stroke-Aware Scene Text Image Super-Resolution method containing a Stroke-Focused Module (SFM) to concentrate on stroke-level internal structures of characters in text images. Specifically, we attempt to design rules for decomposing English characters and digits at stroke-level, then pre-train a text recognizer to provide stroke-level attention maps as positional clues with the purpose of controlling the consistency between the generated super-resolution image and high-resolution ground truth. The extensive experimental results validate that the proposed method can indeed generate more distinguishable images on TextZoom and manually constructed Chinese character dataset Degraded-IC13. Furthermore, since the proposed SFM is only used to provide stroke-level guidance when training, it will not bring any time overhead during the test phase. Code is available at https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt.

----

## [32] Towards High-Fidelity Face Self-Occlusion Recovery via Multi-View Residual-Based GAN Inversion

**Authors**: *Jinsong Chen, Hu Han, Shiguang Shan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19905](https://doi.org/10.1609/aaai.v36i1.19905)

**Abstract**:

Face self-occlusions are inevitable due to the 3D nature of the human face and the loss of information in the projection process from 3D to 2D images. While recovering face self-occlusions based on 3D face reconstruction, e.g., 3D Morphable Model (3DMM) and its variants provides an effective solution, most of the existing methods show apparent limitations in expressing high-fidelity, natural, and diverse facial details. To overcome these limitations, we propose in this paper a new generative adversarial network (MvInvert) for natural face self-occlusion recovery without using paired image-texture data. We design a coarse-to-fine generator for photorealistic texture generation. A coarse texture is computed by inpainting the invisible areas in the photorealistic but incomplete texture sampled directly from the 2D image using the unrealistic but complete statistical texture from 3DMM. Then, we design a multi-view Residual-based GAN Inversion, which re-renders and refines multi-view 2D images, which are used for extracting multiple high-fidelity textures. Finally, these high-fidelity textures are fused based on their visibility maps via Poisson blending. To perform adversarial learning to assure the quality of the recovered texture, we design a discriminator consisting of two heads, i.e., one for global and local discrimination between the recovered texture and a small set of real textures in UV space, and the other for discrimination between the input image and the re-rendered 2D face images via pixel-wise, identity, and adversarial losses. Extensive experiments demonstrate that our approach outperforms the state-of-the-art methods in face self-occlusion recovery under unconstrained scenarios.

----

## [33] ProgressiveMotionSeg: Mutually Reinforced Framework for Event-Based Motion Segmentation

**Authors**: *Jinze Chen, Yang Wang, Yang Cao, Feng Wu, Zheng-Jun Zha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19906](https://doi.org/10.1609/aaai.v36i1.19906)

**Abstract**:

Dynamic Vision Sensor (DVS) can asynchronously output the events reflecting apparent motion of objects with microsecond resolution, and shows great application potential in monitoring and other fields. However, the output event stream of existing DVS inevitably contains background activity noise (BA noise) due to dark current and junction leakage current, which will affect the temporal correlation of objects, resulting in deteriorated motion estimation performance. Particularly, the existing filter-based denoising methods cannot be directly applied to suppress the noise in event stream, since there is no spatial correlation. To address this issue, this paper presents a novel progressive framework, in which a Motion Estimation (ME) module and an Event Denoising (ED) module are jointly optimized in a mutually reinforced manner. Specifically, based on the maximum sharpness criterion, ME module divides the input event into several segments by adaptive clustering in a motion compensating warp field, and captures the temporal correlation of event stream according to the clustered motion parameters. Taking temporal correlation as guidance, ED module calculates the confidence that each event belongs to real activity events, and transmits it to ME module to update energy function of motion segmentation for noise suppression. The two steps are iteratively updated until stable motion segmentation results are obtained. Extensive experimental results on both synthetic and real datasets demonstrate the superiority of our proposed approaches against the State-Of-The-Art (SOTA) methods.

----

## [34] Attacking Video Recognition Models with Bullet-Screen Comments

**Authors**: *Kai Chen, Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19907](https://doi.org/10.1609/aaai.v36i1.19907)

**Abstract**:

Recent research has demonstrated that Deep Neural Networks (DNNs) are vulnerable to adversarial patches which introduce perceptible but localized changes to the input. Nevertheless, existing approaches have focused on generating adversarial patches on images, their counterparts in videos have been less explored. Compared with images, attacking videos is much more challenging as it needs to consider not only spatial cues but also temporal cues. To close this gap, we introduce a novel adversarial attack in this paper, the bullet-screen comment (BSC) attack, which attacks video recognition models with BSCs. Specifically, adversarial BSCs are generated with a Reinforcement Learning (RL) framework, where the environment is set as the target model and the agent plays the role of selecting the position and transparency of each BSC. By continuously querying the target models and receiving feedback, the agent gradually adjusts its selection strategies in order to achieve a high fooling rate with non-overlapping BSCs. As BSCs can be regarded as a kind of meaningful patch, adding it to a clean video will not affect people’s understanding of the video content, nor will arouse people’s suspicion. We conduct extensive experiments to verify the effectiveness of the proposed method. On both UCF-101 and HMDB-51 datasets, our BSC attack method can achieve about 90% fooling rate when attacking three mainstream video recognition models, while only occluding < 8% areas in the video. Our code is available at https://github.com/kay-ck/BSC-attack.

----

## [35] VITA: A Multi-Source Vicinal Transfer Augmentation Method for Out-of-Distribution Generalization

**Authors**: *Minghui Chen, Cheng Wen, Feng Zheng, Fengxiang He, Ling Shao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19908](https://doi.org/10.1609/aaai.v36i1.19908)

**Abstract**:

Invariance to diverse types of image corruption, such as noise, blurring, or colour shifts, is essential to establish robust models in computer vision. Data augmentation has been the major approach in improving the robustness against common corruptions. However, the samples produced by popular augmentation strategies deviate significantly from the underlying data manifold. As a result, performance is skewed toward certain types of corruption. To address this issue, we propose a multi-source vicinal transfer augmentation (VITA) method for generating diverse on-manifold samples. The proposed VITA consists of two complementary parts: tangent transfer and integration of multi-source vicinal samples. The tangent transfer creates initial augmented samples for improving corruption robustness. The integration employs a generative model to characterize the underlying manifold built by vicinal samples, facilitating the generation of on-manifold samples. Our proposed VITA significantly outperforms the current state-of-the-art augmentation methods, demonstrated in extensive experiments on corruption benchmarks.

----

## [36] TransZero: Attribute-Guided Transformer for Zero-Shot Learning

**Authors**: *Shiming Chen, Ziming Hong, Yang Liu, Guo-Sen Xie, Baigui Sun, Hao Li, Qinmu Peng, Ke Lu, Xinge You*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19909](https://doi.org/10.1609/aaai.v36i1.19909)

**Abstract**:

Zero-shot learning (ZSL) aims to recognize novel classes by transferring semantic knowledge from seen classes to unseen ones. Semantic knowledge is learned from attribute descriptions shared between different classes, which are strong prior for localization of object attribute for representing discriminative region features enabling significant visual-semantic interaction. Although few attention-based models have attempted to learn such region features in a single image, the transferability and discriminative attribute localization of visual features are typically neglected. In this paper, we propose an attribute-guided Transformer network to learn the attribute localization for discriminative visual-semantic embedding representations in ZSL, termed TransZero. Specifically, TransZero takes a feature augmentation encoder to alleviate the cross-dataset bias between ImageNet and ZSL benchmarks and improve the transferability of visual features by reducing the entangled relative geometry relationships among region features. To learn locality-augmented visual features, TransZero employs a visual-semantic decoder to localize the most relevant image regions to each attributes from a given image under the guidance of attribute semantic information. Then, the locality-augmented visual features and semantic vectors are used for conducting effective visual-semantic interaction in a visual-semantic embedding network. Extensive experiments show that TransZero achieves a new state-of-the-art on three ZSL benchmarks. The codes are available at: https://github.com/shiming-chen/TransZero.

----

## [37] Structured Semantic Transfer for Multi-Label Recognition with Partial Labels

**Authors**: *Tianshui Chen, Tao Pu, Hefeng Wu, Yuan Xie, Liang Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19910](https://doi.org/10.1609/aaai.v36i1.19910)

**Abstract**:

Multi-label image recognition is a fundamental yet practical task because real-world images inherently possess multiple semantic labels. However, it is difficult to collect large-scale multi-label annotations due to the complexity of both the input images and output label spaces. To reduce the annotation cost, we propose a structured semantic transfer (SST) framework that enables training multi-label recognition models with partial labels, i.e., merely some labels are known while other labels are missing (also called unknown labels) per image. The framework consists of two complementary transfer modules that explore within-image and cross-image semantic correlations to transfer knowledge of known labels to generate pseudo labels for unknown labels. Specifically, an intra-image semantic transfer module learns image-specific label co-occurrence matrix and maps the known labels to complement unknown labels based on this matrix. Meanwhile, a cross-image transfer module learns category-specific feature similarities and helps complement unknown labels with high similarities. Finally, both known and generated labels are used to train the multi-label recognition models. Extensive experiments on the Microsoft COCO, Visual Genome and Pascal VOC datasets show that the proposed SST framework obtains superior performance over current state-of-the-art algorithms. Codes are available at https://github.com/HCPLab-SYSU/HCP-MLR-PL.

----

## [38] SJDL-Vehicle: Semi-supervised Joint Defogging Learning for Foggy Vehicle Re-identification

**Authors**: *Wei-Ting Chen, I-Hsiang Chen, Chih-Yuan Yeh, Hao-Hsiang Yang, Jian-Jiun Ding, Sy-Yen Kuo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19911](https://doi.org/10.1609/aaai.v36i1.19911)

**Abstract**:

Vehicle re-identification (ReID) has attracted considerable attention in computer vision. Although several methods have been proposed to achieve state-of-the-art performance on this topic, re-identifying vehicle in foggy scenes remains a great challenge due to the degradation of visibility. To our knowledge, this problem is still not well-addressed so far. In this paper, to address this problem, we propose a novel training framework called Semi-supervised Joint Defogging Learning (SJDL) framework. First, the fog removal branch and the re-identification branch are integrated to perform simultaneous training. With the collaborative training scheme, defogged features generated by the defogging branch from input images can be shared to learn better representation for the re-identification branch. However, since the fog-free image of real-world data is intractable, this architecture can only be trained on the synthetic data, which may cause the domain gap problem between real-world and synthetic scenarios. To solve this problem, we design a semi-supervised defogging training scheme that can train two kinds of data alternatively in each iteration. Due to the lack of a dataset specialized for vehicle ReID in the foggy weather, we construct a dataset called FVRID which consists of real-world and synthetic foggy images to train and evaluate the performance. Experimental results show that the proposed method is effective and outperforms other existing vehicle ReID methods in the foggy weather. The code and dataset are available in https://github.com/Cihsaing/SJDL-Foggy-Vehicle-Re-Identification--AAAI2022.

----

## [39] Imagine by Reasoning: A Reasoning-Based Implicit Semantic Data Augmentation for Long-Tailed Classification

**Authors**: *Xiaohua Chen, Yucan Zhou, Dayan Wu, Wanqian Zhang, Yu Zhou, Bo Li, Weiping Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19912](https://doi.org/10.1609/aaai.v36i1.19912)

**Abstract**:

Real-world data often follows a long-tailed distribution, which makes the performance of existing classification algorithms degrade heavily. A key issue is that the samples in tail categories fail to depict their intra-class diversity. Humans can imagine a sample in new poses, scenes and view angles with their prior knowledge even if it is the first time to see this category. Inspired by this, we propose a novel reasoning-based implicit semantic data augmentation method to borrow transformation directions from other classes. Since the covariance matrix of each category represents the feature transformation directions, we can sample new directions from similar categories to generate definitely different instances. Specifically, the long-tailed distributed data is first adopted to train a backbone and a classifier. Then, a covariance matrix for each category is estimated, and a knowledge graph is constructed to store the relations of any two categories. Finally, tail samples are adaptively enhanced via propagating information from all the similar categories in the knowledge graph. Experimental results on CIFAR-LT-100, ImageNet-LT, and iNaturalist 2018 have demonstrated the effectiveness of our proposed method compared with the state-of-the-art methods.

----

## [40] Guide Local Feature Matching by Overlap Estimation

**Authors**: *Ying Chen, Dihe Huang, Shang Xu, Jianlin Liu, Yong Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19913](https://doi.org/10.1609/aaai.v36i1.19913)

**Abstract**:

Local image feature matching under large appearance, viewpoint, and distance changes is challenging yet important. Conventional methods detect and match tentative local features across the whole images, with heuristic consistency checks to guarantee reliable matches. In this paper, we introduce a novel Overlap Estimation method conditioned on image pairs with TRansformer, named OETR, to constrain local feature matching in the commonly visible region. OETR performs overlap estimation in a two step process of feature correlation and then overlap regression. As a preprocessing module, OETR can be plugged into any existing local feature detection and matching pipeline, to mitigate potential view angle or scale variance. Intensive experiments show that OETR can boost state of the art local feature matching performance substantially, especially for image pairs with small shared regions. The code will be publicly available at https://github.com/AbyssGaze/OETR.

----

## [41] Causal Intervention for Subject-Deconfounded Facial Action Unit Recognition

**Authors**: *Yingjie Chen, Diqi Chen, Tao Wang, Yizhou Wang, Yun Liang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19914](https://doi.org/10.1609/aaai.v36i1.19914)

**Abstract**:

Subject-invariant facial action unit (AU) recognition remains challenging for the reason that the data distribution varies among subjects. In this paper, we propose a causal inference framework for subject-invariant facial action unit recognition. To illustrate the causal effect existing in AU recognition task, we formulate the causalities among facial images, subjects, latent AU semantic relations, and estimated AU occurrence probabilities via a structural causal model. By constructing such a causal diagram, we clarify the causal-effect among variables and propose a plug-in causal intervention module, CIS, to deconfound the confounder Subject in the causal diagram. Extensive experiments conducted on two commonly used AU benchmark datasets, BP4D and DISFA, show the effectiveness of our CIS, and the model with CIS inserted, CISNet, has achieved state-of-the-art performance.

----

## [42] Deep One-Class Classification via Interpolated Gaussian Descriptor

**Authors**: *Yuanhong Chen, Yu Tian, Guansong Pang, Gustavo Carneiro*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19915](https://doi.org/10.1609/aaai.v36i1.19915)

**Abstract**:

One-class classification (OCC) aims to learn an effective data description to enclose all normal training samples and detect anomalies based on the deviation from the data description. Current state-of-the-art OCC models learn a compact normality description by hyper-sphere minimisation, but they often suffer from overfitting the training data, especially when the training set is small or contaminated with anomalous samples. To address this issue, we introduce the interpolated Gaussian descriptor (IGD) method, a novel OCC model that learns a one-class Gaussian anomaly classifier trained with adversarially interpolated training samples. The Gaussian anomaly classifier differentiates the training samples based on their distance to the Gaussian centre and the standard deviation of these distances, offering the model a discriminability w.r.t. the given samples during training. The adversarial interpolation is enforced to consistently learn a smooth Gaussian descriptor, even when the training data is small or contaminated with anomalous samples. This enables our model to learn the data description based on the representative normal samples rather than fringe or anomalous samples, resulting in significantly improved normality description. In extensive experiments on diverse popular benchmarks, including MNIST, Fashion MNIST, CIFAR10, MVTec AD and two medical datasets, IGD achieves better detection accuracy than current state-of-the-art models. IGD also shows better robustness in problems with small or contaminated training sets.

----

## [43] Towards Ultra-Resolution Neural Style Transfer via Thumbnail Instance Normalization

**Authors**: *Zhe Chen, Wenhai Wang, Enze Xie, Tong Lu, Ping Luo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19916](https://doi.org/10.1609/aaai.v36i1.19916)

**Abstract**:

We present an extremely simple Ultra-Resolution Style Transfer framework, termed URST, to flexibly process arbitrary high-resolution images (e.g., 10000x10000 pixels) style transfer for the first time. Most of the existing state-of-the-art methods would fall short due to massive memory cost and small stroke size when processing ultra-high resolution images. URST completely avoids the memory problem caused by ultra-high resolution images by (1) dividing the image into small patches and (2) performing patch-wise style transfer with a novel Thumbnail Instance Normalization (TIN). Specifically, TIN can extract thumbnail features' normalization statistics and apply them to small patches, ensuring the style consistency among different patches.
 Overall, the URST framework has three merits compared to prior arts. (1) We divide input image into small patches and adopt TIN, successfully transferring image style with arbitrary high-resolution. (2) Experiments show that our URST surpasses existing SOTA methods on ultra-high resolution images benefiting from the effectiveness of the proposed stroke perceptual loss in enlarging the stroke size. (3) Our URST can be easily plugged into most existing style transfer methods and directly improve their performance even without training. Code is available at https://git.io/URST.

----

## [44] DeTarNet: Decoupling Translation and Rotation by Siamese Network for Point Cloud Registration

**Authors**: *Zhi Chen, Fan Yang, Wenbing Tao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19917](https://doi.org/10.1609/aaai.v36i1.19917)

**Abstract**:

Point cloud registration is a fundamental step for many tasks. In this paper, we propose a neural network named DetarNet to decouple the translation t and rotation R, so as to overcome the performance degradation due to their mutual interference in point cloud registration. First, a Siamese Network based Progressive and Coherent Feature Drift (PCFD) module is proposed to align the source and target points in high-dimensional feature space, and accurately recover translation from the alignment process. Then we propose a Consensus Encoding Unit (CEU) to construct more distinguishable features for a set of putative correspondences. After that, a Spatial and Channel Attention (SCA) block is adopted to build a classification network for finding good correspondences. Finally, the rotation is obtained by Singular Value Decomposition (SVD). In this way, the proposed network decouples the estimation of translation and rotation, resulting in better performance for both of them. Experimental results demonstrate that the proposed DetarNet improves registration performance on both indoor and outdoor scenes. Our code will be available in https://github.com/ZhiChen902/DetarNet.

----

## [45] LCTR: On Awakening the Local Continuity of Transformer for Weakly Supervised Object Localization

**Authors**: *Zhiwei Chen, Changan Wang, Yabiao Wang, Guannan Jiang, Yunhang Shen, Ying Tai, Chengjie Wang, Wei Zhang, Liujuan Cao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19918](https://doi.org/10.1609/aaai.v36i1.19918)

**Abstract**:

Weakly supervised object localization (WSOL) aims to learn object localizer solely by using image-level labels. The convolution neural network (CNN) based techniques often result in highlighting the most discriminative part of objects while ignoring the entire object extent. Recently, the transformer architecture has been deployed to WSOL to capture the long-range feature dependencies with self-attention mechanism and multilayer perceptron structure. Nevertheless, transformers lack the locality inductive bias inherent to CNNs and therefore may deteriorate local feature details in WSOL. In this paper, we propose a novel framework built upon the transformer, termed LCTR (Local Continuity TRansformer), which targets at enhancing the local perception capability of global features among long-range feature dependencies. To this end, we propose a relational patch-attention module (RPAM), which considers cross-patch information on a global basis. We further design a cue digging module (CDM), which utilizes local features to guide the learning trend of the model for highlighting the weak local responses. Finally, comprehensive experiments are carried out on two widely used datasets, ie, CUB-200-2011 and ILSVRC, to verify the effectiveness of our method.

----

## [46] Efficient Virtual View Selection for 3D Hand Pose Estimation

**Authors**: *Jian Cheng, Yanguang Wan, Dexin Zuo, Cuixia Ma, Jian Gu, Ping Tan, Hongan Wang, Xiaoming Deng, Yinda Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19919](https://doi.org/10.1609/aaai.v36i1.19919)

**Abstract**:

3D hand pose estimation from single depth is a fundamental problem in computer vision, and has wide applications. However, the existing methods still can not achieve satisfactory hand pose estimation results due to view variation and occlusion of human hand. In this paper, we propose a new virtual view selection and fusion module for 3D hand pose estimation from single depth. We propose to automatically select multiple virtual viewpoints for pose estimation and fuse the results of all and find this empirically delivers accurate and robust pose estimation. In order to select most effective virtual views for pose fusion, we evaluate the virtual views based on the confidence of virtual views using a light-weight network via network distillation. Experiments on three main benchmark datasets including NYU, ICVL and Hands2019 demonstrate that our method outperforms the state-of-the-arts on NYU and ICVL, and achieves very competitive performance on Hands2019-Task1, and our proposed virtual view selection and fusion module is both effective for 3D hand pose estimation.

----

## [47] Pose Adaptive Dual Mixup for Few-Shot Single-View 3D Reconstruction

**Authors**: *Ta Ying Cheng, Hsuan-Ru Yang, Niki Trigoni, Hwann-Tzong Chen, Tyng-Luh Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19920](https://doi.org/10.1609/aaai.v36i1.19920)

**Abstract**:

We present a pose adaptive few-shot learning procedure and a two-stage data interpolation regularization, termed Pose Adaptive Dual Mixup (PADMix), for single-image 3D reconstruction. While augmentations via interpolating feature-label pairs are effective in classification tasks, they fall short in shape predictions potentially due to inconsistencies between interpolated products of two images and volumes when rendering viewpoints are unknown. PADMix targets this issue with two sets of mixup procedures performed sequentially. We first perform an input mixup which, combined with a pose adaptive learning procedure, is helpful in learning 2D feature extraction and pose adaptive latent encoding. The stagewise training allows us to build upon the pose invariant representations to perform a follow-up latent mixup under one-to-one correspondences between features and ground-truth volumes. PADMix significantly outperforms previous literature on few-shot settings over the ShapeNet dataset and sets new benchmarks on the more challenging real-world Pix3D dataset.

----

## [48] PureGaze: Purifying Gaze Feature for Generalizable Gaze Estimation

**Authors**: *Yihua Cheng, Yiwei Bao, Feng Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19921](https://doi.org/10.1609/aaai.v36i1.19921)

**Abstract**:

Gaze estimation methods learn eye gaze from facial features. However, among rich information in the facial image, real gaze-relevant features only correspond to subtle changes in eye region, while other gaze-irrelevant features like illumination, personal appearance and even facial expression may affect the learning in an unexpected way. This is a major reason why existing methods show significant performance degradation in cross-domain/dataset evaluation. In this paper, we tackle the cross-domain problem in gaze estimation. Different from common domain adaption methods, we propose a domain generalization method to improve the cross-domain performance without touching target samples. The domain generalization is realized by gaze feature purification. We eliminate gaze-irrelevant factors such as illumination and identity to improve the cross-domain performance. We design a plug-and-play self-adversarial framework for the gaze feature purification. The framework enhances not only our baseline but also existing gaze estimation methods directly and significantly. To the best of our knowledge, we are the first to propose domain generalization methods in gaze estimation. Our method achieves not only state-of-the-art performance among typical gaze estimation methods but also competitive results among domain adaption methods. The code is released in https://github.com/yihuacheng/PureGaze.

----

## [49] (25+1)D Spatio-Temporal Scene Graphs for Video Question Answering

**Authors**: *Anoop Cherian, Chiori Hori, Tim K. Marks, Jonathan Le Roux*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19922](https://doi.org/10.1609/aaai.v36i1.19922)

**Abstract**:

Spatio-temporal scene-graph approaches to video-based reasoning tasks, such as video question-answering (QA), typically construct such graphs for every video frame. These approaches often ignore the fact that videos are essentially sequences of 2D ``views'' of events happening in a 3D space, and that the semantics of the 3D scene can thus be carried over from frame to frame. Leveraging this insight, we propose a (2.5+1)D scene graph representation to better capture the spatio-temporal information flows inside the videos. Specifically, we first create a 2.5D (pseudo-3D) scene graph by transforming every 2D frame to have an inferred 3D structure using an off-the-shelf 2D-to-3D transformation module, following which we register the video frames into a shared (2.5+1)D spatio-temporal space and ground each 2D scene graph within it. Such a (2.5+1)D graph is then segregated into a static sub-graph and a dynamic sub-graph, corresponding to whether the objects within them usually move in the world. The nodes in the dynamic graph are enriched with motion features capturing their interactions with other graph nodes. Next, for the video QA task, we present a novel transformer-based reasoning pipeline that embeds the (2.5+1)D graph into a spatio-temporal hierarchical latent space, where the sub-graphs and their interactions are captured at varied granularity. To demonstrate the effectiveness of our approach, we present experiments on the NExT-QA and AVSD-QA datasets. Our results show that our proposed (2.5+1)D representation leads to faster training and inference, while our hierarchical model showcases superior performance on the video QA task versus the state of the art.

----

## [50] Event-Image Fusion Stereo Using Cross-Modality Feature Propagation

**Authors**: *Hoonhee Cho, Kuk-Jin Yoon*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19923](https://doi.org/10.1609/aaai.v36i1.19923)

**Abstract**:

Event cameras asynchronously output the polarity values of pixel-level log intensity alterations. They are robust against motion blur and can be adopted in challenging light conditions. Owing to these advantages, event cameras have been employed in various vision tasks such as depth estimation, visual odometry, and object detection. In particular, event cameras are effective in stereo depth estimation to ﬁnd correspondence points between two cameras under challenging illumination conditions and/or fast motion. However, because event cameras provide spatially sparse event stream data, it is difﬁcult to obtain a dense disparity map. Although it is possible to estimate disparity from event data at the edge of a structure where intensity changes are likely to occur, estimating the disparity in a region where event occurs rarely is challenging. In this study, we propose a deep network that combines the features of an image with the features of an event to generate a dense disparity map. The proposed network uses images to obtain spatially dense features that are lacking in events. In addition, we propose a spatial multi-scale correlation between two fused feature maps for an accurate disparity map. To validate our method, we conducted experiments using synthetic and real-world datasets.

----

## [51] Style-Guided and Disentangled Representation for Robust Image-to-Image Translation

**Authors**: *Jaewoong Choi, Dae Ha Kim, Byung Cheol Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19924](https://doi.org/10.1609/aaai.v36i1.19924)

**Abstract**:

Recently, various image-to-image translation (I2I) methods have improved mode diversity and visual quality in terms of neural networks or regularization terms. However, conventional I2I methods relies on a static decision boundary and the encoded representations in those methods are entangled with each other, so they often face with ‘mode collapse’ phenomenon. To mitigate mode collapse, 1) we design a so-called style-guided discriminator that guides an input image to the target image style based on the strategy of flexible decision boundary. 2) Also, we make the encoded representations include independent domain attributes. Based on two ideas, this paper proposes Style-Guided and Disentangled Representation for Robust Image-to-Image Translation (SRIT). SRIT showed outstanding FID by 8%, 22.8%, and 10.1% for CelebA-HQ, AFHQ, and Yosemite datasets, respectively. The translated images of SRIT reflect the styles of target domain successfully. This indicates that SRIT shows better mode diversity than previous works.

----

## [52] Denoised Maximum Classifier Discrepancy for Source-Free Unsupervised Domain Adaptation

**Authors**: *Tong Chu, Yahao Liu, Jinhong Deng, Wen Li, Lixin Duan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19925](https://doi.org/10.1609/aaai.v36i1.19925)

**Abstract**:

Source-Free Unsupervised Domain Adaptation(SFUDA) aims to adapt a pre-trained source model to an unlabeled target domain without access to the original labeled source domain samples. Many existing SFUDA approaches apply the self-training strategy, which involves iteratively selecting confidently predicted target samples as pseudo-labeled samples used to train the model to fit the target domain. However, the self-training strategy may also suffer from  sample selection bias and be impacted by the label noise of the pseudo-labeled samples. In this work, we provide a rigorous theoretical analysis on how these two issues affect the model generalization ability when applying the self-training strategy for the SFUDA problem. Based on this theoretical analysis, we then propose a new Denoised Maximum Classifier Discrepancy (D-MCD) method for SFUDA to effectively address these two issues. In particular, we first minimize the distribution mismatch between the selected pseudo-labeled samples and the remaining target domain samples to alleviate the sample selection bias. Moreover, we design a strong-weak self-training paradigm to denoise the selected pseudo-labeled samples, where the strong network is used to select pseudo-labeled samples while the weak network helps the strong network to filter out hard samples to avoid incorrect labels. In this way, we are able to ensure both the quality of the pseudo-labels and the generalization ability of the trained model on the target domain. We achieve state-of-the-art results on three domain adaptation benchmark datasets, which clearly validates the effectiveness of our proposed approach. Full code is available at https://github.com/kkkkkkon/D-MCD.

----

## [53] Model-Based Image Signal Processors via Learnable Dictionaries

**Authors**: *Marcos V. Conde, Steven McDonagh, Matteo Maggioni, Ales Leonardis, Eduardo Pérez-Pellitero*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19926](https://doi.org/10.1609/aaai.v36i1.19926)

**Abstract**:

Digital cameras transform sensor RAW readings into RGB images by means of their Image Signal Processor (ISP). Computational photography tasks such as image denoising and colour constancy are commonly performed in the RAW domain, in part due to the inherent hardware design, but also due to the appealing simplicity of noise statistics that result from the direct sensor readings. Despite this, the availability of RAW images is limited in comparison with the abundance and diversity of available RGB data. Recent approaches have attempted to bridge this gap by estimating the RGB to RAW mapping: handcrafted model-based methods that are interpretable and controllable usually require manual parameter fine-tuning, while end-to-end learnable neural networks require large amounts of training data, at times with complex training procedures, and generally lack interpretability and parametric control. Towards addressing these existing limitations, we present a novel hybrid model-based and data-driven ISP that builds on canonical ISP operations and is both learnable and interpretable. Our proposed invertible model, capable of bidirectional mapping between RAW and RGB domains, employs end-to-end learning of rich parameter representations, i.e. dictionaries, that are free from direct parametric supervision and additionally enable simple and plausible data augmentation. We evidence the value of our data generation process by extensive experiments under both RAW image reconstruction and RAW image denoising tasks, obtaining state-of-the-art performance in both. Additionally, we show that our ISP can learn meaningful mappings from few data samples, and that denoising models trained with our dictionary-based data augmentation are competitive despite having only few or zero ground-truth labels.

----

## [54] MMA: Multi-Camera Based Global Motion Averaging

**Authors**: *Hainan Cui, Shuhan Shen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19927](https://doi.org/10.1609/aaai.v36i1.19927)

**Abstract**:

In order to fully perceive the surrounding environment, many intelligent robots and self-driving cars are equipped with a multi-camera system. Based on this system, the structure-from-motion (SfM) technology is used to realize scene reconstruction, but the fixed relative poses between cameras in the multi-camera system are usually not considered. This paper presents a tailor-made multi-camera based motion averaging system, where the fixed relative poses are utilized to improve the accuracy and robustness of SfM. Our approach starts by dividing the images into reference images and non-reference images, and edges in view-graph are divided into four categories accordingly. Then, a multi-camera based rotating averaging problem is formulated and solved in two stages, where an iterative re-weighted least squares scheme is used to deal with outliers. Finally, a multi-camera based translation averaging problem is formulated and a l1-norm based optimization scheme is proposed to compute the relative translations of multi-camera system and reference camera positions simultaneously. Experiments demonstrate that our algorithm achieves superior accuracy and robustness on various data sets compared to the state-of-the-art methods.

----

## [55] GenCo: Generative Co-training for Generative Adversarial Networks with Limited Data

**Authors**: *Kaiwen Cui, Jiaxing Huang, Zhipeng Luo, Gongjie Zhang, Fangneng Zhan, Shijian Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19928](https://doi.org/10.1609/aaai.v36i1.19928)

**Abstract**:

Training effective Generative Adversarial Networks (GANs) requires large amounts of training data, without which the trained models are usually sub-optimal with discriminator over-fitting. Several prior studies address this issue by expanding the distribution of the limited training data via massive and hand-crafted data augmentation. We handle data-limited image generation from a very different perspective. Specifically, we design GenCo, a Generative Co-training network that mitigates the discriminator over-fitting issue by introducing multiple complementary discriminators that provide diverse supervision from multiple distinctive views in training. We instantiate the idea of GenCo in two ways. The first way is Weight-Discrepancy Co-training (WeCo) which co-trains multiple distinctive discriminators by diversifying their parameters. The second way is Data-Discrepancy Co-training (DaCo) which achieves co-training by feeding discriminators with different views of the input images. Extensive experiments over multiple benchmarks show that GenCo achieves superior generation with limited training data. In addition, GenCo also complements the augmentation approach with consistent and clear performance gains when combined.

----

## [56] Unbiased IoU for Spherical Image Object Detection

**Authors**: *Feng Dai, Bin Chen, Hang Xu, Yike Ma, Xiaodong Li, Bailan Feng, Peng Yuan, Chenggang Yan, Qiang Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19929](https://doi.org/10.1609/aaai.v36i1.19929)

**Abstract**:

As one of the fundamental components of object detection, intersection-over-union (IoU) calculations between two bounding boxes play an important role in samples selection, NMS operation and evaluation of object detection algorithms. This procedure is well-defined and solved for planar images, while it is challenging for spherical ones. Some existing methods utilize planar bounding boxes to represent spherical objects. However, they are biased due to the distortions of spherical objects. Others use spherical rectangles as unbiased representations, but they adopt excessive approximate algorithms when computing the IoU. In this paper, we propose an unbiased IoU as a novel evaluation criterion for spherical image object detection, which is based on the unbiased representations and utilize unbiased analytical method for IoU calculation. This is the first time that the absolutely accurate IoU calculation is applied to the evaluation criterion, thus object detection algorithms can be correctly evaluated for spherical images. With the unbiased representation and calculation, we also present Spherical CenterNet, an anchor free object detection algorithm for spherical images. The experiments show that our unbiased IoU gives accurate results and the proposed Spherical CenterNet achieves better performance on one real-world and two synthetic spherical object detection datasets than existing methods.

----

## [57] InsCLR: Improving Instance Retrieval with Self-Supervision

**Authors**: *Zelu Deng, Yujie Zhong, Sheng Guo, Weilin Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19930](https://doi.org/10.1609/aaai.v36i1.19930)

**Abstract**:

This work aims at improving instance retrieval with self-supervision. We find that fine-tuning using the recently developed self-supervised learning (SSL) methods, such as SimCLR and MoCo, fails to improve the performance of instance retrieval. In this work, we identify that the learnt representations for instance retrieval should be invariant to large variations in viewpoint and background etc., whereas self-augmented positives applied by the current SSL methods can not provide strong enough signals for learning robust instance-level representations. To overcome this problem, we propose InsCLR, a new SSL method that builds on the instance-level contrast, to learn the intra-class invariance by dynamically mining meaningful pseudo positive samples from both mini-batches and a memory bank during training. Extensive experiments demonstrate that InsCLR achieves similar or even better performance than the state-of-the-art SSL methods on instance retrieval. Code is available at https://github.com/zeludeng/insclr.

----

## [58] Spatio-Temporal Recurrent Networks for Event-Based Optical Flow Estimation

**Authors**: *Ziluo Ding, Rui Zhao, Jiyuan Zhang, Tianxiao Gao, Ruiqin Xiong, Zhaofei Yu, Tiejun Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19931](https://doi.org/10.1609/aaai.v36i1.19931)

**Abstract**:

Event camera has offered promising alternative for visual perception, especially in high speed and high dynamic range scenes. Recently, many deep learning methods have shown great success in providing model-free solutions to many event-based problems, such as optical flow estimation. However, existing deep learning methods did not address the importance of temporal information well from the perspective of architecture design and cannot effectively extract spatio-temporal features. Another line of research that utilizes Spiking Neural Network suffers from training issues for deeper architecture. To address these points, a novel input representation is proposed that captures the events temporal distribution for signal enhancement. Moreover, we introduce a spatio-temporal recurrent encoding-decoding neural network architecture for event-based optical flow estimation, which utilizes Convolutional Gated Recurrent Units to extract feature maps from a series of event images. Besides, our architecture allows some traditional frame-based core modules, such as correlation layer and iterative residual refine scheme, to be incorporated. The network is end-to-end trained with self-supervised learning on the Multi-Vehicle Stereo Event Camera dataset. We have shown that it outperforms all the existing state-of-the-art methods by a large margin.

----

## [59] Construct Effective Geometry Aware Feature Pyramid Network for Multi-Scale Object Detection

**Authors**: *Jinpeng Dong, Yuhao Huang, Songyi Zhang, Shitao Chen, Nanning Zheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19932](https://doi.org/10.1609/aaai.v36i1.19932)

**Abstract**:

Feature Pyramid Network (FPN) has been widely adopted to exploit multi-scale features for scale variation in object detection. However, intrinsic defects in most of the current methods with FPN make it difficult to adapt to the feature of different geometric objects. To address this issue, we introduce geometric prior into FPN to obtain more discriminative features. In this paper, we propose Geometry-aware Feature Pyramid Network (GaFPN), which mainly consists of the novel Geometry-aware Mapping Module and Geometry-aware Predictor Head.The Geometry-aware Mapping Module is proposed to make full use of all pyramid features to obtain better proposal features by the weight-generation subnetwork. The weights generation subnetwork generates fusion weight for each layer proposal features by using the geometric information of the proposal. The Geometry-aware Predictor Head introduces geometric prior into predictor head by the embedding generation network to strengthen feature representation for classification and regression. Our GaFPN can be easily extended to other two-stage object detectors with feature pyramid and applied to instance segmentation task. The proposed GaFPN significantly improves detection performance compared to baseline detectors with ResNet-50-FPN: +1.9, +2.0, +1.7, +1.3, +0.8 points Average Precision (AP) on Faster-RCNN, Cascade R-CNN, Dynamic R-CNN, SABL, and AugFPN respectively on MS COCO dataset.

----

## [60] Complementary Attention Gated Network for Pedestrian Trajectory Prediction

**Authors**: *Jinghai Duan, Le Wang, Chengjiang Long, Sanping Zhou, Fang Zheng, Liushuai Shi, Gang Hua*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19933](https://doi.org/10.1609/aaai.v36i1.19933)

**Abstract**:

Pedestrian trajectory prediction is crucial in many practical applications due to the diversity of pedestrian movements, such as social interactions and individual motion behaviors. With similar observable trajectories and social environments, different pedestrians may make completely different future decisions. However, most existing methods only focus on the frequent modal of the trajectory and thus are difficult to generalize to the peculiar scenario, which leads to the decline of the multimodal fitting ability when facing similar scenarios. In this paper, we propose a complementary attention gated network (CAGN) for pedestrian trajectory prediction, in which a dual-path architecture including normal and inverse attention is proposed to capture both frequent and peculiar modals in spatial and temporal patterns, respectively. Specifically, a complementary block is proposed to guide normal and inverse attention, which are then be summed with learnable weights to get attention features by a gated network. Finally, multiple trajectory distributions are estimated based on the fused spatio-temporal attention features due to the multimodality of future trajectory. Experimental results on benchmark datasets, i.e., the ETH, and the UCY, demonstrate that our method outperforms state-of-the-art methods by 13.8% in Average Displacement Error (ADE) and 10.4% in Final Displacement Error (FDE). Code will be available at https://github.com/jinghaiD/CAGN

----

## [61] SVT-Net: Super Light-Weight Sparse Voxel Transformer for Large Scale Place Recognition

**Authors**: *Zhaoxin Fan, Zhenbo Song, Hongyan Liu, Zhiwu Lu, Jun He, Xiaoyong Du*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19934](https://doi.org/10.1609/aaai.v36i1.19934)

**Abstract**:

Simultaneous Localization and Mapping (SLAM) and Autonomous Driving are becoming increasingly more important in recent years. Point cloud-based large scale place recognition is the spine of them. While many models have been proposed and have achieved acceptable performance by learning short-range local features, they always skip long-range contextual properties. Moreover, the model size also becomes a serious shackle for their wide applications. To overcome these challenges, we propose a super light-weight network model termed SVT-Net. On top of the highly efficient 3D Sparse Convolution (SP-Conv), an Atom-based Sparse Voxel Transformer (ASVT) and a Cluster-based Sparse Voxel Transformer (CSVT) are proposed respectively to learn both short-range local features and long-range contextual features. Consisting of ASVT and CSVT, SVT-Net can achieve state-of-the-art performance in terms of both recognition accuracy and running speed with a super-light model size (0.9M parameters). Meanwhile, for the purpose of further boosting efficiency, we introduce two simplified versions, which also achieve state-of-the-art performance and further reduce the model size to 0.8M and 0.4M respectively.

----

## [62] Backdoor Attacks on the DNN Interpretation System

**Authors**: *Shihong Fang, Anna Choromanska*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19935](https://doi.org/10.1609/aaai.v36i1.19935)

**Abstract**:

Interpretability is crucial to understand the inner workings of deep neural networks (DNNs). Many interpretation methods help to understand the decision-making of DNNs by generating saliency maps that highlight parts of the input image that contribute the most to the prediction made by the DNN. In this paper we design a backdoor attack that alters the saliency map produced by the network for an input image with a specific trigger pattern while not losing the prediction performance significantly. The saliency maps are incorporated in the penalty term of the objective function that is used to train a deep model and its influence on model training is conditioned upon the presence of a trigger. We design two types of attacks: a targeted attack that enforces a specific modification of the saliency map and a non-targeted attack when the importance scores of the top pixels from the original saliency map are significantly reduced. We perform empirical evaluations of the proposed backdoor attacks on gradient-based interpretation methods, Grad-CAM and SimpleGrad, and a gradient-free scheme, VisualBackProp, for a variety of deep learning architectures. We show that our attacks constitute a serious security threat to the reliability of the interpretation methods when deploying models developed by untrusted sources. We furthermore show that existing backdoor defense mechanisms are ineffective in detecting our attacks. Finally, we demonstrate that the proposed methodology can be used in an inverted setting, where the correct saliency map can be obtained only in the presence of a trigger (key), effectively making the interpretation system available only to selected users.

----

## [63] Learning to Learn Transferable Attack

**Authors**: *Shuman Fang, Jie Li, Xianming Lin, Rongrong Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19936](https://doi.org/10.1609/aaai.v36i1.19936)

**Abstract**:

Transfer adversarial attack is a non-trivial black-box adversarial attack that aims to craft adversarial perturbations on the surrogate model and then apply such perturbations to the victim model. However, the transferability of perturbations from existing methods is still limited, since the adversarial perturbations are easily overﬁtting with a single surrogate model and speciﬁc data pattern. In this paper, we propose a Learning to Learn Transferable Attack (LLTA) method, which makes the adversarial perturbations more generalized via learning from both data and model augmentation. For data augmentation, we adopt simple random resizing and padding. For model augmentation, we randomly alter the back propagation instead of the forward propagation to eliminate the effect on the model prediction. By treating the attack of both speciﬁc data and a modiﬁed model as a task, we expect the adversarial perturbations to adopt enough tasks for generalization. To this end, the meta-learning algorithm is further introduced during the iteration of perturbation generation. Empirical results on the widely-used dataset demonstrate the effectiveness of our attack method with a 12.85% higher success rate of transfer attack compared with the state-of-the-art methods. We also evaluate our method on the real-world online system, i.e., Google Cloud Vision API, to further show the practical potentials of our method.

----

## [64] Perceptual Quality Assessment of Omnidirectional Images

**Authors**: *Yuming Fang, Liping Huang, Jiebin Yan, Xuelin Liu, Yang Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19937](https://doi.org/10.1609/aaai.v36i1.19937)

**Abstract**:

Omnidirectional images, also called 360◦images, have attracted extensive attention in recent years, due to the rapid development of virtual reality (VR) technologies. During omnidirectional image processing including capture, transmission, consumption, and so on, measuring the perceptual quality of omnidirectional images is highly desired, since it plays a great role in guaranteeing the immersive quality of experience (IQoE). In this paper, we conduct a comprehensive study on the perceptual quality of omnidirectional images from both subjective and objective perspectives. Specifically, we construct the largest so far subjective omnidirectional image quality database, where we consider several key influential elements, i.e., realistic non-uniform distortion, viewing condition, and viewing behavior, from the user view. In addition to subjective quality scores, we also record head and eye movement data. Besides, we make the first attempt by using the proposed database to train a convolutional neural network (CNN) for blind omnidirectional image quality assessment. To be consistent with the human viewing behavior in the VR device, we extract viewports from each omnidirectional image and incorporate the user viewing conditions naturally in the proposed model. The proposed model is composed of two parts, including a multi-scale CNN-based feature extraction module and a perceptual quality prediction module. The feature extraction module is used to incorporate the multi-scale features, and the perceptual quality prediction module is designed to regress them to perceived quality scores. The experimental results on our database verify that the proposed model achieves the competing performance compared with the state-of-the-art methods.

----

## [65] PatchUp: A Feature-Space Block-Level Regularization Technique for Convolutional Neural Networks

**Authors**: *Mojtaba Faramarzi, Mohammad Amini, Akilesh Badrinaaraayanan, Vikas Verma, Sarath Chandar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19938](https://doi.org/10.1609/aaai.v36i1.19938)

**Abstract**:

Large capacity deep learning models are often prone to a high generalization gap when trained with a limited amount of labeled training data. A recent class of methods to address this problem uses various ways to construct a new training sample by mixing a pair (or more) of training samples. We propose PatchUp, a hidden state block-level regularization technique for Convolutional Neural Networks (CNNs), that is applied on selected contiguous blocks of feature maps from a random pair of samples. Our approach improves the robustness of CNN models against the manifold intrusion problem that may occur in other state-of-the-art mixing approaches. Moreover, since we are mixing the contiguous block of features in the hidden space, which has more dimensions than the input space, we obtain more diverse samples for training towards different dimensions. Our experiments on CIFAR10/100, SVHN, Tiny-ImageNet, and ImageNet using ResNet architectures including PreActResnet18/34, WRN-28-10, ResNet101/152 models show that PatchUp improves upon, or equals, the performance of current state-of-the-art regularizers for CNNs. We also show that PatchUp can provide a better generalization to deformed samples and is more robust against adversarial attacks.

----

## [66] DuMLP-Pin: A Dual-MLP-Dot-Product Permutation-Invariant Network for Set Feature Extraction

**Authors**: *Jiajun Fei, Ziyu Zhu, Wenlei Liu, Zhidong Deng, Mingyang Li, Huanjun Deng, Shuo Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19939](https://doi.org/10.1609/aaai.v36i1.19939)

**Abstract**:

Existing permutation-invariant methods can be divided into two categories according to the aggregation scope, i.e. global aggregation and local one. Although the global aggregation methods, e. g., PointNet and Deep Sets, get involved in simpler structures, their performance is poorer than the local aggregation ones like PointNet++ and Point Transformer. It remains an open problem whether there exists a global aggregation method with a simple structure, competitive performance, and even much fewer parameters. In this paper, we propose a novel global aggregation permutation-invariant network based on dual MLP dot-product, called DuMLP-Pin, which is capable of being employed to extract features for set inputs, including unordered or unstructured pixel, attribute, and point cloud data sets. We strictly prove that any permutation-invariant function implemented by DuMLP-Pin can be decomposed into two or more permutation-equivariant ones in a dot-product way as the cardinality of the given input set is greater than a threshold. We also show that the DuMLP-Pin can be viewed as Deep Sets with strong constraints under certain conditions. The performance of DuMLP-Pin is evaluated on several different tasks with diverse data sets. The experimental results demonstrate that our DuMLP-Pin achieves the best results on the two classification problems for pixel sets and attribute sets. On both the point cloud classification and the part segmentation, the accuracy of DuMLP-Pin is very close to the so-far best-performing local aggregation method with only a 1-2% difference, while the number of required parameters is significantly reduced by more than 85% in classification and 69% in segmentation, respectively. The code is publicly available on https://github.com/JaronTHU/DuMLP-Pin.

----

## [67] Attention-Aligned Transformer for Image Captioning

**Authors**: *Zhengcong Fei*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19940](https://doi.org/10.1609/aaai.v36i1.19940)

**Abstract**:

Recently, attention-based image captioning models, which are expected to ground correct image regions for proper word generations, have achieved remarkable performance. However, some researchers have argued “deviated focus” problem of existing attention mechanisms in determining the effective and influential image features. In this paper, we present A2 - an attention-aligned Transformer for image captioning, which guides attention learning in a perturbation-based self-supervised manner, without any annotation overhead. Specifically, we add mask operation on image regions through a learnable network to estimate the true function in ultimate description generation. We hypothesize that the necessary image region features, where small disturbance causes an obvious performance degradation, deserve more attention weight. Then, we propose four aligned strategies to use this information to refine attention weight distribution. Under such a pattern, image regions are attended correctly with the output words. Extensive experiments conducted on the MS COCO dataset demonstrate that the proposed A2 Transformer consistently outperforms baselines in both automatic metrics and human evaluation. Trained models and code for reproducing the experiments are publicly available.

----

## [68] Model Doctor: A Simple Gradient Aggregation Strategy for Diagnosing and Treating CNN Classifiers

**Authors**: *Zunlei Feng, Jiacong Hu, Sai Wu, Xiaotian Yu, Jie Song, Mingli Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19941](https://doi.org/10.1609/aaai.v36i1.19941)

**Abstract**:

Recently, Convolutional Neural Network (CNN) has achieved excellent performance in the classification task. It is widely known that CNN is deemed as a 'blackbox', which is hard for understanding the prediction mechanism and debugging the wrong prediction. Some model debugging and explanation works are developed for solving the above drawbacks. However, those methods focus on explanation and diagnosing possible causes for model prediction, based on which the researchers handle the following optimization of models manually. In this paper, we propose the first completely automatic model diagnosing and treating tool, termed as Model Doctor. Based on two discoveries that 1) each category is only correlated with sparse and specific convolution kernels, and 2) adversarial samples are isolated while normal samples are successive in the feature space, a simple aggregate gradient constraint is devised for effectively diagnosing and optimizing CNN classifiers. The aggregate gradient strategy is a versatile module for mainstream CNN classifiers. Extensive experiments demonstrate that the proposed Model Doctor applies to all existing CNN classifiers, and improves the accuracy of 16 mainstream CNN classifiers by 1%~5%.

----

## [69] OctAttention: Octree-Based Large-Scale Contexts Model for Point Cloud Compression

**Authors**: *Chunyang Fu, Ge Li, Rui Song, Wei Gao, Shan Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19942](https://doi.org/10.1609/aaai.v36i1.19942)

**Abstract**:

In point cloud compression, sufficient contexts are significant for modeling the point cloud distribution. However, the contexts gathered by the previous voxel-based methods decrease when handling sparse point clouds. To address this problem, we propose a multiple-contexts deep learning framework called OctAttention employing the octree structure, a memory-efficient representation for point clouds. Our approach encodes octree symbol sequences in a lossless way by gathering the information of sibling and ancestor nodes. Expressly, we first represent point clouds with octree to reduce spatial redundancy, which is robust for point clouds with different resolutions. We then design a conditional entropy model with a large receptive field that models the sibling and ancestor contexts to exploit the strong dependency among the neighboring nodes and employ an attention mechanism to emphasize the correlated nodes in the context. Furthermore, we introduce a mask operation during training and testing to make a trade-off between encoding time and performance. Compared to the previous state-of-the-art works, our approach obtains a 10%-35% BD-Rate gain on the LiDAR benchmark (e.g. SemanticKITTI) and object point cloud dataset (e.g. MPEG 8i, MVUB), and saves 95% coding time compared to the voxel-based baseline. The code is available at https://github.com/zb12138/OctAttention.

----

## [70] DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents

**Authors**: *Tsu-Jui Fu, William Yang Wang, Daniel McDuff, Yale Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19943](https://doi.org/10.1609/aaai.v36i1.19943)

**Abstract**:

Creating presentation materials requires complex multimodal reasoning skills to summarize key concepts and arrange them in a logical and visually pleasing manner. Can machines learn to emulate this laborious process? We present a novel task and approach for document-to-slide generation. Solving this involves document summarization, image and text retrieval, slide structure and layout prediction to arrange key elements in a form suitable for presentation. We propose a hierarchical sequence-to-sequence approach to tackle our task in an end-to-end manner. Our approach exploits the inherent structures within documents and slides and incorporates paraphrasing and layout prediction modules to generate slides. To help accelerate research in this domain, we release a dataset about 6K paired documents and slide decks used in our experiments. We show that our approach outperforms strong baselines and produces slides with rich content and aligned imagery.

----

## [71] Unsupervised Underwater Image Restoration: From a Homology Perspective

**Authors**: *Zhenqi Fu, Huangxing Lin, Yan Yang, Shu Chai, Liyan Sun, Yue Huang, Xinghao Ding*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19944](https://doi.org/10.1609/aaai.v36i1.19944)

**Abstract**:

Underwater images suffer from degradation due to light scattering and absorption. It remains challenging to restore such degraded images using deep neural networks since real-world paired data is scarcely available while synthetic paired data cannot approximate real-world data perfectly. In this paper, we propose an UnSupervised Underwater Image Restoration method (USUIR) by leveraging the homology property between a raw underwater image and a re-degraded image. Specifically, USUIR first estimates three latent components of the raw underwater image, i.e., the global background light, the transmission map, and the scene radiance (the clean image). Then, a re-degraded image is generated by randomly mixing up the estimated scene radiance and the raw underwater image. We demonstrate that imposing a homology constraint between the raw underwater image and the re-degraded image is equivalent to minimizing the restoration error and hence can be used for the unsupervised restoration. Extensive experiments show that USUIR achieves promising performance in both inference time and restoration quality.

----

## [72] Playing Lottery Tickets with Vision and Language

**Authors**: *Zhe Gan, Yen-Chun Chen, Linjie Li, Tianlong Chen, Yu Cheng, Shuohang Wang, Jingjing Liu, Lijuan Wang, Zicheng Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19945](https://doi.org/10.1609/aaai.v36i1.19945)

**Abstract**:

Large-scale pre-training has recently revolutionized vision-and-language (VL) research. Models such as LXMERT and UNITER have significantly lifted the state of the art over a wide range of VL tasks. However, the large number of parameters in such models hinders their application in practice. In parallel, work on the lottery ticket hypothesis (LTH) has shown that deep neural networks contain small matching subnetworks that can achieve on par or even better performance than the dense networks when trained in isolation. In this work, we perform the first empirical study to assess whether such trainable subnetworks also exist in pre-trained VL models. We use UNITER as the main testbed (also test on LXMERT and ViLT), and consolidate 7 representative VL tasks for experiments, including visual question answering, visual commonsense reasoning, visual entailment, referring expression comprehension, image-text retrieval, GQA, and NLVR2. Through comprehensive analysis, we summarize our main findings as follows. (i) It is difficult to find subnetworks that strictly match the performance of the full model. However, we can find relaxed winning tickets at 50%-70% sparsity that maintain 99% of the full accuracy. (ii) Subnetworks found by task-specific pruning transfer reasonably well to the other tasks, while those found on the pre-training tasks at 60%/70% sparsity transfer universally, matching 98%/96% of the full accuracy on average over all the tasks. (iii) Besides UNITER, other models such as LXMERT and ViLT can also play lottery tickets. However, the highest sparsity we can achieve for ViLT is far lower than LXMERT and UNITER (30% vs. 70%). (iv) LTH also remains relevant when using other training methods (e.g., adversarial training).

----

## [73] Feature Distillation Interaction Weighting Network for Lightweight Image Super-resolution

**Authors**: *Guangwei Gao, Wenjie Li, Juncheng Li, Fei Wu, Huimin Lu, Yi Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19946](https://doi.org/10.1609/aaai.v36i1.19946)

**Abstract**:

Convolutional neural networks based single-image superresolution (SISR) has made great progress in recent years. However, it is difficult to apply these methods to real-world
scenarios due to the computational and memory cost. Meanwhile, how to take full advantage of the intermediate features under the constraints of limited parameters and calculations
is also a huge challenge. To alleviate these issues, we propose a lightweight yet efficient Feature Distillation Interaction Weighted Network (FDIWN). Specifically, FDIWN utilizes a series of specially designed Feature Shuffle Weighted
Groups (FSWG) as the backbone, and several novel mutual Wide-residual Distillation Interaction Blocks (WDIB) form an FSWG. In addition, Wide Identical Residual Weighting
(WIRW) units and Wide Convolutional Residual Weighting (WCRW) units are introduced into WDIB for better feature distillation. Moreover, a Wide-Residual Distillation Connection (WRDC) framework and a Self-Calibration Fusion
(SCF) unit are proposed to interact features with different scales more flexibly and efficiently. Extensive experiments show that our FDIWN is superior to other models to strike a good balance between model performance and efficiency.
The code is available at https://github.com/IVIPLab/FDIWN.

----

## [74] Weakly-Supervised Salient Object Detection Using Point Supervison

**Authors**: *Shuyong Gao, Wei Zhang, Yan Wang, Qianyu Guo, Chenglong Zhang, Yangji He, Wenqiang Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19947](https://doi.org/10.1609/aaai.v36i1.19947)

**Abstract**:

Current state-of-the-art saliency detection models rely heavily on large datasets of accurate pixel-wise annotations, but manually labeling pixels is time-consuming and labor-intensive. There are some weakly supervised methods developed for alleviating the problem, such as image label, bounding box label, and scribble label, while point label still has not been explored in this field. In this paper, we propose a novel weakly-supervised salient object detection method using point supervision. To infer the saliency map, we first design an adaptive masked flood filling algorithm to generate pseudo labels. Then we develop a transformer-based point-supervised saliency detection model to produce the first round of saliency maps. However, due to the sparseness of the label, the weakly supervised model tends to degenerate into a general foreground detection model. To address this issue, we propose a Non-Salient Suppression (NSS) method to optimize the erroneous saliency maps generated in the first round and leverage them for the second round of training. Moreover, we build a new point-supervised dataset (P-DUTS) by relabeling the DUTS dataset. In P-DUTS, there is only one labeled point for each salient object. Comprehensive experiments on five largest benchmark datasets demonstrate our method outperforms the previous state-of-the-art methods trained with the stronger supervision and even surpass several fully supervised state-of-the-art models. The code is available at: https://github.com/shuyonggao/PSOD.

----

## [75] Latent Space Explanation by Intervention

**Authors**: *Itai Gat, Guy Lorberbom, Idan Schwartz, Tamir Hazan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19948](https://doi.org/10.1609/aaai.v36i1.19948)

**Abstract**:

The success of deep neural nets heavily relies on their ability to encode complex relations between their input and their output. While this property serves to fit the training data well, it also obscures the mechanism that drives prediction. This study aims to reveal hidden concepts by employing an intervention mechanism that shifts the predicted class based on discrete variational autoencoders. An explanatory model then visualizes the encoded information from any hidden layer and its corresponding intervened representation. By the assessment of differences between the original representation and the intervened representation, one can determine the concepts that can alter the class, hence providing interpretability. We demonstrate the effectiveness of our approach on CelebA, where we show various visualizations for bias in the data and suggest different interventions to reveal and change bias.

----

## [76] Lifelong Person Re-identification by Pseudo Task Knowledge Preservation

**Authors**: *Wenhang Ge, Junlong Du, Ancong Wu, Yuqiao Xian, Ke Yan, Feiyue Huang, Wei-Shi Zheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19949](https://doi.org/10.1609/aaai.v36i1.19949)

**Abstract**:

In real world, training data for person re-identification (Re-ID) is collected discretely with spatial and temporal variations, which requires a model to incrementally learn new knowledge without forgetting old knowledge. This problem is called lifelong person re-identification (LReID). Variations of illumination and background for images of each task exhibit task-specific image style and lead to task-wise domain gap. In addition to missing data from the old tasks, task-wise domain gap is a key factor for catastrophic forgetting in LReID, which is ignored in existing approaches for LReID. The model tends to learn task-specific knowledge with task-wise domain gap, which results in stability and plasticity dilemma. To overcome this problem, we cast LReID as a domain adaptation problem and propose a pseudo task knowledge preservation framework to alleviate the domain gap. Our framework is based on a pseudo task transformation module which maps the features of the new task into the feature space of the old tasks to complement the limited saved exemplars of the old tasks. With extra transformed features in the task-specific feature space, we propose a task-specific domain consistency loss to implicitly alleviate the task-wise domain gap for learning task-shared knowledge instead of task-specific one. Furthermore, to guide knowledge preservation with the feature distributions of the old tasks, we propose to preserve knowledge on extra pseudo tasks which jointly distills knowledge and discriminates identity, in order to achieve a better trade-off between stability and plasticity for lifelong learning with task-wise domain gap. Extensive experiments demonstrate the superiority of our method as compared with the state-of-the-art lifelong learning and LReID methods.

----

## [77] Adversarial Robustness in Multi-Task Learning: Promises and Illusions

**Authors**: *Salah Ghamizi, Maxime Cordy, Mike Papadakis, Yves Le Traon*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19950](https://doi.org/10.1609/aaai.v36i1.19950)

**Abstract**:

Vulnerability to adversarial attacks is a well-known weakness of Deep Neural networks. While most of the studies focus on single-task neural networks with computer vision datasets, very little research has considered complex multi-task models that are common in real applications. In this paper, we evaluate the design choices that impact the robustness of multi-task deep learning networks. We provide evidence that blindly adding auxiliary tasks, or weighing the tasks provides a false sense of robustness. Thereby, we tone down the claim made by previous research and study the different factors which may affect robustness. In particular, we show that the choice of the task to incorporate in the loss function are important factors that can be leveraged to yield more robust models. We provide the appendix, all our algorithms, models, and open source-code at https://github.com/yamizi/taskaugment

----

## [78] Deep Confidence Guided Distance for 3D Partial Shape Registration

**Authors**: *Dvir Ginzburg, Dan Raviv*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19951](https://doi.org/10.1609/aaai.v36i1.19951)

**Abstract**:

We present a novel non-iterative learnable method for partial-to-partial 3D shape registration.
The partial alignment task is extremely complex, as it jointly tries to match between points, and identify which points do not appear in the corresponding shape, causing the solution to be non-unique and ill-posed in most cases. 

Until now, two main methodologies have been suggested to solve this problem: sample a subset of points that are likely to have correspondences, or perform soft alignment between the point clouds and try to avoid a match to an occluded part. These heuristics work when the partiality is mild or when the transformation is small but fails for severe occlusions, or when outliers are present.
We present a unique approach named Confidence Guided Distance Network (CGD-net), where we fuse learnable similarity between point embeddings and spatial distance between point clouds, inducing an optimized solution for the overlapping points while ignoring parts that only appear in one of the shapes.
The point feature generation is done by a self-supervised architecture that repels far points to have different embeddings, therefore succeeds to align partial views of shapes, even with excessive internal symmetries, or acute rotations.
We compare our network to recently presented learning-based and axiomatic methods and report a fundamental boost in performance.

----

## [79] Predicting Physical World Destinations for Commands Given to Self-Driving Cars

**Authors**: *Dusan Grujicic, Thierry Deruyttere, Marie-Francine Moens, Matthew B. Blaschko*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19952](https://doi.org/10.1609/aaai.v36i1.19952)

**Abstract**:

In recent years, we have seen significant steps taken in the development of self-driving cars. Multiple companies are starting to roll out impressive systems that work in a variety of settings. These systems can sometimes give the impression that full self-driving is just around the corner and that we would soon build cars without even a steering wheel. The increase in the level of autonomy and control given to an AI provides an opportunity for new modes of human-vehicle interaction. However, surveys have shown that giving more control to an AI in self-driving cars is accompanied by a degree of uneasiness by passengers. In an attempt to alleviate this issue, recent works have taken a natural language-oriented approach by allowing the passenger to give commands that refer to specific objects in the visual scene. Nevertheless, this is only half the task as the car should also understand the physical destination of the command, which is what we focus on in this paper. We propose an extension in which we annotate the 3D destination that the car needs to reach after executing the given command and evaluate multiple different baselines on predicting this destination location. Additionally, we introduce a model that outperforms the prior works adapted for this particular setting.

----

## [80] Towards Light-Weight and Real-Time Line Segment Detection

**Authors**: *Geonmo Gu, ByungSoo Ko, SeoungHyun Go, Sung-Hyun Lee, Jingeun Lee, Minchul Shin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19953](https://doi.org/10.1609/aaai.v36i1.19953)

**Abstract**:

Previous deep learning-based line segment detection (LSD) suffers from the immense model size and high computational cost for line prediction. This constrains them from real-time inference on computationally restricted environments. In this paper, we propose a real-time and light-weight line segment detector for resource-constrained environments named Mobile LSD (M-LSD). We design an extremely efficient LSD architecture by minimizing the backbone network and removing the typical multi-module process for line prediction found in previous methods. To maintain competitive performance with a light-weight network, we present novel training schemes: Segments of Line segment (SoL) augmentation, matching and geometric loss. SoL augmentation splits a line segment into multiple subparts, which are used to provide auxiliary line data during the training process. Moreover, the matching and geometric loss allow a model to capture additional geometric cues. Compared with TP-LSD-Lite, previously the best real-time LSD method, our model (M-LSD-tiny) achieves competitive performance with 2.5% of model size and an increase of 130.5% in inference speed on GPU. Furthermore, our model runs at 56.8 FPS and 48.6 FPS on the latest Android and iPhone mobile devices, respectively. To the best of our knowledge, this is the first real-time deep LSD available on mobile devices.

----

## [81] Exploiting Fine-Grained Face Forgery Clues via Progressive Enhancement Learning

**Authors**: *Qiqi Gu, Shen Chen, Taiping Yao, Yang Chen, Shouhong Ding, Ran Yi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19954](https://doi.org/10.1609/aaai.v36i1.19954)

**Abstract**:

With the rapid development of facial forgery techniques, forgery detection has attracted more and more attention due to security concerns. Existing approaches attempt to use frequency information to mine subtle artifacts under high-quality forged faces. However, the exploitation of frequency information is coarse-grained, and more importantly, their vanilla learning process struggles to extract fine-grained forgery traces. To address this issue, we propose a progressive enhancement learning  framework to exploit both the RGB and fine-grained frequency clues. Specifically, we perform a fine-grained decomposition of RGB images to completely decouple the real and fake traces in the frequency space. Subsequently, we propose a progressive enhancement learning framework based on a two-branch network, combined with self-enhancement and mutual-enhancement modules. The self-enhancement module captures the traces in different input spaces based on spatial noise enhancement and channel attention. The Mutual-enhancement module concurrently enhances RGB and frequency features by communicating in the shared spatial dimension. The progressive enhancement process facilitates the learning of discriminative features with fine-grained face forgery clues. Extensive experiments on several datasets show that our method outperforms the state-of-the-art face forgery detection methods.

----

## [82] Delving into the Local: Dynamic Inconsistency Learning for DeepFake Video Detection

**Authors**: *Zhihao Gu, Yang Chen, Taiping Yao, Shouhong Ding, Jilin Li, Lizhuang Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19955](https://doi.org/10.1609/aaai.v36i1.19955)

**Abstract**:

The rapid development of facial manipulation techniques has aroused public concerns in recent years. Existing deepfake video detection approaches attempt to capture the discrim- inative features between real and fake faces based on tem- poral modelling. However, these works impose supervisions on sparsely sampled video frames but overlook the local mo- tions among adjacent frames, which instead encode rich in- consistency information that can serve as an efficient indica- tor for DeepFake video detection. To mitigate this issue, we delves into the local motion and propose a novel sampling unit named snippet which contains a few successive videos frames for local temporal inconsistency learning. Moreover, we elaborately design an Intra-Snippet Inconsistency Module (Intra-SIM) and an Inter-Snippet Interaction Module (Inter- SIM) to establish a dynamic inconsistency modelling frame- work. Specifically, the Intra-SIM applies bi-directional tem- poral difference operations and a learnable convolution ker- nel to mine the short-term motions within each snippet. The Inter-SIM is then devised to promote the cross-snippet infor- mation interaction to form global representations. The Intra- SIM and Inter-SIM work in an alternate manner and can be plugged into existing 2D CNNs. Our method outperforms the state of the art competitors on four popular benchmark dataset, i.e., FaceForensics++, Celeb-DF, DFDC and Wild- Deepfake. Besides, extensive experiments and visualizations are also presented to further illustrate its effectiveness.

----

## [83] Assessing a Single Image in Reference-Guided Image Synthesis

**Authors**: *Jiayi Guo, Chaoqun Du, Jiangshan Wang, Huijuan Huang, Pengfei Wan, Gao Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19956](https://doi.org/10.1609/aaai.v36i1.19956)

**Abstract**:

Assessing the performance of Generative Adversarial Networks (GANs) has been an important topic due to its practical significance. Although several evaluation metrics have been proposed, they generally assess the quality of the whole generated image distribution. For Reference-guided Image Synthesis (RIS) tasks, i.e., rendering a source image in the style of another reference image, where assessing the quality of a single generated image is crucial, these metrics are not applicable. In this paper, we propose a general learning-based framework, Reference-guided Image Synthesis Assessment (RISA) to quantitatively evaluate the quality of a single generated image. Notably, the training of RISA does not require human annotations. In specific, the training data for RISA are acquired by the intermediate models from the training procedure in RIS, and weakly annotated by the number of models' iterations, based on the positive correlation between image quality and iterations. As this annotation is too coarse as a supervision signal, we introduce two techniques: 1) a pixel-wise interpolation scheme to refine the coarse labels, and 2) multiple binary classifiers to replace a naïve regressor. In addition, an unsupervised contrastive loss is introduced to effectively capture the style similarity between a generated image and its reference image. Empirical results on various datasets demonstrate that RISA is highly consistent with human preference and transfers well across models.

----

## [84] Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-Supervised Action Recognition

**Authors**: *Tianyu Guo, Hong Liu, Zhan Chen, Mengyuan Liu, Tao Wang, Runwei Ding*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19957](https://doi.org/10.1609/aaai.v36i1.19957)

**Abstract**:

In recent years, self-supervised representation learning for skeleton-based action recognition has been developed with the advance of contrastive learning methods. The existing contrastive learning methods use normal augmentations to construct similar positive samples, which limits the ability to explore novel movement patterns. In this paper, to make better use of the movement patterns introduced by extreme augmentations, a Contrastive Learning framework utilizing Abundant Information Mining for self-supervised action Representation (AimCLR) is proposed. First, the extreme augmentations and the Energy-based Attention-guided Drop Module (EADM) are proposed to obtain diverse positive samples, which bring novel movement patterns to improve the universality of the learned representations. Second, since directly using extreme augmentations may not be able to boost the performance due to the drastic changes in original identity, the Dual Distributional Divergence Minimization Loss (D3M Loss) is proposed to minimize the distribution divergence in a more gentle way. Third, the Nearest Neighbors Mining (NNM) is proposed to further expand positive samples to make the abundant information mining process more reasonable. Exhaustive experiments on NTU RGB+D 60, PKU-MMD, NTU RGB+D 120 datasets have verified that our AimCLR can significantly perform favorably against state-of-the-art methods under a variety of evaluation protocols with observed higher quality action representations. Our code is available at https://github.com/Levigty/AimCLR.

----

## [85] Convolutional Neural Network Compression through Generalized Kronecker Product Decomposition

**Authors**: *Marawan Gamal Abdel Hameed, Marzieh S. Tahaei, Ali Mosleh, Vahid Partovi Nia*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19958](https://doi.org/10.1609/aaai.v36i1.19958)

**Abstract**:

Modern Convolutional Neural Network (CNN) architectures, despite their superiority in solving various problems, are generally too large to be deployed on resource constrained edge devices. In this paper, we reduce memory usage and floating-point operations required by convolutional layers in CNNs. We compress these layers by generalizing the Kronecker Product Decomposition to apply to multidimensional tensors, leading to the Generalized Kronecker Product Decomposition (GKPD). Our approach yields a plug-and-play module that can be used as a drop-in replacement for any convolutional layer. Experimental results for image classification on CIFAR-10 and ImageNet datasets using ResNet, MobileNetv2 and SeNet architectures substantiate the effectiveness of our proposed approach. We find that GKPD outperforms state-of-the-art decomposition methods including Tensor-Train and Tensor-Ring as well as other relevant compression methods such as pruning and knowledge distillation.

----

## [86] Meta Faster R-CNN: Towards Accurate Few-Shot Object Detection with Attentive Feature Alignment

**Authors**: *Guangxing Han, Shiyuan Huang, Jiawei Ma, Yicheng He, Shih-Fu Chang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19959](https://doi.org/10.1609/aaai.v36i1.19959)

**Abstract**:

Few-shot object detection (FSOD) aims to detect objects using only a few examples. How to adapt state-of-the-art object detectors to the few-shot domain remains challenging. Object proposal is a key ingredient in modern object detectors. However, the quality of proposals generated for few-shot classes using existing methods is far worse than that of many-shot classes, e.g., missing boxes for few-shot classes due to misclassification or inaccurate spatial locations with respect to true objects. To address the noisy proposal problem, we propose a novel meta-learning based FSOD model by jointly optimizing the few-shot proposal generation and fine-grained few-shot proposal classification. To improve proposal generation for few-shot classes, we propose to learn a lightweight metric-learning based prototype matching network, instead of the conventional simple linear object/nonobject classifier, e.g., used in RPN. Our non-linear classifier with the feature fusion network could improve the discriminative prototype matching and the proposal recall for few-shot classes. To improve the fine-grained few-shot proposal classification, we propose a novel attentive feature alignment method to address the spatial misalignment between the noisy proposals and few-shot classes, thus improving the performance of few-shot object detection. Meanwhile we learn a separate Faster R-CNN detection head for many-shot base classes and show strong performance of maintaining base-classes knowledge. Our model achieves state-of-the-art performance on multiple FSOD benchmarks over most of the shots and metrics.

----

## [87] Delving into Probabilistic Uncertainty for Unsupervised Domain Adaptive Person Re-identification

**Authors**: *Jian Han, Ya-Li Li, Shengjin Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19960](https://doi.org/10.1609/aaai.v36i1.19960)

**Abstract**:

Clustering-based unsupervised domain adaptive (UDA) person re-identification (ReID) reduces exhaustive annotations. However, owing to unsatisfactory feature embedding and imperfect clustering, pseudo labels for target domain data inherently contain an unknown proportion of wrong ones, which would mislead feature learning. In this paper, we propose an approach named probabilistic uncertainty guided progressive label refinery (P2LR) for domain adaptive person re-identification. First, we propose to model the labeling uncertainty with the probabilistic distance along with ideal single-peak distributions. A quantitative criterion is established to measure the uncertainty of pseudo labels and facilitate the network training. Second, we explore a progressive strategy for refining pseudo labels. With the uncertainty-guided alternative optimization, we balance between the exploration of target domain data and the negative effects of noisy labeling. On top of a strong baseline, we obtain significant improvements and achieve the state-of-the-art performance on four UDA ReID benchmarks. Specifically, our method outperforms the baseline by 6.5% mAP on the Duke2Market task, while surpassing the state-of-the-art method by 2.5% mAP on the Market2MSMT task. Code is available at: https://github.com/JeyesHan/P2LR.

----

## [88] Laneformer: Object-Aware Row-Column Transformers for Lane Detection

**Authors**: *Jianhua Han, Xiajun Deng, Xinyue Cai, Zhen Yang, Hang Xu, Chunjing Xu, Xiaodan Liang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19961](https://doi.org/10.1609/aaai.v36i1.19961)

**Abstract**:

We present Laneformer, a conceptually simple yet powerful transformer-based architecture tailored for lane detection that is a long-standing research topic for visual perception in autonomous driving. The dominant paradigms rely on purely CNN-based architectures which often fail in incorporating relations of long-range lane points and global contexts induced by surrounding objects (e.g., pedestrians, vehicles). Inspired by recent advances of the transformer encoder-decoder architecture in various vision tasks, we move forwards to design a new end-to-end Laneformer architecture that revolutionizes the conventional transformers into better capturing the shape and semantic characteristics of lanes, with minimal overhead in latency. First, coupling with deformable pixel-wise self-attention in the encoder, Laneformer presents two new row and column self-attention operations to efficiently mine point context along with the lane shapes. Second, motivated by the appearing objects would affect the decision of predicting lane segments, Laneformer further includes the detected object instances as extra inputs of multi-head attention blocks in the encoder and decoder to facilitate the lane point detection by sensing semantic contexts. Specifically, the bounding box locations of objects are added into Key module to provide interaction with each pixel and query while the ROI-aligned features are inserted into Value module. Extensive experiments demonstrate our Laneformer achieves state-of-the-art performances on CULane benchmark, in terms of 77.1% F1 score. We hope our simple and effective Laneformer will serve as a strong baseline for future research in self-attention models for lane detection.

----

## [89] Modify Self-Attention via Skeleton Decomposition for Effective Point Cloud Transformer

**Authors**: *Jiayi Han, Longbin Zeng, Liang Du, Xiaoqing Ye, Weiyang Ding, Jianfeng Feng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19962](https://doi.org/10.1609/aaai.v36i1.19962)

**Abstract**:

Although considerable progress has been achieved regarding the transformers in recent years, the large number of parameters, quadratic computational complexity, and memory cost conditioned on long sequences make the transformers hard to train and implement, especially in edge computing configurations. In this case, a dizzying number of works have sought to make improvements around computational and memory efficiency upon the original transformer architecture. Nevertheless, many of them restrict the context in the attention to seek a trade-off between cost and performance with prior knowledge of orderly stored data. It is imperative to dig deep into an efficient feature extractor for point clouds due to their irregularity and a large number of points. In this paper, we propose a novel skeleton decomposition-based self-attention (SD-SA) which has no sequence length limit and exhibits favorable scalability in long-sequence models. Due to the numerical low-rank nature of self-attention, we approximate it by the skeleton decomposition method while maintaining its effectiveness. At this point, we have shown that the proposed method works for the proposed approach on point cloud classification, segmentation, and detection tasks on the ModelNet40, ShapeNet, and KITTI datasets, respectively. Our approach significantly improves the efficiency of the point cloud transformer and exceeds other efficient transformers on point cloud tasks in terms of the speed at comparable performance.

----

## [90] Generalizable Person Re-identification via Self-Supervised Batch Norm Test-Time Adaption

**Authors**: *Ke Han, Chenyang Si, Yan Huang, Liang Wang, Tieniu Tan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19963](https://doi.org/10.1609/aaai.v36i1.19963)

**Abstract**:

In this paper, we investigate the generalization problem of person re-identification (re-id), whose major challenge is the distribution shift on an unseen domain. As an important tool of regularizing the distribution, batch normalization (BN) has been widely used in existing methods. However, they neglect that BN is severely biased to the training domain and inevitably suffers the performance drop if directly generalized without being updated. To tackle this issue, we propose Batch Norm Test-time Adaption (BNTA), a novel re-id framework that applies the self-supervised strategy to update BN parameters adaptively. Specifically, BNTA quickly explores the domain-aware information within unlabeled target data before inference, and accordingly modulates the feature distribution normalized by BN to adapt to the target domain. This is accomplished by two designed self-supervised auxiliary tasks, namely part positioning and part nearest neighbor matching, which help the model mine the domain-aware information with respect to the structure and identity of body parts, respectively. To demonstrate the effectiveness of our method, we conduct extensive experiments on three re-id datasets and confirm the superior performance to the state-of-the-art methods.

----

## [91] RRL: Regional Rotate Layer in Convolutional Neural Networks

**Authors**: *Zongbo Hao, Tao Zhang, Mingwang Chen, Kaixu Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19964](https://doi.org/10.1609/aaai.v36i1.19964)

**Abstract**:

Convolutional Neural Networks (CNNs) perform very well in image classification and object detection in recent years, but even the most advanced models have limited rotation invariance. Known solutions include the enhancement of training data and the increase of rotation invariance by globally merging the rotation equivariant features. These methods either increase the workload of training or increase the number of model parameters. To address this problem, this paper proposes a module that can be inserted into the existing networks, and directly incorporates the rotation invariance into the feature extraction layers of the CNNs. This module does not have learnable parameters and will not increase the complexity of the model. At the same time, only by training the upright data, it can perform well on the rotated testing set. These ad-vantages will be suitable for fields such as biomedicine and astronomy where it is difficult to obtain upright samples or the target has no directionality. Evaluate our module with LeNet-5, ResNet-18 and tiny-yolov3, we get impressive results.

----

## [92] QueryProp: Object Query Propagation for High-Performance Video Object Detection

**Authors**: *Fei He, Naiyu Gao, Jian Jia, Xin Zhao, Kaiqi Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19965](https://doi.org/10.1609/aaai.v36i1.19965)

**Abstract**:

Video object detection has been an important yet challenging topic in computer vision. Traditional methods mainly focus on designing the image-level or box-level feature propagation strategies to exploit temporal information. This paper argues that with a more effective and efficient feature propagation framework, video object detectors can gain improvement in terms of both accuracy and speed. For this purpose, this paper studies object-level feature propagation, and proposes an object query propagation (QueryProp) framework for high-performance video object detection. The proposed QueryProp contains two propagation strategies: 1) query propagation is performed from sparse key frames to dense non-key frames to reduce the redundant computation on non-key frames; 2) query propagation is performed from previous key frames to the current key frame to improve feature representation by temporal context modeling. To further facilitate query propagation, an adaptive propagation gate is designed to achieve flexible key frame selection. We conduct extensive experiments on the ImageNet VID dataset. QueryProp achieves comparable accuracy with state-of-the-art methods and strikes a decent accuracy/speed trade-off.

----

## [93] Flow-Based Unconstrained Lip to Speech Generation

**Authors**: *Jinzheng He, Zhou Zhao, Yi Ren, Jinglin Liu, Baoxing Huai, Nicholas Jing Yuan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19966](https://doi.org/10.1609/aaai.v36i1.19966)

**Abstract**:

Unconstrained lip-to-speech aims to generate corresponding speeches based on silent facial videos with no restriction to head pose or vocabulary. It is desirable to generate intelligible and natural speech with a fast speed in unconstrained settings.
  Currently, to handle the more complicated scenarios, most existing methods adopt the autoregressive architecture, which is optimized with the MSE loss. Although these methods have achieved promising performance, they are prone to bring issues including high inference latency and mel-spectrogram over-smoothness. 
  To tackle these problems, we propose a novel 
  flow-based non-autoregressive lip-to-speech model (GlowLTS) to break autoregressive constraints and achieve faster inference. Concretely, we adopt a flow-based decoder which is optimized by maximizing the likelihood of the training data and is capable of more natural and fast speech generation. Moreover, we devise a condition module to improve the intelligibility of generated speech. 
  We demonstrate the superiority of our proposed method through objective and subjective evaluation on Lip2Wav-Chemistry-Lectures and Lip2Wav-Chess-Analysis datasets. Our demo video can be found at https://glowlts.github.io/.

----

## [94] TransFG: A Transformer Architecture for Fine-Grained Recognition

**Authors**: *Ju He, Jieneng Chen, Shuai Liu, Adam Kortylewski, Cheng Yang, Yutong Bai, Changhu Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19967](https://doi.org/10.1609/aaai.v36i1.19967)

**Abstract**:

Fine-grained visual classification (FGVC) which aims at recognizing objects from subcategories is a very challenging task due to the inherently subtle inter-class differences. Most existing works mainly tackle this problem by reusing the backbone network to extract features of detected discriminative regions. However, this strategy inevitably complicates the pipeline and pushes the proposed regions to contain most parts of the objects thus fails to locate the really important parts. Recently, vision transformer (ViT) shows its strong performance in the traditional classification task. The self-attention mechanism of the transformer links every patch token to the classification token. In this work, we first evaluate the effectiveness of the ViT framework in the fine-grained recognition setting. Then motivated by the strength of the attention link can be intuitively considered as an indicator of the importance of tokens, we further propose a novel Part Selection Module that can be applied to most of the transformer architectures where we integrate all raw attention weights of the transformer into an attention map for guiding the network to effectively and accurately select discriminative image patches and compute their relations. A contrastive loss is applied to enlarge the distance between feature representations of confusing classes. We name the augmented transformer-based model TransFG and demonstrate the value of it by conducting experiments on five popular fine-grained benchmarks where we achieve state-of-the-art performance. Qualitative results are presented for better understanding of our model.

----

## [95] Self-Supervised Robust Scene Flow Estimation via the Alignment of Probability Density Functions

**Authors**: *Pan He, Patrick Emami, Sanjay Ranka, Anand Rangarajan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19968](https://doi.org/10.1609/aaai.v36i1.19968)

**Abstract**:

In this paper, we present a new self-supervised scene flow estimation approach for a pair of consecutive point clouds. The key idea of our approach is to represent discrete point clouds as continuous probability density functions using Gaussian mixture models. Scene flow estimation is therefore converted into the problem of recovering motion from the alignment of probability density functions, which we achieve using a closed-form expression of the classic Cauchy-Schwarz divergence. Unlike existing nearest-neighbor-based approaches that use hard pairwise correspondences, our proposed approach establishes soft and implicit point correspondences between point clouds and generates more robust and accurate scene flow in the presence of missing correspondences and outliers. Comprehensive experiments show that our method makes noticeable gains over the Chamfer Distance and the Earth Mover’s Distance in real-world environments and achieves state-of-the-art performance among self-supervised learning methods on FlyingThings3D and KITTI, even outperforming some supervised methods with ground truth annotations.

----

## [96] SVGA-Net: Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds

**Authors**: *Qingdong He, Zhengning Wang, Hao Zeng, Yi Zeng, Yijun Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19969](https://doi.org/10.1609/aaai.v36i1.19969)

**Abstract**:

Accurate 3D object detection from point clouds has become a crucial component in autonomous driving. However, the volumetric representations and the projection methods in previous works fail to establish the relationships between the local point sets. In this paper, we propose Sparse Voxel-Graph Attention Network (SVGA-Net), a novel end-to-end trainable network which mainly contains voxel-graph module and sparse-to-dense regression module to achieve comparable 3D detection tasks from raw LIDAR data. Specifically, SVGA-Net constructs the local complete graph within each divided 3D spherical voxel and global KNN graph through all voxels. The local and global graphs serve as the attention mechanism to enhance the extracted features. In addition, the novel sparse-to-dense regression module enhances the 3D box estimation accuracy through feature maps aggregation at different levels. Experiments on KITTI detection benchmark and Waymo Open dataset demonstrate the efficiency of extending the graph representation to 3D object detection and the proposed SVGA-Net can achieve decent detection accuracy.

----

## [97] SECRET: Self-Consistent Pseudo Label Refinement for Unsupervised Domain Adaptive Person Re-identification

**Authors**: *Tao He, Leqi Shen, Yuchen Guo, Guiguang Ding, Zhenhua Guo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19970](https://doi.org/10.1609/aaai.v36i1.19970)

**Abstract**:

Unsupervised domain adaptive person re-identification aims at learning on an unlabeled target domain with only labeled data in source domain. Currently, the state-of-the-arts usually solve this problem by pseudo-label-based clustering and fine-tuning in target domain. However, the reason behind the noises of pseudo labels is not sufficiently explored, especially for the popular multi-branch models. We argue that the consistency between different feature spaces is the key to the pseudo labels’ quality. Then a SElf-Consistent pseudo label RefinEmenT method, termed as SECRET, is proposed to improve consistency by mutually refining the pseudo labels generated from different feature spaces. The proposed SECRET gradually encourages the improvement of pseudo labels’ quality during training process, which further leads to better cross-domain Re-ID performance. Extensive experiments on benchmark datasets show the superiority of our method. Specifically, our method outperforms the state-of-the-arts by 6.3% in terms of mAP on the challenging dataset MSMT17. In the purely unsupervised setting, our method also surpasses existing works by a large margin. Code is available at https://github.com/LunarShen/SECRET.

----

## [98] Visual Semantics Allow for Textual Reasoning Better in Scene Text Recognition

**Authors**: *Yue He, Chen Chen, Jing Zhang, Juhua Liu, Fengxiang He, Chaoyue Wang, Bo Du*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19971](https://doi.org/10.1609/aaai.v36i1.19971)

**Abstract**:

Existing Scene Text Recognition (STR) methods typically use a language model to optimize the joint probability of the 1D character sequence predicted by a visual recognition (VR) model, which ignore the 2D spatial context of visual semantics within and between character instances, making them not generalize well to arbitrary shape scene text. To address this issue, we make the first attempt to perform textual reasoning based on visual semantics in this paper. Technically, given the character segmentation maps predicted by a VR model, we construct a subgraph for each instance, where nodes represent the pixels in it and edges are added between nodes based on their spatial similarity. Then, these subgraphs are sequentially connected by their root nodes and merged into a complete graph. Based on this graph, we devise a graph convolutional network for textual reasoning (GTR) by supervising it with a cross-entropy loss. GTR can be easily plugged in representative STR models to improve their performance owing to better textual reasoning. Specifically, we construct our model, namely S-GTR, by paralleling GTR to the language model in a segmentation-based STR baseline, which can effectively exploit the visual-linguistic complementarity via mutual learning. S-GTR sets new state-of-the-art on six challenging STR benchmarks and generalizes well to multi-linguistic datasets. Code is available at https://github.com/adeline-cs/GTR.

----

## [99] Ranking Info Noise Contrastive Estimation: Boosting Contrastive Learning via Ranked Positives

**Authors**: *David T. Hoffmann, Nadine Behrmann, Juergen Gall, Thomas Brox, Mehdi Noroozi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19972](https://doi.org/10.1609/aaai.v36i1.19972)

**Abstract**:

This paper introduces Ranking Info Noise Contrastive Estimation (RINCE), a new member in the family of InfoNCE losses that preserves a ranked ordering of positive samples. In contrast to the standard InfoNCE loss, which requires a strict binary separation of the training pairs into similar and dissimilar samples, RINCE can exploit information about a similarity ranking for learning a corresponding embedding space. We show that the proposed loss function learns favorable embeddings compared to the standard InfoNCE whenever at least noisy ranking information can be obtained or when the definition of positives and negatives is blurry. We demonstrate this for a supervised classification task with additional superclass labels and noisy similarity scores. Furthermore, we show that RINCE can also be applied to unsupervised training with experiments on unsupervised representation learning from videos. In particular, the embedding yields higher classification accuracy, retrieval rates and performs better on out-of-distribution detection than the standard InfoNCE loss.

----

## [100] Uncertainty-Driven Dehazing Network

**Authors**: *Ming Hong, Jianzhuang Liu, Cuihua Li, Yanyun Qu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19973](https://doi.org/10.1609/aaai.v36i1.19973)

**Abstract**:

Deep learning has made remarkable achievements for single image haze removal. However, existing deep dehazing models only give deterministic results without discussing the uncertainty of them. 
 There exist two types of uncertainty in the dehazing models: aleatoric uncertainty that comes from noise inherent in the observations and epistemic uncertainty that accounts for uncertainty in the model.
 In this paper, we propose a novel uncertainty-driven dehazing network (UDN) that improves the dehazing results by exploiting the relationship between the uncertain and confident representations. 
 We first introduce an Uncertainty Estimation Block (UEB) to predict the aleatoric and epistemic uncertainty together. Then, we propose an Uncertainty-aware Feature Modulation (UFM) block to adaptively enhance the learned features. UFM predicts a convolution kernel and channel-wise modulation cofficients conitioned on the uncertainty weighted representation.
 Moreover, we develop an uncertainty-driven self-distillation loss to improve the uncertain representation by transferring the knowledge from the confident one.
 Extensive experimental results on synthetic datasets and real-world images show that UDN achieves significant quantitative and qualitative improvements, outperforming the state-of-the-arts.

----

## [101] Shadow Generation for Composite Image in Real-World Scenes

**Authors**: *Yan Hong, Li Niu, Jianfu Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19974](https://doi.org/10.1609/aaai.v36i1.19974)

**Abstract**:

Image composition targets at inserting a foreground object into a background image. Most previous image composition methods focus on adjusting the foreground to make it compatible with background while ignoring the shadow effect of foreground on the background. In this work, we focus on generating plausible shadow for the foreground object in the composite image. First, we contribute a real-world shadow generation dataset DESOBA by generating synthetic composite images based on paired real images and deshadowed images.  Then, we propose a novel shadow generation network SGRNet, which consists of a shadow mask prediction stage and a shadow filling stage. In the shadow mask prediction stage, foreground and background information are thoroughly interacted to generate foreground shadow mask. In the shadow filling stage, shadow parameters are predicted to fill the shadow area. Extensive experiments on our DESOBA dataset and real composite images demonstrate the effectiveness of our proposed method. Our dataset and code are available at https://github.com/bcmi/Object-Shadow-Generation- Dataset-DESOBA.

----

## [102] Shape-Adaptive Selection and Measurement for Oriented Object Detection

**Authors**: *Liping Hou, Ke Lu, Jian Xue, Yuqiu Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19975](https://doi.org/10.1609/aaai.v36i1.19975)

**Abstract**:

The development of detection methods for oriented object detection remains a challenging task. A considerable obstacle is the wide variation in the shape (e.g., aspect ratio) of objects. Sample selection in general object detection has been widely studied as it plays a crucial role in the performance of the detection method and has achieved great progress. However, existing sample selection strategies still overlook some issues: (1) most of them ignore the object shape information; (2) they do not make a potential distinction between selected positive samples; and (3) some of them can only be applied to either anchor-free or anchor-based methods and cannot be used for both of them simultaneously. In this paper, we propose novel flexible shape-adaptive selection (SA-S) and shape-adaptive measurement (SA-M) strategies for oriented object detection, which comprise an SA-S strategy for sample selection and SA-M strategy for the quality estimation of positive samples. Specifically, the SA-S strategy dynamically selects samples according to the shape information and characteristics distribution of objects. The SA-M strategy measures the localization potential and adds quality information on the selected positive samples. The experimental results on both anchor-free and anchor-based baselines and four publicly available oriented datasets (DOTA, HRSC2016, UCAS-AOD, and ICDAR2015) demonstrate the effectiveness of the proposed method.

----

## [103] H^2-MIL: Exploring Hierarchical Representation with Heterogeneous Multiple Instance Learning for Whole Slide Image Analysis

**Authors**: *Wentai Hou, Lequan Yu, Chengxuan Lin, Helong Huang, Rongshan Yu, Jing Qin, Liansheng Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19976](https://doi.org/10.1609/aaai.v36i1.19976)

**Abstract**:

Current representation learning methods for whole slide image (WSI) with pyramidal resolutions are inherently homogeneous and flat, which cannot fully exploit the multiscale and heterogeneous diagnostic information of different structures for comprehensive analysis. This paper presents a novel graph neural network-based multiple instance learning framework (i.e., H^2-MIL) to learn hierarchical representation from a heterogeneous graph with different resolutions for WSI analysis. A heterogeneous graph with the “resolution” attribute is constructed to explicitly model the feature and spatial-scaling relationship of multi-resolution patches. We then design a novel resolution-aware attention convolution (RAConv) block to learn compact yet discriminative representation from the graph, which tackles the heterogeneity of node neighbors with different resolutions and yields more reliable message passing. More importantly, to explore the task-related structured information of WSI pyramid, we elaborately design a novel iterative hierarchical pooling (IHPool) module to progressively aggregate the heterogeneous graph based on scaling relationships of different nodes. We evaluated our method on two public WSI datasets from the TCGA project, i.e., esophageal cancer and kidney cancer. Experimental results show that our method clearly outperforms the state-of-the-art methods on both tumor typing and staging tasks.

----

## [104] Elastic-Link for Binarized Neural Networks

**Authors**: *Jie Hu, Ziheng Wu, Vince Junkai Tan, Zhilin Lu, Mengze Zeng, Enhua Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19977](https://doi.org/10.1609/aaai.v36i1.19977)

**Abstract**:

Recent work has shown that Binarized Neural Networks (BNNs) are able to greatly reduce computational costs and memory footprints, facilitating model deployment on resource-constrained devices. However, in comparison to their full-precision counterparts, BNNs suffer from severe accuracy degradation. Research aiming to reduce this accuracy gap has thus far largely focused on specific network architectures with few or no 1 × 1 convolutional layers, for which standard binarization methods do not work well. Because 1 × 1 convolutions are common in the design of modern architectures (e.g. GoogleNet, ResNet, DenseNet), it is crucial to develop a method to binarize them effectively for BNNs to be more widely adopted. In this work, we propose an “Elastic-Link” (EL) module to enrich information flow within a BNN by adaptively adding real-valued input features to the subsequent convolutional output features. The proposed EL module is easily implemented and can be used in conjunction with other methods for BNNs. We demonstrate that adding EL to BNNs produces a significant improvement on the challenging large-scale ImageNet dataset. For example, we raise the top-1 accuracy of binarized ResNet26 from 57.9% to 64.0%. EL also aids con-vergence in the training of binarized MobileNet, for which a top-1 accuracy of 56.4% is achieved. Finally, with the integration of ReActNet, it yields a new state-of-the-art result of 71.9% top-1 accuracy.

----

## [105] FInfer: Frame Inference-Based Deepfake Detection for High-Visual-Quality Videos

**Authors**: *Juan Hu, Xin Liao, Jinwen Liang, Wenbo Zhou, Zheng Qin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19978](https://doi.org/10.1609/aaai.v36i1.19978)

**Abstract**:

Deepfake has ignited hot research interests in both academia and industry due to its potential security threats. Many countermeasures have been proposed to mitigate such risks. Current Deepfake detection methods achieve superior performances in dealing with low-visual-quality Deepfake media which can be distinguished by the obvious visual artifacts. However, with the development of deep generative models, the realism of Deepfake media has been significantly improved and becomes tough challenging to current detection models. In this paper, we propose a frame inference-based detection framework (FInfer) to solve the problem of high-visual-quality Deepfake detection. Specifically, we first learn the referenced representations of the current and future frames’ faces. Then, the current frames’ facial representations are utilized to predict the future frames’ facial representations by using an autoregressive model. Finally, a representation-prediction loss is devised to maximize the discriminability of real videos and fake videos. We demonstrate the effectiveness of our FInfer framework through information theory analyses. The entropy and mutual information analyses indicate the correlation between the predicted representations and referenced representations in real videos is higher than that of high-visual-quality Deepfake videos. Extensive experiments demonstrate the performance of our method is promising in terms of in-dataset detection performance, detection efficiency, and cross-dataset detection performance in high-visual-quality Deepfake videos.

----

## [106] Bi-volution: A Static and Dynamic Coupled Filter

**Authors**: *Xiwei Hu, Xuanhong Chen, Bingbing Ni, Teng Li, Yutian Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19979](https://doi.org/10.1609/aaai.v36i1.19979)

**Abstract**:

Dynamic convolution has achieved significant gain in performance and computational complexity, thanks to its powerful representation capability given limited filter number/layers.
 However, SOTA dynamic convolution operators are sensitive to input noises (e.g., Gaussian noise, shot noise, e.t.c.) and lack sufficient spatial contextual information in filter generation.
 To alleviate this inherent weakness, we propose a lightweight and heterogeneous-structure (i.e., static and dynamic) operator, named Bi-volution.
 On the one hand, Bi-volution is designed as a dual-branch structure to fully leverage complementary properties of static/dynamic convolution, which endows Bi-volution more robust properties and higher performance.
 On the other hand, the Spatial Augmented Kernel Generation module is proposed to improve the dynamic convolution, realizing the learning of spatial context information with negligible additional computational complexity.
 Extensive experiments illustrate that the ResNet-50 equipped with Bi-volution achieves a highly competitive boost in performance (+2.8% top-1 accuracy on ImageNet classification, +2.4% box AP and +2.2% mask AP on COCO detection and instance segmentation) while maintaining extremely low FLOPs (i.e., ResNet50@2.7 GFLOPs). Furthermore, our Bi-volution shows better robustness than dynamic convolution against various noise and input corruptions. Our code is available at https://github.com/neuralchen/Bivolution.

----

## [107] AFDetV2: Rethinking the Necessity of the Second Stage for Object Detection from Point Clouds

**Authors**: *Yihan Hu, Zhuangzhuang Ding, Runzhou Ge, Wenxin Shao, Li Huang, Kun Li, Qiang Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19980](https://doi.org/10.1609/aaai.v36i1.19980)

**Abstract**:

There have been two streams in the 3D detection from point clouds: single-stage methods and two-stage methods. While the former is more computationally efficient, the latter usually provides better detection accuracy. By carefully examining the two-stage approaches, we have found that if appropriately designed, the first stage can produce accurate box regression. In this scenario, the second stage mainly rescores the boxes such that the boxes with better localization get selected. From this observation, we have devised a single-stage anchor-free network that can fulfill these requirements. This network, named AFDetV2, extends the previous work by incorporating a self-calibrated convolution block in the backbone, a keypoint auxiliary supervision, and an IoU prediction branch in the multi-task head. We take a simple product of the predicted IoU score with the classification heatmap to form the final classification confidence. The enhanced backbone strengthens the box localization capability, and the rescoring approach effectively joins the object presence confidence and the box regression accuracy. As a result, the detection accuracy is drastically boosted in the single-stage. To evaluate our approach, we have conducted extensive experiments on the Waymo Open Dataset and the nuScenes Dataset. We have observed that our AFDetV2 achieves the state-of-the-art results on these two datasets, superior to all the prior arts, including both the single-stage and the two-stage 3D detectors. AFDetV2 won the 1st place in the Real-Time 3D Detection of the Waymo Open Dataset Challenge 2021. In addition, a variant of our model AFDetV2-Base was entitled the "Most Efficient Model" by the Challenge Sponsor, showing a superior computational efficiency. To demonstrate the generality of this single-stage method, we have also applied it to the first stage of the two-stage networks. Without exception, the results show that with the strengthened backbone and the rescoring approach, the second stage refinement is no longer needed.

----

## [108] Divide-and-Regroup Clustering for Domain Adaptive Person Re-identification

**Authors**: *Zhengdong Hu, Yifan Sun, Yi Yang, Jianguang Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19981](https://doi.org/10.1609/aaai.v36i1.19981)

**Abstract**:

Clustering is important for domain adaptive person re-identification(re-ID). A majority of unsupervised domain adaptation (UDA) methods conduct clustering on the target domain and then use the generated pseudo labels for adaptive training. Albeit important, the clustering pipeline adopted by current literature is quite standard and lacks consideration for two characteristics of re-ID, i.e., 1) a single person has various feature distribution in multiple cameras. 2) a person’s occurrence in the same camera are usually temporally continuous. We argue that the multi-camera distribution hinders clustering because it enlarges the intra-class distances. In contrast, the temporal continuity prior is beneficial, because it offers clue for distinguishing some look-alike person (who are temporally far away from each other). These two insight motivate us to propose a novel Divide-And-Regroup Clustering (DARC) pipeline for re-ID UDA. Specifically, DARC divides the unlabeled data into multiple camera-specific groups and conducts local clustering within each camera. Afterwards, it regroups those local clusters potentially belonging to the same person into a unity. Through this divide-and-regroup pipeline, DARC avoids directly clustering across multiple cameras and focuses on the feature distribution within each individual camera. Moreover, during the local clustering, DARC uses the temporal continuity prior to distinguish some look-alike person and thus reduces false positive pseudo labels. Consequentially, DARC effectively reduces clustering errors and improves UDA. Importantly, we show that DARC is compatible to many pseudo label-based UDA methods and brings general improvement. Based on a recent UDA method, DARC advances the state of the art (e.g, 85.1% mAP on MSMT-to-Market and 83.1% mAP on PersonX-to-Market).

----

## [109] CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes

**Authors**: *Hao Huang, Yongtao Wang, Zhaoyu Chen, Yuze Zhang, Yuheng Li, Zhi Tang, Wei Chu, Jingdong Chen, Weisi Lin, Kai-Kuang Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19982](https://doi.org/10.1609/aaai.v36i1.19982)

**Abstract**:

Malicious applications of deepfakes (i.e., technologies generating target facial attributes or entire faces from facial images) have posed a huge threat to individuals' reputation and security. To mitigate these threats, recent studies have proposed adversarial watermarks to combat deepfake models, leading them to generate distorted outputs. Despite achieving impressive results, these adversarial watermarks have low image-level and model-level transferability, meaning that they can protect only one facial image from one specific deepfake model. To address these issues, we propose a novel solution that can generate a Cross-Model Universal Adversarial Watermark (CMUA-Watermark), protecting a large number of facial images from multiple deepfake models. Specifically, we begin by proposing a cross-model universal attack pipeline that attacks multiple deepfake models iteratively. Then, we design a two-level perturbation fusion strategy to alleviate the conflict between the adversarial watermarks generated by different facial images and models. Moreover, we address the key problem in cross-model optimization with a heuristic approach to automatically find the suitable attack step sizes for different models, further weakening the model-level conflict. Finally, we introduce a more reasonable and comprehensive evaluation method to fully test the proposed method and compare it with existing ones. Extensive experimental results demonstrate that the proposed CMUA-Watermark can effectively distort the fake facial images generated by multiple deepfake models while achieving a better performance than existing methods. Our code is available at https://github.com/VDIGPKU/CMUA-Watermark.

----

## [110] Deconfounded Visual Grounding

**Authors**: *Jianqiang Huang, Yu Qin, Jiaxin Qi, Qianru Sun, Hanwang Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19983](https://doi.org/10.1609/aaai.v36i1.19983)

**Abstract**:

We focus on the confounding bias between language and location in the visual grounding pipeline, where we find that the bias is the major visual reasoning bottleneck. For example, the grounding process is usually a trivial languagelocation association without visual reasoning, e.g., grounding any language query containing sheep to the nearly central regions, due to that most queries about sheep have ground-truth locations at the image center. First, we frame the visual grounding pipeline into a causal graph, which shows the causalities among image, query, target location and underlying confounder. Through the causal graph, we know how to break the grounding bottleneck: deconfounded visual grounding. Second, to tackle the challenge that the confounder is unobserved in general, we propose a confounder-agnostic approach called: Referring Expression Deconfounder (RED), to remove the confounding bias. Third, we implement RED as a simple language attention, which can be applied in any grounding method. On popular benchmarks, RED improves various state-of-the-art grounding methods by a significant margin. Code is available at: https://github.com/JianqiangH/Deconfounded_VG.

----

## [111] Learning to Model Pixel-Embedded Affinity for Homogeneous Instance Segmentation

**Authors**: *Wei Huang, Shiyu Deng, Chang Chen, Xueyang Fu, Zhiwei Xiong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19984](https://doi.org/10.1609/aaai.v36i1.19984)

**Abstract**:

Homogeneous instance segmentation aims to identify each instance in an image where all interested instances belong to the same category, such as plant leaves and microscopic cells. Recently, proposal-free methods, which straightforwardly generate instance-aware information to group pixels into different instances, have received increasing attention due to their efficient pipeline. However, they often fail to distinguish adjacent instances due to similar appearances, dense distribution and ambiguous boundaries of instances in homogeneous images. In this paper, we propose a pixel-embedded affinity modeling method for homogeneous instance segmentation, which is able to preserve the semantic information of instances and improve the distinguishability of adjacent instances. Instead of predicting affinity directly, we propose a self-correlation module to explicitly model the pairwise relationships between pixels, by estimating the similarity between embeddings generated from the input image through CNNs. Based on the self-correlation module, we further design a cross-correlation module to maintain the semantic consistency between instances. Specifically, we map the transformed input images with different views and appearances into the same embedding space, and then mutually estimate the pairwise relationships of embeddings generated from the original input and its transformed variants. In addition, to integrate the global instance information, we introduce an embedding pyramid module to model affinity on different scales. Extensive experiments demonstrate the versatile and superior performance of our method on three representative datasets. Code and models are available at https://github.com/weih527/Pixel-Embedded-Affinity.

----

## [112] Channelized Axial Attention - considering Channel Relation within Spatial Attention for Semantic Segmentation

**Authors**: *Ye Huang, Di Kang, Wenjing Jia, Liu Liu, Xiangjian He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19985](https://doi.org/10.1609/aaai.v36i1.19985)

**Abstract**:

Spatial and channel attentions, modelling the semantic interdependencies in spatial and channel dimensions respectively, have recently been widely used for semantic segmentation. However, computing spatial and channel attentions separately sometimes causes errors, especially for those difficult cases. In this paper, we propose Channelized Axial Attention (CAA) to seamlessly integrate channel attention and spatial attention into a single operation with negligible computation overhead. Specifically, we break down the dot-product operation of the spatial attention into two parts and insert channel relation in between, allowing for independently optimized channel attention on each spatial location. We further develop grouped vectorization, which allows our model to run with very little memory consumption without slowing down the running speed. Comparative experiments conducted on multiple benchmark datasets, including Cityscapes, PASCAL Context, and COCO-Stuff, demonstrate that our CAA outperforms many state-of-the-art segmentation models (including dual attention) on all tested datasets.

----

## [113] UFPMP-Det: Toward Accurate and Efficient Object Detection on Drone Imagery

**Authors**: *Yecheng Huang, Jiaxin Chen, Di Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19986](https://doi.org/10.1609/aaai.v36i1.19986)

**Abstract**:

This paper proposes a novel approach to object detection on drone imagery, namely Multi-Proxy Detection Network with Unified Foreground Packing (UFPMP-Det). To deal with the numerous instances of very small scales, different from the common solution that divides the high-resolution input image into quite a number of chips with low foreground ratios to perform detection on them each, the Unified Foreground Packing (UFP) module is designed, where the sub-regions given by a coarse detector are initially merged through clustering to suppress background and the resulting ones are subsequently packed into a mosaic for a single inference, thus significantly reducing overall time cost. Furthermore, to address the more serious confusion between inter-class similarities and intra-class variations of instances, which deteriorates detection performance but is rarely discussed, the Multi-Proxy Detection Network (MP-Det) is presented to model object distributions in a fine-grained manner by employing multiple proxy learning, and the proxies are enforced to be diverse by minimizing a Bag-of-Instance-Words (BoIW) guided optimal transport loss. By such means, UFPMP-Det largely promotes both the detection accuracy and efficiency. Extensive experiments are carried out on the widely used VisDrone and UAVDT datasets, and UFPMP-Det reports new state-of-the-art scores at a much higher speed, highlighting its advantages. The code is available at https://github.com/PuAnysh/UFPMP-Det.

----

## [114] Modality-Adaptive Mixup and Invariant Decomposition for RGB-Infrared Person Re-identification

**Authors**: *Zhipeng Huang, Jiawei Liu, Liang Li, Kecheng Zheng, Zheng-Jun Zha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19987](https://doi.org/10.1609/aaai.v36i1.19987)

**Abstract**:

RGB-infrared person re-identification is an emerging cross-modality re-identification task, which is very challenging due to significant modality discrepancy between RGB and infrared images. In this work, we propose a novel modality-adaptive mixup and invariant decomposition (MID) approach for RGB-infrared person re-identification towards learning modality-invariant and discriminative representations. MID designs a modality-adaptive mixup scheme to generate suitable mixed modality images between RGB and infrared images for mitigating the inherent modality discrepancy at the pixel-level. It formulates modality mixup procedure as Markov decision process, where an actor-critic agent learns dynamical and local linear interpolation policy between different regions of cross-modality images under a deep reinforcement learning framework. Such policy guarantees modality-invariance in a more continuous latent space and avoids manifold intrusion by the corrupted mixed modality samples. Moreover, to further counter modality discrepancy and enforce invariant visual semantics at the feature-level, MID employs modality-adaptive convolution decomposition to disassemble a regular convolution layer into modality-specific basis layers and a modality-shared coefficient layer. Extensive experimental results on two challenging benchmarks demonstrate superior performance of MID over state-of-the-art methods.

----

## [115] MuMu: Cooperative Multitask Learning-Based Guided Multimodal Fusion

**Authors**: *Md Mofijul Islam, Tariq Iqbal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19988](https://doi.org/10.1609/aaai.v36i1.19988)

**Abstract**:

Multimodal sensors (visual, non-visual, and wearable) can provide complementary information to develop robust perception systems for recognizing activities accurately. However, it is challenging to extract robust multimodal representations due to the heterogeneous characteristics of data from multimodal sensors and disparate human activities, especially in the presence of noisy and misaligned sensor data. In this work, we propose a cooperative multitask learning-based guided multimodal fusion approach, MuMu, to extract robust multimodal representations for human activity recognition (HAR). MuMu employs an auxiliary task learning approach to extract features specific to each set of activities with shared characteristics (activity-group). MuMu then utilizes activity-group-specific features to direct our proposed Guided Multimodal Fusion Approach (GM-Fusion) for extracting complementary multimodal representations, designed as the target task. We evaluated MuMu by comparing its performance to state-of-the-art multimodal HAR approaches on three activity datasets. Our extensive experimental results suggest that MuMu outperforms all the evaluated approaches across all three datasets. Additionally, the ablation study suggests that MuMu significantly outperforms the baseline models (p<0.05), which do not use our guided multimodal fusion. Finally, the robust performance of MuMu on noisy and misaligned sensor data posits that our approach is suitable for HAR in real-world settings.

----

## [116] An Unsupervised Way to Understand Artifact Generating Internal Units in Generative Neural Networks

**Authors**: *Haedong Jeong, Jiyeon Han, Jaesik Choi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19989](https://doi.org/10.1609/aaai.v36i1.19989)

**Abstract**:

Despite significant improvements on the image generation performance of Generative Adversarial Networks (GANs), generations with low visual fidelity still have been observed. As widely used metrics for GANs focus more on the overall performance of the model, evaluation on the quality of individual generations or detection of defective generations is challenging. While recent studies try to detect featuremap units that cause artifacts and evaluate individual samples, these approaches require additional resources such as external networks or a number of training data to approximate the real data manifold. 
 In this work, we propose the concept of local activation, and devise a metric on the local activation to detect artifact generations without additional supervision.
 We empirically verify that our approach can detect and correct artifact generations from GANs with various datasets. Finally, we discuss a geometrical analysis to partially reveal the relation between the proposed concept and low visual fidelity.

----

## [117] FrePGAN: Robust Deepfake Detection Using Frequency-Level Perturbations

**Authors**: *Yonghyun Jeong, Doyeon Kim, Youngmin Ro, Jongwon Choi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19990](https://doi.org/10.1609/aaai.v36i1.19990)

**Abstract**:

Various deepfake detectors have been proposed, but challenges still exist to detect images of unknown categories or GAN models outside of the training settings.  Such issues arise from the overfitting issue, which we discover from our own analysis and the previous studies to originate from the frequency-level artifacts in generated images. We find that ignoring the frequency-level artifacts can improve the detector's generalization across various GAN models, but it can reduce the model's performance for the trained GAN models. Thus, we design a framework to generalize the deepfake detector for both the known and unseen GAN models. Our framework generates the frequency-level perturbation maps to make the generated images indistinguishable from the real images. By updating the deepfake detector along with the training of the perturbation generator, our model is trained to detect the frequency-level artifacts at the initial iterations and consider the image-level irregularities at the last iterations. For experiments, we design new test scenarios varying from the training settings in GAN models, color manipulations, and object categories. Numerous experiments validate the state-of-the-art performance of our deepfake detector.

----

## [118] Learning Disentangled Attribute Representations for Robust Pedestrian Attribute Recognition

**Authors**: *Jian Jia, Naiyu Gao, Fei He, Xiaotang Chen, Kaiqi Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19991](https://doi.org/10.1609/aaai.v36i1.19991)

**Abstract**:

Although various methods have been proposed for pedestrian attribute recognition, most studies follow the same feature learning mechanism, \ie, learning a shared pedestrian image feature to classify multiple attributes. However, this mechanism leads to low-confidence predictions and non-robustness of the model in the inference stage. In this paper, we investigate why this is the case. We mathematically discover that the central cause is that the optimal shared feature cannot maintain high similarities with multiple classifiers simultaneously in the context of minimizing classification loss. In addition, this feature learning mechanism ignores the spatial and semantic distinctions between different attributes. To address these limitations, we propose a novel disentangled attribute feature learning (DAFL) framework to learn a disentangled feature for each attribute, which exploits the semantic and spatial characteristics of attributes. The framework mainly consists of learnable semantic queries, a cascaded semantic-spatial cross-attention (SSCA) module, and a group attention merging (GAM) module. Specifically, based on learnable semantic queries, the cascaded SSCA module iteratively enhances the spatial localization of attribute-related regions and aggregates region features into multiple disentangled attribute features, used for classification and updating learnable semantic queries. The GAM module splits attributes into groups based on spatial distribution and utilizes reliable group attention to supervise query attention maps. Experiments on PETA, RAPv1, PA100k, and RAPv2 show that the proposed method performs favorably against state-of-the-art methods.

----

## [119] Degrade Is Upgrade: Learning Degradation for Low-Light Image Enhancement

**Authors**: *Kui Jiang, Zhongyuan Wang, Zheng Wang, Chen Chen, Peng Yi, Tao Lu, Chia-Wen Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19992](https://doi.org/10.1609/aaai.v36i1.19992)

**Abstract**:

Low-light image enhancement aims to improve an image's visibility while keeping its visual naturalness. Different from existing methods, which tend to accomplish the relighting task directly, we investigate the intrinsic degradation and relight the low-light image while refining the details and color in two steps. Inspired by the color image formulation (diffuse illumination color plus environment illumination color), we first estimate the degradation from low-light inputs to simulate the distortion of environment illumination color, and then refine the content to recover the loss of diffuse illumination color. To this end, we propose a novel Degradation-to-Refinement Generation Network (DRGN). Its distinctive features can be summarized as 1) A novel two-step generation network for degradation learning and content refinement. It is not only superior to one-step methods, but also capable of synthesizing sufficient paired samples to benefit the model training; 2) A multi-resolution fusion network to represent the target information (degradation or contents) in a multi-scale cooperative manner, which is more effective to address the complex unmixing problems. Extensive experiments on both the enhancement task and the joint detection task have verified the effectiveness and efficiency of our proposed method, surpassing the SOTA by 1.59dB on average and 3.18\% in mAP on the ExDark dataset. The code will be available soon.

----

## [120] HarmoFL: Harmonizing Local and Global Drifts in Federated Learning on Heterogeneous Medical Images

**Authors**: *Meirui Jiang, Zirui Wang, Qi Dou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19993](https://doi.org/10.1609/aaai.v36i1.19993)

**Abstract**:

Multiple medical institutions collaboratively training a model using federated learning (FL) has become a promising solution for maximizing the potential of data-driven models, yet the non-independent and identically distributed (non-iid) data in medical images is still an outstanding challenge in real-world practice. The feature heterogeneity caused by diverse scanners or protocols introduces a drift in the learning process, in both local (client) and global (server) optimizations, which harms the convergence as well as model performance. Many previous works have attempted to address the non-iid issue by tackling the drift locally or globally, but how to jointly solve the two essentially coupled drifts is still unclear. In this work, we concentrate on handling both local and global drifts and introduce a new harmonizing framework called HarmoFL. First, we propose to mitigate the local update drift by normalizing amplitudes of images transformed into the frequency domain to mimic a unified imaging setting, in order to generate a harmonized feature space across local clients. Second, based on harmonized features, we design a client weight perturbation guiding each local model to reach a flat optimum, where a neighborhood area of the local optimal solution has a uniformly low loss. Without any extra communication cost, the perturbation assists the global model to optimize towards a converged optimal solution by aggregating several local flat optima. We have theoretically analyzed the proposed method and empirically conducted extensive experiments on three medical image classification and segmentation tasks, showing that HarmoFL outperforms a set of recent state-of-the-art methods with promising convergence behavior. Code is available at: https://github.com/med-air/HarmoFL

----

## [121] Coarse-to-Fine Generative Modeling for Graphic Layouts

**Authors**: *Zhaoyun Jiang, Shizhao Sun, Jihua Zhu, Jian-Guang Lou, Dongmei Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19994](https://doi.org/10.1609/aaai.v36i1.19994)

**Abstract**:

Even though graphic layout generation has attracted growing attention recently, it is still challenging to synthesis realistic and diverse layouts, due to the complicated element relationships and varied element arrangements. In this work, we seek to improve the performance of layout generation by incorporating the concept of regions, which consist of a smaller number of elements and appears like a simple layout, into the generation process. Specifically, we leverage Variational Autoencoder (VAE) as the overall architecture and decompose the decoding process into two stages. The first stage predicts representations for regions, and the second stage fills in the detailed position for each element within the region based on the predicted region representation. Compared to prior studies that merely abstract the layout into a list of elements and generate all the element positions in one go, our approach has at least two advantages. First, by the two-stage decoding, our approach decouples the complex layout generation task into several simple layout generation tasks, which reduces the problem difficulty. Second, the predicted regions can help the model roughly know what the graphic layout looks like and serve as global context to improve the generation of detailed element positions. Qualitative and quantitative experiments demonstrate that our approach significantly outperforms the existing methods, especially on the complex graphic layouts.

----

## [122] DarkVisionNet: Low-Light Imaging via RGB-NIR Fusion with Deep Inconsistency Prior

**Authors**: *Shuangping Jin, Bingbing Yu, Minhao Jing, Yi Zhou, Jiajun Liang, Renhe Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19995](https://doi.org/10.1609/aaai.v36i1.19995)

**Abstract**:

RGB-NIR fusion is a promising method for low-light imaging. However, high-intensity noise in low-light images amplifies the effect of structure inconsistency between RGB-NIR images, which fails existing algorithms. To handle this, we propose a new RGB-NIR fusion algorithm called Dark Vision Net (DVN) with two technical novelties: Deep Structure and Deep Inconsistency Prior (DIP). The Deep Structure extracts clear structure details in deep multiscale feature space rather than raw input space, which is more robust to noisy inputs. Based on the deep structures from both RGB and NIR domains, we introduce the DIP to leverage the structure inconsistency to guide the fusion of RGB-NIR. Benefits from this, the proposed DVN obtains high-quality low-light images without the visual artifacts. We also propose a new dataset called Dark Vision Dataset (DVD), consisting of aligned RGB-NIR image pairs, as the first public RGB-NIR fusion benchmark. Quantitative and qualitative results on the proposed benchmark show that DVN significantly outperforms other comparison algorithms in PSNR and SSIM, especially in extremely low light conditions.

----

## [123] LAGConv: Local-Context Adaptive Convolution Kernels with Global Harmonic Bias for Pansharpening

**Authors**: *Zi-Rong Jin, Tian-Jing Zhang, Tai-Xiang Jiang, Gemine Vivone, Liang-Jian Deng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19996](https://doi.org/10.1609/aaai.v36i1.19996)

**Abstract**:

Pansharpening is a critical yet challenging low-level vision task that aims to obtain a higher-resolution image by fusing a multispectral (MS) image and a panchromatic (PAN) image. While most pansharpening methods are based on convolutional neural network (CNN) architectures with standard convolution operations, few attempts have been made with context-adaptive/dynamic convolution, which delivers impressive results on high-level vision tasks. In this paper, we propose a novel strategy to generate local-context adaptive (LCA) convolution kernels and introduce a new global harmonic (GH) bias mechanism, exploiting image local specificity as well as integrating global information, dubbed LAGConv. The proposed LAGConv can replace the standard convolution that is context-agnostic to fully perceive the particularity of each pixel for the task of remote sensing pansharpening. Furthermore, by applying the LAGConv, we provide an image fusion network architecture, which is more effective than conventional CNN-based pansharpening approaches. The superiority of the proposed method is demonstrated by extensive experiments implemented on a wide range of datasets compared with state-of-the-art pansharpening methods. Besides, more discussions testify that the proposed LAGConv outperforms recent adaptive convolution techniques for pansharpening.

----

## [124] Learning the Dynamics of Visual Relational Reasoning via Reinforced Path Routing

**Authors**: *Chenchen Jing, Yunde Jia, Yuwei Wu, Chuanhao Li, Qi Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19997](https://doi.org/10.1609/aaai.v36i1.19997)

**Abstract**:

Reasoning is a dynamic process. In cognitive theories, the dynamics of reasoning refers to reasoning states over time after successive state transitions. Modeling the cognitive dynamics is of utmost importance to simulate human reasoning capability. In this paper, we propose to learn the reasoning dynamics of visual relational reasoning by casting it as a path routing task. We present a reinforced path routing method that represents an input image via a structured visual graph and introduces a reinforcement learning based model to explore paths (sequences of nodes) over the graph based on an input sentence to infer reasoning results. By exploring such paths, the proposed method represents reasoning states clearly and characterizes state transitions explicitly to fully model the reasoning dynamics for accurate and transparent visual relational reasoning. Extensive experiments on referring expression comprehension and visual question answering demonstrate the effectiveness of our method.

----

## [125] Towards To-a-T Spatio-Temporal Focus for Skeleton-Based Action Recognition

**Authors**: *Lipeng Ke, Kuan-Chuan Peng, Siwei Lyu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19998](https://doi.org/10.1609/aaai.v36i1.19998)

**Abstract**:

Graph Convolutional Networks (GCNs) have been widely used to model the high-order dynamic dependencies for skeleton-based action recognition. Most existing approaches do not explicitly embed the high-order spatio-temporal importance to joints’ spatial connection topology and intensity, and they do not have direct objectives on their attention module to jointly learn when and where to focus on in the action sequence. To address these problems, we propose the To-a-T Spatio-Temporal Focus (STF), a skeleton-based action recognition framework that utilizes the spatio-temporal gradient to focus on relevant spatio-temporal features. We first propose the STF modules with learnable gradient-enforced and instance-dependent adjacency matrices to model the high-order spatio-temporal dynamics. Second, we propose three loss terms defined on the gradient-based spatio-temporal focus to explicitly guide the classifier when and where to look at, distinguish confusing classes, and optimize the stacked STF modules. STF outperforms the state-of-the-art methods on the NTU RGB+D 60, NTU RGB+D 120, and Kinetics Skeleton 400 datasets in all 15 settings over different views, subjects, setups, and input modalities, and STF also shows better accuracy on scarce data and dataset shifting settings.

----

## [126] MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition

**Authors**: *Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan, Rynson W. H. Lau*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.19999](https://doi.org/10.1609/aaai.v36i1.19999)

**Abstract**:

Existing portrait matting methods either require auxiliary inputs that are costly to obtain or involve multiple stages that are computationally expensive, making them less suitable for real-time applications. In this work, we present a light-weight matting objective decomposition network (MODNet) for portrait matting in real-time with a single input image. The key idea behind our efficient design is by optimizing a series of sub-objectives simultaneously via explicit constraints. In addition, MODNet includes two novel techniques for improving model efficiency and robustness. First, an Efficient Atrous Spatial Pyramid Pooling (e-ASPP) module is introduced to fuse multi-scale features for semantic estimation. Second, a self-supervised sub-objectives consistency (SOC) strategy is proposed to adapt MODNet to real-world data to address the domain shift problem common to trimap-free methods. MODNet is easy to be trained in an end-to-end manner. It is much faster than contemporaneous methods and runs at 67 frames per second on a 1080Ti GPU. Experiments show that MODNet outperforms prior trimap-free methods by a large margin on both Adobe Matting Dataset and a carefully designed photographic portrait matting (PPM-100) benchmark proposed by us. Further, MODNet achieves remarkable results on daily photos and videos.

----

## [127] Learning Mixture of Domain-Specific Experts via Disentangled Factors for Autonomous Driving

**Authors**: *Inhan Kim, Joonyeong Lee, Daijin Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.20000](https://doi.org/10.1609/aaai.v36i1.20000)

**Abstract**:

Since human drivers only consider the driving-related factors that affect vehicle control depending on the situation, they can drive safely even in diverse driving environments. To mimic this behavior, we propose an autonomous driving framework based on the two-stage representation learning that initially splits the latent features as domain-specific features and domain-general features. Subsequently, the dynamic-object features, which contain information of dynamic objects, are disentangled from latent features using mutual information estimator. In this study, the problem in behavior cloning is divided into several domain-specific subspaces, with experts becoming specialized on each domain-specific policy. The proposed mixture of domain-specific experts (MoDE) model predicts the final control values through the cooperation of experts using a gating function. The domain-specific features are used to calculate the importance weight of the domain-specific experts, and the disentangled domain-general and dynamic-object features are applied in estimating the control values. To validate the proposed MoDE model, we conducted several experiments and achieved a higher success rate on the CARLA benchmarks under several conditions and tasks than state-of-the-art approaches.

----

## [128] Towards Versatile Pedestrian Detector with Multisensory-Matching and Multispectral Recalling Memory

**Authors**: *Jung Uk Kim, Sungjune Park, Yong Man Ro*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.20001](https://doi.org/10.1609/aaai.v36i1.20001)

**Abstract**:

Recently, automated surveillance cameras can change a visible sensor and a thermal sensor for all-day operation. However, existing single-modal pedestrian detectors mainly focus on detecting pedestrians in only one specific modality (i.e., visible or thermal), so they cannot cope with other modal inputs. In addition, recent multispectral pedestrian detectors have shown remarkable performance by adopting multispectral modalities, but they also have limitations in practical applications (e.g., different Field-of-View (FoV) and frame rate). In this paper, we introduce a versatile pedestrian detector that shows robust detection performance in any single modality. We propose a multisensory-matching contrastive loss to reduce the difference between the visual representation of pedestrians in the visible and thermal modalities. Moreover, for the robust detection on a single modality, we design a Multispectral Recalling (MSR) Memory. The MSR Memory enhances the visual representation of the single modal features by recalling that of the multispectral modalities. To guide the MSR Memory to store the multispectral modal contexts, we introduce a multispectral recalling loss. It enables the pedestrian detector to encode more discriminative features with a single input modality. We believe our method is a step forward detector that can be applied to a variety of real-world applications. The comprehensive experimental results verify the effectiveness of the proposed method.

----

## [129] Semantic Feature Extraction for Generalized Zero-Shot Learning

**Authors**: *Junhan Kim, Kyuhong Shim, Byonghyo Shim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.20002](https://doi.org/10.1609/aaai.v36i1.20002)

**Abstract**:

Generalized zero-shot learning (GZSL) is a technique to train a deep learning model to identify unseen classes using the attribute.
In this paper, we put forth a new GZSL technique that improves the GZSL classification performance greatly. 
Key idea of the proposed approach, henceforth referred to as semantic feature extraction-based GZSL (SE-GZSL), is to use the semantic feature containing only attribute-related information in learning the relationship between the image and the attribute. 
In doing so, we can remove the interference, if any, caused by the attribute-irrelevant information contained in the image feature. 
To train a network extracting the semantic feature, we present two novel loss functions, 1) mutual information-based loss to capture all the attribute-related information in the image feature and 2) similarity-based loss to remove unwanted attribute-irrelevant information. 
From extensive experiments using various datasets, we show that the proposed SE-GZSL technique outperforms conventional GZSL approaches by a large margin.

----

## [130] Distinguishing Homophenes Using Multi-Head Visual-Audio Memory for Lip Reading

**Authors**: *Minsu Kim, Jeong Hun Yeo, Yong Man Ro*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.20003](https://doi.org/10.1609/aaai.v36i1.20003)

**Abstract**:

Recognizing speech from silent lip movement, which is called lip reading, is a challenging task due to 1) the inherent information insufficiency of lip movement to fully represent the speech, and 2) the existence of homophenes that have similar lip movement with different pronunciations. In this paper, we try to alleviate the aforementioned two challenges in lip reading by proposing a Multi-head Visual-audio Memory (MVM). Firstly, MVM is trained with audio-visual datasets and remembers audio representations by modelling the inter-relationships of paired audio-visual representations. At the inference stage, visual input alone can extract the saved audio representation from the memory by examining the learned inter-relationships. Therefore, the lip reading model can complement the insufficient visual information with the extracted audio representations. Secondly, MVM is composed of multi-head key memories for saving visual features and one value memory for saving audio knowledge, which is designed to distinguish the homophenes. With the multi-head key memories, MVM extracts possible candidate audio features from the memory, which allows the lip reading model to consider the possibility of which pronunciations can be represented from the input lip movement. This also can be viewed as an explicit implementation of the one-to-many mapping of viseme-to-phoneme. Moreover, MVM is employed in multi-temporal levels to consider the context when retrieving the memory and distinguish the homophenes. Extensive experimental results verify the effectiveness of the proposed method in lip reading and in distinguishing the homophenes.

----

## [131] Deep Translation Prior: Test-Time Training for Photorealistic Style Transfer

**Authors**: *Sunwoo Kim, Soohyun Kim, Seungryong Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.20004](https://doi.org/10.1609/aaai.v36i1.20004)

**Abstract**:

Recent techniques to solve photorealistic style transfer within deep convolutional neural networks (CNNs) generally require intensive training from large-scale datasets, thus having limited applicability and poor generalization ability to unseen images or styles. To overcome this, we propose a novel framework, dubbed Deep Translation Prior (DTP), to accomplish photorealistic style transfer through test-time training on given input image pair with untrained networks, which learns an image pair-specific translation prior and thus yields better performance and generalization. Tailored for such test-time training for style transfer, we present novel network architectures, with two sub-modules of correspondence and generation modules, and loss functions consisting of contrastive content, style, and cycle consistency losses. Our framework does not require offline training phase for style transfer, which has been one of the main challenges in existing methods, but the networks are to be solely learned during test time. Experimental results prove that our framework has a better generalization ability to unseen image pairs and even outperforms the state-of-the-art methods.

----

## [132] PrivateSNN: Privacy-Preserving Spiking Neural Networks

**Authors**: *Youngeun Kim, Yeshwanth Venkatesha, Priyadarshini Panda*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.20005](https://doi.org/10.1609/aaai.v36i1.20005)

**Abstract**:

How can we bring both privacy and energy-efficiency to a neural system? In this paper, we propose PrivateSNN, which aims to build low-power Spiking Neural Networks (SNNs) from a pre-trained ANN model without leaking sensitive information contained in a dataset. Here, we tackle two types of leakage problems: 1) Data leakage is caused when the networks access real training data during an ANN-SNN conversion process. 2) Class leakage is caused when class-related features can be reconstructed from network parameters. In order to address the data leakage issue, we generate synthetic images from the pre-trained ANNs and convert ANNs to SNNs using the generated images. However, converted SNNs remain vulnerable to class leakage since the weight parameters have the same (or scaled) value with respect to ANN parameters. Therefore, we encrypt SNN weights by training SNNs with a temporal spike-based learning rule. Updating weight parameters with temporal data makes SNNs difficult to be interpreted in the spatial domain. We observe that the encrypted PrivateSNN eliminates data and class leakage issues with a slight performance drop (less than ~2%) and significant energy-efficiency gain (about 55x) compared to the standard ANN. We conduct extensive experiments on various datasets including  CIFAR10, CIFAR100, and TinyImageNet, highlighting the importance of privacy-preserving SNN training.

----

## [133] NaturalInversion: Data-Free Image Synthesis Improving Real-World Consistency

**Authors**: *Yujin Kim, Dogyun Park, Dohee Kim, Suhyun Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.20006](https://doi.org/10.1609/aaai.v36i1.20006)

**Abstract**:

We introduce NaturalInversion, a novel model inversion-based method to synthesize images that agrees well with the original data distribution without using real data. In NaturalInversion, we propose: (1) a Feature Transfer Pyramid which uses enhanced image prior of the original data by combining the multi-scale feature maps extracted from the pre-trained classifier, (2) a one-to-one approach generative model where only one batch of images are synthesized by one generator to bring the non-linearity to optimization and to ease the overall optimizing process, (3) learnable Adaptive Channel Scaling parameters which are end-to-end trained to scale the output image channel to utilize the original image prior further. With our NaturalInversion, we synthesize images from classifiers trained on CIFAR-10/100 and show that our images are more consistent with original data distribution than prior works by visualization and additional analysis. Furthermore, our synthesized images outperform prior works on various applications such as knowledge distillation and pruning, demonstrating the effectiveness of our proposed method.

----

## [134] Joint 3D Object Detection and Tracking Using Spatio-Temporal Representation of Camera Image and LiDAR Point Clouds

**Authors**: *Junho Koh, Jaekyum Kim, Jin Hyeok Yoo, Yecheol Kim, Dongsuk Kum, Jun Won Choi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i1.20007](https://doi.org/10.1609/aaai.v36i1.20007)

**Abstract**:

In this paper, we propose a new joint object detection and tracking (JoDT) framework for 3D object detection and tracking based on camera and LiDAR sensors. The proposed method, referred to as 3D DetecTrack, enables the detector and tracker to cooperate to generate a spatio-temporal representation of the camera and LiDAR data, with which 3D object detection and tracking are then performed. The detector constructs the spatio-temporal features via the weighted temporal aggregation of the spatial features obtained by the camera and LiDAR fusion. Then, the detector reconfigures the initial detection results using information from the tracklets maintained up to the previous time step. Based on the spatio-temporal features generated by the detector, the tracker associates the detected objects with previously tracked objects using a graph neural network (GNN). We devise a fully-connected GNN facilitated by a combination of rule-based edge pruning and attention-based edge gating, which exploits both spatial and temporal object contexts to improve tracking performance. The experiments conducted on both KITTI and nuScenes benchmarks demonstrate that the proposed 3D DetecTrack achieves significant improvements in both detection and tracking performances over baseline methods and achieves state-of-the-art performance among existing methods through collaboration between the detector and tracker.

----

## [135] Amplitude Spectrum Transformation for Open Compound Domain Adaptive Semantic Segmentation

**Authors**: *Jogendra Nath Kundu, Akshay R. Kulkarni, Suvaansh Bhambri, Varun Jampani, Venkatesh Babu Radhakrishnan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20008](https://doi.org/10.1609/aaai.v36i2.20008)

**Abstract**:

Open compound domain adaptation (OCDA) has emerged as a practical adaptation setting which considers a single labeled source domain against a compound of multi-modal unlabeled target data in order to generalize better on novel unseen domains. We hypothesize that an improved disentanglement of domain-related and task-related factors of dense intermediate layer features can greatly aid OCDA. Prior-arts attempt this indirectly by employing adversarial domain discriminators on the spatial CNN output. However, we find that latent features derived from the Fourier-based amplitude spectrum of deep CNN features hold a more tractable mapping with domain discrimination. Motivated by this, we propose a novel feature space Amplitude Spectrum Transformation (AST). During adaptation, we employ the AST auto-encoder for two purposes. First, carefully mined source-target instance pairs undergo a simulation of cross-domain feature stylization (AST-Sim) at a particular layer by altering the AST-latent. Second, AST operating at a later layer is tasked to normalize (AST-Norm) the domain content by fixing its latent to a mean prototype. Our simplified adaptation technique is not only clustering-free but also free from complex adversarial alignment. We achieve leading performance against the prior arts on the OCDA scene segmentation benchmarks.

----

## [136] Siamese Network with Interactive Transformer for Video Object Segmentation

**Authors**: *Meng Lan, Jing Zhang, Fengxiang He, Lefei Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20009](https://doi.org/10.1609/aaai.v36i2.20009)

**Abstract**:

Semi-supervised video object segmentation (VOS) refers to segmenting the target object in remaining frames given its annotation in the first frame, which has been actively studied in recent years. The key challenge lies in finding effective ways to exploit the spatio-temporal context of past frames to help learn discriminative target representation of current frame. In this paper, we propose a novel Siamese network with a specifically designed interactive transformer, called SITVOS, to enable effective context propagation from historical to current frames. Technically, we use the transformer encoder and decoder to handle the past frames and current frame separately, i.e., the encoder encodes robust spatio-temporal context of target object from the past frames, while the decoder takes the feature embedding of current frame as the query to retrieve the target from the encoder output. To further enhance the target representation, a feature interaction module (FIM) is devised to promote the information flow between the encoder and decoder. Moreover, we employ the Siamese architecture to extract backbone features of both past and current frames, which enables feature reuse and is more efficient than existing methods. Experimental results on three challenging benchmarks validate the superiority of SITVOS over state-of-the-art methods. Code is available at https://github.com/LANMNG/SITVOS.

----

## [137] Adversarial Attack for Asynchronous Event-Based Data

**Authors**: *Wooju Lee, Hyun Myung*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20010](https://doi.org/10.1609/aaai.v36i2.20010)

**Abstract**:

Deep neural networks (DNNs) are vulnerable to adversarial examples that are carefully designed to cause the deep learning model to make mistakes. Adversarial examples of 2D images and 3D point clouds have been extensively studied, but studies on event-based data are limited. Event-based data can be an alternative to a 2D image under high-speed movements, such as autonomous driving. However, the given adversarial events make the current deep learning model vulnerable to safety issues. In this work, we generate adversarial examples and then train the robust models for event-based data, for the first time. Our algorithm shifts the time of the original events and generates additional adversarial events. Additional adversarial events are generated in two stages. First, null events are added to the event-based data to generate additional adversarial events. The perturbation size can be controlled with the number of null events. Second, the location and time of additional adversarial events are set to mislead DNNs in a gradient-based attack. Our algorithm achieves an attack success rate of 97.95% on the N-Caltech101 dataset. Furthermore, the adversarial training model improves robustness on the adversarial event data compared to the original model.

----

## [138] Iteratively Selecting an Easy Reference Frame Makes Unsupervised Video Object Segmentation Easier

**Authors**: *Youngjo Lee, Hongje Seong, Euntai Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20011](https://doi.org/10.1609/aaai.v36i2.20011)

**Abstract**:

Unsupervised video object segmentation (UVOS) is a per-pixel binary labeling problem which aims at separating the foreground object from the background in the video without using the ground truth (GT) mask of the foreground object. Most of the previous UVOS models use the first frame or the entire video as a reference frame to specify the mask of the foreground object. Our question is why the first frame should be selected as a reference frame or why the entire video should be used to specify the mask. We believe that we can select a better reference frame to achieve the better UVOS performance than using only the first frame or the entire video as a reference frame. In our paper, we propose Easy Frame Selector (EFS). The EFS enables us to select an "easy" reference frame that makes the subsequent VOS become easy, thereby improving the VOS performance. Furthermore, we propose a new framework named as Iterative Mask Prediction (IMP). In the framework, we repeat applying EFS to the given video and selecting an "easier" reference frame from the video than the previous iteration, increasing the VOS performance incrementally. The IMP consists of EFS, Bi-directional Mask Prediction (BMP), and Temporal Information Updating (TIU). From the proposed framework, we achieve state-of-the-art performance in three UVOS benchmark sets: DAVIS16, FBMS, and SegTrack-V2.

----

## [139] SCTN: Sparse Convolution-Transformer Network for Scene Flow Estimation

**Authors**: *Bing Li, Cheng Zheng, Silvio Giancola, Bernard Ghanem*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20012](https://doi.org/10.1609/aaai.v36i2.20012)

**Abstract**:

We propose a novel scene flow estimation approach to capture and infer 3D motions from point clouds. Estimating 3D motions for point clouds is challenging, since a point cloud is unordered and its density is significantly non-uniform. Such unstructured data poses difficulties in matching corresponding points between point clouds, leading to inaccurate flow estimation. We propose a novel architecture named Sparse Convolution-Transformer Network (SCTN) that equips the sparse convolution with the transformer. Specifically, by leveraging the sparse convolution, SCTN transfers irregular point cloud into locally consistent flow features for estimating spatially consistent motions within an object/local object part. We further propose to explicitly learn point relations using a point transformer module, different from exiting methods. We show that the learned relation-based contextual information is rich and helpful for matching corresponding points, benefiting scene flow estimation. In addition, a novel loss function is proposed to adaptively encourage flow consistency according to feature similarity. Extensive experiments demonstrate that our proposed approach achieves a new state of the art in scene flow estimation. Our approach achieves an error of 0.038 and 0.037 (EPE3D) on FlyingThings3D and KITTI Scene Flow respectively, which significantly outperforms previous methods by large margins.

----

## [140] Shrinking Temporal Attention in Transformers for Video Action Recognition

**Authors**: *Bonan Li, Pengfei Xiong, Congying Han, Tiande Guo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20013](https://doi.org/10.1609/aaai.v36i2.20013)

**Abstract**:

Spatiotemporal modeling in an unified architecture is key for video action recognition. This paper proposes a Shrinking Temporal Attention Transformer (STAT), which efficiently builts spatiotemporal attention maps considering the attenuation of spatial attention in short and long temporal sequences. Specifically, for short-term temporal tokens, query token interacts with them in a fine-grained manner in dealing with short-range motion. It then shrinks to a coarse attention in neighborhood for long-term tokens, to provide larger receptive field for long-range spatial aggregation. Both of them are composed in a short-long temporal integrated block to build visual appearances and temporal structure concurrently with lower costly in computation. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple action recognition benchmarks including Kinetics400 and Something-Something v2, outperforming prior methods with 50% less FLOPs and without any pretrained model.

----

## [141] DanceFormer: Music Conditioned 3D Dance Generation with Parametric Motion Transformer

**Authors**: *Buyu Li, Yongchi Zhao, Zhelun Shi, Lu Sheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20014](https://doi.org/10.1609/aaai.v36i2.20014)

**Abstract**:

Generating 3D dances from music is an emerged research task that benefits a lot of applications in vision and graphics. Previous works treat this task as sequence generation, however, it is challenging to render a music-aligned long-term sequence with high kinematic complexity and coherent movements. In this paper, we reformulate it by a two-stage process, i.e., a key pose generation and then an in-between parametric motion curve prediction, where the key poses are easier to be synchronized with the music beats and the parametric curves can be efficiently regressed to render fluent rhythm-aligned movements. We named the proposed method as DanceFormer, which includes two cascading kinematics-enhanced transformer-guided networks (called DanTrans) that tackle each stage, respectively. Furthermore, we propose a large-scale music conditioned 3D dance dataset, called PhantomDance, that is accurately labeled by experienced animators rather than reconstruction or motion capture. This dataset also encodes dances as key poses and parametric motion curves apart from pose sequences, thus benefiting the training of our DanceFormer. Extensive experiments demonstrate that the proposed method, even trained by existing datasets, can generate fluent, performative, and music-matched 3D dances that surpass previous works quantitatively and qualitatively. Moreover, the proposed DanceFormer, together with the PhantomDance dataset, are seamlessly compatible with industrial animation software, thus facilitating the adaptation for various downstream applications.

----

## [142] Interpretable Generative Adversarial Networks

**Authors**: *Chao Li, Kelu Yao, Jin Wang, Boyu Diao, Yongjun Xu, Quanshi Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20015](https://doi.org/10.1609/aaai.v36i2.20015)

**Abstract**:

Learning a disentangled representation is still a challenge in the field of the interpretability of generative adversarial networks (GANs). This paper proposes a generic method to modify a traditional GAN into an interpretable GAN, which ensures that filters in an intermediate layer of the generator encode disentangled localized visual concepts. Each filter in the layer is supposed to consistently generate image regions corresponding to the same visual concept when generating different images. The interpretable GAN learns to automatically discover meaningful visual concepts without any annotations of visual concepts. The interpretable GAN enables people to modify a specific visual concept on generated images by manipulating feature maps of the corresponding filters in the layer. Our method can be broadly applied to different types of GANs. Experiments have demonstrated the effectiveness of our method.

----

## [143] Cross-Modal Object Tracking: Modality-Aware Representations and a Unified Benchmark

**Authors**: *Chenglong Li, Tianhao Zhu, Lei Liu, Xiaonan Si, Zilin Fan, Sulan Zhai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20016](https://doi.org/10.1609/aaai.v36i2.20016)

**Abstract**:

In many visual systems, visual tracking often bases on RGB image sequences, in which some targets are invalid in low-light conditions, and tracking performance is thus affected significantly. Introducing other modalities such as depth and infrared data is an effective way to handle imaging limitations of individual sources, but multi-modal imaging platforms usually require elaborate designs and cannot be applied in many real-world applications at present. Near-infrared (NIR) imaging becomes an essential part of many surveillance cameras, whose imaging is switchable between RGB and NIR based on the light intensity. These two modalities are heterogeneous with very different visual properties and thus bring big challenges for visual tracking. However, existing works have not studied this challenging problem. In this work, we address the cross-modal object tracking problem and contribute a new video dataset, including 654 cross-modal image sequences with over 481K frames in total, and the average video length is more than 735 frames. To promote the research and development of cross-modal object tracking, we propose a new algorithm, which learns the modality-aware target representation to mitigate the appearance gap between RGB and NIR modalities in the tracking process. It is plug-and-play and could thus be flexibly embedded into different tracking frameworks. Extensive experiments on the dataset are conducted, and we demonstrate the effectiveness of the proposed algorithm in two representative tracking frameworks against 19 state-of-the-art tracking methods. Dataset, code, model and results are available at https://github.com/mmic-lcl/source-code.

----

## [144] You Only Infer Once: Cross-Modal Meta-Transfer for Referring Video Object Segmentation

**Authors**: *Dezhuang Li, Ruoqi Li, Lijun Wang, Yifan Wang, Jinqing Qi, Lu Zhang, Ting Liu, Qingquan Xu, Huchuan Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20017](https://doi.org/10.1609/aaai.v36i2.20017)

**Abstract**:

We present YOFO (You Only inFer Once), a new paradigm for referring video object segmentation (RVOS) that operates in an one-stage manner. Our key insight is that the language descriptor should serve as target-specific guidance to identify the target object, while a direct feature fusion of image and language can increase feature complexity and thus may be sub-optimal for RVOS. To this end, we propose a meta-transfer module, which is trained in a learning-to-learn fashion and aims to transfer the target-specific information from the language domain to the image domain, while discarding the uncorrelated complex variations of language description. To bridge the gap between the image and language domains, we develop a multi-scale cross-modal feature mining block that aggregates all the essential features required by RVOS from both domains and generates regression labels for the meta-transfer module. The whole system can be trained in an end-to-end manner and shows competitive performance against state-of-the-art two-stage approaches.

----

## [145] Knowledge Distillation for Object Detection via Rank Mimicking and Prediction-Guided Feature Imitation

**Authors**: *Gang Li, Xiang Li, Yujie Wang, Shanshan Zhang, Yichao Wu, Ding Liang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20018](https://doi.org/10.1609/aaai.v36i2.20018)

**Abstract**:

Knowledge Distillation (KD) is a widely-used technology to inherit information from cumbersome teacher models to compact student models, consequently realizing model compression and acceleration. Compared with image classification, object detection is a more complex task, and designing specific KD methods for object detection is non-trivial. In this work, we elaborately study the behaviour difference between the teacher and student detection models, and obtain two intriguing observations: First, the teacher and student rank their detected candidate boxes quite differently, which results in their precision discrepancy. Second, there is a considerable gap between the feature response differences and prediction differences between teacher and student, indicating that equally imitating all the feature maps of the teacher is the sub-optimal choice for improving the student's accuracy. Based on the two observations, we propose Rank Mimicking (RM) and Prediction-guided Feature Imitation (PFI) for distilling one-stage detectors, respectively. RM takes the rank of candidate boxes from teachers as a new form of knowledge to distill, which consistently outperforms the traditional soft label distillation. PFI attempts to correlate feature differences with prediction differences, making feature imitation directly help to improve the student's accuracy. On MS COCO and PASCAL VOC benchmarks, extensive experiments are conducted on various detectors with different backbones to validate the effectiveness of our method. Specifically, RetinaNet with ResNet50 achieves 40.4% mAP on MS COCO, which is 3.5% higher than its baseline, and also outperforms previous KD methods.

----

## [146] Rethinking Pseudo Labels for Semi-supervised Object Detection

**Authors**: *Hengduo Li, Zuxuan Wu, Abhinav Shrivastava, Larry S. Davis*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20019](https://doi.org/10.1609/aaai.v36i2.20019)

**Abstract**:

Recent advances in semi-supervised object detection (SSOD) are largely driven by consistency-based pseudo-labeling methods for image classification tasks, producing pseudo labels as supervisory signals. However, when using pseudo labels, there is a lack of consideration in localization precision and amplified class imbalance, both of which are critical for detection tasks. In this paper, we introduce certainty-aware pseudo labels tailored for object detection, which can effectively estimate the classification and localization quality of derived pseudo labels. This is achieved by converting conventional localization as a classification task followed by refinement. Conditioned on classification and localization quality scores, we dynamically adjust the thresholds used to generate pseudo labels and reweight loss functions for each category to alleviate the class imbalance problem. Extensive experiments demonstrate that our method improves state-of-the-art SSOD performance by 1-2% AP on COCO and PASCAL VOC while being orthogonal and complementary to most existing methods. In the limited-annotation regime, our approach improves supervised baselines by up to 10% AP using only 1-10% labeled data from COCO.

----

## [147] Action-Aware Embedding Enhancement for Image-Text Retrieval

**Authors**: *Jiangtong Li, Li Niu, Liqing Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20020](https://doi.org/10.1609/aaai.v36i2.20020)

**Abstract**:

Image-text retrieval plays a central role in bridging vision and language, which aims to reduce the semantic discrepancy between images and texts. Most of existing works rely on refined words and objects representation through the data-oriented method to capture the word-object cooccurrence. Such approaches are prone to ignore the asymmetric action relation between images and texts, that is, the text has explicit action representation (i.e., verb phrase) while the image only contains implicit action information. In this paper, we propose Action-aware Memory-Enhanced embedding (AME) method for image-text retrieval, which aims to emphasize the action information when mapping the images and texts into a shared embedding space. Specifically, we integrate action prediction along with an action-aware memory bank to enrich the image and text features with action-similar text features. The effectiveness of our proposed AME method is verified by comprehensive experimental results on two benchmark datasets.

----

## [148] Retinomorphic Object Detection in Asynchronous Visual Streams

**Authors**: *Jianing Li, Xiao Wang, Lin Zhu, Jia Li, Tiejun Huang, Yonghong Tian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20021](https://doi.org/10.1609/aaai.v36i2.20021)

**Abstract**:

Due to high-speed motion blur and challenging illumination, conventional frame-based cameras have encountered an important challenge in object detection tasks. Neuromorphic cameras that output asynchronous visual streams instead of intensity frames, by taking the advantage of high temporal resolution and high dynamic range, have brought a new perspective to address the challenge. In this paper, we propose a novel problem setting, retinomorphic object detection, which is the first trial that integrates foveal-like and peripheral-like visual streams. Technically, we first build a large-scale multimodal neuromorphic object detection dataset (i.e., PKU-Vidar-DVS) over 215.5k spatio-temporal synchronized labels. Then, we design temporal aggregation representations to preserve the spatio-temporal information from asynchronous visual streams. Finally, we present a novel bio-inspired unifying framework to fuse two sensing modalities via a dynamic interaction mechanism. Our experimental evaluation shows that our approach has significant improvements over the state-of-the-art methods with the single-modality, especially in high-speed motion and low-light scenarios. We hope that our work will attract further research into this newly identified, yet crucial research direction. Our dataset can be available at https://www.pkuml.org/resources/pku-vidar-dvs.html.

----

## [149] Learning from Weakly-Labeled Web Videos via Exploring Sub-concepts

**Authors**: *Kunpeng Li, Zizhao Zhang, Guanhang Wu, Xuehan Xiong, Chen-Yu Lee, Zhichao Lu, Yun Fu, Tomas Pfister*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20022](https://doi.org/10.1609/aaai.v36i2.20022)

**Abstract**:

Learning visual knowledge from massive weakly-labeled web videos has attracted growing research interests thanks to the large corpus of easily accessible video data on the Internet. However, for video action recognition, the action of interest might only exist in arbitrary clips of untrimmed web videos, resulting in high label noises in the temporal space. To address this challenge, we introduce a new method for pre-training video action recognition models using queried web videos. Instead of trying to filter out potential noises, we propose to provide fine-grained supervision signals by defining the concept of Sub-Pseudo Label (SPL). Specifically, SPL spans out a new set of meaningful "middle ground" label space constructed by extrapolating the original weak labels during video querying and the prior knowledge distilled from a teacher model. Consequently, SPL provides enriched supervision for video models to learn better representations and improves data utilization efficiency of untrimmed videos. We validate the effectiveness of our method on four video action recognition datasets and a weakly-labeled image dataset. Experiments show that SPL outperforms several existing pre-training strategies and the learned representations lead to competitive results on several benchmarks.

----

## [150] Learning Universal Adversarial Perturbation by Adversarial Example

**Authors**: *Maosen Li, Yanhua Yang, Kun Wei, Xu Yang, Heng Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20023](https://doi.org/10.1609/aaai.v36i2.20023)

**Abstract**:

Deep learning models have shown to be susceptible to universal adversarial perturbation (UAP), which has aroused wide concerns in the community. Compared with the conventional adversarial attacks that generate adversarial samples at the instance level, UAP can fool the target model for different instances with only a single perturbation, enabling us to evaluate the robustness of the model from a more effective and accurate perspective. The existing universal attack methods fail to exploit the differences and connections between the instance and universal levels to produce dominant perturbations. To address this challenge, we propose a new universal attack method that unifies instance-specific and universal attacks from a feature perspective to generate a more dominant UAP. Specifically, we reformulate the UAP generation task as a minimax optimization problem and then utilize the instance-specific attack method to solve the minimization problem thereby obtaining better training data for generating UAP. At the same time, we also introduce a consistency regularizer to explore the relationship between training data, thus further improving the dominance of the generated UAP. Furthermore, our method is generic with no additional assumptions about the training data and hence can be applied to both data-dependent (supervised) and data-independent (unsupervised) manners. Extensive experiments demonstrate that the proposed method improves the performance by a significant margin over the existing methods in both data-dependent and data-independent settings. Code is available at https://github.com/lisenxd/AT-UAP.

----

## [151] Logit Perturbation

**Authors**: *Mengyang Li, Fengguang Su, Ou Wu, Ji Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20024](https://doi.org/10.1609/aaai.v36i2.20024)

**Abstract**:

Features, logits, and labels are the three primary data when a sample passes through a deep neural network. Feature perturbation and label perturbation receive increasing attention in recent years. They have been proven to be useful in various deep learning approaches. For example, (adversarial) feature perturbation can improve the robustness or even generalization capability of learned models. However, limited studies have explicitly explored for the perturbation of logit vectors. This work discusses several existing methods related to logit perturbation. Based on a unified viewpoint between positive/negative data augmentation and loss variations incurred by logit perturbation, a new method is proposed to explicitly learn to perturb logits. A comparative analysis is conducted for the perturbations used in our and existing methods. Extensive experiments on benchmark image classification data sets and their long-tail versions indicated the competitive performance of our learning method. In addition, existing methods can be further improved by utilizing our method.

----

## [152] Neighborhood-Adaptive Structure Augmented Metric Learning

**Authors**: *Pandeng Li, Yan Li, Hongtao Xie, Lei Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20025](https://doi.org/10.1609/aaai.v36i2.20025)

**Abstract**:

Most metric learning techniques typically focus on sample embedding learning, while implicitly assume a homogeneous local neighborhood around each sample, based on the metrics used in training ( e.g., hypersphere for Euclidean distance or unit hyperspherical crown for cosine distance). As real-world data often lies on a low-dimensional manifold curved in a high-dimensional space, it is unlikely that everywhere of the manifold shares the same local structures in the input space. Besides, considering the non-linearity of neural networks, the local structure in the output embedding space may not be homogeneous as assumed. Therefore, representing each sample simply with its embedding while ignoring its individual neighborhood structure would have limitations in Embedding-Based Retrieval (EBR). By exploiting the heterogeneity of local structures in the embedding space, we propose a Neighborhood-Adaptive Structure Augmented metric learning framework (NASA), where the neighborhood structure is realized as a  structure embedding, and learned along with the sample embedding in a self-supervised manner. In this way, without any modifications, most indexing techniques can be used to support large-scale EBR with NASA embeddings. Experiments on six standard benchmarks with two kinds of embeddings, i.e., binary embeddings and real-valued embeddings, show that our method significantly improves and outperforms the state-of-the-art methods.

----

## [153] Stereo Neural Vernier Caliper

**Authors**: *Shichao Li, Zechun Liu, Zhiqiang Shen, Kwang-Ting Cheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20026](https://doi.org/10.1609/aaai.v36i2.20026)

**Abstract**:

We propose a new object-centric framework for learning-based stereo 3D object detection. Previous studies build scene-centric representations that do not consider the significant variation among outdoor instances and thus lack the flexibility and functionalities that an instance-level model can offer. We build such an instance-level model by formulating and tackling a local update problem, i.e., how to predict a refined update given an initial 3D cuboid guess. We demonstrate how solving this problem can complement scene-centric approaches in (i) building a coarse-to-fine multi-resolution system, (ii) performing model-agnostic object location refinement, and (iii) conducting stereo 3D tracking-by-detection. Extensive experiments demonstrate the effectiveness of our approach, which achieves state-of-the-art performance on the KITTI benchmark. Code and pre-trained models are available at https://github.com/Nicholasli1995/SNVC.

----

## [154] EditVAE: Unsupervised Parts-Aware Controllable 3D Point Cloud Shape Generation

**Authors**: *Shidi Li, Miaomiao Liu, Christian Walder*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20027](https://doi.org/10.1609/aaai.v36i2.20027)

**Abstract**:

This paper tackles the problem of parts-aware point cloud generation. Unlike existing works which require the point cloud to be segmented into parts a priori, our parts-aware editing and generation are performed in an unsupervised manner. We achieve this with a simple modification of the Variational Auto-Encoder which yields a joint model of the point cloud itself along with a schematic representation of it as a combination of shape primitives. In particular, we introduce a latent representation of the point cloud which can be decomposed into a disentangled representation for each part of the shape. These parts are in turn disentangled into both a shape primitive and a point cloud representation, along with a standardising transformation to a canonical coordinate system. The dependencies between our standardising transformations preserve the spatial dependencies between the parts in a manner that allows meaningful parts-aware point cloud generation and shape editing. In addition to the flexibility afforded by our disentangled representation, the inductive bias introduced by our joint modeling approach yields state-of-the-art experimental results on the ShapeNet dataset.

----

## [155] Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection

**Authors**: *Shuo Li, Fang Liu, Licheng Jiao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20028](https://doi.org/10.1609/aaai.v36i2.20028)

**Abstract**:

Weakly supervised Video Anomaly Detection (VAD) using Multi-Instance Learning (MIL) is usually based on the fact that the anomaly score of an abnormal snippet is higher than that of a normal snippet. In the beginning of training, due to the limited accuracy of the model, it is easy to select the wrong abnormal snippet. In order to reduce the probability of selection errors, we first propose a Multi-Sequence Learning (MSL) method and a hinge-based MSL ranking loss that uses a sequence composed of multiple snippets as an optimization unit. We then design a Transformer-based MSL network to learn both video-level anomaly probability and snippet-level anomaly scores. In the inference stage, we propose to use the video-level anomaly probability to suppress the fluctuation of snippet-level anomaly scores. Finally, since VAD needs to predict the snippet-level anomaly scores, by gradually reducing the length of selected sequence, we propose a self-training strategy to gradually refine the anomaly scores. Experimental results show that our method achieves significant improvements on ShanghaiTech, UCF-Crime, and XD-Violence.

----

## [156] TA2N: Two-Stage Action Alignment Network for Few-Shot Action Recognition

**Authors**: *Shuyuan Li, Huabin Liu, Rui Qian, Yuxi Li, John See, Mengjuan Fei, Xiaoyuan Yu, Weiyao Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20029](https://doi.org/10.1609/aaai.v36i2.20029)

**Abstract**:

Few-shot action recognition aims to recognize novel action classes (query) using just a few samples (support). The majority of current approaches follow the metric learning paradigm, which learns to compare the similarity between videos. Recently, it has been observed that directly measuring this similarity is not ideal since different action instances may show distinctive temporal distribution, resulting in severe misalignment issues across query and support videos. In this paper, we arrest this problem from two distinct aspects -- action duration misalignment and action evolution misalignment. We address them sequentially through a Two-stage Action Alignment Network (TA2N). The first stage locates the action by learning a temporal affine transform, which warps each video feature to its action duration while dismissing the action-irrelevant feature (e.g. background). Next, the second stage coordinates query feature to match the spatial-temporal action evolution of support by performing temporally rearrange and spatially offset prediction. Extensive experiments on benchmark datasets show the potential of the proposed method in achieving state-of-the-art performance for few-shot action recognition.

----

## [157] Best-Buddy GANs for Highly Detailed Image Super-resolution

**Authors**: *Wenbo Li, Kun Zhou, Lu Qi, Liying Lu, Jiangbo Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20030](https://doi.org/10.1609/aaai.v36i2.20030)

**Abstract**:

We consider the single image super-resolution (SISR) problem, where a high-resolution (HR) image is generated based on a low-resolution (LR) input. Recently, generative adversarial networks (GANs) become popular to hallucinate details. Most methods along this line rely on a predefined single-LR-single-HR mapping, which is not flexible enough for the ill-posed SISR task. Also, GAN-generated fake details may often undermine the realism of the whole image. We address these issues by proposing best-buddy GANs (Beby-GAN) for rich-detail SISR. Relaxing the rigid one-to-one constraint, we allow the estimated patches to dynamically seek trustworthy surrogates of supervision during training, which is beneficial to producing more reasonable details. Besides, we propose a region-aware adversarial learning strategy that directs our model to focus on generating details for textured areas adaptively. Extensive experiments justify the effectiveness of our method. An ultra-high-resolution 4K dataset is also constructed to facilitate future super-resolution research.

----

## [158] SCAN: Cross Domain Object Detection with Semantic Conditioned Adaptation

**Authors**: *Wuyang Li, Xinyu Liu, Xiwen Yao, Yixuan Yuan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20031](https://doi.org/10.1609/aaai.v36i2.20031)

**Abstract**:

The domain gap severely limits the transferability and scalability of object detectors trained in a specific domain when applied to a novel one. Most existing works bridge the domain gap by minimizing the domain discrepancy in the category space and aligning category-agnostic global features. Though great success, these methods model domain discrepancy with prototypes within a batch, yielding a biased estimation of domain-level distribution. Besides, the category-agnostic alignment leads to the disagreement of class-specific distributions in the two domains, further causing inevitable classification errors. To overcome these two challenges, we propose a novel Semantic Conditioned AdaptatioN (SCAN) framework such that well-modeled unbiased semantics can support semantic conditioned adaptation for precise domain adaptive object detection. Specifically, class-specific semantics crossing different images in the source domain are graphically aggregated as the input to learn an unbiased semantic paradigm incrementally. The paradigm is then sent to a lightweight manifestation module to obtain conditional kernels to serve as the role of extracting semantics from the target domain for better adaptation. Subsequently, conditional kernels are integrated into global alignment to support the class-specific adaptation in a well-designed Conditional Kernel guided Alignment (CKA) module. Meanwhile, rich knowledge of the unbiased paradigm is transferred to the target domain with a novel Graph-based Semantic Transfer (GST) mechanism, yielding the adaptation in the category-based feature space. Comprehensive experiments conducted on three adaptation benchmarks demonstrate that SCAN outperforms existing works by a large margin.

----

## [159] Hybrid Instance-Aware Temporal Fusion for Online Video Instance Segmentation

**Authors**: *Xiang Li, Jinglu Wang, Xiao Li, Yan Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20032](https://doi.org/10.1609/aaai.v36i2.20032)

**Abstract**:

Recently, transformer-based image segmentation methods have achieved notable success against previous solutions. While for video domains, how to effectively model temporal context with the attention of object instances across frames remains an open problem. In this paper, we propose an online video instance segmentation framework with a novel instance-aware temporal fusion method. We first leverage the representation, \ie, a latent code in the global context (instance code) and CNN feature maps to represent instance- and pixel-level features. Based on this representation, we introduce a cropping-free temporal fusion approach to model the temporal consistency between video frames. Specifically, we encode global instance-specific information in the instance code and build up inter-frame contextual fusion with hybrid attentions between the instance codes and CNN feature maps. Inter-frame consistency between the instance codes is further enforced with order constraints. By leveraging the learned hybrid temporal consistency, we are able to directly retrieve and maintain instance identities across frames, eliminating the complicated frame-wise instance matching in prior methods. Extensive experiments have been conducted on popular VIS datasets, i.e. Youtube-VIS-19/21. Our model achieves the best performance among all online VIS methods. Notably, our model also eclipses all offline methods when using the ResNet-50 backbone.

----

## [160] Close the Loop: A Unified Bottom-Up and Top-Down Paradigm for Joint Image Deraining and Segmentation

**Authors**: *Yi Li, Yi Chang, Changfeng Yu, Luxin Yan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20033](https://doi.org/10.1609/aaai.v36i2.20033)

**Abstract**:

In this work, we focus on a very practical problem: image segmentation under rain conditions. Image deraining is a classic low-level restoration task, while image segmentation is a typical high-level understanding task. Most of the existing methods intuitively employ the bottom-up paradigm by taking deraining as a preprocessing step for subsequent segmentation. However, our statistical analysis indicates that not only deraining would benefit segmentation (bottom-up), but also segmentation would further improve deraining performance (top-down) in turn. This motivates us to solve the rainy image segmentation task within a novel top-down and bottom-up unified paradigm, in which two sub-tasks are alternatively performed and collaborated with each other. Specifically, the bottom-up procedure yields both clearer images and rain-robust features from both image and feature domains, so as to ease the segmentation ambiguity caused by rain streaks. The top-down procedure adopts semantics to adaptively guide the restoration for different contents via a novel multi-path semantic attentive module (SAM). Thus the deraining and segmentation could boost the performance of each other cooperatively and progressively. Extensive experiments and ablations demonstrate that the proposed method outperforms the state-of-the-art on rainy image segmentation.

----

## [161] Uncertainty Estimation via Response Scaling for Pseudo-Mask Noise Mitigation in Weakly-Supervised Semantic Segmentation

**Authors**: *Yi Li, Yiqun Duan, Zhanghui Kuang, Yimin Chen, Wayne Zhang, Xiaomeng Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20034](https://doi.org/10.1609/aaai.v36i2.20034)

**Abstract**:

Weakly-Supervised Semantic Segmentation (WSSS) segments objects without heavy burden of dense annotation. While as a price, generated pseudo-masks exist obvious noisy pixels, which result in sub-optimal segmentation models trained over these pseudo-masks. But rare studies notice or work on this problem, even these noisy pixels are inevitable after their improvements on pseudo-mask. So we try to improve WSSS in the aspect of noise mitigation. And we observe that many noisy pixels are of high confidences, especially when the response range is too wide or narrow, presenting an uncertain status. Thus, in this paper, we simulate noisy variations of response by scaling the prediction map in multiple times for uncertainty estimation. The uncertainty is then used to weight the segmentation loss to mitigate noisy supervision signals. We call this method URN, abbreviated from Uncertainty estimation via Response scaling for Noise mitigation. Experiments validate the benefits of URN, and our method achieves state-of-the-art results at 71.2% and 41.5% on PASCAL VOC 2012 and MS COCO 2014 respectively, without extra models like saliency detection. Code is available at https://github.com/XMed-Lab/URN.

----

## [162] Multi-Modal Perception Attention Network with Self-Supervised Learning for Audio-Visual Speaker Tracking

**Authors**: *Yidi Li, Hong Liu, Hao Tang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20035](https://doi.org/10.1609/aaai.v36i2.20035)

**Abstract**:

Multi-modal fusion is proven to be an effective method to improve the accuracy and robustness of speaker tracking, especially in complex scenarios. However, how to combine the heterogeneous information and exploit the complementarity of multi-modal signals remains a challenging issue. In this paper, we propose a novel Multi-modal Perception Tracker (MPT) for speaker tracking using both audio and visual modalities. Specifically, a novel acoustic map based on spatial-temporal Global Coherence Field (stGCF) is first constructed for heterogeneous signal fusion, which employs a camera model to map audio cues to the localization space consistent with the visual cues. Then a multi-modal perception attention network is introduced to derive the perception weights that measure the reliability and effectiveness of intermittent audio and video streams disturbed by noise. Moreover, a unique cross-modal self-supervised learning method is presented to model the confidence of audio and visual observations by leveraging the complementarity and consistency between different modalities. Experimental results show that the proposed MPT achieves 98.6% and 78.3% tracking accuracy on the standard and occluded datasets, respectively, which demonstrates its robustness under adverse conditions and outperforms the current state-of-the-art methods.

----

## [163] Defending against Model Stealing via Verifying Embedded External Features

**Authors**: *Yiming Li, Linghui Zhu, Xiaojun Jia, Yong Jiang, Shu-Tao Xia, Xiaochun Cao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20036](https://doi.org/10.1609/aaai.v36i2.20036)

**Abstract**:

Obtaining a well-trained model involves expensive data collection and training procedures, therefore the model is a valuable intellectual property. Recent studies revealed that adversaries can `steal' deployed models even when they have no training samples and can not get access to the model parameters or structures. Currently, there were some defense methods to alleviate this threat, mostly by increasing the cost of model stealing. In this paper, we explore the defense from another angle by verifying whether a suspicious model contains the knowledge of defender-specified external features. Specifically, we embed the external features by tempering a few training samples with style transfer. We then train a meta-classifier to determine whether a model is stolen from the victim. This approach is inspired by the understanding that the stolen models should contain the knowledge of features learned by the victim model. We examine our method on both CIFAR-10 and ImageNet datasets. Experimental results demonstrate that our method is effective in detecting different types of model stealing simultaneously, even if the stolen model is obtained via a multi-stage stealing process. The codes for reproducing main results are available at Github (https://github.com/zlh-thu/StealingVerification).

----

## [164] Towards an Effective Orthogonal Dictionary Convolution Strategy

**Authors**: *Yishi Li, Kunran Xu, Rui Lai, Lin Gu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20037](https://doi.org/10.1609/aaai.v36i2.20037)

**Abstract**:

Orthogonality regularization has proven effective in improving the precision, convergence speed and the training stability of CNNs. Here, we propose a novel Orthogonal Dictionary Convolution Strategy (ODCS) on CNNs to improve orthogonality effect by optimizing the network architecture and changing the regularized object. Specifically, we remove the nonlinear layer in typical convolution block “Conv(BN) + Nonlinear + Pointwise Conv(BN)”, and only impose orthogonal regularization on the front Conv. The structure, “Conv(BN) + Pointwise Conv(BN)”, is then equivalent to a pair of dictionary and encoding, defined in sparse dictionary learning. Thanks to the exact and efficient representation of signal with dictionaries in low-dimensional projections, our strategy could reduce the superfluous information in dictionary Conv kernels. Meanwhile, the proposed strategy relieves the too strict orthogonality regularization in training, which makes hyper-parameters tuning of model to be more flexible. In addition, our ODCS can modify the state-of-the-art models easily without any extra consumption in inference phase. We evaluate it on a variety of CNNs in small-scale (CIFAR), large-scale (ImageNet) and fine-grained (CUB-200-2011) image classification tasks, respectively. The experimental results show that our method achieve a stable and superior improvement.

----

## [165] ELMA: Energy-Based Learning for Multi-Agent Activity Forecasting

**Authors**: *Yu-Ke Li, Pin Wang, Lixiong Chen, Zheng Wang, Ching-Yao Chan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20038](https://doi.org/10.1609/aaai.v36i2.20038)

**Abstract**:

This paper describes an energy-based learning method that predicts the activities of multiple agents simultaneously. It aims to forecast both upcoming actions and paths of all agents in a scene based on their past activities, which can be jointly formulated by a probabilistic model over time. Learning this model is challenging because: 1) it has a large number of time-dependent variables that must scale with the forecast horizon and the number of agents; 2) distribution functions have to contain multiple modes in order to capture the spatio-temporal complexities of each agent's activities. To address these challenges, we put forth a novel Energy-based Learning approach for Multi-Agent activity forecasting (ELMA) to estimate this complex model via maximum log-likelihood estimation. Specifically, by sampling from a sequence of factorized marginalized multi-model distributions, ELMA generates most possible future actions efficiently. Moreover, by graph-based representations, ELMA also explicitly resolves the spatio-temporal dependencies of all agents' activities in a single pass. Our experiments on two large-scale datasets prove that ELMA outperforms recent leading studies by an obvious margin.

----

## [166] Equal Bits: Enforcing Equally Distributed Binary Network Weights

**Authors**: *Yunqiang Li, Silvia-Laura Pintea, Jan C. van Gemert*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20039](https://doi.org/10.1609/aaai.v36i2.20039)

**Abstract**:

Binary networks are extremely efficient as they use only two symbols to define the network: {+1, −1}. One can make the prior distribution of these symbols a design choice. The recent IR-Net of Qin et al. argues that imposing a Bernoulli distribution with equal priors (equal bit ratios) over the binary weights leads to maximum entropy and thus minimizes information loss. However, prior work cannot precisely control the binary weight distribution during training, and therefore cannot guarantee maximum entropy. Here, we show that quantizing using optimal transport can guarantee any bit ratio, including equal ratios. We investigate experimentally that equal bit ratios are indeed preferable and show that our method leads to optimization benefits. We show that our quantization method is effective when compared to state-of-the-art binarization methods, even when using binary weight pruning. Our code is available at https://github.com/liyunqianggyn/Equal-Bits-BNN.

----

## [167] SimIPU: Simple 2D Image and 3D Point Cloud Unsupervised Pre-training for Spatial-Aware Visual Representations

**Authors**: *Zhenyu Li, Zehui Chen, Ang Li, Liangji Fang, Qinhong Jiang, Xianming Liu, Junjun Jiang, Bolei Zhou, Hang Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20040](https://doi.org/10.1609/aaai.v36i2.20040)

**Abstract**:

Pre-training has become a standard paradigm in many computer vision tasks. However, most of the methods are generally designed on the RGB image domain. Due to the discrepancy between the two-dimensional image plane and the three-dimensional space, such pre-trained models fail to perceive spatial information and serve as sub-optimal solutions for 3D-related tasks. To bridge this gap, we aim to learn a spatial-aware visual representation that can describe the three-dimensional space and is more suitable and effective for these tasks. To leverage point clouds, which are much more superior in providing spatial information compared to images, we propose a simple yet effective 2D Image and 3D Point cloud Unsupervised pre-training strategy, called SimIPU. Specifically, we develop a multi-modal contrastive learning framework that consists of an intra-modal spatial perception module to learn a spatial-aware representation from point clouds and an inter-modal feature interaction module to transfer the capability of perceiving spatial information from the point cloud encoder to the image encoder, respectively. Positive pairs for contrastive losses are established by the matching algorithm and the projection matrix. The whole framework is trained in an unsupervised end-to-end fashion. To the best of our knowledge, this is the first study to explore contrastive learning pre-training strategies for outdoor multi-modal datasets, containing paired camera images and LIDAR point clouds.

----

## [168] Improving Human-Object Interaction Detection via Phrase Learning and Label Composition

**Authors**: *Zhimin Li, Cheng Zou, Yu Zhao, Boxun Li, Sheng Zhong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20041](https://doi.org/10.1609/aaai.v36i2.20041)

**Abstract**:

Human-Object Interaction (HOI) detection is a fundamental task in high-level human-centric scene understanding. We propose PhraseHOI, containing a HOI branch and a novel phrase branch, to leverage language prior and improve relation expression. Specifically, the phrase branch is supervised by semantic embeddings, whose ground truths are automatically converted from the original HOI annotations without extra human efforts. Meanwhile, a novel label composition method is proposed to deal with the long-tailed problem in HOI, which composites novel phrase labels by semantic neighbors. Further, to optimize the phrase branch, a loss composed of a distilling loss and a balanced triplet loss is proposed. Extensive experiments are conducted to prove the effectiveness of the proposed PhraseHOI, which achieves significant improvement over the baseline and surpasses previous state-of-the-art methods on Full and NonRare on the challenging HICO-DET benchmark.

----

## [169] Rethinking the Optimization of Average Precision: Only Penalizing Negative Instances before Positive Ones Is Enough

**Authors**: *Zhuo Li, Weiqing Min, Jiajun Song, Yaohui Zhu, Liping Kang, Xiaoming Wei, Xiaolin Wei, Shuqiang Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20042](https://doi.org/10.1609/aaai.v36i2.20042)

**Abstract**:

Optimising the approximation of Average Precision (AP) has been widely studied for image retrieval. Limited by the definition of AP, such methods consider both negative and positive instances ranking before each positive instance. However, we claim that only penalizing negative instances before positive ones is enough, because the loss only comes from these negative instances. To this end, we propose a novel loss, namely Penalizing Negative instances before Positive ones (PNP),  which can directly minimize the number of negative instances before each positive one. In addition, AP-based methods adopt a fixed and sub-optimal gradient assignment strategy. Therefore, we systematically investigate different gradient assignment solutions via constructing derivative functions of the loss, resulting in PNP-I with increasing derivative functions and PNP-D with decreasing ones. PNP-I focuses more on the hard positive instances by assigning larger gradients to them and tries to make all relevant instances closer. In contrast,  PNP-D pays less attention to such instances and slowly corrects them. For most real-world data, one class usually contains several local clusters. PNP-I blindly gathers these clusters while PNP-D keeps them as they were. Therefore, PNP-D is more superior. Experiments on three standard retrieval datasets show consistent results with the above analysis. Extensive evaluations demonstrate that PNP-D achieves the state-of-the-art performance. Code is available at https://github.com/interestingzhuo/PNPloss

----

## [170] Reliability Exploration with Self-Ensemble Learning for Domain Adaptive Person Re-identification

**Authors**: *Zongyi Li, Yuxuan Shi, Hefei Ling, Jiazhong Chen, Qian Wang, Fengfan Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20043](https://doi.org/10.1609/aaai.v36i2.20043)

**Abstract**:

Person re-identifcation (Re-ID) based on unsupervised domain adaptation (UDA) aims to transfer the pre-trained model from one labeled source domain to an unlabeled target domain. Existing methods tackle this problem by using clustering methods to generate pseudo labels. However, pseudo labels produced by these techniques may be unstable and noisy, substantially deteriorating models’ performance. In this paper, we propose a Reliability Exploration with Self-ensemble Learning (RESL) framework for domain adaptive person ReID. First, to increase the feature diversity, multiple branches are presented to extract features from different data augmentations. Taking the temporally average model as a mean teacher model, online label refning is conducted by using its dynamic ensemble predictions from different branches as soft labels. Second, to combat the adverse effects of unreliable samples in clusters, sample reliability is estimated by evaluating the consistency of different clusters’ results, followed by selecting reliable instances for training and re-weighting sample contribution within Re-ID losses. A contrastive loss is also utilized with cluster-level memory features which are updated by the mean feature. The experiments demonstrate that our method can signifcantly surpass the state-of-the-art performance on the unsupervised domain adaptive person ReID.

----

## [171] Deconfounding Physical Dynamics with Global Causal Relation and Confounder Transmission for Counterfactual Prediction

**Authors**: *Zongzhao Li, Xiangyu Zhu, Zhen Lei, Zhaoxiang Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20044](https://doi.org/10.1609/aaai.v36i2.20044)

**Abstract**:

Discovering the underneath causal relations is the fundamental ability for reasoning about the surrounding environment and predicting the future states in the physical world. Counterfactual prediction from visual input, which requires simulating future states based on unrealized situations in the past, is a vital component in causal relation tasks. In this paper, we work on the confounders that have effect on the physical dynamics, including masses, friction coefficients, etc., to bridge relations between the intervened variable and the affected variable whose future state may be altered. We propose a neural network framework combining Global Causal Relation Attention (GCRA) and Confounder Transmission Structure (CTS). The GCRA looks for the latent causal relations between different variables and estimates the confounders by capturing both spatial and temporal information. The CTS integrates and transmits the learnt confounders in a residual way, so that the estimated confounders can be encoded into the network as a constraint for object positions when performing counterfactual prediction. Without any access to ground truth information about confounders, our model outperforms the state-of-the-art method on various benchmarks by fully utilizing the constraints of confounders. Extensive experiments demonstrate that our model can generalize to unseen environments and maintain good performance.

----

## [172] One More Check: Making "Fake Background" Be Tracked Again

**Authors**: *Chao Liang, Zhipeng Zhang, Xue Zhou, Bing Li, Weiming Hu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20045](https://doi.org/10.1609/aaai.v36i2.20045)

**Abstract**:

The one-shot multi-object tracking, which integrates object detection and ID embedding extraction into a unified network, has achieved groundbreaking results in recent years. However, current one-shot trackers solely rely on single-frame detections to predict candidate bounding boxes, which may be unreliable when facing disastrous visual degradation, e.g., motion blur, occlusions. Once a target bounding box is mistakenly classified as background by the detector, the temporal consistency of its corresponding tracklet will be no longer maintained. In this paper, we set out to restore the bounding boxes misclassified as ``fake background'' by proposing a re-check network. The re-check network innovatively expands the role of ID embedding from data association to motion forecasting by effectively propagating previous tracklets to the current frame with a small overhead. Note that the propagation results are yielded by an independent and efficient embedding search, preventing the model from over-relying on detection results. Eventually, it helps to reload the ``fake background'' and repair the broken tracklets. Building on a strong baseline CSTrack, we construct a new one-shot tracker and achieve favorable gains by 70.7 ➡ 76.4, 70.6 ➡ 76.3 MOTA on MOT16 and MOT17, respectively. It also reaches a new state-of-the-art MOTA and IDF1 performance. Code is released at https://github.com/JudasDie/SOTS.

----

## [173] Semantically Contrastive Learning for Low-Light Image Enhancement

**Authors**: *Dong Liang, Ling Li, Mingqiang Wei, Shuo Yang, Liyan Zhang, Wenhan Yang, Yun Du, Huiyu Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20046](https://doi.org/10.1609/aaai.v36i2.20046)

**Abstract**:

Low-light image enhancement (LLE) remains challenging due to the unfavorable prevailing low-contrast and weak-visibility problems of single RGB images. In this paper, we respond to the intriguing learning-related question -- if leveraging both accessible unpaired over/underexposed images and high-level semantic guidance, can improve the performance of cutting-edge LLE models? Here, we propose an effective semantically contrastive learning paradigm for LLE (namely SCL-LLE). Beyond the existing LLE wisdom, it casts the image enhancement task as multi-task joint learning, where LLE is converted into three constraints of contrastive learning, semantic brightness consistency, and feature preservation for simultaneously ensuring the exposure, texture, and color consistency. SCL-LLE allows the LLE model to learn from unpaired positives (normal-light)/negatives (over/underexposed), and enables it to interact with the scene semantics to regularize the image enhancement network, yet the interaction of high-level semantic knowledge and the low-level signal prior is seldom investigated in previous methods. Training on readily available open data, extensive experiments demonstrate that our method surpasses the state-of-the-arts LLE models over six independent cross-scenes datasets. Moreover, SCL-LLE's potential to benefit the downstream semantic segmentation under extremely dark conditions is discussed. Source Code: https://github.com/LingLIx/SCL-LLE.

----

## [174] Self-Supervised Spatiotemporal Representation Learning by Exploiting Video Continuity

**Authors**: *Hanwen Liang, Niamul Quader, Zhixiang Chi, Lizhe Chen, Peng Dai, Juwei Lu, Yang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20047](https://doi.org/10.1609/aaai.v36i2.20047)

**Abstract**:

Recent self-supervised video representation learning methods have found significant success by exploring essential properties of videos, e.g. speed, temporal order, etc.
This work exploits an essential yet under-explored property of videos, the \textit{video continuity}, to obtain supervision signals for self-supervised representation learning.
Specifically, we formulate three novel continuity-related pretext tasks, i.e. continuity justification, discontinuity localization, and missing section approximation, that jointly supervise a shared backbone for video representation learning. 
This self-supervision approach, termed as Continuity Perception Network (CPNet), solves the three tasks altogether and encourages the backbone network to learn local and long-ranged motion and context representations. It outperforms prior arts on multiple downstream tasks, such as action recognition, video retrieval, and action localization.
Additionally, the video continuity can be complementary to other coarse-grained video properties for representation learning, and integrating the proposed pretext task to prior arts can yield much performance gains.

----

## [175] Inharmonious Region Localization by Magnifying Domain Discrepancy

**Authors**: *Jing Liang, Li Niu, Penghao Wu, Fengjun Guo, Teng Long*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20048](https://doi.org/10.1609/aaai.v36i2.20048)

**Abstract**:

Inharmonious region localization aims to localize the region in a synthetic image which is incompatible with surrounding background. The inharmony issue is mainly attributed to the color and illumination inconsistency produced by image editing techniques. In this work, we tend to transform the input image to another color space to magnify the domain discrepancy between inharmonious region and background, so that the model can identify the inharmonious region more easily. To this end, we present a novel framework consisting of a color mapping module and an inharmonious region localization network, in which the former is equipped with a novel domain discrepancy magnification loss and the latter could be an arbitrary localization network. Extensive experiments on image harmonization dataset show the superiority of our designed framework.

----

## [176] Distribution Aware VoteNet for 3D Object Detection

**Authors**: *Junxiong Liang, Pei An, Jie Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20049](https://doi.org/10.1609/aaai.v36i2.20049)

**Abstract**:

Occlusion is common in the actual 3D scenes, causing the boundary ambiguity of the targeted object. This uncertainty brings difficulty for labeling and learning. Current 3D detectors predict the bounding box directly, regarding it as Dirac delta distribution. However, it does not fully consider such ambiguity. To deal with it, distribution learning is used to efficiently represent the boundary ambiguity. In this paper, we revise the common regression method by predicting the distribution of the 3D box and then present a distribution-aware regression (DAR) module for box refinement and localization quality estimation. It contains scale adaptive (SA) encoder and joint localization quality estimator (JLQE). With the adaptive receptive field, SA encoder refines discriminative features for precise distribution learning. JLQE provides a reliable location score by further leveraging the distribution statistics, correlating with the localization quality of the targeted object. Combining DAR module and the baseline VoteNet, we propose a novel 3D detector called DAVNet. Extensive experiments on both ScanNet V2 and SUN RGB-D datasets demonstrate that the proposed DAVNet achieves significant improvement and outperforms state-of-the-art 3D detectors.

----

## [177] Contrastive Instruction-Trajectory Learning for Vision-Language Navigation

**Authors**: *Xiwen Liang, Fengda Zhu, Yi Zhu, Bingqian Lin, Bing Wang, Xiaodan Liang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20050](https://doi.org/10.1609/aaai.v36i2.20050)

**Abstract**:

The vision-language navigation (VLN) task requires an agent to reach a target with the guidance of natural language instruction. Previous works learn to navigate step-by-step following an instruction. However, these works may fail to discriminate the similarities and discrepancies across instruction-trajectory pairs and ignore the temporal continuity of sub-instructions. These problems hinder agents from learning distinctive vision-and-language representations, harming the robustness and generalizability of the navigation policy. In this paper, we propose a Contrastive Instruction-Trajectory Learning (CITL) framework that explores invariance across similar data samples and variance across different ones to learn distinctive representations for robust navigation. Specifically, we propose: (1) a coarse-grained contrastive learning objective to enhance vision-and-language representations by contrasting semantics of full trajectory observations and instructions, respectively; (2) a fine-grained contrastive learning objective to perceive instructions by leveraging the temporal information of the sub-instructions; (3) a pairwise sample-reweighting mechanism for contrastive learning to mine hard samples and hence mitigate the influence of data sampling bias in contrastive learning. Our CITL can be easily integrated with VLN backbones to form a new learning paradigm and achieve better generalizability in unseen environments. Extensive experiments show that the model with CITL surpasses the previous state-of-the-art methods on R2R, R4R, and RxR.

----

## [178] Interventional Multi-Instance Learning with Deconfounded Instance-Level Prediction

**Authors**: *Tiancheng Lin, Hongteng Xu, Canqian Yang, Yi Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20051](https://doi.org/10.1609/aaai.v36i2.20051)

**Abstract**:

When applying multi-instance learning (MIL) to make predictions for bags of instances, the prediction accuracy of an instance often depends on not only the instance itself but also its context in the corresponding bag. From the viewpoint of causal inference, such bag contextual prior works as a confounder and may result in model robustness and interpretability issues. Focusing on this problem, we propose a novel interventional multi-instance learning (IMIL) framework to achieve deconfounded instance-level prediction. Unlike traditional likelihood-based strategies, we design an Expectation-Maximization (EM) algorithm based on causal intervention, providing a robust instance selection in the training phase and suppressing the bias caused by the bag contextual prior. Experiments on pathological image analysis demonstrate that our IMIL method substantially reduces false positives and outperforms state-of-the-art MIL methods.

----

## [179] A Causal Debiasing Framework for Unsupervised Salient Object Detection

**Authors**: *Xiangru Lin, Ziyi Wu, Guanqi Chen, Guanbin Li, Yizhou Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20052](https://doi.org/10.1609/aaai.v36i2.20052)

**Abstract**:

Unsupervised Salient Object Detection (USOD) is a promising yet challenging task that aims to learn a salient object detection model without any ground-truth labels. Self-supervised learning based methods have achieved remarkable success recently and have become the dominant approach in USOD. However, we observed that two distribution biases of salient objects limit further performance improvement of the USOD methods, namely, contrast distribution bias and spatial distribution bias. Concretely, contrast distribution bias is essentially a confounder that makes images with similar high-level semantic contrast and/or low-level visual appearance contrast spuriously dependent, thus forming data-rich contrast clusters and leading the training process biased towards the data-rich contrast clusters in the data. Spatial distribution bias means that the position distribution of all salient objects in a dataset is concentrated on the center of the image plane, which could be harmful to off-center objects prediction. This paper proposes a causal based debiasing framework to disentangle the model from the impact of such biases. Specifically, we use causal intervention to perform de-confounded model training to minimize the contrast distribution bias and propose an image-level weighting strategy that softly weights each image's importance according to the spatial distribution bias map. Extensive experiments on 6 benchmark datasets show that our method significantly outperforms previous unsupervised state-of-the-art methods and even surpasses some of the supervised methods, demonstrating our debiasing framework's effectiveness.

----

## [180] A Causal Inference Look at Unsupervised Video Anomaly Detection

**Authors**: *Xiangru Lin, Yuyang Chen, Guanbin Li, Yizhou Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20053](https://doi.org/10.1609/aaai.v36i2.20053)

**Abstract**:

Unsupervised video anomaly detection, a task that requires no labeled normal/abnormal training data in any form, is challenging yet of great importance to both industrial applications and academic research. Existing methods typically follow an iterative pseudo label generation process. However, they lack a principled analysis of the impact of such pseudo label generation on training. Furthermore, the long-range temporal dependencies also has been overlooked, which is unreasonable since the definition of an abnormal event depends on the long-range temporal context. To this end, first, we propose a causal graph to analyze the confounding effect of the pseudo label generation process. Then, we introduce a simple yet effective causal inference based framework to disentangle the noisy pseudo label's impact. Finally, we perform counterfactual based model ensemble that blends long-range temporal context with local image context in inference to make final anomaly detection. Extensive experiments on six standard benchmark datasets show that our proposed method significantly outperforms previous state-of-the-art methods, demonstrating our framework's effectiveness.

----

## [181] Unpaired Multi-Domain Stain Transfer for Kidney Histopathological Images

**Authors**: *Yiyang Lin, Bowei Zeng, Yifeng Wang, Yang Chen, Zijie Fang, Jian Zhang, Xiangyang Ji, Haoqian Wang, Yongbing Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20054](https://doi.org/10.1609/aaai.v36i2.20054)

**Abstract**:

As an essential step in the pathological diagnosis, histochemical staining can show specific tissue structure information and, consequently, assist pathologists in making accurate diagnoses. Clinical kidney histopathological analyses usually employ more than one type of staining: H&E, MAS, PAS, PASM, etc. However, due to the interference of colors among multiple stains, it is not easy to perform multiple staining simultaneously on one biological tissue. To address this problem, we propose a network based on unpaired training data to virtually generate multiple types of staining from one staining. Our method can preserve the content of input images while transferring them to multiple target styles accurately. To efficiently control the direction of stain transfer, we propose a style guided normalization (SGN). Furthermore, a multiple style encoding (MSE) is devised to represent the relationship among different staining styles dynamically. An improved one-hot label is also proposed to enhance the generalization ability and extendibility of our method. Vast experiments have demonstrated that our model can achieve superior performance on a tiny dataset. The results exhibit not only good performance but also great visualization and interpretability. Especially, our method also achieves satisfactory results over cross-tissue, cross-staining as well as cross-task. We believe that our method will significantly influence clinical stain transfer and reduce the workload greatly for pathologists. Our code and Supplementary materials are available at https://github.com/linyiyang98/UMDST.

----

## [182] Dynamic Spatial Propagation Network for Depth Completion

**Authors**: *Yuankai Lin, Tao Cheng, Qi Zhong, Wending Zhou, Hua Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20055](https://doi.org/10.1609/aaai.v36i2.20055)

**Abstract**:

Image-guided depth completion aims to generate dense depth maps with sparse depth measurements and corresponding RGB images. Currently, spatial propagation networks (SPNs) are the most popular affinity-based methods in depth completion, but they still suffer from the representation limitation of the fixed affinity and the over smoothing during iterations. 
 Our solution is to estimate independent affinity matrices in each SPN iteration, but it is over-parameterized and heavy calculation.This paper introduces an efficient model that learns the affinity among neighboring pixels with an attention-based, dynamic approach. Specifically, the Dynamic Spatial Propagation Network (DySPN) we proposed makes use of a non-linear propagation model (NLPM). It decouples the neighborhood into parts regarding to different distances and recursively generates independent attention maps to refine these parts into adaptive affinity matrices. Furthermore, we adopt a diffusion suppression (DS) operation so that the model converges at an early stage to prevent over-smoothing of dense depth. Finally, in order to decrease the computational cost required, we also introduce three variations that reduce the amount of neighbors and attentions needed while still retaining similar accuracy. In practice, our method requires less iteration to match the performance of other SPNs and yields better results overall. DySPN outperforms other state-of-the-art (SoTA) methods on KITTI Depth Completion (DC) evaluation by the time of submission and is able to yield SoTA performance in NYU Depth v2 dataset as well.

----

## [183] Local Similarity Pattern and Cost Self-Reassembling for Deep Stereo Matching Networks

**Authors**: *Biyang Liu, Huimin Yu, Yangqi Long*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20056](https://doi.org/10.1609/aaai.v36i2.20056)

**Abstract**:

Although convolutional neural network based stereo matching architectures have made impressive achievements, there are still some limitations: 1) Convolutional Feature (CF) tends to capture appearance information, which is inadequate for accurate matching. 2) Due to the static filters, current convolution based disparity refinement modules often produce over-smooth results. In this paper, we present two schemes to address these issues, where some traditional wisdoms are integrated. Firstly, we introduce a pairwise feature for deep stereo matching networks, named LSP (Local Similarity Pattern). Through explicitly revealing the neighbor relationships, LSP contains rich structural information, which can be leveraged to aid CF for more discriminative feature description. Secondly, we design a dynamic self-reassembling refinement strategy and apply it to the cost distribution and the disparity map respectively. The former could be equipped with the unimodal distribution constraint to alleviate the over-smoothing problem, and the latter is more practical. The effectiveness of the proposed methods is demonstrated via incorporating them into two well-known basic architectures, GwcNet and GANet-deep. Experimental results on the SceneFlow and KITTI benchmarks show that our modules significantly improve the performance of the model. Code is available at https://github.com/SpadeLiu/Lac-GwcNet.

----

## [184] FedFR: Joint Optimization Federated Framework for Generic and Personalized Face Recognition

**Authors**: *Chih-Ting Liu, Chien-Yi Wang, Shao-Yi Chien, Shang-Hong Lai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20057](https://doi.org/10.1609/aaai.v36i2.20057)

**Abstract**:

Current state-of-the-art deep learning based face recognition (FR) models require a large number of face identities for central training. However, due to the growing privacy awareness, it is prohibited to access the face images on user devices to continually improve face recognition models. Federated Learning (FL) is a technique to address the privacy issue, which can collaboratively optimize the model without sharing the data between clients. In this work, we propose a FL based framework called FedFR to improve the generic face representation in a privacy-aware manner. Besides, the framework jointly optimizes personalized models for the corresponding clients via the proposed Decoupled Feature Customization module. The client-specific personalized model can serve the need of optimized face recognition experience for registered identities at the local device. To the best of our knowledge, we are the first to explore the personalized face recognition in FL setup. The proposed framework is validated to be superior to previous approaches on several generic and personalized face recognition benchmarks with diverse FL scenarios. The source codes and our proposed personalized FR benchmark under FL setup are available at https://github.com/jackie840129/FedFR.

----

## [185] Memory-Guided Semantic Learning Network for Temporal Sentence Grounding

**Authors**: *Daizong Liu, Xiaoye Qu, Xing Di, Yu Cheng, Zichuan Xu, Pan Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20058](https://doi.org/10.1609/aaai.v36i2.20058)

**Abstract**:

Temporal sentence grounding (TSG) is crucial and fundamental for video understanding. Although existing methods train well-designed deep networks with large amount of data, we find that they can easily forget the rarely appeared cases during training due to the off-balance data distribution, which influences the model generalization and leads to unsatisfactory performance. To tackle this issue, we propose a memory-augmented network, called Memory-Guided Semantic Learning Network (MGSL-Net), that learns and memorizes the rarely appeared content in TSG task. Specifically, our proposed model consists of three main parts: cross-modal interaction module, memory augmentation module, and heterogeneous attention module. We first align the given video-query pair by a cross-modal graph convolutional network, and then utilize memory module to record the cross-modal shared semantic features in the domain-specific persistent memory. During training, the memory slots are dynamically associated with both common and rare cases, alleviating the forgetting issue. In testing, the rare cases can thus be enhanced by retrieving the stored memories, leading to better generalization. At last, the heterogeneous attention module is utilized to integrate the enhanced multi-modal features in both video and query domains. Experimental results on three benchmarks show the superiority of our method on both effectiveness and efficiency, which substantially improves the accuracy not only on the entire dataset but also on the rare cases.

----

## [186] Exploring Motion and Appearance Information for Temporal Sentence Grounding

**Authors**: *Daizong Liu, Xiaoye Qu, Pan Zhou, Yang Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20059](https://doi.org/10.1609/aaai.v36i2.20059)

**Abstract**:

This paper addresses temporal sentence grounding. Previous works typically solve this task by learning frame-level video features and align them with the textual information. A major limitation of these works is that they fail to distinguish ambiguous video frames with subtle appearance differences due to frame-level feature extraction. Recently, a few methods adopt Faster R-CNN to extract detailed object features in each frame to differentiate the fine-grained appearance similarities. However, the object-level features extracted by Faster R-CNN suffer from missing motion analysis since the object detection model lacks temporal modeling. To solve this issue, we propose a novel Motion-Appearance Reasoning Network (MARN), which incorporates both motion-aware and appearance-aware object features to better reason object relations for modeling the activity among successive frames.
Specifically, we first introduce two individual video encoders to embed the video into corresponding motion-oriented and appearance-aspect object representations. Then, we develop separate motion and appearance branches to learn motion-guided and appearance-guided object relations, respectively. At last, both motion and appearance information from two branches are associated to generate more representative features for final grounding. Extensive experiments on two challenging datasets (Charades-STA and TACoS) show that our proposed MARN significantly outperforms previous state-of-the-art methods by a large margin.

----

## [187] Unsupervised Temporal Video Grounding with Deep Semantic Clustering

**Authors**: *Daizong Liu, Xiaoye Qu, Yinzhen Wang, Xing Di, Kai Zou, Yu Cheng, Zichuan Xu, Pan Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20060](https://doi.org/10.1609/aaai.v36i2.20060)

**Abstract**:

Temporal video grounding (TVG) aims to localize a target segment in a video according to a given sentence query. Though respectable works have made decent achievements in this task, they severely rely on abundant video-query paired data, which is expensive to collect in real-world scenarios. In this paper, we explore whether a video grounding model can be learned without any paired annotations. To the best of our knowledge, this paper is the first work trying to address TVG in an unsupervised setting. Considering there is no paired supervision, we propose a novel Deep Semantic Clustering Network (DSCNet) to leverage all semantic information from the whole query set to compose the possible activity in each video for grounding. Specifically, we first develop a language semantic mining module, which extracts implicit semantic features from the whole query set. Then, these language semantic features serve as the guidance to compose the activity in video via a video-based semantic aggregation module. Finally, we utilize a foreground attention branch to filter out the redundant background activities and refine the grounding results. To validate the effectiveness of our DSCNet, we conduct experiments on both ActivityNet Captions and Charades-STA datasets. The results demonstrate that our DSCNet achieves competitive performance, and even outperforms most weakly-supervised approaches.

----

## [188] SpikeConverter: An Efficient Conversion Framework Zipping the Gap between Artificial Neural Networks and Spiking Neural Networks

**Authors**: *Fangxin Liu, Wenbo Zhao, Yongbiao Chen, Zongwu Wang, Li Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20061](https://doi.org/10.1609/aaai.v36i2.20061)

**Abstract**:

Spiking Neural Networks (SNNs) have recently attracted enormous research interest since their event-driven and brain-inspired structure enables low-power computation. In image recognition tasks, the best results are achieved by SNN so far utilizing ANN-SNN conversion methods that replace activation functions in artificial neural networks~(ANNs) with integrate-and-fire neurons. Compared to source ANNs, converted SNNs usually suffer from accuracy loss and require a considerable number of time steps to achieve competitive accuracy. We find that the performance degradation of converted SNN stems from the fact that the information capacity of spike trains in transferred networks is smaller than that of activation values in source ANN, resulting in less information being passed during SNN inference.
 
 To better correlate ANN and SNN for better performance, we propose a conversion framework to mitigate the gap between the activation value of source ANN and the generated spike train of target SNN. The conversion framework originates from exploring an identical relation in the conversion and exploits temporal separation scheme and novel neuron model for the relation to hold. We demonstrate almost lossless ANN-SNN conversion using SpikeConverter for VGG-16, ResNet-20/34, and MobileNet-v2 SNNs on challenging datasets including CIFAR-10, CIFAR-100, and ImageNet. Our results also show that SpikeConverter achieves the abovementioned accuracy across different network architectures and datasets using 32X - 512X fewer inference time-steps than state-of-the-art ANN-SNN conversion methods.

----

## [189] Perceiving Stroke-Semantic Context: Hierarchical Contrastive Learning for Robust Scene Text Recognition

**Authors**: *Hao Liu, Bin Wang, Zhimin Bao, Mobai Xue, Sheng Kang, Deqiang Jiang, Yinsong Liu, Bo Ren*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20062](https://doi.org/10.1609/aaai.v36i2.20062)

**Abstract**:

We introduce Perceiving Stroke-Semantic Context (PerSec), a new approach to self-supervised representation learning tailored for Scene Text Recognition (STR) task. Considering scene text images carry both visual and semantic properties, we equip our PerSec with dual context perceivers which can contrast and learn latent representations from low-level stroke and high-level semantic contextual spaces simultaneously via hierarchical contrastive learning on unlabeled text image data. Experiments in un- and semi-supervised learning settings on STR benchmarks demonstrate our proposed framework can yield a more robust representation for both CTC-based and attention-based decoders than other contrastive learning methods. To fully investigate the potential of our method, we also collect a dataset of 100 million unlabeled text images, named UTI-100M, covering 5 scenes and 4 languages. By leveraging hundred-million-level unlabeled data, our PerSec shows significant performance improvement when fine-tuning the learned representation on the labeled data. Furthermore, we observe that the representation learned by PerSec presents great generalization, especially under few labeled data scenes.

----

## [190] AnchorFace: Boosting TAR@FAR for Practical Face Recognition

**Authors**: *Jiaheng Liu, Haoyu Qin, Yichao Wu, Ding Liang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20063](https://doi.org/10.1609/aaai.v36i2.20063)

**Abstract**:

Within the field of face recognition (FR), it is widely accepted that the key objective is to optimize the entire feature space in the training process and acquire robust feature representations. However, most real-world FR systems tend to operate at a pre-defined False Accept Rate (FAR), and the corresponding True Accept Rate (TAR) represents the performance of the FR systems, which indicates that the optimization on the pre-defined FAR is more meaningful and important in the practical evaluation process. In this paper, we call the predefined FAR as Anchor FAR, and we argue that the existing FR loss functions cannot guarantee the optimal TAR under the Anchor FAR, which impedes further improvements of FR systems. To this end, we propose AnchorFace to bridge the aforementioned gap between the training and practical evaluation process for FR. Given the Anchor FAR, AnchorFace can boost the performance of FR systems by directly optimizing
the non-differentiable FR evaluation metrics. Specifically, in AnchorFace, we first calculate the similarities of the positive and negative pairs based on both the features of the current batch and the stored features in the maintained online-updating set. Then, we generate the differentiable TAR loss and FAR loss using a soften strategy. Our AnchorFace can be readily integrated into most existing FR loss functions, and extensive experimental results on multiple benchmark datasets demonstrate the effectiveness of AnchorFace.

----

## [191] Memory-Based Jitter: Improving Visual Recognition on Long-Tailed Data with Diversity in Memory

**Authors**: *Jialun Liu, Wenhui Li, Yifan Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20064](https://doi.org/10.1609/aaai.v36i2.20064)

**Abstract**:

This paper considers deep visual recognition on long-tailed data. To make our method general, we tackle two applied scenarios, i.e. , deep classification and deep metric learning. Under the long-tailed data distribution, the most classes (i.e., tail classes) only occupy relatively few samples and are prone to lack of within-class diversity. A radical solution is to augment the tail classes with higher diversity. To this end, we introduce a simple and reliable method named Memory-based Jitter (MBJ). We observe that during training, the deep model constantly changes its parameters after every iteration, yielding the phenomenon of weight jitters. Consequentially, given a same image as the input, two historical editions of the model generate two different features in the deeply-embedded space, resulting in feature jitters. Using a memory bank, we collect these (model or feature) jitters across multiple training iterations and get the so-called Memory-based Jitter. The accumulated jitters enhance the within-class diversity for the tail classes and consequentially improves long-tailed visual recognition. With slight modifications, MBJ is applicable for two fundamental visual recognition tasks, i.e., deep image classification and deep metric learning (on long-tailed data). Extensive experiments on five long-tailed classification benchmarks and two deep metric learning benchmarks demonstrate significant improvement. Moreover, the achieved performance are on par with the state of the art on both tasks.

----

## [192] Debiased Batch Normalization via Gaussian Process for Generalizable Person Re-identification

**Authors**: *Jiawei Liu, Zhipeng Huang, Liang Li, Kecheng Zheng, Zheng-Jun Zha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20065](https://doi.org/10.1609/aaai.v36i2.20065)

**Abstract**:

Generalizable person re-identification aims to learn a model with only several labeled source domains that can perform well on unseen domains. Without access to the unseen domain, the feature statistics of the batch normalization (BN) layer learned from a limited number of source domains is doubtlessly biased for unseen domain. This would mislead the feature representation learning for unseen domain and deteriorate the generalizaiton ability of the model. In this paper, we propose a novel Debiased Batch Normalization via Gaussian Process approach (GDNorm) for generalizable person re-identification, which models the feature statistic estimation from BN layers as a dynamically self-refining Gaussian process to alleviate the bias to unseen domain for improving the generalization. Specifically, we establish a lightweight model with multiple set of domain-specific BN layers to capture the discriminability of individual source domain, and learn the corresponding parameters of the domain-specific BN layers. These parameters of different source domains are employed to deduce a Gaussian process. We randomly sample several paths from this Gaussian process served as the BN estimations of potential new domains outside of existing source domains, which can further optimize these learned parameters from source domains, and estimate more accurate Gaussian process by them in return, tending to real data distribution. Even without a large number of source domains, GDNorm can still provide debiased BN estimation by using the mean path of the Gaussian process, while maintaining low computational cost during testing. Extensive experiments demonstrate that our GDNorm effectively improves the generalization ability of the model on unseen domain.

----

## [193] Parallel and High-Fidelity Text-to-Lip Generation

**Authors**: *Jinglin Liu, Zhiying Zhu, Yi Ren, Wencan Huang, Baoxing Huai, Nicholas Jing Yuan, Zhou Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20066](https://doi.org/10.1609/aaai.v36i2.20066)

**Abstract**:

As a key component of talking face generation, lip movements generation determines the naturalness and coherence of the generated talking face video. Prior literature mainly focuses on speech-to-lip generation while there is a paucity in text-to-lip (T2L) generation. T2L is a challenging task and existing end-to-end works depend on the attention mechanism and autoregressive (AR) decoding manner. However, the AR decoding manner generates current lip frame conditioned on frames generated previously, which inherently hinders the inference speed, and also has a detrimental effect on the quality of generated lip frames due to error propagation. This encourages the research of parallel T2L generation. In this work, we propose a parallel decoding model for fast and high-fidelity text-to-lip generation (ParaLip). Specifically, we predict the duration of the encoded linguistic features and model the target lip frames conditioned on the encoded linguistic features with their duration in a non-autoregressive manner. Furthermore, we incorporate the structural similarity index loss and adversarial learning to improve perceptual quality of generated lip frames and alleviate the blurry prediction problem. Extensive experiments conducted on GRID and TCD-TIMIT datasets demonstrate the superiority of proposed methods.

----

## [194] SiamTrans: Zero-Shot Multi-Frame Image Restoration with Pre-trained Siamese Transformers

**Authors**: *Lin Liu, Shanxin Yuan, Jianzhuang Liu, Xin Guo, Youliang Yan, Qi Tian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20067](https://doi.org/10.1609/aaai.v36i2.20067)

**Abstract**:

We propose a novel zero-shot multi-frame image restoration method for removing unwanted obstruction elements (such as rains, snow, and moire patterns) that vary in successive frames. It has three stages: transformer pre-training, zero-shot restoration, and hard patch refinement. Using the pre-trained transformers, our model is able to tell the motion difference between the true image information and the obstructing elements. For zero-shot image restoration, we design a novel model, termed SiamTrans, which is constructed by Siamese transformers, encoders, and decoders. Each transformer has a temporal attention layer and several self-attention layers, to capture both temporal and spatial information of multiple frames. Only self-supervisedly pre-trained on the denoising task, SiamTrans is tested on three different low-level vision tasks (deraining, demoireing, and desnowing). Compared with related methods, SiamTrans achieves the best performances, even outperforming those with supervised learning.

----

## [195] Single-Domain Generalization in Medical Image Segmentation via Test-Time Adaptation from Shape Dictionary

**Authors**: *Quande Liu, Cheng Chen, Qi Dou, Pheng-Ann Heng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20068](https://doi.org/10.1609/aaai.v36i2.20068)

**Abstract**:

Domain generalization typically requires data from multiple source domains for model learning. However, such strong assumption may not always hold in practice, especially in medical field where the data sharing is highly concerned and sometimes prohibitive due to privacy issue. This paper studies the important yet challenging single domain generalization problem, in which a model is learned under the worst-case scenario with only one source domain to directly generalize to different unseen target domains. We present a novel approach to address this problem in medical image segmentation, which extracts and integrates the semantic shape prior information of segmentation that are invariant across domains and can be well-captured even from single domain data to facilitate segmentation under distribution shifts. Besides, a test-time adaptation strategy with dual-consistency regularization is further devised to promote dynamic incorporation of these shape priors under each unseen domain to improve model generalizability. Extensive experiments on two medical image segmentation tasks demonstrate the consistent improvements of our method across various unseen domains, as well as its superiority over state-of-the-art approaches in addressing domain generalization under the worst-case scenario.

----

## [196] Learning to Predict 3D Lane Shape and Camera Pose from a Single Image via Geometry Constraints

**Authors**: *Ruijin Liu, Dapeng Chen, Tie Liu, Zhiliang Xiong, Zejian Yuan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20069](https://doi.org/10.1609/aaai.v36i2.20069)

**Abstract**:

Detecting 3D lanes from the camera is a rising problem for autonomous vehicles. In this task, the correct camera pose is the key to generating accurate lanes, which can transform an image from perspective-view to the top-view. With this transformation, we can get rid of the perspective effects so that 3D lanes would look similar and can accurately be fitted by low-order polynomials. However, mainstream 3D lane detectors rely on perfect camera poses provided by other sensors, which is expensive and encounters multi-sensor calibration issues. To overcome this problem, we propose to predict 3D lanes by estimating camera pose from a single image with a two-stage framework. The first stage aims at the camera pose task from perspective-view images. To improve pose estimation, we introduce an auxiliary 3D lane task and geometry constraints to benefit from multi-task learning, which enhances consistencies between 3D and 2D, as well as compatibility in the above two tasks. The second stage targets the 3D lane task. It uses previously estimated pose to generate top-view images containing distance-invariant lane appearances for predicting accurate 3D lanes. Experiments demonstrate that, without ground truth camera pose, our method outperforms the state-of-the-art perfect-camera-pose-based methods and has the fewest parameters and computations. Codes are available at https://github.com/liuruijin17/CLGo.

----

## [197] OVIS: Open-Vocabulary Visual Instance Search via Visual-Semantic Aligned Representation Learning

**Authors**: *Sheng Liu, Kevin Lin, Lijuan Wang, Junsong Yuan, Zicheng Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20070](https://doi.org/10.1609/aaai.v36i2.20070)

**Abstract**:

We introduce the task of open-vocabulary visual instance search (OVIS). Given an arbitrary textual search query, Open-vocabulary Visual Instance Search (OVIS) aims to return a ranked list of visual instances, i.e., image patches, that satisfies the search intent from an image database. The term ``open vocabulary'' means that there are neither restrictions to the visual instance to be searched nor restrictions to the word that can be used to compose the textual search query. We propose to address such a search challenge via visual-semantic aligned representation learning (ViSA). ViSA leverages massive image-caption pairs as weak image-level (not instance-level) supervision to learn a rich cross-modal semantic space where the representations of visual instances (not images) and those of textual queries are aligned, thus allowing us to measure the similarities between any visual instance and an arbitrary textual query. To evaluate the performance of ViSA, we build two datasets named OVIS40 and OVIS1600 and also introduce a pipeline for error analysis. Through extensive experiments on the two datasets, we demonstrate ViSA's ability to search for visual instances in images not available during training given a wide range of textual queries including those composed of uncommon words. Experimental results show that ViSA achieves an mAP@50 of 27.8% on OVIS40 and achieves a recall@30 of 21.3% on OVIS1400 dataset under the most challenging settings.

----

## [198] Feature Generation and Hypothesis Verification for Reliable Face Anti-spoofing

**Authors**: *Shice Liu, Shitao Lu, Hongyi Xu, Jing Yang, Shouhong Ding, Lizhuang Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20071](https://doi.org/10.1609/aaai.v36i2.20071)

**Abstract**:

Although existing face anti-spoofing (FAS) methods achieve high accuracy in intra-domain experiments, their effects drop severely in cross-domain scenarios because of poor generalization. Recently, multifarious techniques have been explored, such as domain generalization and representation disentanglement. However, the improvement is still limited by two issues: 1) It is difficult to perfectly map all faces to a shared feature space. If faces from unknown domains are not mapped to the known region in the shared feature space, accidentally inaccurate predictions will be obtained. 2) It is hard to completely consider various spoof traces for disentanglement. In this paper, we propose a Feature Generation and Hypothesis Verification framework to alleviate the two issues. Above all, feature generation networks which generate hypotheses of real faces and known attacks are introduced for the first time in the FAS task. Subsequently, two hypothesis verification modules are applied to judge whether the input face comes from the real-face space and the real-face distribution respectively. Furthermore, some analyses of the relationship between our framework and Bayesian uncertainty estimation are given, which provides theoretical support for reliable defense in unknown domains. Experimental results show our framework achieves promising results and outperforms the state-of-the-art approaches on extensive public datasets.

----

## [199] Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions

**Authors**: *Wenyu Liu, Gaofeng Ren, Runsheng Yu, Shi Guo, Jianke Zhu, Lei Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20072](https://doi.org/10.1609/aaai.v36i2.20072)

**Abstract**:

Though deep learning-based object detection methods have achieved promising results on the conventional datasets, it is still challenging to locate objects from the low-quality images captured in adverse weather conditions. The existing methods either have difficulties in balancing the tasks of image enhancement and object detection, or often ignore the latent information beneficial for detection. To alleviate this problem, we propose a novel Image-Adaptive YOLO (IA-YOLO) framework, where each image can be adaptively enhanced for better detection performance. Specifically, a differentiable image processing (DIP) module is presented to take into account the adverse weather conditions for YOLO detector, whose parameters are predicted by a small convolutional neural network (CNN-PP). We learn CNN-PP and YOLOv3 jointly in an end-to-end fashion, which ensures that CNN-PP can learn an appropriate DIP to enhance the image for detection in a weakly supervised manner. Our proposed IA-YOLO approach can adaptively process images in both normal and adverse weather conditions. The experimental results are very encouraging, demonstrating the effectiveness of our proposed IA-YOLO method in both foggy and low-light scenarios. The source code can be found at https://github.com/wenyyu/Image-Adaptive-YOLO.

----



[Go to the next page](AAAI-2022-list02.md)

[Go to the catalog section](README.md)