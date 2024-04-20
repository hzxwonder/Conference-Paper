## [2000] NODEO: A Neural Ordinary Differential Equation Based Optimization Framework for Deformable Image Registration

**Authors**: *Yifan Wu, Tom Z. Jiahao, Jiancong Wang, Paul A. Yushkevich, M. Ani Hsieh, James C. Gee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02014](https://doi.org/10.1109/CVPR52688.2022.02014)

**Abstract**:

Deformable image registration (DIR), aiming to find spatial correspondence between images, is one of the most critical problems in the domain of medical image analysis. In this paper, we present a novel, generic, and accurate diffeomorphic image registration framework that utilizes neural ordinary differential equations (NODEs). We model each voxel as a moving particle and consider the set of all voxels in a 3D image as a high-dimensional dynamical system whose trajectory determines the targeted deformation field. Our method leverages deep neural networks for their expressive power in modeling dynamical systems, and simultaneously optimizes for a dynamical system between the image pairs and the corresponding transformation. Our formulation allows various constraints to be imposed along the transformation to maintain desired regularities. Our experiment results show that our method outperforms the benchmarks under various metrics. Additionally, we demonstrate the feasibility to expand our framework to register multiple image sets using a unified form of transformation, which could possibly serve a wider range of applications.

----

## [2001] SMPL-A: Modeling Person-Specific Deformable Anatomy

**Authors**: *Hengtao Guo, Benjamin Planche, Meng Zheng, Srikrishna Karanam, Terrence Chen, Ziyan Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02015](https://doi.org/10.1109/CVPR52688.2022.02015)

**Abstract**:

A variety of diagnostic and therapeutic protocols rely on locating in vivo target anatomical structures, which can be obtained from medical scans. However, organs move and deform as the patient changes his/her pose. In order to obtain accurate target location information, clinicians have to either conduct frequent intraoperative scans, resulting in higher exposition of patients to radiations, or adopt proxy procedures (e.g., creating and using custom molds to keep patients in the exact same pose during both preoperative organ scanning and subsequent treatment. Such custom proxy methods are typically sub-optimal, constraining the clinicians and costing precious time and money to the patients. To the best of our knowledge, this work is the first to present a learning-based approach to estimate the patient's internal organ deformation for arbitrary human poses in order to assist with radiotherapy and similar medical protocols. The underlying method first leverages medical scans to learn a patient-specific representation that potentially encodes the organ's shape and elastic properties. During inference, given the patient's current body pose information and the organ's representation extracted from previous medical scans, our method can estimate their current organ deformation to offer guidance to clinicians. We conduct experiments on a well-sized dataset which is augmented through real clinical data using finite element modeling. Our results suggest that pose-dependent organ deformation can be learned through a point cloud autoencoder conditioned on the parametric pose input. We hope that this work can be a starting point for future research towards closing the loop between human mesh recovery and anatomical reconstruction, with applications beyond the medical domain.

----

## [2002] DiRA: Discriminative, Restorative, and Adversarial Learning for Self-supervised Medical Image Analysis

**Authors**: *Fatemeh Haghighi, Mohammad Reza Hosseinzadeh Taher, Michael B. Gotway, Jianming Liang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02016](https://doi.org/10.1109/CVPR52688.2022.02016)

**Abstract**:

Discriminative learning, restorative learning, and adversarial learning have proven beneficial for self-supervised learning schemes in computer vision and medical imaging. Existing efforts, however, omit their synergistic effects on each other in a ternary setup, which, we envision, can sig-nificantly benefit deep semantic representation learning. To realize this vision, we have developed DiRA, thefirstframework that unites discriminative, restorative, and adversarial learning in a unified manner to collaboratively glean complementary visual information from unlabeled medical images for fine-grained semantic representation learning. Our extensive experiments demonstrate that DiRA (1) encourages collaborative learning among three learning ingredients, resulting in more generalizable representation across organs, diseases, and modalities; (2) outperforms fully supervised ImageNet models and increases robustness in small data regimes, reducing annotation cost across multiple medical imaging applications; (3) learns fine-grained semantic representation, facilitating accurate lesion localization with only image-level annotation; and (4) enhances state-of-the-art restorative approaches, revealing that DiRA is a general mechanism for united representation learning. All code and pretrained models are available at https://github.com/JLiangLab/DiRA.

----

## [2003] Affine Medical Image Registration with Coarse-to-Fine Vision Transformer

**Authors**: *Tony C. W. Mok, Albert C. S. Chung*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02017](https://doi.org/10.1109/CVPR52688.2022.02017)

**Abstract**:

Affine registration is indispensable in a comprehensive medical image registration pipeline. However, only a few studies focus on fast and robust affine registration algorithms. Most of these studies utilize convolutional neural networks (CNNs) to learn joint affine and non-parametric registration, while the standalone performance of the affine subnetwork is less explored. Moreover, existing CNN-based affine registration approaches focus either on the local mis-alignment or the global orientation and position of the input to predict the affine transformation matrix, which are sensitive to spatial initialization and exhibit limited generalizability apart from the training dataset. In this paper, we present a fast and robust learning-based algorithm, Coarse-to-Fine Vision Transformer (C2FViT), for 3D affine medical image registration. Our method naturally leverages the global connectivity and locality of the convolutional vision transformer and the multi-resolution strategy to learn the global affine registration. We evaluate our method on 3D brain atlas registration and template-matching normalization. Comprehensive results demonstrate that our method is superior to the existing CNNs-based affine registration methods in terms of registration accuracy, robustness and generalizability while preserving the runtime advantage of the learning-based methods. The source code is available at https://github.com/cwmok/C2FViT.

----

## [2004] Topology-Preserving Shape Reconstruction and Registration via Neural Diffeomorphic Flow

**Authors**: *Shanlin Sun, Kun Han, Deying Kong, Hao Tang, Xiangyi Yan, Xiaohui Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02018](https://doi.org/10.1109/CVPR52688.2022.02018)

**Abstract**:

Deep Implicit Functions (DIFs) represent 3D geometry with continuous signed distance functions learned through deep neural nets. Recently DIFs-based methods have been proposed to handle shape reconstruction and dense point correspondences simultaneously, capturing semantic relationships across shapes of the same class by learning a DIFs-modeled shape template. These methods provide great flexibility and accuracy in reconstructing 3D shapes and inferring correspondences. However, the point correspondences built from these methods do not intrinsically preserve the topology of the shapes, unlike mesh-based template matching methods. This limits their applications on 3D geometries where underlying topological structures exist and matter, such as anatomical structures in medical images. In this paper, we propose a new model called Neural Diffeomorphic Flow (NDF) to learn deep implicit shape templates, representing shapes as conditional diffeomorphic deformations of templates, intrinsically preserving shape topologies. The diffeomorphic deformation is realized by an autodecoder consisting of Neural Ordinary Differential Equation (NODE) blocks that progressively map shapes to implicit templates. We conduct extensive experiments on several medical image organ segmentation datasets to evaluate the effectiveness of NDF on reconstructing and aligning shapes. NDF achieves consistently state-of-the-art organ shape reconstruction and registration results in both accuracy and quality. The source code is publicly available at https://github.com/Siwensun/Neural_Diffeomorphic_Flow-NDF.

----

## [2005] Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization

**Authors**: *Ziqi Zhou, Lei Qi, Xin Yang, Dong Ni, Yinghuan Shi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02019](https://doi.org/10.1109/CVPR52688.2022.02019)

**Abstract**:

For medical image segmentation, imagine if a model was only trained using MR images in source domain, how about its performance to directly segment CT images in target domain? This setting, namely generalizable cross-modality segmentation, owning its clinical potential, is much more challenging than other related settings, e.g., domain adaptation. To achieve this goal, we in this paper propose a novel dual-normalization model by leveraging the augmented source-similar and source-dissimilar images during our generalizable segmentation. To be specific, given a single source domain, aiming to simulate the possible appearance change in unseen target domains, we first utilize a nonlinear transformation to augment source-similar and source-dissimilar images. Then, to sufficiently exploit these two types of augmentations, our proposed dualnormalization based model employs a shared backbone yet independent batch normalization layer for separate normalization. Afterward, we put forward a style-based selection scheme to automatically choose the appropriate path in the test stage. Extensive experiments on three publicly available datasets, i.e., BraTS, Cross-Modality Cardiac, and Abdominal Multi-Organ datasets, have demonstrated that our method outperforms other state-of-the-art domain generalization methods. Code is available at https://github.com/zzzqzhou/Dual-Normalization.

----

## [2006] Closing the Generalization Gap of Cross-silo Federated Medical Image Segmentation

**Authors**: *An Xu, Wenqi Li, Pengfei Guo, Dong Yang, Holger Roth, Ali Hatamizadeh, Can Zhao, Daguang Xu, Heng Huang, Ziyue Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02020](https://doi.org/10.1109/CVPR52688.2022.02020)

**Abstract**:

Cross-silo federated learning (FL) has attracted much attention in medical imaging analysis with deep learning in recent years as it can resolve the critical issues of insufficient data, data privacy, and training efficiency. However, there can be a generalization gap between the model trained from FL and the one from centralized training. This important issue comes from the non-iid data distribution of the local data in the participating clients and is well-known as client drift. In this work, we propose a novel training frame-work FedSM to avoid the client drift issue and successfully close the generalization gap compared with the centralized training for medical image segmentation tasks for the first time. We also propose a novel personalized FL objective formulation and a new method SoftPull to solve it in our proposed framework FedSM. We conduct rigorous theoretical analysis to guarantee its convergence for optimizing the non-convex smooth objective function. Real-world medical image segmentation experiments using deep FL validate the motivations and effectiveness of our proposed method.

----

## [2007] FIBA: Frequency-Injection based Backdoor Attack in Medical Image Analysis

**Authors**: *Yu Feng, Benteng Ma, Jing Zhang, Shanshan Zhao, Yong Xia, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02021](https://doi.org/10.1109/CVPR52688.2022.02021)

**Abstract**:

In recent years, the security of AI systems has drawn increasing research attention, especially in the medical imaging realm. To develop a secure medical image analysis (MIA) system, it is a must to study possible backdoor attacks (BAs), which can embed hidden malicious behaviors into the system. However, designing a unified BA method that can be applied to various MIA systems is challenging due to the diversity of imaging modalities (e.g., X-Ray, CT, and MRI) and analysis tasks (e.g., classification, detection, and segmentation). Most existing BA methods are designed to attack natural image classification models, which apply spatial triggers to training images and inevitably corrupt the semantics of poisoned pixels, leading to the failures of attacking dense prediction models. To address this issue, we propose a novel Frequency-Injection based Backdoor Attack method (FIBA) that is capable of delivering attacks in various MIA tasks. Specifically, FIBA leverages a trigger function in the frequency domain that can inject the low-frequency information of a trigger image into the poisoned image by linearly combining the spectral amplitude of both images. Since it preserves the semantics of the poisoned image pixels, FIBA can perform attacks on both classification and dense prediction models. Experiments on three benchmarks in MIA (i.e., ISIC-2019 [4] for skin lesion classification, KiTS-19 [17] for kidney tumor segmentation, and EAD-2019 [1] for endoscopic artifact detection), validate the effectiveness of FIBA and its superiority over stateof-the-art methods in attacking MIA models and bypassing backdoor defense. Source code will be available at code.

----

## [2008] Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning

**Authors**: *Soumen Basu, Mayank Gupta, Pratyaksha Rana, Pankaj Gupta, Chetan Arora*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02022](https://doi.org/10.1109/CVPR52688.2022.02022)

**Abstract**:

We explore the potential of CNN-based models for gall-bladder cancer (GBC) detection from ultrasound (USG) images as no prior study is known. USG is the most common diagnostic modality for GB diseases due to its low cost and accessibility. However, USG images are challenging to analyze due to low image quality, noise, and varying viewpoints due to the handheld nature of the sensor. Our exhaustive study of state-of-the-art (SOTA) image classification techniques for the problem reveals that they often fail to learn the salient GB region due to the presence of shadows in the USG images. SOTA object detection techniques also achieve low accuracy because of spurious textures due to noise or adjacent organs. We propose GBCNet to tackle the challenges in our problem. GBCNet first extracts the regions of interest (ROIs) by detecting the GB (and not the cancer), and then uses a new multi-scale, second-order pooling architecture specializing in classifying GBC. To effectively handle spurious textures, we propose a curriculum inspired by human visual acuity, which reduces the texture biases in GBCNet. Experimental results demonstrate that GBC-Net significantly outperforms SOTA CNN models, as well as the expert radiologists. Our technical innovations are generic to other USG image analysis tasks as well. Hence, as a validation, we also show the efficacy of GBCNet in detecting breast cancer from USG images. Project page with source code, trained models, and data is available at https://GBC-iitd.github.io/GBCnet.

----

## [2009] CellTypeGraph: A New Geometric Computer Vision Benchmark

**Authors**: *Lorenzo Cerrone, Athul Vijayan, Tejasvinee Mody, Kay Schneitz, Fred A. Hamprecht*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02023](https://doi.org/10.1109/CVPR52688.2022.02023)

**Abstract**:

Classifying all cells in an organ is a relevant and difficult problem from plant developmental biology. We here abstract the problem into a new benchmark for node classification in a geo-referenced graph. Solving it requires learning the spatial layout of the organ including symmetries. To allow the convenient testing of new geometrical learning methods, the benchmark of Arabidopsis thaliana ovules is made available as a PyTorch data loader, along with a large number of precomputed features. Finally, we benchmark eight recent graph neural network architectures, finding that DeeperGCN currently works best on this problem.

----

## [2010] ContIG: Self-supervised Multimodal Contrastive Learning for Medical Imaging with Genetics

**Authors**: *Aiham Taleb, Matthias Kirchler, Remo Monti, Christoph Lippert*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02024](https://doi.org/10.1109/CVPR52688.2022.02024)

**Abstract**:

High annotation costs are a substantial bottleneck in applying modern deep learning architectures to clinically relevant medical use cases, substantiating the need for novel algorithms to learn from unlabeled data. In this work, we propose ContIG, a self-supervised method that can learn from large datasets of unlabeled medical images and genetic data. Our approach aligns images and several genetic modalities in the feature space using a contrastive loss. We design our method to integrate multiple modalities of each individual person in the same model end-to-end, even when the available modalities vary across individuals. Our procedure outperforms state-of-the-art self-supervised methods on all evaluated downstream benchmark tasks. We also adapt gradient-based explainability algorithms to better understand the learned cross-modal associations between the images and genetic modalities. Finally, we perform genome-wide association studies on the features learned by our models, uncovering interesting relationships between images and genetic data.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Source code at: https://github.com/HealthML/ContIG

----

## [2011] FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos

**Authors**: *Yan Wang, Yixuan Sun, Yiwen Huang, Zhongying Liu, Shuyong Gao, Wei Zhang, Weifeng Ge, Wenqiang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02025](https://doi.org/10.1109/CVPR52688.2022.02025)

**Abstract**:

Current benchmarks for facial expression recognition (FER) mainly focus on static images, while there are limited datasets for FER in videos. It is still ambiguous to evaluate whether performances of existing methods remain satisfactory in real-world application-oriented scenes. For example, the “Happy” expression with high intensity in Talk-Show is more discriminating than the same expression with low intensity in Official-Event. To fill this gap, we build a large-scale multi-scene dataset, coined as FERV39k. We analyze the important ingredients of constructing such a novel dataset in three aspects: (1) multi-scene hierarchy and expression class, (2) generation of candidate video clips, (3) trusted manual labelling process. Based on these guidelines, we select 4 scenarios subdivided into 22 scenes, annotate 86k samples automatically obtained from 4k videos based on the well-designed workflow, and finally build 38,935 video clips labeled with 7 classic expressions. Experiment benchmarks on four kinds of baseline frame-works were also provided and further analysis on their performance across different scenes and some challenges for future research were given. Besides, we systematically investigate key components of DFER by ablation studies. The baseline framework and our project are available on https://github.com/wangyanckxx/FERV39k.

----

## [2012] Multi-Dimensional, Nuanced and Subjective - Measuring the Perception of Facial Expressions

**Authors**: *De'Aira Bryant, Siqi Deng, Nashlie Sephus, Wei Xia, Pietro Perona*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02026](https://doi.org/10.1109/CVPR52688.2022.02026)

**Abstract**:

Humans can perceive multiple expressions, each one with varying intensity, in the picture of a face. We propose a methodology for collecting and modeling multidimensional modulated expression annotations from human annotators. Our data reveals that the perception of some expressions can be quite different across observers; thus, our model is designed to represent ambiguity alongside intensity. An empirical exploration of how many dimensions are necessary to capture the perception of facial expression suggests six principal expression dimensions are sufficient. Using our method, we collected multidimensional modulated expression annotations for 1,000 images culled from the popular ExpW in-the-wild dataset. As a proof of principle of our improved measurement technique, we used these annotations to benchmark four public domain algorithms for automated facial expression prediction.

----

## [2013] DAD-3DHeads: A Large-scale Dense, Accurate and Diverse Dataset for 3D Head Alignment from a Single Image

**Authors**: *Tetiana Martyniuk, Orest Kupyn, Yana Kurlyak, Igor Krashenyi, Jiri Matas, Viktoriia Sharmanska*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02027](https://doi.org/10.1109/CVPR52688.2022.02027)

**Abstract**:

We present DAD-3DHeads, a dense and diverse large-scale dataset, and a robust model for 3D Dense Head Alignment in-the-wild. It contains annotations of over 3.5K land-marks that accurately represent 3D head shape compared to the ground-truth scans. The data-driven model, DAD-3DNet, trained on our dataset, learns shape, expression, and pose parameters, and performs 3D reconstruction of a FLAME mesh. The model also incorporates a landmark prediction branch to take advantage of rich supervision and co-training of multiple related tasks. Experimentally, DAD-3DNet outperforms or is comparable to the state-of-the-art models in (i) 3D Head Pose Estimation on AFLW2000-3D and BIWI, (ii) 3D Face Shape Reconstruction on NoW and Feng, and (iii) 3D Dense Head Alignment and 3D Land-marks Estimation on DAD-3DHeads dataset. Finally, diver-sity of DAD-3DHeads in camera angles, facial expressions, and occlusions enables a benchmark to study in-the-wild generalization and robustness to distribution shifts. The dataset webpage is https://p.farm/research/dad-3dheads.

----

## [2014] OakInk: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction

**Authors**: *Lixin Yang, Kailin Li, Xinyu Zhan, Fei Wu, Anran Xu, Liu Liu, Cewu Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02028](https://doi.org/10.1109/CVPR52688.2022.02028)

**Abstract**:

Learning how humans manipulate objects requires machines to acquire knowledge from two perspectives: one for understanding object affordances and the other for learning human's interactions based on the affordances. Even though these two knowledge bases are crucial, we find that current databases lack a comprehensive awareness of them. In this work, we propose a multi-modal and rich-annotated knowledge repository, OakInk, for visual and cognitive understanding of hand-object interactions. We start to collect 1,800 common household objects and annotate their affordances to construct the first knowledge base: Oak. Given the affordance, we record rich human interactions with 100 selected objects in Oak. Finally, we transfer the interactions on the 100 recorded objects to their virtual counterparts through a novel method: Tink. The recorded and transferred hand-object interactions constitute the second knowledge base: Ink. As a result, OakInk contains 50,000 distinct affordance-aware and intent-oriented hand-object interactions. We benchmark OakInk on pose estimation and grasp generation tasks. Moreover, we propose two practical applications of OakInk: intent-based interaction generation and handover generation. Our dataset and source code are publicly available at www.oakink.net.

----

## [2015] PoseTrack21: A Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking

**Authors**: *Andreas Doering, Di Chen, Shanshan Zhang, Bernt Schiele, Juergen Gall*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02029](https://doi.org/10.1109/CVPR52688.2022.02029)

**Abstract**:

Current research evaluates person search, multi-object tracking and multi-person pose estimation as separate tasks and on different datasets although these tasks are very akin to each other and comprise similar sub-tasks, e.g. person detection or appearance-based association of detected persons. Consequently, approaches on these respective tasks are eligible to complement each other. Therefore, we introduce PoseTrack21, a large-scale dataset for person search, multi-object tracking and multi-person pose tracking in real-world scenarios with a high diversity of poses. The dataset provides rich annotations like human pose annotations including annotations of joint occlusions, bounding box annotations even for small persons, and person-ids within and across video sequences. The dataset allows to evaluate multi-object tracking and multi-person pose tracking jointly with person re-identification or exploit structural knowledge of human poses to improve person search and tracking, particularly in the context of severe occlusions. With PoseTrack21, we want to encourage researchers to work on joint approaches that perform reasonably well on all three tasks.

----

## [2016] Learning Modal-Invariant and Temporal-Memory for Video-based Visible-Infrared Person Re-Identification

**Authors**: *Xinyu Lin, Jinxing Li, Zeyu Ma, Huafeng Li, Shuang Li, Kaixiong Xu, Guangming Lu, David Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02030](https://doi.org/10.1109/CVPR52688.2022.02030)

**Abstract**:

Thanks for the cross-modal retrieval techniques, visible-infrared (RGB-IR) person re-identification (Re-ID) is achieved by projecting them into a common space, allowing person Re-ID in 24-hour surveillance systems. However, with respect to the probe-to- gallery, almost all existing RGB-IR based cross-modal person Re-ID methods focus on image-to-image matching, while the video-to-video matching which contains much richer spatial- and temporal-information remains under-explored. In this paper, we primarily study the video-based cross-modal per-son Re-ID method. To achieve this task, a video-based RGB-IR dataset is constructed, in which 927 valid identities with 463,259 frames and 21,863 tracklets captured by 12 RGB/IR cameras are collected. Based on our constructed dataset, we prove that with the increase of frames in a tracklet, the performance does meet more enhancement, demonstrating the significance of video-to-video matching in RGB-IR person Re-ID. Additionally, a novel method is further proposed, which not only projects two modalities to a modal-invariant subspace, but also extracts the temporal-memory for motion-invariant. Thanks to these two strategies, much better results are achieved on our video-based cross-modal person Re-ID. The code and dataset are released at: https://github.com/VCM-project233/MITML.

----

## [2017] JRDB-Act: A Large-scale Dataset for Spatio-temporal Action, Social Group and Activity Detection

**Authors**: *Mahsa Ehsanpour, Fatemeh Sadat Saleh, Silvio Savarese, Ian D. Reid, Hamid Rezatofighi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02031](https://doi.org/10.1109/CVPR52688.2022.02031)

**Abstract**:

The availability of large-scale video action understanding datasets has facilitated advances in the interpretation of visual scenes containing people. However, learning to recognise human actions and their social interactions in an unconstrained real-world environment comprising numerous people, with potentially highly unbalanced and longtailed distributed action labels from a stream of sensory data captured from a mobile robot platform remains a significant challenge, not least owing to the lack of a reflective large-scale dataset. In this paper, we introduce JRDB-Act, as an extension of the existing JRDB, which is captured by a social mobile manipulator and reflects a real distribution of human daily-life actions in a university campus environment. JRDB-Act has been densely annotated with atomic actions, comprises over 2.8M action labels, constituting a large-scale spatio-temporal action detection dataset. Each human bounding box is labeled with one pose-based action label and multiple (optional) interaction-based action labels. Moreover JRDB-Act provides social group annotation, conducive to the task of grouping individuals based on their interactions in the scene to infer their social activities (common activities in each social group). Each annotated label in JRDB-Act is tagged with the annotators' confidence level which contributes to the development of reliable evaluation strategies. In order to demonstrate how one can effectively utilise such annotations, we develop an end-to-end trainable pipeline to learn and infer these tasks, i.e. individual action and social group detection. The data and the evaluation code will be publicly available at https://jrdb.erc.monash.edu/

----

## [2018] DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion

**Authors**: *Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, Ping Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02032](https://doi.org/10.1109/CVPR52688.2022.02032)

**Abstract**:

A typical pipeline for multi-object tracking (MOT) is to use a detector for object localization, and following re-identification (re-ID)for object association. This pipeline is partially motivated by recent progress in both object detection and re- ID, and partially motivated by biases in existing tracking datasets, where most objects tend to have distin-guishing appearance and re-ID models are sufficient for es-tablishing associations. In response to such bias, we would like to re-emphasize that methods for multi-object tracking should also work when object appearance is not sufficiently discriminative. To this end, we propose a large-scale dataset for multi-human tracking, where humans have sim-ilar appearance, diverse motion and extreme articulation. As the dataset contains mostly group dancing videos, we name it “DanceTrack”. We expect DanceTrack to provide a better platform to develop more MOT algorithms that rely less on visual discrimination and depend more on motion analysis. We benchmark several state-of-the-art trackers on our dataset and observe a significant performance drop on DanceTrack when compared against existing benchmarks. The dataset, project code and competition is released at: https://github.com/DanceTrack.

----

## [2019] Egocentric Prediction of Action Target in 3D

**Authors**: *Yiming Li, Ziang Cao, Andrew Liang, Benjamin Liang, Luoyao Chen, Hang Zhao, Chen Feng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02033](https://doi.org/10.1109/CVPR52688.2022.02033)

**Abstract**:

We are interested in anticipating as early as possible the target location of a person's object manipulation action in a 3D workspace from egocentric vision. It is important in fields like human-robot collaboration, but has not yet received enough attention from vision and learning communities. To stimulate more research on this challenging egocentric vision task, we propose a large multimodality dataset of more than 1 million frames of RGB-D and IMU streams, and provide evaluation metrics based on our high-quality 2D and 3D labels from semi-automatic annotation. Meanwhile, we design baseline methods using recurrent neural networks and conduct various ablation studies to validate their effectiveness. Our results demonstrate that this new task is worthy of further study by researchers in robotics, vision, and learning communities.

----

## [2020] HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction

**Authors**: *Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, Li Yi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02034](https://doi.org/10.1109/CVPR52688.2022.02034)

**Abstract**:

We present HOI4D, a large-scale 4D egocentric dataset with rich annotations, to catalyze the research of category-level human-object interaction. HOI4D consists of 2.4M RGB-D egocentric video frames over 4000 sequences col-lected by 9 participants interacting with 800 different ob-ject instances from 16 categories over 610 different indoor rooms. Frame-wise annotations for panoptic segmentation, motion segmentation, 3D hand pose, category-level object pose and hand action have also been provided, together with reconstructed object meshes and scene point clouds. With HOI4D, we establish three benchmarking tasks to pro-mote category-level HOI from 4D visual signals including semantic segmentation of 4D dynamic point cloud se-quences, category-level object pose tracking, and egocen-tric action segmentation with diverse interaction targets. In-depth analysis shows HOI4D poses great challenges to existing methods and produces huge research opportunities.

----

## [2021] Amodal Panoptic Segmentation

**Authors**: *Rohit Mohan, Abhinav Valada*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02035](https://doi.org/10.1109/CVPR52688.2022.02035)

**Abstract**:

Humans have the remarkable ability to perceive objects as a whole, even when parts of them are occluded. This ability of amodal perception forms the basis of our perceptual and cognitive understanding of our world. To enable robots to reason with this capability, we formulate and propose a novel task that we name amodal panoptic segmentation. The goal of this task is to simultaneously predict the pixel-wise semantic segmentation labels of the visible regions of stuff classes and the instance segmentation labels of both the visible and occluded regions of thing classes. To facilitate research on this new task, we extend two established benchmark datasets with pixel-level amodal panoptic segmentation labels that we make publicly available as KITTI-360-APS and BDD100K-APS. We present several strong baselines, along with the amodal panoptic quality (APQ) and amodal parsing coverage (APC) metrics to quantify the performance in an interpretable manner. Furthermore, we propose the novel amodal panoptic segmentation network (APSNet), as a first step towards addressing this task by explicitly modeling the complex relationships between the occluders and occludes. Extensive experimental evaluations demonstrate that APSNet achieves state-of-the-art performance on both benchmarks and more importantly exemplifies the utility of amodal recognition. The datasets are available at http://amodal-panoptic.cs.uni-freiburg.de.

----

## [2022] Large-scale Video Panoptic Segmentation in the Wild: A Benchmark

**Authors**: *Jiaxu Miao, Xiaohan Wang, Yu Wu, Wei Li, Xu Zhang, Yunchao Wei, Yi Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02036](https://doi.org/10.1109/CVPR52688.2022.02036)

**Abstract**:

In this paper, we present a new large-scale dataset for the video panoptic segmentation task, which aims to assign semantic classes and track identities to all pixels in a video. As the ground truth for this task is difficult to annotate, previous datasets for video panoptic segmentation are limited by either small scales or the number of scenes. In contrast, our large-scale VIdeo Panoptic Segmentation in the Wild (VIPSeg) dataset provides 3,536 videos and 84,750 frames with pixel-level panoptic annotations, covering a wide range of real-world scenarios and categories. To the best of our knowledge, our VIPSeg is the first attempt to tackle the challenging video panoptic segmentation task in the wild by considering diverse scenarios. Based on VIPSeg, we evaluate existing video panoptic segmentation approaches and propose an efficient and effective clip-based baseline method to analyze our VIPSeg dataset. Our dataset is available at https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/.

----

## [2023] YouMVOS: An Actor-centric Multi-shot Video Object Segmentation Dataset

**Authors**: *Donglai Wei, Siddhant Kharbanda, Sarthak Arora, Roshan Roy, Nishant Jain, Akash Palrecha, Tanav Shah, Shray Mathur, Ritik Mathur, Abhijay Kemkar, Anirudh Srinivasan Chakravarthy, Zudi Lin, Won-Dong Jang, Yansong Tang, Song Bai, James Tompkin, Philip H. S. Torr, Hanspeter Pfister*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02037](https://doi.org/10.1109/CVPR52688.2022.02037)

**Abstract**:

Many video understanding tasks require analyzing multishot videos, but existing datasets for video object segmentation (VOS) only consider single-shot videos. To address this challenge, we collected a new dataset-YouMVaS-of 200 popular YouTube videos spanning ten genres, where each video is on average five minutes long and with 75 shots. We selected recurring actors and annotated 431K segmentation masks at a frame rate of six, exceeding previous datasets in average video duration, object variation, and narrative structure complexity. We incorporated good practices of model architecture design, memory management, and multi-shot tracking into an existing video segmentation method to build competitive baseline methods. Through error analysis, we found that these baselines still fail to cope with cross-shot appearance variation on our YouMVOS dataset. Thus, our dataset poses new challenges in multi-shot segmentation towards better video analysis. Data, code, and pre-trained models are available at https://donglaiw.github.io/proj/youMVOS

----

## [2024] The DEVIL is in the Details: A Diagnostic Evaluation Benchmark for Video Inpainting

**Authors**: *Ryan Szeto, Jason J. Corso*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02038](https://doi.org/10.1109/CVPR52688.2022.02038)

**Abstract**:

Quantitative evaluation has increased dramatically among recent video inpainting work, but the video and mask content used to gauge performance has received relatively little attention. Although attributes such as camera and background scene motion inherently change the difficulty of the task and affect methods differently, existing evaluation schemes fail to control for them, thereby providing minimal insight into inpainting failure modes. To address this gap, we propose the Diagnostic Evaluation of Video Inpainting on Landscapes (DEVIL) benchmark, which consists of two contributions: (i) a novel dataset of videos and masks labeled according to several key inpainting failure modes, and (ii) an evaluation scheme that samples slices of the dataset characterized by a fixed content attribute, and scores performance on each slice according to reconstruction, realism, and temporal consistency quality. By revealing systematic changes in performance induced by particular characteristics of the input content, our challenging benchmark enables more insightful analysis into video inpainting methods and serves as an invaluable diagnostic tool for the field. Our code and data are available at github.com/MichiganCOG/devil.

----

## [2025] 3MASSIV: Multilingual, Multimodal and Multi-Aspect dataset of Social Media Short Videos

**Authors**: *Vikram Gupta, Trisha Mittal, Puneet Mathur, Vaibhav Mishra, Mayank Maheshwari, Aniket Bera, Debdoot Mukherjee, Dinesh Manocha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02039](https://doi.org/10.1109/CVPR52688.2022.02039)

**Abstract**:

We present 3MASSIV, a multilingual, multimodal and multi-aspect, expertly-annotated dataset of diverse short videos extracted from short-video social media platform - Moj. 3MASSIV comprises of 50k short videos (20 seconds average duration) and 100K unlabeled videos in 11 different languages and captures popular short video trends like pranks, fails, romance, comedy expressed via unique audio-visual formats like self-shot videos, reaction videos, lip-synching, self-sung songs, etc. 3MASSIV presents an opportunity for multimodal and multilingual semantic understanding on these unique videos by annotating them for concepts, affective states, media types, and audio language. We present a thorough analysis of 3MASSIV and highlight the variety and unique aspects of our dataset compared to other contemporary popular datasets with strong baselines. We also show how the social media content in 3MASSIV is dynamic and temporal in nature, which can be used for semantic understanding tasks and cross-lingual analysis.

----

## [2026] AxIoU: An Axiomatically Justified Measure for Video Moment Retrieval

**Authors**: *Riku Togashi, Mayu Otani, Yuta Nakashima, Esa Rahtu, Janne Heikkilä, Tetsuya Sakai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02040](https://doi.org/10.1109/CVPR52688.2022.02040)

**Abstract**:

Evaluation measures have a crucial impact on the direction of research. Therefore, it is of utmost importance to develop appropriate and reliable evaluation measures for new applications where conventional measures are not well suited. Video Moment Retrieval (VMR) is one such application, and the current practice is to use R@K, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\theta$</tex>
 for evaluating VMR systems. However, this measure has two disadvantages. First, it is rank-insensitive: It ignores the rank positions of successfully localised moments in the top-K ranked list by treating the list as a set. Second, it binarizes the Intersection over Union (IoU) of each retrieved video moment using the threshold 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\theta$</tex>
 and thereby ignoring fine-grained localisation quality of ranked moments. We propose an alternative measure for evaluating VMR, called Average Max IoU (AxIoU), which is free from the above two problems. We show that AxIoU satisfies two important axioms for VMR evaluation, namely, Invariance against Redundant Moments and Monotonicity with respect to the Best Moment, and also that R@ K, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\theta$</tex>
 satisfies the first axiom only. We also empirically examine how Ax-IoU agrees with R@K, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\theta$</tex>
, as well as its stability with respect to change in the test data and human-annotated temporal boundaries.

----

## [2027] A Large-scale Comprehensive Dataset and Copy-overlap Aware Evaluation Protocol for Segment-level Video Copy Detection

**Authors**: *Sifeng He, Xudong Yang, Chen Jiang, Gang Liang, Wei Zhang, Tan Pan, Qing Wang, Furong Xu, Chunguang Li, Jingxiong Liu, Hui Xu, Kaiming Huang, Yuan Cheng, Feng Qian, Xiaobo Zhang, Lei Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02041](https://doi.org/10.1109/CVPR52688.2022.02041)

**Abstract**:

In this paper, we introduce VCSL (Video Copy Segment Localization), a new comprehensive segment-level annotated video copy dataset. Compared with existing copy detection datasets restricted by either video-level annotation or small-scale, VCSL not only has two orders of magnitude more segment-level labelled data, with 160k realistic video copy pairs containing more than 280k localized copied segment pairs, but also covers a variety of video categories and a wide range of video duration. All the copied segments inside each collected video pair are manually extracted and accompanied by precisely annotated starting and ending timestamps. Alongside the dataset, we also propose a novel evaluation protocol that better measures the prediction accuracy of copy overlapping segments between a video pair and shows improved adaptability in different scenarios. By benchmarking several baseline and state-of-the-art segment-level video copy detection methods with the proposed dataset and evaluation metric, we provide a comprehensive analysis that uncovers the strengths and weaknesses of current approaches, hoping to open up promising directions for future works. The VCSL dataset, metric and benchmark codes are all publicly available at https://github.com/alipay/vCSL.

----

## [2028] Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities

**Authors**: *Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, Angela Yao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02042](https://doi.org/10.1109/CVPR52688.2022.02042)

**Abstract**:

Assembly101 is a new procedural activity dataset fea-turing 4321 videos of people assembling and disassembling 101 “take-apart” toy vehicles. Participants work without fixed instructions, and the sequences feature rich and natu-ral variations in action ordering, mistakes, and corrections. Assembly101 is the first multi-view action dataset, with si-multaneous static (8) and egocentric (4) recordings. Se-quences are annotated with more than 100K coarse and 1M fine-grained action segments, and I8M 3D hand poses. We benchmark on three action understanding tasks: recognition, anticipation and temporal segmentation. Ad-ditionally, we propose a novel task of detecting mistakes. The unique recording format and rich set of annotations al-low us to investigate generalization to new toys, cross-view transfer, long-tailed distributions, and pose vs. appearance. We envision that Assemblyl0l will serve as a new challenge to investigate various activity understanding problems.

----

## [2029] Optimal Correction Cost for Object Detection Evaluation

**Authors**: *Mayu Otani, Riku Togashi, Yuta Nakashima, Esa Rahtu, Janne Heikkilä, Shin'ichi Satoh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02043](https://doi.org/10.1109/CVPR52688.2022.02043)

**Abstract**:

Mean Average Precision (mAP) is the primary evaluation measure for object detection. Although object detection has a broad range of applications, mAP evaluates detectors in terms of the performance of ranked instance retrieval. Such the assumption for the evaluation task does not suit some downstream tasks. To alleviate the gap between downstream tasks and the evaluation scenario, we propose Optimal Correction Cost (OC-cost), which assesses detection accuracy at image level. OC-cost computes the cost of correcting detections to ground truths as a measure of accuracy. The cost is obtained by solving an optimal transportation problem between the detections and the ground truths. Unlike mAp, OC-cost is designed to penalize false positive and false negative detections properly, and every image in a dataset is treated equally. Our experimental result validates that OCscost has better agreement with human preference than a ranking-based measure, i.e., mAP for a single image. We also show that detectors' rankings by OC-cost are more consistent on different data splits than mAP. Our goal is not to replace mAP with OC-cost but provide an additional tool to evaluate detectors from another aspect. To help future researchers and developers choose a target measure, we provide a series of experiments to clarify how mAP and OC-cost differ.

----

## [2030] GrainSpace: A Large-scale Dataset for Fine-grained and Domain-adaptive Recognition of Cereal Grains

**Authors**: *Lei Fan, Yiwen Ding, Dongdong Fan, Donglin Di, Maurice Pagnucco, Yang Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02044](https://doi.org/10.1109/CVPR52688.2022.02044)

**Abstract**:

Cereal grains are a vital part of human diets and are important commodities for people's livelihood and international trade. Grain Appearance Inspection (GAI) serves as one of the crucial steps for the determination of grain quality and grain stratification for proper circulation, storage and food processing, etc. GAI is routinely performed manually by qualified inspectors with the aid of some hand tools. Automated GAI has the benefit of greatly assisting inspectors with their jobs but has been limited due to the lack of datasets and clear definitions of the tasks. In this paper we formulate GAI as three ubiquitous computer vision tasks: fine-grained recognition, domain adaptation and out-of-distribution recognition. We present a large-scale and publicly available cereal grains dataset called GrainSpace. Specifically, we construct three types of device prototypes for data acquisition, and a total of 5.25 million images determined by professional inspectors. The grain samples including wheat, maize and rice are collected from five countries and more than 30 regions. We also develop a comprehensive benchmark based on semi-supervised learning and self-supervised learning techniques. To the best of our knowledge, GrainSpace is the first publicly released dataset for cereal grain inspection, https://github.com/hellodfan/GrainSpace.

----

## [2031] ABO: Dataset and Benchmarks for Real-World 3D Object Understanding

**Authors**: *Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F. Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, Jitendra Malik*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02045](https://doi.org/10.1109/CVPR52688.2022.02045)

**Abstract**:

We introduce Amazon Berkeley Objects (ABO), a new large-scale dataset designed to help bridge the gap between real and virtual 3D worlds. ABO contains product catalog images, metadata, and artist-created 3D models with com-plex geometries and physically-based materials that cor-respond to real, household objects. We derive challenging benchmarks that exploit the unique properties of ABO and measure the current limits of the state-of-the-art on three open problems for real-world 3D object understanding: single-view 3D reconstruction, material estimation, and cross-domain multi-view object retrieval.

----

## [2032] Improving Segmentation of the Inferior Alveolar Nerve through Deep Label Propagation

**Authors**: *Marco Cipriano, Stefano Allegretti, Federico Bolelli, Federico Pollastri, Costantino Grana*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02046](https://doi.org/10.1109/CVPR52688.2022.02046)

**Abstract**:

Many recent works in dentistry and maxillofacial imagery focused on the Inferior Alveolar Nerve (IAN) canal detection. Unfortunately, the small extent of available 3D maxillofacial datasets has strongly limited the performance of deep learning-based techniques. On the other hand, a huge amount of sparsely annotated data is produced every day from the regular procedures in the maxillofacial practice. Despite the amount of sparsely labeled images being significant, the adoption of those data still raises an open problem. Indeed, the deep learning approach frames the presence of dense annotations as a crucial factor. Recent efforts in literature have hence focused on developing label propagation techniques to expand sparse annotations into dense labels. However, the proposed methods proved only marginally effective for the purpose of segmenting the alveolar nerve in CBCT scans. This paper exploits and publicly releases a new 3D densely annotated dataset, through which we are able to train a deep label propagation model which obtains better results than those available in literature. By combining a segmentation model trained on the 3D annotated data and label propagation, we significantly improve the state of the art in the Inferior Alveolar Nerve segmentation.

----

## [2033] ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes

**Authors**: *Dina Bashkirova, Mohamed Abdelfattah, Ziliang Zhu, James Akl, Fadi M. Alladkani, Ping Hu, Vitaly Ablavsky, Berk Çalli, Sarah Adel Bargal, Kate Saenko*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02047](https://doi.org/10.1109/CVPR52688.2022.02047)

**Abstract**:

Less than 35% of recyclable waste is being actually recycled in the US [2], which leads to increased soil and sea pollution and is one of the major concerns of environmental researchers as well as the common public. At the heart of the problem are the inefficiencies of the waste sorting process (separating paper, plastic, metal, glass, etc.) due to the extremely complex and cluttered nature of the waste stream. Recyclable waste detection poses a unique computer vision challenge as it requires detection of highly deformable and often translucent objects in cluttered scenes without the kind of context information usually present in human-centric datasets. This challenging computer vision task currently lacks suitable datasets or methods in the available literature. In this paper, we take a step towards computer-aided waste detection and present the first in-the-wild industrial-grade waste detection and segmentation dataset, ZeroWaste. We believe that ZeroWaste will catalyze research in object detection and semantic segmentation in extreme clutter as well as applications in the recycling domain. Our project page can be found at http://ai.bu.edu/zerowaste/

----

## [2034] DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation

**Authors**: *Aysim Toker, Lukas Kondmann, Mark Weber, Marvin Eisenberger, Andrés Camero, Jingliang Hu, Ariadna Pregel Hoderlein, Çaglar Senaras, Timothy Davis, Daniel Cremers, Giovanni Marchisio, Xiao Xiang Zhu, Laura Leal-Taixé*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02048](https://doi.org/10.1109/CVPR52688.2022.02048)

**Abstract**:

Earth observation is a fundamental tool for monitoring the evolution of land use in specific areas of interest. Observing and precisely defining change, in this context, requires both time-series data and pixel-wise segmentations. To that end, we propose the DynamicEarthNet dataset that consists of daily, multi-spectral satellite observations of 75 selected areas of interest distributed over the globe with imagery from Planet Labs. These observations are paired with pixel-wise monthly semantic segmentation labels of 7 land use and land cover (LULC) classes. DynamicEarthNet is the first dataset that provides this unique combination of daily measurements and high-quality labels. In our experiments, we compare several established baselines that either utilize the daily observations as additional training data (semi-supervised learning) or multiple observations at once (spatio-temporal learning) as a point of reference for future research. Finally, we propose a new evaluation metric SCS that addresses the specific challenges associated with time-series semantic change segmentation. The data is available at: https://mediatum.ub.tum.de/1650201.

----

## [2035] Open Challenges in Deep Stereo: the Booster Dataset

**Authors**: *Pierluigi Zama Ramirez, Fabio Tosi, Matteo Poggi, Samuele Salti, Stefano Mattoccia, Luigi Di Stefano*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02049](https://doi.org/10.1109/CVPR52688.2022.02049)

**Abstract**:

We present a novel high-resolution and challenging stereo dataset framing indoor scenes annotated with dense and accurate ground-truth disparities. Peculiar to our dataset is the presence of several specular and transparent surfaces, i.e. the main causes of failures for state-of-the-art stereo networks. Our acquisition pipeline leverages a novel deep space-time stereo framework which allows for easy and accurate labeling with sub-pixel precision. We re-lease a total of 419 samples collected in 64 different scenes and annotated with dense ground-truth disparities. Each sample include a high-resolution pair (12 Mpx) as well as an unbalanced pair (Left: 12 Mpx, Right: 1.1 Mpx). Additionally, we provide manually annotated material segmentation masks and 15K unlabeled samples. We evaluate state-of-the-art deep networks based on our dataset, highlighting their limitations in addressing the open challenges in stereo and drawing hints for future research.

----

## [2036] No-Reference Point Cloud Quality Assessment via Domain Adaptation

**Authors**: *Qi Yang, Yipeng Liu, Siheng Chen, Yiling Xu, Jun Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02050](https://doi.org/10.1109/CVPR52688.2022.02050)

**Abstract**:

We present a novel no-reference quality assessment metric, the image transferred point cloud quality assessment (IT-PCQA), for 3D point clouds. For quality assessment, deep neural network (DNN) has shown compelling performance on no-reference metric design. However, the most challenging issue for no-reference PCQA is that we lack large-scale subjective databases to drive robust networks. Our motivation is that the human visual system (HVS) is the decision-maker regardless of the type of media for quality assessment. Leveraging the rich subjective scores of the natural images, we can quest the evaluation criteria of human perception via DNN and transfer the capability of prediction to 3D point clouds. In particular, we treat natural images as the source domain and point clouds as the target domain, and infer point cloud quality via unsupervised adversarial domain adaptation. To extract effective latent features and minimize the domain discrepancy, we propose a hierarchical feature encoder and a conditional-discriminative network. Considering that the ultimate pur-pose is regressing objective score, we introduce a novel con-ditional cross entropy loss in the conditional-discriminative network to penalize the negative samples which hinder the convergence of the quality regression network. Experi-mental results show that the proposed method can achieve higher performance than traditional no-reference metrics, even comparable results with full-reference metrics. The proposed method also suggests the feasibility of assessing the quality of specific media content without the expensive and cumbersome subjective evaluations. Code is available at https://github.com/Qi-Yangsjtu/IT-PCQA.

----

## [2037] Exploring Endogenous Shift for Cross-domain Detection: A Large-scale Benchmark and Perturbation Suppression Network

**Authors**: *Renshuai Tao, Hainan Li, Tianbo Wang, Yanlu Wei, Yifu Ding, Bowei Jin, Hongping Zhi, Xianglong Liu, Aishan Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02051](https://doi.org/10.1109/CVPR52688.2022.02051)

**Abstract**:

Existing cross-domain detection methods mostly study the domain shifts where differences between domains are often caused by external environment and perceivable for humans. However, in real-world scenarios (e.g., MRI medical diagnosis, X-ray security inspection), there still exists another type of shift, named endogenous shift, where the differences between domains are mainly caused by the intrinsic factors (e.g., imaging mechanisms, hardware components, etc.), and usually inconspicuous. This shift can also severely harm the cross-domain detection performance but has been rarely studied. To support this study, we contribute the first Endogenous Domain Shift (EDS) benchmark, X-ray security inspection, where the endogenous shifts among the domains are mainly caused by different X-ray machine types with different hardware parameters, wear degrees, etc. EDS consists of 14,219 images including 31,654 common instances from three domains (X-ray machines), with bounding-box annotations from 10 categories. To handle the endogenous shift, we further introduce the Perturbation Suppression Network (PSN), motivated by the fact that this shift is mainly caused by two types of perturbations: category-dependent and category-independent ones. PSN respectively exploits local prototype alignment and global adversarial learning mechanism to suppress these two types of perturbations. The comprehensive evaluation results show that PSN outperforms SOTA methods, serving a new perspective to the cross-domain research community.

----

## [2038] How Good Is Aesthetic Ability of a Fashion Model?

**Authors**: *Xingxing Zou, Kaicheng Pang, Wen Zhang, Waikeung Wong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02052](https://doi.org/10.1109/CVPR52688.2022.02052)

**Abstract**:

We introduce A100 (Aesthetic 100) to assess the aesthetic ability of the fashion compatibility models. To date, it is the first work to address the AI model's aesthetic ability with detailed characterization based on the professional fashion domain knowledge. A100 has several desirable characteristics: 1. Completeness. It covers all types of standards in the fashion aesthetic system through two tests, namely LAT (Liberalism Aesthetic Test) and AAT (Academicism Aesthetic Test); 2. Reliability. It is training data agnostic and consistent with major indicators. It provides a fair and objective judgment for model comparison. 3. Explainability. Better than all previous indicators, the A100 further identifies essential characteristics of fashion aesthetics, thus showing the model's performance on more fine-grained dimensions, such as Color, Balance, Material, etc. Experimental results prove the advance of the A100 in the aforementioned aspects. All data can be found at https://github.com/AemikaChow/AiDLab-fAshIon-Data.

----

## [2039] Instance-wise Occlusion and Depth Orders in Natural Scenes

**Authors**: *Hyunmin Lee, Jaesik Park*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02053](https://doi.org/10.1109/CVPR52688.2022.02053)

**Abstract**:

In this paper, we introduce a new dataset, named InstaOrder, that can be used to understand the geometrical relationships of instances in an image. The dataset consists of 2.9M annotations of geometric orderings for class-labeled instances in 101K natural scenes. The scenes were annotated by 3,659 crowd-workers regarding (1) occlusion order that identifies occluder/occludee and (2) depth order that describes ordinal relations that consider relative distance from the camera. The dataset provides joint annotation of two kinds of orderings for the same instances, and we discover that the occlusion order and depth order are complementary. We also introduce a geometric order prediction network called InstaOrderNet, which is superior to state-of-the-art approaches. Moreover, we propose a dense depth prediction network called InstaDepthNet that uses auxiliary geometric order loss to boost the accuracy of the state-of-the-art depth prediction approach, MiDaS [54].

----

## [2040] PhoCaL: A Multi-Modal Dataset for Category-Level Object Pose Estimation with Photometrically Challenging Objects

**Authors**: *Pengyuan Wang, HyunJun Jung, Yitong Li, Siyuan Shen, Rahul Parthasarathy Srikanth, Lorenzo Garattoni, Sven Meier, Nassir Navab, Benjamin Busam*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02054](https://doi.org/10.1109/CVPR52688.2022.02054)

**Abstract**:

Object pose estimation is crucial for robotic applications and augmented reality. Beyond instance level 6D object pose estimation methods, estimating category-level pose and shape has become a promising trend. As such, a new research field needs to be supported by well-designed datasets. To provide a benchmark with high-quality ground truth annotations to the community, we introduce a multimodal dataset for category-level object pose estimation with photometrically challenging objects termed PhoCaL. PhoCaL comprises 60 high quality 3D models of household objects over 8 categories including highly reflective, transparent and symmetric objects. We developed a novel robot-supported multi-modal (RGB, depth, polarisation) data acquisition and annotation process. It ensures sub-millimeter accuracy of the pose for opaque textured, shiny and transparent objects, no motion blur and perfect camera synchronisation. To set a benchmark for our dataset, state-of-the-art RGB-D and monocular RGB methods are evaluated on the challenging scenes of PhoCaL.

----

## [2041] Replacing Labeled Real-image Datasets with Auto-generated Contours

**Authors**: *Hirokatsu Kataoka, Ryo Hayamizu, Ryosuke Yamada, Kodai Nakashima, Sora Takashima, Xinyu Zhang, Edgar Josafat Martinez-Noriega, Nakamasa Inoue, Rio Yokota*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02055](https://doi.org/10.1109/CVPR52688.2022.02055)

**Abstract**:

In the present work, we show that the performance of formula-driven supervised learning (FDSL) can match or even exceed that of ImageNet-21k without the use of real images, human-, and self-supervision during the pre-training of Vision Transformers (ViTs). For example, ViT-Base pre-trained on ImageNet-21k shows 81.8% top-1 accuracy when fine-tuned on ImageNet-1k and FDSL shows 82.7% top-1 accuracy when pre-trained under the same conditions (number of images, hyperparameters, and number of epochs). Images generated by formulas avoid the privacy/copyright issues, labeling cost and errors, and biases that real images suffer from, and thus have tremendous potential for pre-training general models. To understand the performance of the synthetic images, we tested two hypotheses, namely (i) object contours are what matter in FDSL datasets and (ii) increased number of parameters to create labels affects performance improvement in FDSL pre-training. To test the former hypothesis, we constructed a dataset that consisted of simple object contour combinations. We found that this dataset can match the performance of fractals. For the latter hypothesis, we found that increasing the difficulty of the pre-training task generally leads to better fine-tuning accuracy.

----

## [2042] V2C: Visual Voice Cloning

**Authors**: *Qi Chen, Mingkui Tan, Yuankai Qi, Jiaqiu Zhou, Yuanqing Li, Qi Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02056](https://doi.org/10.1109/CVPR52688.2022.02056)

**Abstract**:

Existing Voice Cloning (VC) tasks aim to convert a para-graph text to a speech with desired voice specified by a ref-erence audio. This has significantly boosted the development of artificial speech applications. However, there also exist many scenarios that cannot be well reflected by these VC tasks, such as movie dubbing, which requires the speech to be with emotions consistent with the movie plots. To fill this gap, in this work we propose a new task named Vi-sual Voice Cloning (V2C), which seeks to convert a para-graph of text to a speech with both desired voice speci-fied by a reference audio and desired emotion specified by a reference video. To facilitate research in this field, we construct a dataset, V2C-Animation, and propose a strong baseline based on existing state-of-the-art (SoTA) VC techniques. Our dataset contains 10,217 animated movie clips covering a large variety of genres (e.g., Comedy, Fantasy) and emotions (e.g., happy, sad). We further design a set of evaluation metrics, named MCD-DTW-SL, which help eval-uate the similarity between ground-truth speeches and the synthesised ones. Extensive experimental results show that even SoTA VC methods cannot generate satisfying speeches for our V2C task. We hope the proposed new task together with the constructed dataset and evaluation metric will fa-cilitate the research in the field of voice cloning and broader vision-and-language community. Source code and dataset will be released in https://github.com/chenqi008/V2C.

----

## [2043] M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining

**Authors**: *Xiao Dong, Xunlin Zhan, Yangxin Wu, Yunchao Wei, Michael C. Kampffmeyer, Xiaoyong Wei, Minlong Lu, Yaowei Wang, Xiaodan Liang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02057](https://doi.org/10.1109/CVPR52688.2022.02057)

**Abstract**:

Despite the potential of multi-modal pre-training to learn highly discriminative feature representations from complementary data modalities, current progress is being slowed by the lack of large-scale modality-diverse datasets. By leveraging the natural suitability of E-commerce, where different modalities capture complementary semantic information, we contribute a large-scale multi-modal pretraining dataset M5Product. The dataset comprises 5 modalities (image, text, table, video, and audio), covers over 6,000 categories and 5,000 attributes, and is 500× larger than the largest publicly available dataset with a similar number of modalities. Furthermore, M5Product contains incomplete modality pairs and noise while also having a long-tailed distribution, resembling most real-world problems. We further propose Self-harmonized ContrAstive LEarning (SCALE), a novel pretraining framework that integrates the different modalities into a unified model through an adaptive feature fusion mechanism, where the importance of each modality is learned directly from the modality embeddings and impacts the inter-modality contrastive learning and masked tasks within a multi-modal transformer model. We evaluate the current multi-modal pre-training state-of-the-art approaches and benchmark their ability to learn from unlabeled data when faced with the large number of modalities in the M5Product dataset. We conduct extensive experiments on four downstream tasks and demonstrate the superiority of our SCALE model, providing insights into the importance of dataset scale and diversity. Dataset and codes are available at 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://xiaodongsuper.github.io/M5Product_dataset/.

----

## [2044] It is Okay to Not Be Okay: Overcoming Emotional Bias in Affective Image Captioning by Contrastive Data Collection

**Authors**: *Youssef Mohamed, Faizan Farooq Khan, Kilichbek Haydarov, Mohamed Elhoseiny*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02058](https://doi.org/10.1109/CVPR52688.2022.02058)

**Abstract**:

Datasets that capture the connection between vision, language, and affection are limited, causing a lack of understanding of the emotional aspect of human intelligence. As a step in this direction, the ArtEmis dataset was recently introduced as a large-scale dataset of emotional reactions to images along with language explanations of these chosen emotions. We observed a significant emotional bias towards instance-rich emotions, making trained neural speakers less accurate in describing under-represented emotions. We show that collecting new data, in the same way, is not effective in mitigating this emotional bias. To remedy this problem, we propose a contrastive data collection approach to balance ArtEmis with a new complementary dataset such that a pair of similar images have contrasting emotions (one positive and one negative). We collected 260,533 instances using the proposed method, we combine them with ArtEmis, creating a second iteration of the dataset. The new combined dataset, dubbed ArtEmis v2.0, has a balanced distribution of emotions with explanations revealing more fine details in the associated painting. Our experiments show that neural speakers trained on the new dataset improve CIDEr and METEOR evaluation metrics by 20% and 7%, respectively, compared to the biased dataset. Finally, we also show that the performance per emotion of neural speakers is improved across all the emotion categories, significantly on under-represented emotions. The collected dataset and code are available at https://artemisdataset-v2.org.

----

## [2045] From Representation to Reasoning: Towards both Evidence and Commonsense Reasoning for Video Question-Answering

**Authors**: *Jiangtong Li, Li Niu, Liqing Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02059](https://doi.org/10.1109/CVPR52688.2022.02059)

**Abstract**:

Video understanding has achieved great success in representation learning, such as video caption, video object grounding, and video descriptive question-answer. However, current methods still struggle on video reasoning, including evidence reasoning and commonsense reasoning. To facilitate deeper video understanding towards video reasoning, we present the task of Causal-VidQA, which includes four types of questions ranging from scene description (description) to evidence reasoning (explanation) and commonsense reasoning (prediction and counterfactual). For commonsense reasoning, we set up a two-step solution by answering the question and providing a proper reason. Through extensive experiments on existing VideoQA methods, we find that the state-of-the-art methods are strong in descriptions but weak in reasoning. We hope that Causal-VidQA can guide the research of video understanding from representation learning to deeper reasoning. The dataset and related resources are available at https://github.com/bcmi/Causal-VidQA.git.

----

## [2046] Point Cloud Pre-training with Natural 3D Structures

**Authors**: *Ryosuke Yamada, Hirokatsu Kataoka, Naoya Chiba, Yukiyasu Domae, Tetsuya Ogata*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02060](https://doi.org/10.1109/CVPR52688.2022.02060)

**Abstract**:

The construction of 3D point cloud datasets requires a great deal of human effort. Therefore, constructing a large-scale 3D point clouds dataset is difficult. In order to rem-edy this issue, we propose a newly developed point cloud fractal database (PC-FractalDB), which is a novel family of formula-driven supervised learning inspired by fractal geometry encountered in natural 3D structures. Our re-search is based on the hypothesis that we could learn rep-resentations from more real-world 3D patterns than con-ventional 3D datasets by learning fractal geometry. We show how the PC-FractalDB facilitates solving several re-cent dataset-related problems in 3D scene understanding, such as 3D model collection and labor-intensive annotation. The experimental section shows how we achieved the performance rate of up to 61.9% and 59.0% for the Scan-NetV2 and SUN RGB-D datasets, respectively, over the current highest scores obtained with the PointContrast, con-trastive scene contexts (CSC), and RandomRooms. More-over, the PC-FractalDB pre-trained model is especially ef-fective in training with limited data. For example, in 10% of training data on ScanNetV2, the PC-FractalDB pre-trained VoteNet performs at 38.3%, which is +14.8% higher accu-racy than CSC. Of particular note, we found that the pro-posed method achieves the highest results for 3D object de-tection pre-training in limited point cloud data. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Dataset release: https://ryosuke-yamada.github.io/PointCloud-FractalDataBase/

----

## [2047] The Auto Arborist Dataset: A Large-Scale Benchmark for Multiview Urban Forest Monitoring Under Domain Shift

**Authors**: *Sara Beery, Guanhang Wu, Trevor Edwards, Filip Pavetic, Bo Majewski, Shreyasee Mukherjee, Stanley Chan, John Morgan, Vivek Rathod, Jonathan Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02061](https://doi.org/10.1109/CVPR52688.2022.02061)

**Abstract**:

Generalization to novel domains is a fundamental chal-lenge for computer vision. Near-perfect accuracy on bench-marks is common, but these models do not work as expected when deployed outside of the training distribution. To build computer vision systems that truly solve real-world prob-lems at global scale, we need benchmarks that fully capture real-world complexity, including geographic domain shift, long-tailed distributions, and data noise. We propose urban forest monitoring as an ideal testbed for studying and improving upon these computer vision challenges, while working towards filling a crucial environ-mental and societal need. Urban forests provide significant benefits to urban societies. However, planning and main-taining these forests is expensive. One particularly costly aspect of urban forest management is monitoring the ex-isting trees in a city: e.g., tracking tree locations, species, and health. Monitoring efforts are currently based on tree censuses built by human experts, costing cities millions of dollars per census and thus collected infrequently. Previous investigations into automating urban forest monitoring focused on small datasets from single cities, covering only common categories. To address these short-comings, we introduce a new large-scale dataset that joins public tree censuses from 23 cities with a large collection of street level and aerial imagery. Our Auto Arborist dataset contains over 2.5M trees and 344 genera and is >2 or-ders of magnitude larger than the closest dataset in the literature. We introduce baseline results on our dataset across modalities as well as metrics for the detailed analy-sis of generalization with respect to geographic distribution shifts, vital for such a system to be deployed at-scale.

----

## [2048] AutoMine: An Unmanned Mine Dataset

**Authors**: *Yuchen Li, Zixuan Li, Siyu Teng, Yu Zhang, Yuhang Zhou, Yuchang Zhu, Dongpu Cao, Bin Tian, Yunfeng Ai, Zhe XuanYuan, Long Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02062](https://doi.org/10.1109/CVPR52688.2022.02062)

**Abstract**:

Autonomous driving datasets have played an important role in validating the advancement of intelligent vehicle algorithms including localization, perception and prediction in academic areas. However, current existing datasets pay more attention to the structured urban road, which hampers the exploration on unstructured special scenarios. Moreover, the open-pit mine is one of the typical representatives for them. Therefore, we introduce the Autonomous driving dataset on the Mining scene (AutoMine) for positioning and perception tasks in this paper. The AutoMine is collected by multiple acquisition platforms including an SUV, a wide-body mining truck and an ordinary mining truck, depending on the actual mine operation scenarios. The dataset consists of 18+ driving hours, 18K annotated lidar and image frames for 3D perception with various mines, time-of-the-day and weather conditions. The main contributions of the AutoMine dataset are as follows: I.The first autonomous driving dataset for perception and localization in mine scenarios. 2.There are abundant dynamic obstacles of 9 degrees of freedom with large dimension difference (mining trucks and pedestrians) and extreme climatic conditions (the dust and snow) in the mining area. 3.Multi-platform acquisition strategies could capture mining data from multiple perspectives that fit the actual operation. More details can be found in our website(https://automine.cc).

----

## [2049] SmartPortraits: Depth Powered Handheld Smartphone Dataset of Human Portraits for State Estimation, Reconstruction and Synthesis

**Authors**: *Anastasiia Kornilova, Marsel Faizullin, Konstantin Pakulev, Andrey Sadkov, Denis Kukushkin, Azat Akhmetyanov, Timur Akhtyamov, Hekmat Taherinejad, Gonzalo Ferrer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02063](https://doi.org/10.1109/CVPR52688.2022.02063)

**Abstract**:

We present a dataset of 1000 video sequences of human portraits recorded in real and uncontrolled conditions by using a handheld smartphone accompanied by an external high-quality depth camera. The collected dataset contains 200 people captured in different poses and locations and its main purpose is to bridge the gap between raw measurements obtained from a smartphone and downstream applications, such as state estimation, 3D reconstruction, view synthesis, etc. The sensors employed in data collection are the smartphone's camera and Inertial Measurement Unit (IMU), and an external Azure Kinect DK depth camera software synchronized with sub-millisecond precision to the smartphone system. During the recording, the smartphone flash is used to provide a periodic secondary source of lightning. Accurate mask of the foremost person is provided as well as its impact on the camera alignment accuracy. For evaluation purposes, we compare multiple state-of-the-art camera alignment methods by using a Motion Cap-ture system. We provide a smartphone visual-inertial bench-mark for portrait capturing, where we report results for multiple methods and motivate further use of the provided trajectories, available in the dataset, in view synthesis and 3D reconstruction tasks.

----

## [2050] BigDatasetGAN: Synthesizing ImageNet with Pixel-wise Annotations

**Authors**: *Daiqing Li, Huan Ling, Seung Wook Kim, Karsten Kreis, Sanja Fidler, Antonio Torralba*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02064](https://doi.org/10.1109/CVPR52688.2022.02064)

**Abstract**:

Annotating images with pixel-wise labels is a time consuming and costly process. Recently, DatasetGAN [78] showcased a promising alternative - to synthesize a large labeled dataset via a generative adversarial network (GAN) by exploiting a small set of manually labeled, GAN generated images. Here, we scale DatasetGAN to ImageNet scale of class diversity. We take image samples from the class-conditional generative model BigGAN [5] trained on ImageNet, and manually annotate only 5 images per class, for all 1k classes. By training an effective feature segmentation architecture on top of BigGAN, we turn Big GAN into a labeled dataset generator. We further show that VQGAN [18] can similarly serve as a dataset generator, leveraging the already annotated data. We create a new ImageNet benchmark by labeling an additional set of real images and evaluate segmentation performance in a variety of settings. Through an extensive ablation study, we show big gains in leveraging a large generated dataset to train different supervised and self-supervised backbone models on pixel-wise tasks. Furthermore, we demonstrate that using our synthesized datasets for pre-training leads to improvements over standard ImageNet pre-training on several downstream datasets, such as PASCAL-VOC, MS-COCO, Cityscapes and chest X-ray, as well as tasks (detection, segmentation). Our benchmark will be made public and maintain a leaderboard for this challenging task. Project Page: https://nv-tlabs.github.io/big-datasetgan/

----

## [2051] Rope3D: The Roadside Perception Dataset for Autonomous Driving and Monocular 3D Object Detection Task

**Authors**: *Xiaoqing Ye, Mao Shu, Hanyu Li, Yifeng Shi, Yingying Li, Guangjie Wang, Xiao Tan, Errui Ding*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02065](https://doi.org/10.1109/CVPR52688.2022.02065)

**Abstract**:

Concurrent perception datasets for autonomous driving are mainly limited to frontal view with sensors mounted on the vehicle. None of them is designed for the overlooked roadside perception tasks. On the other hand, the data captured from roadside cameras have strengths over frontal-view data, which is believed to facilitate a safer and more intelligent autonomous driving system. To accelerate the progress of roadside perception, we present the first high-diversity challenging Roadside Perception 3D dataset- Rope3D from a novel view. The dataset consists of 50k images and over 1.5M 3D objects in various scenes, which are captured under different settings including various cameras with ambiguous mounting positions, camera specifications, viewpoints, and different environmental conditions. We conduct strict 2D-3D joint annotation and comprehensive data analysis, as well as set up a new 3D roadside perception benchmark with metrics and evaluation devkit. Furthermore, we tailor the existing frontal-view monocular 3D object detection approaches and propose to leverage the geometry constraint to solve the inherent ambiguities caused by various sensors, viewpoints. Our dataset is available on https://thudair.baai.ac.cn/rope.

----

## [2052] Unifying Panoptic Segmentation for Autonomous Driving

**Authors**: *Oliver Zendel, Matthias Schörghuber, Bernhard Rainer, Markus Murschitz, Csaba Beleznai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02066](https://doi.org/10.1109/CVPR52688.2022.02066)

**Abstract**:

This paper aims to improve panoptic segmentation for real-world applications in three ways. First, we present a label policy that unifies four of the most popular panoptic segmentation datasets for autonomous driving. We also clean up label confusion by adding the new vehicle labels pickup and van. Full relabeling information for the popular Mapillary Vistas, IDD, and Cityscapes dataset are provided to add these new labels to existing setups. Second, we introduce Wilddash2 (WD2), a new dataset and public benchmark service for panoptic segmentation. The dataset consists of more than 5000 unique driving scenes from all over the world with a focus on visually challenging scenes, such as diverse weather conditions, lighting situations, and camera characteristics. We showcase experimental visual hazard classifiers which help to pre-filter challenging frames during dataset creation. Finally, to characterize the robustness of algorithms in out-of-distribution situations, we introduce hazard-aware and negative testing for panoptic segmentation as well as statistical significance calculations that increase confidence for both concepts. Additionally, we present a novel technique for visualizing panoptic segmentation errors. Our experiments show the negative impact of visual hazards on panoptic segmentation quality. Additional data from the WD2 dataset improves performance for visually challenging scenes and thus robustness in real-world scenarios.

----

## [2053] DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection

**Authors**: *Haibao Yu, Yizhen Luo, Mao Shu, Yiyi Huo, Zebang Yang, Yifeng Shi, Zhenglong Guo, Hanyu Li, Xing Hu, Jirui Yuan, Zaiqing Nie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02067](https://doi.org/10.1109/CVPR52688.2022.02067)

**Abstract**:

Autonomous driving faces great safety challenges for a lack of global perspective and the limitation of long-range perception capabilities. It has been widely agreed that vehicle-infrastructure cooperation is required to achieve Level 5 autonomy. However, there is still NO dataset from real scenarios available for computer vision researchers to work on vehicle-infrastructure cooperation-related problems. To accelerate computer vision research and innovation for Vehicle-Infrastructure Cooperative Autonomous Driving (VICAD), we release DAIR-V2X Dataset, which is the first large-scale, multi-modality, multi-view dataset from real scenarios for VICAD. DAIR-V2X comprises 71254 LiDAR frames and 71254 Camera frames, and all frames are captured from real scenes with 3D annotations. The Vehicle-Infrastructure Cooperative 3D Object Detection problem (VIC3D) is introduced, formulating the problem of collaboratively locating and identifying 3D objects using sensory inputs from both vehicle and infrastructure. In addition to solving traditional 3D object detection problems, the solution of VIC3D needs to consider the temporal asynchrony problem between vehicle and infrastructure sensors and the data transmission cost between them. Furthermore, we propose Time Compensation Late Fusion (TCLF), a late fusion framework for the VIC3D task as a benchmark based on DAIR-V2X. Find data, code, and more up-to-date information at https://thudair.baai.ac.cn/index and https://github.com/AIR-Thu/dair-V2x.

----

## [2054] SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation

**Authors**: *Tao Sun, Mattia Segù, Janis Postels, Yuxuan Wang, Luc Van Gool, Bernt Schiele, Federico Tombari, Fisher Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02068](https://doi.org/10.1109/CVPR52688.2022.02068)

**Abstract**:

Adapting to a continuously evolving environment is a safety-critical challenge inevitably faced by all autonomous-driving systems. Existing image- and video-based driving datasets, however, fall short of capturing the mutable nature of the real world. In this paper, we introduce the largest multi-task synthetic dataset for autonomous driving, SHIFT. It presents discrete and continuous shifts in cloudiness, rain and fog intensity, time of day, and vehicle and pedestrian density. Featuring a comprehensive sensor suite and annotations for several mainstream perception tasks, SHIFT allows to investigate how a perception systems' performance degrades at increasing levels of domain shift, fostering the development of continuous adaptation strategies to mitigate this problem and assessing the robustness and generality of a model. Our dataset and benchmark toolkit are publicly available at www.vis.xyz/shift.

----

## [2055] Ithaca365: Dataset and Driving Perception under Repeated and Challenging Weather Conditions

**Authors**: *Carlos Andres Diaz-Ruiz, Youya Xia, Yurong You, Jose Nino, Junan Chen, Josephine Monica, Xiangyu Chen, Katie Luo, Yan Wang, Marc Emond, Wei-Lun Chao, Bharath Hariharan, Kilian Q. Weinberger, Mark E. Campbell*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02069](https://doi.org/10.1109/CVPR52688.2022.02069)

**Abstract**:

Advances in perception for self-driving cars have accelerated in recent years due to the availability of large-scale datasets, typically collected at specific locations and under nice weather conditions. Yet, to achieve the high safety requirement, these perceptual systems must operate robustly under a wide variety of weather conditions including snow and rain. In this paper, we present a new dataset to enable robust autonomous driving via a novel data collection process - data is repeatedly recorded along a 15 km route under diverse scene (urban, highway, rural, campus), weather (snow, rain, sun), time (day/night), and traffic conditions (pedestrians, cyclists and cars). The dataset includes images and point clouds from cameras and LiDAR sensors, along with high-precision GPS/INS to establish correspondence across routes. The dataset includes road and object annotations using amodal masks to capture partial occlusions and 3D bounding boxes. We demonstrate the uniqueness of this dataset by analyzing the performance of baselines in amodal segmentation of road and objects, depth estimation, and 3D object detection. The repeated routes opens new research directions in object discovery, continual learning, and anomaly detection. Link to Ithaca365: https://ithaca365.mae.cornell.edu/

----

## [2056] SCENIC: A JAX Library for Computer Vision Research and Beyond

**Authors**: *Mostafa Dehghani, Alexey A. Gritsenko, Anurag Arnab, Matthias Minderer, Yi Tay*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02070](https://doi.org/10.1109/CVPR52688.2022.02070)

**Abstract**:

Scenic is an open-source
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/google-research/scenic JAX library with a focus on transformer-based models for computer vision research and beyond. The goal of this toolkit is to facilitate rapid experimentation, prototyping, and research of new architectures and models. Scenic supports a diverse range of tasks (e.g., classification, segmentation, detection) and facilitates working on multi-modal problems, along with GPU/TPU support for large-scale, multi-host and multi-device training. Scenic also offers optimized implementations of state-of-the-art research models spanning a wide range of modalities. Scenic has been successfully used for numerous projects and published papers and continues serving as the library of choice for rapid prototyping and publication of new research ideas.

----

## [2057] DeepLIIF: An Online Platform for Quantification of Clinical Pathology Slides

**Authors**: *Parmida Ghahremani, Joseph Marino, Ricardo Dodds, Saad Nadeem*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02071](https://doi.org/10.1109/CVPR52688.2022.02071)

**Abstract**:

In the clinic, resected tissue samples are stained with Hematoxylin-and-Eosin (H&E) and/or Immunhistochem-istry (IHC) stains and presented to the pathologists on glass slides or as digital scans for diagnosis and assess-ment of disease progression. Cell-level quantification, e.g. in IHC protein expression scoring, can be extremely in-efficient and subjective. We present DeepLIIF (https://deepliif.org), a first free online platform for ef-ficient and reproducible IHC scoring. DeepLIIF outper-forms current state-of-the-art approaches (relying on man-ual error-prone annotations) by virtually restaining clinical IHC slides with more informative multiplex immunofluores-cence staining. Our DeepLIIF cloud-native platform sup-ports (1) more than 150 proprietary/non-proprietary input formats via the Bio-Formats standard, (2) interactive ad-justment, visualization, and downloading of the IHC quan-tification results and the accompanying restained images, (3) consumption of an exposed workflow API programmat-ically or through interactive plugins for open source whole slide image viewers such as QuPath/ImageJ, and (4) auto scaling to efficiently scale GPU resources based on user demand.

----

## [2058] VL-InterpreT: An Interactive Visualization Tool for Interpreting Vision-Language Transformers

**Authors**: *Estelle Aflalo, Meng Du, Shao-Yen Tseng, Yongfei Liu, Chenfei Wu, Nan Duan, Vasudev Lal*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02072](https://doi.org/10.1109/CVPR52688.2022.02072)

**Abstract**:

Breakthroughs in transformer-based models have revolutionized not only the NLP field, but also vision and multimodal systems. However, although visualization and interpretability tools have become available for NLP models, internal mechanisms of vision and multimodal transformers remain largely opaque. With the success of these transformers, it is increasingly critical to understand their inner workings, as unraveling these black-boxes will lead to more capable and trustworthy models. To contribute to this quest, we propose VL-InterpreT, which provides novel interactive visualizations for interpreting the attentions and hidden representations in multimodal transformers. VL-InterpreT is a task agnostic and integrated tool that (1) tracks a variety of statistics in attention heads throughout all layers for both vision and language components, (2) visualizes cross-modal and intra-modal attentions through easily readable heatmaps, and (3) plots the hidden representations of vision and language tokens as they pass through the transformer layers. In this paper, we demonstrate the functionalities of VL-InterpreT through the analysis of KD-VLP, an end-to-end pretraining vision-language multimodal transformer-based model, in the tasks of Visual Commonsense Reasoning (VCR) and WebQA, two visual question answering benchmarks. Furthermore, we also present a few interesting findings about multimodal transformer behaviors that were learned through our tool.

----

## [2059] GeoEngine: A Platform for Production-Ready Geospatial Research

**Authors**: *Sagar Verma, Siddharth Gupta, Hal Shin, Akash Panigrahi, Shubham Goswami, Shweta Pardeshi, Natanael Exe, Ujwal Dutta, Tanka Raj Joshi, Nitin Bhojwani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02073](https://doi.org/10.1109/CVPR52688.2022.02073)

**Abstract**:

Geospatial machine learning has seen tremendous aca-demic advancement, but its practical application has been constrained by difficulties with operationalizing performant and reliable solutions. Sourcing satellite imagery in real-world settings, handling terabytes of training data, and managing machine learning artifacts are a few of the chal-lenges that have severely limited downstream innovation. In this paper we introduce the GeoEngine
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://apps.granular.ai/apps platform for re-producible and production-ready geospatial machine learning research. GeoEngine removes key technical hurdles to adopting computer vision and deep learning-based geospa-tial solutions at scale. It is the first end-to-end geospatial machine learning platform, simplifying access to insights locked behind petabytes of imagery. Backed by a rigor-ous research methodology, this geospatial framework em-powers researchers with powerful abstractions for image sourcing, dataset development, model development, large scale training, and model deployment. In this paper we pro-vide the GeoEngine architecture explaining our design rationale in detail. We provide several real-world use cases of image sourcing, dataset development, and model building that have helped different organisations build and deploy geospatial solutions.

----

## [2060] Talking Face Generation with Multilingual TTS

**Authors**: *Hyoung-Kyu Song, Sang Hoon Woo, Junhyeok Lee, Seungmin Yang, Hyunjae Cho, Youseong Lee, Dongho Choi, Kang-wook Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02074](https://doi.org/10.1109/CVPR52688.2022.02074)

**Abstract**:

Recent studies in talking face generation have focused on building a model that can generalize from any source speech to any target identity. A number of works have already claimed this functionality and have added that their models will also generalize to any language. However, we show, using languages from different language families, that these models do not translate well when the training language and the testing language are sufficiently different. We reduce the scope of the problem to building a language-robust talking face generation system on seen identities, i.e., the target identity is the same as the training identity. In this work, we introduce a talking face generation system that generalizes to different languages. We evaluate the efficacy of our system using a multilingual text-to-speech system. We present the joint text-to-speech system and the talking face generation system as a neural dubber system. Our demo is available at https://bit.ly/ml-face-generation-cvpr22-demo. Also, our screencast is uploaded at https://youtu.be/F6h0s0M4vBI.

----

## [2061] Real-Time, Accurate, and Consistent Video Semantic Segmentation via Unsupervised Adaptation and Cross-Unit Deployment on Mobile Device

**Authors**: *Hyojin Park, Alan Yessenbayev, Tushar Singhal, Navin Kumar Adhikari, Yizhe Zhang, Shubhankar Mangesh Borse, Hong Cai, Frank Mayer, Balaji Calidas, Nilesh Prasad Pandey, Fei Yin, Fatih Porikli*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02075](https://doi.org/10.1109/CVPR52688.2022.02075)

**Abstract**:

This demonstration showcases our innovations on efficient, accurate, and temporally consistent video semantic segmentation on mobile device. We employ our test-time unsupervised scheme, AuxAdapt, to enable the segmentation model to adapt to a given video in an online manner. More specifically, we leverage a small auxiliary network to perform weight updates and keep the large, main segmen-tation network frozen. This significantly reduces the computational cost of adaptation when compared to previous methods (e.g., Tent, DVP), and at the same time, prevents catastrophic forgetting. By running AuxAdapt, we can considerably improve the temporal consistency of video segmentation while maintaining the accuracy. We demonstrate how to efficiently deploy our adaptive video segmentation algorithm on a smartphone powered by a Snapdragon® Mobile Platform
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Snapdragon is a product of Qualcomm Technologies, Inc. and/or its subsidiaries., Rather than simply running the entire algorithm on the GPU, we adopt a crossunit deployment strategy. The main network, which will be frozen during test time, will perform inferences on a highly optimized AI accelerator unit, while the small auxiliary net-work, which will be updated on the fly, will run forward passes and back-propagations on the GPU. Such a deployment scheme best utilizes the available processing power on the smartphone and enables real-time operation of our adaptive video segmentation algorithm. We provide example videos in supplementary material.

----

## [2062] BigDL 20: Seamless Scaling of AI Pipelines from Laptops to Distributed Cluster

**Authors**: *Jason Jinquan Dai, Ding Ding, Dongjie Shi, Shengsheng Huang, Jiao Wang, Xin Qiu, Kai Huang, Guoqiong Song, Yang Wang, Qiyuan Gong, Jiaming Song, Shan Yu, Le Zheng, Yina Chen, Junwei Deng, Ge Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02076](https://doi.org/10.1109/CVPR52688.2022.02076)

**Abstract**:

Most AI projects start with a Python notebook running on a single laptop; however, one usually needs to go through a mountain of pains to scale it to handle larger dataset (for both experimentation and production deployment). These usually entail many manual and error-prone steps for the data scientists to fully take advantage of the available hardware resources (e.g., SIMD instructions, multi-processing, quantization, memory allocation optimization, data partitioning, distributed computing, etc.). To address this challenge, we have open sourced BigDL 2.0 at https://github.com/intel-analytics/BigDL/ under Apache 2.0 license (combining the original BigDL [19] and Analytics Zoo [18] projects); using BigDL 2.0, users can simply build conventional Python notebooks on their laptops (with possible AutoML support), which can then be transparently accelerated on a single node (with up-to 9.6x speedup in our experiments), and seamlessly scaled out to a large cluster (across several hundreds servers in real-world use cases). BigDL 2.0 has already been adopted by many real-world users (such as Mastercard, Burger King, Inspur, etc.) in production.

----

## [2063] Interactive Segmentation and Visualization for Tiny Objects in Multi-megapixel Images

**Authors**: *Chengyuan Xu, Boning Dong, Noah Stier, Curtis McCully, D. Andrew Howell, Pradeep Sen, Tobias Höllerer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02077](https://doi.org/10.1109/CVPR52688.2022.02077)

**Abstract**:

We introduce an interactive image segmentation and visualization framework for identifying, inspecting, and editing tiny objects (just a few pixels wide) in large multi-megapixel high-dynamic-range (HDR) images. Detecting cosmic rays (CRs) in astronomical observations is a cum-bersome workflow that requires multiple tools, so we developed an interactive toolkit that unifies model inference, HDR image visualization, segmentation mask inspection and editing into a single graphical user interface. The feature set, initially designed for astronomical data, makes this work a useful research-supporting tool for human-in-the-loop tiny-object segmentation in scientific areas like biomedicine, materials science, remote sensing, etc., as well as computer vision. Our interface features mouse-controlled, synchronized, dual-window visualization of the image and the segmentation mask, a critical feature for locating tiny objects in multi-megapixel images. The browser-based tool can be readily hosted on the web to provide multi-user access and GPU acceleration for any device. The toolkit can also be used as a high-precision annotation tool, or adapted as the frontend for an interactive machine learning framework. Our open-source dataset, CR detection model, and visualization toolkit are available at https://github.com/cy-xu/cosmic-com.

----

## [2064] A Low-cost & Realtime Motion Capture System

**Authors**: *Anargyros Chatzitofis, Georgios Albanis, Nikolaos Zioulis, Spyridon Thermos*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02078](https://doi.org/10.1109/CVPR52688.2022.02078)

**Abstract**:

Traditional marker-based motion capture requires excessive and specialized equipment, hindering accessibility and wider adoption. In this work, we demonstrate such a system but rely on a very sparse set of low-cost consumer-grade sensors. Our system exploits a data-driven backend to infer the captured subject's joint positions from noisy marker estimates in real-time. In addition to reduced costs and portability, its inherent denoising nature allows for quicker captures by alleviating the need for precise marker placement and post-processing, making it suitable for interactive virtual reality applications.

----

## [2065] PyMiceTracking: An Open-Source Toolbox For Real-Time Behavioral Neuroscience Experiments

**Authors**: *Richardson Santiago Teles de Menezes, Aron de Miranda, Helton Maia Peixoto*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02079](https://doi.org/10.1109/CVPR52688.2022.02079)

**Abstract**:

The development of computational tools allows the advancement of research in behavioral neuroscience and elevates the limits of experiment design. Many behavioral experiments need to determine the animal's position from its tracking, which is crucial for real-time decision-making and further analysis of experimental data. Modern experimental designs usually generate the recording of a large amount of data, requiring the development of automatic computational tools and intelligent algorithms for timely data acquisition and processing. The proposed tool in this study initially operates with the acquisition of images. Then the animal tracking step begins with background subtraction, followed by the animal contour detection and morphological operations to remove noise in the detected shapes. Finally, in the final stage of the algorithm, the principal components analysis (PCA) is applied in the obtained shape, resulting in the animal's gaze direction.

----

## [2066] Effective conditioned and composed image retrieval combining CLIP-based features

**Authors**: *Alberto Baldrati, Marco Bertini, Tiberio Uricchio, Alberto Del Bimbo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02080](https://doi.org/10.1109/CVPR52688.2022.02080)

**Abstract**:

Conditioned and composed image retrieval extend CBIR systems by combining a query image with an additional text that expresses the intent of the user, describing additional requests w.r.t. the visual content of the query image. This type of search is interesting for e-commerce applications, e.g. to develop interactive multimodal searches and chat-bots. In this demo, we present an interactive system based on a combiner network, trained using contrastive learning, that combines visual and textual features obtained from the OpenAI CLIP network to address conditioned CBIR. The system can be used to improve e-shop search engines. For example, considering the fashion domain it lets users search for dresses, shirts and toptees using a candidate start image and expressing some visual differences w.r.t. its visual con-tent, e.g. asking to change color, pattern or shape. The pro-posed network obtains state-of-the-art performance on the FashionIQ dataset and on the more recent CIRR dataset, showing its applicability to the fashion domain for conditioned retrieval, and to more generic content considering the more general task of composed image retrieval.

----

## [2067] VIsCUIT: Visual Auditor for Bias in CNN Image Classifier

**Authors**: *Seongmin Lee, Judy Hoffman, Zijie J. Wang, Duen Horng Chau*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02081](https://doi.org/10.1109/CVPR52688.2022.02081)

**Abstract**:

CNN image classifiers are widely used, thanks to their efficiency and accuracy. However, they can suffer from biases that impede their practical applications. Most existing bias investigation techniques are either inapplicable to general image classification tasks or require significant user efforts in perusing all data subgroups to manually specify which data attributes to inspect. We present VIsCUIT, an interactive visualization system that reveals how and why a CNN classifier is biased. VIsCUIT visually summarizes the subgroups on which the classifier underperforms and helps users discover and characterize the cause of the underperformances by revealing image concepts responsible for activating neurons that contribute to misclassifications. VIsCUIT runs in modern browsers and is opensource, allowing people to easily access and extend the tool to other model architectures and datasets. VIsCUIT is available at the following public demo link: https://poloclub.github.io/VisCUIT. A video demo is available at https://youtu.be/eNDbSyM4R_4.

----

## [2068] DetectorDetective: Investigating the Effects of Adversarial Examples on Object Detectors

**Authors**: *Sivapriya Vellaichamy, Matthew Hull, Zijie J. Wang, Nilaksh Das, Sheng-Yun Peng, Haekyu Park, Duen Horng (Polo) Chau*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02082](https://doi.org/10.1109/CVPR52688.2022.02082)

**Abstract**:

With deep learning based systems performing exceedingly well in many vision-related tasks, a major concern with their widespread deployment especially in safety-critical applications is their susceptibility to adversarial attacks. We propose DetectorDetective, an interactive visual tool that aims to help users better understand the behaviors of a model as adversarial images journey through an object detector. DetectorDetective enables users to easily learn about how the three key modules of the Faster R-CNN object detector — Feature Pyramidal Network, Region Proposal Network, and Region Of Interest Head — respond to a user-selected benign image and its adversarial version. Visualizations about the progressive changes in the intermediate features among such modules help users gain insights into the impact of adversarial attacks, and perform side-by-side comparisons between the benign and adversarial responses. Furthermore, DetectorDetective displays saliency maps for the input images to comparatively highlight image regions that contribute to attack success. DetectorDetective complements adversarial machine learning research on object detection by providing a user-friendly interactive tool for inspecting and understanding model responses. DetectorDetective is available at the following public demo link: https://poloclub.github.io/detector-detective. A video demo is available at https://youtu.be/5C3Klh87CZI.

----

## [2069] V-Doc : Visual questions answers with Documents

**Authors**: *Yihao Ding, Zhe Huang, Runlin Wang, Yanhang Zhang, Xianru Chen, Yuzhong Ma, Hyunsuk Chung, Soyeon Caren Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02083](https://doi.org/10.1109/CVPR52688.2022.02083)

**Abstract**:

We propose V-Doc, a question-answering tool using document images and PDF, mainly for researchers and general non-deep learning experts looking to generate, process, and understand the document visual question answering tasks. The V-Doc supports generating and using both extractive and abstractive question-answer pairs using documents images. The extractive QA selects a subset of tokens or phrases from the document contents to predict the answers, while the abstractive QA recognises the language in the content and generates the answer based on the trained model. Both aspects are crucial to understanding the documents, especially in an image format. We include a detailed scenario of question generation for the abstractive QA task. V-Doc supports a wide range of datasets and models, and is highly extensible through a declarative, framework-agnostic platform.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Data and demo video: https://github.com/usydnlp/vdoc

----

## [2070] Clustering Plotted Data by Image Segmentation

**Authors**: *Tarek Naous, Srinjay Sarkar, Abubakar Abid, James Zou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.02084](https://doi.org/10.1109/CVPR52688.2022.02084)

**Abstract**:

Clustering is a popular approach to detecting patterns in unlabeled data. Existing clustering methods typically treat samples in a dataset as points in a metric space and compute distances to group together similar points. In this paper, we present a different way of clustering points in 2-dimensional space, inspired by how humans cluster data: by training neural networks to perform instance segmentation on plotted data. Our approach, Visual Clustering, has several advantages over traditional clustering algorithms: it is much faster than most existing clustering algorithms (making it suitable for very large datasets), it agrees strongly with human intuition for clusters, and it is by default hyperparameter free (although additional steps with hyperparameters can be introduced for more control of the algorithm). We describe the method and compare it to ten other clustering methods on synthetic data to illustrate its advantages and disadvantages. We then demonstrate how our approach can be extended to higher-dimensional data and illustrate its performance on real-world data. Our implementation of Visual Clustering is publicly available as a python package that can be installed and used on any dataset in a few lines of code
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://hithub.com/tareknaous/visual-clustering. A demo on synthetic datasets is provided
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
https://huggingface.co/spaces/CVPR/visual-clustering.

----

## [2071] Spatial-Temporal Parallel Transformer for Arm-Hand Dynamic Estimation

**Authors**: *Shuying Liu, Wenbin Wu, Jiaxian Wu, Yue Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01987](https://doi.org/10.1109/CVPR52688.2022.01987)

**Abstract**:

We propose an approach to estimate arm and hand dynamics from monocular video by utilizing the relationship between arm and hand. Although monocular full human motion capture technologies have made great progress in recent years, recovering accurate and plausible arm twists and hand gestures from in-the-wild videos still remains a challenge. To solve this problem, our solution is proposed based on the fact that arm poses and hand gestures are highly correlated in most real situations. To fully exploit arm-hand correlation as well as inter-frame information, we carefully design a Spatial-Temporal Parallel Arm-Hand Motion Transformer (PAHMT) to predict the arm and hand dynamics simultaneously. We also introduce new losses to encourage the estimations to be smooth and accurate. Besides, we collect a motion capture dataset including 200K frames of hand gestures and use this data to train our model. By integrating a 2D hand pose estimation model and a 3D human pose estimation model, the proposed method can produce plausible arm and hand dynamics from monocular video. Extensive evaluations demonstrate that the proposed method has advantages over previous state-of-the-art approaches and shows robustness under various challenging scenarios.

----



[Go to the previous page](CVPR-2022-list10.md)

[Go to the catalog section](README.md)