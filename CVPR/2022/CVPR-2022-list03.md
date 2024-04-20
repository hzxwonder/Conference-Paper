## [400] Exploring Frequency Adversarial Attacks for Face Forgery Detection

**Authors**: *Shuai Jia, Chao Ma, Taiping Yao, Bangjie Yin, Shouhong Ding, Xiaokang Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00407](https://doi.org/10.1109/CVPR52688.2022.00407)

**Abstract**:

Various facial manipulation techniques have drawn seri-ous public concerns in morality, security, and privacy. Al- though existing face forgery classifiers achieve promising performance on detecting fake images, these methods are vulnerable to adversarial examples with injected impercep- tible perturbations on the pixels. Meanwhile, many face forgery detectors always utilize the frequency diversity be-tween real and fake faces as a crucial clue. In this paper, in- stead of injecting adversarial perturbations into the spatial domain, we propose a frequency adversarial attack method against face forgery detectors. Concretely, we apply dis-crete cosine transform (DCT) on the input images and in-troduce a fusion module to capture the salient region of ad-versary in the frequency domain. Compared with existing adversarial attacks (e.g. FGSM, PGD) in the spatial do-main, our method is more imperceptible to human observers and does not degrade the visual quality of the original images. Moreover, inspired by the idea of meta-learning, we also propose a hybrid adversarial attack that performs at-tacks in both the spatial and frequency domains. Exten-sive experiments indicate that the proposed method fools not only the spatial-based detectors but also the state-of- the-art frequency-based detectors effectively. In addition, the proposed frequency attack enhances the transferability across face forgery detectors as black-box attacks.

----

## [401] End-to-End Reconstruction-Classification Learning for Face Forgery Detection

**Authors**: *Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, Xiaokang Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00408](https://doi.org/10.1109/CVPR52688.2022.00408)

**Abstract**:

Existing face forgery detectors mainly focus on specific forgery patterns like noise characteristics, local textures, or frequency statistics for forgery detection. This causes specialization of learned representations to known forgery patterns presented in the training set, and makes it difficult to detect forgeries with unknown patterns. In this paper, from a new perspective, we propose a forgery detection frame-work emphasizing the common compact representations of genuine faces based on reconstruction-classification learning. Reconstruction learning over real images enhances the learned representations to be aware of forgery patterns that are even unknown, while classification learning takes the charge of mining the essential discrepancy between real and fake images, facilitating the understanding of forgeries. To achieve better representations, instead of only using the encoder in reconstruction learning, we build bipartite graphs over the encoder and decoder features in a multi-scale fashion. We further exploit the reconstruction difference as guidance of forgery traces on the graph output as the final representation, which is fed into the classifier for forgery detection. The reconstruction and classification learning is optimized end-to-end. Extensive experiments on large-scale benchmark datasets demonstrate the superiority of the proposed method over state of the arts.

----

## [402] Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing

**Authors**: *Zhuo Wang, Zezheng Wang, Zitong Yu, Weihong Deng, Jiahong Li, Tingting Gao, Zhongyuan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00409](https://doi.org/10.1109/CVPR52688.2022.00409)

**Abstract**:

With diverse presentation attacks emerging continually, generalizable face anti-spoofing (FAS) has drawn growing attention. Most existing methods implement domain generalization (DG) on the complete representations. However, different image statistics may have unique properties for the FAS tasks. In this work, we separate the complete representation into content and style ones. A novel Shuffled Style Assembly Network (SSAN) is proposed to extract and reassemble different content and style features for a stylized feature space. Then, to obtain a generalized representation, a contrastive learning strategy is developed to emphasize liveness-related style information while suppress the domain-specific one. Finally, the representations of the correct assemblies are used to distinguish between living and spoofing during the inferring. On the other hand, despite the decent performance, there still exists a gap between academia and industry, due to the difference in data quantity and distribution. Thus, a new large-scale benchmark for FAS is built up to further evaluate the performance of algorithms in reality. Both qualitative and quantitative results on existing and proposed benchmarks demonstrate the effectiveness of our methods. The codes will be available at https://github.com/wangzhuo2019/SSAN.

----

## [403] Privacy-preserving Online AutoML for Domain-Specific Face Detection

**Authors**: *Chenqian Yan, Yuge Zhang, Quanlu Zhang, Yaming Yang, Xinyang Jiang, Yuqing Yang, Baoyuan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00410](https://doi.org/10.1109/CVPR52688.2022.00410)

**Abstract**:

Despite the impressive progress of general face detection, the tuning of hyper-parameters and architectures is still critical for the performance of a domain-specific face detector. Though existing AutoML works can speedup such process, they either require tuning from scratch for a new scenario or do not consider data privacy. To scale up, we derive a new AutoML setting from a platform perspective. In such setting, new datasets sequentially arrive at the platform, where an architecture and hyper-parameter configuration is recommended to train the optimal face detector for each dataset. This, however, brings two major challenges: (1) how to predict the best configuration for any given dataset without touching their raw images due to the privacy concern? and (2) how to continuously improve the AutoML algorithm from previous tasks and offer a better warm-up for future ones? We introduce “HyperFD”, a new privacy-preserving online AutoML framework for face detection. At its core part, a novel meta-feature representation of a dataset as well as its learning paradigm is proposed. Thanks to HyperFD, each local task (client) is able to effectively leverage the learning “experience” of previous tasks without uploading raw images to the platform; meanwhile, the meta-feature extractor is continuously learned to better trade off the bias and variance. Extensive experiments demonstrate the effectiveness and efficiency of our design.

----

## [404] Simulated Adversarial Testing of Face Recognition Models

**Authors**: *Nataniel Ruiz, Adam Kortylewski, Weichao Qiu, Cihang Xie, Sarah Adel Bargal, Alan L. Yuille, Stan Sclaroff*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00411](https://doi.org/10.1109/CVPR52688.2022.00411)

**Abstract**:

Most machine learning models are validated and tested on fixed datasets. This can give an incomplete picture of the capabilities and weaknesses of the model. Such weaknesses can be revealed at test time in the real world. The risks involved in such failures can be loss of profits, loss of time or even loss of life in certain critical applications. In order to alleviate this issue, simulators can be controlled in a finegrained manner using interpretable parameters to explore the semantic image manifold. In this work, we propose a framework for learning how to test machine learning algorithms using simulators in an adversarial manner in order to find weaknesses in the model before deploying it in critical scenarios. We apply this method in a face recognition setup. We show that certain weaknesses of models trained on real data can be discovered using simulated samples. Using our proposed method, we can find adversarial synthetic faces that fool contemporary face recognition models. This demonstrates the fact that these models have weaknesses that are not measured by commonly used validation datasets. We hypothesize that this type of adversarial examples are not isolated, but usually lie in connected spaces in the latent space of the simulator. We present a method to find these adversarial regions as opposed to the typical adversarial points found in the adversarial example literature.

----

## [405] Decoupled Multi-task Learning with Cyclical Self-Regulation for Face Parsing

**Authors**: *Qingping Zheng, Jiankang Deng, Zheng Zhu, Ying Li, Stefanos Zafeiriou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00412](https://doi.org/10.1109/CVPR52688.2022.00412)

**Abstract**:

This paper probes intrinsic factors behind typical failure cases (e.g. spatial inconsistency and boundary confusion) produced by the existing state-of-the-art method in face parsing. To tackle these problems, we propose a novel Decoupled Multi-task Learning with Cyclical Self-Regulation (DML-CSR) for face parsing. Specifically, DML-CSR designs a multi-task model which comprises face parsing, binary edge, and category edge detection. These tasks only share low-level encoder weights without high-level interactions between each other, enabling to decouple auxiliary modules from the whole network at the inference stage. To address spatial inconsistency, we develop a dynamic dual graph convolutional network to capture global contextual information without using any extra pooling operation. To handle boundary confusion in both single and multiple face scenarios, we exploit binary and category edge detection to jointly obtain generic geometric structure and fine-grained semantic clues of human faces. Besides, to prevent noisy labels from degrading model generalization during training, cyclical self-regulation is proposed to self-ensemble several model instances to get a new model and the resulting model then is used to self-distill subsequent models, through alternating iterations. Experiments show that our method achieves the new state-of-the-art performance on the Helen, CelebAMask-HQ, and Lapa datasets. The source code is available at https://github.com/deepinsight/insightface/tree/master/parsing/dml_csr.

----

## [406] Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin

**Authors**: *Hangyu Li, Nannan Wang, Xi Yang, Xiaoyu Wang, Xinbo Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00413](https://doi.org/10.1109/CVPR52688.2022.00413)

**Abstract**:

Only parts of unlabeled data are selected to train models for most semi-supervised learning methods, whose confidence scores are usually higher than the pre-defined threshold (i.e., the confidence margin). We argue that the recognition performance should be further improved by making full use of all unlabeled data. In this paper, we learn an Adaptive Confidence Margin (Ada-CM) to fully leverage all unlabeled data for semi-supervised deep facial expression recognition. All unlabeled samples are partitioned into two subsets by comparing their confidence scores with the adaptively learned confidence margin at each training epoch: (1) subset I including samples whose confidence scores are no lower than the margin; (2) subset II including samples whose confidence scores are lower than the margin. For samples in subset I, we constrain their predictions to match pseudo labels. Meanwhile, samples in subset II participate in the feature-level contrastive objective to learn effective facial expression features. We extensively evaluate Ada-CM on four challenging datasets, showing that our method achieves state-of-the-art performance, especially surpassing fully-supervised baselines in a semi-supervised manner. Ablation study further proves the effectiveness of our method. The source code is available at https://github.com/hangyu94/Ada-CM.

----

## [407] Towards Accurate Facial Landmark Detection via Cascaded Transformers

**Authors**: *Hui Li, Zidong Guo, Seon-Min Rhee, Seungju Han, Jae-Joon Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00414](https://doi.org/10.1109/CVPR52688.2022.00414)

**Abstract**:

Accurate facial landmarks are essential prerequisites for many tasks related to human faces. In this paper, an accurate facial landmark detector is proposed based on cascaded transformers. We formulate facial landmark detection as a coordinate regression task such that the model can be trained end-to-end. With self-attention in transformers, our model can inherently exploit the structured relationships between landmarks, which would benefit landmark detection under challenging conditions such as large pose and occlusion. During cascaded refinement, our model is able to extract the most relevant image features around the target landmark for coordinate prediction, based on deformable attention mechanism, thus bringing more accurate alignment. In addition, we propose a novel decoder that refines image features and landmark positions simultaneously. With few parameter increasing, the detection performance improves further. Our model achieves new state-of-the-art performance on several standard facial landmark detection benchmarks, and shows good generalization ability in cross-dataset evaluation.

----

## [408] PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer

**Authors**: *Zitong Yu, Yuming Shen, Jingang Shi, Hengshuang Zhao, Philip H. S. Torr, Guoying Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00415](https://doi.org/10.1109/CVPR52688.2022.00415)

**Abstract**:

Remote photoplethysmography (rPPG), which aims at measuring heart activities and physiological signals from facial video without any contact, has great potential in many applications. Recent deep learning approaches focus on mining subtle rPPG clues using convolutional neural networks with limited spatio-temporal receptive fields, which neglect the long-range spatio-temporal perception and interaction for rPPG modeling. In this paper, we propose the PhysFormer, an end-to-end video transformer based architecture, to adaptively aggregate both local and global spatio-temporal features for rPPG representation enhancement. As key modules in PhysFormer, the temporal difference transformers first enhance the quasi-periodic rPPG features with temporal difference guided global attention, and then refine the local spatio-temporal representation against interference. Furthermore, we also propose the label distribution learning and a curriculum learning inspired dynamic constraint in frequency domain, which provide elaborate supervisions for PhysFormer and alleviate overfitting. Comprehensive experiments are performed on four benchmark datasets to show our superior performance on both intra- and cross-dataset testings. One highlight is that, unlike most transformer networks needed pretraining from large-scale datasets, the proposed PhysFormer can be easily trained from scratch on rPPG datasets, which makes it promising as a novel transformer baseline for the rPPG community. The codes are available at https://github.com/ZitongYu/PhysFormer.

----

## [409] GazeOnce: Real-Time Multi-Person Gaze Estimation

**Authors**: *Mingfang Zhang, Yunfei Liu, Feng Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00416](https://doi.org/10.1109/CVPR52688.2022.00416)

**Abstract**:

Appearance-based gaze estimation aims to predict the 3D eye gaze direction from a single image. While recent deep learning-based approaches have demonstrated excellent performance, they usually assume one calibrated face in each input image and cannot output multi-person gaze in real time. However, simultaneous gaze estimation for multiple people in the wild is necessary for real-world applications. In this paper, we propose the first one-stage end-to-end gaze estimation method, GazeOnce, which is capable of simultaneously predicting gaze directions for multiple faces (> 10) in an image. In addition, we design a sophisticated data generation pipeline and propose a new dataset, MPSGaze, which contains full images of multiple people with 3D gaze ground truth. Experimental results demonstrate that our unified framework not only offers a faster speed, but also provides a lower gaze estimation error compared with state-of-the-art methods. This technique can be useful in real-time applications with multiple users.

----

## [410] Generalizing Gaze Estimation with Rotation Consistency

**Authors**: *Yiwei Bao, Yunfei Liu, Haofei Wang, Feng Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00417](https://doi.org/10.1109/CVPR52688.2022.00417)

**Abstract**:

Recent advances of deep learning-based approaches have achieved remarkable performance on appearance-based gaze estimation. However, due to the shortage of target domain data and absence of target labels, generalizing gaze estimation algorithm to unseen environments is still challenging. In this paper, we discover the rotation-consistency property in gaze estimation and introduce the ‘sub-label’ for unsupervised domain adaptation. Consequently, we propose the Rotation-enhanced Unsupervised Domain Adaptation (RUDA) for gaze estimation. First, we rotate the original images with different angles for training. Then we conduct domain adaptation under the constraint of rotation consistency. The target domain images are assigned with sub-labels, derived from relative rotation angles rather than untouchable real labels. With such sub-labels, we propose a novel distribution loss that facilitates the domain adaptation. We evaluate the RUDA framework on four cross-domain gaze estimation tasks. Experimental results demonstrate that it improves the performance over the baselines with gains ranging from 12.2% to 30.5%. Our framework has the potential to be used in other computer vision tasks with physical constraints.

----

## [411] Face Relighting with Geometrically Consistent Shadows

**Authors**: *Andrew Z. Hou, Michel Sarkis, Ning Bi, Yiying Tong, Xiaoming Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00418](https://doi.org/10.1109/CVPR52688.2022.00418)

**Abstract**:

Most face relighting methods are able to handle diffuse shadows, but struggle to handle hard shadows, such as those cast by the nose. Methods that propose techniques for handling hard shadows often do not produce geometrically consistent shadows since they do not directly leverage the estimated face geometry while synthesizing them. We propose a novel differentiable algorithm for synthesizing hard shadows based on ray tracing, which we incorporate into training our face relighting model. Our proposed algorithm directly utilizes the estimated face geometry to synthesize geometrically consistent hard shadows. We demonstrate through quantitative and qualitative experiments on Multi-PIE and FFHQ that our method produces more geometrically consistent shadows than previous face relighting methods while also achieving state-of-the-art face relighting performance under directional lighting. In addition, we demonstrate that our differentiable hard shadow modeling improves the quality of the estimated face geometry over diffuse shading models.

----

## [412] HairMapper: Removing Hair from Portraits Using GANs

**Authors**: *Yiqian Wu, Yong-Liang Yang, Xiaogang Jin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00419](https://doi.org/10.1109/CVPR52688.2022.00419)

**Abstract**:

Removing hair from portrait images is challenging due to the complex occlusions between hair and face, as well as the lack of paired portrait data with/without hair. To this end, we present a dataset and a baseline method for removing hair from portrait images using generative adversarial networks (GANs). Our core idea is to train a fully connected network HairMapper to find the direction of hair removal in the latent space of StyleGAN for the training stage. We develop a new separation boundary and diffuse method to generate paired training data for males, and a novel “female-male-bald” pipeline for paired data of females. Experiments show that our method can naturally deal with portrait images with variations on gender, age, etc. We validate the superior performance of our method by comparing it to state-of-the-art methods through extensive experiments and user studies. We also demonstrate its applications in hair design and 3D face reconstruction.

----

## [413] Learning to Restore 3D Face from In-the-Wild Degraded Images

**Authors**: *Zhenyu Zhang, Yanhao Ge, Ying Tai, Xiaoming Huang, Chengjie Wang, Hao Tang, Dongjin Huang, Zhifeng Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00420](https://doi.org/10.1109/CVPR52688.2022.00420)

**Abstract**:

In-the-wild 3D face modelling is a challenging problem as the predicted facial geometry and texture suffer from a lack of reliable clues or priors, when the input images are degraded. To address such a problem, in this paper we propose a novel Learning to Restore (L2R) 3D face framework for unsupervised high-quality face reconstruction from low-resolution images. Rather than directly refining 2D image appearance, L2R learns to recover fine-grained 3D details on the proxy against degradation via extracting generative facial priors. Concretely, L2R proposes a novel albedo restoration network to model high-quality 3D facial texture, in which the diverse guidance from the pre-trained Generative Adversarial Networks (GANs) is leveraged to complement the lack of input facial clues. With the finer details of the restored 3D texture, L2R then learns displacement maps from scratch to enhance the significant facial structure and geometry. Both of the procedures are mutually optimized with a novel 3D-aware adversarial loss, which further improves the modelling performance and suppresses the potential uncertainty. Extensive experiments on benchmarks show that L2R outperforms state-of-the-art methods under the condition of low-quality inputs, and obtains superior performances than 2D pre-processed modelling approaches with limited 3D proxy.

----

## [414] Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels

**Authors**: *Yuchao Wang, Haochen Wang, Yujun Shen, Jingjing Fei, Wei Li, Guoqiang Jin, Liwei Wu, Rui Zhao, Xinyi Le*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00421](https://doi.org/10.1109/CVPR52688.2022.00421)

**Abstract**:

The crux of semi-supervised semantic segmentation is to assign adequate pseudo-labels to the pixels of unlabeled images. A common practice is to select the highly confident predictions as the pseudo ground-truth, but it leads to a problem that most pixels may be left unused due to their unreliability. We argue that every pixel matters to the model training, even its prediction is ambiguous. Intuitively, an unreliable prediction may get confused among the top classes (i.e., those with the highest probabilities), however, it should be confident about the pixel not belonging to the remaining classes. Hence, such a pixel can be convincingly treated as a negative sample to those most unlikely categories. Based on this insight, we develop an effective pipeline to make sufficient use of unlabeled data. Concretely, we separate reliable and unreliable pixels via the entropy of predictions, push each unreliable pixel to a category-wise queue that consists of negative samples, and manage to train the model with all candidate pixels. Considering the training evolution, where the prediction becomes more and more accurate, we adaptively adjust the threshold for the reliable-unreliable partition. Experimental results on various benchmarks and training settings demonstrate the superiority of our approach over the state-of-the-art alternatives. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project: https://haochen-wang409.github.io/U2PL.

----

## [415] Perturbed and Strict Mean Teachers for Semi-supervised Semantic Segmentation

**Authors**: *Yuyuan Liu, Yu Tian, Yuanhong Chen, Fengbei Liu, Vasileios Belagiannis, Gustavo Carneiro*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00422](https://doi.org/10.1109/CVPR52688.2022.00422)

**Abstract**:

Consistency learning using input image, feature, or network perturbations has shown remarkable results in semi-supervised semantic segmentation, but this approach can be seriously affected by inaccurate predictions of unlabelled training images. There are two consequences of these inaccurate predictions: 1) the training based on the “strict” cross-entropy (CE) loss can easily overfit prediction mistakes, leading to confirmation bias; and 2) the perturbations applied to these inaccurate predictions will use potentially erroneous predictions as training signals, degrading consistency learning. In this paper, we address the prediction accuracy problem of consistency learning methods with novel extensions of the mean-teacher (MT) model, which include a new auxiliary teacher, and the replacement of MT's mean square error (MSE) by a stricter confidence-weighted cross-entropy (Conf-CE) loss. The accurate prediction by this model allows us to use a challenging combination of network, input data and feature perturbations to improve the consistency learning generalisation, where the feature perturbations consist of a new adversarial perturbation. Results on public benchmarks show that our approach achieves remarkable improvements over the previous SOTA methods in the field.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Supported by Australian Research Council through grants DP180103232 and FT190100525. Our code is available at https://github.com/yyliu01/PS-MT.

----

## [416] ST++: Make Self-trainingWork Better for Semi-supervised Semantic Segmentation

**Authors**: *Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, Yang Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00423](https://doi.org/10.1109/CVPR52688.2022.00423)

**Abstract**:

Self-training via pseudo labeling is a conventional, simple, and popular pipeline to leverage unlabeled data. In this work, we first construct a strong baseline of self-training (namely ST) for semi-supervised semantic segmentation via injecting strong data augmentations (SDA) on unlabeled images to alleviate overfitting noisy labels as well as decouple similar predictions between the teacher and student. With this simple mechanism, our ST outperforms all existing methods without any bells and whistles, e.g., iterative retraining. Inspired by the impressive results, we thoroughly investigate the SDA and provide some empirical analysis. Nevertheless, incorrect pseudo labels are still prone to accumulate and degrade the performance. To this end, we further propose an advanced self-training framework (namely ST++), that performs selective re-training via prioritizing reliable unlabeled images based on holistic prediction-level stability. Concretely, several model checkpoints are saved in the first stage supervised training, and the discrepancy of their predictions on the unlabeled image serves as a measurement for reliability. Our image-level selection offers holistic contextual information for learning. We demonstrate that it is more suitable for segmentation than common pixel-wise selection. As a result, ST+ further boosts the performance of our ST. Code is available at https://github.com/LiheYoung/ST-PlusPlus.

----

## [417] Beyond Semantic to Instance Segmentation: Weakly-Supervised Instance Segmentation via Semantic Knowledge Transfer and Self-Refinement

**Authors**: *Beomyoung Kim, Youngjoon Yoo, Chaeeun Rhee, Junmo Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00424](https://doi.org/10.1109/CVPR52688.2022.00424)

**Abstract**:

Weakly-supervised instance segmentation (WSIS) has been considered as a more challenging task than weakly-supervised semantic segmentation (WSSS). Compared to WSSS, WSIS requires instance-wise localization, which is difficult to extract from image-level labels. To tackle the problem, most WSIS approaches use off-the-shelf proposal techniques that require pre-training with instance or object level labels, deviating the fundamental definition of the fully-image-level supervised setting. In this paper, we propose a novel approach including two innovative components. First, we propose a semantic knowledge transfer to obtain pseudo instance labels by transferring the knowledge of WSSS to WSIS while eliminating the need for the off-the-shelf proposals. Second, we propose a self-refinement method to refine the pseudo instance labels in a self-supervised scheme and to use the refined labels for training in an online manner. Here, we discover an erroneous phenomenon, semantic drift, that occurred by the missing instances in pseudo instance labels categorized as background class. This semantic drift occurs confusion between background and instance in training and consequently degrades the segmentation performance. We term this problem as semantic drift problem and show that our proposed self-refinement method eliminates the semantic drift problem. The extensive experiments on PASCAL VOC 2012 and MS COCO demonstrate the effectiveness of our approach, and we achieve a considerable performance without off-the-shelf proposal techniques. The code is available at https://github.com/clovaai/BESTIE.

----

## [418] Self-supervised Image-specific Prototype Exploration for Weakly Supervised Semantic Segmentation

**Authors**: *Qi Chen, Lingxiao Yang, Jianhuang Lai, Xiaohua Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00425](https://doi.org/10.1109/CVPR52688.2022.00425)

**Abstract**:

Weakly Supervised Semantic Segmentation (WSSS) based on image-level labels has attracted much attention due to low annotation costs. Existing methods often rely on Class Activation Mapping (CAM) that measures the correlation between image pixels and classifier weight. However, the classifier focuses only on the discriminative regions while ignoring other useful information in each image, resulting in incomplete localization maps. To address this issue, we propose a Self-supervised Image-specific Prototype Exploration (SIPE) that consists of an Image-specific Prototype Exploration (IPE) and a General-Specific Consistency (GSC) loss. Specifically, IPE tailors prototypes for every image to capture complete regions, formed our Image-Specific CAM (IS-CAM), which is realized by two sequential steps. In addition, GSC is proposed to construct the consistency of general CAM and our specific IS-CAM, which further optimizes the feature representation and empowers a self-correction ability of prototype exploration. Extensive experiments are conducted on PASCAL VOC 2012 and MS COCO 2014 segmentation benchmark and results show our SIPE achieves new state-of-the-art performance using only image-level labels. The code is available at https://github.com/chenqi1126/SIPE.

----

## [419] Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation

**Authors**: *Tianfei Zhou, Meijie Zhang, Fang Zhao, Jianwu Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00426](https://doi.org/10.1109/CVPR52688.2022.00426)

**Abstract**:

Learning semantic segmentation from weakly-labeled (e.g., image tags only) data is challenging since it is hard to infer dense object regions from sparse semantic tags. Despite being broadly studied, most current efforts directly learn from limited semantic annotations carried by individual image or image pairs, and struggle to obtain integral localization maps. Our work alleviates this from a novel perspective, by exploring rich semantic contexts synergistically among abundant weakly-labeled training data for network learning and inference. In particular, we propose regional semantic contrast and aggregation (RCA). RCA is equipped with a regional memory bank to store massive, diverse object patterns appearing in training data, which acts as strong support for exploration of dataset-level semantic structure. Particularly, we propose i) semantic contrast to drive network learning by contrasting massive categorical object regions, leading to a more holistic object pattern understanding, and ii) semantic aggregation to gather diverse relational contexts in the memory to enrich semantic repre-sentations. In this manner, RCA earns a strong capability of fine-grained semantic understanding, and eventually establishes new state-of-the-art results on two popular benchmarks, i.e., PASCAL VOC 2012 and COCO 2014.

----

## [420] Multi-class Token Transformer for Weakly Supervised Semantic Segmentation

**Authors**: *Lian Xu, Wanli Ouyang, Mohammed Bennamoun, Farid Boussaïd, Dan Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00427](https://doi.org/10.1109/CVPR52688.2022.00427)

**Abstract**:

This paper proposes a new transformer-based framework to learn class-specific object localization maps as pseudo labels for weakly supervised semantic segmentation (WSSS). Inspired by the fact that the attended regions of the one-class token in the standard vision transformer can be leveraged to form a class-agnostic localization map, we investigate if the transformer model can also effectively capture class-specific attention for more discriminative object localization by learning multiple class tokens within the transformer. To this end, we propose a Multi-class Token Transformer, termed as MCTformer, which uses multiple class tokens to learn interactions between the class tokens and the patch tokens. The proposed MCTformer can successfully produce class-discriminative object localization maps from the class-to-patch attentions corresponding to different class tokens. We also propose to use a patch-level pairwise affinity, which is extracted from the patch-to-patch transformer attention, to further refine the localization maps. Moreover, the proposed framework is shown to fully complement the Class Activation Mapping (CAM) method, leading to remarkably superior WSSS results on the PASCAL VOC and MS COCO datasets. These results underline the importance of the class token for WSSS. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/xulianuwa/MCTformer

----

## [421] Weakly Supervised Semantic Segmentation by Pixel-to-Prototype Contrast

**Authors**: *Ye Du, Zehua Fu, Qingjie Liu, Yunhong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00428](https://doi.org/10.1109/CVPR52688.2022.00428)

**Abstract**:

Though image-level weakly supervised semantic seg-mentation (WSSS) has achieved great progress with Class Activation Maps (CAMs) as the cornerstone, the large su-pervision gap between classification and segmentation still hampers the model to generate more complete and precise pseudo masks for segmentation. In this study, we propose weakly-supervised pixel-to-prototype contrast that can provide pixel-level supervisory signals to narrow the gap. Guided by two intuitive priors, our method is executed across different views and within per single view of an image, aiming to impose cross-view feature semantic consistency regularization and facilitate intra(inter)-class compactness(dispersion) of the feature space. Our method can be seamlessly incorporated into existing WSSS models with-out any changes to the base networks and does not incur any extra inference burden. Extensive experiments manifest that our method consistently improves two strong baselines by large margins, demonstrating the effectiveness. Specifically, built on top of SEAM, we improve the initial seed mIoU on PASCAL VOC 2012 from 55.4% to 61.5%. Moreover, armed with our method, we increase the segmentation mIoU of EPS from 70.8% to 73.6%, achieving new state-of-the-art.

----

## [422] Threshold Matters in WSSS: Manipulating the Activation for the Robust and Accurate Segmentation Model Against Thresholds

**Authors**: *Minhyun Lee, Dongseob Kim, Hyunjung Shim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00429](https://doi.org/10.1109/CVPR52688.2022.00429)

**Abstract**:

Weakly-supervised semantic segmentation (WSSS) has recently gained much attention for its promise to train segmentation models only with image-level labels. Existing WSSS methods commonly argue that the sparse coverage of CAM incurs the performance bottleneck of WSSS. This paper provides analytical and empirical evidence that the actual bottleneck may not be sparse coverage but a global thresholding scheme applied after CAM. Then, we show that this issue can be mitigated by satisfying two conditions; 1) reducing the imbalance in the foreground activation and 2) increasing the gap between the foreground and the background activation. Based on these findings, we propose a novel activation manipulation network with a per-pixel classification loss and a label conditioning module. Per-pixel classification naturally induces two-level activation in activation maps, which can penalize the most discriminative parts, promote the less discriminative parts, and deactivate the background regions. Label conditioning imposes that the output label of pseudo-masks should be any of true image-level labels; it penalizes the wrong activation assigned to non-target classes. Based on extensive analysis and evaluations, we demonstrate that each component helps produce accurate pseudo-masks, achieving the robustness against the choice of the global threshold. Finally, our model achieves state-of-the-art records on both PAS-CAL VOC 2012 and MS COCO 2014 datasets. The code is available at https://github.com/gaviotas/AMN.

----

## [423] Novel Class Discovery in Semantic Segmentation

**Authors**: *Yuyang Zhao, Zhun Zhong, Nicu Sebe, Gim Hee Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00430](https://doi.org/10.1109/CVPR52688.2022.00430)

**Abstract**:

We introduce a new setting of Novel Class Discovery in Semantic Segmentation (NCDSS), which aims at segmenting unlabeled images containing new classes given prior knowledge from a labeled set of disjoint classes. In contrast to existing approaches that look at novel class dis-covery in image classification, we focus on the more chal-lenging semantic segmentation. In NCDSS, we need to dis-tinguish the objects and background, and to handle the existence of multiple classes within an image, which in-creases the difficulty in using the unlabeled data. To tackle this new setting, we leverage the labeled base data and a saliency model to coarsely cluster novel classes for model training in our basic framework. Additionally, we propose the Entropy-based Uncertainty Modeling and Self-training (EUMS) framework to overcome noisy pseudo-labels, fur-ther improving the model performance on the novel classes. Our EUMS utilizes an entropy ranking technique and a dy-namic reassignment to distill clean labels, thereby making full use of the noisy data via self-supervised learning. We build the NCDSS benchmark on the PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 dataset and COCO-20i dataset. Extensive experiments demonstrate the feasibility of the basic framework (achieving an average mIoU of 49.81% on PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
) and the effectiveness of EUMS framework (outperforming the basic framework by 9.28% mIoU on PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
).

----

## [424] Pin the Memory: Learning to Generalize Semantic Segmentation

**Authors**: *Jin Kim, Jiyoung Lee, Jungin Park, Dongbo Min, Kwanghoon Sohn*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00431](https://doi.org/10.1109/CVPR52688.2022.00431)

**Abstract**:

The rise of deep neural networks has led to several break-throughs for semantic segmentation. In spite of this, a model trained on source domain often fails to work properly in new challenging domains, that is directly concerned with the generalization capability of the model. In this paper, we present a novel memory-guided domain generalization method for semantic segmentation based on meta-learning framework. Especially, our method abstracts the conceptual knowledge of semantic classes into categorical memory which is constant beyond the domains. Upon the meta-learning concept, we repeatedly train memory-guided networks and simulate virtual test to 1) learn how to memorize a domain-agnostic and distinct information of classes and 2) offer an externally settled memory as a class-guidance to reduce the ambiguity of representation in the test data of arbitrary unseen domain. To this end, we also propose memory divergence and feature cohesion losses, which encourage to learn memory reading and update processes for category-aware domain generalization. Extensive experiments for semantic segmentation demonstrate the superior generalization capability of our method over state-of-the-art works on various benchmarks.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/Genie-Kim/PintheMemory

----

## [425] ISDNet: Integrating Shallow and Deep Networks for Efficient Ultra-high Resolution Segmentation

**Authors**: *Shaohua Guo, Liang Liu, Zhenye Gan, Yabiao Wang, Wuhao Zhang, Chengjie Wang, Guannan Jiang, Wei Zhang, Ran Yi, Lizhuang Ma, Ke Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00432](https://doi.org/10.1109/CVPR52688.2022.00432)

**Abstract**:

The huge burden of computation and memory are two obstacles in ultra-high resolution image segmentation. To tackle these issues, most of the previous works follow the global-local refinement pipeline, which pays more attention to the memory consumption but neglects the inference speed. In comparison to the pipeline that partitions the large image into small local regions, we focus on inferring the whole image directly. In this paper, we propose ISDNet, a novel ultra-high resolution segmentation framework that integrates the shallow and deep networks in a new manner, which significantly accelerates the inference speed while achieving accurate segmentation. To further exploit the relationship between the shallow and deep features, we propose a novel Relational-Aware feature Fusion module, which ensures high performance and robustness of our framework. Extensive experiments on Deepglobe, Inria Aerial, and Cityscapes datasets demonstrate our performance is consistently superior to state-of-the-arts. Specifically, it achieves 73.30 mIoU with a speed of 27.70 FPS on Deepglobe, which is more accurate and 172 × faster than the recent competitor. Code available at https://github.com/cedricgsh/ISDNet.

----

## [426] Incremental Learning in Semantic Segmentation from Image Labels

**Authors**: *Fabio Cermelli, Dario Fontanel, Antonio Tavera, Marco Ciccone, Barbara Caputo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00433](https://doi.org/10.1109/CVPR52688.2022.00433)

**Abstract**:

Although existing semantic segmentation approaches achieve impressive results, they still struggle to update their models incrementally as new categories are uncovered. Furthermore, pixel-by-pixel annotations are expensive and time-consuming. This paper proposes a novel framework for Weakly Incremental Learning for Semantic Segmentation, that aims at learning to segment new classes from cheap and largely available image-level labels. As opposed to existing approaches, that need to generate pseudolabels offline, we use a localizer, trained with image-level labels and regularized by the segmentation model, to obtain pseudo-supervision online and update the model incrementally. We cope with the inherent noise in the process by using soft-labels generated by the localizer. We demonstrate the effectiveness of our approach on the Pascal VOC and COCO datasets, outperforming offline weakly-supervised methods and obtaining results comparable with incremental learning methods with full supervision.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code can be found at https://github.com/fcd194/WILSON.

----

## [427] Instance Segmentation with Mask-supervised Polygonal Boundary Transformers

**Authors**: *Justin Lazarow, Weijian Xu, Zhuowen Tu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00434](https://doi.org/10.1109/CVPR52688.2022.00434)

**Abstract**:

In this paper, we present an end-to-end instance segmentation method that regresses a polygonal boundary for each object instance. This sparse, vectorized boundary representation for objects, while attractive in many downstream computer vision tasks, quickly runs into issues of parity that need to be addressed: parity in supervision and parity in performance when compared to existing pixel-based methods. This is due in part to object instances being annotated with ground-truth in the form of polygonal boundaries or segmentation masks, yet being evaluated in a convenient manner using only segmentation masks. Our method, BoundaryFormer, is a Transformer based architecture that directly predicts polygons yet uses instance mask segmentations as the ground-truth supervision for computing the loss. We achieve this by developing an end-to-end differentiable model that solely relies on supervision within the mask space through differentiable rasterization. Boundary-Former matches or surpasses the Mask R-CNN method in terms of instance segmentation quality on both COCO and Cityscapes while exhibiting significantly better transferability across datasets.

----

## [428] SharpContour: A Contour-based Boundary Refinement Approach for Efficient and Accurate Instance Segmentation

**Authors**: *Chenming Zhu, Xuanye Zhang, Yanran Li, Liangdong Qiu, Kai Han, Xiaoguang Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00435](https://doi.org/10.1109/CVPR52688.2022.00435)

**Abstract**:

Excellent performance has been achieved on instance segmentation but the quality on the boundary area remains unsatisfactory, which leads to a rising attention on boundary refinement. For practical use, an ideal post-processing refinement scheme are required to be accurate, generic and efficient. However, most of existing approaches propose pixel-wise refinement, which either introduce a massive computation cost or design specifically for different backbone models. Contour-based models are efficient and generic to be incorporated with any existing segmentation methods, but they often generate over-smoothed contour and tend to fail on corner areas. In this paper, we propose an efficient contour-based boundary refinement approach, named SharpContour, to tackle the segmentation of boundary area. We design a novel contour evolution process together with an Instance-aware Point Classifier. Our method deforms the contour iteratively by updating offsets in a discrete manner. Differing from existing contour evolution methods, SharpContour estimates each offset more independently so that it predicts much sharper and accurate contours. Notably, our method is generic to seamlessly work with diverse existing models with a small computational cost. Experiments show that SharpContour achieves competitive gains whilst preserving high efficiency.

----

## [429] Sparse Object-level Supervision for Instance Segmentation with Pixel Embeddings

**Authors**: *Adrian Wolny, Qin Yu, Constantin Pape, Anna Kreshuk*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00436](https://doi.org/10.1109/CVPR52688.2022.00436)

**Abstract**:

Most state-of-the-art instance segmentation methods have to be trained on densely annotated images. While difficult in general, this requirement is especially daunting for biomedical images, where domain expertise is often required for annotation and no large public data collections are available for pre-training. We propose to address the dense annotation bottleneck by introducing a proposal-free segmentation approach based on non-spatial embeddings, which exploits the structure of the learned embedding space to extract individual instances in a differentiable way. The segmentation loss can then be applied directly to instances and the overall pipeline can be trained in a fully-or weakly supervised manner. We consider the challenging case of positive-unlabeled supervision, where a novel self-supervised consistency loss is introduced for the unlabeled parts of the training data. We evaluate the proposed method on 2D and 3D segmentation problems in different microscopy modalities as well as on the Cityscapes and CVPPP instance segmentation benchmarks, achieving state-of-the-art results on the latter.

----

## [430] Mask Transfiner for High-Quality Instance Segmentation

**Authors**: *Lei Ke, Martin Danelljan, Xia Li, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00437](https://doi.org/10.1109/CVPR52688.2022.00437)

**Abstract**:

Two-stage and query-based instance segmentation methods have achieved remarkable results. However, their segmented masks are still very coarse. In this paper, we present Mask Transfiner for high-quality and efficient instance segmentation. Instead of operating on regular dense tensors, our Mask Transfiner decomposes and represents the image regions as a quadtree. Our transformer-based approach only processes detected error-prone tree nodes and self-corrects their errors in parallel. While these sparse pixels only constitute a small proportion of the total number, they are critical to the final mask quality. This allows Mask Transfiner to predict highly accurate instance masks, at a low computational cost. Extensive experiments demonstrate that Mask Transfiner outperforms current instance segmentation methods on three popular benchmarks, significantly improving both two-stage and query-based frameworks by a large margin of +3.0 mask AP on COCO and BDD100K, and +6.6 boundary AP on Cityscapes. Our code and trained models are available at https://github.com/SysCV/transfiner.

----

## [431] Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity

**Authors**: *Weiyao Wang, Matt Feiszli, Heng Wang, Jitendra Malik, Du Tran*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00438](https://doi.org/10.1109/CVPR52688.2022.00438)

**Abstract**:

Open-world instance segmentation is the task of grouping pixels into object instances without any pre-determined taxonomy. This is challenging, as state-of-the-art methods rely on explicit class semantics obtained from large labeled datasets, and out-of-domain evaluation performance drops significantly. Here we propose a novel approach for mask proposals, Generic Grouping Networks (GGNs), constructed without semantic supervision. Our approach combines a local measure of pixel affinity with instance-level mask supervision, producing a training regimen designed to make the model as generic as the data diversity allows. We introduce a method for predicting Pairwise Affinities (PA), a learned local relationship between pairs of pixels. PA generalizes very well to unseen categories. From PA we construct a large set of pseudo-ground-truth instance masks; combined with human-annotated instance masks we train GGNs and significantly outperform the SOTA on open-world instance segmentation on various benchmarks including COCO, LVIS, ADE20K, and UVO.

----

## [432] Sparse Instance Activation for Real-Time Instance Segmentation

**Authors**: *Tianheng Cheng, Xinggang Wang, Shaoyu Chen, Wenqiang Zhang, Qian Zhang, Chang Huang, Zhaoxiang Zhang, Wenyu Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00439](https://doi.org/10.1109/CVPR52688.2022.00439)

**Abstract**:

In this paper, we propose a conceptually novel, efficient, and fully convolutional framework for real-time instance segmentation. Previously, most instance segmentation methods heavily rely on object detection and perform mask prediction based on bounding boxes or dense centers. In contrast, we propose a sparse set of instance activation maps, as a new object representation, to high-light informative regions for each foreground object. Then instance-level features are obtained by aggregating features according to the highlighted regions for recognition and segmentation. Moreover, based on bipartite matching, the instance activation maps can predict objects in a one-to-one style, thus avoiding non-maximum suppression (NMS) in post-processing. Owing to the simple yet effective designs with instance activation maps, SparseInst has extremely fast inference speed and achieves 40 FPS and 37.9 AP on the COCO benchmark, which significantly out-performs the counterparts in terms of speed and accuracy. Code and models are available at https://github.com/hustvl/SparseInst.

----

## [433] E2EC: An End-to-End Contour-based Method for High-Quality High-Speed Instance Segmentation

**Authors**: *Tao Zhang, Shiqing Wei, Shunping Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00440](https://doi.org/10.1109/CVPR52688.2022.00440)

**Abstract**:

Contour-based instance segmentation methods have developed rapidly recently but feature rough and hand-crafted front-end contour initialization, which restricts the model performance, and an empirical and fixed backend predicted-label vertex pairing, which contributes to the learning difficulty. In this paper, we introduce a novel contour-based method, named E2EC, for high-quality instance segmentation. Firstly, E2EC applies a novel learnable contour initialization architecture instead of hand-crafted contour initialization. This consists of a contour initialization module for constructing more explicit learning goals and a global contour deformation module for taking advantage of all of the vertices' features better. Secondly, we propose a novel label sampling scheme, named multi-direction alignment, to reduce the learning difficulty. Thirdly, to improve the quality of the boundary details, we dynamically match the most appropriate predicted-ground truth vertex pairs and propose the corresponding loss function named dynamic matching loss. The experiments showed that E2EC can achieve a state-of-the-art performance on the KITTI INStance (KINS) dataset, the Semantic Boundaries Dataset (SBD), the Cityscapes and the COCO dataset. E2EC is also efficient for use in real-time applications, with an inference speed of 36 fps for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$512\times 512$</tex>
 images on an NVIDIA A6000 GPU. Code will be released at https://github.com/zhang-tao-whu/e2ec.

----

## [434] Hyperbolic Image Segmentation

**Authors**: *Mina Ghadimi Atigh, Julian Schoep, Erman Acar, Nanne van Noord, Pascal Mettes*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00441](https://doi.org/10.1109/CVPR52688.2022.00441)

**Abstract**:

For image segmentation, the current standard is to perform pixel-level optimization and inference in Euclidean output embedding spaces through linear hyperplanes. In this work, we show that hyperbolic manifolds provide a valuable alternative for image segmentation and propose a tractable formulation of hierarchical pixel-level classification in hyperbolic space. Hyperbolic Image Segmentation opens up new possibilities and practical benefits for segmentation, such as uncertainty estimation and boundary information for free, zero-label generalization, and increased performance in low-dimensional output embeddings.

----

## [435] SeeThroughNet: Resurrection of Auxiliary Loss by Preserving Class Probability Information

**Authors**: *Dasol Han, Jaewook Yoo, Dokwan Oh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00442](https://doi.org/10.1109/CVPR52688.2022.00442)

**Abstract**:

Auxiliary loss is additional loss besides the main branch loss to help optimize the learning process of neural networks. In order to calculate the auxiliary loss between the feature maps of intermediate layers and the ground truth in the field of semantic segmentation, the size of each feature map must match the ground truth. In all studies using the auxiliary losses with the segmentation models, from what we have investigated, they either use a down-sampling function to reduce the size of the ground truth or use an up-sampling function to increase the size of the feature map in order to match the resolution between the feature map and the ground truth. However, in the process of selecting representative values through down-sampling and up-sampling, information loss is inevitable. In this paper, we introduce Class Probability Preserving (CPP) pooling to alleviate information loss in down-sampling the ground truth in semantic segmentation tasks. We demonstrated the superiority of the proposed method on Cityscapes, Pascal VOC, Pascal Context, and NYU-Depth-v2 datasets by using CPP pooling with auxiliary losses based on seven popular segmentation models. In addition, we propose See-Through Network (SeeThroughNet) that adopts an improved multi-scale attention-coupled decoder structure to maximize the effect of CPP pooling. SeeThroughNet shows cutting-edge results in the field of semantic understanding of urban street scenes, which ranked #1 on the Cityscapes benchmark.

----

## [436] CDGNet: Class Distribution Guided Network for Human Parsing

**Authors**: *Kunliang Liu, Ouk Choi, Jianming Wang, Wonjun Hwang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00443](https://doi.org/10.1109/CVPR52688.2022.00443)

**Abstract**:

The objective of human parsing is to partition a human in an image into constituent parts. This task involves labeling each pixel of the human image according to the classes. Since the human body comprises hierarchically structured parts, each body part of an image can have its sole position distribution characteristic. Probably, a human head is less likely to be under the feet, and arms are more likely to be near the torso. Inspired by this observation, we make instance class distributions by accumulating the original human parsing label in the horizontal and vertical directions, which can be utilized as supervision signals. Using these horizontal and vertical class distribution labels, the network is guided to exploit the intrinsic position distribution of each class. We combine two guided features to form a spatial guidance map, which is then superimposed onto the baseline network by multiplication and concatenation to distinguish the human parts precisely. We conducted extensive experiments to demonstrate the effectiveness and superiority of our method on three well-known benchmarks: LIP, ATR, and CIHP databases.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
Our code is available at https://github.com/tjpulkl/CDGNet.

----

## [437] CLIMS: Cross Language Image Matching for Weakly Supervised Semantic Segmentation

**Authors**: *Jinheng Xie, Xianxu Hou, Kai Ye, Linlin Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00444](https://doi.org/10.1109/CVPR52688.2022.00444)

**Abstract**:

It has been widely known that CAM (Class Activation Map) usually only activates discriminative object regions and falsely includes lots of object-related backgrounds. As only a fixed set of image-level object labels are available to the WSSS (weakly supervised semantic segmentation) model, it could be very difficult to suppress those diverse background regions consisting of open set objects. In this paper, we propose a novel Cross Language Image Matching (CLIMS) framework, based on the recently introduced Contrastive Language-Image Pre-training (CLIP) model, for WSSS. The core idea of our framework is to introduce natural language supervision to activate more complete object regions and suppress closely-related open background regions. In particular, we design object, background region and text label matching losses to guide the model to excite more reasonable object regions for CAM of each category. In addition, we design a co-occurring background suppression loss to prevent the model from activating closely-related background regions, with a predefined set of class-related background text descriptions. These designs enable the proposed CLIMS to generate a more complete and compact activation map for the target objects. Extensive experiments on PASCAL VOC2012 dataset show that our CLIMS significantly outperforms the previous state-of-the-art methods. Code will be available at https://github.com/CVI-SZU/CLIMS.

----

## [438] Sparse Non-local CRF

**Authors**: *Olga Veksler, Yuri Boykov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00445](https://doi.org/10.1109/CVPR52688.2022.00445)

**Abstract**:

CRF is a classical computer vision model which is also useful for deep learning. There are two common CRF types: sparse and dense. Sparse CRF connects only the nearby pixels, while dense CRF has global connectivity. Therefore dense CRF is a more general model, but it is much harder to optimize compared to sparse CRF. In fact, only a certain form of dense CRF is optimized in practice, and even then approximately. We propose a new sparse non-local CRF: it has a sparse number of connections, but it has both local and non-local ones. Like sparse CRF, the total number of connections is small, and our model is easy to optimize exactly. Like dense CRF, our model is more general than sparse CRF due to non-local connections. We show that our sparse non-local CRF can model properties similar to that of the popular Gaussian edge dense CRF. Besides efficiency, another advantage is that our edge weights are less restricted compared to Gaussian edge dense CRF. We design models that take advantage of this flexibility. We also discuss connection of our model to other CRF models. Finally, to prove the usefulness of our model, we evaluate it on the classical application of segmentation from a bounding box and for deep learning based salient object segmentation. We improve state of the art for both applications.

----

## [439] Detecting Camouflaged Object in Frequency Domain

**Authors**: *Yijie Zhong, Bo Li, Lv Tang, Senyun Kuang, Shuang Wu, Shouhong Ding*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00446](https://doi.org/10.1109/CVPR52688.2022.00446)

**Abstract**:

Camouflaged object detection (COD) aims to identify objects that are perfectly embedded in their environment, which has various downstream applications in fields such as medicine, art, and agriculture. However, it is an extremely challenging task to spot camouflaged objects with the perception ability of human eyes. Hence, we claim that the goal of COD task is not just to mimic the human visual ability in a single RGB domain, but to go beyond the human biological vision. We then introduce the frequency domain as an additional clue to better detect camouflaged objects from backgrounds. To well involve the frequency clues into the CNN models, we present a powerful network with two special components. We first design a novel frequency enhancement module (FEM) to dig clues of camouflaged objects in the frequency domain. It contains the offline discrete cosine transform followed by the learnable enhancement. Then we use a feature alignment to fuse the features from RGB domain and frequency domain. Moreover, to further make full use of the frequency information, we propose the high-order relation module (HOR) to handle the rich fusion feature. Comprehensive experiments on three widely-used COD datasets show the proposed method significantly outperforms other state-of-the-art methods by a large margin.

----

## [440] Progressive Minimal Path Method with Embedded CNN

**Authors**: *Wei Liao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00447](https://doi.org/10.1109/CVPR52688.2022.00447)

**Abstract**:

We propose Path-CNN, a method for the segmentation of centerlines of tubular structures by embedding convolutional neural networks (CNNs) into the progressive minimal path method. Minimal path methods are widely used for topology-aware centerline segmentation, but usually these methods rely on weak, hand-tuned image features. In contrast, CNNs use strong image features which are learned automatically from images. But CNNs usually do not take the topology of the results into account, and often require a large amount of annotations for training. We integrate CNNs into the minimal path method, so that both techniques benefit from each other: CNNs employ learned image features to improve the determination of minimal paths, while the minimal path method ensures the correct topology of the segmented centerlines, provides strong geometric priors to increase the performance of CNNs, and reduces the amount of annotations for the training of CNNs significantly. Our method has lower hardware requirements than many recent methods. Qualitative and quantitative comparison with other methods shows that Path-CNN achieves better performance, especially when dealing with tubular structures with complex shapes in challenging environments.

----

## [441] Open-Set Text Recognition via Character-Context Decoupling

**Authors**: *Chang Liu, Chun Yang, Xu-Cheng Yin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00448](https://doi.org/10.1109/CVPR52688.2022.00448)

**Abstract**:

The open-set text recognition task is an emerging chal-lenge that requires an extra capability to cognize novel characters during evaluation. We argue that a major cause of the limited performance for current methods is the con-founding effect of contextual information over the visual information of individual characters. Under open-set sce-narios, the intractable bias in contextual information can be passed down to visual information, consequently im-pairing the classification performance. In this paper, a Character-Context Decoupling framework is proposed to alleviate this problem by separating contextual information and character-visual information. Contextual information can be decomposed into temporal information and lin-guistic information. Here, temporal information that mod-els character order and word length is isolated with a de-tached temporal attention module. Linguistic information that models n- gram and other linguistic statistics is sepa-rated with a decoupled context anchor mechanism. A va-riety of quantitative and qualitative experiments show that our method achieves promising performance on open-set, zero-shot, and close-set text recognition datasets.

----

## [442] Neural Collaborative Graph Machines for Table Structure Recognition

**Authors**: *Hao Liu, Xin Li, Bing Liu, Deqiang Jiang, Yinsong Liu, Bo Ren*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00449](https://doi.org/10.1109/CVPR52688.2022.00449)

**Abstract**:

Recently, table structure recognition has achieved impressive progress with the help of deep graph models. Most of them exploit single visual cues of tabular elements or simply combine visual cues with other modalities via early fusion to reason their graph relationships. However, neither early fusion nor individually reasoning in terms of multiple modalities can be appropriate for all varieties of table structures with great diversity. Instead, different modalities are expected to collaborate with each other in different patterns for different table cases. In the community, the importance of intrainter modality interactions for table structure reasoning is still unexplored. In this paper, we define it as heterogeneous table structure recognition (HeteroTSR) problem. With the aim offilling this gap, we present a novel Neural Collaborative Graph Machines (NCGM) equipped with stacked collaborative blocks, which alternatively extracts intramodality context and models inter-modality interactions in a hierarchical way. It can represent the intrainter modality relationships of tabular elements more robustly, which significantly improves the recognition performance. We also show that the proposed NCGM can modulate collaborative pattern of different modalities conditioned on the context of intramodality cues, which is vital for diversified table cases. Experimental results on benchmarks demonstrate our proposed NCGM achieves state-of-the-art performance and beats other contemporary methods by a large margin especially under challenging scenarios.

----

## [443] Revisiting Document Image Dewarping by Grid Regularization

**Authors**: *Xiangwei Jiang, Rujiao Long, Nan Xue, Zhibo Yang, Cong Yao, Gui-Song Xia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00450](https://doi.org/10.1109/CVPR52688.2022.00450)

**Abstract**:

This paper addresses the problem of document image dewarping, which aims at eliminating the geometric distortion in document images for document digitization. Instead of designing a better neural network to approximate the optical flow fields between the inputs and outputs, we pursue the best readability by taking the text lines and the document boundaries into account from a constrained optimization perspective. Specifically, our proposed method first learns the boundary points and the pixels in the text lines and then follows the most simple observation that the boundaries and text lines in both horizontal and vertical directions should be kept after dewarping to introduce a novel grid regularization scheme. To obtain the final forward mapping for dewarping, we solve an optimization problem with our proposed grid regularization. The experiments comprehensively demonstrate that our proposed approach outperforms the prior arts by large margins in terms of readability (with the metrics of Character Errors Rate and the Edit Distance) while maintaining the best image quality on the publicly-available DocUNet benchmark.

----

## [444] Syntax-Aware Network for Handwritten Mathematical Expression Recognition

**Authors**: *Ye Yuan, Xiao Liu, Wondimu Dikubab, Hui Liu, Zhilong Ji, Zhongqin Wu, Xiang Bai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00451](https://doi.org/10.1109/CVPR52688.2022.00451)

**Abstract**:

Handwritten mathematical expression recognition (HMER) is a challenging task that has many potential applications. Recent methods for HMER have achieved outstanding performance with an encoder-decoder architecture. However, these methods adhere to the paradigm that the prediction is made “from one character to another”, which inevitably yields prediction errors due to the complicated structures of mathematical expressions or crabbed handwritings. In this paper, we propose a simple and efficient method for HMER, which is the first to incorporate syntax information into an encoder-decoder network. Specifically, we present a set of grammar rules for converting the LaTeX markup sequence of each expression into a parsing tree; then, we model the markup sequence prediction as a tree traverse process with a deep neural network. In this way, the proposed method can effectively describe the syntax context of expressions, alleviating the structure prediction errors of HMER. Experiments on three benchmark datasets demonstrate that our method achieves better recognition performance than prior arts. To further validate the effectiveness of our method, we create a large-scale dataset consisting of 100k handwritten mathematical expression images acquired from ten thousand writers. The source code, new dataset
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
https://ai.100tal.com/dataset, and pre-trained models of this work will be publicly available.

----

## [445] Few Could Be Better Than All: Feature Sampling and Grouping for Scene Text Detection

**Authors**: *Jingqun Tang, Wenqing Zhang, Hongye Liu, Mingkun Yang, Bo Jiang, Guanglong Hu, Xiang Bai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00452](https://doi.org/10.1109/CVPR52688.2022.00452)

**Abstract**:

Recently, transformer-based methods have achieved promising progresses in object detection, as they can eliminate the post-processes like NMS and enrich the deep representations. However, these methods cannot well cope with scene text due to its extreme variance of scales and aspect ratios. In this paper, we present a simple yet effective transformer-based architecture for scene text detection. Different from previous approaches that learn robust deep representations of scene text in a holistic manner, our method performs scene text detection based on a few representative features, which avoids the disturbance by background and reduces the computational cost. Specifically, we first select a few representative features at all scales that are highly relevant to foreground text. Then, we adopt a transformer for modeling the relationship of the sampled features, which effectively divides them into reasonable groups. As each feature group corresponds to a text instance, its bounding box can be easily obtained without any post-processing operation. Using the basic feature pyramid network for feature extraction, our method consistently achieves state-of-the-art results on several popular datasets for scene text detection.

----

## [446] Fourier Document Restoration for Robust Document Dewarping and Recognition

**Authors**: *Chuhui Xue, Zichen Tian, Fangneng Zhan, Shijian Lu, Song Bai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00453](https://doi.org/10.1109/CVPR52688.2022.00453)

**Abstract**:

State-of-the-art document dewarping techniques learn to predict 3-dimensional information of documents which are prone to errors while dealing with documents with irregular distortions or large variations in depth. This paper presents FDRNet, a Fourier Document Restoration Network that can restore documents with different distortions and improve document recognition in a reliable and simpler manner. FDRNet focuses on high-frequency components in the Fourier space that capture most structural information but are largely free of degradation in appearance. It dewarps documents by a flexible Thin-Plate Spline transformation which can handle various deformations effectively without requiring deformation annotations in training. These features allow FDRNet to learn from a small amount of simply labeled training images, and the learned model can dewarp documents with complex geometric distortion and recognize the restored texts accurately. To facilitate document restoration research, we create a benchmark dataset consisting of over one thousand camera documents with different types of geometric and photometric distortion. Extensive experiments show that FDRNet outperforms the state-of-the-art by large margins on both dewarping and text recognition tasks. In addition, FDRNet requires a small amount of simply labeled training data and is easy to deploy. The proposed dataset is available at https://sg-vilab.github.io/event/warpdoc/.

----

## [447] XYLayoutLM: Towards Layout-Aware Multimodal Networks For Visually-Rich Document Understanding

**Authors**: *Zhangxuan Gu, Changhua Meng, Ke Wang, Jun Lan, Weiqiang Wang, Ming Gu, Liqing Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00454](https://doi.org/10.1109/CVPR52688.2022.00454)

**Abstract**:

Recently, various multimodal networks for Visually-Rich Document Understanding(VRDU) have been proposed, showing the promotion of transformers by integrating visual and layout information with the text embeddings. However, most existing approaches utilize the position embeddings to incorporate the sequence information, neglecting the noisy improper reading order obtained by OCR tools. In this paper, we propose a robust layout-aware multimodal network named XYLayoutLM to capture and leverage rich layout information from proper reading orders produced by our Augmented XY Cut. Moreover, a Dilated Conditional Position Encoding module is proposed to deal with the input sequence of variable lengths, and it additionally extracts local layout information from both textual and vi-sual modalities while generating position embeddings. Experiment results show that our XYLayoutLM achieves competitive results on document understanding tasks.

----

## [448] SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition

**Authors**: *Mingxin Huang, Yuliang Liu, Zhenghao Peng, Chongyu Liu, Dahua Lin, Shenggao Zhu, Nicholas Jing Yuan, Kai Ding, Lianwen Jin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00455](https://doi.org/10.1109/CVPR52688.2022.00455)

**Abstract**:

End-to-end scene text spotting has attracted great attention in recent years due to the success of excavating the intrinsic synergy of the scene text detection and recognition. However, recent state-of-the-art methods usually incorporate detection and recognition simply by sharing the backbone, which does not directly take advantage of the feature interaction between the two tasks. In this paper, we propose a new end-to-end scene text spotting framework termed SwinTextSpotter. Using a transformer encoder with dynamic head as the detector, we unify the two tasks with a novel Recognition Conversion mechanism to explicitly guide text localization through recognition loss. The straightforward design results in a concise framework that requires neither additional rectification module nor character-level annotation for the arbitrarily-shaped text. Qualitative and quantitative experiments on multi-oriented datasets RoIC13 and ICDAR 2015, arbitrarily-shaped datasets Total-Text and CTW1500, and multi-lingual datasets ReCTS (Chinese) and VinText (Viet-namese) demonstrate SwinTextSpotter significantly outperforms existing methods. Code is available at https://github.com/mxin262/SwinTextSpotter.

----

## [449] Towards Weakly-Supervised Text Spotting using a Multi-Task Transformer

**Authors**: *Yair Kittenplon, Inbal Lavi, Sharon Fogel, Yarin Bar, R. Manmatha, Pietro Perona*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00456](https://doi.org/10.1109/CVPR52688.2022.00456)

**Abstract**:

Text spotting end-to-end methods have recently gained attention in the literature due to the benefits of jointly optimizing the text detection and recognition components. Existing methods usually have a distinct separation between the detection and recognition branches, requiring exact annotations for the two tasks. We introduce TextTranSpotter (TTS), a transformer-based approach for text spotting and the first text spotting framework which may be trained with both fully- and weakly-supervised settings. By learning a single latent representation per word detection, and using a novel loss function based on the Hungarian loss, our method alleviates the need for expensive localization annotations. Trained with only text transcription annotations on real data, our weakly-supervised method achieves competitive performance with previous state-of-the-art fully-supervised methods. When trained in a fully-supervised manner, TextTranSpotter shows state-of-the-art results on multiple benchmarks.

----

## [450] TableFormer: Table Structure Understanding with Transformers

**Authors**: *Ahmed S. Nassar, Nikolaos Livathinos, Maksym Lysak, Peter W. J. Staar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00457](https://doi.org/10.1109/CVPR52688.2022.00457)

**Abstract**:

Tables organize valuable content in a concise and compact representation. This content is extremely valuable for systems such as search engines, Knowledge Graph's, etc, since they enhance their predictive capabilities. Unfortu-nately, tables come in a large variety of shapes and sizes. Furthermore, they can have complex column/row-header configurations, multiline rows, different variety of separation lines, missing entries, etc. As such, the correct iden-tification of the table-structure from an image is a nontrivial task. In this paper, we present a new table-structure identification model. The latter improves the latest end-to-end deep learning model (i.e. encoder-dual-decoder from PubTabNet) in two significant ways. First, we introduce a new object detection decoder for table-cells. In this way, we can obtain the content of the table-cells from program-matic PDF's directly from the PDF source and avoid the training of the custom OCR decoders. This architectural change leads to more accurate table-content extraction and allows us to tackle non-english tables. Second, we replace the LSTM decoders with transformer based decoders. This upgrade improves significantly the previous state-of-the-art tree-editing-distance-score (TEDS) from 91% to 98.5% on simple tables and from 88.7% to 95% on complex tables.

----

## [451] Knowledge Mining with Scene Text for Fine-Grained Recognition

**Authors**: *Hao Wang, Junchao Liao, Tianheng Cheng, Zewen Gao, Hao Liu, Bo Ren, Xiang Bai, Wenyu Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00458](https://doi.org/10.1109/CVPR52688.2022.00458)

**Abstract**:

Recently, the semantics of scene text has been proven to be essential in fine-grained image classification. However, the existing methods mainly exploit the literal meaning of scene text for fine-grained recognition, which might be irrelevant when it is not significantly related to objects/scenes. We propose an end-to-end trainable network that mines implicit contextual knowledge behind scene text image and enhance the semantics and correlation to fine-tune the image representation. Unlike the existing methods, our model integrates three modalities: visual feature extraction, text semantics extraction, and correlating background knowledge to fine-grained image classification. Specifically, we employ KnowBert to retrieve relevant knowledge for semantic representation and combine it with image features for fine-grained classification. Experiments on two benchmark datasets, Con-Text, and Drink Bottle, show that our method outperforms the state-of-the-art by 3.72% mAP and 5.39% mAp, respectively. To further validate the effectiveness of the proposed method, we create a new dataset on crowd activity recognition for the evaluation. The source code and new dataset of this work are available at this repository
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/lanfeng4659/KnowledgeMiningWithSceneText.

----

## [452] PubTables-1M: Towards comprehensive table extraction from unstructured documents

**Authors**: *Brandon Smock, Rohith Pesala, Robin Abraham*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00459](https://doi.org/10.1109/CVPR52688.2022.00459)

**Abstract**:

Recently, significant progress has been made applying machine learning to the problem of table structure inference and extraction from unstructured documents. However, one of the greatest challenges remains the creation of datasets with complete, unambiguous ground truth at scale. To address this, we develop a new, more comprehensive dataset for table extraction, called PubTables-1M. PubTables-1M contains nearly one million tables from scientific articles, supports multiple input modalities, and contains detailed header and location information for table structures, making it useful for a wide variety of modeling approaches. It also addresses a significant source of ground truth inconsistency observed in prior datasets called oversegmentation, using a novel canonicalization procedure. We demonstrate that these improvements lead to a significant increase in training performance and a more reliable estimate of model performance at evaluation for table structure recognition. Further, we show that transformer-based object detection models trained on PubTables-1M produce excellent results for all three tasks of detection, structure recognition, and functional analysis without the need for any special customization for these tasks. Data and code will be released at https://github.com/microsoft/table-transformer.

----

## [453] Focal and Global Knowledge Distillation for Detectors

**Authors**: *Zhendong Yang, Zhe Li, Xiaohu Jiang, Yuan Gong, Zehuan Yuan, Danpei Zhao, Chun Yuan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00460](https://doi.org/10.1109/CVPR52688.2022.00460)

**Abstract**:

Knowledge distillation has been applied to image classification successfully. However, object detection is much more sophisticated and most knowledge distillation methods have failed on it. In this paper, we point out that in object detection, the features of the teacher and student vary greatly in different areas, especially in the foreground and background. If we distill them equally, the uneven differences between feature maps will negatively affect the distillation. Thus, we propose Focal and Global Distillation (FGD). Focal distillation separates the foreground and background, forcing the student to focus on the teacher's critical pixels and channels. Global distillation rebuilds the relation between different pixels and transfers it from teachers to students, compensating for missing global information in focal distillation. As our method only needs to calculate the loss on the feature map, FGD can be applied to various detectors. We experiment on various detectors with different backbones and the results show that the student detector achieves excellent mAP improvement. For example, ResNet-50 based RetinaNet, Faster RCNN, RepPoints and Mask RCNN with our distillation method achieve 40.7%, 42.0%, 42.0% and 42.1% mAP on COCO2017, which are 3.3, 3.6, 3.4 and 2.9 higher than the baseline, respectively. Our codes are available at https://github.com/yzd-v/FGD.

----

## [454] Speed up Object Detection on Gigapixel-level Images with Patch Arrangement

**Authors**: *Jiahao Fan, Huabin Liu, Wenjie Yang, John See, Aixin Zhang, Weiyao Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00461](https://doi.org/10.1109/CVPR52688.2022.00461)

**Abstract**:

With the appearance of super high-resolution (e.g., gigapixel-level) images, performing efficient object detection on such images becomes an important issue. Most ex-isting works for efficient object detection on high-resolution images focus on generating local patches where objects may exist, and then every patch is detected independently. How-ever, when the image resolution reaches gigapixel-level, they will suffer from a huge time cost for detecting numerous patches. Different from them, we devise a novel patch ar-rangement frameworkfor fast object detection on gigapixel-level images. Under this framework, a Patch Arrangement Network (PAN) is proposed to accelerate the detection by determining which patches could be packed together into a compact canvas. Specifically, PAN consists of (1) a Patch Filter Module (PFM) (2) a Patch Packing Module (PPM). PFM filters patch candidates by learning to select patches between two granularities. Subsequently, from the remaining patches, PPM determines how to pack these patches to-gether into a smaller number of canvases. Meanwhile, it generates an ideal layout of patches on canvas. These can-vases are fed to the detector to get final results. Experiments show that our method could improve the inference speed on gigapixel-level images by 5 x while maintaining great performance.

----

## [455] Training Object Detectors from Scratch: An Empirical Study in the Era of Vision Transformer

**Authors**: *Weixiang Hong, Jiangwei Lao, Wang Ren, Jian Wang, Jingdong Chen, Wei Chu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00462](https://doi.org/10.1109/CVPR52688.2022.00462)

**Abstract**:

Modeling in computer vision has long been dominated by convolutional neural networks (CNNs). Recently, in light of the excellent performances of self-attention mech-anism in the language field, transformers tailored for visual data have drawn numerous attention and triumphed CNNs in various vision tasks. These vision transformers heavily rely on large-scale pre-training to achieve competitive accuracy, which not only hinders the freedom of architectural design in downstream tasks like object detection, but also causes learning bias and domain mismatch in the fine-tuning stages. To this end, we aim to get rid of the “pre-train & fine-tune” paradigm of vision transformer and train transformer based object detector from scratch. Some earlier work in the CNNs era have successfully trained CNNs based detectors without pre-training, unfortunately, their findings do not generalize well when the backbone is switched from CNNs to vision transformer. Instead of proposing a specific vision transformer based detector, in this work, our goal is to reveal the insights of training vision transformer based detectors from scratch. In particular, we expect those insights can help other re-searchers and practitioners, and inspire more interesting research in other fields, such as semantic segmentation, visual-linguistic pre-training, etc. One of the key findings is that both architectural changes and more epochs play critical roles in training vision transformer based detectors from scratch. Experiments on MS COCO datasets demonstrate that vision transformer based detectors trained from scratch can also achieve similar performances to their counterparts with ImageNet pre-training.

----

## [456] Learning with Neighbor Consistency for Noisy Labels

**Authors**: *Ahmet Iscen, Jack Valmadre, Anurag Arnab, Cordelia Schmid*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00463](https://doi.org/10.1109/CVPR52688.2022.00463)

**Abstract**:

Recent advances in deep learning have relied on large, labelled datasets to train high-capacity models. However, collecting large datasets in a time- and cost-efficient manner often results in label noise. We present a method for learning from noisy labels that leverages similarities between training examples in feature space, encouraging the prediction of each example to be similar to its nearest neighbours. Compared to training algorithms that use multiple models or distinct stages, our approach takes the form of a simple, additional regularization term. It can be interpreted as an inductive version of the classical, transductive label propagation algorithm. We thoroughly evaluate our method on datasets evaluating both synthetic (CIFAR-10, CIFAR-100) and realistic (mini-WebVision, WebVision, Clothing1M, mini-ImageNet-Red) noise, and achieve competitive or state-of-the-art accuracies across all of them.

----

## [457] Meta Convolutional Neural Networks for Single Domain Generalization

**Authors**: *Chaoqun Wan, Xu Shen, Yonggang Zhang, Zhiheng Yin, Xinmei Tian, Feng Gao, Jianqiang Huang, Xian-Sheng Hua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00464](https://doi.org/10.1109/CVPR52688.2022.00464)

**Abstract**:

In single domain generalization, models trained with data from only one domain are required to perform well on many unseen domains. In this paper, we propose a new model, termed meta convolutional neural network, to solve the single domain generalization problem in image recognition. The key idea is to decompose the convolutional features of images into meta features. Acting as “visual words”, meta features are defined as universal and basic visual elements for image representations (like words for documents in language). Taking meta features as reference, we propose compositional operations to eliminate irrelevant features of local convolutional features by an addressing process and then to reformulate the convolutional feature maps as a composition of related meta features. In this way, images are universally coded without biased information from the unseen domain, which can be processed by following modules trained in the source domain. The compositional operations adopt a regression analysis technique to learn the meta features in an online batch learning manner. Extensive experiments on multiple benchmark datasets verify the superiority of the proposed model in improving single domain generalization ability.

----

## [458] Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification

**Authors**: *Haowei Zhu, Wenjing Ke, Dong Li, Ji Liu, Lu Tian, Yi Shan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00465](https://doi.org/10.1109/CVPR52688.2022.00465)

**Abstract**:

Recently, self-attention mechanisms have shown impressive performance in various NLP and CV tasks, which can help capture sequential characteristics and derive global information. In this work, we explore how to extend self-attention modules to better learn subtle feature embeddings for recognizing fine-grained objects, e.g., different bird species or person identities. To this end, we propose a dual cross-attention learning (DCAL) algorithm to coordinate with self-attention learning. First, we propose global-local cross-attention (GLCA) to enhance the interactions between global images and local high-response regions, which can help reinforce the spatial-wise discriminative clues for recognition. Second, we propose pairwise cross-attention (PWCA) to establish the interactions between image pairs. PWCA can regularize the attention learning of an image by treating another image as distractor and will be removed during inference. We observe that DCAL can reduce misleading attentions and diffuse the attention response to discover more complementary parts for recognition. We conduct extensive evaluations on finegrained visual categorization and object reidentification. Experiments demonstrate that DCAL performs on par with state-of-the-art methods and consistently improves multiple self-attention baselines, e.g., surpassing DeiT-Tiny and ViT-Base by 2.8% and 2.4% mAP on MSMT17, respectively.

----

## [459] Geometry-Aware Guided Loss for Deep Crack Recognition

**Authors**: *Zhuangzhuang Chen, Jin Zhang, Zhuonan Lai, Jie Chen, Zun Liu, Jianqiang Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00466](https://doi.org/10.1109/CVPR52688.2022.00466)

**Abstract**:

Despite the substantial progress of deep models for crack recognition, due to the inconsistent cracks in varying sizes, shapes, and noisy background textures, there still lacks the discriminative power of the deeply learned features when supervised by the cross-entropy loss. In this paper, we propose the geometry-aware guided loss (GAGL) that enhances the discrimination ability and is only applied in the training stage without extra computation and memory during inference. The GAGL consists of the feature-based geometry-aware projected gradient descent method (FGA-PGD) that approximates the geometric distances of the features to the class boundaries, and the geometry-aware update rule that learns an anchor of each class as the approximation of the feature expected to have the largest geometric distance to the corresponding class boundary. Then the discriminative power can be enhanced by minimizing the distances between the features and their corresponding class anchors in the feature space. To address the limited availability of related benchmarks, we collect a fully annotated dataset, namely, NPP2021, which involves inconsistent cracks and noisy backgrounds in real-world nuclear power plants. Our proposed GAGL outperforms the state of the arts on various benchmark datasets including CRACK2019, SDNET2018, and our NPP2021.

----

## [460] Segment, Magnify and Reiterate: Detecting Camouflaged Objects the Hard Way

**Authors**: *Qi Jia, Shuilian Yao, Yu Liu, Xin Fan, Risheng Liu, Zhongxuan Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00467](https://doi.org/10.1109/CVPR52688.2022.00467)

**Abstract**:

It is challenging to accurately detect camouflaged objects from their highly similar surroundings. Existing methods mainly leverage a single-stage detection fashion, while neglecting small objects with low-resolution fine edges requires more operations than the larger ones. To tackle camouflaged object detection (COD), we are inspired by humans attention coupled with the coarse-to-fine detection strategy, and thereby propose an iterative refinement framework, coined SegMaR, which integrates Segment, Magnify and Reiterate in a multi-stage detection fashion. Specifically, we design a new discriminative mask which makes the model attend on the fixation and edge regions. In addition, we leverage an attention-based sampler to magnify the object region progressively with no need of enlarging the image size. Extensive experiments show our SegMaR achieves remarkable and consistent improvements over other state-of-the-art methods. Especially, we surpass two competitive methods 7.4% and 20.0% respectively in average over standard evaluation metrics on small camouflaged objects. Additional studies provide more promising insights into Seg-MaR, including its effectiveness on the discriminative mask and its generalization to other network architectures. Code is available at https://github.com/dlut-dimt/SegMaR.

----

## [461] Dynamic Sparse R-CNN

**Authors**: *Qinghang Hong, Fengming Liu, Dong Li, Ji Liu, Lu Tian, Yi Shan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00468](https://doi.org/10.1109/CVPR52688.2022.00468)

**Abstract**:

Sparse R-CNN is a recent strong object detection baseline by set prediction on sparse, learnable proposal boxes and proposal features. In this work, we propose to improve Sparse R-CNN with two dynamic designs. First, Sparse R-CNN adopts a one-to-one label assignment scheme, where the Hungarian algorithm is applied to match only one positive sample for each ground truth. Such one-to-one assignment may not be optimal for the matching between the learned proposal boxes and ground truths. To address this problem, we propose dynamic label assignment (DLA) based on the optimal transport algorithm to assign increasing positive samples in the iterative training stages of Sparse R-CNN. We constrain the matching to be gradually looser in the sequential stages as the later stage produces the refined proposals with improved precision. Second, the learned proposal boxes and features remain fixed for different images in the inference process of Sparse R-CNN. Motivated by dynamic convolution, we propose dynamic proposal generation (DPG) to assemble multiple proposal experts dynamically for providing better initial proposal boxes and features for the consecutive training stages. DPG thereby can derive sample-dependent proposal boxes and features for inference. Experiments demonstrate that our method, named Dynamic Sparse R-CNN, can boost the strong Sparse R-CNN baseline with different backbones for object detection. Particularly, Dynamic Sparse R-CNN reaches the state-of-the-art 47.2% AP on the COCO 2017 validation set, surpassing Sparse R-CNN by 2.2% AP with the same ResNet-50 backbone.

----

## [462] Deep Hybrid Models for Out-of-Distribution Detection

**Authors**: *Senqi Cao, Zhongfei Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00469](https://doi.org/10.1109/CVPR52688.2022.00469)

**Abstract**:

We propose a principled and practical method for out-of-distribution (OoD) detection with deep hybrid models (DHMs), which model the joint density p(x, y) of features and labels with a single forward pass. By factorizing the joint density p(x, y) into three sources of uncertainty, we show that our approach has the ability to identify samples semantically different from the training data. To ensure computational scalability, we add a weight normalization step during training, which enables us to plug in state-of-the-art (SoTA) deep neural network (DNN) architectures for approximately modeling and inferring expressive probability distributions. Our method provides an efficient, general, and flexible framework for predictive uncertainty estimation with promising results and theoretical support. To our knowledge, this is the first work to reach 100% in OoD detection tasks on both vision and language datasets, especially on notably difficult dataset pairs such as CIFAR -10 vs. SVHN and CIFAR-100 vs. CIFAR-10. This work is a step towards enabling DNNs in real-world deployment for safety-critical applications.

----

## [463] AutoLoss-GMS: Searching Generalized Margin-based Softmax Loss Function for Person Re-identification

**Authors**: *Hongyang Gu, Jianmin Li, Guangyuan Fu, Chifong Wong, Xinghao Chen, Jun Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00470](https://doi.org/10.1109/CVPR52688.2022.00470)

**Abstract**:

Person re-identification is a hot topic in computer vision, and the loss function plays a vital role in improving the discrimination of the learned features. However, most existing models utilize the hand-crafted loss functions, which are usually sub-optimal and challenging to be designed. In this paper, we propose a novel method, AutoLoss-GMS, to search the better loss function in the space of generalized margin-based softmax loss function for person reidentification automatically. Specifically, the generalized margin-based softmax loss function is first decomposed into two computational graphs and a constant. Then a general searching framework built upon the evolutionary algorithm is proposed to search for the loss function efficiently. The computational graph is constructed with a forward method, which can construct much richer loss function forms than the backward method used in existing works. In addition to the basic in-graph mutation operations, the cross-graph mutation operation is designed to further improve the offspring's diversity. The loss-rejection protocol, equivalence-check strategy and the predictor-based promising-loss chooser are developed to improve the search efficiency. Finally, experimental results demonstrate that the searched loss functions can achieve state-of-the-art performance and be transferable across different models and datasets in person re-identification.

----

## [464] Feature Erasing and Diffusion Network for Occluded Person Re-Identification

**Authors**: *Zhikang Wang, Feng Zhu, Shixiang Tang, Rui Zhao, Lihuo He, Jiangning Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00471](https://doi.org/10.1109/CVPR52688.2022.00471)

**Abstract**:

Occluded person re-identification (ReID) aims at matching occluded person images to holistic ones across different camera views. Target Pedestrians (TP) are often disturbed by Non-Pedestrian Occlusions (NPO) and Non-Target Pedestrians (NTP). Previous methods mainly focus on increasing the model's robustness against NPO while ignoring feature contamination from NTP. In this paper, we propose a novel Feature Erasing and Diffusion Network (FED) to simultaneously handle challenges from NPO and NTP. Specifically, aided by the NPO augmentation strategy that simulates NPO on holistic pedestrian images and gen-erates precise occlusion masks, NPO features are explicitly eliminated by our proposed Occlusion Erasing Module (OEM). Subsequently, we diffuse the pedestrian representations with other memorized features to synthesize the NTP characteristics in the feature space through the novel Feature Diffusion Module (FDM). With the guidance of the occlusion scores from OEM, the feature diffusion process is conducted on visible body parts, thereby improving the quality of the synthesized NTP characteristics. We can greatly improve the model's perception ability towards TP and alleviate the influence of NPO and NTP by jointly optimizing OEM and FDM. Furthermore, the proposed FDM works as an auxiliary module for training and will not be engaged in the inference phase, thus with high flexibility. Experiments on occluded and holistic person ReID benchmarks demonstrate the superiority of FED over state-of-the-art methods.

----

## [465] Multi-label Classification with Partial Annotations using Class-aware Selective Loss

**Authors**: *Emanuel Ben Baruch, Tal Ridnik, Itamar Friedman, Avi Ben-Cohen, Nadav Zamir, Asaf Noy, Lihi Zelnik-Manor*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00472](https://doi.org/10.1109/CVPR52688.2022.00472)

**Abstract**:

Large-scale multi-label classification datasets are commonly, and perhaps inevitably, partially annotated. That is, only a small subset of labels are annotated per sample. Different methods for handling the missing labels induce different properties on the model and impact its accuracy. In this work, we analyze the partial labeling problem, then propose a solution based on two key ideas. First, un-annotated labels should be treated selectively according to two probability quantities: the class distribution in the overall dataset and the specific label likelihood for a given data sample. We propose to estimate the class distribution using a dedicated temporary model, and we show its improved efficiency over a naive estimation computed using the dataset's partial annotations. Second, during the training of the target model, we emphasize the contribution of annotated labels over originally un-annotated labels by using a dedicated asymmetric loss. With our novel approach, we achieve state-of-the-art results on OpenImages dataset (e.g. reaching 87.3 mAP on V6). In addition, experiments conducted on LVIS and simulated-COCO demonstrate the effectiveness of our approach. Code is available at https://github.com/Alibaba-MIIL/PartialLabelingCSL.

----

## [466] BoxeR: Box-Attention for 2D and 3D Transformers

**Authors**: *Duy-Kien Nguyen, Jihong Ju, Olaf Booij, Martin R. Oswald, Cees G. M. Snoek*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00473](https://doi.org/10.1109/CVPR52688.2022.00473)

**Abstract**:

In this paper, we propose a simple attention mechanism, we call Box-Attention. It enables spatial interaction between grid features, as sampled from boxes of interest, and improves the learning capability of transformers for several vision tasks. Specifically, we present BoxeR, short for Box Transformer, which attends to a set of boxes by predicting their transformation from a reference window on an input feature map. The BoxeR computes attention weights on these boxes by considering its grid structure. Notably, BoxeR-2D naturally reasons about box information within its attention module, making it suitable for end-to-end instance detection and segmentation tasks. By learning invariance to rotation in the box-attention module, BoxeR-3D is capable of generating discriminative information from a bird's-eye view plane for 3D end-to-end object detection. Our experiments demonstrate that the proposed BoxeR-2D achieves state-of-the-art results on COCO detection and instance segmentation. Besides, BoxeR-3D improves over the end-to-end 3D object detection baseline and already obtains a compelling performance for the vehicle category of Waymo Open, without any class-specific optimization. Code is available at https://github.com/kienduynguyen/BoxeR.

----

## [467] Multi-label Iterated Learning for Image Classification with Label Ambiguity

**Authors**: *Sai Rajeswar, Pau Rodríguez, Soumye Singhal, David Vázquez, Aaron C. Courville*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00474](https://doi.org/10.1109/CVPR52688.2022.00474)

**Abstract**:

Transfer learning from large-scale pre-trained models has become essential for many computer vision tasks. Recent studies have shown that datasets like ImageNet are weakly labeled since images with multiple object classes present are assigned a single label. This ambiguity biases models towards a single prediction, which could result in the suppression of classes that tend to co-occur in the data. Inspired by language emergence literature, we propose multi-label iterated learning (MILe) to incorporate the inductive biases of multi-label learning from single labels using the framework of iterated learning. MILe is a simple yet effective procedure that builds a multi-label description of the image by propagating binary predictions through successive generations of teacher and student networks with a learning bottleneck. Experiments show that our approach exhibits systematic benefits on ImageNet accuracy as well as ReaL F1 score, which indicates that MILe deals better with label ambiguity than the standard training procedure, even when fine-tuning from self-supervised weights. We also show that MILe is effective reducing label noise, achieving state-of-the-art performance on real-world large-scale noisy data such as WebVision. Furthermore, MILe improves performance in class incremental settings such as IIRC and it is robust to distribution shifts. Code: https://github.com/rajeswar18/MILe

----

## [468] Vision Transformer with Deformable Attention

**Authors**: *Zhuofan Xia, Xuran Pan, Shiji Song, Li Erran Li, Gao Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00475](https://doi.org/10.1109/CVPR52688.2022.00475)

**Abstract**:

Transformers have recently shown superior performances on various vision tasks. The large, sometimes even global, receptive field endows Transformer models with higher representation power over their CNN counterparts. Nevertheless, simply enlarging receptive field also gives rise to several concerns. On the one hand, using dense attention e.g., in ViT, leads to excessive memory and computational cost, and features can be influenced by irrelevant parts which are beyond the region of interests. On the other hand, the sparse attention adopted in PVT or Swin Transformer is data agnostic and may limit the ability to model long range relations. To mitigate these issues, we propose a novel deformable selfattention module, where the positions of key and value pairs in selfattention are selected in a data-dependent way. This flexible scheme enables the self-attention module to focus on relevant re-gions and capture more informative features. On this basis, we present Deformable Attention Transformer, a general backbone model with deformable attention for both image classification and dense prediction tasks. Extensive experi-ments show that our models achieve consistently improved results on comprehensive benchmarks. Code is available at https://github.com/LeapLabTHU/DAT.

----

## [469] MViTv2: Improved Multiscale Vision Transformers for Classification and Detection

**Authors**: *Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00476](https://doi.org/10.1109/CVPR52688.2022.00476)

**Abstract**:

In this paper, we study Multiscale Vision Transformers (MViTv2) as a unified architecture for image and video classification, as well as object detection. We present an improved version of MViT that incorporates decomposed relative positional embeddings and residual pooling connections. We instantiate this architecture in five sizes and evaluate it for ImageNet classification, COCO detection and Kinetics video recognition where it outperforms prior work. We further compare MViTv2s' pooling attention to window attention mechanisms where it outperforms the latter in accuracy/compute. Without bells-and-whistles, MViTv2 has state-of-the-art performance in 3 domains: 88.8% accuracy on ImageNet classification, 58.7 AP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">box</sup>
 on COCO object detection as well as 86.1% on Kinetics-400 video classification. Code and models are available at https://github.com/facebookresearch/mvit.

----

## [470] Dense Learning based Semi-Supervised Object Detection

**Authors**: *Binghui Chen, Pengyu Li, Xiang Chen, Biao Wang, Lei Zhang, Xian-Sheng Hua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00477](https://doi.org/10.1109/CVPR52688.2022.00477)

**Abstract**:

Semi-supervised object detection (SSOD) aims to facilitate the training and deployment of object detectors with the help of a large amount of unlabeled data. Though various self-training based and consistency-regularization based SSOD methods have been proposed, most of them are anchor-based detectors, ignoring the fact that in many real-world applications anchor-free detectors are more demanded. In this paper, we intend to bridge this gap and propose a DenSe Learning (DSL) based anchor-free SSOD algorithm. Specifically, we achieve this goal by introducing several novel techniques, including an Adaptive Filtering strategy for assigning multi-level and accurate dense pixel-wise pseudo-labels, an Aggregated Teacher for producing stable and precise pseudo-labels, and an uncertainty-consistency-regularization term among scales and shuffled patches for improving the generalization capability of the detector. Extensive experiments are conducted on MS-COCO and PASCAL-VOC, and the results show that our proposed DSL method records new state-of-the-art SSOD performance, surpassing existing methods by a large margin. Codes can be found at https://github.com/chenbinghui1/DSL.

----

## [471] R(Det)2: Randomized Decision Routing for Object Detection

**Authors**: *Yali Li, Shengjin Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00478](https://doi.org/10.1109/CVPR52688.2022.00478)

**Abstract**:

In the paradigm of object detection, the decision head is an important part, which affects detection performance significantly. Yet how to design a high-performance decision head remains to be an open issue. In this paper, we propose a novel approach to combine decision trees and deep neural networks in an end-to-end learning manner for object detection. First, we disentangle the decision choices and prediction values by plugging soft decision trees into neural networks. To facilitate effective learning, we propose randomized decision routing with node selective and associative losses, which can boost the feature representative learning and network decision simultaneously. Second, we develop the decision head for object detection with narrow branches to generate the routing probabilities and masks, for the purpose of obtaining divergent decisions from different nodes. We name this approach as the randomized decision routing for object detection, abbreviated as R(Det)2. Experiments on MS-COCO dataset demonstrate that R(Det)2 is effective to improve the detection performance. Equipped with existing detectors, it achieves 1.4 ~ 3.6% AP improvement.

----

## [472] GlideNet: Global, Local and Intrinsic based Dense Embedding NETwork for Multi-category Attributes Prediction

**Authors**: *Kareem Metwaly, Aerin Kim, Elliot Branson, Vishal Monga*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00479](https://doi.org/10.1109/CVPR52688.2022.00479)

**Abstract**:

Attaching attributes (such as color, shape, state, action) to object categories is an important computer vision problem. Attribute prediction has seen exciting recent progress and is often formulated as a multi-label classification problem. Yet significant challenges remain in: 1) predicting a large number of attributes over multiple object categories, 2) modeling category-dependence of attributes, 3) methodically capturing both global and local scene context, and 4) robustly predicting attributes of objects with low pixel-count. To address these issues, we propose a novel multi-category attribute prediction deep architecture named GlideNet, which contains three distinct feature extractors. A global feature extractor recognizes what objects are present in a scene, whereas a local one focuses on the area surrounding the object of interest. Meanwhile, an intrinsic feature extractor uses an extension of standard convolution dubbed Informed Convolution to retrieve features of objects with low pixel-count utilizing its binary mask. GlideNet then uses gating mechanisms with binary masks and its self-learned category embedding to combine the dense embeddings. Collectively, the Global-Local-Intrinsic blocks comprehend the scene's global context while attending to the characteristics of the local object of interest. The architecture adapts the feature composition based on the category via category embedding. Finally, using the combined features, an interpreter predicts the attributes, and the length of the output is determined by the category, thereby removing unnecessary attributes. GlideNet can achieve compelling results on two recent and challenging datasets - VAW and CAR -for large-scale attribute prediction. For instance, it obtains more than 5% gain over state of the art in the mean recall (mR) metric. GlideNet's advantages are especially apparent when predicting attributes of objects with low pixel counts as well as attributes that demand global context understanding. Finally, we show that GlideNet excels in training starved real-world scenarios.

----

## [473] Self-Supervised Equivariant Learning for Oriented Keypoint Detection

**Authors**: *Jongmin Lee, Byungjin Kim, Minsu Cho*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00480](https://doi.org/10.1109/CVPR52688.2022.00480)

**Abstract**:

Detecting robust keypoints from an image is an integral part of many computer vision problems, and the characteristic orientation and scale of keypoints play an important role for keypoint description and matching. Existing learning-based methods for keypoint detection rely on standard translation-equivariant CNNs but often fail to detect reliable keypoints against geometric variations. To learn to detect robust oriented keypoints, we introduce a self-supervised learning framework using rotation-equivariant CNNs. We propose a dense orientation alignment loss by an image pair generated by synthetic transformations for training a histogram-based orientation map. Our method outperforms the previous methods on an image matching benchmark and a camera pose estimation benchmark.

----

## [474] Label Relation Graphs Enhanced Hierarchical Residual Network for Hierarchical Multi-Granularity Classification

**Authors**: *Jingzhou Chen, Peng Wang, Jian Liu, Yuntao Qian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00481](https://doi.org/10.1109/CVPR52688.2022.00481)

**Abstract**:

Hierarchical multi-granularity classification (HMC) assigns hierarchical multi-granularity labels to each object and focuses on encoding the label hierarchy, e.g., [“Albatross”, “Laysan Albatross”] from coarse-to-fine levels. However, the definition of what is fine-grained is subjective, and the image quality may affect the identification. Thus, samples could be observed at any level of the hierarchy, e.g., [“Albatross”] or [“Albatross”, “Laysan Albatross”], and examples discerned at coarse categories are often neglected in the conventional setting of HMC. In this paper, we study the HMC problem in which objects are labeled at any level of the hierarchy. The essential designs of the proposed method are derived from two motivations: (1) learning with objects labeled at various levels should transfer hierarchical knowledge between levels; (2) lower-level classes should inherit attributes related to upper-level superclasses. The proposed combinatorial loss maximizes the marginal probability of the observed ground truth label by aggregating information from related labels defined in the tree hierarchy. If the observed label is at the leaf level, the combinatorial loss further imposes the multi-class cross-entropy loss to increase the weight of fine-grained classification loss. Considering the hierarchical feature interaction, we propose a hierarchical residual network (HRN), in which granularity-specific features from parent levels acting as residual connections are added to features of children levels. Experiments on three commonly used datasets demonstrate the effectiveness of our approach compared to the state-of-the-art HMC approaches. The code will be available at https://github.com/MonsterZhZh/HRN.

----

## [475] Object Localization under Single Coarse Point Supervision

**Authors**: *Xuehui Yu, Pengfei Chen, Di Wu, Najmul Hassan, Guorong Li, Junchi Yan, Humphrey Shi, Qixiang Ye, Zhenjun Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00482](https://doi.org/10.1109/CVPR52688.2022.00482)

**Abstract**:

Point-based object localization (POL), which pursues high-performance object sensing under low-cost data annotation, has attracted increased attention. However, the point annotation mode inevitably introduces semantic variance for the inconsistency of annotated points. Existing POL methods heavily reply on accurate keypoint annotations which are difficult to define. In this study, we propose a POL method using coarse point annotations, relaxing the supervision signals from accurate key points to freely spotted points. To this end, we propose a coarse point refinement (CPR) approach, which to our best knowledge is the first attempt to alleviate semantic variance from the perspective of algorithm. CPR constructs point bags, selects semantic-correlated points, and produces semantic center points through multiple instance learning (MIL). In this way, CPR defines a weakly supervised evolution procedure, which ensures training high-performance object localizer under coarse point supervision. Experimental results on COCO, DOTA and our proposed SeaPerson dataset validate the effectiveness of the CPR approach. The dataset and code will be available at https://github.com/ucas-vg/PointTinyBenchmark/

----

## [476] Rethinking Visual Geo-localization for Large-Scale Applications

**Authors**: *Gabriele Moreno Berton, Carlo Masone, Barbara Caputo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00483](https://doi.org/10.1109/CVPR52688.2022.00483)

**Abstract**:

Visual Geo-localization (VG) is the task of estimating the position where a given photo was taken by comparing it with a large database of images of known locations. To investigate how existing techniques would perform on a real-world city-wide VG application, we build San Francisco eXtra Large, a new dataset covering a whole city and providing a wide range of challenging cases, with a size 30x bigger than the previous largest dataset for visual geo-localization. We find that current methods fail to scale to such large datasets, therefore we design a new highly scalable training technique, called CosPlace, which casts the training as a classification problem avoiding the expensive mining needed by the commonly used contrastive learning. We achieve state-of-the-art performance on a wide range of datasets and find that CosPlace is robust to heavy domain changes. Moreover, we show that, compared to the previous state-of-the-art, CosPlace requires roughly 80% less GPU memory at train time, and it achieves better results with 8x smaller descriptors, paving the way for city-wide real-world visual geo-localization. Dataset, code and trained models are available for research purposes at https://github.com/gmberton/CosPlace.

----

## [477] Whose Hands are These? Hand Detection and Hand-Body Association in the Wild

**Authors**: *Supreeth Narasimhaswamy, Thanh Nguyen, Mingzhen Huang, Minh Hoai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00484](https://doi.org/10.1109/CVPR52688.2022.00484)

**Abstract**:

We study a new problem of detecting hands and finding the location of the corresponding person for each detected hand. This task is helpful for many downstream tasks such as hand tracking and hand contact estimation. Associating hands with people is challenging in unconstrained conditions since multiple people can be present in the scene with varying overlaps and occlusions. We propose a novel end-to-end trainable convolutional network that can Jointly detect hands and the body location for the corresponding person. Our method first detects a set of hands and bodies and uses a novel Hand-Body Association Network to predict association scores between them. We use these association scores to find the body location for each detected hand. We also introduce a new challenging dataset called BodyHands containing uncon-strained images with hand and their corresponding body locations annotations. We conduct extensive experiments on BodyHands and another public dataset to show the effectiveness of our method. Finally, we demonstrate the benefits of hand-body association in two critical applications: hand tracking and hand contact estimation. Our experiments show that hand tracking and hand contact estimation methods can be improved significantly by reasoning about the hand-body association. Code and data can be found at http://vision.cs.stonybrook.edu/~supreeth/BodyHands/.

----

## [478] Cloning Outfits from Real-World Images to 3D Characters for Generalizable Person Re-Identification

**Authors**: *Yanan Wang, Xuezhi Liang, Shengcai Liao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00485](https://doi.org/10.1109/CVPR52688.2022.00485)

**Abstract**:

Recently, large-scale synthetic datasets are shown to be very useful for generalizable person re-identification. However, synthesized persons in existing datasets are mostly cartoon-like and in random dress collocation, which limits their performance. To address this, in this work, an automatic approach is proposed to directly clone the whole outfits from real-world person images to virtual 3D characters, such that any virtual person thus created will appear very similar to its real-world counterpart. Specifically, based on UV texture mapping, two cloning methods are designed, namely registered clothes mapping and homogeneous cloth expansion. Given clothes keypoints detected on person images and labeled on regular UV maps with clear clothes structures, registered mapping applies perspective homography to warp real-world clothes to the counterparts on the UV map. As for invisible clothes parts and irregular UV maps, homogeneous expansion segments a homogeneous area on clothes as a realistic cloth pattern or cell, and expand the cell to fill the UV map. Furthermore, a similarity-diversity expansion strategy is proposed, by clustering person images, sampling images per cluster, and cloning outfits for 3D character generation. This way, virtual persons can be scaled up densely in visual similarity to challenge model learning, and diversely in population to enrich sample distribution. Finally, by rendering the cloned characters in Unity3D scenes, a more realistic virtual dataset called ClonedPerson is created, with 5,621 identities and 887,766 images. Experimental results show that the model trained on ClonedPerson has a better generalization performance, superior to that trained on other popular real-world and synthetic person re-identification datasets. The ClonedPerson project is available at https://github.com/Yanan-Wang-cs/ClonedPerson.

----

## [479] Towards Unsupervised Domain Generalization

**Authors**: *Xingxuan Zhang, Linjun Zhou, Renzhe Xu, Peng Cui, Zheyan Shen, Haoxin Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00486](https://doi.org/10.1109/CVPR52688.2022.00486)

**Abstract**:

Domain generalization (DG) aims to help models trained on a set of source domains generalize better on unseen target domains. The performances of current DG methods largely rely on sufficient labeled data, which are usually costly or unavailable, however. Since unlabeled data are far more accessible, we seek to explore how unsupervised learning can help deep models generalize across domains. Specifically, we study a novel generalization problem called unsupervised domain generalization (UDG), which aims to learn generalizable models with unlabeled data and analyze the effects of pre-training on DG. In UDG, models are pretrained with unlabeled data from various source domains before being trained on labeled source data and eventually tested on unseen target domains. Then we propose a method named Domain-Aware Representation LearnING (DARLING) to cope with the significant and misleading heterogeneity within unlabeled pretraining data and severe distribution shifts between source and target data. Surprisingly we observe that DARLING can not only counterbalance the scarcity of labeled data but also further strengthen the generalization ability of models when the labeled data are insufficient. As a pretraining approach, DARLING shows superior or comparable performance compared with ImageNet pretraining protocol even when the available data are unlabeled and of a vastly smaller amount compared to ImageNet, which may shed light on improving generalization with large-scale unlabeled data.

----

## [480] ViM: Out-Of-Distribution with Virtual-logit Matching

**Authors**: *Haoqi Wang, Zhizhong Li, Litong Feng, Wayne Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00487](https://doi.org/10.1109/CVPR52688.2022.00487)

**Abstract**:

Most of the existing Out-Of-Distribution (OOD) detection algorithms depend on single input source: the feature, the logit, or the softmax probability. However, the immense diversity of the OOD examples makes such methods fragile. There are OOD samples that are easy to identify in the feature space while hard to distinguish in the logit space and vice versa. Motivated by this observation, we propose a novel OOD scoring method named Virtual-logit Matching (ViM), which combines the class-agnostic score from feature space and the In-Distribution (ID) class-dependent logits. Specifically, an additional logit representing the virtual OOD class is generated from the residual of the feature against the principal space, and then matched with the original logits by a constant scaling. The probability of this virtual logit after softmax is the indicator of OOD-ness. To facilitate the evaluation of large-scale OOD detection in academia, we create a new OOD dataset for ImageNet1K, which is human-annotated and is 8.8× the size of existing datasets. We conducted extensive experiments, including CNNs and vision transformers, to demonstrate the effectiveness of the proposed ViM score. In particular, using the BiT-S model, our method gets an average AUROC 90.91% on four difficult OOD benchmarks, which is 4% ahead of the best baseline. Code and dataset are available at https://github.com/haoqiwang/vim.

----

## [481] Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space

**Authors**: *Arnav Chavan, Zhiqiang Shen, Zhuang Liu, Zechun Liu, Kwang-Ting Cheng, Eric P. Xing*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00488](https://doi.org/10.1109/CVPR52688.2022.00488)

**Abstract**:

This paper explores the feasibility of finding an optimal sub-model from a vision transformer and introduces a pure vision transformer slimming (ViT-Slim) framework. It can search a sub-structure from the original model end-to-end across multiple dimensions, including the input tokens, MHSA and MLP modules with state-of-the-art performance. Our method is based on a learnable and unified ℓ
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</inf>
sparsity constraint with pre-defined factors to reflect the global importance in the continuous searching space of different dimensions. The searching process is highly efficient through a single-shot training scheme. For instance, on DeiT-S, ViT-Slim only takes ~43 GPU hours for the searching process, and the searched structure is flexible with diverse dimensionalities in different modules. Then, a budget threshold is employed according to the requirements of accuracy-FLOPs trade-off on running devices, and a retraining process is performed to obtain the final model. The extensive experiments show that our ViT-Slim can compress up to 40% of parameters and 40% FLOPs on various vision transformers while increasing the accuracy by ~0.6% on ImageNet. We also demonstrate the advantage of our searched models on several downstream datasets. Our code is available at https://github.com/Arnav0400/ViT-Slim.

----

## [482] Nonuniform-to-Uniform Quantization: Towards Accurate Quantization via Generalized Straight-Through Estimation

**Authors**: *Zechun Liu, Kwang-Ting Cheng, Dong Huang, Eric P. Xing, Zhiqiang Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00489](https://doi.org/10.1109/CVPR52688.2022.00489)

**Abstract**:

The nonuniform quantization strategy for compressing neural networks usually achieves better performance than its counterpart, i.e., uniform strategy, due to its superior representational capacity. However, many nonuniform quantization methods overlook the complicated projection process in implementing the nonuniformly quantized weights/activations, which incurs non-negligible time and space overhead in hardware deployment. In this study, we propose Nonuniform-to-Uniform Quantization (N2UQ), a method that can maintain the strong representation ability of nonuniform methods while being hardware-friendly and efficient as the uniform quantization for model inference. We achieve this through learning the flexible inequidistant input thresholds to better fit the underlying distribution while quantizing these real-valued inputs into equidistant output levels. To train the quantized network with learnable input thresholds, we introduce a generalized straight-through estimator (G-STE) for intractable backward derivative calculation w.r.t. threshold parameters. Additionally, we consider entropy preserving regularization to further reduce information loss in weight quantization. Even under this adverse constraint of imposing uniformly quantized weights and activations, our N2UQ outperforms state-of-the-art nonuniform quantization methods by 0.5 ~ 1.7% on ImageNet, demonstrating the contribution of N2UQ design. Code and models are available at: https://github.com/liuzechun/Nonuniform-to-Uniform-Quantization.

----

## [483] Align and Prompt: Video-and-Language Pre-training with Entity Prompts

**Authors**: *Dongxu Li, Junnan Li, Hongdong Li, Juan Carlos Niebles, Steven C. H. Hoi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00490](https://doi.org/10.1109/CVPR52688.2022.00490)

**Abstract**:

Yidco-and-language pre-training has shown promising improvements on various downstream tasks. Most previous methods capture cross-modal interactions with a standard transformer-based multimodal encoder, not fully addressing the misalignment between unimodal video and text features. Besides, learning finegrained visual-language alignment usually requires off-the-shelf object detectors to provide object information, which is bottlenecked by the detector's limited vocabulary and expensive computation cost. In this paper, we propose Align and Prompt: a new video-and-language pre-training framework (AlPro), which operates on sparsely-sampled video frames and achieves more effective cross-modal alignment without explicit object detectors. First, we introduce a video-text contrastive (VTC) loss to align unimodal video-text features at the instance level, which eases the modeling of cross-modal interactions. Then, we propose a novel visually-grounded pre-training task, prompting entity modeling (PEM), which learns finegrained alignment between visual region and text entity via an entity prompter module in a self-supervised way. Finally, we pretrain the video-and-language transformer models on large webly-source video-text pairs using the proposed VTC and PEM losses as well as two standard losses of masked language modeling (MLM) and video-text matching (VTM). The resulting pre-trained model achieves state-of-the-art performance on both text-video retrieval and videoQA, outperforming prior work by a substantial margin. Implementation and pre-trained models are available at https://github.com/salesforce/ALPRO.

----

## [484] Language-Bridged Spatial-Temporal Interaction for Referring Video Object Segmentation

**Authors**: *Zihan Ding, Tianrui Hui, Junshi Huang, Xiaoming Wei, Jizhong Han, Si Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00491](https://doi.org/10.1109/CVPR52688.2022.00491)

**Abstract**:

Referring video object segmentation aims to predict foreground labels for objects referred by natural language expressions in videos. Previous methods either depend on 3D ConvNets or incorporate additional 2D ConvNets as encoders to extract mixed spatial-temporal features. However, these methods suffer from spatial misalignment or false distractors due to delayed and implicit spatial-temporal interaction occurring in the decoding phase. To tackle these limitations, we propose a Language-Bridged Duplex Transfer (LBDT) module which utilizes language as an intermediary bridge to accomplish explicit and adaptive spatial-temporal interaction earlier in the encoding phase. Concretely, cross-modal attention is performed among the temporal encoder, referring words and the spatial encoder to aggregate and transfer language-relevant motion and appearance information. In addition, we also propose a Bilateral Channel Activation (BCA) module in the decoding phase for further denoising and highlighting the spatial-temporal consistent features via channel-wise activation. Extensive experiments show our method achieves new state-of-the-art performances on four popular benchmarks with 6.8% and 6.9% absolute AP gains on A2D Sentences and J-HMDB Sentences respectively, while consuming around 7× less computational overhead
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/dzh19990407/LBDT.

----

## [485] Language as Queries for Referring Video Object Segmentation

**Authors**: *Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, Ping Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00492](https://doi.org/10.1109/CVPR52688.2022.00492)

**Abstract**:

Referring video object segmentation (R-VOS) is an emerging cross-modal task that aims to segment the target object referred by a language expression in all video frames. In this work, we propose a simple and unified framework built upon Transformer, termed ReferFormer. It views the language as queries and directly attends to the most relevant regions in the video frames. Concretely, we introduce a small set of object queries conditioned on the language as the input to the Transformer. In this manner, all the queries are obligated to find the referred objects only. They are eventually transformed into dynamic kernels which capture the crucial object-level information, and play the role of convolution filters to generate the segmentation masks from feature maps. The object tracking is achieved naturally by linking the corresponding queries across frames. This mechanism greatly simplifies the pipeline and the end-to-end framework is significantly different from the previous methods. Extensive experiments on Ref-Youtube-vos, Ref-DAVIS17, A2D-Sentences and JHMDB-Sentences show the effectiveness of ReferFormer. On Ref-Youtube-vos, ReferFormer achieves 55.6 J&F with a ResNet-50 backbone without bells and whistles, which exceeds the previous state-of-the-art performance by 8.4 points. In addition, with the strong Video-Swin-Base backbone, ReferFormer achieves the best J&F of 64.9 among all existing methods. Moreover, we show the impressive results of 55.0 mAP and 43.7 mAP on A2D-Sentences and JHMDB-Sentences respectively, which significantly outperforms the previous methods by a large margin. Code is publicly available at https://github.com/wjn922/ReferFormer.

----

## [486] End-to-End Referring Video Object Segmentation with Multimodal Transformers

**Authors**: *Adam Botach, Evgenii Zheltonozhskii, Chaim Baskin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00493](https://doi.org/10.1109/CVPR52688.2022.00493)

**Abstract**:

The referring video object segmentation task (RVOS) involves segmentation of a text-referred object instance in the frames of a given video. Due to the complex nature of this multimodal task, which combines text reasoning, video understanding, instance segmentation and tracking, existing approaches typically rely on sophisticated pipelines in order to tackle it. In this paper, we propose a simple Transformer-based approach to RVOS. Our framework, termed Multimodal Tracking Transformer (MTTR), models the RVOS task as a sequence prediction problem. Following recent advancements in computer vision and natural language processing, MTTR is based on the realization that video and text can be processed together effectively and elegantly by a single multimodal Transformer model. MTTR is end-to-end trainable, free of text-related inductive bias components and requires no additional mask-refinement post-processing steps. As such, it simplifies the RVOS pipeline considerably compared to existing methods. Evaluation on standard benchmarks reveals that MTTR significantly outperforms previous art across multiple metrics. In particular, MTTR shows impressive +5.7 and +5.0 mAP gains on the A2D-Sentences and JHMDB-Sentences datasets respectively, while processing 76 frames per second. In addition, we report strong results on the public validation set of Refer-YouTube-VOS, a more challenging RVOS dataset that has yet to receive the attention of researchers. The code to reproduce our experiments is avail-able at https://github.com/mttr2021/MTTR.

----

## [487] Multi-Level Representation Learning with Semantic Alignment for Referring Video Object Segmentation

**Authors**: *Dongming Wu, Xingping Dong, Ling Shao, Jianbing Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00494](https://doi.org/10.1109/CVPR52688.2022.00494)

**Abstract**:

Referring video object segmentation (RVOS) is a challenging language-guided video grounding task, which requires comprehensively understanding the semantic information of both video content and language queries for object prediction. However, existing methods adopt multi-modal fusion at a frame-based spatial granularity. The limitation of visual representation is prone to causing vision-language mismatching and producing poor segmentation results. To address this, we propose a novel multi-level representation learning approach, which explores the inherent structure of the video content to provide a set of discriminative visual embedding, enabling more effective vision-language semantic alignment. Specifically, we embed different visual cues in terms of visual granularity, including multi-frame long-temporal information at video level, intra-frame spatial semantics at frame level, and enhanced object-aware feature prior at object level. With the powerful multi-level visual embedding and carefully-designed dynamic alignment, our model can generate a robust representation for accurate video object segmentation. Extensive experiments on Refer-DAVIS
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">17</inf>
 and Refer-YouTube-VOS demonstrate that our model achieves superior performance both in segmentation accuracy and inference speed.

----

## [488] X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval

**Authors**: *Satya Krishna Gorti, Noël Vouitsis, Junwei Ma, Keyvan Golestan, Maksims Volkovs, Animesh Garg, Guangwei Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00495](https://doi.org/10.1109/CVPR52688.2022.00495)

**Abstract**:

In text-video retrieval, the objective is to learn a cross-modal similarity function between a text and a video that ranks relevant text-video pairs higher than irrelevant pairs. However, videos inherently express a much wider gamut of information than texts. Instead, texts often capture sub-regions of entire videos and are most semantically similar to certain frames within videos. Therefore, for a given text, a retrieval model should focus on the text's most semantically similar video sub-regions to make a more relevant comparison. Yet, most existing works aggregate entire videos with-out directly considering text. Common text-agnostic ag-gregations schemes include mean-pooling or self-attention over the frames, but these are likely to encode misleading vi-sual information not described in the given text. To address this, we propose a cross-modal attention model called X-Pool that reasons between a text and the frames of a video. Our core mechanism is a scaled dot product attention for a text to attend to its most semantically similar frames. We then generate an aggregated video representation conditioned on the text's attention weights over the frames. We evaluate our method on three benchmark datasets of MSR-VTT, MSVD and LSMDC, achieving new state-of-the-art re-sults by up to 12% in relative improvement in Recall@ 1. Our findings thereby highlight the importance of joint text-video reasoning to extract important visual cues according to text. Full code and demo can be found at: layer6ai-labs.github.iolxpooll.

----

## [489] Video-Text Representation Learning via Differentiable Weak Temporal Alignment

**Authors**: *Dohwan Ko, Joonmyung Choi, Juyeon Ko, Shinyeong Noh, Kyoung-Woon On, Eun-Sol Kim, Hyunwoo J. Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00496](https://doi.org/10.1109/CVPR52688.2022.00496)

**Abstract**:

Learning generic joint representations for video and text by a supervised method requires a prohibitively substantial amount of manually annotated video datasets. As a practical alternative, a large-scale but uncurated and narrated video dataset, HowTo100M, has recently been introduced. But it is still challenging to learn joint embeddings of video and text in a self-supervised manner, due to its ambiguity and non-sequential alignment. In this paper, we propose a novel multi-modal self-supervised framework Video-Text Temporally Weak Alignment-based Contrastive Learning (VT-TWINS) to capture significant information from noisy and weakly correlated data using a variant of Dynamic Time Warping (DTW). We observe that the standard DTW inherently cannot handle weakly correlated data and only considers the globally optimal alignment path. To address these problems, we develop a differentiable DTW which also reflects local information with weak temporal alignment. Moreover, our proposed model applies a contrastive learning scheme to learn feature representations on weakly correlated data. Our extensive experiments demonstrate that VT-TWINS attains significant improvements in multi-modal representation learning and outperforms various challenging downstream tasks. Code is available at https://github.com/mlvlab/VT-Twins.

----

## [490] MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions

**Authors**: *Mattia Soldan, Alejandro Pardo, Juan León Alcázar, Fabian Caba Heilbron, Chen Zhao, Silvio Giancola, Bernard Ghanem*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00497](https://doi.org/10.1109/CVPR52688.2022.00497)

**Abstract**:

The recent and increasing interest in video-language research has driven the development of large-scale datasets that enable data-intensive machine learning techniques. In comparison, limited effort has been made at assessing the fitness of these datasets for the video-language grounding task. Recent works have begun to discover significant limitations in these datasets, suggesting that state-of-the-art techniques commonly overfit to hidden dataset biases. In this work, we present MAD (Movie Audio Descriptions), a novel benchmark that departs from the paradigm of augmenting existing video datasets with text annotations and focuses on crawling and aligning available audio descriptions of mainstream movies. MAD contains over 384, 000 natural language sentences grounded in over 1, 200 hours of videos and exhibits a significant reduction in the currently diagnosed biases for video-language grounding datasets. MAD's collection strategy enables a novel and more challenging version of video-language grounding, where short temporal moments (typically seconds long) must be accurately grounded in diverse long-form videos that can last up to three hours. We have released MAD's data and baselines code at https://github.com/Soldelli/MAD.

----

## [491] Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions

**Authors**: *Hongwei Xue, Tiankai Hang, Yanhong Zeng, Yuchong Sun, Bei Liu, Huan Yang, Jianlong Fu, Baining Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00498](https://doi.org/10.1109/CVPR52688.2022.00498)

**Abstract**:

We study joint video and language (VL) pretraining to enable cross-modality learning and benefit plentiful downstream VL tasks. Existing works either extract low-quality video features or learn limited text embedding, while neglecting that high-resolution videos and diversified semantics can significantly improve cross-modality learning. In this paper, we propose a novel High-resolution and Diversified VIdeo-LAnguage pre-training model (HD-VILA) for many visual tasks. In particular, we collect a large dataset with two distinct properties: 1) the first high-resolution dataset including 371.5k hours of 720p videos, and 2) the most diversified dataset covering 15 popular YouTube categories. To enable VL pre-training, we jointly optimize the HD-VILA model by a hybrid Transformer that learns rich spatiotemporal features, and a multimodal Transformer that enforces interactions of the learned video features with diversified texts. Our pre-training model achieves new state-of-the-art results in 10 VL understanding tasks and 2 more novel text-to-visual generation tasks. For example, we outperform SOTA models with relative increases of 40.4% R@1 in zero-shot MSR-VTT text-to-video retrieval task, and 55.4% in high-resolution dataset LSMDC. The learned VL embedding is also effective in generating visually pleasing and semantically relevant results in text-to-visual editing and super-resolution tasks.

----

## [492] Measuring Compositional Consistency for Video Question Answering

**Authors**: *Mona Gandhi, Mustafa Omer Gul, Eva Prakash, Madeleine Grunde-McLaughlin, Ranjay Krishna, Maneesh Agrawala*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00499](https://doi.org/10.1109/CVPR52688.2022.00499)

**Abstract**:

Recent video question answering benchmarks indicate that state-of-the-art models struggle to answer compositional questions. However, it remains unclear which types of compositional reasoning cause models to mispredict. Furthermore, it is difficult to discern whether models arrive at answers using compositional reasoning or by leveraging data biases. In this paper, we develop a question decomposition engine that programmatically deconstructs a compositional question into a directed acyclic graph of sub-questions. The graph is designed such that each parent question is a composition of its children. We present AGQA-Decomp, a benchmark containing 2.3M question graphs, with an average of 11.49 sub-questions per graph, and 4.55M total new sub-questions. Using question graphs, we evaluate three state-of-the-art models with a suite of novel compositional consistency metrics. We find that models either cannot reason correctly through most compositions or are reliant on incorrect reasoning to reach answers, frequently contradicting themselves or achieving high accuracies when failing at intermediate reasoning steps.

----

## [493] Sim VQA: Exploring Simulated Environments for Visual Question Answering

**Authors**: *Paola Cascante-Bonilla, Hui Wu, Letao Wang, Rogério Feris, Vicente Ordonez*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00500](https://doi.org/10.1109/CVPR52688.2022.00500)

**Abstract**:

Existing work on VQA explores data augmentation to achieve better generalization by perturbing images in the dataset or modifying existing questions and answers. While these methods exhibit good performance, the diversity of the questions and answers are constrained by the available images. In this work we explore using synthetic computer-generated data to fully control the visual and language space, allowing us to provide more diverse scenarios. We quantify the effectiveness of leveraging synthetic data for real-world VQA. By exploiting 3D and physics simulation platforms, we provide a pipeline to generate synthetic data to expand and replace type-specific questions and answers without risking exposure of sensitive or personal data that might be present in real images. We offer a comprehensive analysis while expanding existing hyper-realistic datasets to be usedfor VQA. We also propose Feature Swapping (F-SWAP) - where we randomly switch object-level features during training to make a VQA model more domain invariant. We show that F-SWAP is effective for improving VQA models on real images without compromising on their accuracy to answer existing questions in the dataset.

----

## [494] Transform-Retrieve-Generate: Natural Language-Centric Outside-Knowledge Visual Question Answering

**Authors**: *Feng Gao, Qing Ping, Govind Thattai, Aishwarya N. Reganti, Ying Nian Wu, Prem Natarajan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00501](https://doi.org/10.1109/CVPR52688.2022.00501)

**Abstract**:

Outside-knowledge visual question answering (OK-VQA) requires the agent to comprehend the image, make use of relevant knowledge from the entire web, and digest all the information to answer the question. Most previous works address the problem by first fusing the image and question in the multi-modal space, which is inflexible for further fusion with a vast amount of external knowledge. In this paper, we call for an alternative paradigm for the OK-VQA task, which transforms the image into plain text, so that we can enable knowledge passage retrieval, and generative question-answering in the natural language space. This paradigm takes advantage of the sheer volume of gigantic knowledge bases and the richness of pretrained language models. A Transform-Retrieve-Generate framework (TRiG) framework is proposed
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code of this work will be made public., which can be plug-and-played with alternative image-to-text models and textual knowledge bases. Experimental results show that our TRiG framework outperforms all state-of-the-art supervised methods by at least 11.1 % absolute margin.

----

## [495] SwapMix: Diagnosing and Regularizing the Over-Reliance on Visual Context in Visual Question Answering

**Authors**: *Vipul Gupta, Zhuowan Li, Adam Kortylewski, Chenyu Zhang, Yingwei Li, Alan L. Yuille*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00502](https://doi.org/10.1109/CVPR52688.2022.00502)

**Abstract**:

While Visual Question Answering (VQA) has progressed rapidly, previous works raise concerns about robustness of current VQA models. In this work, we study the robustness of VQA models from a novel perspective: visual context. We suggest that the models over-rely on the visual context, i.e., irrelevant objects in the image, to make predictions. To diagnose the models' reliance on visual context and measure their robustness, we propose a simple yet effective perturbation technique, SwapMix. SwapMix perturbs the visual context by swapping features of irrelevant context objects with features from other objects in the dataset. Using SwapMix we are able to change answers to more than 45% of the questions for a representative VQA model. Additionally, we train the models with perfect sight and find that the context over-reliance highly depends on the quality of visual representations. In addition to diagnosing, SwapMix can also be applied as a data augmentation strategy during training in order to regularize the context over-reliance. By swapping the context object features, the model reliance on context can be suppressed effectively. Two representative VQA models are studied using SwapMix: a co-attention model MCAN and a large-scale pretrained model LXMERT. Our experiments on the popular GQA dataset show the effectiveness of SwapMix for both diagnosing model robustness, and regularizing the over-reliance on visual context. The code for our method is available at https://github.com/vipulgupta1011/swapmix

----

## [496] MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering

**Authors**: *Yang Ding, Jing Yu, Bang Liu, Yue Hu, Mingxin Cui, Qi Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00503](https://doi.org/10.1109/CVPR52688.2022.00503)

**Abstract**:

Knowledge-based visual question answering requires the ability of associating external knowledge for open-ended cross-modal scene understanding. One limitation of existing solutions is that they capture relevant knowledge from text-only knowledge bases, which merely contain facts expressed by first-order predicates or language descriptions while lacking complex but indispensable multimodal knowledge for visual understanding. How to construct vision-relevant and explainable multimodal knowledge for the VQA scenario has been less studied. In this paper, we propose MuKEA to represent multimodal knowledge by an explicit triplet to correlate visual objects and fact answers with implicit relations. To bridge the heterogeneous gap, we propose three objective losses to learn the triplet representations from complementary views: embedding structure, topological relation and semantic space. By adopting a pretraining and fine-tuning learning strategy, both basic and domain-specific multimodal knowledge are progressively accumulated for answer prediction. We outperform the state-of-the-art by 3.35% and 6.08% respectively on two challenging knowledge-required datasets: OK-VQA and KRVQA. Experimental results prove the complementary benefits of the multimodal knowledge with existing knowledge bases and the advantages of our end-to-end framework over the existing pipeline methods. The code is available at https://github.com/AndersonStra/MuKEA.

----

## [497] Maintaining Reasoning Consistency in Compositional Visual Question Answering

**Authors**: *Chenchen Jing, Yunde Jia, Yuwei Wu, Xinyu Liu, Qi Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00504](https://doi.org/10.1109/CVPR52688.2022.00504)

**Abstract**:

A compositional question refers to a question that contains multiple visual concepts (e.g., objects, attributes, and relationships) and requires compositional reasoning to answer. Existing VQA models can answer a compositional question well, but cannot work well in terms of reasoning consistency in answering the compositional question and its sub-questions. For example, a compositional question for an image is: “Are there any elephants to the right of the white bird?” and one of its sub-questions is “Is any bird visible in the scene?”. The models may answer “yes” to the compositional question, but “no” to the sub-question. This paper presents a dialog-like reasoning method for maintaining reasoning consistency in answering a compositional question and its sub-questions. Our method integrates the reasoning processes for the sub-questions into the reasoning process for the compositional question like a dialog task, and uses a consistency constraint to penalize inconsistent answer predictions. In order to enable quantitative evaluation of reasoning consistency, we construct a GQA-Sub dataset based on the well-organized GQA dataset. Experimental results on the GQA dataset and the GQA-Sub dataset demonstrate the effectiveness of our method.

----

## [498] MLSLT: Towards Multilingual Sign Language Translation

**Authors**: *Aoxiong Yin, Zhou Zhao, Weike Jin, Meng Zhang, Xingshan Zeng, Xiaofei He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00505](https://doi.org/10.1109/CVPR52688.2022.00505)

**Abstract**:

Most of the research to date focuses on bilingual sign language translation (BSLT). However, such models are in-efficient in building multilingual sign language translation systems. To solve this problem, we introduce the multilin-gual sign language translation (MSLT) task. It aims to use a single model to complete the translation between multiple sign languages and spoken languages. Then, we propose MLSLT, the first MSLT model, which contains two novel dy-namic routing mechanisms for controlling the degree ofpa-rameter sharing between different languages. Intra-layer language-specific routing controls the proportion of data flowing through shared parameters and language-specific parameters from the token level through a soft gate within the layer, and inter-layer language-specific routing controls and learns the data flow path of different languages at the language level through a soft gate between layers. In order to evaluate the performance of MLSLT, we collect the first publicly available multilingual sign language understanding dataset, Spreadthesign-Ten (SP-10), which contains up to 100 language pairs, e.g., CSL→en, GSG→zh. Experi-mental results show that the average performance of ML-SLT outperforms the baseline MSLT model and the com-bination of multiple BSLT models in many cases. In ad-dition, we also explore zero-shot translation in sign language and find that our model can achieve comparable performance to the supervised BSLT model on some language pairs. Dataset and more details are at https://mlslt.github.io/.

----

## [499] A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation

**Authors**: *Yutong Chen, Fangyun Wei, Xiao Sun, Zhirong Wu, Stephen Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00506](https://doi.org/10.1109/CVPR52688.2022.00506)

**Abstract**:

This paper proposes a simple transfer learning baseline for sign language translation. Existing sign language datasets (e.g. PHOENIX-2014T, CSL-Daily) contain only about 10 K-20K pairs of sign videos, gloss annotations and texts, which are an order of magnitude smaller than typical parallel data for training spoken language translation models. Data is thus a bottleneck for training effective sign language translation models. To mitigate this problem, we propose to progressively pretrain the model from general- domain datasets that include a large amount of external supervision to within-domain datasets. Concretely, we pretrain the sign-to-gloss visual network on the general domain of human actions and the within-domain of a sign-to-gloss dataset, and pretrain the gloss-to-text translation network on the general domain of a multilingual corpus and the within-domain of a gloss-to-text corpus. The joint model is fine-tuned with an additional module named the visual-language mapper that connects the two networks. This simple baseline surpasses the previous state-of-the-art results on two sign language translation benchmarks, demonstrating the effectiveness of transfer learning. With its simplicity and strong performance, this approach can serve as a solid baseline for future research.

----

## [500] C2SLR: Consistency-enhanced Continuous Sign Language Recognition

**Authors**: *Ronglai Zuo, Brian Mak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00507](https://doi.org/10.1109/CVPR52688.2022.00507)

**Abstract**:

The backbone of most deep-learning-based continuous sign language recognition (CSLR) models consists of a visual module, a sequential module, and an alignment module. However, such CSLR backbones are hard to be trained sufficiently with a single connectionist temporal classification loss. In this work, we propose two auxiliary constraints to enhance the CSLR backbones from the perspective of consistency. The first constraint aims to enhance the visual module, which easily suffers from the insufficient training problem. Specifically, since sign languages convey information mainly with signers' faces and hands, we insert a keypoint-guided spatial attention module into the visual module to enforce it to focus on informative regions, i.e., spatial attention consistency. Nevertheless, only enhancing the visual module may not fully exploit the power of the backbone. Motivated by that both the output features of the visual and sequential modules represent the same sentence, we further impose a sentence embedding consistency constraint between them to enhance the representation power of both the features. Experimental results over three representative backbones validate the effectiveness of the two constraints. More remarkably, with a transformer-based backbone, our model achieves state-of-the-art or competitive performance on three benchmarks, PHOENIX-2014, PHOENIX-2014-T, and CSL.

----

## [501] Signing at Scale: Learning to Co-Articulate Signs for Large-Scale Photo-Realistic Sign Language Production

**Authors**: *Ben Saunders, Necati Cihan Camgöz, Richard Bowden*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00508](https://doi.org/10.1109/CVPR52688.2022.00508)

**Abstract**:

Sign languages are visual languages, with vocabularies as rich as their spoken language counterparts. However, current deep-learning based Sign Language Production (SLP) models produce under-articulated skeleton pose sequences from constrained vocabularies and this limits applicability. To be understandable and accepted by the deaf, an automatic SLP system must be able to generate co-articulated photo-realistic signing sequences for large domains of discourse. In this work, we tackle large-scale SLP by learning to co-articulate between dictionary signs, a method capable of producing smooth signing while scaling to unconstrained domains of discourse. To learn sign co-articulation, we propose a novel Frame Selection Network (FS-NET) that improves the temporal alignment of interpolated dictionary signs to continuous signing sequences. Additionally, we propose SIGNGAN, a pose-conditioned human synthesis model that produces photo-realistic sign language videos direct from skeleton pose. We propose a novel keypoint-based loss function which improves the quality of synthe-sized hand images. We evaluate our SLP model on the large-scale meineDGS (mDGS) corpus, conducting extensive user evaluation showing our FS-NET approach improves coarticulation of interpolated dictionary signs. Additionally, we show that SIGNGAN significantly outperforms all baseline methods for quantitative metrics, human perceptual studies and native deaf signer comprehension.

----

## [502] Generating Diverse and Natural 3D Human Motions from Text

**Authors**: *Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, Li Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00509](https://doi.org/10.1109/CVPR52688.2022.00509)

**Abstract**:

Automated generation of 3D human motions from text is a challenging problem. The generated motions are expected to be sufficiently diverse to explore the text-grounded motion space, and more importantly, accurately depicting the content in prescribed text descriptions. Here we tackle this problem with a two-stage approach: text2length sampling and text2motion generation. Text2length involves sampling from the learned distribution function of motion lengths conditioned on the input text. This is followed by our text2motion module using temporal variational autoen-coder to synthesize a diverse set of human motions of the sampled lengths. Instead of directly engaging with pose sequences, we propose motion snippet code as our internal motion representation, which captures local semantic motion contexts and is empirically shown to facilitate the generation of plausible motions faithful to the input text. Moreover, a large-scale dataset of scripted 3D Human motions, HumanML3D, is constructed, consisting of 14,616 motion clips and 44,970 text descriptions.

----

## [503] Sub-word Level Lip Reading With Visual Attention

**Authors**: *K. R. Prajwal, Triantafyllos Afouras, Andrew Zisserman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00510](https://doi.org/10.1109/CVPR52688.2022.00510)

**Abstract**:

The goal of this paper is to learn strong lip reading models that can recognise speech in silent videos. Most prior works deal with the open-set visual speech recognition problem by adapting existing automatic speech recognition techniques on top of trivially pooled visual features. Instead, in this paper, we focus on the unique challenges encountered in lip reading and propose tailored solutions. To this end, we make the following contributions: (1) we propose an attention-based pooling mechanism to aggregate visual speech representations; (2) we use sub-word units for lip reading for the first time and show that this allows us to better model the ambiguities of the task; (3) we propose a model for Visual Speech Detection (VSD), trained on top of the lip reading network. Following the above, we obtain state-of-the-art results on the challenging LRS2 and LRS3 benchmarks when training on public datasets, and even surpass models trained on large-scale industrial datasets by using an order of magnitude less data. Our best model achieves 22.6% word error rate on the LRS2 dataset, a performance unprecedented for lip reading models, significantly reducing the performance gap between lip reading and automatic speech recognition. Moreover, on the AVA-ActiveSpeaker benchmark, our VSD model surpasses all visual-only baselines and even outperforms several recent audio-visual methods.

----

## [504] Habitat-Web: Learning Embodied Object-Search Strategies from Human Demonstrations at Scale

**Authors**: *Ram Ramrakhya, Eric Undersander, Dhruv Batra, Abhishek Das*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00511](https://doi.org/10.1109/CVPR52688.2022.00511)

**Abstract**:

We present a large-scale study of imitating human demonstrations on tasks that require a virtual robot to search for objects in new environments - (1) ObjectGoal Navigation (e.g. 'find & go to a chair’) and (2) Pick&place (e.g. 'find mug, pick mug, find counter, place mug on counter’). First, we develop a virtual teleoperation data-collection infrastructure - connecting Habitat simulator running in a web browser to Amazon Mechanical Turk, allowing remote users to teleoperate virtual robots, safely and at scale. We collect 80k demonstrations for OBJECTNAV and 12k demonstrations for PICK&PLACE, which is an order of magnitude larger than existing human demonstration datasets in simulation or on real robots. Our virtual teleoperation data contains 29.3M actions, and is equivalent to 22.6k hours of real-world teleoperation time, and illustrates rich, diverse strategies for solving the tasks. Second, we use this data to answer the question - how does large-scale imitation learning (IL) (which has not been hitherto possible) compare to reinforcement learning (RL) (which is the status quo)? On OBJECTNAV, we find that IL (with no bells or whistles) using 70k human demonstrations outperforms RL using 240k agent-gathered trajectories. This effectively establishes an ‘exchange rate’ - a single human demonstration appears to be worth ~4 agent-gathered ones. More importantly, we find the IL-trained agent learns efficient object-search behavior from humans - it peeks into rooms, checks corners for small objects, turns in place to get a panoramic view - none of these are exhibited as prominently by the RL agent, and to induce these behaviors via contemporary RL techniques would require tedious reward engineering. Finally, accuracy vs. training data size plots show promising scaling behavior, suggesting that simply collecting more demonstrations is likely to advance the state of art further. On PICK&PLACE, the comparison is starker - IL agents achieve ~18% success on episodes with new object-receptacle locations when trained with 9.5k human demonstrations, while RL agents fail to get beyond 0%. Overall, our work provides compelling evidence for investing in large-scale imitation learning.

----

## [505] ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval

**Authors**: *Mengjun Cheng, Yipeng Sun, Longchao Wang, Xiongwei Zhu, Kun Yao, Jie Chen, Guoli Song, Junyu Han, Jingtuo Liu, Errui Ding, Jingdong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00512](https://doi.org/10.1109/CVPR52688.2022.00512)

**Abstract**:

Visual appearance is considered to be the most important cue to understand images for cross-modal retrieval, while sometimes the scene text appearing in images can provide valuable information to understand the visual semantics. Most of existing cross-modal retrieval approaches ignore the usage of scene text information and directly adding this information may lead to performance degradation in scene text free scenarios. To address this issue, we propose a full transformer architecture to unify these cross-modal retrieval scenarios in a single Vision and Scene Text Aggregation framework (ViSTA). Specifically, ViSTA utilizes transformer blocks to directly encode image patches and fuse scene text embedding to learn an aggregated visual representation for cross-modal retrieval. To tackle the modality missing problem of scene text, we propose a novel fusion token based transformer aggregation approach to exchange the necessary scene text information only through the fusion token and concentrate on the most important features in each modality. To further strengthen the visual modality, we develop dual contrastive learning losses to embed both image-text pairs and fusion-text pairs into a common cross-modal space. Compared to existing methods, ViSTA enables to aggregate relevant scene text semantics with visual appearance, and hence improve results under both scene text free and scene text aware scenarios. Experimental results show that ViSTA outperforms other methods by at least 8.4% at Recall@ 1 for scene text aware retrieval task. Compared with state-of-the-art scene text free retrieval methods, ViSTA can achieve better accuracy on Flicker30K and MSCOCO while running at least three times faster during the inference stage, which validates the effectiveness of the proposed framework.

----

## [506] Cross Modal Retrieval with Querybank Normalisation

**Authors**: *Simion-Vlad Bogolin, Ioana Croitoru, Hailin Jin, Yang Liu, Samuel Albanie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00513](https://doi.org/10.1109/CVPR52688.2022.00513)

**Abstract**:

Profiting from large-scale training datasets, advances in neural architecture design and efficient inference, joint embeddings have become the dominant approach for tackling cross-modal retrieval. In this work we first show that, despite their effectiveness, state-of-the-art joint embeddings suffer significantly from the longstanding “hubness problem” in which a small number of gallery embeddings form the nearest neighbours of many queries. Drawing inspiration from the NLP literature, we formulate a simple but effective framework called Querybank Normalisation (QB-NORM) that re-normalises query similarities to account for hubs in the embedding space. QB-NORM improves retrieval performance without requiring retraining. Differently from prior work, we show that QB-NORM works effectively without concurrent access to any test set queries. Within the QB-NORM framework, we also propose a novel similarity normalisation method, the Dynamic Inverted Softmax, that is significantly more robust than existing approaches. We showcase QB-NORM across a range of cross modal retrieval models and benchmarks where it consistently enhances strong baselines beyond the state of the art. Code is available at https://vladbogo.github.io/QB-Norm/.

----

## [507] Prompt Distribution Learning

**Authors**: *Yuning Lu, Jianzhuang Liu, Yonggang Zhang, Yajing Liu, Xinmei Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00514](https://doi.org/10.1109/CVPR52688.2022.00514)

**Abstract**:

We present prompt distribution learning for effectively adapting a pre-trained vision-language model to address downstream recognition tasks. Our method not only learns low-bias prompts from a few samples but also captures the distribution of diverse prompts to handle the varying visual representations. In this way, we provide high-quality task-related content for facilitating recognition. This prompt distribution learning is realized by an efficient approach that learns the output embeddings of prompts instead of the input embeddings. Thus, we can employ a Gaussian distribution to model them effectively and derive a surrogate loss for efficient training. Extensive experiments on 12 datasets demonstrate that our method consistently and significantly outperforms existing methods. For example, with 1 sample per category, it relatively improves the average result by 9.1% compared to human-crafted prompts.

----

## [508] VALHALLA: Visual Hallucination for Machine Translation

**Authors**: *Yi Li, Rameswar Panda, Yoon Kim, Chun-Fu Richard Chen, Rogério Feris, David D. Cox, Nuno Vasconcelos*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00515](https://doi.org/10.1109/CVPR52688.2022.00515)

**Abstract**:

Designing better machine translation systems by considering auxiliary inputs such as images has attracted much attention in recent years. While existing methods show promising performance over the conventional text-only translation systems, they typically require paired text and image as input during inference, which limits their applicability to real-world scenarios. In this paper, we introduce a visual hallucination framework, called VALHALLA, which requires only source sentences at inference time and instead uses hallucinated visual representations for multi-modal machine translation. In particular, given a source sentence an autoregressive hallucination transformer is used to predict a discrete visual representation from the input text, and the combined text and hallucinated representations are utilized to obtain the target translation. We train the hallucination transformer jointly with the translation transformer using standard backpropagation with crossentropy losses while being guided by an additional loss that encourages consistency between predictions using either groundtruth or hallucinated visual representations. Extensive experiments on three standard translation datasets with a diverse set of language pairs demonstrate the effectiveness of our approach over both text-only baselines and state-of-the-art methods. Project page: http://www.svcl.ucsd.jects/valhalla.edu/pro.

----

## [509] VL-ADAPTER: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks

**Authors**: *Yi-Lin Sung, Jaemin Cho, Mohit Bansal*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00516](https://doi.org/10.1109/CVPR52688.2022.00516)

**Abstract**:

Recently, fine-tuning language models pre-trained on large text corpora have provided huge improvements on vision-and-language (V&L) tasks as well as on pure language tasks. However, fine-tuning the entire parameter set of pre-trained models becomes impractical since the model size is growing rapidly. Hence, in this paper, we introduce adapter-based parameter-efficient transfer learning techniques to V&L models such as VL-BART and VL-T5. We evaluate our methods in a unified multi-task setup on both image-text and video-text benchmarks. For the image-text tasks, we use four diverse V&L datasets: VQAv2, GQA, NLVR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
, and MSCOCO image captioning. For video-text tasks, we use TVQA, How2QA, TVC, and YC2C. With careful training and thorough experiments, we benchmark three popular adapter-based methods (Adapter, Hyperformer, Compacter) against the standard full fine-tuning and the recently proposed prompt-tuning approach. We also enhance the efficiency and performance of adapters by sharing their weights to attain knowledge across tasks. Our results demonstrate that training the adapter with the weight-sharing technique (4.18% of total parameters for image-text tasks and 3.39% for video-text tasks) can match the performance of fine-tuning the entire model. Lastly, we present a comprehensive analysis including the combination of adapter and task-specific prompts and the impact of V&L pre-training on adapters. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code for our CVPR 2022 paper is available at: https://github.com/ylsung/VL_adapter.

----

## [510] Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality

**Authors**: *Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, Candace Ross*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00517](https://doi.org/10.1109/CVPR52688.2022.00517)

**Abstract**:

We present a novel task and dataset for evaluating the ability of vision and language models to conduct visio-linguistic compositional reasoning, which we call Winoground. Given two images and two captions, the goal is to match them correctly-but crucially, both captions contain a completely identical set of words, only in a different order. The dataset was carefully hand-curated by expert annotators and is labeled with a rich set offine-grained tags to assist in analyzing model performance. We probe a diverse range of state-of-the-art vision and language models and find that, surprisingly, none of them do much better than chance. Evidently, these models are not as skilled at visio-linguistic compositional reasoning as we might have hoped. We perform an extensive analysis to obtain insights into how future work might try to mitigate these models' shortcomings. We aim for Winoground to serve as a useful evaluation set for advancing the state of the art and driving further progress in the field. The dataset is available at https://huggingface.co/datasets/facebook/winoground.

----

## [511] MixFormer: Mixing Features across Windows and Dimensions

**Authors**: *Qiang Chen, Qiman Wu, Jian Wang, Qinghao Hu, Tao Hu, Errui Ding, Jian Cheng, Jingdong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00518](https://doi.org/10.1109/CVPR52688.2022.00518)

**Abstract**:

While local-window self-attention performs notably in vision tasks, it suffers from limited receptive field and weak modeling capability issues. This is mainly because it performs self-attention within non-overlapped windows and shares weights on the channel dimension. We propose Mix-Former to find a solution. First, we combine local-window self-attention with depth-wise convolution in a parallel design, modeling cross-window connections to enlarge the receptive fields. Second, we propose bi-directional interactions across branches to provide complementary clues in the channel and spatial dimensions. These two designs are integrated to achieve efficient feature mixing among windows and dimensions. Our MixFormer provides competitive results on image classification with EfficientNet and shows better results than RegNet and Swin Transformer. Performance in downstream tasks outperforms its alternatives by significant margins with less computational costs in 5 dense prediction tasks on MS COCO, ADE20k, and LVIS. Code is available at https://github.com/PaddlePaddle/PaddleClas.

----

## [512] Recurrent Glimpse-based Decoder for Detection with Transformer

**Authors**: *Zhe Chen, Jing Zhang, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00519](https://doi.org/10.1109/CVPR52688.2022.00519)

**Abstract**:

Although detection with Transformer (DETR) is increasingly popular, its global attention modeling requires an extremely long training period to optimize and achieve promising detection performance. Alternative to existing studies that mainly develop advanced feature or embedding designs to tackle the training issue, we point out that the Region-of-Interest (RoI) based detection refinement can easily help mitigate the difficulty of training for DETR methods. Based on this, we introduce a novel REcurrent Glimpse-based decOder (REGO) in this paper. In particular, the REGO employs a multi-stage recurrent processing structure to help the attention of DETR gradually focus on foreground objects more accurately. In each processing stage, visual features are extracted as glimpse features from RoIs with enlarged bounding box areas of detection results from the previous stage. Then, a glimpse-based decoder is introduced to provide refined detection results based on both the glimpse features and the attention modeling outputs of the previous stage. In practice, REGO can be easily embedded in representative DETR variants while maintaining their fully end-to-end training and inference pipelines. In particular, REGO helps Deformable DETR achieve 44.8 AP on the MSCOCO dataset with only 36 training epochs, compared with the first DETR and the Deformable DETR that require 500 and 50 epochs to achieve comparable performance, respectively. Experiments also show that REGO consistently boosts the performance of different DETR detectors by up to 7% relative gain at the same setting of 50 training epochs. Code is available via https://github.com/zhechen/Deformable-DETR-REGO.

----

## [513] Mobile-Former: Bridging MobileNet and Transformer

**Authors**: *Yinpeng Chen, Xiyang Dai, Dongdong Chen, Mengchen Liu, Xiaoyi Dong, Lu Yuan, Zicheng Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00520](https://doi.org/10.1109/CVPR52688.2022.00520)

**Abstract**:

We present Mobile-Former, a parallel design of MobileNet and transformer with a two-way bridge in between. This structure leverages the advantages of MobileNet at local processing and transformer at global interaction. And the bridge enables bidirectional fusion of local and global features. Different from recent works on vision transformer, the transformer in Mobile-Former contains very few tokens (e.g. 6 or fewer tokens) that are randomly initialized to learn global priors, resulting in low computational cost. Combining with the proposed light-weight cross attention to model the bridge, Mobile-Former is not only computationally efficient, but also has more representation power. It outperforms MobileNetV3 at low FLOP regime from 25M to 500M FLOPs on ImageNet classification. For instance, Mobile-Former achieves 77.9% top-1 accuracy at 294M FLOPs, gaining 1.3% over MobileNetV3 but saving 17% of computations. When transferring to object detection, Mobile-Former outperforms MobileNetV3 by 8.6 AP in RetinaNet framework. Furthermore, we build an efficient end-to-end detector by replacing backbone, encoder and decoder in DETR with Mobile-Former, which outperforms DETR by 1.3 AP but saves 52% of computational cost and 36% of parameters. Code will be released at https://github.com/aaboys/mobileformer.

----

## [514] Unsupervised Domain Generalization by Learning a Bridge Across Domains

**Authors**: *Sivan Harary, Eli Schwartz, Assaf Arbelle, Peter W. J. Staar, Shady Abu-Hussein, Elad Amrani, Roei Herzig, Amit Alfassy, Raja Giryes, Hilde Kuehne, Dina Katabi, Kate Saenko, Rogério Feris, Leonid Karlinsky*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00521](https://doi.org/10.1109/CVPR52688.2022.00521)

**Abstract**:

The ability to generalize learned representations across significantly different visual domains, such as between real photos, clipart, paintings, and sketches, is a fundamental capacity of the human visual system. In this paper, different from most cross-domain works that utilize some (or full) source domain supervision, we approach a relatively new and very practical Unsupervised Domain Generalization (UDG) setup of having no training supervision in neither source nor target domains. Our approach is based on self-supervised learning of a Bridge Across Domains (BrAD) - an auxiliary bridge domain accompanied by a set of semantics preserving visual (image-to-image) mappings to BrAD from each of the training domains. The BrAD and mappings to it are learned jointly (end-to-end) with a contrastive self-supervised representation model that semantically aligns each of the domains to its BrAD-projection, and hence implicitly drives all the domains (seen or unseen) to semantically align to each other. In this work, we show how using an edge-regularized BrAD our approach achieves significant gains across multiple benchmarks and a range of tasks, including UDG, Few-shot UDA, and unsupervised generalization across multi-domain datasets (including generalization to unseen domains and classes).

----

## [515] SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection

**Authors**: *Wuyang Li, Xinyu Liu, Yixuan Yuan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00522](https://doi.org/10.1109/CVPR52688.2022.00522)

**Abstract**:

Domain Adaptive Object Detection (DAOD) leverages a labeled domain to learn an object detector generalizing to a novel domain free of annotations. Recent advances align class-conditional distributions by narrowing down cross-domain prototypes (class centers). Though great success, they ignore the significant within-class variance and the domain-mismatched semantics within the training batch, leading to a sub-optimal adaptation. To overcome these challenges, we propose a novel SemantIc-complete Graph MAtching (SIGMA) framework for DAOD, which completes mismatched semantics and reformulates the adaptation with graph matching. Specifically, we design a Graph-embedded Semantic Completion module (GSC) that completes mis-matched semantics through generating hallucination graph nodes in missing categories. Then, we establish cross-image graphs to model class-conditional distributions and learn a graph-guided memory bank for better semantic completion in turn. After representing the source and target data as graphs, we reformulate the adaptation as a graph matching problem, i.e., finding well-matched node pairs across graphs to reduce the domain gap, which is solved with a novel Bipartite Graph Matching adaptor (BGM). In a nutshell, we utilize graph nodes to establish semantic-aware node affinity and leverage graph edges as quadratic constraints in a structure-aware matching loss, achieving fine-grained adaptation with a node-to-node graph matching. Extensive experiments verify that SIGMA outperforms existing works significantly. Our code is available at https://github.com/CityU-AIM-Group/SIGMA.

----

## [516] Target-Relevant Knowledge Preservation for Multi-Source Domain Adaptive Object Detection

**Authors**: *Jiaxi Wu, Jiaxin Chen, Mengzhe He, Yiru Wang, Bo Li, Bingqi Ma, Weihao Gan, Wei Wu, Yali Wang, Di Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00523](https://doi.org/10.1109/CVPR52688.2022.00523)

**Abstract**:

Domain adaptive object detection (DAOD) is a promising way to alleviate performance drop of detectors in new scenes. Albeit great effort made in single source domain adaptation, a more generalized task with multiple source domains remains not being well explored, due to knowledge degradation during their combination. To address this issue, we propose a novel approach, namely target-relevant knowledge preservation (TRKP), to unsupervised multi-source DAOD. Specifically, TRKP adopts the teacher-student framework, where the multi-head teacher network is built to extract knowledge from labeled source domains and guide the student network to learn detectors in unlabeled target domain. The teacher network is further equipped with an adversarial multi-source disentanglement (AMSD) module to preserve source domain-specific knowledge and simultaneously perform cross-domain alignment. Besides, a holistic target-relevant mining (HTRM) scheme is developed to re-weight the source images according to the source-target relevance. By this means, the teacher network is enforced to capture target-relevant knowledge, thus benefiting decreasing domain shift when mentoring object detection in the target domain. Extensive experiments are conducted on various widely used benchmarks with new state-of-the-art scores reported, highlighting the effectiveness.

----

## [517] PNP: Robust Learning from Noisy Labels by Probabilistic Noise Prediction

**Authors**: *Zeren Sun, Fumin Shen, Dan Huang, Qiong Wang, Xiangbo Shu, Yazhou Yao, Jinhui Tang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00524](https://doi.org/10.1109/CVPR52688.2022.00524)

**Abstract**:

Label noise has been a practical challenge in deep learning due to the strong capability of deep neural networks in fitting all training data. Prior literature primarily resorts to sample selection methods for combating noisy labels. However, these approaches focus on dividing samples by order sorting or threshold selection, inevitably introducing hyperparameters (e.g., selection ratio / threshold) that are hard-to-tune and dataset-dependent. To this end, we propose a simple yet effective approach named PNP (Probabilistic Noise Prediction) to explicitly model label noise. Specifically, we simultaneously train two networks, in which one predicts the category label and the other predicts the noise type. By predicting label noise probabilistically, we identify noisy samples and adopt dedicated optimization objectives accordingly. Finally, we establish a joint loss for network update by unifying the classification loss, the auxiliary constraint loss, and the in-distribution consistency loss. Comprehensive experimental results on synthetic and realworld datasets demonstrate the superiority of our proposed method. The source code and models have been made available at https://github.com/NUST-Machine-Intelligence-Laboratory/PNP.

----

## [518] Few-Shot Object Detection with Fully Cross-Transformer

**Authors**: *Guangxing Han, Jiawei Ma, Shiyuan Huang, Long Chen, Shih-Fu Chang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00525](https://doi.org/10.1109/CVPR52688.2022.00525)

**Abstract**:

Few-shot object detection (FSOD), with the aim to detect novel objects using very few training examples, has recently attracted great research interest in the community. Metric-learning based methods have been demonstrated to be effective for this task using a two-branch based siamese network, and calculate the similarity between image regions and few-shot examples for detection. However, in previous works, the interaction between the two branches is only restricted in the detection head, while leaving the remaining hundreds of layers for separate feature extraction. Inspired by the recent work on vision transformers and vision-language transformers, we propose a novel Fully Cross-Transformer based model (FCT) for FSOD by incorporating cross-transformer into both the feature backbone and detection head. The asymmetric-batched cross-attention is proposed to aggregate the key information from the two branches with different batch sizes. Our model can improve the few-shot similarity learning between the two branches by introducing the multi-level interactions. Comprehensive experiments on both PASCAL VOC and MSCOCO FSOD benchmarks demonstrate the effectiveness of our model.

----

## [519] Task Discrepancy Maximization for Fine-grained Few-Shot Classification

**Authors**: *Su Been Lee, WonJun Moon, Jae-Pil Heo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00526](https://doi.org/10.1109/CVPR52688.2022.00526)

**Abstract**:

Recognizing discriminative details such as eyes and beaks is important for distinguishing fine-grained classes since they have similar overall appearances. In this regard, we introduce Task Discrepancy Maximization (TDM), a simple module for fine-grained few-shot classification. Our objective is to localize the class-wise discriminative regions by highlighting channels encoding distinct information of the class. Specifically, TDM learns task-specific channel weights based on two novel components: Support Attention Module (SAM) and Query Attention Module (QAM). SAM produces a support weight to represent channel-wise discriminative power for each class. Still, since the SAM is basically only based on the labeled support sets, it can be vulnerable to bias toward such support set. Therefore, we propose QAM which complements SAM by yielding a query weight that grants more weight to object-relevant channels for a given query image. By combining these two weights, a class-wise task-specific channel weight is defined. The weights are then applied to produce task-adaptive feature maps more focusing on the discriminative details. Our experiments validate the effectiveness of TDM and its complementary benefits with prior methods in fine- grained few-shot classification.

----

## [520] Leveraging Self-Supervision for Cross-Domain Crowd Counting

**Authors**: *Weizhe Liu, Nikita Durasov, Pascal Fua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00527](https://doi.org/10.1109/CVPR52688.2022.00527)

**Abstract**:

State-of-the-art methods for counting people in crowded scenes rely on deep networks to estimate crowd density. While effective, these data-driven approaches rely on large amount of data annotation to achieve good performance, which stops these models from being deployed in emergencies during which data annotation is either too costly or cannot be obtained fast enough. One popular solution is to use synthetic data for training. Unfortunately, due to domain shift, the resulting models generalize poorly on real imagery. We remedy this shortcoming by training with both synthetic images, along with their associated labels, and unlabeled real images. To this end, we force our network to learn perspective-aware features by training it to recognize upside-down real images from regular ones and incorporate into it the ability to predict its own uncertainty so that it can generate useful pseudo labels for fine-tuning purposes. This yields an algorithm that consistently outperforms state-of-the-art cross-domain crowd counting ones without any extra computation at inference time. Code is publicly available at https://github.com/weizheliu/Cross-Domain-Crowd-Counting.

----

## [521] What to look at and where: Semantic and Spatial Refined Transformer for detecting human-object interactions

**Authors**: *A. S. M. Iftekhar, Hao Chen, Kaustav Kundu, Xinyu Li, Joseph Tighe, Davide Modolo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00528](https://doi.org/10.1109/CVPR52688.2022.00528)

**Abstract**:

We propose a novel one-stage Transformer-based semantic and spatial refined transformer (SSRT) to solve the Human-Object Interaction detection task, which requires to localize humans and objects, and predicts their interactions. Differently from previous Transformer-based HOI approaches, which mostly focus at improving the design of the decoder outputs for the final detection, SSRT introduces two new modules to help select the most relevant object-action pairs within an image and refine the queries' representation using rich semantic and spatial features. These enhancements lead to state-of-the-art results on the two most popular HOI benchmarks: V-COCO and HICO-DET.

----

## [522] AdaMixer: A Fast-Converging Query-Based Object Detector

**Authors**: *Ziteng Gao, Limin Wang, Bing Han, Sheng Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00529](https://doi.org/10.1109/CVPR52688.2022.00529)

**Abstract**:

Traditional object detectors employ the dense paradigm of scanning over locations and scales in an image. The recent query-based object detectors break this convention by decoding image features with a set of learnable queries. However, this paradigm still suffers from slow convergence, limited performance, and design complexity of extra networks between backbone and decoder. In this paper, we find that the key to these issues is the adaptability of decoders for casting queries to varying objects. Accordingly, we propose a fast-converging query-based detector, named AdaMixer, by improving the adaptability of query-based decoding processes in two aspects. First, each query adaptively samples features over space and scales based on estimated offsets, which allows AdaMixer to efficiently attend to the coherent regions of objects. Then, we dynamically decode these sampled features with an adaptive MLP-Mixer under the guidance of each query. Thanks to these two critical designs, AdaMixer enjoys architectural simplicity without requiring dense attentional encoders or explicit pyramid networks. On the challenging MS COCO benchmark, AdaMixer with ResNet-50 as the backbone, with 12 training epochs, reaches up to 45.0 AP on the validation set along with 27.9 APs in detecting small objects. With the longer training scheme, AdaMixer with ResNeXt-101-DCN and Swin-S reaches 49.5 and 51.3 AP. Our work sheds light on a simple, accurate, and fast converging architecture for query-based object detectors. The code is made available at https://github.com/MCG-NJU/AdaMixer.

----

## [523] Correlation Verification for Image Retrieval

**Authors**: *Seongwon Lee, Hongje Seong, Suhyeon Lee, Euntai Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00530](https://doi.org/10.1109/CVPR52688.2022.00530)

**Abstract**:

Geometric verification is considered a de facto solution for the re-ranking task in image retrieval. In this study, we propose a novel image retrieval re-ranking network named Correlation Verification Networks (CVNet). Our proposed network, comprising deeply stacked 4D convolutional layers, gradually compresses dense feature correlation into image similarity while learning diverse geometric matching patterns from various image pairs. To enable cross-scale matching, it builds feature pyramids and constructs cross-scale feature correlations within a single inference, replacing costly multi-scale inferences. In addition, we use curriculum learning with the hard negative mining and Hide-and-Seek strategy to handle hard samples without losing generality. Our proposed re-ranking network shows state-of-the-art performance on several retrieval benchmarks with a significant margin (+12.6% in mAP on ROxford-Hard+1M set) over state-of-the-art methods. The source code and models are available online: ht tps: / /gi thub. com/ sungonce/CVNet.

----

## [524] Real-time Object Detection for Streaming Perception

**Authors**: *Jinrong Yang, Songtao Liu, Zeming Li, Xiaoping Li, Jian Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00531](https://doi.org/10.1109/CVPR52688.2022.00531)

**Abstract**:

Autonomous driving requires the model to perceive the environment and (re)act within a low latency for safety. While past works ignore the inevitable changes in the environment after processing, streaming perception is proposed to jointly evaluate the latency and accuracy into a single metric for video online perception. In this paper, instead of searching trade-offs between accuracy and speed like previous works, we point out that endowing real-time models with the ability to predict the future is the key to dealing with this problem. We build a simple and effective frame-work for streaming perception. It equips a novel Dual-Flow Perception module (DFP), which includes dynamic and static flows to capture the moving trend and basic detection feature for streaming prediction. Further, we introduce a Trend-Aware Loss (TAL) combined with a trend factor to generate adaptive weights for objects with different moving speeds. Our simple method achieves competitive performance on Argoverse-HD dataset and improves the AP by 4.9% compared to the strong baseline, validating its effectiveness. Our code will be made available at https://github.com/yancie-yjr/StreamYOLO.

----

## [525] Deep Visual Geo-localization Benchmark

**Authors**: *Gabriele Moreno Berton, Riccardo Mereu, Gabriele Trivigno, Carlo Masone, Gabriela Csurka, Torsten Sattler, Barbara Caputo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00532](https://doi.org/10.1109/CVPR52688.2022.00532)

**Abstract**:

In this paper, we propose a new open-source benchmarkingframeworkfor Visual Geo-localization (VG) that allows to build, train, and test a wide range of commonly used ar-chitectures, with the flexibility to change individual components of a geo-localization pipeline. The purpose of this framework is twofold: i) gaining insights into how differ-ent components and design choices in a VG pipeline im-pact the final results, both in terms of performance (re-call@N metric) and system requirements (such as execution time and memory consumption); ii) establish a system-atic evaluation protocol for comparing different methods. Using the proposed framework, we perform a large suite of experiments which provide criteria for choosing back-bone, aggregation and negative mining depending on the use-case and requirements. We also assess the impact of engineering techniques like pre/post-processing, data aug-mentation and image resizing, showing that better performance can be obtained through somewhat simple procedures: for example, downscaling the images' resolution to 80% can lead to similar results with a 36% savings in ex-traction time and dataset storage requirement. Code and trained models are available at dataset storage requirement. https://deep-vg-bench.herokuapp.com/.

----

## [526] RendNet: Unified 2D/3D Recognizer with Latent Space Rendering

**Authors**: *Ruoxi Shi, Xinyang Jiang, Caihua Shan, Yansen Wang, Dongsheng Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00533](https://doi.org/10.1109/CVPR52688.2022.00533)

**Abstract**:

Vector graphics (VG) have been ubiquitous in our daily life with vast applications in engineering, architecture, designs, etc. The VG recognition process of most existing methods is to first render the VG into raster graphics (RG) and then conduct recognition based on RG formats. However, this procedure discards the structure of geometries and loses the high resolution of VG. Recently, another category of algorithms is proposed to recognize directly from the original VG format. But it is affected by the topological errors that can be filtered out by RG rendering. Instead of looking at one format, it is a good solution to utilize the formats of VG and RG together to avoid these shortcomings. Besides, we argue that the VG-to-RG rendering process is essential to effectively combine VG and RG information. By specifying the rules on how to transfer VG primitives to RG pixels, the rendering process depicts the interaction and correlation between VG and RG. As a result, we propose RendNet, a unified architecture for recognition on both 2D and 3D scenarios, which considers both VG/RG representations and exploits their interaction by incorporating the VG-to-RG rasterization process. Experiments show that Rend-Net can achieve state-of-the-art performance on 2D and 3D object recognition tasks on various VG datasets.

----

## [527] Sparse Fuse Dense: Towards High Quality 3D Detection with Depth Completion

**Authors**: *Xiaopei Wu, Liang Peng, Honghui Yang, Liang Xie, Chenxi Huang, Chengqi Deng, Haifeng Liu, Deng Cai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00534](https://doi.org/10.1109/CVPR52688.2022.00534)

**Abstract**:

Current LiDAR-only 3D detection methods inevitably suffer from the sparsity of point clouds. Many multi-modal methods are proposed to alleviate this issue, while different representations of images and point clouds make it difficult to fuse them, resulting in suboptimal performance. In this paper, we present a novel multi-modal framework SFD (Sparse Fuse Dense), which utilizes pseudo point clouds generated from depth completion to tackle the issues mentioned above. Different from prior works, we propose a new RoI fusion strategy 3D-GAF (3D Grid-wise Attentive Fusion) to make fuller use of information from different types of point clouds. Specifically, 3D-GAF fuses 3D RoI features from the pair of point clouds in a grid-wise attentive way, which is more fine- grained and more precise. In addition, we propose a SynAugment (Synchronized Augmentation) to enable our multi-modal framework to utilize all data augmentation approaches tailored to LiDAR-only methods. Lastly, we customize an effective and efficient feature extractor CPConv (Color Point Convolution) for pseudo point clouds. It can explore 2D image features and 3D geometric features of pseudo point clouds simultaneously. Our method holds the highest entry on the KITTI car 3D object detection leaderboard
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
†On the date of CVPR deadline, i.e., Nov.16, 2021, demonstrating the effectiveness of our SFD. Code will be made publicly available.

----

## [528] Focal Sparse Convolutional Networks for 3D Object Detection

**Authors**: *Yukang Chen, Yanwei Li, Xiangyu Zhang, Jian Sun, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00535](https://doi.org/10.1109/CVPR52688.2022.00535)

**Abstract**:

Non-uniformed 3D sparse data, e.g., point clouds or voxels in different spatial positions, make contribution to the task of 3D object detection in different ways. Existing basic components in sparse convolutional networks (Sparse CNNs) process all sparse data, regardless of regular or submanifold sparse convolution. In this paper, we introduce two new modules to enhance the capability of Sparse CNNs, both are based on making feature sparsity learnable with position-wise importance prediction. They are focal sparse convolution (Focals Conv) and its multi-modal variant of focal sparse convolution with fusion, or Focals Conv-F for short. The new modules can readily substitute their plain counterparts in existing Sparse CNNs and be jointly trained in an end-to-end fashion. For the first time, we show that spatially learnable sparsity in sparse convolution is essential for sophisticated 3D object detection. Extensive experiments on the KITTI, nuScenes and Waymo benchmarks validate the effectiveness of our approach. Without bells and whistles, our results outperform all existing single-model entries on the nuScenes test benchmark. Code and models are at github.com/dvlab-research/FocalsConv.

----

## [529] Point-NeRF: Point-based Neural Radiance Fields

**Authors**: *Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, Ulrich Neumann*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00536](https://doi.org/10.1109/CVPR52688.2022.00536)

**Abstract**:

Volumetric neural rendering methods like NeRF [34] generate high-quality view synthesis results but are optimized per-scene leading to prohibitive reconstruction time. On the other hand, deep multi-view stereo methods can quickly reconstruct scene geometry via direct network inference. Point-NeRF combines the advantages of these two approaches by using neural 3D point clouds, with associated neural features, to model a radiance field. Point-NeRF can be rendered efficiently by aggregating neural point features near scene surfaces, in a ray marching-based rendering pipeline. Moreover, Point-NeRF can be initialized via direct inference of a pre-trained deep network to produce a neural point cloud; this point cloud can be finetuned to surpass the visual quality of NeRF with 30× faster training time. Point-NeRF can be combined with other 3D re-construction methods and handles the errors and outliers in such methods via a novel pruning and growing mechanism.

----

## [530] NeRFusion: Fusing Radiance Fields for Large-Scale Scene Reconstruction

**Authors**: *Xiaoshuai Zhang, Sai Bi, Kalyan Sunkavalli, Hao Su, Zexiang Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00537](https://doi.org/10.1109/CVPR52688.2022.00537)

**Abstract**:

While NeRF [28] has shown great success for neural reconstruction and rendering, its limited MLP capacity and long per-scene optimization times make it challenging to model large-scale indoor scenes. In contrast, classical 3D reconstruction methods can handle large-scale scenes but do not produce realistic renderings. We propose NeRFusion, a method that combines the advantages of NeRF and TSDF-based fusion techniques to achieve efficient large-scale reconstruction and photo-realistic rendering. We process the input image sequence to predict per-frame local radiance fields via direct network inference. These are then fused using a novel recurrent neural network that incrementally reconstructs a global, sparse scene representation in real-time at 22 fps. This global volume can be further fine-tuned to boost rendering quality. We demonstrate that NeR-Fusionachieves state-of-the-art quality on both large-scale indoor and small-scale object scenes, with substantially faster reconstruction than NeRF and other recent methods.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://jetd1.github.io/NeRFusion-Web/

----

## [531] Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction

**Authors**: *Cheng Sun, Min Sun, Hwann-Tzong Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00538](https://doi.org/10.1109/CVPR52688.2022.00538)

**Abstract**:

We present a super-fast convergence approach to reconstructing the per-scene radiance field from a set of images that capture the scene with known poses. This task, which is often applied to novel view synthesis, is recently revolution-ized by Neural Radiance Field (NeRF) for its state-of-the-art quality and fiexibility. However, NeRF and its variants require a lengthy training time ranging from hours to days for a single scene. In contrast, our approach achieves NeRF-comparable quality and converges rapidly from scratch in less than 15 minutes with a single GPU. We adopt a representation consisting of a density voxel grid for scene geometry and a feature voxel grid with a shallow network for complex view-dependent appearance. Modeling with explicit and discretized volume representations is not new, but we propose two simple yet non-trivial techniques that contribute to fast convergence speed and high-quality output. First, we introduce the post-activation interpolation on voxel density, which is capable of producing sharp surfaces in lower grid resolution. Second, direct voxel density optimization is prone to suboptimal geometry solutions, so we robustify the optimization process by imposing several priors. Finally, evaluation on five inward-facing benchmarks shows that our method matches, if not surpasses, NeRF's quality, yet it only takes about 15 minutes to train from scratch for a new scene. Code: https://github.com/sunset1995/DirectVoxGO.

----

## [532] Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields

**Authors**: *Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, Peter Hedman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00539](https://doi.org/10.1109/CVPR52688.2022.00539)

**Abstract**:

Though neural radiance fields (NeRF) have demon-strated impressive view synthesis results on objects and small bounded regions of space, they struggle on “un-bounded” scenes, where the camera may point in any di-rection and content may exist at any distance. In this set-ting, existing NeRF-like models often produce blurry or low-resolution renderings (due to the unbalanced detail and scale of nearby and distant objects), are slow to train, and may exhibit artifacts due to the inherent ambiguity of the task of reconstructing a large scene from a small set of images. We present an extension of mip-NeRF (a NeRF variant that addresses sampling and aliasing) that uses a non-linear scene parameterization, online distillation, and a novel distortion-based regularizer to overcome the chal-lenges presented by unbounded scenes. Our model, which we dub “mip-NeRF 360” as we target scenes in which the camera rotates 360 degrees around a point, reduces mean-squared error by 57% compared to mip-NeRF, and is able to produce realistic synthesized views and detailed depth maps for highly intricate, unbounded real-world scenes.

----

## [533] RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs

**Authors**: *Michael Niemeyer, Jonathan T. Barron, Ben Mildenhall, Mehdi S. M. Sajjadi, Andreas Geiger, Noha Radwan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00540](https://doi.org/10.1109/CVPR52688.2022.00540)

**Abstract**:

Neural Radiance Fields (NeRF) have emerged as a powerful representation for the task of novel view synthesis due to their simplicity and state-of-the-art performance. Though NeRF can produce photorealistic renderings of unseen viewpoints when many input views are available, its performance drops significantly when this number is reduced. We observe that the majority of artifacts in sparse input scenarios are caused by errors in the estimated scene geometry, and by divergent behavior at the start of training. We address this by regularizing the geometry and appearance of patches rendered from unobserved viewpoints, and annealing the ray sampling space during training. We additionally use a normalizing flow model to regularize the color of unobserved viewpoints. Our model outperforms not only other methods that optimize over a single scene, but in many cases also conditional models that are extensively pre-trained on large multi-view datasets.

----

## [534] Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields

**Authors**: *Dor Verbin, Peter Hedman, Ben Mildenhall, Todd E. Zickler, Jonathan T. Barron, Pratul P. Srinivasan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00541](https://doi.org/10.1109/CVPR52688.2022.00541)

**Abstract**:

Neural Radiance Fields (NeRF) is a popular view synthesis technique that represents a scene as a continuous volumetric function, parameterized by multilayer perceptrons that provide the volume density and view-dependent emitted radiance at each location. While NeRF-based techniques excel at representing fine geometric structures with smoothly varying view-dependent appearance, they often fail to accurately capture and reproduce the appearance of glossy surfaces. We address this limitation by introducing Ref-NeRF, which replaces NeRF's parameterization of view-dependent outgoing radiance with a representation of reflected radiance and structures this function using a collection of spatially-varying scene properties. We show that together with a regularizer on normal vectors, our model significantly improves the realism and accuracy of specular reflections. Furthermore, we show that our model's internal representation of outgoing radiance is interpretable and useful for scene editing.

----

## [535] Plenoxels: Radiance Fields without Neural Networks

**Authors**: *Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, Angjoo Kanazawa*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00542](https://doi.org/10.1109/CVPR52688.2022.00542)

**Abstract**:

We introduce Plenoxels (plenoptic voxels), a systemfor photorealistic view synthesis. Plenoxels represent a scene as a sparse 3D grid with spherical harmonics. This representation can be optimized from calibrated images via gradient methods and regularization without any neural components. On standard, benchmark tasks, Plenoxels are optimized two orders of magnitude faster than Neural Radiance Fields with no loss in visual quality. For video and code, please see https://alexyu.net/plenoxels.

----

## [536] Neural 3D Scene Reconstruction with the Manhattan-world Assumption

**Authors**: *Haoyu Guo, Sida Peng, Haotong Lin, Qianqian Wang, Guofeng Zhang, Hujun Bao, Xiaowei Zhou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00543](https://doi.org/10.1109/CVPR52688.2022.00543)

**Abstract**:

This paper addresses the challenge of reconstructing 3D indoor scenes from multi-view images. Many previous works have shown impressive reconstruction results on textured objects, but they still have difficulty in handling low-textured planar regions, which are common in indoor scenes. An approach to solving this issue is to incorporate planer constraints into the depth map estimation in multiview stereo-based methods, but the per-view plane estimation and depth optimization lack both efficiency and multiview consistency. In this work, we show that the planar constraints can be conveniently integrated into the recent implicit neural representation-based reconstruction methods. Specifically, we use an MLP network to represent the signed distance function as the scene geometry. Based on the Manhattan-world assumption, planar constraints are employed to regularize the geometry in floor and wall regions predicted by a 2D semantic segmentation network. To resolve the inaccurate segmentation, we encode the semantics of 3D points with another MLP and design a novel loss that jointly optimizes the scene geometry and semantics in 3D space. Experiments on ScanNet and 7-Scenes datasets show that the proposed method outperforms previous methods by a large margin on 3D reconstruction quality. The code and supplementary materials are available at https://zju3dv.github.io/manhattan_sdf.

----

## [537] Neural 3D Video Synthesis from Multi-view Video

**Authors**: *Tianye Li, Mira Slavcheva, Michael Zollhöfer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard A. Newcombe, Zhaoyang Lv*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00544](https://doi.org/10.1109/CVPR52688.2022.00544)

**Abstract**:

We propose a novel approach for 3D video synthesis that is able to represent multi-view video recordings of a dynamic real-world scene in a compact, yet expressive representation that enables high-quality view synthesis and motion interpolation. Our approach takes the high quality and compactness of static neural radiance fields in a new direction: to a model-free, dynamic setting. At the core of our approach is a novel time-conditioned neural radiance field that represents scene dynamics using a set of compact latent codes. We are able to significantly boost the training speed and perceptual quality of the generated imagery by a novel hierarchical training scheme in combination with ray importance sampling. Our learned representation is highly compact and able to represent a 10 second 30 FPS multi-view video recording by 18 cameras with a model size of only 28MB. We demonstrate that our method can render high-fidelity wide-angle novel views at over 1K resolution, even for complex and dynamic scenes. We perform an extensive qualitative and quantitative evaluation that shows that our approach outperforms the state of the art. Project website: https://neural-3d-video.github.io/.

----

## [538] Learning to Solve Hard Minimal Problems

**Authors**: *Petr Hruby, Timothy Duff, Anton Leykin, Tomás Pajdla*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00545](https://doi.org/10.1109/CVPR52688.2022.00545)

**Abstract**:

We present an approach to solving hard geometric optimization problems in the RANSAC framework. The hard minimal problems arise from relaxing the original geometric optimization problem into a minimal problem with many spurious solutions. Our approach avoids computing large numbers of spurious solutions. We design a learning strategy for selecting a starting problem-solution pair that can be numerically continued to the problem and the solution of interest. We demonstrate our approach by developing a RANSAC solver for the problem of computing the relative pose of three calibrated cameras, via a minimal relaxation using four points in each view. On average, we can solve a single problem in under 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$70\ \mu s$</tex>
. We also benchmark and study our engineering choices on the very familiar problem of computing the relative pose of two calibrated cameras, via the minimal case of five points in two views.

----

## [539] Learning a Structured Latent Space for Unsupervised Point Cloud Completion

**Authors**: *Yingjie Cai, Kwan-Yee Lin, Chao Zhang, Qiang Wang, Xiaogang Wang, Hongsheng Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00546](https://doi.org/10.1109/CVPR52688.2022.00546)

**Abstract**:

Unsupervised point cloud completion aims at estimating the corresponding complete point cloud of a partial point cloud in an unpaired manner. It is a crucial but challenging problem since there is no paired partial-complete supervision that can be exploited directly. In this work, we pro-pose a novel framework, which learns a unified and structured latent space that encoding both partial and complete point clouds. Specifically, we map a series of related par-tial point clouds into multiple complete shape and occlusion code pairs and fuse the codes to obtain their repre-sentations in the unified latent space. To enforce the learning of such a structured latent space, the proposed method adopts a series of constraints including structured ranking regularization, latent code swapping constraint, and distribution supervision on the related partial point clouds. By establishing such a unified and structured latent space, better partial-complete geometry consistency and shape completion accuracy can be achieved. Extensive experi-ments show that our proposed method consistently outper-forms state-of-the-art unsupervised methods on both syn-thetic ShapeNet and real-world KITTI, ScanNet, and Mat- terport3D datasets.

----

## [540] Lepard: Learning partial point cloud matching in rigid and deformable scenes

**Authors**: *Yang Li, Tatsuya Harada*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00547](https://doi.org/10.1109/CVPR52688.2022.00547)

**Abstract**:

We present Lepard, a Learning based approach for partial point cloud matching in rigid and deformable scenes. The key characteristics are the following techniques that exploit 3D positional knowledge for point cloud matching: 1) An architecture that disentangles point cloud representation into feature space and 3D position space. 2) A position encoding method that explicitly reveals 3D relative distance information through the dot product of vectors. 3) A repositioning technique that modifies the cross-point-cloud relative positions. Ablation studies demonstrate the effectiveness of the above techniques. In rigid cases, Lepard combined with RANSAC and ICP demonstrates state-of-the-art registration recall of 93.9% / 71.3% on the 3DMatch / 3DLoMatch. In deformable cases, Lepard achieves +27.1% / +34.8% higher non-rigid feature matching recall than the prior art on our newly constructed 4DMatch / 4DLoMatch benchmark. Code and data are available at https://github.com/rabbityl/lepard.

----

## [541] IRON: Inverse Rendering by Optimizing Neural SDFs and Materials from Photometric Images

**Authors**: *Kai Zhang, Fujun Luan, Zhengqi Li, Noah Snavely*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00548](https://doi.org/10.1109/CVPR52688.2022.00548)

**Abstract**:

We propose a neural inverse rendering pipeline called IRON that operates on photometric images and outputs high-quality 3D content in the format of triangle meshes and material textures readily deployable in existing graphics pipelines. Our method adopts neural representations for geometry as signed distance fields (SDFs) and materials during optimization to enjoy their flexibility and compactness, and features a hybrid optimization scheme for neural SDFs: first, optimize using a volumetric radiance field approach to recover correct topology, then optimize further using edgeaware physics-based surface rendering for geometry refinement and disentanglement of materials and lighting. In the second stage, we also draw inspiration from mesh-based differentiable rendering, and design a novel edge sampling algorithm for neural SDFs to further improve performance. We show that our IRON achieves significantly better inverse rendering quality compared to prior works.

----

## [542] Learning Multi-View Aggregation In the Wild for Large-Scale 3D Semantic Segmentation

**Authors**: *Damien Robert, Bruno Vallet, Loïc Landrieu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00549](https://doi.org/10.1109/CVPR52688.2022.00549)

**Abstract**:

Recent works on 3D semantic segmentation propose to exploit the synergy between images and point clouds by processing each modality with a dedicated network and projecting learned 2D features onto 3D points. Merging large-scale point clouds and images raises several challenges, such as constructing a mapping between points and pixels, and aggregating features between multiple views. Current methods require mesh reconstruction or specialized sensors to recover occlusions, and use heuristics to select and aggregate available images. In contrast, we propose an end-to-end trainable multi-view aggregation model leveraging the viewing conditions of 3D points to merge features from images taken at arbitrary positions. Our method can combine standard 2D and 3D networks and outperforms both 3D models operating on colorized point clouds and hybrid 2D/3D networks without requiring colorization, meshing, or true depth maps. We set a new state-of-the-art for large-scale indoor/outdoor semantic segmentation on S3DIS (74.7 mIoU 6-Fold) and on KITTI-360 (58.3 mIoU). Our full pipeline is accessible at https://github.com/drprojects/DeepViewAgg, and only requires raw 3D scans and a set of images and poses.

----

## [543] HyperDet3D: Learning a Scene-conditioned 3D Object Detector

**Authors**: *Yu Zheng, Yueqi Duan, Jiwen Lu, Jie Zhou, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00550](https://doi.org/10.1109/CVPR52688.2022.00550)

**Abstract**:

A bathtub in a library, a sink in an office, a bed in a laundry room - the counter-intuition suggests that scene provides important prior knowledge for 3D object detection, which instructs to eliminate the ambiguous detection of similar objects. In this paper, we propose HyperDet3D to explore scene-conditioned prior knowledge for 3D object detection. Existing methods strive for better representation of local elements and their relations without scene-conditioned knowledge, which may cause ambiguity merely based on the understanding of individual points and object candidates. Instead, HyperDet3D simultaneously learns scene-agnostic embeddings and scene-specific knowledge through scene-conditioned hypernetworks. More specifically, our HyperDet3D not only explores the sharable abstracts from various 3D scenes, but also adapts the detector to the given scene at test time. We propose a discriminative Multi-head Scene-specific Attention (MSA) module to dynamically control the layer parameters of the detector conditioned on the fusion of scene-conditioned knowledge. Our HyperDet3D achieves state-of-the-art results on the 3D object detection benchmark of the ScanNet and SUN RGB-D datasets. Moreover, through cross-dataset evaluation, we show the acquired scene-conditioned prior knowledge still takes effect when facing 3D scenes with domain gap.

----

## [544] KeyTr: Keypoint Transporter for 3D Reconstruction of Deformable Objects in Videos

**Authors**: *David Novotný, Ignacio Rocco, Samarth Sinha, Alexandre Carlier, Gael Kerchenbaum, Roman Shapovalov, Nikita Smetanin, Natalia Neverova, Benjamin Graham, Andrea Vedaldi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00551](https://doi.org/10.1109/CVPR52688.2022.00551)

**Abstract**:

We consider the problem of reconstructing the depth of dynamic objects from videos. Recent progress in dynamic video depth prediction has focused on improving the output of monocular depth estimators by means of multi-view constraints while imposing little to no restrictions on the deformation of the dynamic parts of the scene. However, the theory of Non-Rigid Structure from Motion prescribes to constrain the deformations for 3D reconstruction. We thus propose a new model that departs significantly from this prior work. The idea is to fit a dynamic point cloud to the video data using Sinkhorn's algorithm to associate the 3D points to 2D pixels and use a differentiable point renderer to ensure the compatibility of the 3D deformations with the measured optical flow. In this manner, our algorithm, called Keypoint Transporter, models the overall deformation of the object within the entire video, so it can constrain the reconstruction correspondingly. Compared to weaker deformation models, this significantly reduces the reconstruction ambiguity and, for dynamic objects, allows Keypoint Transporter to obtain reconstructions of the quality superior or at least comparable to prior approaches while being much faster and reliant on a pre-trained monocular depth estimator network. To assess the method, we evaluate on new datasets of synthetic videos depicting dynamic humans and animals with ground-truth depth. We also show qualitative results on crowd-sourced real-world videos of pets.

----

## [545] SelfRecon: Self Reconstruction Your Digital Avatar from Monocular Video

**Authors**: *Boyi Jiang, Yang Hong, Hujun Bao, Juyong Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00552](https://doi.org/10.1109/CVPR52688.2022.00552)

**Abstract**:

We propose SelfRecon, a clothed human body reconstruction method that combines implicit and explicit repre-sentations to recover space-time coherent geometries from a monocular self-rotating human video. Explicit methods require a predefined template mesh for a given sequence, while the template is hard to acquire for a specific subject. Meanwhile, the fixed topology limits the reconstruction accuracy and clothing types. Implicit representation supports arbitrary topology and can represent high-fidelity geometry shapes due to its continuous nature. However, it is difficult to integrate multi-frame information to produce a consistent registration sequence for downstream applications. We propose to combine the advantages of both representations. We utilize differential mask loss of the explicit mesh to obtain the coherent overall shape, while the details on the implicit surface are refined with the differentiable neural rendering. Meanwhile, the explicit mesh is updated periodically to adjust its topology changes, and a consistency loss is designed to match both representations. Compared with existing methods, SelfRecon can produce high-fidelity surfaces for arbitrary clothed humans with self-supervised optimization. Extensive experimental results demonstrate its effectiveness on real captured monocular videos. The source code is available at https://github.com/jby1993/SelfReconCode.

----

## [546] Ditto: Building Digital Twins of Articulated Objects from Interaction

**Authors**: *Zhenyu Jiang, Cheng-Chun Hsu, Yuke Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00553](https://doi.org/10.1109/CVPR52688.2022.00553)

**Abstract**:

Digitizing physical objects into the virtual world has the potential to unlock new research and applications in embodied AI and mixed reality. This work focuses on recreating interactive digital twins of real-world articulated objects, which can be directly imported into virtual environments. We introduce Ditto to learn articulation model estimation and 3D geometry reconstruction of an articulated object through interactive perception. Given a pair of visual observations of an articulated object before and after interaction, Ditto reconstructs part-level geometry and estimates the articulation model of the object. We employ implicit neural representations for joint geometry and articulation modeling. Our experiments show that Ditto effectively builds digital twins of articulated objects in a category-agnostic way. We also apply Ditto to real-world objects and deploy the recreated digital twins in physical simulation. Code and additional results are available at https://ut-austin-rpl.github.io/Ditto/

----

## [547] Bijective Mapping Network for Shadow Removal

**Authors**: *Yurui Zhu, Jie Huang, Xueyang Fu, Feng Zhao, Qibin Sun, Zheng-Jun Zha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00554](https://doi.org/10.1109/CVPR52688.2022.00554)

**Abstract**:

Shadow removal, which aims to restore the background in the shadow regions, is challenging due to its highly ill-posed nature. Most existing deep learning-based methods individually remove the shadow by only considering the content of the matched paired images, barely taking into account the auxiliary supervision of shadow generation in the shadow removal procedure. In this work, we argue that shadow removal and generation are interrelated and could provide useful informative supervision for each other. Specifically, we propose a new Bijective Mapping Network (BMNet), which couples the learning procedures of shadow removal and shadow generation in a unified parameter-shared framework. With consistent two way constraints and synchronous optimization of the two procedures, BMNet could effectively recover the underlying background contents during the forward shadow removal procedure. In addition, through statistical analysis of real world datasets, we observe and verify that shadow appearances under different color spectrums are inconsistent. This motivates us to design a Shadow-Invariant Color Guidance Module (SICGM), which can explicitly utilize the learned shadow-invariant color information to guide network color restoration, thereby further reducing color-bias effects. Experiments on the representative ISTD, ISTD+ and SRD benchmarks show that our proposed network outperforms the state-of-the-art method [11] in de-shadowing performance, while only using its 0.25% network parameters and 6.25% floating point operations (FLOPs).

----

## [548] Toward Fast, Flexible, and Robust Low-Light Image Enhancement

**Authors**: *Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, Zhongxuan Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00555](https://doi.org/10.1109/CVPR52688.2022.00555)

**Abstract**:

Existing low-light image enhancement techniques are mostly not only difficult to deal with both visual quality and computational efficiency but also commonly invalid in unknown complex scenarios. In this paper, we develop a new Self-Calibrated Illumination (SCI) learning framework for fast, flexible, and robust brightening images in real-world low-light scenarios. To be specific, we establish a cascaded illumination learning process with weight sharing to handle this task. Considering the computational burden of the cascaded pattern, we construct the self-calibrated module which realizes the convergence between results of each stage, producing the gains that only use the single basic block for inference (yet has not been exploited in previous works), which drastically diminishes computation cost. We then define the unsupervised training loss to elevate the model capability that can adapt general scenes. Further, we make comprehensive explorations to excavate SCI's inherent properties (lacking in existing works) including operation-insensitive adaptability (acquiring stable performance under the settings of different simple operations) and model-irrelevant generality (can be applied to illumination-based existing works to improve performance). Finally, plenty of experiments and ablation studies fully indicate our superiority in both quality and efficiency. Applications on low-light face detection and nighttime semantic segmentation fully reveal the latent practical values for SCI. The source code is available at https://github.com/vis-opt-group/SCI.

----

## [549] Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements

**Authors**: *Dongdong Chen, Julián Tachella, Mike E. Davies*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00556](https://doi.org/10.1109/CVPR52688.2022.00556)

**Abstract**:

Deep networks provide state-of-the-art performance in multiple imaging inverse problems ranging from medical imaging to computational photography. However, most existing networks are trained with clean signals which are often hard or impossible to obtain. Equivariant imaging (EI) is a recent self-supervised learning framework that exploits the group invariance present in signal distributions to learn a reconstruction function from partial measurement data alone. While EI results are impressive, its performance degrades with increasing noise. In this paper, we propose a Robust Equivariant Imaging (REI) framework which can learn to image from noisy partial measurements alone. The proposed method uses Stein's Unbiased Risk Estimator (SURE) to obtain a fully unsupervised training loss that is robust to noise. We show that REI leads to considerable performance gains on linear and nonlinear inverse problems, thereby paving the way for robust unsupervised imaging with deep networks. Code is available at https://github.com/edongdongchen/REI.

----

## [550] Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution

**Authors**: *Jie Liang, Hui Zeng, Lei Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00557](https://doi.org/10.1109/CVPR52688.2022.00557)

**Abstract**:

Single image super-resolution (SISR) with generative adversarial networks (GAN) has recently attracted increasing attention due to its potentials to generate rich details. However, the training of GAN is unstable, and it often introduces many perceptually unpleasant artifacts along with the generated details. In this paper, we demonstrate that it is possible to train a GAN-based SISR model which can stably generate perceptually realistic details while inhibiting visual artifacts. Based on the observation that the local statistics (e.g., residual variance) of artifact areas are often different from the areas of perceptually friendly details, we develop a framework to discriminate between GAN-generated artifacts and realistic details, and consequently generate an artifact map to regularize and stabilize the model training process. Our proposed locally discriminative learning (LDL) method is simple yet effective, which can be easily plugged in off-the-shelf SISR methods and boost their performance. Experiments demonstrate that LDL outperforms the state-of-the-art GAN based SISR methods, achieving not only higher reconstruction accuracy but also superior perceptual quality on both synthetic and real-world datasets. Codes and models are available at https://github.com/csjliang/LDL.

----

## [551] Dual Adversarial Adaptation for Cross-Device Real-World Image Super-Resolution

**Authors**: *Xiaoqian Xu, Pengxu Wei, Weikai Chen, Yang Liu, Mingzhi Mao, Liang Lin, Guanbin Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00558](https://doi.org/10.1109/CVPR52688.2022.00558)

**Abstract**:

Due to the sophisticated imaging process, an identical scene captured by different cameras could exhibit distinct imaging patterns, introducing distinct proficiency among the super-resolution (SR) models trained on images from different devices. In this paper, we investigate a novel and practical task coded cross-device SR, which strives to adapt a real-world SR model trained on the paired images captured by one camera to low-resolution (LR) images captured by arbitrary target devices. The proposed task is highly challenging due to the absence of paired data from various imaging devices. To address this issue, we propose an unsupervised domain adaptation mechanism for real-world SR, named Dual ADversarial Adaptation (DADA), which only requires LR images in the target domain with available real paired data from a source camera. DADA employs the Domain-Invariant Attention (DIA) module to establish the basis of target model training even without HR supervision. Furthermore, the dual framework of DADA facilitates an Inter-domain Adversarial Adaptation (InterAA) in one branch for two LR input images from two domains, and an Intra-domain Adversarial Adaptation (IntraAA) in two branches for an LR input image. InterAA and IntraAA together improve the model transferability from the source domain to the target. We empirically conduct experiments under six 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\text{Real} \rightarrow \text{Real}$</tex>
 adaptation settings among three different cameras, and achieve superior performance compared with existing state-of-the-art approaches. We also evaluate the proposed DADA to address the adaptation to the video camera, which presents a promising re-search topic to promote the wide applications of real-world super-resolution. Our source code is publicly available at https://github.com/lonelyhopeIDADA.

----

## [552] SphereSR: 360° Image Super-Resolution with Arbitrary Projection via Continuous Spherical Image Representation

**Authors**: *Youngho Yoon, Inchul Chung, Lin Wang, Kuk-Jin Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00559](https://doi.org/10.1109/CVPR52688.2022.00559)

**Abstract**:

The 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$360^{\circ}$</tex>
 imaging has recently gained much attention; however, its angular resolution is relatively lower than that of a narrow field-of-view (FOV) perspective image as it is captured using a fisheye lens with the same sensor size. Therefore, it is beneficial to super-resolve a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$360^{\circ}$</tex>
 image. Several attempts have been made, but mostly considered equirectangular projection (ERP) as one of the ways for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$360^{\circ}$</tex>
 image representation despite the latitude-dependent distortions. In that case, as the output high-resolution (HR) image is always in the same ERP format as the low-resolution (LR) input, additional information loss may occur when transforming the HR image to other projection types. In this paper, we propose SphereSR, a novel framework to generate a continuous spherical image representation from an LR 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$360^{\circ}$</tex>
 image, with the goal of predicting the RGB values at given spherical coordinates for super-resolution with an arbitrary 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$360^{\circ}$</tex>
 image projection. Specifically, first we propose a feature extraction module that represents the spherical data based on an icosahedron and that efficiently extracts features on the spherical surface. We then propose a spherical local implicit image function (SLIIF) to predict RGB values at the spherical coordinates. As such, SphereSR flexibly reconstructs an HR image given an arbitrary projection type. Experiments on various benchmark datasets show that the proposed method significantly surpasses existing methods in terms of performance.

----

## [553] Learning Trajectory-Aware Transformer for Video Super-Resolution

**Authors**: *Chengxu Liu, Huan Yang, Jianlong Fu, Xueming Qian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00560](https://doi.org/10.1109/CVPR52688.2022.00560)

**Abstract**:

Video super-resolution (VSR) aims to restore a sequence of high-resolution (HR) frames from their low-resolution (LR) counterparts. Although some progress has been made, there are grand challenges to effectively utilize temporal dependency in entire video sequences. Existing approaches usually align and aggregate video frames from limited adjacent frames (e.g., 5 or 7 frames), which prevents these approaches from satisfactory results. In this paper, we take one step further to enable effective spatio-temporal learning in videos. We propose a novel Trajectory-aware Transformer for Video Super-Resolution (TTVSR). In particular, we formulate video frames into several pre-aligned trajectories which consist of continuous visual tokens. For a query token, self-attention is only learned on relevant visual tokens along spatio-temporal trajectories. Compared with vanilla vision Transformers, such a design significantly reduces the computational cost and enables Transformers to model long-range features. We further propose a cross-scale feature tokenization module to over-come scale-changing problems that often occur in long-range videos. Experimental results demonstrate the superiority of the proposed TTVSR over state-of-the-art models, by extensive quantitative and qualitative evaluations in four widely-used video super-resolution benchmarks. Both code and pre-trained models can be downloaded at https://github.com/researchmm/TTVSR.

----

## [554] Discrete Cosine Transform Network for Guided Depth Map Super-Resolution

**Authors**: *Zixiang Zhao, Jiangshe Zhang, Shuang Xu, Zudi Lin, Hanspeter Pfister*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00561](https://doi.org/10.1109/CVPR52688.2022.00561)

**Abstract**:

Guided depth super-resolution (GDSR) is an essential topic in multi-modal image processing, which reconstructs high-resolution (HR) depth maps from low-resolution ones collected with suboptimal conditions with the help of HR RGB images of the same scene. To solve the challenges in interpreting the working mechanism, extracting cross-modal features and RGB texture over-transferred, we propose a novel Discrete Cosine Transform Network (DCTNet) to alleviate the problems from three aspects. First, the Discrete Cosine Transform (DCT) module reconstructs the multi-channel HR depth features by using DCT to solve the channel-wise optimization problem derived from the image domain. Second, we introduce a semi-coupled feature extraction module that uses shared convolutional kernels to extract common information and private kernels to extract modality-specific information. Third, we employ an edge attention mechanism to highlight the contours informative for guided upsampling. Extensive quantitative and qualitative evaluations demonstrate the effectiveness of our DCTNet, which outperforms previous state-of-the-art methods with a relatively small number of parameters. The code is available at https://github.com/Zhaozixiang1228/GDSR-DCTNet.

----

## [555] Faithful Extreme Rescaling via Generative Prior Reciprocated Invertible Representations

**Authors**: *Zhixuan Zhong, Liangyu Chai, Yang Zhou, Bailin Deng, Jia Pan, Shengfeng He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00562](https://doi.org/10.1109/CVPR52688.2022.00562)

**Abstract**:

This paper presents a Generative prior ReciprocAted Invertible rescaling Network (GRAIN) for generating faithful high-resolution (HR) images from low-resolution (LR) invertible images with an extreme upscaling factor (64×). Previous researches have leveraged the prior knowledge of a pretrained GAN model to generate high-quality upscaling results. However, they fail to produce pixel-accurate results due to the highly ambiguous extreme mapping process. We remedy this problem by introducing a reciprocated invertible image rescaling process, in which high-resolution information can be delicately embedded into an invertible low-resolution image and generative prior for a faithful HR reconstruction. In particular, the invertible LR features not only carry significant HR semantics, but also are trained to predict scale-specific latent codes, yielding a preferable utilization of generative features. On the other hand, the enhanced generative prior is re-injected to the rescaling process, compensating the lost details of the invertible rescaling. Our reciprocal mechanism perfectly integrates the advantages of invertible encoding and generative prior, leading to the first feasible extreme rescaling solution. Extensive experiments demonstrate superior performance against state-of-the-art upscaling methods. Code is available at https://github.com/cszzx/GRAIN.

----

## [556] ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding

**Authors**: *Dailan He, Ziming Yang, Weikun Peng, Rui Ma, Hongwei Qin, Yan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00563](https://doi.org/10.1109/CVPR52688.2022.00563)

**Abstract**:

Recently, learned image compression techniques have achieved remarkable performance, even surpassing the best manually designed lossy image coders. They are promising to be large-scale adopted. For the sake of practicality, a thorough investigation of the architecture design of learned image compression, regarding both compression performance and running speed, is essential. In this paper, we first propose uneven channel-conditional adaptive coding, motivated by the observation of energy compaction in learned image compression. Combining the proposed uneven grouping model with existing context models, we obtain a spatial-channel contextual adaptive model to improve the coding performance without damage to running speed. Then we study the structure of the main transform and propose an efficient model, ELIC, to achieve state-of-the-art speed and compression ability. With superior performance, the proposed model also supports extremely fast preview decoding and progressive decoding, which makes the coming application of learning-based image compression more promising.

----

## [557] Restormer: Efficient Transformer for High-Resolution Image Restoration

**Authors**: *Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00564](https://doi.org/10.1109/CVPR52688.2022.00564)

**Abstract**:

Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising). The source code and pre-trained models are available at https://github.com/swz30/Restormer.

----

## [558] Deep Rectangling for Image Stitching: A Learning Baseline

**Authors**: *Lang Nie, Chunyu Lin, Kang Liao, Shuaicheng Liu, Yao Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00565](https://doi.org/10.1109/CVPR52688.2022.00565)

**Abstract**:

Stitched images provide a wide field-of-view (FoV) but suffer from unpleasant irregular boundaries. To deal with this problem, existing image rectangling methods devote to searching an initial mesh and optimizing a target mesh to form the mesh deformation in two stages. Then rectangu-lar images can be generated by warping stitched images. However, these solutions only work for images with rich linear structures, leading to noticeable distortions for por-traits and landscapes with non-linear objects. In this paper, we address these issues by proposing the first deep learning solution to image rectangling. Con-cretely, we predefine a rigid target mesh and only estimate an initial mesh to form the mesh deformation, contributing to a compact one-stage solution. The initial mesh is predicted using a fully convolutional network with a resid-ual progressive regression strategy. To obtain results with high content fidelity, a comprehensive objective function is proposed to simultaneously encourage the boundary rect-angular, mesh shape-preserving, and content perceptually natural. Besides, we build the first image stitching rectan-gling dataset with a large diversity in irregular boundaries and scenes. Experiments demonstrate our superiority over traditional methods both quantitatively and qualitatively.

----

## [559] Parametric Scattering Networks

**Authors**: *Shanel Gauthier, Benjamin Thérien, Laurent Alsène-Racicot, Muawiz Chaudhary, Irina Rish, Eugene Belilovsky, Michael Eickenberg, Guy Wolf*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00566](https://doi.org/10.1109/CVPR52688.2022.00566)

**Abstract**:

The wavelet scattering transform creates geometric in-variants and deformation stability. In multiple signal do-mains, it has been shown to yield more discriminative rep-resentations compared to other non-learned representations and to outperform learned representations in certain tasks, particularly on limited labeled data and highly structured signals. The wavelet filters used in the scattering trans-form are typically selected to create a tight frame via a pa-rameterized mother wavelet. In this work, we investigate whether this standard wavelet filterbank construction is op-timal. Focusing on Morlet wavelets, we propose to learn the scales, orientations, and aspect ratios of the filters to produce problem-specific parameterizations of the scattering transform. We show that our learned versions of the scattering transform yield significant performance gains in small-sample classification settings over the standard scat-tering transform. Moreover, our empirical results suggest that traditional filterbank constructions may not always be necessary for scattering transforms to extract effective rep-resentations.

----

## [560] Burst Image Restoration and Enhancement

**Authors**: *Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, Ming-Hsuan Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00567](https://doi.org/10.1109/CVPR52688.2022.00567)

**Abstract**:

Modern handheld devices can acquire burst image sequence in a quick succession. However, the individual acquired frames suffer from multiple degradations and are misaligned due to camera shake and object motions. The goal of Burst Image Restoration is to effectively combine complimentary cues across multiple burst frames to generate high-quality outputs. Towards this goal, we develop a novel approach by solely focusing on the effective information exchange between burst frames, such that the degradations get filtered out while the actual scene details are preserved and enhanced. Our central idea is to create a set of pseudo-burst features that combine complimentary information from all the input burst frames to seamlessly exchange information. However, the pseudo-burst cannot be successfully created unless the individual burst frames are properly aligned to discount inter-frame movements. Therefore, our approach initially extracts pre-processed features from each burst frame and matches them using an edge-boosting burst alignment module. The pseudo-burst features are then created and enriched using multi-scale contextual information. Our final step is to adaptively aggregate information from the pseudo-burst features to progressively increase resolution in multiple stages while merging the pseudo-burst features. In comparison to existing works that usually follow a late fusion scheme with single-stage upsampling, our approach performs favorably, delivering state-of-the-art performance on burst super-resolution, burst low-light image enhancement and burst denoising tasks. The source code and pre-trained models are available at https://github.com/akshaydudhane16/BIPNet.

----

## [561] MAXIM: Multi-Axis MLP for Image Processing

**Authors**: *Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan C. Bovik, Yinxiao Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00568](https://doi.org/10.1109/CVPR52688.2022.00568)

**Abstract**:

Recent progress on Transformers and multilayer perceptron (MLP) models provide new network architectural designs for computer vision tasks. Although these models proved to be effective in many vision tasks such as image recognition, there remain challenges in adapting them for lowlevel vision. The inflexibility to support high-resolution images and limitations of local attention are perhaps the main bottlenecks. In this work, we present a multi-axis MLP based architecture called MAXIM, that can serve as an efficient and flexible general-purpose vision backbone for image processing tasks. MAXIM uses a UNet-shaped hierarchical structure and supports long-range interactions enabled by spatially-gated MLPs. Specifically, MAXIM contains two MLP-based building blocks: a multi-axis gated MLP that allows for efficient and scalable spatial mixing of local and global visual cues, and a cross-gating block, an alternative to cross-attention, which accounts for cross-feature conditioning. Both these modules are exclusively based on MLPs, but also benefit from being both global and ‘fully-convolutional’, two properties that are desirable for image processing. Our extensive experimental results show that the proposed MAXIM model achieves state-of-the-art performance on more than ten benchmarks across a range of image processing tasks, including denoising, deblurring, de raining, dehazing, and enhancement while requiring fewer or comparable numbers of parameters and FLOPs than competitive models. The source code and trained models will be available at https://github.com/google-research/maxim.

----

## [562] Event-aided Direct Sparse Odometry

**Authors**: *Javier Hidalgo-Carrió, Guillermo Gallego, Davide Scaramuzza*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00569](https://doi.org/10.1109/CVPR52688.2022.00569)

**Abstract**:

We introduce EDS, a direct monocular visual odometry using events and frames. Our algorithm leverages the event generation model to track the camera motion in the blind time between frames. The method formulates a direct probabilistic approach of observed brightness increments. Per-pixel brightness increments are predicted using a sparse number of selected 3D points and are compared to the events via the brightness increment error to estimate camera motion. The method recovers a semi-dense 3D map using photometric bundle adjustment. EDS is the first method to perform 6-DOF VO using events and frames with a direct approach. By design it overcomes the problem of changing appearance in indirect methods. Our results outperform all previous event-based odometry solutions. We also show that, for a target error performance, EDS can work at lower frame rates than state-of-the-art frame-based VO solutions. This opens the door to low-power motion-tracking applications where frames are sparingly triggered “on demand” and our method tracks the motion in between. We release code and datasets to the public.

----

## [563] CamLiFlow: Bidirectional Camera-LiDAR Fusion for Joint Optical Flow and Scene Flow Estimation

**Authors**: *Haisong Liu, Tao Lu, Yihui Xu, Jia Liu, Wenjie Li, Lijun Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00570](https://doi.org/10.1109/CVPR52688.2022.00570)

**Abstract**:

In this paper, we study the problem of jointly estimating the optical flow and scene flow from synchronized 2D and 3D data. Previous methods either employ a complex pipeline that splits the joint task into independent stages, or fuse 2D and 3D information in an “early-fusion“ or “late-fusion“ manner. Such one-size-fits-all approaches suffer from a dilemma of failing to fully utilize the characteristic of each modality or to maximize the inter-modality complementarity. To address the problem, we propose a novel end-to-end framework, called CamLiFlow. It consists of 2D and 3D branches with multiple bidirectional connections between them in specific layers. Different from previous work, we apply a point-based 3D branch to better extract the geometric features and design a symmetric learnable operator to fuse dense image features and sparse point features. Experiments show that CamLiFlow achieves better performance with fewer parameters. Our method ranks 1st on the KITTI Scene Flow benchmark, outperforming the previous art with 1/7 parameters. Code is available at https://github.com/MCG-NJU/CamLiFlow.

----

## [564] Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection

**Authors**: *Jinyuan Liu, Xin Fan, Zhanbo Huang, Guanyao Wu, Risheng Liu, Wei Zhong, Zhongxuan Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00571](https://doi.org/10.1109/CVPR52688.2022.00571)

**Abstract**:

This study addresses the issue of fusing infrared and visible images that appear differently for object detection. Aiming at generating an image of high visual quality, previous approaches discover commons underlying the two modalities and fuse upon the common space either by iterative optimization or deep networks. These approaches neglect that modality differences implying the complementary information are extremely important for both fusion and subsequent detection task. This paper proposes a bilevel optimization formulation for the joint problem of fusion and detection, and then unrolls to a target-aware Dual Adversarial Learning (TarDAL) network for fusion and a commonly used detection network. The fusion network with one generator and dual discriminators seeks commons while learning from differences, which preserves structural information of targets from the infrared and textural details from the visible. Furthermore, we build a synchronized imaging system with calibrated infrared and optical sensors, and collect currently the most comprehensive benchmark covering a wide range of scenarios. Extensive experiments on several public datasets and our benchmark demonstrate that our method outputs not only visually appealing fusion but also higher detection mAP than the state-of-the-art approaches. The source code and benchmark are available at https://github.com/dlut-dimt/TarDAL.

----

## [565] Image Dehazing Transformer with Transmission-Aware 3D Position Embedding

**Authors**: *Chunle Guo, Qixin Yan, Saeed Anwar, Runmin Cong, Wenqi Ren, Chongyi Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00572](https://doi.org/10.1109/CVPR52688.2022.00572)

**Abstract**:

Despite single image dehazing has been made promising progress with Convolutional Neural Networks (CNNs), the inherent equivariance and locality of convolution still bottleneck deharing performance. Though Transformer has occupied various computer vision tasks, directly leveraging Transformer for image dehazing is challenging: 1) it tends to result in ambiguous and coarse details that are undesired for image reconstruction; 2) previous position embedding of Transformer is provided in logic or spatial position order that neglects the variational haze densities, which results in the sub-optimal dehazlng performance. The key insight of this study is to investigate how to combine CNN and Transformer for image dehazing. To solve the feature inconsistency issue between Transformer and CNN, we propose to modulate CNN features via learning modulation matrices (i.e., coefficient matrix and bias matrix) conditioned on Transformer features instead of simple feature addition or concatenation. The feature modulation naturally inherits the global context modeling capability of Transformer and the local representation capability of CNN. We bring a haze density-related prior into Trans-former via a novel transmission-aware 3D position embedding module, which not only provides the relative position but also suggests the haze density of different spatial regions. Extensive experiments demonstrate that our method, DeHamer, attains state-of-the-art performance on several image dehazing benchmarks.

----

## [566] Unsupervised Deraining: Where Contrastive Learning Meets Self-similarity

**Authors**: *Yuntong Ye, Changfeng Yu, Yi Chang, Lin Zhu, Xi-Le Zhao, Luxin Yan, Yonghong Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00573](https://doi.org/10.1109/CVPR52688.2022.00573)

**Abstract**:

Image deraining is a typical low-level image restoration task, which aims at decomposing the rainy image into two distinguishable layers: clean image layer and rain layer. Most of the existing learning-based deraining methods are supervisedly trained on synthetic rainy-clean pairs. The domain gap between the synthetic and real rains makes them less generalized to different real rainy scenes. Moreover, the existing methods mainly utilize the property of the two layers independently, while few of them have considered the mutually exclusive relationship between the two layers. In this work, we propose a novel non-local contrastive learning (NLCL) method for unsupervised image deraining. Consequently, we not only utilize the intrinsic self-similarity property within samples, but also the mutually exclusive property between the two layers, so as to better differ the rain layer from the clean image. Specifically, the non-local self-similarity image layer patches as the positives are pulled together and similar rain layer patches as the negatives are pushed away. Thus the similar positive/negative samples that are close in the original space benefit us to enrich more discriminative representation. Apart from the self-similarity sampling strategy, we analyze how to choose an appropriate feature encoder in NLCL. Extensive experiments on different real rainy datasets demonstrate that the proposed method obtains state-of-the-art performance in real deraining.

----

## [567] Towards Multi-domain Single Image Dehazing via Test-time Training

**Authors**: *Huan Liu, Zijun Wu, Liangyan Li, Sadaf Salehkalaibar, Jun Chen, Keyan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00574](https://doi.org/10.1109/CVPR52688.2022.00574)

**Abstract**:

Recent years have witnessed significant progress in the area of single image dehazing, thanks to the employment of deep neural networks and diverse datasets. Most of the existing methods perform well when the training and testing are conducted on a single dataset. However, they are not able to handle different types of hazy images using a dehazing model trained on a particular dataset. One possible remedy is to perform training on multiple datasets jointly. However, we observe that this training strategy tends to compromise the model performance on individual datasets. Motivated by this observation, we propose a test-time training method which leverages a helper network to assist the dehazing model in better adapting to a domain of interest. Specifically, during the test time, the helper network evaluates the quality of the dehazing results, then directs the dehazing network to improve the quality by adjusting its parameters via self-supervision. Nevertheless, the inclusion of the helper network does not automatically ensure the desired performance improvement. For this reason, a metalearning approach is employed to make the objectives of the dehazing and helper networks consistent with each other. We demonstrate the effectiveness of the proposed method by providing extensive supporting experiments.

----

## [568] Physically Disentangled Intra- and Inter-domain Adaptation for Varicolored Haze Removal

**Authors**: *Yi Li, Yi Chang, Yan Gao, Changfeng Yu, Luxin Yan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00575](https://doi.org/10.1109/CVPR52688.2022.00575)

**Abstract**:

Learning-based image dehazing methods have achieved marvelous progress during the past few years. On one hand, most approaches heavily rely on synthetic data and may face difficulties to generalize well in real scenes, due to the huge domain gap between synthetic and real images. On the other hand, very few works have considered the varicolored haze, caused by chromatic casts in real scenes. In this work, our goal is to handle the new task: real-world varicolored haze removal. To this end, we propose a physically disentangled joint intra- and inter-domain adaptation paradigm, in which intra-domain adaptation focuses on color correction and inter-domain procedure transfers knowledge between synthetic and real domains. We first learn to physically disentangle haze images into three components complying with the scattering model: background, transmission map, and atmospheric light. Since haze color is determined by atmospheric light, we perform intra-domain adaptation by specifically translating atmospheric light from varicolored space to unified color-balanced space, and then reconstructing color-balanced haze image through the scattering model. Consequently, we perform inter-domain adaptation between the synthetic and real images by mutually exchanging the background and other two components. Then we can reconstruct both identity and domain-translated haze images with self-consistency and adversarial loss. Extensive experiments demonstrate the superiority of the proposed method over the state-of-the-art for real varicolored image dehazing.

----

## [569] Incorporating Semi-Supervised and Positive-Unlabeled Learning for Boosting Full Reference Image Quality Assessment

**Authors**: *Yue Cao, Zhaolin Wan, Dongwei Ren, Zifei Yan, Wangmeng Zuo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00576](https://doi.org/10.1109/CVPR52688.2022.00576)

**Abstract**:

Full-reference (FR) image quality assessment (IQA) evaluates the visual quality of a distorted image by measuring its perceptual difference with pristine-quality reference, and has been widely used in low-level vision tasks. Pairwise labeled data with mean opinion score (MOS) are required in training FR-IQA model, but is time-consuming and cumbersome to collect. In contrast, unlabeled data can be easily collected from an image degradation or restoration process, making it encouraging to exploit unlabeled training data to boost FR-IQA performance. Moreover, due to the distribution inconsistency between labeled and unlabeled data, outliers may occur in unlabeled data, further increasing the training difficulty. In this paper, we suggest to incorporate semi-supervised and positive-unlabeled (PU) learning for exploiting unlabeled data while mitigating the adverse effect of outliers. Particularly, by treating all labeled data as positive samples, PU learning is leveraged to identify negative samples (i.e., outliers) from unlabeled data. Semi-supervised learning (SSL) is further deployed to exploit positive unlabeled data by dynamically generating pseudo-MOS. We adopt a dual-branch network including reference and distortion branches. Furthermore, spatial attention is introduced in the reference branch to concentrate more on the informative regions, and sliced Wasserstein distance is used for robust difference map computation to address the misalignment issues caused by images recovered by GAN models. Extensive experiments show that our method performs favorably against state-of-the-arts on the benchmark datasets PIPAL, KADID-10k, TID2013, LIVE and CSIQ. The source code and model are available at https://github.com/happycaoyue/JSPL.

----

## [570] Practical Learned Lossless JPEG Recompression with Multi-Level Cross-Channel Entropy Model in the DCT Domain

**Authors**: *Lina Guo, Xinjie Shi, Dailan He, Yuanyuan Wang, Rui Ma, Hongwei Qin, Yan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00577](https://doi.org/10.1109/CVPR52688.2022.00577)

**Abstract**:

JPEG is a popular image compression method widely used by individuals, data center, cloud storage and network filesystems. However, most recent progress on image compression mainly focuses on uncompressed images while ignoring trillions of already-existing JPEG images. To compress these JPEG images adequately and restore them back to JPEG format losslessly when needed, we propose a deep learning based JPEG recompression method that operates on DCT domain and propose a Multi-Level Cross-Channel Entropy Model to compress the most informative Y component. Experiments show that our method achieves state-of-the-art performance compared with traditional JPEG recompression methods including Lepton, JPEG XL and CMIX. To the best of our knowledge, this is the first learned compression method that losslessly transcodes JPEG images to more storage-saving bitstreams.

----

## [571] Neural Compression-Based Feature Learning for Video Restoration

**Authors**: *Cong Huang, Jiahao Li, Bin Li, Dong Liu, Yan Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00578](https://doi.org/10.1109/CVPR52688.2022.00578)

**Abstract**:

How to efficiently utilize the temporal features is crucial, yet challenging, for video restoration. The temporal features usually contain various noisy and uncorrelated information, and they may interfere with the restoration of the current frame. This paper proposes learning noiserobust feature representations to help video restoration. We are inspired by that the neural codec is a natural denoiser: In neural codec, the noisy and uncorrelated contents which are hard to predict but cost lots of bits are more inclined to be discarded for bitrate saving. Therefore, we design a neural compression module to filter the noise and keep the most useful information in features for video restoration. To achieve robustness to noise, our compression module adopts a spatial-channel-wise quantization mechanism to adaptively determine the quantization step size for each position in the latent. Experiments show that our method can significantly boost the performance on video denoising, where we obtain 0.13 dB improvement over BasicVSR++ with only 0.23x FLOPs. Meanwhile, our method also obtains SOTA results on video deraining and dehazing.

----

## [572] Bi-directional Object-Context Prioritization Learning for Saliency Ranking

**Authors**: *Xin Tian, Ke Xu, Xin Yang, Lin Du, Baocai Yin, Rynson W. H. Lau*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00579](https://doi.org/10.1109/CVPR52688.2022.00579)

**Abstract**:

The saliency ranking task is recently proposed to study the visual behavior that humans would typically shift their attention over different objects of a scene based on their degrees of saliency. Existing approaches focus on learning either object-object or object-scene relations. Such a strategy follows the idea of object-based attention in Psychology, but it tends to favor objects with strong semantics (e.g., humans), resulting in unrealistic saliency ranking. We observe that spatial attention works concurrently with object-based attention in the human visual recognition system. During the recognition process, the human spatial attention mechanism would move, engage, and disengage from region to region (i.e., context to context). This inspires us to model region-level interactions, in addition to object-level reasoning, for saliency ranking. Hence, we propose a novel bi-directional method to unify spatial attention and object-based attention for saliency ranking. Our model has two novel modules: (1) a selective object saliency (SOS) module to model object-based attention via inferring the semantic representation of salient objects, and (2) an object-context-object relation (OCOR) module to allocate saliency ranks to objects by jointly modeling object-context and context-object interactions of salient objects. Extensive experiments show that our approach outperforms existing state-of-the-art methods. Code and pretrained model are available at https://github.com/GrassBro/OCOR.

----

## [573] URetinex-Net: Retinex-based Deep Unfolding Network for Low-light Image Enhancement

**Authors**: *Wenhui Wu, Jian Weng, Pingping Zhang, Xu Wang, Wenhan Yang, Jianmin Jiang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00581](https://doi.org/10.1109/CVPR52688.2022.00581)

**Abstract**:

Retinex model-based methods have shown to be effective in layer-wise manipulation with well-designed priors for low-light image enhancement. However, the commonly used handcrafted priors and optimization-driven solutions lead to the absence of adaptivity and efficiency. To address these issues, in this paper, we propose a Retinex-based deep unfolding network (URetinex-Net), which unfolds an optimization problem into a learnable network to decompose a low-light image into reflectance and illumination layers. By formulating the decomposition problem as an implicit priors regularized model, three learning-based modules are carefully designed, responsible for data-dependent initialization, high-efficient unfolding optimization, and user-specified illumination enhancement, respectively. Particularly, the proposed unfolding optimization module, introducing two networks to adaptively fit implicit priors in data-driven manner, can realize noise suppression and details preservation for the final decomposition results. Extensive experiments on real-world low-light images qualitatively and quantitatively demonstrate the effectiveness and superiority of the proposed method over state-of-the-art methods. The code is available at https://github.com/AndersonYong/URetinex-Net.

----

## [574] A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution

**Authors**: *Jianqi Ma, Zhetong Liang, Lei Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00582](https://doi.org/10.1109/CVPR52688.2022.00582)

**Abstract**:

Scene text image super-resolution aims to increase the resolution and readability of the text in low-resolution images. Though significant improvement has been achieved by deep convolutional neural networks (CNNs), it remains difficult to reconstruct high-resolution images for spatially deformed texts, especially rotated and curve-shaped ones. This is because the current CNN-based methods adopt locality-based operations, which are not effective to deal with the variation caused by deformations. In this paper, we propose a CNN based Text ATTention network (TATT) to address this problem. The semantics of the text are firstly extracted by a text recognition module as text prior information. Then we design a novel transformer-based module, which leverages global attention mechanism, to exert the semantic guidance of text prior to the text reconstruction process. In addition, we propose a text structure consistency loss to refine the visual appearance by imposing structural consistency on the reconstructions of regular and deformed texts. Experiments on the benchmark TextZoom dataset show that the proposed TATT not only achieves state-of-the-art performance in terms of PSNR/SSIM metrics, but also significantly improves the recognition accuracy in the downstream text recognition task, particularly for text instances with multi-orientation and curved shapes. Code is available at https://github.com/mjq11302010044/TATT.

----

## [575] Coarse-To-Fine Deep Video Coding with Hyperprior-Guided Mode Prediction

**Authors**: *Zhihao Hu, Guo Lu, Jinyang Guo, Shan Liu, Wei Jiang, Dong Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00583](https://doi.org/10.1109/CVPR52688.2022.00583)

**Abstract**:

The previous deep video compression approaches only use the single scale motion compensation strategy and rarely adopt the mode prediction technique from the traditional standards like H.264/H.265 for both motion and residual compression. In this work, we first propose a coarse-to-fine (C2F) deep video compression framework for better motion compensation, in which we perform motion estimation, compression and compensation twice in a coarse to fine manner. Our C2F framework can achieve better motion compensation results without significantly increasing bit costs. Observing hyperprior information (i.e., the mean and variance values) from the hyperprior networks contains discriminant statistical information of different patches, we also propose two efficient hyperprior-guided mode prediction methods. Specifically, using hyper-prior information as the input, we propose two mode prediction networks to respectively predict the optimal block resolutions for better motion coding and decide whether to skip residual information from each block for better residual coding without introducing additional bit cost while bringing negligible extra computation cost. Comprehensive experimental results demonstrate our proposed C2F video compression framework equipped with the new hyperprior-guided mode prediction methods achieves the state-of-the-art performance on HEVC, UVG and MCL-JCV datasets.

----

## [576] Task Decoupled Framework for Reference-based Super-Resolution

**Authors**: *Yixuan Huang, Xiaoyun Zhang, Yu Fu, Siheng Chen, Ya Zhang, Yanfeng Wang, Dazhi He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00584](https://doi.org/10.1109/CVPR52688.2022.00584)

**Abstract**:

Reference-based super-resolution(RefSR) has achieved impressive progress on the recovery of high-frequency details thanks to an additional reference high-resolution(HR) image input. Although the superiority compared with Single-Image Super-Resolution(SISR), existing RefSR methods easily result in the reference-underuse issue and the reference-misuse as shown in Fig. I. In this work, we deeply investigate the cause of the two issues and further propose a novel framework to mitigate them. Our studies find that the issues are mostly due to the improper coupled framework design of current methods. Those methods conduct the super-resolution task of the input low-resolution(LR) image and the texture transfer task from the reference image together in one module, easily introducing the interference between LR and reference features. Inspired by this finding, we propose a novel framework, which decouples the two tasks of RefSR, eliminating the interference between the LR image and the reference image. The super-resolution task upsamples the LR image leveraging only the LR image itself. The texture transfer task extracts and transfers abundant textures from the reference image to the coarsely upsampled result of the super-resolution task. Extensive experiments demonstrate clear improvements in both quantitative and qualitative evaluations over state-of-the-art methods.

----

## [577] Learning Semantic Associations for Mirror Detection

**Authors**: *Huankang Guan, Jiaying Lin, Rynson W. H. Lau*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00585](https://doi.org/10.1109/CVPR52688.2022.00585)

**Abstract**:

Mirrors generally lack a consistent visual appearance, making mirror detection very challenging. Although recent works that are based on exploiting contextual contrasts and corresponding relations have achieved good results, heavily relying on contextual contrasts and corresponding relations to discover mirrors tend to fail in complex real-world scenes, where a lot of objects, e.g., doorways, may have similar features as mirrors. We observe that humans tend to place mirrors in relation to certain objects for specific functional purposes, e.g., a mirror above the sink. Inspired by this observation, we propose a model to exploit the semantic associations between the mirror and its surrounding objects for a reliable mirror localization. Our model first acquires class-specific knowledge of the surrounding objects via a semantic side-path. It then uses two novel modules to exploit semantic associations: 1) an Associations Exploration (AE) Module to extract the associations of the scene objects based on fully connected graph models, and 2) a Quadruple-Graph (QG) Module to facilitate the diffusion and aggregation of semantic association knowledge using graph convolutions. Extensive experiments show that our method outperforms the existing methods and sets the new state-of-the-art on both PMD dataset (f-measure: 0.844) and MSD dataset (f-measure: 0.889). Code is available at https://github.com/guanhuankang/Learning-Semantic-Associations-for-Mirror-Detection.

----

## [578] SketchEdit: Mask-Free Local Image Manipulation with Partial Sketches

**Authors**: *Yu Zeng, Zhe Lin, Vishal M. Patel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00586](https://doi.org/10.1109/CVPR52688.2022.00586)

**Abstract**:

Sketch-based image manipulation is an interactive image editing task to modify an image based on input sketches from users. Existing methods typically formulate this task as a conditional inpainting problem, which requires users to draw an extra mask indicating the region to modify in addition to sketches. The masked regions are regarded as holes and filled by an inpainting model conditioned on the sketch. With this formulation, paired training data can be easily obtained by randomly creating masks and extracting edges or contours. Although this setup simplifies data preparation and model design, it complicates user interaction and discards useful information in masked regions. To this end, we investigate a new paradigm of sketch-based image manipulation: mask-free local image manipulation, which only requires sketch inputs from users and utilizes the entire original image. Given an image and sketch, our model automatically predicts the target modification region and encodes it into a structure agnostic style vector. A generator then synthesizes the new image content based on the style vector and sketch. The manipulated image is finally produced by blending the generator output into the modification region of the original image. Our model can be trained in a self-supervised fashion by learning the reconstruction of an image region from the style vector and sketch. The proposed method offers simpler and more intuitive user workflows for sketch-based image manipulation and provides better results than previous approaches. More results, code and interactive demo will be available at https://zengxianyu.github.io/sketchedit.

----

## [579] Investigating Tradeoffs in Real-World Video Super-Resolution

**Authors**: *Kelvin C. K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00587](https://doi.org/10.1109/CVPR52688.2022.00587)

**Abstract**:

The diversity and complexity of degradations in real-world video super-resolution (VSR) pose non-trivial challenges in inference and training. First, while long-term propagation leads to improved performance in cases of mild degradations, severe in-the-wild degradations could be exaggerated through propagation, impairing output quality. To balance the tradeoff between detail synthesis and artifact suppression, we found an image precleaning stage in-dispensable to reduce noises and artifacts prior to propagation. Equipped with a carefully designed cleaning module, our RealBasicVSR outperforms existing methods in both quality and efficiency (Fig. 1). Second, real-world VSR models are often trained with diverse degradations to improve generalizability, requiring increased batch size to produce a stable gradient. Inevitably, the increased computational burden results in various problems, including 1) speed-performance tradeoff and 2) batch-length trade-off. To alleviate the first tradeoff, we propose a stochastic degradation scheme that reduces up to 40% of training time without sacrificing performance. We then analyze different training settings and suggest that employing longer sequences rather than larger batches during training allows more effective uses of temporal information, leading to more stable performance during inference. To facilitate fair comparisons, we propose the new VideoLQ dataset, which contains a large variety of real-world low-quality video sequences containing rich textures and patterns. Our dataset can serve as a common ground for benchmarking. Code, models, and the dataset are publicly available at https://github.com/ckkelvinchan/RealBasicVSR.

----

## [580] BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment

**Authors**: *Kelvin C. K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00588](https://doi.org/10.1109/CVPR52688.2022.00588)

**Abstract**:

A recurrent structure is a popular framework choice for the task of video super-resolution. The state-of-the-art method BasicVSR adopts bidirectional propagation with feature alignment to effectively exploit information from the entire input video. In this study, we redesign BasicVsr by proposing second-order grid propagation and flow-guided deformable alignment. We show that by empowering the re-current framework with enhanced propagation and align-ment, one can exploit spatiotemporal information across misaligned video frames more effectively. The new components lead to an improved performance under a simi-lar computational constraint. In particular, our model Ba-sicVSR++ surpasses BasicVSR by a significant 0.82 dB in PSNR with similar number of parameters. BasicVSR++ is generalizable to other video restoration tasks, and obtains three champions and one first runner-up in NTIRE 2021 video restoration challenge.

----

## [581] Inertia-Guided Flow Completion and Style Fusion for Video Inpainting

**Authors**: *Kaidong Zhang, Jingjing Fu, Dong Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00589](https://doi.org/10.1109/CVPR52688.2022.00589)

**Abstract**:

Physical objects have inertia, which resists changes in the velocity and motion direction. Inspired by this, we introduce inertia prior that optical flow, which reflects object motion in a local temporal window, keeps unchanged in the adjacent preceding or subsequent frame. We propose a flow completion network to align and aggregate flow features from the consecutive flow sequences based on the inertia prior. The corrupted flows are completed under the supervision of customized losses on reconstruction, flow smoothness, and consistent ternary census transform. The completed flows with high fidelity give rise to significant improvement on the video inpainting quality. Nevertheless, the existing flow-guided cross-frame warping methods fail to consider the lightening and sharpness variation across video frames, which leads to spatial incoherence after warping from other frames. To alleviate such problem, we propose the Adaptive Style Fusion Network (ASFN), which utilizes the style information extracted from the valid regions to guide the gradient refinement in the warped regions. Moreover, we design a data simulation pipeline to reduce the training difficulty of ASFN. Extensive experiments show the superiority of our method against the state-of-the-art methods quantitatively and qualitatively. The project page is at https://github.com/hitachinsk/ISVI.

----

## [582] Joint Global and Local Hierarchical Priors for Learned Image Compression

**Authors**: *Jun-Hyuk Kim, Byeongho Heo, Jong-Seok Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00590](https://doi.org/10.1109/CVPR52688.2022.00590)

**Abstract**:

Recently, learned image compression methods have out-performed traditional hand-crafted ones including BPG. One of the keys to this success is learned entropy models that estimate the probability distribution of the quantized latent representation. Like other vision tasks, most recent learned entropy models are based on convolutional neural networks (CNNs). However, CNNs have a limitation in modeling long-range dependencies due to their nature of local connectivity, which can be a significant bottleneck in image compression where reducing spatial redundancy is a key point. To overcome this issue, we propose a novel entropy model called Information Transformer (Informer) that exploits both global and local information in a content-dependent manner using an attention mechanism. Our experiments show that Informer improves rate-distortion performance over the state-of-the-art methods on the Kodak and Tecnick datasets without the quadratic computational complexity problem. Our source code is available at https://github.com/naver-ai/informer.

----

## [583] Reflash Dropout in Image Super-Resolution

**Authors**: *Xiangtao Kong, Xina Liu, Jinjin Gu, Yu Qiao, Chao Dong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00591](https://doi.org/10.1109/CVPR52688.2022.00591)

**Abstract**:

Dropout is designed to relieve the overfitting problem in high-level vision tasks but is rarely applied in lowlevel vision tasks, like image super-resolution (SR). As a classic regression problem, SR exhibits a different behaviour as high-level tasks and is sensitive to the dropout operation. However, in this paper, we show that appropriate usage of dropout benefits SR networks and improves the generalization ability. Specifically, dropout is better embedded at the end of the network and is significantly helpful for the multi-degradation settings. This discovery breaks our common sense and inspires us to explore its working mechanism. We further use two analysis tools - one is from a recent network interpretation work, and the other is specially designed for this task. The analysis results provide side proofs to our experimental findings and show us a new perspective to understand SR networks.

----

## [584] Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond

**Authors**: *Yi Yu, Wenhan Yang, Yap-Peng Tan, Alex C. Kot*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00592](https://doi.org/10.1109/CVPR52688.2022.00592)

**Abstract**:

Rain removal aims to remove rain streaks from images/videos and reduce the disruptive effects caused by rain. It not only enhances image/video visibility but also allows many computer vision algorithms to function properly. This paper makes the first attempt to conduct a comprehensive study on the robustness of deep learning-based rain removal methods against adversarial attacks. Our study shows that, when the image/video is highly degraded, rain removal methods are more vulnerable to the adversarial attacks as small distortions/perturbations become less noticeable or detectable. In this paper, we first present a comprehensive empirical evaluation of various methods at different levels of attacks and with various losses/targets to generate the perturbations from the perspective of human perception and machine analysis tasks. A systematic evaluation of key modules in existing methods is performed in terms of their robustness against adversarial attacks. From the insights of our analysis, we construct a more robust deraining method by integrating these effective modules. Finally, we examine various types of adversarial attacks that are specific to deraining problems and their effects on both human and machine vision tasks, including 1) rain region attacks, adding perturbations only in the rain regions to make the perturbations in the attacked rain images less visible; 2) object-sensitive attacks, adding perturbations only in regions near the given objects. Code is available at https://github.com/yuyi-sd/Robust_Rain_Removal.

----

## [585] Dreaming to Prune Image Deraining Networks

**Authors**: *Weiqi Zou, Yang Wang, Xueyang Fu, Yang Cao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00593](https://doi.org/10.1109/CVPR52688.2022.00593)

**Abstract**:

Convolutional image deraining networks have achieved great success while suffering from tremendous computational and memory costs. Most model compression methods require original data for iterative fine-tuning, which is limited in real-world applications due to storage, privacy, and transmission constraints. We note that it is overstretched to fine-tune the compressed model using self-collected data, as it exhibits poor generalization over images with different degradation characteristics. To address this problem, we propose a novel data-free compression framework for de-raining networks. It is based on our observation that deep degradation representations can be clustered by degradation characteristics (types of rain) while independent of image content. Therefore, in our framework, we “dream” diverse in-distribution degraded images using a deep inversion paradigm, thus leveraging them to distill the pruned model. Specifically, we preserve the performance of the pruned model in a dual-branch way. In one branch, we invert the pre-trained model (teacher) to reconstruct the degraded inputs that resemble the original distribution and employ the orthogonal regularization for deep features to yield degradation diversity. In the other branch, the pruned model (student) is distilled to fit the teacher's original statistical modeling on these dreamed inputs. Further, an adaptive pruning scheme is proposed to determine the hierarchical sparsity, which alleviates the regression drift of the initial pruned model. Experiments on various deraining datasets demonstrate that our method can reduce about 40% FLOPs of the state-of-the-art models while maintaining comparable performance without original data.

----

## [586] LC-FDNet: Learned Lossless Image Compression with Frequency Decomposition Network

**Authors**: *Hochang Rhee, Yeong Il Jang, Seyun Kim, Nam Ik Cho*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00594](https://doi.org/10.1109/CVPR52688.2022.00594)

**Abstract**:

Recent learning-based lossless image compression methods encode an image in the unit of subimages and achieve comparable performances to conventional non-learning algorithms. However, these methods do not consider the performance drop in the high-frequency region, giving equal consideration to the low and high-frequency areas. In this paper, we propose a new lossless image compression method that proceeds the encoding in a coarse-to-fine manner to separate and process low and high-frequency regions differently. We initially compress the low-frequency components and then use them as additional input for encoding the remaining high-frequency region. The low-frequency components act as a strong prior in this case, which leads to improved estimation in the high-frequency area. In addition, we design the frequency decomposition process to be adptive to color channel, spatial location, and image characteristics. As a result, our method derives an image-specific optimal ratio of low/high-frequency components. Experiments show that the proposed method achieves state-of-the-art performance for benchmark high-resolution datasets.

----

## [587] Exposure Normalization and Compensation for Multiple-Exposure Correction

**Authors**: *Jie Huang, Yajing Liu, Xueyang Fu, Man Zhou, Yang Wang, Feng Zhao, Zhiwei Xiong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00595](https://doi.org/10.1109/CVPR52688.2022.00595)

**Abstract**:

Images captured with improper exposures usually bring unsatisfactory visual effects. Previous works mainly focus on either underexposure or overexposure correction, resulting in poor generalization to various exposures. An alternative solution is to mix the multiple exposure data for training a single network. However, the procedures of correcting underexposure and overexposure to normal exposures are much different from each other, leading to large discrepancies for the network in correcting multiple-exposures, thus resulting in poor performance. The key point to address this issue lies in bridging different exposure representations. To achieve this goal, we design a multiple exposure correction framework based on an Exposure Normalization and Compensation (ENC) module. Specifically, the ENC module consists of an exposure normalization part for mapping different exposure features to the exposure-invariant feature space, and a compensation part for integrating the initial features unprocessed by the exposure normalization part to ensure the completeness of information. Besides, to further alleviate the imbalanced performance caused by variations in the optimization process, we introduce a parameter regularization fine-tuning strategy to improve the performance of the worst-performed exposure without degrading other exposures. Our model empowered by ENC outperforms the existing methods by more than 2dB and is robust to multiple image enhancement tasks, demonstrating its effectiveness and generalization capability for real-world applications. Code: https://github.com/KevinJ-Huang/ExposureNorm-Compensation.

----

## [588] Revisiting Temporal Alignment for Video Restoration

**Authors**: *Kun Zhou, Wenbo Li, Liying Lu, Xiaoguang Han, Jiangbo Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00596](https://doi.org/10.1109/CVPR52688.2022.00596)

**Abstract**:

Long-range temporal alignment is critical yet challenging for video restoration tasks. Recently, some works attempt to divide the long-range alignment into several sub-alignments and handle them progressively. Although this operation is helpful in modeling distant correspondences, error accumulation is inevitable due to the propagation mechanism. In this work, we present a novel, generic iterative alignment module which employs a gradual refinement scheme for sub-alignments, yielding more accurate motion compensation. To further enhance the alignment accuracy and temporal consistency, we develop a non-parametric re-weighting method, where the importance of each neighboring frame is adaptively evaluated in a spatial-wise way for aggregation. By virtue of the proposed strategies, our model achieves state-of-the-art performance on multiple benchmarks across a range of video restoration tasks including video super-resolution, denoising and deblurring.

----

## [589] LSVC: A Learning-based Stereo Video Compression Framework

**Authors**: *Zhenghao Chen, Guo Lu, Zhihao Hu, Shan Liu, Wei Jiang, Dong Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00598](https://doi.org/10.1109/CVPR52688.2022.00598)

**Abstract**:

In this work, we propose the first end-to-end optimized framework for compressing automotive stereo videos (i.e., stereo videos from autonomous driving applications) from both left and right views. Specifically, when compressing the current frame from each view, our framework reduces temporal redundancy by performing motion compensation using the reconstructed intra-view adjacent frame and at the same time exploits binocular redundancy by conducting disparity compensation using the latest reconstructed cross-view frame. Moreover, to effectively compress the introduced motion and disparity offsets for better compensation, we further propose two novel schemes called motion residual compression and disparity residual compression to respectively generate the predicted motion offset and disparity offset from the previously compressed motion offset and disparity offset, such that we can more effectively compress residual offset information for better bit-rate saving. Overall, the entire framework is implemented by the fully-differentiable modules and can be optimized in an end-to-end manner. Our comprehensive experiments on three automotive stereo video benchmarks Cityscapes, KITTI 2012 and KITTI 2015 demonstrate that our proposed framework outperforms the learning-based single-view video codec and the traditional hand-crafted multi-view video codec.

----

## [590] Learning based Multi-modality Image and Video Compression

**Authors**: *Guo Lu, Tianxiong Zhong, Jing Geng, Qiang Hu, Dong Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00599](https://doi.org/10.1109/CVPR52688.2022.00599)

**Abstract**:

Multi-modality (i.e., multi-sensor) data is widely used in various vision tasks for more accurate or robust perception. However, the increased data modalities bring new challenges for data storage and transmission. The existing data compression approaches usually adopt individual codecs for each modality without considering the correlation between different modalities. This work proposes a multi-modality compression framework for infrared and visible image pairs by exploiting the cross-modality redun-dancy. Specifically, given the image in the reference modality (e.g., the infrared image), we use the channel-wise alignment module to produce the aligned features based on the affine transform. Then the aligned feature is used as the context information for compressing the image in the current modality (e.g., the visible image), and the corresponding affine coefficients are losslessly compressed at negligible cost. Furthermore, we introduce the Transformer-based spatial alignment module to exploit the correlation between the intermediate features in the decoding procedures for different modalities. Our framework is very flexible and easily extended for multi-modality video compression. Experimental results show our proposed framework outperforms the traditional and learning-based single modality compression methods on the FLIR and KAIST datasets.

----

## [591] Transformer Based Line Segment Classifier with Image Context for Real-Time Vanishing Point Detection in Manhattan World

**Authors**: *Xin Tong, Xianghua Ying, Yongjie Shi, Ruibin Wang, Jinfa Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00600](https://doi.org/10.1109/CVPR52688.2022.00600)

**Abstract**:

Previous works on vanishing point detection usually use geometric prior for line segment clustering. We find that image context can also contribute to accurate line classification. Based on this observation, we propose to classify line segments into three groups according to three unknown-but-sought vanishing points with Manhattan world assumption, using both geometric information and image context in this work. To achieve this goal, we propose a novel Transformer based Line segment Classifier (TLC) that can group line segments in images and estimate the corresponding vanishing points. In TLC, we design a line segment descriptor to represent line segments using their positions, directions and local image contexts. Transformer based feature fusion module is used to capture global features from all line segments, which is proved to improve the classification performance significantly in our experiments. By using a network to score line segments for outlier rejection, vanishing points can be got by Singular Value Decomposition (SVD) from the classified lines. The proposed method runs at 25 fps on one NVIDIA 2080Ti card for vanishing point detection. Experimental results on synthetic and real-world datasets demonstrate that our method is superior to other state-of-the-art methods on the balance between accuracy and efficiency, while keeping stronger generalization capability when trained and evaluated on different datasets.

----

## [592] Deep vanishing point detection: Geometric priors make dataset variations vanish

**Authors**: *Yancong Lin, Ruben Wiersma, Silvia L. Pintea, Klaus Hildebrandt, Elmar Eisemann, Jan C. van Gemert*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00601](https://doi.org/10.1109/CVPR52688.2022.00601)

**Abstract**:

Deep learning has improved vanishing point detection in images. Yet, deep networks require expensive annotated datasets trained on costly hardware and do not generalize to even slightly different domains, and minor problem variants. Here, we address these issues by injecting deep vanishing point detection networks with prior knowledge. This prior knowledge no longer needs to be learned from data, saving valuable annotation efforts and compute, unlocking realistic few-sample scenarios, and reducing the impact of domain changes. Moreover, the interpretability of the priors allows to adapt deep networks to minor problem variations such as switching between Manhattan and non-Manhattan worlds. We seamlessly incorporate two geometric priors: (i) Hough Transform – mapping image pixels to straight lines, and (ii) Gaussian sphere – mapping lines to great circles whose intersections denote vanishing points. Experimentally, we ablate our choices and show comparable accuracy to existing models in the large-data setting. We validate our model's improved data efficiency, robustness to domain changes, adaptability to non-Manhattan settings.

----

## [593] Stereo Depth from Events Cameras: Concentrate and Focus on the Future

**Authors**: *Yeongwoo Nam, S. Mohammad Mostafavi I., Kuk-Jin Yoon, Jonghyun Choi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00602](https://doi.org/10.1109/CVPR52688.2022.00602)

**Abstract**:

Neuromorphic cameras or event cameras mimic human vision by reporting changes in the intensity in a scene, instead of reporting the whole scene at once in a form of an image frame as performed by conventional cameras. Events are streamed data that are often dense when either the scene changes or the camera moves rapidly. The rapid movement causes the events to be overridden or missed when creating a tensor for the machine to learn on. To alleviate the event missing or overriding issue, we propose to learn to concentrate on the dense events to produce a compact event representation with high details for depth estimation. Specifically, we learn a model with events from both past and future but infer only with past data with the predicted future. We initially estimate depth in an event-only setting but also propose to further incorporate images and events by a hier-archical event and intensity combination network for better depth estimation. By experiments in challenging real-world scenarios, we validate that our method outperforms prior arts even with low computational cost. Code is available at: https://github.com/yonseivnl/se-cff.

----

## [594] Volumetric Bundle Adjustment for Online Photorealistic Scene Capture

**Authors**: *Ronald Clark*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00603](https://doi.org/10.1109/CVPR52688.2022.00603)

**Abstract**:

Efficient photorealistic scene capture is a challenging task. Current online reconstruction systems can operate very efficiently, but images generated from the models captured by these systems are often not photorealistic. Recent approaches based on neural volume rendering can render novel views at high fidelity, but they often require a long time to train, making them impractical for applications that require real-time scene capture. In this paper, we propose a system that can reconstruct photorealistic models of complex scenes in an efficient manner. Our system processes images online, i.e. it can obtain a good quality estimate of both the scene geometry and appearance at roughly the same rate the video is captured. To achieve the efficiency, we propose a hierarchical feature volume using VDB grids. This representation is memory efficient and allows for fast querying of the scene information. Secondly, we introduce a novel optimization technique that improves the efficiency of the bundle adjustment which allows our system to converge to the target camera poses and scene geometry much faster. Experiments on real-world scenes show that our method outperforms existing systems in terms of efficiency and capture quality. To the best of our knowledge, this is the first method that can achieve online photorealistic scene capture.

----

## [595] Neural Volumetric Object Selection

**Authors**: *Zhongzheng Ren, Aseem Agarwala, Bryan C. Russell, Alexander G. Schwing, Oliver Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00604](https://doi.org/10.1109/CVPR52688.2022.00604)

**Abstract**:

We introduce an approach for selecting objects in neural volumetric 3D representations, such as multi-plane images (MPI) and neural radiance fields (NeRF). Our approach takes a set of foreground and background 2D user scribbles in one view and automatically estimates a 3D segmentation of the desired object, which can be rendered into novel views. To achieve this result, we propose a novel voxel feature embedding that incorporates the neural volumetric 3D representation and multi-view image features from all input views. To evaluate our approach, we introduce a new dataset of human-provided segmentation masks for depicted objects in real-world multi-view scene captures. We show that our approach out-performs strong baselines, including 2D segmentation and 3D segmentation approaches adapted to our task.

----

## [596] HVH: Learning a Hybrid Neural Volumetric Representation for Dynamic Hair Performance Capture

**Authors**: *Ziyan Wang, Giljoo Nam, Tuur Stuyck, Stephen Lombardi, Michael Zollhöfer, Jessica K. Hodgins, Christoph Lassner*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00605](https://doi.org/10.1109/CVPR52688.2022.00605)

**Abstract**:

Capturing and rendering life-like hair is particularly challenging due to its fine geometric structure, the complex physical interaction and its non-trivial visual appearance. Yet, hair is a critical component for believable avatars. In this paper, we address the aforementioned problems: 1) we use a novel, volumetric hair representation that is composed of thousands of primitives. Each primitive can be rendered efficiently, yet realistically, by building on the latest advances in neural rendering. 2) To have a reliable control signal, we present a novel way of tracking hair on the strand level. To keep the computational effort manageable, we use guide hairs and classic techniques to expand those into a dense hood of hair. 3) To better enforce temporal consistency and generalization ability of our model, we further optimize the 3D scene flow of our representation with multiview optical flow, using volumetric raymarching. Our method can not only create realistic renders of recorded multi-view sequences, but also create renderings for new hair configurations by providing new control signals. We compare our method with existing work on viewpoint synthesis and drivable animation and achieve state-of-the-art results. https://ziyanw1.github.io/hvh.

----

## [597] NeuralHOFusion: Neural Volumetric Rendering under Human-object Interactions

**Authors**: *Yuheng Jiang, Suyi Jiang, Guoxing Sun, Zhuo Su, Kaiwen Guo, Minye Wu, Jingyi Yu, Lan Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00606](https://doi.org/10.1109/CVPR52688.2022.00606)

**Abstract**:

4D modeling of human-object interactions is critical for numerous applications. However, efficient volumetric capture and rendering of complex interaction scenarios, especially from sparse inputs, remain challenging. In this paper, we propose NeuralHOFusion, a neural approach for volumetric human-object capture and rendering using sparse consumer RGBD sensors. It marries traditional non-rigid fusion with recent neural implicit modeling and blending advances, where the captured humans and objects are layer-wise disentangled. For geometry modeling, we propose a neural implicit inference scheme with non-rigid key-volume fusion, as well as a template-aid robust object tracking pipeline. Our scheme enables detailed and complete geometry generation under complex interactions and occlusions. Moreover, we introduce a layer-wise human-object texture rendering scheme, which combines volumetric and image-based rendering in both spatial and temporal domains to obtain photo-realistic results. Extensive experiments demonstrate the effectiveness and efficiency of our approach in synthesizing photo-realistic free-view results under complex human-object interactions.

----

## [598] BNV-Fusion: Dense 3D Reconstruction using Bi-level Neural Volume Fusion

**Authors**: *Kejie Li, Yansong Tang, Victor Adrian Prisacariu, Philip H. S. Torr*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00607](https://doi.org/10.1109/CVPR52688.2022.00607)

**Abstract**:

Dense 3D reconstruction from a stream of depth images is the key to many mixed reality and robotic applications. Although methods based on Truncated Signed Distance Function (TSDF) Fusion have advanced the field over the years, the TSDF volume representation is confronted with striking a balance between the robustness to noisy measurements and maintaining the level of detail. We present Bi-level Neural Volume Fusion (BNV-Fusion), which leverages recent advances in neural implicit representations and neural rendering for dense 3D reconstruction. In order to incrementally integrate new depth maps into a global neural implicit representation, we propose a novel bi-level fusion strategy that considers both efficiency and reconstruction quality by design. We evaluate the proposed method on multiple datasets quantitatively and qualitatively, demonstrating a significant improvement over existing methods.

----

## [599] Input-level Inductive Biases for 3D Reconstruction

**Authors**: *Wang Yifan, Carl Doersch, Relja Arandjelovic, João Carreira, Andrew Zisserman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00608](https://doi.org/10.1109/CVPR52688.2022.00608)

**Abstract**:

Much of the recent progress in 3D vision has been driven by the development of specialized architectures that incorporate geometrical inductive biases. In this paper we tackle 3D reconstruction using a domain agnostic architecture and study how to inject the same type of inductive biases directly as extra inputs to the model. This approach makes it possible to apply existing general models, such as Perceivers, on this rich domain, without the need for architectural changes, while simultaneously maintaining data efficiency of bespoke models. In particular we study how to encode cameras, projective ray incidence and epipolar geometry as model inputs, and demonstrate competitive multi-view depth estimation performance on multiple benchmarks.

----



[Go to the previous page](CVPR-2022-list02.md)

[Go to the next page](CVPR-2022-list04.md)

[Go to the catalog section](README.md)