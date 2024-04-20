## [2000] Zero-Shot Text-to-Parameter Translation for Game Character Auto-Creation

        **Authors**: *Rui Zhao, Wei Li, Zhipeng Hu, Lincheng Li, Zhengxia Zou, Zhenwei Shi, Changjie Fan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02013](https://doi.org/10.1109/CVPR52729.2023.02013)

        **Abstract**:

        Recent popular Role-Playing Games (RPGs) saw the great success of character auto-creation systems. The bone-drivenface model controlled by continuous parameters (like the position of bones) and discrete parameters (like the hairstyles) makes it possible for users to personalize and customize in-game characters. Previous in-game character auto-creation systems are mostly image-driven, where facial parameters are optimized so that the rendered character looks similar to the reference face photo. This paper proposes a novel text-to-parameter translation method (T2P) to achieve zero-shot text-driven game character auto-creation. With our method, users can create a vivid in-game character with arbitrary text description without using any reference photo or editing hundreds of parameters manually. In our method, taking the power of large-scale pre-trained multi-modal CLIP and neural rendering, T2P searches both continuous facial parameters and discrete facial parameters in a unified framework. Due to the discontinuous parameter representation, previous methods have difficulty in effectively learning discrete facial parameters. T2p, to our best knowledge, is the first method that can handle the optimization of both discrete and continuous parameters. Experimental results show that T2P can generate high-quality and vivid game characters with given text prompts. T2P outperforms other SOTA text-to-3D generation methods on both objective evaluations and subjective evaluations.

        ----

        ## [2001] Learning Locally Editable Virtual Humans

        **Authors**: *Hsuan-I Ho, Lixin Xue, Jie Song, Otmar Hilliges*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02014](https://doi.org/10.1109/CVPR52729.2023.02014)

        **Abstract**:

        In this paper, we propose a novel hybrid representation and end-to-end trainable network architecture to model fully editable and customizable neural avatars. At the core of our work lies a representation that combines the modeling power of neural fields with the ease of use and inherent 3D consistency of skinned meshes. To this end, we construct a trainable feature codebook to store local geometry and texture features on the vertices of a deformable body model, thus exploiting its consistent topology under articulation. This representation is then employed in a generative auto-decoder architecture that admits fitting to unseen scans and sampling of realistic avatars with varied appearances and geometries. Furthermore, our representation allows local editing by swapping local features between 3D assets. To verify our method for avatar creation and editing, we contribute a new highquality dataset, dubbed CustomHumans, for training and evaluation. Our experiments quantitatively and qualitatively show that our method generates diverse detailed avatars and achieves better model fitting performance compared to state-of-the-art methods. Our code and dataset are available at https://ait.ethz.ch/customhumans.

        ----

        ## [2002] Auto-CARD: Efficient and Robust Codec Avatar Driving for Real-time Mobile Telepresence

        **Authors**: *Yonggan Fu, Yuecheng Li, Chenghui Li, Jason M. Saragih, Peizhao Zhang, Xiaoliang Dai, Yingyan Celine Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02015](https://doi.org/10.1109/CVPR52729.2023.02015)

        **Abstract**:

        Real-time and robust photorealistic avatars for telepresence in AR/VR have been highly desired for enabling im-mersive photorealistic telepresence. However, there still exists one key bottleneck: the considerable computational expense needed to accurately infer facial expressions captured from headset-mounted cameras with a quality level that can match the realism of the avatar's human appearance. To this end, we propose a framework called Auto-CARD, which for the first time enables realtime and robust driving of Codec Avatars when exclusively using merely on-device computing resources. This is achieved by minimizing two sources of redundancy. First, we develop a dedicated neural architecture search technique called AVE-NAS for avatar encoding in AR/VR, which explicitly boosts both the searched architectures' robustness in the presence of extreme facial ex-pressions and hardware friendliness on fast evolving AR/VR headsets. Second, we leverage the temporal redundancy in consecutively captured images during continuous rendering and develop a mechanism dubbed LATEX to skip the computation of redundant frames. Specifically, we first identify an opportunity from the linearity of the latent space derived by the avatar decoder and then propose to perform adaptive latent extrapolation for redundant frames. For evaluation, we demonstrate the efficacy of our Auto-CARD framework in realtime Codec Avatar driving settings, where we achieve a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$5.05\times$</tex>
 speedup on Meta Quest 2 while maintaining a compa-rable or even better animation quality than state-of-the-art avatar encoder designs.

        ----

        ## [2003] Ham2Pose: Animating Sign Language Notation into Pose Sequences

        **Authors**: *Rotem Shalev-Arkushin, Amit Moryossef, Ohad Fried*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02016](https://doi.org/10.1109/CVPR52729.2023.02016)

        **Abstract**:

        Translating spoken languages into Sign languages is necessary for open communication between the hearing and hearing-impaired communities. To achieve this goal, we propose the first method for animating a text written in HamNoSys, a lexical Sign language notation, into signed pose sequences. As HamNoSys is universal by design, our proposed method offers a generic solution invariant to the target Sign language. Our method gradually generates pose predictions using transformer encoders that create meaningful representations of the text and poses while considering their spatial and temporal information. We use weak supervision for the training process and show that our method succeeds in learning from partial and inaccurate data. Additionally, we offer a new distance measurement that considers missing keypoints, to measure the distance between pose sequences using DTW-MJE. We validate its correctness using AUTSL, a large-scale Sign language dataset, show that it measures the distance between pose sequences more accurately than existing measurements, and use it to assess the quality of our generated pose sequences. Code for the data pre-processing, the model, and the distance measurement is publicly released for future research.

        ----

        ## [2004] PointAvatar: Deformable Point-Based Head Avatars from Videos

        **Authors**: *Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J. Black, Otmar Hilliges*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02017](https://doi.org/10.1109/CVPR52729.2023.02017)

        **Abstract**:

        The ability to create realistic animatable and relightable head avatars from casual video sequences would open up wide ranging applications in communication and entertainment. Current methods either build on explicit 3D morphable meshes (3DMM) or exploit neural implicit representations. The former are limited by fixed topology, while the latter are non-trivial to deform and inefficient to render. Furthermore, existing approaches entangle lighting and albedo, limiting the ability to re-render the avatar in new environments. In contrast, we propose PointAvatar, a deformable point-based representation that disentangles the source color into intrinsic albedo and normal-dependent shading. We demonstrate that PointAvatar bridges the gap between existing mesh- and implicit representations, combining high-quality geometry and appearance with topological flexibility, ease of deformation and rendering efficiency. We show that our method is able to generate animatable 3D avatars using monocular videos from multiple sources including hand-held smartphones, laptop webcams and internet videos, achieving state-of-the-art quality in challenging cases where previous methods fail, e.g., thin hair strands, while being significantly more efficient in training than competing methods.

        ----

        ## [2005] PAniC-3D: Stylized Single-view 3D Reconstruction from Portraits of Anime Characters

        **Authors**: *Shuhong Chen, Kevin Zhang, Yichun Shi, Heng Wang, Yiheng Zhu, Guoxian Song, Sizhe An, Janus Kristjansson, Xiao Yang, Matthias Zwicker*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02018](https://doi.org/10.1109/CVPR52729.2023.02018)

        **Abstract**:

        We propose PAniC-3D, a system to reconstruct stylized 3D character heads directly from illustrated (p)ortraits of (ani)me (c)haracters. Our anime-style domain poses unique challenges to single-view reconstruction; compared to natural images of human heads, character portrait illustrations have hair and accessories with more complex and diverse geometry, and are shaded with non-photorealistic contour lines. In addition, there is a lack of both 3D model and portrait illustration data suitable to train and evaluate this ambiguous stylized reconstruction task. Facing these challenges, our proposed PAniC-3D architecture crosses the illustration-to-3D domain gap with a line-filling model, and represents sophisticated geometries with a volumetric radiance field. We train our system with two large new datasets (11.2k Vroid 3D models, 1k Vtuber portrait illustrations), and evaluate on a novel AnimeRecon benchmark of illustration-to-3D pairs. PAniC-3D significantly outper-forms baseline methods, and provides data to establish the task of stylized reconstruction from portrait illustrations.

        ----

        ## [2006] HandNeRF: Neural Radiance Fields for Animatable Interacting Hands

        **Authors**: *Zhiyang Guo, Wengang Zhou, Min Wang, Li Li, Houqiang Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02019](https://doi.org/10.1109/CVPR52729.2023.02019)

        **Abstract**:

        We propose a novel framework to reconstruct accurate appearance and geometry with neural radiance fields (NeRF) for interacting hands, enabling the rendering of photo-realistic images and videos for gesture animation from arbitrary views. Given multi-view images of a single hand or interacting hands, an off-the-shelf skeleton estimator is first employed to parameterize the hand poses. Then we design a pose-driven deformation field to establish correspondence from those different poses to a shared canonical space, where a pose-disentangled NeRF for one hand is optimized. Such unified modeling efficiently complements the geometry and texture cues in rarely-observed areas for both hands. Meanwhile, we further leverage the pose priors to generate pseudo depth maps as guidance for occlusion aware density learning. Moreover, a neural feature distillation method is proposed to achieve cross-domain alignment for color optimization. We conduct extensive experiments to verify the merits of our proposed HandNeRF and report a series of state-of-the-art results both qualitatively and quantitatively on the large-scale InterHand2.6M dataset.

        ----

        ## [2007] VGFlow: Visibility guided Flow Network for Human Reposing

        **Authors**: *Rishabh Jain, Krishna Kumar Singh, Mayur Hemani, Jingwan Lu, Mausoom Sarkar, Duygu Ceylan, Balaji Krishnamurthy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02020](https://doi.org/10.1109/CVPR52729.2023.02020)

        **Abstract**:

        The task of human reposing involves generating a realistic image of a person standing in an arbitrary conceivable pose. There are multiple difficulties in generating perceptually accurate images, and existing methods suffer from limitations in preserving texture, maintaining pattern co-herence, respecting cloth boundaries, handling occlusions, manipulating skin generation, etc. These difficulties are further exacerbated by the fact that the possible space of pose orientation for humans is large and variable, the nature of clothing items is highly non-rigid, and the diversity in body shape differs largely among the population. To alle-viate these difficulties and synthesize perceptually accurate images, we propose VGFlow. Our model uses a visibility-guided flow module to disentangle the flow into visible and invisible parts of the target for simultaneous texture preser-vation and style manipulation. Furthermore, to tackle dis-tinct body shapes and avoid network artifacts, we also in-corporate a self-supervised patch-wise “realness” loss to improve the output. VGFlow achieves state-of-the-art results as observed qualitatively and quantitatively on different image quality metrics (SSIM, LPIPS, FID). Results can be downloaded from Project Webpage

        ----

        ## [2008] Clothed Human Performance Capture with a Double-layer Neural Radiance Fields

        **Authors**: *Kangkan Wang, Guofeng Zhang, Suxu Cong, Jian Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02021](https://doi.org/10.1109/CVPR52729.2023.02021)

        **Abstract**:

        This paper addresses the challenge of capturing performance for the clothed humans from sparse-view or monocular videos. Previous methods capture the performance of full humans with a personalized template or recover the garments from a single frame with static human poses. However, it is inconvenient to extract cloth semantics and capture clothing motion with one-piece template, while single frame-based methods may suffer from instable tracking across videos. To address these problems, we propose a novel method for human performance capture by tracking clothing and human body motion separately with a double-layer neural radiance fields (NeRFs). Specifically, we propose a double-layer NeRFsfor the body and garments, and track the densely deforming template of the clothing and body by jointly optimizing the deformation fields and the canonical double-layer NeRFs. In the optimization, we introduce a physics-aware cloth simulation network which can help generate physically plausible cloth dynamics and body-cloth interactions. Compared with existing methods, our method is fully differentiable and can capture both the body and clothing motion robustly from dynamic videos. Also, our method represents the clothing with an independent NeRFs, allowing us to model implicit fields of general clothes feasibly. The experimental evaluations validate its effectiveness on real multi-view or monocular videos.

        ----

        ## [2009] POEM: Reconstructing Hand in a Point Embedded Multi-view Stereo

        **Authors**: *Lixin Yang, Jian Xu, Licheng Zhong, Xinyu Zhan, Zhicheng Wang, Kejian Wu, Cewu Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02022](https://doi.org/10.1109/CVPR52729.2023.02022)

        **Abstract**:

        Enable neural networks to capture 3D geometrical-aware features is essential in multi-view based vision tasks. Previous methods usually encode the 3D information of multi-view stereo into the 2D features. In contrast, we present a novel method, named POEM, that directly operates on the 3D POints Embedded in the Multi-view stereo for reconstructing hand mesh in it. Point is a natural form of 3D information and an ideal medium for fusing features across views, as it has different projections on different views. Our method is thus in light of a simple yet effective idea, that a complex 3D hand mesh can be represented by a set of 3D points that 1) are embedded in the multi-view stereo, 2) carry features from the multi-view images, and 3) encircle the hand. To leverage the power of points, we design two operations: point-based feature fusion and cross-set point attention mechanism. Evaluation on three challenging multi-view datasets shows that POEM outperforms the state-of-the-art in hand mesh reconstruction. Code and models are available for research at github.com/lixiny/POEM

        ----

        ## [2010] FlexNeRF: Photorealistic Free-viewpoint Rendering of Moving Humans from Sparse Views

        **Authors**: *Vinoj Jayasundara, Amit Agrawal, Nicolas Heron, Abhinav Shrivastava, Larry S. Davis*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02023](https://doi.org/10.1109/CVPR52729.2023.02023)

        **Abstract**:

        We present FlexNeRF, a method for photorealistic free-viewpoint rendering of humans in motion from monocular videos. Our approach works well with sparse views, which is a challenging scenario when the subject is exhibiting fast/complex motions. We propose a novel approach which jointly optimizes a canonical time and pose configuration, with a pose-dependent motion field and pose-independent temporal deformations complementing each other. Thanks to our novel temporal and cyclic consistency constraints along with additional losses on intermediate representation such as segmentation, our approach provides high quality outputs as the observed views become sparser. We empirically demonstrate that our method significantly outperforms the state-of-the-art on public benchmark datasets as well as a self-captured fashion dataset. The project page is available at: https://flex-nerf.github.io/

        ----

        ## [2011] Flow Supervision for Deformable NeRF

        **Authors**: *Chaoyang Wang, Lachlan Ewen MacDonald, László A. Jeni, Simon Lucey*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02024](https://doi.org/10.1109/CVPR52729.2023.02024)

        **Abstract**:

        In this paper we present a new method for deformable NeRF that can directly use optical flow as supervision. We overcome the major challenge with respect to the computationally inefficiency of enforcing the flow constraints to the backward deformation field, used by deformable NeRFs. Specifically, we show that inverting the backward deformation function is actually not needed for computing scene flows between frames. This insight dramatically simplifies the problem, as one is no longer constrained to deformation functions that can be analytically inverted. Instead, thanks to the weak assumptions required by our derivation based on the inverse function theorem, our approach can be extended to a broad class of commonly used backward deformation field. We present results on monocular novel view synthesis with rapid object motion, and demonstrate significant improvements over baselines without flow supervision.

        ----

        ## [2012] Building Rearticulable Models for Arbitrary 3D Objects from 4D Point Clouds

        **Authors**: *Shaowei Liu, Saurabh Gupta, Shenlong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02025](https://doi.org/10.1109/CVPR52729.2023.02025)

        **Abstract**:

        We build rearticulable models for arbitrary everyday man-made objects containing an arbitrary number of parts that are connected together in arbitrary ways via 1 degree-of-freedom joints. Given point cloud videos of such everyday objects, our method identifies the distinct object parts, what parts are connected to what other parts, and the properties of the joints connecting each part pair. We do this by jointly optimizing the part segmentation, transformation, and kinematics using a novel energy minimization frame-work. Our inferred animatable models, enables retargeting to novel poses with sparse point correspondences guidance. We test our method on a new articulating robot dataset, and the Sapiens dataset with common daily objects. Experiments show that our method outperforms two leading prior works on various metrics.

        ----

        ## [2013] Implicit 3D Human Mesh Recovery using Consistency with Pose and Shape from Unseen-view

        **Authors**: *Hanbyel Cho, Yooshin Cho, Jaesung Ahn, Junmo Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02026](https://doi.org/10.1109/CVPR52729.2023.02026)

        **Abstract**:

        From an image of a person, we can easily infer the natural 3D pose and shape of the person even if ambiguity exists. This is because we have a mental model that allows us to imagine a person's appearance at different viewing directions from a given image and utilize the consistency between them for inference. However, existing human mesh recovery methods only consider the direction in which the image was taken due to their structural limitations. Hence, we propose “Implicit 3D Human Mesh Recovery (ImpHMR)” that can implicitly imagine a person in 3D space at the feature-level via Neural Feature Fields. In ImpHMR, feature fields are generated by CNN-based image encoder for a given image. Then, the 2D feature map is volume-rendered from the feature field for a given viewing direction, and the pose and shape parameters are regressed from the feature. To utilize consistency with pose and shape from unseen-view, if there are 3D labels, the model predicts results including the silhouette from an arbitrary direction and makes it equal to the rotated ground-truth. In the case of only 2D labels, we perform self-supervised learning through the constraint that the pose and shape parameters inferred from different directions should be the same. Extensive evaluations show the efficacy of the proposed method.

        ----

        ## [2014] One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer

        **Authors**: *Jing Lin, Ailing Zeng, Haoqian Wang, Lei Zhang, Yu Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02027](https://doi.org/10.1109/CVPR52729.2023.02027)

        **Abstract**:

        Whole-body mesh recovery aims to estimate the 3D human body, face, and hands parameters from a single image. It is challenging to perform this task with a single network due to resolution issues, i.e., the face and hands are usually located in extremely small regions. Existing works usually detect hands and faces, enlarge their resolution to feed in a specific network to predict the parameter, and finally fuse the results. While this copy-paste pipeline can capture the fine-grained details of the face and hands, the connections between different parts cannot be easily recovered in late fusion, leading to implausible 3D rotation and unnatural pose. In this work, we propose a one-stage pipeline for expressive whole-body mesh recovery, named OSX, without separate networks for each part. Specifically, we design a Component Aware Transformer (CAT) composed of a global body encoder and a local face/hand decoder. The encoder predicts the body parameters and provides a high-quality feature map for the decoder, which performs a feature-level upsample-crop scheme to extract highresolution part-specific features and adopt keypointguided deformable attention to estimate hand and face precisely. The whole pipeline is simple yet effective without any manual post-processing and naturally avoids implausible prediction. Comprehensive experiments demonstrate the effectiveness of OSX. Lastly, we build a large-scale Upper-Body dataset (UBody) with high-quality 2D and 3D whole-body annotations. It contains persons with partially visible bodies in diverse real-life scenarios to bridge the gap between the basic task and downstream applications.

        ----

        ## [2015] Im2Hands: Learning Attentive Implicit Representation of Interacting Two-Hand Shapes

        **Authors**: *Jihyun Lee, Minhyuk Sung, Honggyu Choi, Tae-Kyun Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02028](https://doi.org/10.1109/CVPR52729.2023.02028)

        **Abstract**:

        We present Implicit Two Hands (Im2Hands), the first neural implicit representation of two interacting hands. Unlike existing methods on two-hand reconstruction that rely on a parametric hand model and/or low-resolution meshes, Im2Hands can produce fine-grained geometry of two hands with high hand-to-hand and hand-to-image coherency. To handle the shape complexity and interaction context between two hands, Im2Hands models the occupancy volume of two hands – conditioned on an RGB image and coarse 3D keypoints – by two novel attention-based modules responsible for (1) initial occupancy estimation and (2) context-aware occupancy refinement, respectively. Im2Hands first learns per-hand neural articulated occupancy in the canonical space designed for each hand using query-image attention. It then refines the initial two-hand occupancy in the posed space to enhance the coherency between the two hand shapes using query-anchor attention. In addition, we introduce an optional keypoint refinement module to enable robust two-hand shape estimation from predicted hand keypoints in a single-image reconstruction scenario. We experimentally demonstrate the effectiveness of Im2Hands on two-hand reconstruction in comparison to related methods, where ours achieves state-of-the-art results. Our code is publicly available at https://github.com/jyunlee/Im2Hands.

        ----

        ## [2016] FLEX: Full-Body Grasping Without Full-Body Grasps

        **Authors**: *Purva Tendulkar, Dídac Surís, Carl Vondrick*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02029](https://doi.org/10.1109/CVPR52729.2023.02029)

        **Abstract**:

        Synthesizing 3D human avatars interacting realistically with a scene is an important problem with applications in AR/VR, video games, and robotics. Towards this goal, we address the task of generating a virtual human – hands and full body – grasping everyday objects. Existing methods approach this problem by collecting a 3D dataset of humans interacting with objects and training on this data. However, 1) these methods do not generalize to different object positions and orientations or to the presence of furniture in the scene, and 2) the diversity of their generated full-body poses is very limited. In this work, we address all the above challenges to generate realistic, diverse full-body grasps in everyday scenes without requiring any 3D full-body grasping data. Our key insight is to leverage the existence of both full-body pose and hand-grasping priors, composing them using 3D geometrical constraints to obtain full-body grasps. We empirically validate that these constraints can generate a variety of feasible human grasps that are superior to baselines both quantitatively and qualitatively. See our webpage for more details: flex.cs.columbia.edu.

        ----

        ## [2017] DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects

        **Authors**: *Chen Bao, Helin Xu, Yuzhe Qin, Xiaolong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02030](https://doi.org/10.1109/CVPR52729.2023.02030)

        **Abstract**:

        To enable general-purpose robots, we will require the robot to operate daily articulated objects as humans do. Current robot manipulation has heavily relied on using a parallel gripper, which restricts the robot to a limited set of objects. On the other hand, operating with a multi-finger robot hand will allow better approximation to human behavior and enable the robot to operate on diverse articulated objects. To this end, we propose a new benchmark called DexArt, which involves Dexterous manipulation with Articulated objects in a physical simulator. In our benchmark, we define multiple complex manipulation tasks, and the robot hand will need to manipulate diverse articulated objects within each task. Our main focus is to evaluate the generalizability of the learned policy on unseen articulated objects. This is very challenging given the high degrees of freedom of both hands and objects. We use Reinforcement Learning with 3D representation learning to achieve generalization. Through extensive studies, we provide new insights into how 3D representation learning affects decision making in RL with 3D point cloud inputs. More details can be found at https://www.chenbao.tech/dexart/.

        ----

        ## [2018] CARTO: Category and Joint Agnostic Reconstruction of ARTiculated Objects

        **Authors**: *Nick Heppert, Muhammad Zubair Irshad, Sergey Zakharov, Katherine Liu, Rares Andrei Ambrus, Jeannette Bohg, Abhinav Valada, Thomas Kollar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02031](https://doi.org/10.1109/CVPR52729.2023.02031)

        **Abstract**:

        We present CARTO, a novel approach for reconstructing multiple articulated objects from a single stereo RGB observation. We use implicit object-centric representations and learn a single geometry and articulation decoder for multiple object categories. Despite training on multiple categories, our decoder achieves a comparable reconstruction accuracy to methods that train bespoke decoders separately for each category. Combined with our stereo image encoder we infer the 3D shape, 6D pose, size, joint type, and the joint state of multiple unknown objects in a single forward pass. Our method achieves a 20.4% absolute improvement in mAP 3D IOU50 for novel instances when compared to a two-stage pipeline. Inference time is fast and can run on a NVIDIA TITAN XP GPU at 1 HZ for eight or less objects present. While only trained on simulated data, CARTO transfers to real-world object instances. Code and evaluation data is available at: carto. cs. uni - freiburg. de

        ----

        ## [2019] CIRCLE: Capture In Rich Contextual Environments

        **Authors**: *João Pedro Araújo, Jiaman Li, Karthik Vetrivel, Rishi Agarwal, Jiajun Wu, Deepak Gopinath, Alexander Clegg, C. Karen Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02032](https://doi.org/10.1109/CVPR52729.2023.02032)

        **Abstract**:

        Synthesizing 3D human motion in a contextual, ecological environment is important for simulating realistic activities people perform in the real world. However, conventional optics-based motion capture systems are not suited for simultaneously capturing human movements and complex scenes. The lack of rich contextual 3D human motion datasets presents a roadblock to creating high-quality generative human motion models. We propose a novel motion acquisition system in which the actor perceives and operates in a highly contextual virtual world while being motion captured in the real world. Our system enables rapid collection of high-quality human motion in highly diverse scenes, without the concern of occlusion or the need for physical scene construction in the real world. We present CIRCLE, a dataset containing 10 hours of full-body reaching motion from 5 subjects across nine scenes, paired with ego-centric information of the environment represented in various forms, such as RGBD videos. We use this dataset to train a model that generates human motion conditioned on scene information. Leveraging our dataset, the model learns to use ego-centric scene information to achieve non-trivial reaching tasks in the context of complex 3D scenes. To download the data please visit our website.

        ----

        ## [2020] Decoupling Human and Camera Motion from Videos in the Wild

        **Authors**: *Vickie Ye, Georgios Pavlakos, Jitendra Malik, Angjoo Kanazawa*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02033](https://doi.org/10.1109/CVPR52729.2023.02033)

        **Abstract**:

        We propose a method to reconstruct global human trajectories from videos in the wild. Our optimization method decouples the camera and human motion, which allows us to place people in the same world coordinate frame. Most existing methods do not model the camera motion; methods that rely on the background pixels to infer 3D human motion usually require a full scene reconstruction, which is often not possible for in-the-wild videos. However, even when existing SLAM systems cannot recover accurate scene reconstructions, the background pixel motion still provides enough signal to constrain the camera motion. We show that relative camera estimates along with data-driven human motion priors can resolve the scene scale ambiguity and recover global human trajectories. Our method robustly recovers the global 3D trajectories of people in challenging in-the-wild videos, such as PoseTrack. We quantify our improvement over existing methods on 3D human dataset Egobody. We further demonstrate that our recovered camera scale allows us to reason about motion of multiple people in a shared coordinate frame, which improves performance of downstream tracking in PoseTrack.

        ----

        ## [2021] GarmentTracking: Category-Level Garment Pose Tracking

        **Authors**: *Han Xue, Wenqiang Xu, Jieyi Zhang, Tutian Tang, Yutong Li, Wenxin Du, Ruolin Ye, Cewu Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02034](https://doi.org/10.1109/CVPR52729.2023.02034)

        **Abstract**:

        Garments are important to humans. A visual system that can estimate and track the complete garment pose can be useful for many downstream tasks and real-world applications. In this work, we present a complete package to address the category-level garment pose tracking task: (1) A recording system VR-Garment, with which users can manipulate virtual garment models in simulation through a VR interface. (2) A large-scale dataset VR-Folding, with complex garment pose configurations in manipulation like flattening and folding. (3) An end-to-end online tracking framework GarmentTracking, which predicts complete garment pose both in canonical space and task space given a point cloud sequence. Extensive experiments demonstrate that the proposed GarmentTracking achieves great performance even when the garment has large non-rigid deformation. It outperforms the baseline approach on both speed and accuracy. We hope our proposed solution can serve as a platform for future research. Codes and datasets are available in https://garment-tracking.robotflow.ai.

        ----

        ## [2022] Hierarchical Temporal Transformer for 3D Hand Pose Estimation and Action Recognition from Egocentric RGB Videos

        **Authors**: *Yilin Wen, Hao Pan, Lei Yang, Jia Pan, Taku Komura, Wenping Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02035](https://doi.org/10.1109/CVPR52729.2023.02035)

        **Abstract**:

        Understanding dynamic hand motions and actions from egocentric RGB videos is a fundamental yet challenging task due to self-occlusion and ambiguity. To address occlusion and ambiguity, we develop a transformer-based framework to exploit temporal information for robust estimation. Noticing the different temporal granularity of and the semantic correlation between hand pose estimation and action recognition, we build a network hierarchy with two cascaded transformer encoders, where the first one exploits the short-term temporal cue for hand pose estimation, and the latter aggregates per-frame pose and object information over a longer time span to recognize the action. Our approach achieves competitive results on two first-person hand action benchmarks, namely FPHA and H2O. Extensive ablation studies verify our design choices.

        ----

        ## [2023] PSVT: End-to-End Multi-Person 3D Pose and Shape Estimation with Progressive Video Transformers

        **Authors**: *Zhongwei Qiu, Qiansheng Yang, Jian Wang, Haocheng Feng, Junyu Han, Errui Ding, Chang Xu, Dongmei Fu, Jingdong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02036](https://doi.org/10.1109/CVPR52729.2023.02036)

        **Abstract**:

        Existing methods of multi-person video 3D human Pose and Shape Estimation (PSE) typically adopt a two-stage strategy, which first detects human instances in each frame and then performs single-person PSE with temporal model. However, the global spatio-temporal context among spatial instances can not be captured. In this paper, we propose a new end-to-end multi-person 3D Pose and Shape estimation framework with progressive Video Transformer, termed PSVT. In PSVT, a spatio-temporal encoder (STE) captures the global feature dependencies among spatial objects. Then, spatio-temporal pose decoder (STPD) and shape decoder (STSD) capture the global dependencies between pose queries and feature tokens, shape queries and feature tokens, respectively. To handle the variances of objects as time proceeds, a novel scheme of progressive decoding is used to update pose and shape queries at each frame. Besides, we propose a novel pose-guided attention (PGA) for shape decoder to better predict shape parameters. The two components strengthen the decoder of PSVT to improve performance. Extensive experiments on the four datasets show that PSVT achieves stage-of-the-art results.

        ----

        ## [2024] Delving into Discrete Normalizing Flows on SO(3) Manifold for Probabilistic Rotation Modeling

        **Authors**: *Yulin Liu, Haoran Liu, Yingda Yin, Yang Wang, Baoquan Chen, He Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02037](https://doi.org/10.1109/CVPR52729.2023.02037)

        **Abstract**:

        Normalizing flows (NFs) provide a powerful tool to construct an expressive distribution by a sequence of trackable transformations of a base distribution and form a probabilistic model of underlying data. Rotation, as an important quantity in computer vision, graphics, and robotics, can exhibit many ambiguities when occlusion and symmetry occur and thus demands such probabilistic models. Though much progress has been made for NFs in Euclidean space, there are no effective normalizing flows without discontinuity or many-to-one mapping tailored for SO(3) manifold. Given the unique non-Euclidean properties of the rotation manifold, adapting the existing NFs to SO(3) manifold is non-trivial. In this paper, we propose a novel normalizing flow on SO (3) by combining a Mobius transformation-based coupling layer and a quaternion affine transformation. With our proposed rotation normalizing flows, one can not only effectively express arbitrary distributions on SO(3), but also conditionally build the target distribution given input observations. Extensive experiments show that our rotation normalizing flows significantly outperform the baselines on both unconditional and conditional tasks.

        ----

        ## [2025] 3D-POP - An Automated Annotation Approach to Facilitate Markerless 2D-3D Tracking of Freely Moving Birds with Marker-Based Motion Capture

        **Authors**: *Hemal Naik, Alex Hoi Hang Chan, Junran Yang, Mathilde Delacoux, Iain D. Couzin, Fumihiro Kano, Nagy Máté*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02038](https://doi.org/10.1109/CVPR52729.2023.02038)

        **Abstract**:

        Recent advances in machine learning and computer vision are revolutionizing the field of animal behavior by enabling researchers to track the poses and locations of freely moving animals without any marker attachment. However, large datasets of annotated images of animals for markerless pose tracking, especially high-resolution images taken from multiple angles with accurate 3D annotations, are still scant. Here, we propose a method that uses a motion capture (mo-cap) system to obtain a large amount of annotated data on animal movement and posture (2D and 3D) in a semi-automatic manner. Our method is novel in that it extracts the 3D positions of morphological keypoints (e.g eyes, beak, tail) in reference to the positions of markers attached to the animals. Using this method, we obtained, and offer here, a new dataset - 3D-POP with approximately 300k annotated frames (4 million instances) in the form of videos having groups of one to ten freely moving birds from 4 different camera views in a 3.6m x 4.2m area. 3D-POP is the first dataset of flocking birds with accurate keypoint annotations in 2D and 3D along with bounding box and individual identities and will facilitate the development of solutions for problems of 2D to 3D markerless pose, trajectory tracking, and identification in birds.

        ----

        ## [2026] TTA-COPE: Test-Time Adaptation for Category-Level Object Pose Estimation

        **Authors**: *Taeyeop Lee, Jonathan Tremblay, Valts Blukis, Bowen Wen, Byeong-Uk Lee, Inkyu Shin, Stan Birchfield, In So Kweon, Kuk-Jin Yoon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02039](https://doi.org/10.1109/CVPR52729.2023.02039)

        **Abstract**:

        Test-time adaptation methods have been gaining attention recently as a practical solution for addressing source-to-target domain gaps by gradually updating the model without requiring labels on the target data. In this paper, we propose a method of test-time adaptation for category-level object pose estimation called TTA-COPE. We design a pose ensemble approach with a self-training loss using pose-aware confidence. Unlike previous unsupervised domain adaptation methods for category-level object pose estimation, our approach processes the test data in a sequential, online manner, and it does not require access to the source domain at runtime. Extensive experimental results demonstrate that the proposed pose ensemble and the self-training loss improve category-level object pose performance during test time under both semi-supervised and unsupervised settings.

        ----

        ## [2027] Markerless Camera-to-Robot Pose Estimation via Self-Supervised Sim-to-Real Transfer

        **Authors**: *Jingpei Lu, Florian Richter, Michael C. Yip*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02040](https://doi.org/10.1109/CVPR52729.2023.02040)

        **Abstract**:

        Solving the camera-to-robot pose is a fundamental requirement for vision-based robot control, and is a process that takes considerable effort and cares to make accurate. Traditional approaches require modification of the robot via markers, and subsequent deep learning approaches enabled markerless feature extraction. Mainstream deep learning methods only use synthetic data and rely on Domain Randomization to fill the sim-to-real gap, because acquiring the 3D annotation is labor-intensive. In this work, we go beyond the limitation of 3D annotations for real-world data. We propose an end-to-end pose estimation framework that is capable of online camera-to-robot calibration and a self-supervised training method to scale the training to unlabeled real-world data. Our framework combines deep learning and geometric vision for solving the robot pose, and the pipeline is fully differentiable. To train the Camera-to-Robot Pose Estimation Network (CtRNet), we leverage foreground segmentation and differentiable rendering for image-level self-supervision. The pose prediction is visualized through a renderer and the image loss with the input image is back-propagated to train the neural network. Our experimental results on two public real datasets confirm the effectiveness of our approach over existing works. We also integrate our framework into a visual servoing system to demonstrate the promise of real-time precise robot pose estimation for automation tasks.

        ----

        ## [2028] SMOC-Net: Leveraging Camera Pose for Self-Supervised Monocular Object Pose Estimation

        **Authors**: *Tao Tan, Qiulei Dong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02041](https://doi.org/10.1109/CVPR52729.2023.02041)

        **Abstract**:

        Recently, self-supervised 6D object pose estimation, where synthetic images with object poses (sometimes jointly with un-annotated real images) are used for training, has attracted much attention in computer vision. Some typical works in literature employ a time-consuming differentiable renderer for object pose prediction at the training stage, so that (i) their performances on real images are generally limited due to the gap between their rendered images and real images and (ii) their training process is computationally expensive. To address the two problems, we propose a novel Network for Self-supervised Monocular Object pose estimation by utilizing the predicted Camera poses from unannotated real images, called SMOC-Net. The proposed network is explored under a knowledge distillation framework, consisting of a teacher model and a student model. The teacher model contains a backbone estimation module for initial object pose estimation, and an object pose refiner for refining the initial object poses using a geometric constraint (called relative-pose constraint) derived from relative camera poses. The student model gains knowledge for object pose estimation from the teacher model by imposing the relative-pose constraint. Thanks to the relative-pose constraint, SMOC-Net could not only narrow the domain gap between synthetic and real data but also reduce the training cost. Experimental results on two public datasets demonstrate that SMOC-Net outperforms several state-of-the-art methods by a large margin while requiring much less training time than the differentiable-renderer-based methods.

        ----

        ## [2029] IMP: Iterative Matching and Pose Estimation with Adaptive Pooling

        **Authors**: *Fei Xue, Ignas Budvytis, Roberto Cipolla*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02042](https://doi.org/10.1109/CVPR52729.2023.02042)

        **Abstract**:

        Previous methods solve feature matching and pose estimation using a two-stage process by first finding matches and then estimating the pose. As they ignore the geometric relationships between the two tasks, they focus on either improving the quality of matches or filtering potential outliers, leading to limited efficiency or accuracy. In contrast, we propose an iterative matching and pose estimation framework (IMP) leveraging the geometric connections between the two tasks: a few good matches are enough for a roughly accurate pose estimation; a roughly accurate pose can be used to guide the matching by providing geometric constraints. To this end, we implement a geometry-aware recurrent attention-based module which jointly outputs sparse matches and camera poses. Specifically, for each iteration, we first implicitly embed geometric information into the module via a pose-consistency loss, allowing it to predict geometry-aware matches progressively. Second, we introduce an efficient IMP, called EIMP, to dynamically discard keypoints without potential matches, avoiding redundant updating and significantly reducing the quadratic time complexity of attention computation in transformers. Experiments on YFCC100m, Scannet, and Aachen Day-Night datasets demonstrate that the proposed method outperforms previous approaches in terms of accuracy and efficiency. Code is available at https://github.com/feixue94/imp-release

        ----

        ## [2030] Self-Supervised Representation Learning for CAD

        **Authors**: *Benjamin T. Jones, Michael Hu, Milin Kodnongbua, Vladimir G. Kim, Adriana Schulz*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02043](https://doi.org/10.1109/CVPR52729.2023.02043)

        **Abstract**:

        Virtually every object in the modern world was created, modified, analyzed and optimized using computer aided design (CAD) tools. An active CAD research area is the use of data-driven machine learning methods to learn from the massive repositories of geometric and program representations. However, the lack of labeled data in CAD's native format, i.e., the parametric boundary representation (B-Rep), poses an obstacle at present difficult to overcome. Several datasets of mechanical parts in B-Rep format have recently been released for machine learning research. However, large-scale databases are mostly unlabeled, and labeled datasets are small. Additionally, task-specific label sets are rare and costly to annotate. This work proposes to leverage unlabeled CAD geometry on supervised learning tasks. We learn a novel, hybrid implicit/explicit surface representation for B-Rep geometry. Further, we show that this pre-training both significantly improves few-shot learning performance and achieves state-of-the-art performance on several current B-Rep benchmarks.

        ----

        ## [2031] Few-Shot Geometry-Aware Keypoint Localization

        **Authors**: *Xingzhe He, Gaurav Bharaj, David Ferman, Helge Rhodin, Pablo Garrido*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02044](https://doi.org/10.1109/CVPR52729.2023.02044)

        **Abstract**:

        Supervised keypoint localization methods rely on large manually labeled image datasets, where objects can deform, articulate, or occlude. However, creating such large keypoint labels is time-consuming and costly, and is often error-prone due to inconsistent labeling. Thus, we desire an approach that can learn keypoint localization with fewer yet consistently annotated images. To this end, we present a novel formulation that learns to localize semantically consistent keypoint definitions, even for occluded regions, for varying object categories. We use a few user-labeled 2D images as input examples, which are extended via self-supervision using a larger unlabeled dataset. Unlike unsupervised methods, the few-shot images act as semantic shape constraints for object localization. Furthermore, we introduce 3D geometry-aware constraints to uplift keypoints, achieving more accurate 2D localization. Our general-purpose formulation paves the way for semantically conditioned generative modeling and attains competitive or state-of-the-art accuracy on several datasets, including human faces, eyes, animals, cars, and never-before-seen mouth interior (teeth) localization tasks, not attempted by the previous few-shot methods. Project page: https://xingzhehe.github.io/FewShot3DKP/

        ----

        ## [2032] SparsePose: Sparse-View Camera Pose Regression and Refinement

        **Authors**: *Samarth Sinha, Jason Y. Zhang, Andrea Tagliasacchi, Igor Gilitschenski, David B. Lindell*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02045](https://doi.org/10.1109/CVPR52729.2023.02045)

        **Abstract**:

        Camera pose estimation is a key step in standard 3D reconstruction pipelines that operate on a dense set of images of a single object or scene. However, methods for pose estimation often fail when only a few images are available because they rely on the ability to robustly identify and match visual features between image pairs. While these methods can work robustly with dense camera views, capturing a large set of images can be time-consuming or impractical. We propose SparsePose for recovering accurate camera poses given a sparse set of wide-baseline images (fewer than 10). The method learns to regress initial camera poses and then iteratively refine them after training on a large-scale dataset of objects (Co3D: Common Objects in 3D). SparsePose significantly outperforms conventional and learning-based baselines in recovering accurate camera rotations and translations. We also demonstrate our pipeline for high-fidelity 3D reconstruction using only 5–9 images of an object. Project webpage: https://sparsepose.github.io/

        ----

        ## [2033] A Large-Scale Homography Benchmark

        **Authors**: *Daniel Barath, Dmytro Mishkin, Michal Polic, Wolfgang Förstner, Jiri Matas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02046](https://doi.org/10.1109/CVPR52729.2023.02046)

        **Abstract**:

        We present a large-scale dataset of Planes in 3D, Pi3D, of roughly 1000 planes observed in 10 000 images from the 1DSfM dataset, and HEB, a large-scale homography estimation benchmark leveraging Pi3D. The applications of the Pi3D dataset are diverse, e.g. training or evaluating monocular depth, surface normal estimation and image matching algorithms. The HEB dataset consists of 226 260 homographies and includes roughly 4M correspondences. The homographies link images that often undergo significant viewpoint and illumination changes. As applications of HEB, we perform a rigorous evaluation of a wide range of robust estimators and deep learning-based correspondence filtering methods, establishing the current state-of-the-art in robust homography estimation. We also evaluate the uncertainty of the SIFT orientations and scales w.r.t. the ground truth coming from the underlying homographies and provide codes for comparing uncertainty of custom detectors. The dataset is available at https://github.com/danini/homography-benchmark.

        ----

        ## [2034] Learning Geometric-Aware Properties in 2D Representation Using Lightweight CAD Models, or Zero Real 3D Pairs

        **Authors**: *Pattaramanee Arsomngern, Sarana Nutanong, Supasorn Suwajanakorn*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02047](https://doi.org/10.1109/CVPR52729.2023.02047)

        **Abstract**:

        Cross-modal training using 2D-3D paired datasets, such as those containing multi-view images and 3D scene scans, presents an effective way to enhance 2D scene understanding by introducing geometric and view-invariance priors into 2D features. However, the need for large-scale scene datasets can impede scalability and further improvements. This paper explores an alternative learning method by leveraging a lightweight and publicly available type of 3D data in the form of CAD models. We construct a 3D space with geometric-aware alignment where the similarity in this space reflects the geometric similarity of CAD models based on the Chamfer distance. The acquired geometric-aware properties are then induced into 2D features, which boost performance on downstream tasks more effectively than existing RGB-CAD approaches. Our technique is not limited to paired RGB-CAD datasets. By training exclu-sively on pseudo pairs generated from CAD-based reconstruction methods, we enhance the performance of SOTA 2D pretrained models that use ResNet-50 or ViT-B back-bones on various 2D understanding tasks. We also achieve comparable results to SOTA methods trained on scene scans on four tasks in NYUv2, SUNRGB-D, indoor ADE20k, and indoor/outdoor COCO, despite using lightweight CAD models or pseudo data. Please visit our page: https://GeoAware2dRepUsingCAD.github.io/

        ----

        ## [2035] AutoRecon: Automated 3D Object Discovery and Reconstruction

        **Authors**: *Yuang Wang, Xingyi He, Sida Peng, Haotong Lin, Hujun Bao, Xiaowei Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02048](https://doi.org/10.1109/CVPR52729.2023.02048)

        **Abstract**:

        A fully automated object reconstruction pipeline is crucial for digital content creation. While the area of 3D reconstruction has witnessed profound developments, the removal of background to obtain a clean object model still relies on different forms of manual labor, such as bounding box labeling, mask annotations, and mesh manipulations. In this paper, we propose a novel framework named AutoRecon for the automated discovery and reconstruction of an object from multi-view images. We demonstrate that foreground objects can be robustly located and segmented from SfM point clouds by leveraging self-supervised 2D vision transformer features. Then, we reconstruct decomposed neural scene representations with dense supervision provided by the decomposed point clouds, resulting in accurate object reconstruction and segmentation. Experiments on the DTU, BlendedMVS and CO3D-V2 datasets demonstrate the effectiveness and robustness of AutoRecon. The code and supplementary material are available on the project page: https://zju3dv.github.io/autorecon/.

        ----

        ## [2036] Multi-Sensor Large-Scale Dataset for Multi-View 3D Reconstruction

        **Authors**: *Oleg Voynov, Gleb Bobrovskikh, Pavel A. Karpyshev, Saveliy Galochkin, Andrei-Timotei Ardelean, Arseniy Bozhenko, Ekaterina Karmanova, Pavel Kopanev, Yaroslav Labutin-Rymsho, Ruslan Rakhimov, Aleksandr Safin, Valerii Serpiva, Alexey Artemov, Evgeny Burnaev, Dzmitry Tsetserukou, Denis Zorin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02049](https://doi.org/10.1109/CVPR52729.2023.02049)

        **Abstract**:

        We present a new multi-sensor dataset for multi-view 3D surface reconstruction. It includes registered RGB and depth data from sensors of different resolutions and modalities: smartphones, Intel RealSense, Microsoft Kinect, industrial cameras, and structured-light scanner. The scenes are selected to emphasize a diverse set of material properties challenging for existing algorithms. We provide around 1.4 million images of 107 different scenes acquired from 100 viewing directions under 14 lighting conditions. We expect our dataset will be useful for evaluation and training of 3D reconstruction algorithms and for related tasks. The dataset is available at skol tech3d. appliedai. tech.

        ----

        ## [2037] NeurOCS: Neural NOCS Supervision for Monocular 3D Object Localization

        **Authors**: *Zhixiang Min, Bingbing Zhuang, Samuel Schulter, Buyu Liu, Enrique Dunn, Manmohan Chandraker*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02050](https://doi.org/10.1109/CVPR52729.2023.02050)

        **Abstract**:

        Monocular 3D object localization in driving scenes is a crucial task, but challenging due to its ill-posed nature. Estimating 3D coordinates for each pixel on the object surface holds great potential as it provides dense 2D-3D geometric constraints for the underlying PnP problem. However, high-quality ground truth supervision is not available in driving scenes due to sparsity and various artifacts of Lidar data, as well as the practical infeasibility of collecting per-instance CAD models. In this work, we present NeurOCS, a framework that uses instance masks and 3D boxes as input to learn 3D object shapes by means of differentiable rendering, which further serves as supervision for learning dense object coordinates. Our approach rests on insights in learning a category-level shape prior directly from real driving scenes, while properly handling single-view ambiguities. Furthermore, we study and make critical design choices to learn object coordinates more effectively from an object-centric view. Altogether, our framework leads to new state-of-the-art in monocular 3D localization that ranks 1st on the KITTI-Object [16] benchmark among published monocular methods.

        ----

        ## [2038] Self-Supervised Super-Plane for Neural 3D Reconstruction

        **Authors**: *Botao Ye, Sifei Liu, Xueting Li, Ming-Hsuan Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02051](https://doi.org/10.1109/CVPR52729.2023.02051)

        **Abstract**:

        Neural implicit surface representation methods show impressive reconstruction results but struggle to handle texture-less planar regions that widely exist in indoor scenes. Existing approaches addressing this leverage image prior that requires assistive networks trained with large-scale annotated datasets. In this work, we introduce a self-supervised super-plane constraint by exploring the free geometry cues from the predicted surface, which can further regularize the reconstruction of plane regions without any other ground truth annotations. Specifically, we introduce an iterative training scheme, where (i) grouping of pixels to formulate a super-plane (analogous to super-pixels), and (ii) optimizing of the scene reconstruction network via a super-plane constraint, are progressively conducted. We demonstrate that the model trained with superplanes surprisingly outperforms the one using conventional annotated planes, as individual super-plane statistically occupies a larger area and leads to more stable training. Extensive experiments show that our self-supervised super-plane constraint significantly improves 3D reconstruction quality even better than using ground truth plane segmentation. Additionally, the plane reconstruction results from our model can be used for auto-labeling for other vision tasks. The code and models are available at https://github.com/botaoye/S3PRecon.

        ----

        ## [2039] PlaneDepth: Self-Supervised Depth Estimation via Orthogonal Planes

        **Authors**: *Ruoyu Wang, Zehao Yu, Shenghua Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02052](https://doi.org/10.1109/CVPR52729.2023.02052)

        **Abstract**:

        Multiple near frontal-parallel planes based depth representation demonstrated impressive results in self-supervised monocular depth estimation (MDE). Whereas, such a representation would cause the discontinuity of the ground as it is perpendicular to the frontal-parallel planes, which is detrimental to the identification of drivable space in autonomous driving. In this paper, we propose the PlaneDepth, a novel orthogonal planes based presentation, including vertical planes and ground planes. PlaneDepth estimates the depth distribution using a Laplacian Mixture Model based on orthogonal planes for an input image. These planes are used to synthesize a reference view to provide the self-supervision signal. Further, we find that the widely used resizing and cropping data augmentation breaks the orthogonality assumptions, leading to inferior plane predictions. We address this problem by explicitly constructing the resizing cropping transformation to rectify the predefined planes and predicted camera pose. Moreover, we propose an augmented self-distillation loss supervised with a bilateral occlusion mask to boost the robustness of orthogonal planes representation for occlusions. Thanks to our orthogonal planes representation, we can extract the ground plane in an unsupervised manner, which is important for autonomous driving. Extensive experiments on the KITTI dataset demonstrate the effectiveness and efficiency of our method. The code is available at https://github.com/svip-lab/PlaneDepth.

        ----

        ## [2040] Single View Scene Scale Estimation using Scale Field

        **Authors**: *Byeong-Uk Lee, Jianming Zhang, Yannick Hold-Geoffroy, In So Kweon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02053](https://doi.org/10.1109/CVPR52729.2023.02053)

        **Abstract**:

        In this paper, we propose a single image scale estimation method based on a novel scale field representation. A scale field defines the local pixel-to-metric conversion ratio along the gravity direction on all the ground pixels. This representation resolves the ambiguity in camera parameters, allowing us to use a simple yet effective way to collect scale annotations on arbitrary images from human annotators. By training our model on calibrated panoramic image data and the in-the-wild human annotated data, our single image scene scale estimation network generates robust scale field on a variety of image, which can be utilized in various 3D understanding and scale-aware image editing applications.

        ----

        ## [2041] 3D Line Mapping Revisited

        **Authors**: *Shaohui Liu, Yifan Yu, Rémi Pautrat, Marc Pollefeys, Viktor Larsson*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02054](https://doi.org/10.1109/CVPR52729.2023.02054)

        **Abstract**:

        In contrast to sparse keypoints, a handful of line segments can concisely encode the high-level scene layout, as they often delineate the main structural elements. In addition to offering strong geometric cues, they are also omnipresent in urban landscapes and indoor scenes. Despite their apparent advantages, current line-based reconstruction methods are far behind their point-based counterparts. In this paper we aim to close the gap by introducing LIMAP, a library for 3D line mapping that robustly and efficiently creates 3D line maps from multi-view imagery. This is achieved through revisiting the degeneracy problem of line triangulation, carefully crafted scoring and track building, and exploiting structural priors such as line coincidence, parallelism, and orthogonality. Our code integrates seamlessly with existing point-based Structure-from-Motion methods and can leverage their 3D points to further improve the line reconstruction. Furthermore, as a byproduct, the method is able to recover 3D association graphs between lines and points / vanishing points (VPs). In thorough experiments, we show that LIMAP significantly outperforms existing approaches for 3D line mapping. Our robust 3D line maps also open up new research directions. We show two example applications: visual localization and bundle adjustment, where integrating lines alongside points yields the best results. Code is available at https://github.com/cvg/limap.

        ----

        ## [2042] Inverting the Imaging Process by Learning an Implicit Camera Model

        **Authors**: *Xin Huang, Qi Zhang, Ying Feng, Hongdong Li, Qing Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02055](https://doi.org/10.1109/CVPR52729.2023.02055)

        **Abstract**:

        Representing visual signals with implicit coordinate-based neural networks, as an effective replacement of the traditional discrete signal representation, has gained considerable popularity in computer vision and graphics. In contrast to existing implicit neural representations which focus on modelling the scene only, this paper proposes a novel implicit camera model which represents the physical imaging process of a camera as a deep neural network. We demonstrate the power of this new implicit camera model on two inverse imaging tasks: i) generating all-in-focus photos, and ii) HDR imaging. Specifically, we devise an implicit blur generator and an implicit tone mapper to model the aperture and exposure of the camera's imaging process, respectively. Our implicit camera model is jointly learned together with implicit scene models under multi-focus stack and multi-exposure bracket supervision. We have demonstrated the effectiveness of our new model on a large number of test images and videos, producing accurate and visually appealing all-in-focus and high dynamic range images. In principle, our new implicit neural camera model has the potential to benefit a wide array of other inverse imaging tasks.

        ----

        ## [2043] SfM-TTR: Using Structure from Motion for Test-Time Refinement of Single-View Depth Networks

        **Authors**: *Sergio Izquierdo, Javier Civera*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02056](https://doi.org/10.1109/CVPR52729.2023.02056)

        **Abstract**:

        Estimating a dense depth map from a single view is geometrically ill-posed, and state-of-the-art methods rely on learning depth's relation with visual appearance using deep neural networks. On the other hand, Structure from Motion (SfM) leverages multi-view constraints to produce very accurate but sparse maps, as matching across images is typically limited by locally discriminative texture. In this work, we combine the strengths of both approaches by proposing a novel test-time refinement (TTR) method, denoted as SfM-TTR, that boosts the performance of single-view depth networks at test time using SfM multi-view cues. Specifically, and differently from the state of the art, we use sparse SfM point clouds as test-time self-supervisory signal, fine-tuning the network encoder to learn a better representation of the test scene. Our results show how the addition of SfM-TTR to several state-of-the-art self-supervised and supervised networks improves significantly their performance, outperforming previous TTR baselines mainly based on photometric multi-view consistency. The code is available at https://github.com/serizba/SfM-TTR.

        ----

        ## [2044] iDisc: Internal Discretization for Monocular Depth Estimation

        **Authors**: *Luigi Piccinelli, Christos Sakaridis, Fisher Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02057](https://doi.org/10.1109/CVPR52729.2023.02057)

        **Abstract**:

        Monocular depth estimation is fundamental for 3D scene understanding and downstream applications. However, even under the supervised setup, it is still challenging and ill-posed due to the lack of full geometric constraints. Although a scene can consist of millions of pixels, there are fewer high-level patterns. We propose iDisc to learn those patterns with internal discretized representations. The method implicitly partitions the scene into a set of high-level patterns. In particular, our new module, Internal Discretization (ID), implements a continuous-discrete-continuous bottleneck to learn those concepts without supervision. In contrast to state-of-the-art methods, the proposed model does not enforce any explicit constraints or priors on the depth output. The whole network with the ID module can be trained end-to-end, thanks to the bottleneck module based on attention. Our method sets the new state of the art with significant improvements on NYU-Depth v2 and KITTI, outperforming all published methods on the official KITTI benchmark. iDisc can also achieve state-of-the-art results on surface normal estimation. Further, we explore the model generalization capability via zero-shot testing. We observe the compelling need to promote diversification in the outdoor scenario. Hence, we introduce splits of two autonomous driving datasets, DDAD and Argoverse. Code is available at http://vis.xyz/pub/idisc.

        ----

        ## [2045] DC2: Dual-Camera Defocus Control by Learning to Refocus

        **Authors**: *Hadi Alzayer, Abdullah Abuolaim, Leung Chun Chan, Yang Yang, Ying Chen Lou, Jia-Bin Huang, Abhishek Kar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02058](https://doi.org/10.1109/CVPR52729.2023.02058)

        **Abstract**:

        Smartphone cameras today are increasingly approaching the versatility and quality of professional cameras through a combination of hardware and software advancements. However, fixed aperture remains a key limitation, preventing users from controlling the depth of field (DoF) of captured images. At the same time, many smartphones now have multiple cameras with different fixed apertures - specifically, an ultra-wide camera with wider field of view and deeper DoF and a higher resolution primary camera with shallower DoF. In this work, we propose 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$DC^{2}$</tex>
, a system for defocus control for synthetically varying camera aperture, focus distance and arbitrary defocus effects by fusing information from such a dual-camera system. Our key insight is to leverage real-world smartphone camera dataset by using image refocus as a proxy task for learning to control defocus. Quantitative and qualitative evaluations on real-world data demonstrate our system's efficacy where we outperform state-of-the-art on defocus deblurring, bokeh rendering, and image refocus. Finally, we demonstrate creative post-capture defocus control enabled by our method, including tilt-shift and content-based defocus effects.

        ----

        ## [2046] A Practical Stereo Depth System for Smart Glasses

        **Authors**: *Jialiang Wang, Daniel Scharstein, Akash Bapat, Kevin Blackburn-Matzen, Matthew Yu, Jonathan Lehman, Suhib Alsisan, Yanghan Wang, Sam S. Tsai, Jan-Michael Frahm, Zijian He, Peter Vajda, Michael F. Cohen, Matt Uyttendaele*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02059](https://doi.org/10.1109/CVPR52729.2023.02059)

        **Abstract**:

        We present the design of a productionized end-to-end stereo depth sensing system that does pre-processing, online stereo rectification, and stereo depth estimation with a fallback to monocular depth estimation when rectification is unreliable. The output of our depth sensing system is then used in a novel view generation pipeline to create 3D computational photography effects using point-of-view images captured by smart glasses. All these steps are executed on-device on the stringent compute budget of a mobile phone, and because we expect the users can use a wide range of smartphones, our design needs to be general and cannot be dependent on a particular hardware or ML accelerator such as a smartphone GPU. Although each of these steps is well studied, a description of a practical system is still lacking. For such a system, all these steps need to work in tandem with one another and fallback gracefully on failures within the system or less than ideal input data. We show how we handle unforeseen changes to calibration, e.g., due to heat, robustly support depth estimation in the wild, and still abide by the memory and latency constraints required for a smooth user experience. We show that our trained models are fast, and run in less than 1s on a six-year-old Samsung Galaxy S8 phone's CPU. Our models generalize well to unseen data and achieve good results on Middlebury and in-the-wild images captured from the smart glasses.

        ----

        ## [2047] GeoMVSNet: Learning Multi-View Stereo with Geometry Perception

        **Authors**: *Zhe Zhang, Rui Peng, Yuxi Hu, Ronggang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02060](https://doi.org/10.1109/CVPR52729.2023.02060)

        **Abstract**:

        Recent cascade Multi- View Stereo (MVS) methods can efficiently estimate high-resolution depth maps through narrowing hypothesis ranges. However, previous methods ignored the vital geometric information embedded in coarse stages, leading to vulnerable cost matching and sub-optimal reconstruction results. In this paper, we propose a geometry awareness model, termed GeoMVSNet, to explic-itly integrate geometric clues implied in coarse stages for delicate depth estimation. In particular, we design a two-branch geometry fusion network to extract geometric pri-ors from coarse estimations to enhance structural feature extraction at finer stages. Besides, we embed the coarse probability volumes, which encode valuable depth distri-bution attributes, into the lightweight regularization network to further strengthen depth-wise geometry intuition. Meanwhile, we apply the frequency domain filtering to mit-igate the negative impact of the high-frequency regions and adopt the curriculum learning strategy to progressively boost the geometry integration of the model. To intensify the full-scene geometry perception of our model, we present the depth distribution similarity loss based on the Gaussian-Mixture Model assumption. Extensive experiments on DTU and Tanks and Temples (T&T) datasets demonstrate that our GeoMVSNet achieves state-of-the-art results and ranks first on the T&T-Advanced set. Code is available at https://github.com/doubleZ0108IGeoMVSNet.

        ----

        ## [2048] DINN360: Deformable Invertible Neural Network for Latitude-aware 360° Image Rescaling

        **Authors**: *Yichen Guo, Mai Xu, Lai Jiang, Leonid Sigal, Yunjin Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02061](https://doi.org/10.1109/CVPR52729.2023.02061)

        **Abstract**:

        With the rapid development of virtual reality, 360° images have gained increasing popularity. Their wide field of view necessitates high resolution to ensure image quality. This, however, makes it harder to acquire, store and even process such 360° images. To alleviate this issue, we propose the first attempt at 360° image rescaling, which refers to downscaling a 360° image to a visually valid lowresolution (LR) counterpart and then upscaling to a highresolution (HR) 360° image given the LR variant. Specifically, we first analyze two 360° image datasets and observe several findings that characterize how 360° images typically change along their latitudes. Inspired by these findings, we propose a novel deformable invertible neural network (INN), named DINN360, for latitude-aware 360° image rescaling. In DINN360, a deformable INN is designed to downscale the LR image, and project the high-frequency (HF) component to the latent space by adaptively handling various deformations occurring at different latitude regions. Given the downscaled LR image, the high-quality HR image is then reconstructed in a conditional latitude-aware manner by recovering the structure-related HF component from the latent space. Extensive experiments over four public datasets show that our DINN360 method performs considerably better than other state-of-the-art methods for 2 x, 4 x and 8 x 360° image rescaling.

        ----

        ## [2049] OmniVidar: Omnidirectional Depth Estimation from Multi-Fisheye Images

        **Authors**: *Sheng Xie, Daochuan Wang, Yunhui Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02062](https://doi.org/10.1109/CVPR52729.2023.02062)

        **Abstract**:

        Estimating depth from four large field of view (FoV) cameras has been a difficult and understudied problem. In this paper, we proposed a novel and simple system that can convert this difficult problem into easier binocular depth estimation. We name this system OmniVidar, as its results are similar to LiDAR, but rely only on vision. OmniVidar contains three components: (1) a new camera model to address the shortcomings of existing models, (2) a new multi-fisheye camera based epipolar rectification method for solving the image distortion and simplifying the depth estimation problem, (3) an improved binocular depth estimation network, which achieves a better balance between accuracy and efficiency. Unlike other omnidirectional stereo vision methods, OmniVidar does not contain any 3D convolution, so it can achieve higher resolution depth estimation at fast speed. Results demonstrate that OmniVidar outperforms all other methods in terms of accuracy and performance.

        ----

        ## [2050] Learning to Fuse Monocular and Multi-view Cues for Multi-frame Depth Estimation in Dynamic Scenes

        **Authors**: *Rui Li, Dong Gong, Wei Yin, Hao Chen, Yu Zhu, Kaixuan Wang, Xiaozhi Chen, Jinqiu Sun, Yanning Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02063](https://doi.org/10.1109/CVPR52729.2023.02063)

        **Abstract**:

        Multi-frame depth estimation generally achieves high accuracy relying on the multi-view geometric consistency. When applied in dynamic scenes, e.g., autonomous driving, this consistency is usually violated in the dynamic areas, leading to corrupted estimations. Many multi-frame methods handle dynamic areas by identifying them with explicit masks and compensating the multi-view cues with monocular cues represented as local monocular depth or features. The improvements are limited due to the uncontrolled quality of the masks and the underutilized benefits of the fusion of the two types of cues. In this paper, we propose a novel method to learn to fuse the multi-view and monocular cues encoded as volumes without needing the heuristically crafted masks. As unveiled in our analyses, the multiview cues capture more accurate geometric information in static areas, and the monocular cues capture more useful contexts in dynamic areas. To let the geometric perception learned from multi-view cues in static areas propagate to the monocular representation in dynamic areas and let monocular cues enhance the representation of multi-view cost volume, we propose a cross-cue fusion (CCF) module, which includes the cross-cue attention (CCA) to encode the spatially non-local relative intra-relations from each source to enhance the representation of the other. Experiments on real-world datasets prove the significant effectiveness and generalization ability of the proposed method.

        ----

        ## [2051] Modality-invariant Visual Odometry for Embodied Vision

        **Authors**: *Marius Memmel, Roman Bachmann, Amir Zamir*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02064](https://doi.org/10.1109/CVPR52729.2023.02064)

        **Abstract**:

        Effectively localizing an agent in a realistic, noisy setting is crucial for many embodied vision tasks. Visual Odometry (VO) is a practical substitute for unreliable GPS and compass sensors, especially in indoor environments. While SLAM-based methods show a solid performance without large data requirements, they are less flexible and robust w.r.t. to noise and changes in the sensor suite compared to learning-based approaches. Recent deep VO models, however, limit themselves to a fixed set of input modalities, e.g., RGB and depth, while training on millions of samples. When sensors fail, sensor suites change, or modalities are intentionally looped out due to available resources, e.g., power consumption, the models fail catastrophically. Furthermore, training these models from scratch is even more expensive without simulator access or suitable existing models that can be fine-tuned. While such scenarios get mostly ignored in simulation, they commonly hinder a model's reusability in real-world applications. We propose a Transformer-based modality-invariant VO approach that can deal with diverse or changing sensor suites of navigation agents. Our model outperforms previous methods while training on only a fraction of the data. We hope this method opens the door to a broader range of real-world applications that can benefit from flexible and learned VO models.

        ----

        ## [2052] VL-SAT: Visual-Linguistic Semantics Assisted Training for 3D Semantic Scene Graph Prediction in Point Cloud

        **Authors**: *Ziqin Wang, Bowen Cheng, Lichen Zhao, Dong Xu, Yang Tang, Lu Sheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02065](https://doi.org/10.1109/CVPR52729.2023.02065)

        **Abstract**:

        The task of 3D semantic scene graph (3D SSG) prediction in the point cloud is challenging since (1) the 3D point cloud only captures geometric structures with limited semantics compared to 2D images, and (2) long-tailed relation distribution inherently hinders the learning of unbiased prediction. Since 2D images provide rich semantics and scene graphs are in nature coped with languages, in this study, we propose Visual-Linguistic Semantics Assisted Training (VL-SAT) scheme that can significantly empower 3DSSG prediction models with discrimination about long-tailed and ambiguous semantic relations. The key idea is to train a powerful multi-modal oracle model to assist the 3D model. This oracle learns reliable structural representations based on semantics from vision, language, and 3D geometry, and its benefits can be heterogeneously passed to the 3D model during the training stage. By effectively utilizing visual-linguistic semantics in training, our VL-SAT can significantly boost common 3DSSG prediction models, such as SGFN and SGGpoint, only with 3D inputs in the inference stage, especially when dealing with tail relation triplets. Comprehensive evaluations and ablation studies on the 3DSSG dataset have validated the effectiveness of the proposed scheme. Code is available at https://github.com/wz7in/CVPR2023-VLSAT.

        ----

        ## [2053] CAPE: Camera View Position Embedding for Multi-View 3D Object Detection

        **Authors**: *Kaixin Xiong, Shi Gong, Xiaoqing Ye, Xiao Tan, Ji Wan, Errui Ding, Jingdong Wang, Xiang Bai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02066](https://doi.org/10.1109/CVPR52729.2023.02066)

        **Abstract**:

        In this paper, we address the problem of detecting 3D ob-jects from multi-view images. Current query-based methods rely on global 3D position embeddings (PE) to learn the ge-ometric correspondence between images and 3D space. We claim that directly interacting 2D image features with global 3D PE could increase the difficulty of learning view trans-formation due to the variation of camera extrinsics. Thus we propose a novel method based on CAmera view Position Embedding, called CAPE. We form the 3D position embed-dings under the local camera-view coordinate system instead of the global coordinate system, such that 3D position em-bedding is free of encoding camera extrinsic parameters. Furthermore, we extend our CAPE to temporal modeling by exploiting the object queries of previous frames and encoding the ego motion for boosting 3D object detection. CAPE achieves the state-of-the-art performance (61.0% NDS and 52.5% mAP) among all LiDAR-free methods on nuScenes dataset. Codes and models are available. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Codes of Paddle3D and PyTorch Implementation.

        ----

        ## [2054] AeDet: Azimuth-Invariant Multi-View 3D Object Detection

        **Authors**: *Chengjian Feng, Zequn Jie, Yujie Zhong, Xiangxiang Chu, Lin Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02067](https://doi.org/10.1109/CVPR52729.2023.02067)

        **Abstract**:

        Recent LSS-based multi-view 3D object detection has made tremendous progress, by processing the features in Brid-Eye-View (BEV) via the convolutional detector. However, the typical convolution ignores the radial symmetry of the BEV features and increases the difficulty of the detector optimization. To preserve the inherent property of the BEV features and ease the optimization, we propose an azimuth-equivariant convolution (AeConv) and an azimuth-equivariant anchor. The sampling grid of AeConv is always in the radial direction, thus it can learn azimuth-invariant BEV features. The proposed anchor enables the detection head to learn predicting azimuth-irrelevant targets. In addition, we introduce a camera-decoupled virtual depth to unify the depth prediction for the images with different camera intrinsic parameters. The resultant detector is dubbed Azimith-equivariant Detector (AeDet). Extensive experiments are conducted on nuScenes, and AeDet achieves a 62.0% NDS, surpassing the recent multi-view 3D object detectors such as PETRv2 and BEVDepth by a large margin. Project page: https://fcjian.github.io/aedet.

        ----

        ## [2055] Object Detection with Self-Supervised Scene Adaptation

        **Authors**: *Zekun Zhang, Minh Hoai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02068](https://doi.org/10.1109/CVPR52729.2023.02068)

        **Abstract**:

        This paper proposes a novel method to improve the performance of a trained object detector on scenes with fixed camera perspectives based on self-supervised adaptation. Given a specific scene, the trained detector is adapted using pseudo-ground truth labels generated by the detector itself and an object tracker in a cross-teaching manner. When the camera perspective is fixed, our method can utilize the background equivariance by proposing artifact-free object mixup as a means of data augmentation, and utilize accurate background extraction as an additional input modality. We also introduce a large-scale and diverse dataset for the development and evaluation of scene-adaptive object detection. Experiments on this dataset show that our method can improve the average precision of the original detector, outperforming the previous state-of-the-art selfsupervised domain adaptive object detection methods by a large margin. Our dataset and code are published at https://github.com/cvlab-stonybrook/scenes100.

        ----

        ## [2056] Understanding the Robustness of 3D Object Detection with Bird'View Representations in Autonomous Driving

        **Authors**: *Zijian Zhu, Yichi Zhang, Hai Chen, Yinpeng Dong, Shu Zhao, Wenbo Ding, Jiachen Zhong, Shibao Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02069](https://doi.org/10.1109/CVPR52729.2023.02069)

        **Abstract**:

        3D object detection is an essential perception task in autonomous driving to understand the environments. The Bird's-Eye-View (BEV) representations have significantly improved the performance of 3D detectors with camera inputs on popular benchmarks. However, there still lacks a systematic understanding of the robustness of these vision-dependent BEV models, which is closely related to the safety of autonomous driving systems. In this paper, we evaluate the natural and adversarial robustness of various representative models under extensive settings, to fully understand their behaviors influenced by explicit BEV features compared with those without BEV. In addition to the classic settings, we propose a 3D consistent patch attack by applying adversarial patches in the 3D space to guarantee the spatiotemporal consistency, which is more realistic for the scenario of autonomous driving. With substantial experiments, we draw several findings: 1) BEV models tend to be more stable than previous methods under different natural conditions and common corruptions due to the expressive spatial representations; 2) BEV models are more vulnerable to adversarial noises, mainly caused by the redundant BEV features; 3) Camera-LiDARfusion models have superior performance under different settings with multi-modal inputs, but BEV fusion model is still vulnerable to adversarial noises of both point cloud and image. These findings alert the safety issue in the applications of BEV detectors and could facilitate the development of more robust models.

        ----

        ## [2057] BEV-LaneDet: An Efficient 3D Lane Detection Based on Virtual Camera via Key-Points

        **Authors**: *Ruihao Wang, Jian Qin, Kaiying Li, Yaochen Li, Dong Cao, Jintao Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00103](https://doi.org/10.1109/CVPR52729.2023.00103)

        **Abstract**:

        3D lane detection which plays a crucial role in vehicle routing, has recently been a rapidly developing topic in autonomous driving. Previous works struggle with practicality due to their complicated spatial transformations and inflexible representations of 3D lanes. Faced with the issues, our work proposes an efficient and robust monocular 3D lane detection called BEV-LaneDet with three main contributions. First, we introduce the Virtual Camera that unifies the in/extrinsic parameters of cameras mounted on different vehicles to guarantee the consistency of the spatial relationship among cameras. It can effectively promote the learning procedure due to the unified visual space. We secondly propose a simple but efficient 3D lane representation called Key-Points Representation. This module is more suitable to represent the complicated and diverse 3D lane structures. At last, we present a light-weight and chip-friendly spatial transformation module named Spatial Transformation Pyramid to transform multiscale front-view features into BEV features. Experimental results demonstrate that our work outperforms the state-of-the-art approaches in terms of F-Score, being 10.6% higher on the OpenLane dataset and 4.0% higher on the Apollo 3D synthetic dataset, with a speed of 185 FPS. Code is released at https://github.com/gigo-team/bev_lane_det.

        ----

        ## [2058] BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection

        **Authors**: *Lei Yang, Kaicheng Yu, Tao Tang, Jun Li, Kun Yuan, Li Wang, Xinyu Zhang, Peng Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02070](https://doi.org/10.1109/CVPR52729.2023.02070)

        **Abstract**:

        While most recent autonomous driving system focuses on developing perception methods on ego-vehicle sensors, people tend to overlook an alternative approach to leverage intelligent roadside cameras to extend the perception ability beyond the visual range. We discover that the state-of-the-art vision-centric bird's eye view detection methods have inferior performances on roadside cameras. This is because these methods mainly focus on recovering the depth regarding the camera center, where the depth difference between the car and the ground quickly shrinks while the distance increases. In this paper, we propose a simple yet effective approach, dubbed BEVHeight, to address this issue. In essence, instead of predicting the pixel-wise depth, we regress the height to the ground to achieve a distance-agnostic formulation to ease the optimization process of camera-only perception methods. On popular 3D detection benchmarks of roadside cameras, our method surpasses all previous vision-centric methods by a significant margin. The code is available at https://github.com/ADLab-AutoDrive/BEVHeight.

        ----

        ## [2059] Uncertainty-Aware Vision-Based Metric Cross-View Geolocalization

        **Authors**: *Florian Fervers, Sebastian Bullinger, Christoph Bodensteiner, Michael Arens, Rainer Stiefelhagen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02071](https://doi.org/10.1109/CVPR52729.2023.02071)

        **Abstract**:

        This paper proposes a novel method for vision-based metric cross-view geolocalization (CVGL) that matches the camera images captured from a ground-based vehicle with an aerial image to determine the vehicle's geo-pose. Since aerial images are globally available at low cost, they represent a potential compromise between two established paradigms of autonomous driving, i.e. using expensive high-definition prior maps or relying entirely on the sensor data captured at runtime. We present an end-to-end differentiable model that uses the ground and aerial images to predict a probability distribution over possible vehicle poses. We combine multiple vehicle datasets with aerial images from orthophoto providers on which we demonstrate the feasibility of our method. Since the ground truth poses are often inaccurate w.r.t. the aerial images, we implement a pseudo-label approach to produce more accurate ground truth poses and make them publicly available. While previous works require training data from the target region to achieve reasonable localization accuracy (i.e. same-area evaluation), our approach overcomes this limitation and outperforms previous results even in the strictly more challenging cross-area case. We improve the previous state-of-the-art by a large margin even without ground or aerial data from the test region, which highlights the model's potential for global-scale application. We further integrate the uncertainty-aware predictions in a tracking framework to determine the vehicle's trajectory over time resulting in a mean position error on KITTI-360 of 0.78m.

        ----

        ## [2060] OrienterNet: Visual Localization in 2D Public Maps with Neural Matching

        **Authors**: *Paul-Edouard Sarlin, Daniel DeTone, Tsun-Yi Yang, Armen Avetisyan, Julian Straub, Tomasz Malisiewicz, Samuel Rota Bulò, Richard A. Newcombe, Peter Kontschieder, Vasileios Balntas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02072](https://doi.org/10.1109/CVPR52729.2023.02072)

        **Abstract**:

        Humans can orient themselves in their 3D environments using simple 2D maps. Differently, algorithms for visual localization mostly rely on complex 3D point clouds that are expensive to build, store, and maintain over time. We bridge this gap by introducing OrienterNet, the first deep neural network that can localize an image with sub-meter accuracy using the same 2D semantic maps that humans use. OrienterNet estimates the location and orientation of a query image by matching a neural Bird's-Eye View with open and globally available maps from OpenStreetMap, enabling anyone to localize anywhere such maps are available. OrienterNet is supervised only by camera poses but learns to perform semantic matching with a wide range of map elements in an end-to-end manner. To enable this, we introduce a large crowd-sourced dataset of images captured across 12 cities from the diverse viewpoints of cars, bikes, and pedestrians. OrienterNet generalizes to new datasets and pushes the state of the art in both robotics and AR scenarios. The code is available at github.com/facebookresearch/OrienterNet.

        ----

        ## [2061] MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with Multi-Depth Seeds for 3D Object Detection

        **Authors**: *Yang Jiao, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, Yu-Gang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02073](https://doi.org/10.1109/CVPR52729.2023.02073)

        **Abstract**:

        Fusing LiDAR and camera information is essential for accurate and reliable 3D object detection in autonomous driving systems. This is challenging due to the difficulty of combining multi-granularity geometric and semantic features from two drastically different modalities. Recent approaches aim at exploring the semantic densities of camera features through lifting points in 2D camera images (referred to as “seeds”) into 3D space, and then incorporate 2D semantics via cross-modal interaction or fusion techniques. However, depth information is under-investigated in these approaches when lifting points into 3D space, thus 2D semantics can not be reliably fused with 3D points. Moreover, their multi-modal fusion strategy, which is implemented as concatenation or attention, either can not effectively fuse 2D and 3D information or is unable to perform fine-grained interactions in the voxel space. To this end, we propose a novel framework with better utilization of the depth information and fine-grained cross-modal interaction between LiDAR and camera, which consists of two important components. First, a Multi-Depth Unprojection (MDU) method is used to enhance the depth quality of the lifted points at each interaction level. Second, a Gated Modality-Aware Convolution (GMA-Conv) block is applied to modulate voxels involved with the camera modality in a fine-grained manner and then aggregate multi-modal features into a unified space. Together they provide the detection head with more comprehensive features from LiDAR and camera. On the nuScenes test benchmark, our proposed method, abbreviated as MSMD-Fusion, achieves state-of-the-art results on both 3D object detection and tracking tasks without using test-time-augmentation and ensemble techniques. The code is available at https://github.com/SxJyJay/MSMDFusion.

        ----

        ## [2062] Virtual Sparse Convolution for Multimodal 3D Object Detection

        **Authors**: *Hai Wu, Chenglu Wen, Shaoshuai Shi, Xin Li, Cheng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02074](https://doi.org/10.1109/CVPR52729.2023.02074)

        **Abstract**:

        Recently, virtuall pseudo-point-based 3D object detection that seamlessly fuses RGB images and LiDAR data by depth completion has gained great attention. However, virtual points generated from an image are very dense, introducing a huge amount of redundant computation during detection. Meanwhile, noises brought by inaccurate depth completion significantly degrade detection precision. This paper proposes a fast yet effective backbone, termed Vir-ConvNet, based on a new operator VirConv (Virtual Sparse Convolution), for virtual-point-based 3D object detection. VirConv consists of two key designs: (1) StVD (Stochastic Voxel Discard) and (2) NRConv (Noise-Resistant Sub-manifold Convolution). StVD alleviates the computation problem by discarding large amounts of nearby redundant voxels. NRConv tackles the noise problem by encoding voxel features in both 2D image and 3D LiDAR space. By integrating VirConv, we first develop an efficient pipeline VirConv-L based on an early fusion design. Then, we build a high-precision pipeline Vir Conv-T based on a transformed refinement scheme. Finally, we develop a semi-supervised pipeline VirConv-S based on a pseudo-label framework. On the KITTI car 3D detection test leader-board, our VirConv-L achieves 85% AP with a fast running speed of 56ms. Our VirConv-T and VirConv-S attains a high-precision of 86.3% and 87.2% AP, and currently rank 2
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">nd</sup>
 and 1
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">st</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
On the date of CVPR deadline, i. e., Nov.11, 2022, respectively. The code is available at https://github.com/hailanyi/VirConv.

        ----

        ## [2063] Optimal Transport Minimization: Crowd Localization on Density Maps for Semi-Supervised Counting

        **Authors**: *Wei Lin, Antoni B. Chan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02075](https://doi.org/10.1109/CVPR52729.2023.02075)

        **Abstract**:

        The accuracy of crowd counting in images has improved greatly in recent years due to the development of deep neural networks for predicting crowd density maps. However, most methods do not further explore the ability to localize people in the density map, with those few works adopting simple methods, like finding the local peaks in the density map. In this paper, we propose the optimal transport minimization (OT-M) algorithm for crowd localization with density maps. The objective of OT-M is to find a target point map that has the minimal Sinkhorn distance with the input density map, and we propose an iterative algorithm to compute the solution. We then apply OT-M to generate hard pseudo-labels (point maps) for semi-supervised counting, rather than the soft pseudo-labels (density maps) used in previous methods. Our hard pseudo-labels provide stronger supervision, and also enable the use of recent density-to-point loss functions for training. We also propose a confidence weighting strategy to give higher weight to the more reliable unlabeled data. Extensive experiments show that our methods achieve outstanding performance on both crowd localization and semi-supervised counting. Code is available at https://github.com/Elin24/OT-M.

        ----

        ## [2064] VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking

        **Authors**: *Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, Jiaya Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02076](https://doi.org/10.1109/CVPR52729.2023.02076)

        **Abstract**:

        3D object detectors usually rely on hand-crafted proxies, e.g., anchors or centers, and translate well-studied 2D frameworks to 3D. Thus, sparse voxel features need to be densified and processed by dense prediction heads, which inevitably costs extra computation. In this paper, we instead propose VoxelNext for fully sparse 3D object detection. Our core insight is to predict objects directly based on sparse voxel features, without relying on hand-crafted proxies. Our strong sparse convolutional network VoxelNeXt detects and tracks 3D objects through voxel features entirely. It is an elegant and efficient framework, with no need for sparse-to-dense conversion or NMS post-processing. Our method achieves a better speed-accuracy trade-off than other mainframe detectors on the nuScenes dataset. For the first time, we show that a fully sparse voxel-based representation works decently for LIDAR 3D object detection and tracking. Extensive experiments on nuScenes, Waymo, and Argoverse2 benchmarks validate the effectiveness of our approach. Without bells and whistles, our model outperforms all existing LIDAR methods on the nuScenes tracking test benchmark. Code and models are available at github.com/dvlab-research/VoxelNeXt.

        ----

        ## [2065] GraVoS: Voxel Selection for 3D Point-Cloud Detection

        **Authors**: *Oren Shrout, Yizhak Ben-Shabat, Ayellet Tal*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02077](https://doi.org/10.1109/CVPR52729.2023.02077)

        **Abstract**:

        3D object detection within large 3D scenes is challenging not only due to the sparsity and irregularity of 3D point clouds, but also due to both the extreme foreground-background scene imbalance and class imbalance. A common approach is to add ground-truth objects from other scenes. Differently, we propose to modify the scenes by removing elements (voxels), rather than adding ones. Our approach selects the “meaningful” voxels, in a manner that addresses both types of dataset imbalance. The approach is general and can be applied to any voxel-based detector, yet the meaningfulness of a voxel is network-dependent. Our voxel selection is shown to improve the performance of several prominent 3D detection methods.

        ----

        ## [2066] MSeg3D: Multi-Modal 3D Semantic Segmentation for Autonomous Driving

        **Authors**: *Jiale Li, Hang Dai, Hao Han, Yong Ding*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02078](https://doi.org/10.1109/CVPR52729.2023.02078)

        **Abstract**:

        LiDAR and camera are two modalities available for 3D semantic segmentation in autonomous driving. The popular LiDAR-only methods severely suffer from inferior segmentation on small and distant objects due to insufficient laser points, while the robust multi-modal solution is under-explored, where we investigate three crucial inherent difficulties: modality heterogeneity, limited sensor field of view intersection, and multi-modal data augmentation. We propose a multi-modal 3D semantic segmentation model (MSeg3D) with joint intra-modal feature extraction and inter-modal feature fusion to mitigate the modality heterogeneity. The multi-modal fusion in MSeg3D consists of geometry-based feature fusion GF-Phase, cross-modal feature completion, and semantic-based feature fusion SF-Phase on all visible points. The multi-modal data augmentation is reinvigorated by applying asymmetric transformations on LiDAR point cloud and multi-camera images individually, which benefits the model training with diversified augmentation transformations. MSeg3D achieves state-of-the-art results on nuScenes, Waymo, and SemanticKITTI datasets. Under the malfunctioning multi-camera input and the multi-frame point clouds input, MSeg3D still shows robustness and improves the LiDAR-only baseline. Our code is publicly available at https://github.com/jialeli1/lidarseg3d.

        ----

        ## [2067] LaserMix for Semi-Supervised LiDAR Semantic Segmentation

        **Authors**: *Lingdong Kong, Jiawei Ren, Liang Pan, Ziwei Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02079](https://doi.org/10.1109/CVPR52729.2023.02079)

        **Abstract**:

        Densely annotating LiDAR point clouds is costly, which often restrains the scalability of fully-supervised learning methods. In this work, we study the underexplored semi-supervised learning (SSL) in LiDAR semantic segmentation. Our core idea is to leverage the strong spatial cues of LiDAR point clouds to better exploit unlabeled data. We propose LaserMix to mix laser beams from different LiDAR scans and then encourage the model to make consistent and confident predictions before and after mixing. Our framework has three appealing properties. 1) Generic: LaserMix is agnostic to LiDAR representations (e.g., range view and voxel), and hence our SSL framework can be universally applied. 2) Statistically grounded: We provide a detailed analysis to theoretically explain the applicability of the proposed framework. 3) Effective: Comprehensive experimental analysis on popular LiDAR segmentation datasets (nuScenes, SemanticKITTI, and ScribbleKITTI) demonstrates our effectiveness and superiority. Notably, we achieve competitive results over fully-supervised counterparts with 2× to 5× fewer labels and improve the supervised-only baseline significantly by relatively 10.8%. We hope this concise yet high-performing framework could facilitate future research in semi-supervised LiDAR segmentation. Code is publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/ldkong1205/LaserMix..

        ----

        ## [2068] Implicit Surface Contrastive Clustering for LiDAR Point Clouds

        **Authors**: *Zaiwei Zhang, Min Bai, Li Erran Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02080](https://doi.org/10.1109/CVPR52729.2023.02080)

        **Abstract**:

        Self-supervised pretraining on large unlabeled datasets has shown tremendous success in improving the task performance of many 2D and small scale 3D computer vision tasks. However, the popular pretraining approaches have not been impactfully applied to outdoor LiDAR point cloud perception due to the latter's scene complexity and wide range. We propose a new self-supervised pretraining method ISCC with two novel pretext tasks for LiDAR point clouds. The first task uncovers semantic information by sorting local groups of points in the scene into a globally consistent set of semantically meaningful clusters using contrastive learning, complemented by a second task which reasons about precise surfaces of various parts of the scene through implicit surface reconstruction to learn geometric structures. We demonstrate their effectiveness through transfer learning on 3D object detection and semantic segmentation in real world LiDAR scenes. We further design an unsupervised semantic grouping task to show that our approach learns highly semantically meaningful features.

        ----

        ## [2069] Semi-Weakly Supervised Object Kinematic Motion Prediction

        **Authors**: *Gengxin Liu, Qian Sun, Haibin Huang, Chongyang Ma, Yulan Guo, Li Yi, Hui Huang, Ruizhen Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02081](https://doi.org/10.1109/CVPR52729.2023.02081)

        **Abstract**:

        Given a 3D object, kinematic motion prediction aims to identify the mobile parts as well as the corresponding motion parameters. Due to the large variations in both topological structure and geometric details of 3D objects, this remains a challenging task and the lack of large scale labeled data also constrain the performance of deep learning based approaches. In this paper, we tackle the task of object kinematic motion prediction problem in a semi-weakly supervised manner. Our key observations are two-fold. First, although 3D dataset with fully annotated motion labels is limited, there are existing datasets and methods for object part semantic segmentation at large scale. Second, semantic part segmentation and mobile part segmentation is not always consistent but it is possible to detect the mobile parts from the underlying 3D structure. Towards this end, we propose a graph neural network to learn the map between hierarchical part-level segmentation and mobile parts parameters, which are further refined based on geometric alignment. This network can be first trained on PartNet-Mobility dataset with fully labeled mobility information and then applied on PartNet dataset with fine-grained and hierarchical part-level segmentation. The network predictions yield a large scale of 3D objects with pseudo labeled mobility information and can further be used for weakly-supervised learning with pre-existing segmentation. Our experiments show there are significant performance boosts with the augmented data for previous method designed for kinematic motion prediction on 3D partial scans.

        ----

        ## [2070] PartSLIP: Low-Shot Part Segmentation for 3D Point Clouds via Pretrained Image-Language Models

        **Authors**: *Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, Hao Su*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02082](https://doi.org/10.1109/CVPR52729.2023.02082)

        **Abstract**:

        Generalizable 3D part segmentation is important but challenging in vision and robotics. Training deep models via conventional supervised methods requires large-scale 3D datasets with fine-grained part annotations, which are costly to collect. This paper explores an alternative way for low-shot part segmentation of 3D point clouds by leveraging a pretrained image-language model, GLIP. which achieves superior performance on open-vocabulary 2D detection. We transfer the rich knowledge from 2D to 3D through GLIP-based part detection on point cloud rendering and a novel 2D-to-3D label lifting algorithm. We also utilize multi-view 3D priors and few-shot prompt tuning to boost performance significantly. Extensive evaluation on PartNet and PartNet-Mobility datasets shows that our method enables excellent zero-shot 3D part segmentation. Our few-shot version not only outperforms existing few-shot approaches by a large margin but also achieves highly competitive results compared to the fully supervised counterpart. Furthermore, we demonstrate that our method can be directly applied to iPhone-scanned point clouds without significant domain gaps.

        ----

        ## [2071] Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions

        **Authors**: *Yurui Zhu, Tianyu Wang, Xueyang Fu, Xuanyu Yang, Xin Guo, Jifeng Dai, Yu Qiao, Xiaowei Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02083](https://doi.org/10.1109/CVPR52729.2023.02083)

        **Abstract**:

        Image restoration under multiple adverse weather conditions aims to remove weather-related artifacts by using a single set of network parameters. In this paper, we find that image degradations under different weather conditions contain general characteristics as well as their specific characteristics. Inspired by this observation, we design an efficient unified framework with a two-stage training strategy to explore the weather-general and weather-specific features. The first training stage aims to learn the weather-general features by taking the images under various weather conditions as inputs and outputting the coarsely restored results. The second training stage aims to learn to adaptively expand the specific parameters for each weather type in the deep model, where the requisite positions for expanding weather-specific parameters are automatically learned. Hence, we can obtain an efficient and unified model for image restoration under multiple adverse weather conditions. Moreover, we build the first real-world benchmark dataset with multiple weather conditions to better deal with realworld weather scenarios. Experimental results show that our method achieves superior performance on all the synthetic and real-world benchmarks. Codes and datasets are available at this repository.

        ----

        ## [2072] Geometry and Uncertainty-Aware 3D Point Cloud Class-Incremental Semantic Segmentation

        **Authors**: *Yuwei Yang, Munawar Hayat, Zhao Jin, Chao Ren, Yinjie Lei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02084](https://doi.org/10.1109/CVPR52729.2023.02084)

        **Abstract**:

        Despite the significant recent progress made on 3D point cloud semantic segmentation, the current methods require training data for all classes at once, and are not suitable for real-life scenarios where new categories are being continuously discovered. Substantial memory storage and expensive re-training is required to update the model to sequentially arriving data for new concepts. In this paper, to continually learn new categories using previous knowledge, we introduce class-incremental semantic segmentation of 3D point cloud. Unlike 2D images, 3D point clouds are disordered and unstructured, making it difficult to store and transfer knowledge especially when the previous data is not available. We further face the challenge of semantic shift, where previous/future classes are indiscriminately collapsed and treated as the background in the current step, causing a dramatic performance drop on past classes. We exploit the structure of point cloud and propose two strategies to address these challenges. First, we design a geometry-aware distillation module that transfers point-wise feature associations in terms of their geometric characteristics. To counter forgetting caused by the semantic shift, we further develop an uncertainty-aware pseudo-labelling scheme that eliminates noise in uncertain pseudo-labels by label propagation within a local neighborhood. Our extensive experiments on S3DIS and ScanNet in a class-incremental setting show impressive results comparable to the joint training strategy (upper bound). Code is available at: https://github.com/leolyj/3DPC-CISS

        ----

        ## [2073] Learning 3D Representations from 2D Pre-Trained Models via Image-to-Point Masked Autoencoders

        **Authors**: *Renrui Zhang, Liuhui Wang, Yu Qiao, Peng Gao, Hongsheng Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02085](https://doi.org/10.1109/CVPR52729.2023.02085)

        **Abstract**:

        Pre-training by numerous image data has become defacto for robust 2D representations. In contrast, due to the expensive data processing, a paucity of 3D datasets severely hinders the learning for high-quality 3D features. In this paper, we propose an alternative to obtain superior 3D representations from 2D pre-trained models via Image-to-Point Masked Autoencoders, named as I2P-MAE. By self-supervised pre-training, we leverage the well learned 2D knowledge to guide 3D masked autoencoding, which reconstructs the masked point tokens with an encoder-decoder architecture. Specifically, we first utilize off-the-shelf 2D models to extract the multi-view visual features of the input point cloud, and then conduct two types of image-to-point learning schemes. For one, we introduce a 2D-guided masking strategy that maintains semantically important point tokens to be visible. Compared to random masking, the network can better concentrate on significant 3D structures with key spatial cues. For another, we enforce these visible tokens to reconstruct multi-view 2D features after the decoder. This enables the network to effectively inherit high-level 2D semantics for discriminative 3D modeling. Aided by our image-to-point pre-training, the frozen I2P-MAE, without any fine-tuning, achieves 93.4% accuracy for linear SVM on ModelNet40, competitive to existing fully trained methods. By further fine-tuning on on ScanObjectNN's hardest split, I2P-MAE attains the state-of-the-art 90.11% accuracy, +3.68% to the second-best, demonstrating superior transferable capacity. Code is available at https://github.com/ZrrSkywalker/I2P-MAE.

        ----

        ## [2074] ToThePoint: Efficient Contrastive Learning of 3D Point Clouds via Recycling

        **Authors**: *Xinglin Li, Jiajing Chen, Jinhui Ouyang, Hanhui Deng, Senem Velipasalar, Di Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02086](https://doi.org/10.1109/CVPR52729.2023.02086)

        **Abstract**:

        Recent years have witnessed significant developments in point cloud processing, including classification and segmentation. However, supervised learning approaches need a lot of well-labeled data for training, and annotation is labor-and time-intensive. Self-supervised learning, on the other hand, uses unlabeled data, and pretrains a back-bone with a pretext task to extract latent representations to be used with the downstream tasks. Compared to 2D images, self-supervised learning of 3D point clouds is under-explored. Existing models, for self-supervised learning of 3D point clouds, rely on a large number of data samples, and require significant amount of computational re-sources and training time. To address this issue, we propose a novel contrastive learning approach, referred to as To ThePoint. Different from traditional contrastive learning methods, which maximize agreement between features obtained from a pair of point clouds formed only with different types of augmentation, ToThePoint also maximizes the agreement between the permutation invariant features and features discarded after max pooling. We first perform self-supervised learning on the ShapeNet dataset, and then evaluate the performance of the network on different downstream tasks. In the downstream task experiments, performed on the ModelNet40, ModelNet40C, ScanobjectNN and ShapeNet-Part datasets, our proposed ToThe-Point achieves competitive, if not better results compared to the state-of-the-art baselines, and does so with significantly less training time (200 times faster than baselines).

        ----

        ## [2075] PointDistiller: Structured Knowledge Distillation Towards Efficient and Compact 3D Detection

        **Authors**: *Linfeng Zhang, Runpei Dong, Hung-Shuo Tai, Kaisheng Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02087](https://doi.org/10.1109/CVPR52729.2023.02087)

        **Abstract**:

        The remarkable breakthroughs in point cloud representation learning have boosted their usage in real-world applications such as self-driving cars and virtual reality. However, these applications usually have a strict requirement for not only accurate but also efficient 3D object detection. Recently, knowledge distillation has been proposed as an effective model compression technique, which transfers the knowledge from an over-parameterized teacher to a lightweight student and achieves consistent effectiveness in 2D vision. However, due to point clouds' sparsity and irregularity, directly applying previous image-based knowledge distillation methods to point cloud detectors usually leads to unsatisfactory performance. To fill the gap, this paper proposes PointDistiller; a structured knowledge distillation framework for point clouds-based 3D detection. Concretely, PointDistiller includes local distillation which extracts and distills the local geometric structure of point clouds with dynamic graph convolution and reweighted learning strategy, which highlights student learning on the crucial points or voxels to improve knowledge distillation efficiency. Extensive experiments on both voxels-based and raw points-based detectors have demonstrated the effectiveness of our method over seven previous knowledge distillation methods. For instance, our 4 x compressed PointPillars student achieves 2.8 and 3.4 mAP improvements on BEV and 3D object detection, outperforming its teacher by 0.9 and 1.8 mAP, respectively. Codes are available in https://github.com/RunpeiDong/PointDistiller.

        ----

        ## [2076] PointConvFormer: Revenge of the Point-based Convolution

        **Authors**: *Wenxuan Wu, Fuxin Li, Qi Shan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02088](https://doi.org/10.1109/CVPR52729.2023.02088)

        **Abstract**:

        We introduce PointConvFormer, a novel building block for point cloud based deep network architectures. Inspired by generalization theory, PointConvFormer combines ideas from point convolution, where filter weights are only based on relative position, and Transformers which utilize feature-based attention. In PointConvFormer, attention computed from feature difference between points in the neighborhood is used to modify the convolutional weights at each point. Hence, we preserved the invariances from point convolution, whereas attention helps to select relevant points in the neighborhood for convolution. PointConvFormer is suitable for multiple tasks that require details at the point level, such as segmentation and scene flow estimation tasks. We experiment on both tasks with multiple datasets including ScanNet, SemanticKitti, FlyingThings3D and KITTI. Our results show that PointConvFormer offers a better accuracy-speed tradeoff than classic convolutions, regular transformers, and voxelized sparse convolution approaches. Visualizations show that PointConvFormer performs similarly to convolution on flat areas, whereas the neighborhood selection effect is stronger on object boundaries, showing that it has got the best of both worlds. The code will be available.

        ----

        ## [2077] Self-Positioning Point-Based Transformer for Point Cloud Understanding

        **Authors**: *Jinyoung Park, Sanghyeok Lee, Sihyeon Kim, Yunyang Xiong, Hyunwoo J. Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02089](https://doi.org/10.1109/CVPR52729.2023.02089)

        **Abstract**:

        Transformers have shown superior performance on various computer vision tasks with their capabilities to capture long-range dependencies. Despite the success, it is challenging to directly apply Transformers on point clouds due to their quadratic cost in the number of points. In this paper, we present a Self-Positioning point-based Transformer (SPoTr), which is designed to capture both local and global shape contexts with reduced complexity. Specifically, this architecture consists of local self-attention and self-positioning point-based global cross-attention. The self-positioning points, adaptively located based on the input shape, consider both spatial and semantic information with disentangled attention to improve expressive power. With the self-positioning points, we propose a novel global cross-attention mechanism for point clouds, which improves the scalability of global self-attention by allowing the attention module to compute attention weights with only a small set of self-positioning points. Experiments show the effectiveness of SPoTr on three point cloud tasks such as shape classification, part segmentation, and scene segmentation. In particular, our proposed model achieves an accuracy gain of 2.6% over the previous best models on shape classification with ScanObjectNN. We also provide qualitative analyses to demonstrate the interpretability of self-positioning points. The code of SPoTr is available at https://github.com/mlvlab/SPoTr.

        ----

        ## [2078] PointClustering: Unsupervised Point Cloud Pre-training using Transformation Invariance in Clustering

        **Authors**: *Fuchen Long, Ting Yao, Zhaofan Qiu, Lusong Li, Tao Mei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02090](https://doi.org/10.1109/CVPR52729.2023.02090)

        **Abstract**:

        Feature invariance under different data transformations, i.e., transformation invariance, can be regarded as a type of self-supervision for representation learning. In this paper, we present PointClustering, a new unsupervised representation learning scheme that leverages transformation invariance for point cloud pre-training. PointClustering formulates the pretext task as deep clustering and employs transformation invariance as an inductive bias, following the philosophy that common point cloud transformation will not change the geometric properties and semantics. Technically, PointClustering iteratively optimizes the feature clusters and backbone, and delves into the transformation invariance as learning regularization from two perspectives: point level and instance level. Point-level invariance learning maintains local geometric properties through gathering point features of one instance across transformations, while instance-level invariance learning further measures clusters over the entire dataset to explore semantics of instances. Our PointClustering is architecture-agnostic and readily applicable to MLP-based, CNN-based and Transformer-based backbones. We empirically demonstrate that the models pre-learnt on the ScanNet dataset by PointClustering provide superior performances on six benchmark-s, across downstream tasks of classification and segmentation. More remarkably, PoinrClustering achieves an accuracy of 94.5% on ModelNet40 with Transformer backbone. Source code is available at https://github.com/FuchenUSTC/PointClustering.

        ----

        ## [2079] Neural Intrinsic Embedding for Non-Rigid Point Cloud Matching

        **Authors**: *Puhua Jiang, Mingze Sun, Ruqi Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02091](https://doi.org/10.1109/CVPR52729.2023.02091)

        **Abstract**:

        As a primitive 3D data representation, point clouds are prevailing in 3D sensing, yet short of intrinsic structural information of the underlying objects. Such discrepancy poses great challenges in directly establishing correspondences between point clouds sampled from deformable shapes. In light of this, we propose Neural Intrinsic Embedding (NIE) to embed each vertex into a high-dimensional space in a way that respects the intrinsic structure. Based upon NIE, we further present a weakly-supervised learning framework for non-rigid point cloud registration. Unlike the prior works, we do not require expansive and sensitive off-line basis construction (e.g., eigen-decomposition of Laplacians), nor do we require ground-truth correspondence labels for supervision. We empirically show that our framework performs on par with or even better than the state-of-the-art baselines, which generally require more supervision and/or more structural geometric input.

        ----

        ## [2080] HGNet: Learning Hierarchical Geometry from Points, Edges, and Surfaces

        **Authors**: *Ting Yao, Yehao Li, Yingwei Pan, Tao Mei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02092](https://doi.org/10.1109/CVPR52729.2023.02092)

        **Abstract**:

        Parsing an unstructured point set into constituent local geometry structures (e.g., edges or surfaces) would be helpful for understanding and representing point clouds. This motivates us to design a deep architecture to model the hierarchical geometry from points, edges, surfaces (triangles), to super-surfaces (adjacent surfaces) for the thorough analysis of point clouds. In this paper, we present a novel Hierarchical Geometry Network (HGNet) that integrates such hierarchical geometry structures from super-surfaces, surfaces, edges, to points in a top-down manner for learning point cloud representations. Technically, we first construct the edges between every two neighbor points. A point-level representation is learnt with edge-to-point aggregation, i.e., aggregating all connected edges into the anchor point. Next, as every two neighbor edges compose a surface, we obtain the edge-level representation of each anchor edge via surface-to-edge aggregation over all neighbor surfaces. Furthermore, the surface-level representation is achieved through super-surface-to-surface aggregation by transforming all super-surfaces into the anchor surface. A Transformer structure is finally devised to unify all the point-level, edge-level, and surface-level features into the holistic point cloud representations. Extensive experiments on four point cloud analysis datasets demonstrate the superiority of HGNet for 3D object classification and part/semantic segmentation tasks. More remarkably, HGNet achieves the overall accuracy of 89.2% on ScanObjectNN, improving PointNeXt-S by 1.5%.

        ----

        ## [2081] LP-DIF: Learning Local Pattern-Specific Deep Implicit Function for 3D Objects and Scenes

        **Authors**: *Meng Wang, Yu-Shen Liu, Yue Gao, Kanle Shi, Yi Fang, Zhizhong Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02093](https://doi.org/10.1109/CVPR52729.2023.02093)

        **Abstract**:

        Deep Implicit Function (DIF) has gained much popularity as an efficient 3D shape representation. To capture geometry details, current mainstream methods divide 3D shapes into local regions and then learn each one with a local latent code via a decoder. Such local methods can capture more local details due to less diversity among local regions than global shapes. Although the diversity of local regions has been decreased compared to global approaches, the diversity in different local regions still poses a challenge in learning an implicit function when treating all regions equally using only a single decoder. What is worse, these local regions often exhibit imbalanced distributions, where certain regions have significantly fewer observations. This leads that fine geometry details could not be preserved well. To solve this problem, we propose a novel Local Pattern-specific Implicit Function, named LP-DIF, to represent a shape with clusters of local regions and multiple decoders, where each decoder only focuses on one cluster of local regions which share a certain pattern. Specifically, we first extract local codes for all regions, and then cluster them into multiple groups in the latent space, where similar regions sharing a common pattern fall into one group. After that, we train multiple decoders for mining local patterns of different groups, which simplifies the learning of fine geometric details by reducing the diversity of local regions seen by each decoder. To further alleviate the data-imbalance problem, we introduce a region re-weighting module to each pattern-specific decoder using a kernel density estimator, which dynamically re-weights the regions during learning. Our LP-DIF can restore more geometry details, and thus improve the quality of 3D reconstruction. Experiments demonstrate that our method can achieve the state-of-the-art performance over previous methods. Code is available at https://github.com/gtyxyz/lpdif.

        ----

        ## [2082] Conjugate Product Graphs for Globally Optimal 2D-3D Shape Matching

        **Authors**: *Paul Roetzer, Zorah Lähner, Florian Bernard*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02094](https://doi.org/10.1109/CVPR52729.2023.02094)

        **Abstract**:

        We consider the problem of finding a continuous and non-rigid matching between a 2D contour and a 3D mesh. While such problems can be solved to global optimality by finding a shortest path in the product graph between both shapes, existing solutions heavily rely on unrealistic prior assumptions to avoid degenerate solutions (e.g. knowledge to which region of the 3D shape each point of the 2D contour is matched). To address this, we propose a novel 2D-3D shape matching formalism based on the conjugate prod-uct graph between the 2D contour and the 3D shape. Doing so allows us for the first time to consider higher-order costs, i.e. defined for edge chains, as opposed to costs de-fined for single edges. This offers substantially more flexi-bility, which we utilise to incorporate a local rigidity prior. By doing so, we effectively circumvent degenerate solutions and thereby obtain smoother and more realistic matchings, even when using only a one-dimensional feature descrip-tor. Overall, our method finds globally optimal and contin-uous 2D-3D matchings, has the same asymptotic complex-ity as previous solutions, produces state-of-the-art results for shape matching and is even capable of matching partial shapes. Our code is publicly available.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/paulOnoah/sm-2D3D

        ----

        ## [2083] UTM: A Unified Multiple Object Tracking Model with Identity-Aware Feature Enhancement

        **Authors**: *Sisi You, Hantao Yao, Bing-Kun Bao, Changsheng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02095](https://doi.org/10.1109/CVPR52729.2023.02095)

        **Abstract**:

        Recently, Multiple Object Tracking has achieved great success, which consists of object detection, feature embedding, and identity association. Existing methods apply the three-step or two-step paradigm to generate robust trajectories, where identity association is independent of other components. However, the independent identity association results in the identity-aware knowledge contained in the tracklet not be used to boost the detection and embedding modules. To overcome the limitations of existing methods, we introduce a novel Unified Tracking Model (UTM) to bridge those three components for generating a positive feedback loop with mutual benefits. The key insight of UTM is the Identity-Aware Feature Enhancement (IAFE), which is applied to bridge and benefit these three components by utilizing the identity-aware knowledge to boost detection and embedding. Formally, IAFE contains the Identity-Aware Boosting Attention (IABA) and the Identity-Aware Erasing Attention (IAEA), where IABA enhances the consistent regions between the current frame feature and identity-aware knowledge, and IAEA suppresses the distracted regions in the current frame feature. With better detections and embeddings, higher-quality tracklets can also be generated. Extensive experiments of public and private detections on three benchmarks demonstrate the robustness of UTM.

        ----

        ## [2084] Learning Rotation-Equivariant Features for Visual Correspondence

        **Authors**: *Jongmin Lee, Byungjin Kim, Seungwook Kim, Minsu Cho*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02096](https://doi.org/10.1109/CVPR52729.2023.02096)

        **Abstract**:

        Extracting discriminative local features that are invariant to imaging variations is an integral part of establishing correspondences between images. In this work, we introduce a self-supervised learning framework to extract discriminative rotation-invariant descriptors using group-equivariant CNNs. Thanks to employing group-equivariant CNNs, our method effectively learns to obtain rotation-equivariant features and their orientations explicitly, without having to perform sophisticated data augmentations. The resultant features and their orientations are further processed by group aligning, a novel invariant mapping technique that shifts the group-equivariant features by their orientations along the group dimension. Our group aligning technique achieves rotation-invariance without any collapse of the group dimension and thus eschews loss of discriminability. The proposed method is trained end-to-end in a self-supervised manner, where we use an orientation alignment loss for the orientation estimation and a contrastive descriptor loss for robust local descriptors to geometric/photometric variations. Our method demonstrates state-of-the-art matching accuracy among existing rotation-invariant descriptors under varying rotation and also shows competitive results when transferred to the task of keypoint matching and camera pose estimation.

        ----

        ## [2085] Adaptive Spot-Guided Transformer for Consistent Local Feature Matching

        **Authors**: *Jiahuan Yu, Jiahao Chang, Jianfeng He, Tianzhu Zhang, Jiyang Yu, Feng Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02097](https://doi.org/10.1109/CVPR52729.2023.02097)

        **Abstract**:

        Local feature matching aims at finding correspondences between a pair of images. Although current detector-free methods leverage Transformer architecture to obtain an impressive performance, few works consider maintaining local consistency. Meanwhile, most methods struggle with large scale variations. To deal with the above issues, we propose Adaptive Spot-Guided Transformer (ASTR) for local feature matching, which jointly models the local consistency and scale variations in a unified coarse-to-fine architecture. The proposed ASTR enjoys several merits. First, we design a spot-guided aggregation module to avoid interfering with irrelevant areas during feature aggregation. Second, we design an adaptive scaling module to adjust the size of grids according to the calculated depth information at fine stage. Extensive experimental results on five standard benchmarks demonstrate that our ASTR performs favorably against state-of-the-art methods. Our code will be released on https://astr2023.github.io.

        ----

        ## [2086] PMatch: Paired Masked Image Modeling for Dense Geometric Matching

        **Authors**: *Shengjie Zhu, Xiaoming Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02098](https://doi.org/10.1109/CVPR52729.2023.02098)

        **Abstract**:

        Dense geometric matching determines the dense pixel-wise correspondence between a source and support image corresponding to the same 3D structure. Prior works employ an encoder of transformer blocks to correlate the two-frame features. However, existing monocular pretraining tasks, e.g., image classification, and masked image modeling (MIM), can not pretrain the cross-frame module, yielding less optimal performance. To resolve this, we reformulate the MIM from reconstructing a single masked image to reconstructing a pair of masked images, enabling the pretraining of transformer module. Additionally, we incorporate a decoder into pretraining for improved upsampling results. Further, to be robust to the textureless area, we propose a novel cross-frame global matching module (CFGM). Since the most textureless area is planar surfaces, we propose a homography loss to further regularize its learning. Combined together, we achieve the State-of-The-Art (SoTA) performance on geometric matching. Codes and models are available at https://github.com/ShngJZ/PMatch.

        ----

        ## [2087] Iterative Geometry Encoding Volume for Stereo Matching

        **Authors**: *Gangwei Xu, Xianqi Wang, Xiaohuan Ding, Xin Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02099](https://doi.org/10.1109/CVPR52729.2023.02099)

        **Abstract**:

        Recurrent All-Pairs Field Transforms (RAFT) has shown great potentials in matching tasks. However, all-pairs correlations lack non-local geometry knowledge and have difficulties tackling local ambiguities in ill-posed regions. In this paper, we propose Iterative Geometry Encoding Volume (IGEV-Stereo), a new deep network architecture for stereo matching. The proposed IGEV-Stereo builds a combined geometry encoding volume that encodes geometry and context information as well as local matching details, and iteratively indexes it to update the disparity map. To speed up the convergence, we exploit GEV to regress an accurate starting point for ConvGRUs iterations. Our IGEV-Stereo ranks 1st on KITTI 2015 and 2012 (Reflective) among all published methods and is the fastest among the top 10 methods. In addition, IGEV-Stereo has strong cross-dataset generalization as well as high inference efficiency. We also extend our IGEV to multi-view stereo (MVS), i.e. IGEV-MVS, which achieves competitive accuracy on DTU benchmark. Code is available at https://github.com/gangweiX/IGEV.

        ----

        ## [2088] Adaptive Annealing for Robust Geometric Estimation

        **Authors**: *Chitturi Sidhartha, Lalit Manam, Venu Madhav Govindu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02100](https://doi.org/10.1109/CVPR52729.2023.02100)

        **Abstract**:

        Geometric estimation problems in vision are often solved via minimization of statistical loss functions which account for the presence of outliers in the observations. The corresponding energy landscape often has many local minima. Many approaches attempt to avoid local minima by an-nealing the scale parameter of loss functions using methods such as graduated non-convexity (GNC). However, little attention has been paid to the annealing schedule, which is often carried out in a fixed manner, resulting in a poor speed-accuracy trade-off and unreliable convergence to the global minimum. In this paper, we propose a principled approach for adaptively annealing the scale for GNC by tracking the positive-definiteness (i.e. local convexity) of the Hessian of the cost function. We illustrate our approach using the classic problem of registering 3D correspondences in the presence of noise and outliers. We also develop approximations to the Hessian that significantly speeds up our method. The effectiveness of our approach is validated by comparing its performance with state-of-the-art 3D registration approaches on a number of synthetic and real datasets. Our approach is accurate and efficient and converges to the global solution more reliably than the state-of-the-art methods.

        ----

        ## [2089] Tangentially Elongated Gaussian Belief Propagation for Event-Based Incremental Optical Flow Estimation

        **Authors**: *Jun Nagata, Yusuke Sekikawa*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02101](https://doi.org/10.1109/CVPR52729.2023.02101)

        **Abstract**:

        Optical flow estimation is a fundamental functionality in computer vision. An event-based camera, which asynchronously detects sparse intensity changes, is an ideal device for realizing low-latency estimation of the optical flow owing to its low-latency sensing mechanism. An existing method using local plane fitting of events could utilize the sparsity to realize incremental updates for low-latency estimation; however, its output is merely a normal component of the full optical flow. An alternative approach using a frame-based deep neural network could estimate the full flow; however, its intensive non-incremental dense operation prohibits the low-latency estimation. We propose tangentially elongated Gaussian (TEG) belief propagation (BP) that realizes incremental full-flow estimation. We model the probability of full flow as the joint distribution of TEGs from the normal flow measurements, such that the marginal of this distribution with correct prior equals the full flow. We formulate the marginalization using a message-passing based on the BP to realize efficient incremental updates using sparse measurements. In addition to the theoretical justification, we evaluate the effectiveness of the TEGBP in real-world datasets; it outperforms SOTA incremental quasi-full flow method by a large margin. (The code is available at https://github.com/DensoITLab/tegbp/).

        ----

        ## [2090] Robust and Scalable Gaussian Process Regression and Its Applications

        **Authors**: *Yifan Lu, Jiayi Ma, Leyuan Fang, Xin Tian, Junjun Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02102](https://doi.org/10.1109/CVPR52729.2023.02102)

        **Abstract**:

        This paper introduces a robust and scalable Gaussian process regression (GPR) model via variational learning. This enables the application of Gaussian processes to a wide range of real data, which are often large-scale and contaminated by outliers. Towards this end, we employ a mixture likelihood model where outliers are assumed to be sampled from a uniform distribution. We next derive a variational formulation that jointly infers the mode of data, i.e., inlier or outlier, as well as hyperparameters by maximizing a lower bound of the true log marginal likelihood. Compared to previous robust GPR, our formulation approximates the exact posterior distribution. The inducing variable approximation and stochastic variational inference are further introduced to our variational framework, extending our model to large-scale data. We apply our model to two challenging real-world applications, namely feature matching and dense gene expression imputation. Extensive experiments demonstrate the superiority of our model in terms of robustness and speed. Notably, when matching 4k feature points, its inference is completed in milliseconds with almost no false matches. The code is at github.com/YifanLu2000/Robust-Scalable-GPR.

        ----

        ## [2091] BEV-Guided Multi-Modality Fusion for Driving Perception

        **Authors**: *Yunze Man, Liang-Yan Gui, Yu-Xiong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02103](https://doi.org/10.1109/CVPR52729.2023.02103)

        **Abstract**:

        Integrating multiple sensors and addressing diverse tasks in an end-to-end algorithm are challenging yet critical topics for autonomous driving. To this end, we introduce BEVGuide, a novel Bird's Eye- View (BEV) representation learning framework, representing the first attempt to unify a wide range of sensors under direct BEV guidance in an end-to-end fashion. Our architecture accepts input from a diverse sensor pool, including but not limited to Camera, Lidar and Radar sensors, and extracts BEV feature embeddings using a versatile and general transformer backbone. We design a BEV-guided multi-sensor attention block to take queries from BEV embeddings and learn the BEV representation from sensor-specific features. BEVGuide is efficient due to its lightweight backbone design and highly flexible as it supports almost any input sensor configurations. Extensive experiments demonstrate that our framework achieves exceptional performance in BEV perception tasks with a diverse sensor set. Project page is at https://yunzeman.github.io/BEVGuide.

        ----

        ## [2092] HumanBench: Towards General Human-Centric Perception with Projector Assisted Pretraining

        **Authors**: *Shixiang Tang, Cheng Chen, Qingsong Xie, Meilin Chen, Yizhou Wang, Yuanzheng Ci, Lei Bai, Feng Zhu, Haiyang Yang, Li Yi, Rui Zhao, Wanli Ouyang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02104](https://doi.org/10.1109/CVPR52729.2023.02104)

        **Abstract**:

        Human-centric perceptions include a variety of vision tasks, which have widespread industrial applications, including surveillance, autonomous driving, and the metaverse. It is desirable to have a general pretrain model for versatile human-centric downstream tasks. This paper forges ahead along this path from the aspects of both benchmark and pretraining methods. Specifically, we propose a HumanBench based on existing datasets to comprehensively evaluate on the common ground the generalization abilities of different pretraining methods on 19 datasets from 6 diverse downstream tasks, including person ReID, pose estimation, human parsing, pedestrian attribute recognition, pedestrian detection, and crowd counting. To learn both coarse-grained and fine-grained knowledge in human bodies, we further propose a Projector AssisTed Hierarchical pretraining method (PATH) to learn diverse knowledge at different granularity levels. Comprehensive evaluations on HumanBench show that our PATH achieves new state-of-the-art results on 17 downstream datasets and on-par results on the other 2 datasets. The code will be publicly at https://github.com/OpenGVLab/HumanBench.

        ----

        ## [2093] Think Twice before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving

        **Authors**: *Xiaosong Jia, Penghao Wu, Li Chen, Jiangwei Xie, Conghui He, Junchi Yan, Hongyang Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02105](https://doi.org/10.1109/CVPR52729.2023.02105)

        **Abstract**:

        End-to-end autonomous driving has made impressive progress in recent years. Existing methods usually adopt the decoupled encoder-decoder paradigm, where the encoder extracts hidden features from raw sensor data, and the decoder outputs the ego-vehicle's future trajectories or actions. Under such a paradigm, the encoder does not have access to the intended behavior of the ego agent, leaving the burden of finding out safety-critical regions from the massive receptive field and inferring about future situations to the decoder. Even worse, the decoder is usually composed of several simple multi-layer perceptrons (MLP) or GRUs while the encoder is delicately designed (e.g., a combination of heavy ResNets or Transformer). Such an imbalanced resource-task division hampers the learning process. In this work, we aim to alleviate the aforementioned problem by two principles: (1) fully utilizing the capacity of the encoder; (2) increasing the capacity of the decoder. Concretely, we first predict a coarse-grained future position and action based on the encoder features. Then, conditioned on the position and action, the future scene is imagined to check the ramification if we drive accordingly. We also retrieve the encoder features around the predicted coordinate to obtain fine-grained information about the safety-critical region. Finally, based on the predicted future and the retrieved salient feature, we refine the coarse-grained position and action by predicting its offset from ground-truth. The above refinement module could be stacked in a cascaded fashion, which extends the capacity of the decoder with spatial-temporal prior knowledge about the conditioned future. We conduct experiments on the CARLA simulator and achieve state-of-the-art performance in closed-loop benchmarks. Extensive ablation studies demonstrate the effectiveness of each proposed module.

        ----

        ## [2094] ProphNet: Efficient Agent-Centric Motion Forecasting with Anchor-Informed Proposals

        **Authors**: *Xishun Wang, Tong Su, Fang Da, Xiaodong Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02106](https://doi.org/10.1109/CVPR52729.2023.02106)

        **Abstract**:

        Motion forecasting is a key module in an autonomous driving system. Due to the heterogeneous nature of multi-sourced input, multimodality in agent behavior, and low latency required by onboard deployment, this task is notoriously challenging. To cope with these difficulties, this paper proposes a novel agent-centric model with anchor-informed proposals for efficient multimodal motion prediction. We design a modality-agnostic strategy to concisely encode the complex input in a unified manner. We generate diverse proposals, fused with anchors bearing goal-oriented scene context, to induce multimodal prediction that covers a wide range of future trajectories. Our network architecture is highly uniform and succinct, leading to an efficient model amenable for real-world driving deployment. Experiments reveal that our agent-centric network compares favorably with the state-of-the-art methods in prediction accuracy, while achieving scene-centric level inference latency.

        ----

        ## [2095] StarCraftImage: A Dataset For Prototyping Spatial Reasoning Methods For Multi-Agent Environments

        **Authors**: *Sean Kulinski, Nicholas R. Waytowich, James Z. Hare, David I. Inouye*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02107](https://doi.org/10.1109/CVPR52729.2023.02107)

        **Abstract**:

        Spatial reasoning tasks in multi-agent environments such as event prediction, agent type identification, or missing data imputation are important for multiple applications (e.g., autonomous surveillance over sensor networks and subtasks for reinforcement learning (RL)). StarCraft II game replays encode intelligent (and adversarial) multiagent behavior and could provide a testbed for these tasks; however, extracting simple and standardized representations for prototyping these tasks is laborious and hinders reproducibility. In contrast, MNIST and CIFAR10, despite their extreme simplicity, have enabled rapid prototyping and reproducibility of ML methods. Following the simplicity of these datasets, we construct a benchmark spatial reasoning dataset based on StarCraft II replays that exhibit complex multi-agent behaviors, while still being as easy to use as MNIST and CIFAR10. Specifically, we carefully summarize a window of 255 consecutive game states to create 3.6 million summary images from 60,000 replays, including all relevant metadata such as game outcome and player races. We develop three formats of decreasing complexity: Hyperspectral images that include one channel for every unit type (similar to multispectral geospatial images), RGB images that mimic CIFAR10, and grayscale images that mimic MNIST. We show how this dataset can be used for prototyping spatial reasoning methods. All datasets, code for extraction, and code for dataset loading can be found at https://starcraftdata.davidinouye.com/.

        ----

        ## [2096] Stimulus Verification is a Universal and Effective Sampler in Multi-modal Human Trajectory Prediction

        **Authors**: *Jianhua Sun, Yuxuan Li, Liang Chai, Cewu Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02108](https://doi.org/10.1109/CVPR52729.2023.02108)

        **Abstract**:

        To comprehensively cover the uncertainty of the future, the common practice of multi-modal human trajectory prediction is to first generate a set/distribution of candidate future trajectories and then sample required numbers of trajectories from them as final predictions. Even though a large number of previous researches develop various strong models to predict candidate trajectories, how to effectively sample the final ones has not received much attention yet. In this paper, we propose stimulus verification, serving as a universal and effective sampling process to improve the multi-modal prediction capability, where stimulus refers to the factor in the observation that may affect the future movements such as social interaction and scene context. Stimulus verification introduces a probabilistic model, denoted as stimulus verifier, to verify the coherence between a predicted future trajectory and its corresponding stimulus. By highlighting prediction samples with better stimulus-coherence, stimulus verification ensures sampled trajectories plausible from the stimulus' point of view and therefore aids in better multi-modal prediction performance. We implement stimulus verification on five representative prediction frameworks and conduct exhaustive experiments on three widely-used benchmarks. Superior results demonstrate the effectiveness of our approach.

        ----

        ## [2097] PyPose: A Library for Robot Learning with Physics-based Optimization

        **Authors**: *Chen Wang, Dasong Gao, Kuan Xu, Junyi Geng, Yaoyu Hu, Yuheng Qiu, Bowen Li, Fan Yang, Brady G. Moon, Abhinav Pandey, Aryan, Jiahe Xu, Tianhao Wu, Haonan He, Daning Huang, Zhongqiang Ren, Shibo Zhao, Taimeng Fu, Pranay Reddy, Xiao Lin, Wenshan Wang, Jingnan Shi, Rajat Talak, Kun Cao, Yi Du, Han Wang, Huai Yu, Shanzhao Wang, Siyu Chen, Ananth Kashyap, Rohan Bandaru, Karthik Dantu, Jiajun Wu, Lihua Xie, Luca Carlone, Marco Hutter, Sebastian A. Scherer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02109](https://doi.org/10.1109/CVPR52729.2023.02109)

        **Abstract**:

        Deep learning has had remarkable success in robotic perception, but its data-centric nature suffers when it comes to generalizing to ever-changing environments. By contrast, physics-based optimization generalizes better, but it does not perform as well in complicated tasks due to the lack of high-level semantic information and reliance on manual parametric tuning. To take advantage of these two complementary worlds, we present PyPose: a robotics-oriented, PyTorch-based library that combines deep perceptual models with physics-based optimization. PyPose's architecture is tidy and well-organized, it has an imperative style interface and is efficient and user-friendly, making it easy to integrate into real-world robotic applications. Besides, it supports parallel computing of any order gradients of Lie groups and Lie algebras and 2
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">nd</sup>
-order optimizers, such as trust region methods. Experiments show that PyPose achieves more than 10× speedup in computation compared to the state-of-the-art libraries. To boost future research, we provide concrete examples for several fields of robot learning, including SLAM, planning, control, and inertial navigation.

        ----

        ## [2098] Source-Free Adaptive Gaze Estimation by Uncertainty Reduction

        **Authors**: *Xin Cai, Jiabei Zeng, Shiguang Shan, Xilin Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02110](https://doi.org/10.1109/CVPR52729.2023.02110)

        **Abstract**:

        Gaze estimation across domains has been explored recently because the training data are usually collected under controlled conditions while the trained gaze estimators are used in nature and diverse environments. However, due to privacy and efficiency concerns, simultaneous access to annotated source data and to-be-predicted target data can be challenging. In light of this, we present an unsupervised source-free domain adaptation approach for gaze estimation, which adapts a source-trained gaze estimator to unlabeled target domains without source data. We propose the Uncertainty Reduction Gaze Adaptation (UnReGA) framework, which achieves adaptation by reducing both sample and model uncertainty. Sample uncertainty is mitigated by enhancing image quality and making them gaze-estimation-friendly, whereas model uncertainty is reduced by minimizing prediction variance on the same inputs. Extensive experiments are conducted on six cross-domain tasks, demonstrating the effectiveness of UnReGA and its components. Results show that UnReGA outperforms other state-of-the-art cross-domain gaze estimation methods under both protocols, with and without source data. The code is available at https://github.com/caixin1998/UnReGA.

        ----

        ## [2099] Camouflaged Object Detection with Feature Decomposition and Edge Reconstruction

        **Authors**: *Chunming He, Kai Li, Yachao Zhang, Longxiang Tang, Yulun Zhang, Zhenhua Guo, Xiu Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02111](https://doi.org/10.1109/CVPR52729.2023.02111)

        **Abstract**:

        Camouflaged object detection (COD) aims to address the tough issue of identifying camouflaged objects visually blended into the surrounding backgrounds. COD is a challenging task due to the intrinsic similarity of camouflaged objects with the background, as well as their ambiguous boundaries. Existing approaches to this problem have developed various techniques to mimic the human visual system. Albeit effective in many cases, these methods still struggle when camouflaged objects are so deceptive to the vision system. In this paper, we propose the FEature Decomposition and Edge Reconstruction (FEDER) model for COD. The FEDER model addresses the intrinsic similarity of foreground and background by decomposing the features into different frequency bands using learnable wavelets. It then focuses on the most informative bands to mine subtle cues that differentiate foreground and background. To achieve this, a frequency attention module and a guidance-based feature aggregation module are developed. To combat the ambiguous boundary problem, we propose to learn an auxiliary edge reconstruction task alongside the COD task. We design an ordinary differential equation-inspired edge reconstruction module that generates exact edges. By learning the auxiliary task in conjunction with the COD task, the FEDER model can generate precise prediction maps with accurate object boundaries. Experiments show that our FEDER model significantly outperforms state-of-the-art methods with cheaper computational and memory costs. The code will be available at https://github.com/ChunmingHe/FEDER.

        ----

        ## [2100] MOTRv2: Bootstrapping End-to-End Multi-Object Tracking by Pretrained Object Detectors

        **Authors**: *Yuang Zhang, Tiancai Wang, Xiangyu Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02112](https://doi.org/10.1109/CVPR52729.2023.02112)

        **Abstract**:

        In this paper, we propose MOTRv2, a simple yet effective pipeline to bootstrap end-to-end multi-object tracking with a pretrained object detector. Existing end-to-end methods, e.g. MOTR [43] and TrackFormer [20] are inferior to their tracking-by-detection counterparts mainly due to their poor detection performance. We aim to improve MOTR by elegantly incorporating an extra object detector. We first adopt the anchor formulation of queries and then use an extra object detector to generate proposals as anchors, providing detection prior to MOTR. The simple modification greatly eases the conflict between joint learning detection and association tasks in MOTR. MOTRv2 keeps the query propogation feature and scales well on large-scale benchmarks. MOTRv2 ranks the 1st place (73.4% HOTA on DanceTrack) in the 1st Multiple People Tracking in Group Dance Challenge. Moreover, MOTRv2 reaches state-of-the-art performance on the BDD100K dataset. We hope this simple and effective pipeline can provide some new insights to the end-to-end MOT community. Code is available at https://github.com/megvii-research/MOTRv2.

        ----

        ## [2101] Dynamic Aggregated Network for Gait Recognition

        **Authors**: *Kang Ma, Ying Fu, Dezhi Zheng, Chunshui Cao, Xuecai Hu, Yongzhen Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02114](https://doi.org/10.1109/CVPR52729.2023.02114)

        **Abstract**:

        Gait recognition is beneficial for a variety of applications, including video surveillance, crime scene investigation, and social security, to mention a few. However, gait recognition often suffers from multiple exterior factors in real scenes, such as carrying conditions, wearing overcoats, and diverse viewing angles. Recently, various deep learning-based gait recognition methods have achieved promising results, but they tend to extract one of the salient features using fixed-weighted convolutional networks, do not well consider the relationship within gait features in key regions, and ignore the aggregation of complete motion patterns. In this paper, we propose a new perspective that actual gait features include global motion patterns in multiple key regions, and each global motion pattern is composed of a series of local motion patterns. To this end, we propose a Dynamic Aggregation Network (DANet) to learn more discriminative gait features. Specifically, we create a dynamic attention mechanism between the features of neighboring pixels that not only adaptively focuses on key regions but also generates more expressive local motion patterns. In addition, we develop a selfattention mechanism to select representative local motion patterns and further learn robust global motion patterns. Extensive experiments on three popular public gait datasets, i.e., CASIA-B, OUMVLP, and Gait3D, demonstrate that the proposed method can provide substantial improvements over the current state-of-the-art methods.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>

        ----

        ## [2102] Feature Representation Learning with Adaptive Displacement Generation and Transformer Fusion for Micro-Expression Recognition

        **Authors**: *Zhijun Zhai, Jianhui Zhao, Chengjiang Long, Wenju Xu, Shuangjiang He, Huijuan Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02115](https://doi.org/10.1109/CVPR52729.2023.02115)

        **Abstract**:

        Micro-expressions are spontaneous, rapid and subtle facial movements that can neither be forged nor suppressed. They are very important nonverbal communication clues, but are transient and of low intensity thus difficult to recognize. Recently deep learning based methods have been developed for micro-expression (ME) recognition using feature extraction and fusion techniques, however, targeted feature learning and efficient feature fusion still lack further study according to the ME characteristics. To address these issues, we propose a novel framework Feature Representation Learning with adaptive Displacement Generation and Transformer fusion (FRL-DGT), in which a convolutional Displacement Generation Module (DGM) with self-supervised learning is used to extract dynamic features from onset/apex frames targeted to the subsequent ME recognition task, and a well-designed Transformer Fusion mechanism composed of three Transformer-based fusion modules (local, global fusions based on AU regions and full-face fusion) is applied to extract the multi-level informative features after DGM for the final ME prediction. The extensive experiments with solid leave-one-subject-out (LOSO) evaluation results have demonstrated the superiority of our proposed FRL-DGT to state-of-the-art methods.

        ----

        ## [2103] MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation

        **Authors**: *Bowen Zhang, Chenyang Qi, Pan Zhang, Bo Zhang, HsiangTao Wu, Dong Chen, Qifeng Chen, Yong Wang, Fang Wen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02116](https://doi.org/10.1109/CVPR52729.2023.02116)

        **Abstract**:

        In this work, we propose an ID-preserving talking head generation framework, which advances previous methods in two aspects. First, as opposed to interpolating from sparse flow, we claim that dense landmarks are crucial to achieving accurate geometry-aware flow fields. Second, inspired by face-swapping methods, we adaptively fuse the source identity during synthesis, so that the network better preserves the key characteristics of the image portrait. Although the proposed model surpasses prior generation fidelity on established benchmarks, personalized fine-tuning is still needed to further make the talking head generation qualified for real usage. However, this process is rather computationally demanding that is unaffordable to standard users. To alleviate this, we propose a fast adaptation model using a metalearning approach. The learned model can be adapted to a high-quality personalized model as fast as 30 seconds. Last but not least, a spatial-temporal enhancement module is proposed to improve the fine details while ensuring temporal coherency. Extensive experiments prove the significant superiority of our approach over the state of the arts in both one-shot and personalized settings.

        ----

        ## [2104] FLAG3D: A 3D Fitness Activity Dataset with Language Instruction

        **Authors**: *Yansong Tang, Jinpeng Liu, Aoyang Liu, Bin Yang, Wenxun Dai, Yongming Rao, Jiwen Lu, Jie Zhou, Xiu Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02117](https://doi.org/10.1109/CVPR52729.2023.02117)

        **Abstract**:

        With the continuously thriving popularity around the world, fitness activity analytic has become an emerging research topic in computer vision. While a variety of new tasks and algorithms have been proposed recently, there are growing hunger for data resources involved in high-quality data, fine-grained labels, and diverse environments. In this paper, we present FLAG3D, a large-scale 3D fitness activity dataset with language instruction containing 180K sequences of 60 categories. FLAG3D features the following three aspects: 1) accurate and dense 3D human pose captured from advanced MoCap system to handle the complex activity and large movement, 2) detailed and professional language instruction to describe how to perform a specific activity, 3) versatile video resources from a high-tech MoCap system, rendering software, and cost-effective smartphones in natural environments. Extensive experiments and in-depth analysis show that FLAG3D contributes great research value for various challenges, such as cross-domain human action recognition, dynamic human mesh recovery, and language-guided human action generation. Our dataset and source code are publicly available at https://andytang15.github.io/FLAG3D.

        ----

        ## [2105] TranSG: Transformer-Based Skeleton Graph Prototype Contrastive Learning with Structure-Trajectory Prompted Reconstruction for Person Re-Identification

        **Authors**: *Haocong Rao, Chunyan Miao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02118](https://doi.org/10.1109/CVPR52729.2023.02118)

        **Abstract**:

        Person re-identification (re-ID) via 3D skeleton data is an emerging topic with prominent advantages. Existing methods usually design skeleton descriptors with raw body joints or perform skeleton sequence representation learning. However, they typically cannot concurrently model different body-component relations, and rarely explore useful semantics from fine-grained representations of body joints. In this paper, we propose a generic Transformer-based Skeleton Graph prototype contrastive learning (TranSG) approach with structure-trajectory prompted reconstruction to fully capture skeletal relations and valuable spatial-temporal semantics from skeleton graphs for person re-ID. Specifically, we first devise the Skeleton Graph Transformer (SGT) to simultaneously learn body and motion relations within skeleton graphs, so as to aggregate key correlative node features into graph representations. Then, we propose the Graph Prototype Contrastive learning (GPC) to mine the most typical graph features (graph prototypes) of each identity, and contrast the inherent similarity between graph representations and different prototypes from both skeleton and sequence levels to learn discriminative graph representations. Last, a graph Structure-Trajectory Prompted Reconstruction (STPR) mechanism is proposed to exploit the spatial and temporal contexts of graph nodes to prompt skeleton graph reconstruction, which facilitates capturing more valuable patterns and graph semantics for person re-ID. Empirical evaluations demonstrate that TranSG significantly outperforms existing state-of-the-art methods. We further show its generality under different graph modeling, RGB-estimated skeletons, and unsupervised scenarios. Our codes are available at https://github.com/Kali-Hac/TranSG.

        ----

        ## [2106] NeMo: 3D Neural Motion Fields from Multiple Video Instances of the Same Action

        **Authors**: *Kuan-Chieh Wang, Zhenzhen Weng, Maria Xenochristou, João Pedro Araújo, Jeffrey Gu, C. Karen Liu, Serena Yeung*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02119](https://doi.org/10.1109/CVPR52729.2023.02119)

        **Abstract**:

        The task of reconstructing 3D human motion has wide-ranging applications. The gold standard Motion capture (MoCap) systems are accurate but inaccessible to the general public due to their cost, hardware, and space constraints. In contrast, monocular human mesh recovery (HMR) methods are much more accessible than MoCap as they take single-view videos as inputs. Replacing the multi-view MoCap systems with a monocular HMR method would break the current barriers to collecting accurate 3D motion thus making exciting applications like motion analysis and motion-driven animation accessible to the general public. However, the performance of existing HMR methods degrades when the video contains challenging and dynamic motion that is not in existing MoCap datasets used for training. This reduces its appeal as dynamic motion is frequently the target in 3D motion recovery in the aforementioned applications. Our study aims to bridge the gap between monocular HMR and multi-view MoCap systems by leveraging information shared across multiple video instances of the same action. We introduce the Neural Motion (NeMo) field. It is optimized to represent the underlying 3D motions across a set of videos of the same action. Empirically, we show that NeMo can recover 3D motion in sports using videos from the Penn Action dataset, where NeMo outperforms existing HMR methods in terms of 2D keypoint detection. To further validate NeMo using 3D metrics, we collected a small MoCap dataset mimicking actions in Penn Action, and show that NeMo achieves better 3D reconstruction compared to various baselines.

        ----

        ## [2107] Unsupervised Space-Time Network for Temporally-Consistent Segmentation of Multiple Motions

        **Authors**: *Etienne Meunier, Patrick Bouthemy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02120](https://doi.org/10.1109/CVPR52729.2023.02120)

        **Abstract**:

        Motion segmentation is one of the main tasks in computer vision and is relevant for many applications. The optical flow (OF) is the input generally used to segment every frame of a video sequence into regions of coherent motion. Temporal consistency is a key feature of motion segmentation, but it is often neglected. In this paper, we propose an original unsupervised spatiotemporal framework for motion segmentation from optical flow that fully investigates the temporal dimension of the problem. More specifically, we have defined a 3D network for multiple motion segmentation that takes as input a sub-volume of successive optical flows and delivers accordingly a sub-volume of coherent segmentation maps. Our network is trained in a fully unsupervised way, and the loss function combines a flow reconstruction term involving spatio-temporal parametric motion models, and a regularization term enforcing temporal consistency on the masks. We have specified an easy temporal linkage of the predicted segments. Besides, we have proposed a flexible and efficient way of coding U-nets. We report experiments on several VOS benchmarks with convincing quantitative results, while not using appearance and not training with any ground-truth data. We also highlight through visual results the distinctive contribution of the short- and long-term temporal consistency brought by our OF segmentation method.

        ----

        ## [2108] Deep Polarization Reconstruction with PDAVIS Events

        **Authors**: *Haiyang Mei, Zuowen Wang, Xin Yang, Xiaopeng Wei, Tobi Delbruck*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02121](https://doi.org/10.1109/CVPR52729.2023.02121)

        **Abstract**:

        The polarization event camera PDAVIS is a novel bio-inspired neuromorphic vision sensor that reports both conventional polarization frames and asynchronous, continuously per-pixel polarization brightness changes (polarization events) with fast temporal resolution and large dynamic range. A deep neural network method (Polarization FireNet) was previously developed to reconstruct the polarization angle and degree from polarization events for bridging the gap between the polarization event camera and mainstream computer vision. However, Polarization FireNet applies a network pretrained for normal event-based frame reconstruction independently on each of four channels of polarization events from four linear polarization angles, which ignores the correlations between channels and inevitably introduces content inconsistency between the four reconstructed frames, resulting in unsatisfactory polarization reconstruction performance. In this work, we strive to train an effective, yet efficient, DNN model that directly outputs polarization from the input raw polarization events. To this end, we constructed the first large-scale event-to-polarization dataset, which we subsequently employed to train our events-to-polarization network E2P. E2P extracts rich polarization patterns from input polarization events and enhances features through cross-modality context integration. We demonstrate that E2P outperforms Polarization FireNet by a significant margin with no additional computing cost. Experimental results also show that E2P produces more accurate measurement of polarization than the PDAVIS frames in challenging fast and high dynamic range scenes. Code and data are publicly available at: https://github.com/SensorsINI/e2p.

        ----

        ## [2109] Range-nullspace Video Frame Interpolation with Focalized Motion Estimation

        **Authors**: *Zhiyang Yu, Yu Zhang, Dongqing Zou, Xijun Chen, Jimmy S. Ren, Shunqing Ren*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02122](https://doi.org/10.1109/CVPR52729.2023.02122)

        **Abstract**:

        Continuous-time video frame interpolation is a fundamental technique in computer vision for its flexibility in synthesizing motion trajectories and novel video frames at arbitrary intermediate time steps. Yet, how to infer accurate intermediate motion and synthesize high-quality video frames are two critical challenges. In this paper, we present a novel VFI framework with improved treatment for these challenges. To address the former, we propose focalized trajectory fitting, which performs confidence-aware motion trajectory estimation by learning to pay focus to reliable optical flow candidates while suppressing the outliers. The second is range-nullspace synthesis, a novel frame renderer cast as solving an ill-posed problem addressed by learning decoupled components in orthogonal subspaces. The proposed framework sets new records on 7 of 10 public VFI benchmarks.

        ----

        ## [2110] Exploring Motion Ambiguity and Alignment for High-Quality Video Frame Interpolation

        **Authors**: *Kun Zhou, Wenbo Li, Xiaoguang Han, Jiangbo Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02123](https://doi.org/10.1109/CVPR52729.2023.02123)

        **Abstract**:

        For video frame interpolation (VFI), existing deep-learning-based approaches strongly rely on the ground-truth (GT) intermediate frames, which sometimes ignore the non-unique nature of motion judging from the given adjacent frames. As a result, these methods tend to produce averaged solutions that are not clear enough. To alleviate this issue, we propose to relax the requirement of reconstructing an intermediate frame as close to the GT as possible. Towards this end, we develop a texture consistency loss (TCL) upon the assumption that the interpolated content should maintain similar structures with their counterparts in the given frames. Predictions satisfying this constraint are encouraged, though they may differ from the predefined GT. Without the bells and whistles, our plug-and-play TCL is capable of improving the performance of existing VFI frameworks consistently. On the other hand, previous methods usually adopt the cost volume or correlation map to achieve more accurate image or feature warping. However, the O (N
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
) (N refers to the pixel count) computational complexity makes it infeasible for high-resolution cases. In this work, we design a simple, efficient O (N) yet powerful guided cross-scale pyramid alignment (GCSPA) module, where multi-scale information is highly exploited. Extensive experiments justify the efficiency and effectiveness of the proposed strategy.

        ----

        ## [2111] 1000 FPS HDR Video with a Spike-RGB Hybrid Camera

        **Authors**: *Yakun Chang, Chu Zhou, Yuchen Hong, Liwen Hu, Chao Xu, Tiejun Huang, Boxin Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02124](https://doi.org/10.1109/CVPR52729.2023.02124)

        **Abstract**:

        Capturing high frame rate and high dynamic range (HFR&HDR) color videos in high-speed scenes with conventional frame-based cameras is very challenging. The increasing frame rate is usually guaranteed by using shorter exposure time so that the captured video is severely interfered by noise. Alternating exposures can alleviate the noise issue but sacrifice frame rate due to involving long-exposure frames. The neuromorphic spiking camera records high-speed scenes of high dynamic range without colors using a completely different sensing mechanism and visual representation. We introduce a hybrid camera system composed of a spiking and an alternating-exposure RGB camera to capture HFR&HDR scenes with high fidelity. Our insight is to bring each camera's superiority into full play. The spike frames, with accurate fast motion information encoded, are firstly reconstructed for motion representation, from which the spike-based optical flows guide the recovery of missing temporal information for long-exposure RGB images while retaining their reliable color appearances. With the strong temporal constraint estimated from spike trains, both missing and distorted colors cross RGB frames are recovered to generate time-consistent and HFR color frames. We collect a new Spike-RGB dataset that contains 300 sequences of synthetic data and 20 groups of real-world data to demonstrate 1000 FPS HDR videos outperforming HDR video reconstruction methods and commercial high-speed cameras.

        ----

        ## [2112] Deep Discriminative Spatial and Temporal Network for Efficient Video Deblurring

        **Authors**: *Jinshan Pan, Boming Xu, Jiangxin Dong, Jianjun Ge, Jinhui Tang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02125](https://doi.org/10.1109/CVPR52729.2023.02125)

        **Abstract**:

        How to effectively explore spatial and temporal information is important for video deblurring. In contrast to existing methods that directly align adjacent frames without discrimination, we develop a deep discriminative spatial and temporal network to facilitate the spatial and temporal feature exploration for better video deblurring. We first develop a channel-wise gated dynamic network to adaptively explore the spatial information. As adjacent frames usually contain different contents, directly stacking features of adjacent frames without discrimination may affect the latent clear frame restoration. Therefore, we develop a simple yet effective discriminative temporal feature fusion module to obtain useful temporal features for latent frame restoration. Moreover, to utilize the information from long-range frames, we develop a wavelet-based feature propagation method that takes the discriminative temporal feature fusion module as the basic unit to effectively propagate main structures from long-range frames for better video deblurring. We show that the proposed method does not require additional alignment methods and performs favorably against state-of-the-art ones on benchmark datasets in terms of accuracy and model complexity.

        ----

        ## [2113] Gated Multi-Resolution Transfer Network for Burst Restoration and Enhancement

        **Authors**: *Nancy Mehta, Akshay Dudhane, Subrahmanyam Murala, Syed Waqas Zamir, Salman H. Khan, Fahad Shahbaz Khan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02126](https://doi.org/10.1109/CVPR52729.2023.02126)

        **Abstract**:

        Burst image processing is becoming increasingly popular in recent years. However, it is a challenging task since individual burst images undergo multiple degradations and often have mutual misalignments resulting in ghosting and zipper artifacts. Existing burst restoration methods usually do not consider the mutual correlation and non-local contextual information among burst frames, which tends to limit these approaches in challenging cases. Another key challenge lies in the robust up-sampling of burst frames. The existing up-sampling methods cannot effectively utilize the advantages of single-stage and progressive up-sampling strategies with conventional and/or recent up-samplers at the same time. To address these challenges, we propose a novel Gated Multi-Resolution Transfer Network (GMTNet) to reconstruct a spatially precise high-quality image from a burst of low-quality raw images. GMT-Net consists of three modules optimized for burst processing tasks: Multi-scale Burst Feature Alignment (MBFA) for feature denoising and alignment, Transposed-Attention Feature Merging (TAFM) for multi-frame feature aggregation, and Resolution Transfer Feature Up-sampler (RTFU) to up-scale merged features and construct a high-quality output image. Detailed experimental analysis on five datasets validate our approach and sets a state-of-the-art for burst super-resolution, burst denoising, and low-light burst enhancement. Our codes and models are available at https://github.com/nanmehta/GMTNet.

        ----

        ## [2114] A Unified HDR Imaging Method with Pixel and Patch Level

        **Authors**: *Qingsen Yan, Weiye Chen, Song Zhang, Yu Zhu, Jinqiu Sun, Yanning Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02127](https://doi.org/10.1109/CVPR52729.2023.02127)

        **Abstract**:

        Mapping Low Dynamic Range (LDR) images with different exposures to High Dynamic Range (HDR) remains nontrivial and challenging on dynamic scenes due to ghosting caused by object motion or camera jitting. With the success of Deep Neural Networks (DNNs), several DNNs-based methods have been proposed to alleviate ghosting, they cannot generate approving results when motion and saturation occur. To generate visually pleasing HDR images in various cases, we propose a hybrid HDR deghosting network, called HyHDRNet, to learn the complicated relationship between reference and non-reference images. The proposed HyHDRNet consists of a content alignment subnetwork and a Transformer-based fusion subnetwork. Specifically, to effectively avoid ghosting from the source, the content alignment subnetwork uses patch aggregation and ghost attention to integrate similar content from other non-reference images with patch level and suppress undesired components with pixel level. To achieve mutual guidance between patch-level and pixel-level, we leverage a gating module to sufficiently swap useful information both in ghosted and saturated regions. Furthermore, to obtain a high-quality HDR image, the Transformer-based fusion subnetwork uses a Residual Deformable Transformer Block (RDTB) to adaptively merge information for different exposed regions. We examined the proposed method on four widely used public HDR image deghosting datasets. Experiments demonstrate that HyHDRNet outperforms state-of-the-art methods both quantitatively and qualitatively, achieving appealing HDR visualization with unified textures and colors.

        ----

        ## [2115] BiasBed - Rigorous Texture Bias Evaluation

        **Authors**: *Nikolai Kalischek, Rodrigo Caye Daudt, Torben Peters, Reinhard Furrer, Jan D. Wegner, Konrad Schindler*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02128](https://doi.org/10.1109/CVPR52729.2023.02128)

        **Abstract**:

        The well-documented presence of texture bias in modern convolutional neural networks has led to a plethora of algorithms that promote an emphasis on shape cues, often to support generalization to new domains. Yet, common datasets, benchmarks and general model selection strategies are missing, and there is no agreed, rigorous evaluation protocol. In this paper, we investigate difficulties and limitations when training networks with reduced texture bias. In particular, we also show that proper evaluation and meaningful comparisons between methods are not trivial. We introduce BiasBed, a testbed for texture- and style-biased training, including multiple datasets and a range of existing algorithms. It comes with an extensive evaluation protocol that includes rigorous hypothesis testing to gauge the significance of the results, despite the considerable training instability of some style bias methods. Our extensive experiments, shed new light on the need for careful, statistically founded evaluation protocols for style bias (and beyond). E.g., we find that some algorithms proposed in the literature do not significantly mitigate the impact of style bias at all. With the release of BiasBed, we hope to foster a common understanding of consistent and meaningful comparisons, and consequently faster progress towards learning methods free of texture bias. Code is available at https://github.com/D1noFuzi/BiasBed

        ----

        ## [2116] Learning a Practical SDR-to-HDRTV Up-conversion using New Dataset and Degradation Models

        **Authors**: *Cheng Guo, Leidong Fan, Ziyu Xue, Xiuhua Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02129](https://doi.org/10.1109/CVPR52729.2023.02129)

        **Abstract**:

        In media industry, the demand of SDR-to-HDRTV upconversion arises when users possess HDR-WCG (high dynamic range-wide color gamut) TVs while most off-the-shelf footage is still in SDR (standard dynamic range). The research community has started tackling this low-level vision task by learning-based approaches. When applied to real SDR, yet, current methods tend to produce dim and desaturated result, making nearly no improvement on viewing experience. Different from other network-oriented methods, we attribute such deficiency to training set (HDR-SDR pair). Consequently, we propose new HDRTV dataset (dubbed HDRTV4K) and new HDR-to-SDR degradation models. Then, it's used to train a luminance-segmented network (LSN) consisting of a global mapping trunk, and two Transformer branches on bright and dark luminance range. We also update assessment criteria by tailored metrics and subjective experiment. Finally, ablation studies are conducted to prove the effectiveness. Our work is available at: https://github.com/AndreGuo/HDRTVDM

        ----

        ## [2117] Learning a Deep Color Difference Metric for Photographic Images

        **Authors**: *Haoyu Chen, Zhihua Wang, Yang Yang, Qilin Sun, Kede Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02130](https://doi.org/10.1109/CVPR52729.2023.02130)

        **Abstract**:

        Most well-established and widely used color difference (CD) metrics are handcrafted and subjectcalibrated against uniformly colored patches, which do not generalize well to photographic images characterized by natural scene complexities. Constructing CD formulae for photo-graphic images is still an active research topic in imaging/illumination, vision science, and color science communities. In this paper, we aim to learn a deep CD metric for photographic images with four desirable properties. First, it well aligns with the observations in vision science that color and form are linked inextricably in visual cortical processing. Second, it is a proper metric in the mathematical sense. Third, it computes accurate CDs between photographic images, differing mainly in color appearances. Fourth, it is robust to mild geometric distortions (e.g., translation or due to parallax), which are often present in photographic images of the same scene captured by different digital cameras. We show that all four properties can be satisfied at once by learning a multi-scale autoregressive normalizing flow for feature transform, followed by the Euclidean distance which is linearly proportional to the human perceptual CD. Quantitative and qualitative experiments on the large-scale SPCD dataset demonstrate the promise of the learned CD metric. Source code is available at https://github.com/haoychen3/CD-Flow.

        ----

        ## [2118] Learning a Simple Low-Light Image Enhancer from Paired Low-Light Instances

        **Authors**: *Zhenqi Fu, Yan Yang, Xiaotong Tu, Yue Huang, Xinghao Ding, Kai-Kuang Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02131](https://doi.org/10.1109/CVPR52729.2023.02131)

        **Abstract**:

        Low-light Image Enhancement (LIE) aims at improving contrast and restoring details for images captured in lowlight conditions. Most of the previous LIE algorithms adjust illumination using a single input image with several handcrafted priors. Those solutions, however, often fail in revealing image details due to the limited information in a single image and the poor adaptability of handcrafted priors. To this end, we propose PairLIE, an unsupervised approach that learns adaptive priors from low-light image pairs. First, the network is expected to generate the same clean images as the two inputs share the same image content. To achieve this, we impose the network with the Retinex theory and make the two reflectance components consistent. Second, to assist the Retinex decomposition, we propose to remove inappropriate features in the raw image with a simple self-supervised mechanism. Extensive experiments on public datasets show that the proposed PairLIE achieves comparable performance against the state-of-the-art approaches with a simpler network and fewer handcrafted priors. Code is available at: https://github.com/zhenqifu/PairLIE.

        ----

        ## [2119] Residual Degradation Learning Unfolding Framework with Mixing Priors Across Spectral and Spatial for Compressive Spectral Imaging

        **Authors**: *Yubo Dong, Dahua Gao, Tian Qiu, Yuyan Li, Minxi Yang, Guangming Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02132](https://doi.org/10.1109/CVPR52729.2023.02132)

        **Abstract**:

        To acquire a snapshot spectral image, coded aperture snapshot spectral imaging (CASSI) is proposed. A core problem of the CASSI system is to recover the reliable and fine underlying 3D spectral cube from the 2D measurement. By alternately solving a data subproblem and a prior subproblem, deep unfolding methods achieve good performance. However, in the data subproblem, the used sensing matrix is ill-suited for the real degradation process due to the device errors caused by phase aberration, distortion; in the prior subproblem, it is important to design a suitable model to jointly exploit both spatial and spectral priors. In this paper, we propose a Residual Degradation Learning Unfolding Framework (RDLUF), which bridges the gap between the sensing matrix and the degradation process. Moreover, a MixS
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 Transformer is designed via mixing priors across spectral and spatial to strengthen the spectral-spatial representation capability. Finally, plugging the MixS
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 Transformer into the RDLUF leads to an end-to-end trainable neural network RDLUF-MixS2. Experimental results establish the superior performance of the proposed method over existing ones. Code is available: https://github.com/ShawnDong98/RDLUF_MixS2

        ----

        ## [2120] Toward Stable, Interpretable, and Lightweight Hyperspectral Super-Resolution

        **Authors**: *Wen-jin Guo, Weiying Xie, Kai Jiang, Yunsong Li, Jie Lei, Leyuan Fang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02133](https://doi.org/10.1109/CVPR52729.2023.02133)

        **Abstract**:

        For real applications, existing HSI-SR methods are not only limited to unstable performance under unknown scenarios but also suffer from high computation consumption. In this paper, we develop a new coordination optimization framework for stable, interpretable, and lightweight HSI-SR. Specifically, we create a positive cycle between fusion and degradation estimation under a new probabilistic framework. The estimated degradation is applied to fusion as guidance for a degradation-aware HSI-SR. Under the framework, we establish an explicit degradation estimation method to tackle the indeterminacy and unstable performance caused by the black-box simulation in previous methods. Considering the interpretability in fusion, we integrate spectral mixing prior into the fusion process, which can be easily realized by a tiny autoencoder, leading to a dramatic release of the computation burden. Based on the spectral mixing prior, we then develop a partial fine-tune strategy to reduce the computation cost further. Comprehensive experiments demonstrate the superiority of our method against the state-of-the-arts under synthetic and real datasets. For instance, we achieve a 2.3 dB promotion on PSNR with 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$120\times$</tex>
 model size reduction and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$4300 \times$</tex>
 FLOPs reduction under the CAVE dataset. Code is available in https://github.com/WenjinGuo/DAEM.

        ----

        ## [2121] RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors

        **Authors**: *Ruiqi Wu, Zheng-Peng Duan, Chun-Le Guo, Zhi Chai, Chongyi Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02134](https://doi.org/10.1109/CVPR52729.2023.02134)

        **Abstract**:

        Existing dehazing approaches struggle to process real-world hazy images owing to the lack of paired real data and robust priors. In this work, we present a new paradigm for real image dehazing from the perspectives of synthesizing more realistic hazy data and introducing more robust priors into the network. Specifically, (1) instead of adopting the de facto physical scattering model, we rethink the degradation of real hazy images and propose a phenomenological pipeline considering diverse degradation types. (2) We propose a Real Image Dehazing network via high-quality Codebook Priors (RIDCP). Firstly, a VQGAN is pre-trained on a large-scale high-quality dataset to obtain the discrete codebook, encapsulating high-quality priors (HQPs). After replacing the negative effects brought by haze with HQPs, the decoder equipped with a novel normalized feature alignment module can effectively utilize high-quality features and produce clean results. However, although our degradation pipeline drastically mitigates the domain gap between synthetic and real data, it is still intractable to avoid it, which challenges HQPs matching in the wild. Thus, we recalculate the distance when matching the features to the HQPs by a controllable matching operation, which facilitates finding better counterparts. We provide a recommendation to control the matching based on an explainable solution. Users can also flexibly adjust the enhancement degree as per their preference. Extensive experiments verify the effectiveness of our data synthesis pipeline and the superior performance of RIDCP in real image dehazing. Code and data are released at https://rqwu.github.io/projects/RIDCP.

        ----

        ## [2122] Robust Unsupervised StyleGAN Image Restoration

        **Authors**: *Yohan Poirier-Ginter, Jean-François Lalonde*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02135](https://doi.org/10.1109/CVPR52729.2023.02135)

        **Abstract**:

        GAN-based image restoration inverts the generative process to repair images corrupted by known degradations. Existing unsupervised methods must be carefully tuned for each task and degradation level. In this work, we make StyleGAN image restoration robust: a single set of hyperparameters works across a wide range of degradation levels. This makes it possible to handle combinations of several degradations, without the need to retune. Our proposed approach relies on a 3-phase progressive latent space extension and a conservative optimizer, which avoids the need for any additional regularization terms. Extensive experiments demonstrate robustness on inpainting, upsampling, denoising, and deartifacting at varying degradations levels, outperforming other StyleGAN-based inversion techniques. Our approach also favorably compares to diffusion-based restoration by yielding much more realistic inversion results. Code is available at the above URL.

        ----

        ## [2123] Quality-aware Pretrained Models for Blind Image Quality Assessment

        **Authors**: *Kai Zhao, Kun Yuan, Ming Sun, Mading Li, Xing Wen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02136](https://doi.org/10.1109/CVPR52729.2023.02136)

        **Abstract**:

        Blind image quality assessment (BIQA) aims to auto-matically evaluate the perceived quality of a single image, whose performance has been improved by deep learning-based methods in recent years. However, the paucity of labeled data somewhat restrains deep learning-based BIQA methods from unleashing their full potential. In this paper, we propose to solve the problem by a pretext task customized for BIQA in a self-supervised learning manner, which enables learning representations from orders of mag-nitude more data. To constrain the learning process, we propose a quality-aware contrastive loss based on a simple assumption: the quality of patches from a distorted image should be similar, but vary from patches from the same image with different degradations and patches from different images. Further, we improve the existing degradation process and form a degradation space with the size of roughly 2 × 10
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">7</sup>
. After pretrained on ImageNet using our method, models are more sensitive to image quality and perform significantly better on downstream BIQA tasks. Experimental results show that our method obtains remarkable improvements on popular BIQA datasets.

        ----

        ## [2124] Learning to Exploit the Sequence-Specific Prior Knowledge for Image Processing Pipelines Optimization

        **Authors**: *Haina Qin, Longfei Han, Weihua Xiong, Juan Wang, Wentao Ma, Bing Li, Weiming Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02137](https://doi.org/10.1109/CVPR52729.2023.02137)

        **Abstract**:

        The hardware image signal processing (ISP) pipeline is the intermediate layer between the imaging sensor and the downstream application, processing the sensor signal into an RGB image. The ISP is less programmable and consists of a series of processing modules. Each processing module handles a subtask and contains a set of tunable hyperparameters. A large number of hyperparameters form a complex mapping with the ISP output. The industry typically relies on manual and time-consuming hyperparameter tuning by image experts, biased towards human perception. Recently, several automatic ISP hyperparameter optimization methods using downstream evaluation metrics come into sight. However, existing methods for ISP tuning treat the high-dimensional parameter space as a global space for optimization and prediction all at once without inducing the structure knowledge of ISP. To this end, we propose a sequential ISP hyperparameter prediction framework that utilizes the sequential relationship within ISP modules and the similarity among parameters to guide the model sequence process. We validate the proposed method on object detection, image segmentation, and image quality tasks.

        ----

        ## [2125] Multi-Realism Image Compression with a Conditional Generator

        **Authors**: *Eirikur Agustsson, David Minnen, George Toderici, Fabian Mentzer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02138](https://doi.org/10.1109/CVPR52729.2023.02138)

        **Abstract**:

        By optimizing the rate-distortion-realism trade-off, generative compression approaches produce detailed, realistic images, even at low bit rates, instead of the blurry reconstructions produced by rate-distortion optimized models. However, previous methods do not explicitly control how much detail is synthesized, which results in a common criticism of these methods: users might be worried that a misleading reconstruction far from the input image is generated. In this work, we alleviate these concerns by training a decoder that can bridge the two regimes and navigate the distortion-realism tradeoff. From a single compressed representation, the receiver can decide to either reconstruct a low mean squared error reconstruction that is close to the input, a realistic reconstruction with high perceptual quality, or anything in between. With our method, we set a new state-of-the-art in distortion-realism, pushing the frontier of achievable distortion-realism pairs, i.e., our method achieves better distortions at high realism and better realism at low distortion than ever before.

        ----

        ## [2126] RGB No More: Minimally-Decoded JPEG Vision Transformers

        **Authors**: *Jeongsoo Park, Justin Johnson*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02139](https://doi.org/10.1109/CVPR52729.2023.02139)

        **Abstract**:

        Most neural networks for computer vision are designed to infer using RGB images. However, these RGB images are commonly encoded in JPEG before saving to disk; decoding them imposes an unavoidable overhead for RGB networks. Instead, our work focuses on training Vision Transformers (ViT) directly from the encoded features of JPEG. This way, we can avoid most of the decoding overhead, accelerating data load. Existing works have studied this aspect but they focus on CNNs. Due to how these encoded features are structured, CNNs require heavy modification to their architecture to accept such data. Here, we show that this is not the case for ViTs. In addition, we tackle data augmentation directly on these encoded features, which to our knowledge, has not been explored in-depth for training in this setting. With these two improvements - ViT and data augmentation - we show that our ViT-Ti model achieves up to 39.2% faster training and 17.9% faster inference with no accuracy loss compared to the RGB counterpart.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://github.com/JeongsooP/RGB-no-more

        ----

        ## [2127] Kernel Aware Resampler

        **Authors**: *Michael Bernasconi, Abdelaziz Djelouah, Farnood Salehi, Markus H. Gross, Christopher Schroers*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02140](https://doi.org/10.1109/CVPR52729.2023.02140)

        **Abstract**:

        Deep learning based methods for super-resolution have become state-of-the-art and outperform traditional approaches by a significant margin. From the initial models designed for fixed integer scaling factors (e.g. 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\times 2$</tex>
 or 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\times 4)$</tex>
), efforts were made to explore different directions such as modeling blur kernels or addressing non-integer scaling factors. However, existing works do not provide a sound framework to handle them jointly. In this paper we propose a framework for generic image resampling that not only addresses all the above mentioned issues but extends the sets of possible transforms from upscaling to generic transforms. A key aspect to unlock these capabilities is the faithful modeling of image warping and changes of the sampling rate during the training data preparation. This allows a localized representation of the implicit image degradation that takes into account the reconstruction kernel, the local geometric distortion and the anti-aliasing kernel. Using this spatially variant degradation map as conditioning for our resampling model, we can address with the same model both global transformations, such as upscaling or rotation, and locally varying transformations such lens distortion or undistortion. Another important contribution is the automatic estimation of the degradation map in this more complex resampling setting (i.e. blind image resampling). Fi-nally, we show that state-of-the-art results can be achieved by predicting kernels to apply on the input image instead of direct color prediction. This renders our model applicable for different types of data not seen during the training such as normals.

        ----

        ## [2128] Spatial-Frequency Mutual Learning for Face Super-Resolution

        **Authors**: *Chenyang Wang, Junjun Jiang, Zhiwei Zhong, Xianming Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02141](https://doi.org/10.1109/CVPR52729.2023.02141)

        **Abstract**:

        Face super-resolution (FSR) aims to reconstruct high-resolution (HR) face images from the low-resolution (LR) ones. With the advent of deep learning, the FSR technique has achieved significant breakthroughs. However, existing FSR methods either have a fixed receptive field or fail to maintain facial structure, limiting the FSRperformance. To circumvent this problem, Fourier transform is introduced, which can capture global facial structure information and achieve image-size receptive field. Relying on the Fourier transform, we devise a spatial-frequency mutual network (SFMNet) for FSR, which is the first FSR method to explore the correlations between spatial and frequency domains as far as we know. To be specific, our SFMNet is a two-branch network equipped with a spatial branch and a frequency branch. Benefiting from the property of Fourier transform, the frequency branch can achieve image-size receptive field and capture global dependency while the spatial branch can extract local dependency. Considering that these dependencies are complementary and both favorable for FSR, we further develop a frequency-spatial interaction block (FSIB) which mutually amalgamates the complementary spatial and frequency information to enhance the capability of the model. Quantitative and qualitative experimental results show that the proposed method out-performs state-of-the-art FSR methods in recovering face images. The implementation and model will be released at https://github.com/wcy-cs/SFMNet.

        ----

        ## [2129] Activating More Pixels in Image Super-Resolution Transformer

        **Authors**: *Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, Chao Dong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02142](https://doi.org/10.1109/CVPR52729.2023.02142)

        **Abstract**:

        Transformer-based methods have shown impressive performance in low-level vision tasks, such as image super-resolution. However, we find that these networks can only utilize a limited spatial range of input information through attribution analysis. This implies that the potential of Transformer is still not fully exploited in existing networks. In order to activate more input pixels for better reconstruction, we propose a novel Hybrid Attention Transformer (HAT). It combines both channel attention and window-based self-attention schemes, thus making use of their complementary advantages of being able to utilize global statistics and strong local fitting capability. Moreover, to better aggregate the cross-window information, we introduce an overlapping cross-attention module to enhance the interaction between neighboring window features. In the training stage, we additionally adopt a same-task pre-training strategy to exploit the potential of the model for further improvement. Extensive experiments show the effectiveness of the proposed modules, and we further scale up the model to demonstrate that the performance of this task can be greatly improved. Our overall method significantly outperforms the state-of-the-art methods by more than 1dB.

        ----

        ## [2130] Omni Aggregation Networks for Lightweight Image Super-Resolution

        **Authors**: *Hang Wang, Xuanhong Chen, Bingbing Ni, Yutian Liu, Jinfan Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02143](https://doi.org/10.1109/CVPR52729.2023.02143)

        **Abstract**:

        While lightweight ViT framework has made tremendous progress in image super-resolution, its uni-dimensional self-attention modeling, as well as homogeneous aggregation scheme, limit its effective receptive field (ERF) to include more comprehensive interactions from both spatial and channel dimensions. To tackle these drawbacks, this work proposes two enhanced components under a new Omni-SR architecture. First, an Omni Self-Attention (OSA) block is proposed based on dense interaction principle, which can simultaneously model pixel-interaction from both spatial and channel dimensions, mining the potential correlations across omni-axis (i.e., spatial and channel). Coupling with mainstream window partitioning strategies, OSA can achieve superior performance with compelling computational budgets. Second, a multi-scale interaction scheme is proposed to mitigate sub-optimal ERF (i.e., premature saturation) in shallow models, which facilitates local propagation and meso-/global-scale interactions, rendering an omni-scale aggregation building block. Extensive experiments demonstrate that Omni-SR achieves recordhigh performance on lightweight super-resolution benchmarks (e.g., 26.95dB@Urban100 x4 with only 792K parameters). Our code is available at https://github.com/Francis0625/Omni-SR.

        ----

        ## [2131] Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method

        **Authors**: *Ran Yi, Haoyuan Tian, Zhihao Gu, Yu-Kun Lai, Paul L. Rosin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02144](https://doi.org/10.1109/CVPR52729.2023.02144)

        **Abstract**:

        Image aesthetics assessment (IAA) is a challenging task due to its highly subjective nature. Most of the current studies rely on large-scale datasets (e.g., AVA and AADB) to learn a general model for all kinds of photography images. However, little light has been shed on measuring the aesthetic quality of artistic images, and the existing datasets only contain relatively few artworks. Such a defect is a great obstacle to the aesthetic assessment of artistic images. To fill the gap in the field of artistic image aesthetics assessment (AIAA), we first introduce a large-scale AIAA dataset: Boldbrush Artistic Image Dataset (BAlD), which consists of 60,337 artistic images covering various art forms, with more than 360,000 votes from online users. We then propose a new method, SAAN (Style-specific Art Assessment Network), which can effectively extract and utilize style-specific and generic aesthetic information to evaluate artistic images. Experiments demonstrate that our proposed approach outperforms existing lAA methods on the proposed BAlD dataset according to quantitative comparisons. We believe the proposed dataset and method can serve as a foundation for future AIAA works and inspire more research in this field. Dataset and code are available at: https://github.com/Dreemurr-T/BAID.git

        ----

        ## [2132] RWSC-Fusion: Region-Wise Style-Controlled Fusion Network for the Prohibited X-ray Security Image Synthesis

        **Authors**: *Luwen Duan, Min Wu, Lijian Mao, Jun Yin, Jianping Xiong, Xi Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02145](https://doi.org/10.1109/CVPR52729.2023.02145)

        **Abstract**:

        Automatic prohibited item detection in security inspection X-ray images is necessary for transportation. The abundance and diversity of the X-ray security images with prohibited item, termed as prohibited X-ray security images, are essential for training the detection model. In order to solve the data insufficiency, we propose a Region-Wise Style-Controlled Fusion (RWSC-Fusion) network, which superimposes the prohibited items onto the normal X-ray security images, to synthesize the prohibited X-ray security images. The proposed RWSC-Fusion innovates both network structure and loss functions to generate more realistic X-ray security images. Specifically, a RWSC-Fusion module is designed to enable the region-wise fusion by controlling the appearance of the overlapping region with novel modulation parameters. In addition, an Edge-Attention (EA) module is proposed to effectively improve the sharpness of the synthetic images. As for the unsupervised loss function, we propose the Luminance loss in Logarithmic form (LL) and Correlation loss of Saturation Difference (CSD), to optimize the fused X-ray security images in terms of luminance and saturation. We evaluate the authenticity and the training effect of the synthetic X-ray security images on private and public SIXray dataset. The results confirm that our synthetic images are reliable enough to augment the prohibited X-ray security images.

        ----

        ## [2133] Efficient Scale-Invariant Generator with Column-Row Entangled Pixel Synthesis

        **Authors**: *Thuan Hoang Nguyen, Thanh Van Le, Anh Tran*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02146](https://doi.org/10.1109/CVPR52729.2023.02146)

        **Abstract**:

        Any-scale image synthesis offers an efficient and scalable solution to synthesize photo-realistic images at any scale, even going beyond 2K resolution. However, existing GAN-based solutions depend excessively on convolutions and a hierarchical architecture, which introduce inconsistency and the “texture sticking” issue when scaling the output resolution. From another perspective, INR-based generators are scale-equivariant by design, but their huge memory footprint and slow inference hinder these networks from being adopted in large-scale or real-time systems. In this work, we propose Column-Row Entangled Pixel Synthesis (CREPS), a new generative model that is both efficient and scale-equivariant without using any spatial convolutions or coarse-to-fine design. To save memory footprint and make the system scalable, we employ a novel bi-line representation that decomposes layer-wise feature maps into separate “thick” column and row encodings. Experiments on various datasets, including FFHQ, LSUN-Church, MetFaces, and Flickr-Scenery, confirm CREPS’ ability to synthesize scale-consistent and alias-free images at any arbitrary resolution with proper training and inference speed. Code is available at https://github.com/VinAIResearch/CREPS.

        ----

        ## [2134] Masked and Adaptive Transformer for Exemplar Based Image Translation

        **Authors**: *Chang Jiang, Fei Gao, Biao Ma, Yuhao Lin, Nannan Wang, Gang Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02147](https://doi.org/10.1109/CVPR52729.2023.02147)

        **Abstract**:

        We present a novel framework for exemplar based image translation. Recent advanced methods for this task mainly focus on establishing cross-domain semantic correspondence, which sequentially dominates image generation in the manner of local style control. Unfortunately, cross-domain semantic matching is challenging; and matching errors ultimately degrade the quality of generated images. To overcome this challenge, we improve the accuracy of matching on the one hand, and diminish the role of matching in image generation on the other hand. To achieve the former, we propose a masked and adaptive transformer (MAT) for learning accurate cross-domain correspondence, and executing context-aware feature augmentation. To achieve the latter, we use source features of the input and global style codes of the exemplar, as sup-plementary information, for decoding an image. Besides, we devise a novel contrastive style learning method, for acquire quality-discriminative style representations, which in turn benefit high-quality image generation. Experimen-tal results show that our method, dubbed MATEBIT, performs considerably better than state-of-the-art methods, in diverse image translation tasks. The codes are available at https://github.com/AiArt-HDU/MATEBIT.

        ----

        ## [2135] SmartBrush: Text and Shape Guided Object Inpainting with Diffusion Model

        **Authors**: *Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, Kun Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02148](https://doi.org/10.1109/CVPR52729.2023.02148)

        **Abstract**:

        Generic image inpainting aims to complete a corrupted image by borrowing surrounding information, which barely generates novel content. By contrast, multi-modal inpainting provides more flexible and useful controls on the inpainted content, e.g., a text prompt can be used to describe an object with richer attributes, and a mask can be used to constrain the shape of the inpainted object rather than being only considered as a missing area. We propose a new diffusion-based model named SmartBrush for completing a missing region with an object using both text and shape-guidance. While previous work such as DALLE-2 and Stable Diffusion can do text-guided inapinting they do not support shape guidance and tend to modify background texture surrounding the generated object. Our model incorporates both text and shape guidance with precision control. To preserve the background better, we propose a novel training and sampling strategy by augmenting the diffusion U-net with object-mask prediction. Lastly, we introduce a multi-task training strategy by jointly training inpainting with text-to-image generation to leverage more training data. We conduct extensive experiments showing that our model outperforms all baselines in terms of visual quality, mask controllability, and background preservation.

        ----

        ## [2136] Neural Transformation Fields for Arbitrary-Styled Font Generation

        **Authors**: *Bin Fu, Junjun He, Jianjun Wang, Yu Qiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02149](https://doi.org/10.1109/CVPR52729.2023.02149)

        **Abstract**:

        Few-shot font generation (FFG), aiming at generating font images with a few samples, is an emerging topic in recent years due to the academic and commercial values. Typically, the FFG approaches follow the style-content disentanglement paradigm, which transfers the target font styles to characters by combining the content representations of source characters and the style codes of reference samples. Most existing methods attempt to increase font generation ability via exploring powerful style representations, which may be a sub-optimal solution for the FFG task due to the lack of modeling spatial transformation in transferring font styles. In this paper, we model font generation as a continuous transformation process from the source character image to the target font image via the creation and dissipation of font pixels, and embed the corresponding transformations into a neural transformation field. With the estimated transformation path, the neural transformation field generates a set of intermediate transformation results via the sampling process, and a font rendering formula is developed to accumulate them into the target font image. Extensive experiments show that our method achieves state-of-the-art performance on few-shot font generation task, which demonstrates the effectiveness of our proposed model. Our implementation is available at: https://github.com/fubinfb/NTF.

        ----

        ## [2137] Referring Image Matting

        **Authors**: *Jizhizi Li, Jing Zhang, Dacheng Tao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02150](https://doi.org/10.1109/CVPR52729.2023.02150)

        **Abstract**:

        Different from conventional image matting, which either requires user-defined scribbles/trimap to extract a specific foreground object or directly extracts all the foreground objects in the image indiscriminately, we introduce a new task named Referring Image Matting (RIM) in this paper, which aims to extract the meticulous alpha matte of the specific object that best matches the given natural language description, thus enabling a more natural and simpler instruction for image matting. First, we establish a large-scale challenging dataset Ref Matte by designing a comprehensive image composition and expression generation engine to automatically produce high-quality images along with diverse text attributes based on public datasets. RefMatte consists of 230 object categories, 47,500 images, 118,749 expression-region entities, and 474,996 expressions. Additionally, we construct a real-world test set with 100 high-resolution natural images and manually annotate complex phrases to evaluate the out-of-domain generalization abilities of RIM methods. Furthermore, we present a novel baseline method CLIPMat for RIM, including a context-embedded prompt, a text-driven semantic pop-up, and a multi-level details extractor. Extensive experiments on RefMatte in both keyword and expression settings validate the superiority of CLIPMat over representative methods. We hope this work could provide novel insights into image matting and encourage more follow-up studies. The dataset, code and models are available at https://github.com/JizhiziLi/RIM.

        ----

        ## [2138] Handwritten Text Generation from Visual Archetypes

        **Authors**: *Vittorio Pippi, Silvia Cascianelli, Rita Cucchiara*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02151](https://doi.org/10.1109/CVPR52729.2023.02151)

        **Abstract**:

        Generating synthetic images of handwritten text in a writer-specific style is a challenging task, especially in the case of unseen styles and new words, and even more when these latter contain characters that are rarely encountered during training. While emulating a writer's style has been recently addressed by generative models, the generalization towards rare characters has been disregarded. In this work, we devise a Transformer-based model for Few-Shot styled handwritten text generation and focus on obtaining a robust and informative representation of both the text and the style. In particular, we propose a novel representation of the textual content as a sequence of dense vectors obtained from images of symbols written as standard GNU Unifont glyphs, which can be considered their visual archetypes. This strategy is more suitable for generating characters that, despite having been seen rarely during training, possibly share visual details with the frequently observed ones. As for the style, we obtain a robust representation of unseen writers' calligraphy by exploiting specific pre-training on a large synthetic dataset. Quantitative and qualitative results demonstrate the effectiveness of our proposal in generating words in unseen styles and with rare characters more faithfully than existing approaches relying on independent one-hot encodings of the characters.

        ----

        ## [2139] SceneComposer: Any-Level Semantic Image Synthesis

        **Authors**: *Yu Zeng, Zhe Lin, Jianming Zhang, Qing Liu, John P. Collomosse, Jason Kuen, Vishal M. Patel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02152](https://doi.org/10.1109/CVPR52729.2023.02152)

        **Abstract**:

        We propose a new framework for conditional image synthesis from semantic layouts of any precision levels, ranging from pure text to a 2D semantic canvas with precise shapes. More specifically, the input layout consists of one or more semantic regions with free-form text descriptions and adjustable precision levels, which can be set based on the desired controllability. The framework naturally reduces to text-to-image (T2I) at the lowest level with no shape information, and it becomes segmentation-to-image (S2I) at the highest level. By supporting the levels in-between, our framework is flexible in assisting users of different drawing expertise and at different stages of their creative workflow. We introduce several novel techniques to address the challenges coming with this new setup, including a pipeline for collecting training data; a precision-encoded mask pyramid and a text feature map representation to jointly encode precision level, semantics, and composition information; and a multi-scale guided diffusion model to synthesize images. To evaluate the proposed method, we collect a test dataset containing user-drawn layouts with diverse scenes and styles. Experimental results show that the proposed method can generate high-quality images following the layout at given precision, and compares favorably against existing methods. Project page https://zengxianyu.github.io/scenec/

        ----

        ## [2140] Affordance Diffusion: Synthesizing Hand-Object Interactions

        **Authors**: *Yufei Ye, Xueting Li, Abhinav Gupta, Shalini De Mello, Stan Birchfield, Jiaming Song, Shubham Tulsiani, Sifei Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02153](https://doi.org/10.1109/CVPR52729.2023.02153)

        **Abstract**:

        Recent successes in image synthesis are powered by large-scale diffusion models. However, most methods are currently limited to either text- or image-conditioned generation for synthesizing an entire image, texture transfer or inserting objects into a user-specified region. In contrast, in this work we focus on synthesizing complex interactions (i.e., an articulated hand) with a given object. Given an RGB image of an object, we aim to hallucinate plausible images of a human hand interacting with it. We propose a two-step generative approach: a LayoutNet that samples an articulation-agnostic hand-object-interaction layout, and a ContentNet that synthesizes images of a hand grasping the object given the predicted layout. Both are built on top of a large-scale pretrained diffusion model to make use of its latent representation. Compared to baselines, the proposed method is shown to generalize better to novel objects and perform surprisingly well on out-of-distribution in-the-wild scenes of portable-sized objects. The resulting system allows us to predict descriptive affordance information, such as hand articulation and approaching orientation.

        ----

        ## [2141] LayoutDiffusion: Controllable Diffusion Model for Layout-to-Image Generation

        **Authors**: *Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhongang Qi, Ying Shan, Xi Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02154](https://doi.org/10.1109/CVPR52729.2023.02154)

        **Abstract**:

        Recently, diffusion models have achieved great success in image synthesis. However, when it comes to the layout-to-image generation where an image often has a complex scene of multiple objects, how to make strong control over both the global layout map and each detailed object remains a challenging task. In this paper, we propose a diffusion model named LayoutDiffusion that can obtain higher generation quality and greater controllability than the previous works. To overcome the difficult multimodal fusion of image and layout, we propose to construct a structural image patch with region information and transform the patched image into a special layout to fuse with the normal layout in a unified form. Moreover, Layout Fusion Module (LFM) and Object-aware Cross Attention (OaCA) are proposed to model the relationship among multiple objects and designed to be object-aware and position-sensitive, allowing for precisely controlling the spatial related information. Extensive experiments show that our LayoutDiffusion out-performs the previous SOTA methods on FID, CAS by relatively 46.35%,26.70% on COCO-stuff and 44.29%,41.82% on VG. Code is available at https://github.com/ZGCTroy/LayoutDiffusion.

        ----

        ## [2142] DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

        **Authors**: *Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, Kfir Aberman*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02155](https://doi.org/10.1109/CVPR52729.2023.02155)

        **Abstract**:

        Large text-to-image models achieved a remarkable leap in the evolution of AI, enabling high-quality and diverse synthesis of images from a given text prompt. However, these models lack the ability to mimic the appearance of subjects in a given reference set and synthesize novel renditions of them in different contexts. In this work, we present a new approach for “personalization” of text-to-image diffusion models. Given as input just a few images of a subject, we fine-tune a pretrained text-to-image model such that it learns to bind a unique identifier with that specific subject. Once the subject is embedded in the output domain of the model, the unique identifier can be used to synthesize novel photorealistic images of the subject contextualized in different scenes. By leveraging the semantic prior embedded in the model with a new autogenous class-specific prior preservation loss, our technique enables synthesizing the subject in diverse scenes, poses, views and lighting conditions that do not appear in the reference images. We apply our technique to several previously-unassailable tasks, including subject recontextualization, text-guided view synthesis, and artistic rendering, all while preserving the subject's key features. We also provide a new dataset and evaluation protocol for this new task of subject-driven generation. Project page: https://dreambooth.github.io/

        ----

        ## [2143] GLIGEN: Open-Set Grounded Text-to-Image Generation

        **Authors**: *Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, Yong Jae Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02156](https://doi.org/10.1109/CVPR52729.2023.02156)

        **Abstract**:

        Large-scale text-to-image diffusion models have made amazing advances. However, the status quo is to use text input alone, which can impede controllability. In this work, we propose GLIGEN, Grounded-Language-to-Image Generation, a novel approach that builds upon and extends the functionality of existing pre-trained text-to-image diffusion models by enabling them to also be conditioned on grounding inputs. To preserve the vast concept knowledge of the pre-trained model, we freeze all of its weights and inject the grounding information into new trainable layers via a gated mechanism. Our model achieves open-world grounded text2img generation with caption and bounding box condition inputs, and the grounding ability generalizes well to novel spatial configurations and concepts. GLIGEN's zero-shot performance on COCO and LVIS outperforms existing supervised layout-to-image baselines by a large margin.

        ----

        ## [2144] Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models

        **Authors**: *Patrick Schramowski, Manuel Brack, Björn Deiseroth, Kristian Kersting*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02157](https://doi.org/10.1109/CVPR52729.2023.02157)

        **Abstract**:

        Text-conditioned image generation models have recently achieved astonishing results in image quality and text alignment and are consequently employed in a fast-growing number of applications. Since they are highly data-driven, relying on billion-sized datasets randomly scraped from the internet, they also suffer, as we demonstrate, from degenerated and biased human behavior. In turn, they may even reinforce such biases. To help combat these undesired side effects, we present safe latent diffusion (SLD). Specifically, to measure the inappropriate degeneration due to unfiltered and imbalanced training sets, we establish a novel image generation test bed-inappropriate image prompts (I2P)-containing dedicated, real-world image-to-text prompts covering concepts such as nudity and violence. As our exhaustive empirical evaluation demonstrates, the introduced SLD removes and suppresses inappropriate image parts during the diffusion process, with no additional training required and no adverse effect on overall image quality or text alignment.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://huggingface.co/docs/diffusers/api/pipelines/stable.diffusion.safe

        ----

        ## [2145] EDICT: Exact Diffusion Inversion via Coupled Transformations

        **Authors**: *Bram Wallace, Akash Gokul, Nikhil Naik*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02158](https://doi.org/10.1109/CVPR52729.2023.02158)

        **Abstract**:

        Finding an initial noise vector that produces an input image when fed into the diffusion process (known as inversion) is an important problem in denoising diffusion models (DDMs), with applications for real image editing. The standard approach for real image editing with inversion uses denoising diffusion implicit models (DDIMs [29]) to deterministically noise the image to the intermediate state along the path that the denoising would follow given the original conditioning. However, DDIM inversion for real images is unstable as it relies on local linearization assumptions, which result in the propagation of errors, leading to incorrect image reconstruction and loss of content. To alleviate these problems, we propose Exact Diffusion Inversion via Coupled Transformations (EDICT), an inversion method that draws inspiration from affine coupling layers. EDICT enables mathematically exact inversion of real and model-generated images by maintaining two coupled noise vectors which are used to invert each other in an alternating fashion. Using Stable Diffusion [25], a state-of-the-art latent diffusion model, we demonstrate that EDICT successfully reconstructs real images with high fidelity. On complex image datasets like MS-COCO, EDICT reconstruction significantly outperforms DDIM, improving the mean square error of reconstruction by a factor of two. Using noise vectors inverted from real images, EDICT enables a wide range of image edits—from local and global semantic edits to image stylization—while maintaining fidelity to the original image structure. EDICT requires no model training/finetuning, prompt tuning, or extra data and can be combined with any pretrained DDM.

        ----

        ## [2146] Solving 3D Inverse Problems Using Pre-Trained 2D Diffusion Models

        **Authors**: *Hyungjin Chung, Dohoon Ryu, Michael T. McCann, Marc Louis Klasky, Jong Chul Ye*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02159](https://doi.org/10.1109/CVPR52729.2023.02159)

        **Abstract**:

        Diffusion models have emerged as the new state-of-the-art generative model with high quality samples, with intriguing properties such as mode coverage and high flexibility. They have also been shown to be effective inverse problem solvers, acting as the prior of the distribution, while the information of the forward model can be granted at the sampling stage. Nonetheless, as the generative process remains in the same high dimensional (i.e. identical to data dimension) space, the models have not been extended to 3D inverse problems due to the extremely high memory and computational cost. In this paper, we combine the ideas from the conventional model-based iterative reconstruction with the modern diffusion models, which leads to a highly effective method for solving 3D medical image reconstruction tasks such as sparse-view tomography, limited angle tomography, compressed sensing MRI from pre-trained 2D diffusion models. In essence, we propose to augment the 2D diffusion prior with a model-based prior in the remaining direction at test time, such that one can achieve coherent reconstructions across all dimensions. Our method can be run in a single commodity GPU, and establishes the new state-of-the-art, showing that the proposed method can perform reconstructions of high fidelity and accuracy even in the most extreme cases (e.g. 2-view 3D tomography). We further reveal that the generalization capacity of the proposed method is surprisingly high, and can be used to reconstruct volumes that are entirely different from the training dataset. Code available: https://github.com/HJ-harry/DiffusionMBIR

        ----

        ## [2147] Diffusion Probabilistic Model Made Slim

        **Authors**: *Xingyi Yang, Daquan Zhou, Jiashi Feng, Xinchao Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02160](https://doi.org/10.1109/CVPR52729.2023.02160)

        **Abstract**:

        Despite the recent visually-pleasing results achieved, the massive computational cost has been a long-standing flaw for diffusion probabilistic models (DPMs), which, in turn, greatly limits their applications on resource-limited platforms. Prior methods towards efficient DPM, however, have largely focused on accelerating the testing yet overlooked their huge complexity and sizes. In this paper, we make a dedicated attempt to lighten DPM while striving to preserve its favourable performance. We start by training a small-sized latent diffusion model (LDM) from scratch, but observe a significant fidelity drop in the synthetic images. Through a thorough assessment, we find that DPM is intrinsically biased against high-frequency generation, and learns to recover different frequency components at different time-steps. These properties make compact networks unable to represent frequency dynamics with accurate high-frequency estimation. Towards this end, we introduce a customized design for slim DPM, which we term as Spectral Diffusion (SD), for light-weight image synthesis. SD incorporates wavelet gating in its architecture to enable frequency dynamic feature extraction at every reverse step, and conducts spectrum-aware distillation to promote high-frequency recovery by inverse weighting the objective based on spectrum magnitude. Experimental results demonstrate that, SD achieves 8–18 × computational complexity reduction as compared to the latent diffusion models on a series of conditional and unconditional image generation tasks while retaining competitive image fidelity.

        ----

        ## [2148] Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models

        **Authors**: *Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, Karsten Kreis*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02161](https://doi.org/10.1109/CVPR52729.2023.02161)

        **Abstract**:

        Latent Diffusion Models (LDMs) enable high-quality image synthesis while avoiding excessive compute demands by training a diffusion model in a compressed lower-dimensional latent space. Here, we apply the LDM paradigm to high-resolution video generation, a particularly resource-intensive task. We first pre-train an LDM on images only; then, we turn the image generator into a video generator by introducing a temporal dimension to the latent space diffusion model and finetuning on encoded image sequences, i.e., videos. Similarly, we temporally align diffusion model upsamplers, turning them into temporally consistent video super resolution models. We focus on two relevant real-world applications: Simulation of in-the-wild driving data and creative content creation with text-to-video modeling. In particular, we validate our Video LDM on real driving videos of resolution 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$512 \times 1024$</tex>
, achieving state-of-the-art performance. Furthermore, our approach can easily leverage off-the-shelf pretrained image LDMs, as we only need to train a temporal alignment model in that case. Doing so, we turn the publicly available, state-of-the-art text-to-image LDM Stable Diffusion into an efficient and expressive text-to-video model with resolution up to 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$1280 \times 2048$</tex>
. We show that the temporal layers trained in this way generalize to different finetuned text-to-image LDMs. Utilizing this property, we show the first results for personalized text-to-video generation, opening exciting directions for future content creation. Project page: https://nv-tlabs.github.io/VideoLDM/

        ----

        ## [2149] Binary Latent Diffusion

        **Authors**: *Ze Wang, Jiang Wang, Zicheng Liu, Qiang Qiu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02162](https://doi.org/10.1109/CVPR52729.2023.02162)

        **Abstract**:

        In this paper, we show that a binary latent space can be explored for compact yet expressive image representations. We model the bi-directional mappings between an image and the corresponding latent binary representation by training an auto-encoder with a Bernoulli encoding distribution. On the one hand, the binary latent space provides a compact discrete image representation of which the distribution can be modeled more efficiently than pixels or continuous latent representations. On the other hand, we now represent each image patch as a binary vector instead of an index of a learned cookbook as in discrete image representations with vector quantization. In this way, we obtain binary latent representations that allow for better image quality and high-resolution image representations without any multi-stage hierarchy in the latent space. In this binary latent space, images can now be generated effectively using a binary latent diffusion model tailored specifically for modeling the prior over the binary image representations. We present both conditional and unconditional image generation experiments with multiple datasets, and show that the proposed method performs comparably to state-of-the-art methods while dramatically improving the sampling efficiency to as few as 16 steps without using any test-time acceleration. The proposed framework can also be seamlessly scaled to 1024 x 1024 high-resolution image generation without resorting to latent hierarchy or multi-stage refinements.

        ----

        ## [2150] Semi-Supervised Video Inpainting with Cycle Consistency Constraints

        **Authors**: *Zhiliang Wu, Hanyu Xuan, Changchang Sun, Weili Guan, Kang Zhang, Yan Yan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02163](https://doi.org/10.1109/CVPR52729.2023.02163)

        **Abstract**:

        Deep learning-based video inpainting has yielded promising results and gained increasing attention from re-searchers. Generally, these methods assume that the cor-rupted region masks of each frame are known and easily ob-tained. However, the annotation of these masks are labor-intensive and expensive, which limits the practical application of current methods. Therefore, we expect to relax this assumption by defining a new semi-supervised inpainting setting, making the networks have the ability of completing the corrupted regions of the whole video using the anno-tated mask of only one frame. Specifically, in this work, we propose an end-to-end trainable framework consisting of completion network and mask prediction network, which are designed to generate corrupted contents of the current frame using the known mask and decide the regions to be filled of the next frame, respectively. Besides, we introduce a cycle consistency loss to regularize the training parameters of these two networks. In this way, the completion network and the mask prediction network can constrain each other, and hence the overall performance of the trained model can be maximized. Furthermore, due to the natural existence of prior knowledge (e.g., corrupted contents and clear bor-ders), current video inpainting datasets are not suitable in the context of semi-supervised video inpainting. Thus, we create a new dataset by simulating the corrupted video of real-world scenarios. Extensive experimental results are reported to demonstrate the superiority of our model in the video inpainting task. Remarkably, although our model is trained in a semi-supervised manner, it can achieve compa-rable performance as fully-supervised methods.

        ----

        ## [2151] Towards Accurate Image Coding: Improved Autoregressive Image Generation with Dynamic Vector Quantization

        **Authors**: *Mengqi Huang, Zhendong Mao, Zhuowei Chen, Yongdong Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02164](https://doi.org/10.1109/CVPR52729.2023.02164)

        **Abstract**:

        Existing vector quantization (VQ) based autoregressive models follow a two-stage generation paradigm that first learns a codebook to encode images as discrete codes, and then completes generation based on the learned code-book. However, they encode fixed-size image regions into fixed-length codes and ignore their naturally different information densities, which results in insufficiency in important regions and redundancy in unimportant ones, and finally degrades the generation quality and speed. Moreover, the fixed-length coding leads to an unnatural rasterscan autoregressive generation. To address the problem, we propose a novel two-stage framework: (1) Dynamic-Quantization VAE (DQ-VAE) which encodes image regions into variable-length codes based on their information densities for an accurate & compact code representation. (2) DQ-Transformer which thereby generates images autoregressively from coarse-grained (smooth regions with fewer codes) to fine-grained (details regions with more codes) by modeling the position and content of codes in each granularity alternately, through a novel stacked-transformer architecture and shared-content, non-shared position input layers designs. Comprehensive experiments on various generation tasks validate our superiorities in both effectiveness and efficiency. Code will be released at https://github.com/CrossmodalGroup/DynamicVectorQuantization.

        ----

        ## [2152] Large-Capacity and Flexible Video Steganography via Invertible Neural Network

        **Authors**: *Chong Mou, Youmin Xu, Jiechong Song, Chen Zhao, Bernard Ghanem, Jian Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02165](https://doi.org/10.1109/CVPR52729.2023.02165)

        **Abstract**:

        Video steganography is the art of unobtrusively concealing secret data in a cover video and then recovering the secret data through a decoding protocol at the receiver end. Although several attempts have been made, most of them are limited to low-capacity and fixed steganography. To rectify these weaknesses, we propose a Large-capacity and Flexible Video Steganography Network (LF-VSN) in this paper. For large-capacity, we present a reversible pipeline to perform multiple videos hiding and recovering through a single invertible neural network (INN). Our method can hide/recover 7 secret videos in/from 1 cover video with promising performance. For flexibility, we propose a key-controllable scheme, enabling different receivers to recover particular secret videos from the same cover video through specific keys. Moreover, we further improve the flexibility by proposing a scalable strategy in multiple videos hiding, which can hide variable numbers of secret videos in a cover video with a single model and a single training session. Extensive experiments demonstrate that with the significant improvement of the video steganography performance, our proposed LF-VSN has high security, large hiding capacity, and flexibility. The source code is available at https://github.com/MC-E/LF-VSN.

        ----

        ## [2153] Neural Video Compression with Diverse Contexts

        **Authors**: *Jiahao Li, Bin Li, Yan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02166](https://doi.org/10.1109/CVPR52729.2023.02166)

        **Abstract**:

        For any video codecs, the coding efficiency highly relies on whether the current signal to be encoded can find the relevant contexts from the previous reconstructed signals. Traditional codec has verified more contexts bring substantial coding gain, but in a time-consuming manner. However, for the emerging neural video codec (NVC), its contexts are still limited, leading to low compression ratio. To boost NVC, this paper proposes increasing the context diversity in both temporal and spatial dimensions. First, we guide the model to learn hierarchical quality patterns across frames, which enriches long-term and yet highquality temporal contexts. Furthermore, to tap the potential of optical flow-based coding framework, we introduce a group-based offset diversity where the cross-group interaction is proposed for better context mining. In addition, this paper also adopts a quadtree-based partition to increase spatial context diversity when encoding the latent representation in parallel. Experiments show that our codec obtains 23.5% bitrate saving over previous SOTA NVC. Better yet, our codec has surpassed the under-developing next generation traditional codec/ECM in both RGB and YUV420 colorspaces, in terms of PSNR. The codes are at https://github.com/microsoft/DCVC.

        ----

        ## [2154] Efficient Semantic Segmentation by Altering Resolutions for Compressed Videos

        **Authors**: *Yubin Hu, Yuze He, Yanghao Li, Jisheng Li, Yuxing Han, Jiangtao Wen, Yong-Jin Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02167](https://doi.org/10.1109/CVPR52729.2023.02167)

        **Abstract**:

        Video semantic segmentation (VSS) is a computationally expensive task due to the per-frame prediction for videos of high frame rates. In recent work, compact models or adaptive network strategies have been proposed for efficient VSS. However, they did not consider a crucial factor that affects the computational cost from the input side: the input resolution. In this paper, we propose an altering resolution framework called AR-Seg for compressed videos to achieve efficient VSS. AR-Seg aims to reduce the computational cost by using low resolution for non-keyframes. To prevent the performance degradation caused by downsampling, we design a Cross Resolution Feature Fusion (CR-eFF) module, and supervise it with a novel Feature Similarity Training (FST) strategy. Specifically, CReFF first makes use of motion vectors stored in a compressed video to warp features from high-resolution keyframes to low-resolution non-keyframes for better spatial alignment, and then selectively aggregates the warped features with local attention mechanism. Furthermore, the proposed FST supervises the aggregated features with high-resolution features through an explicit similarity loss and an implicit constraint from the shared decoding layer. Extensive experiments on CamVid and Cityscapes show that AR-Seg achieves state-of-the-art performance and is compatible with different segmentation backbones. On CamVid, AR-Seg saves 67% computational cost (measured in GFLOPs) with the PSPNet18 back-bone while maintaining high segmentation accuracy. Code: https://github.com/THU-LYJ-Lab/AR-Seg.

        ----

        ## [2155] Structured Sparsity Learning for Efficient Video Super-Resolution

        **Authors**: *Bin Xia, Jingwen He, Yulun Zhang, Yitong Wang, Yapeng Tian, Wenming Yang, Luc Van Gool*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02168](https://doi.org/10.1109/CVPR52729.2023.02168)

        **Abstract**:

        The high computational costs of video super-resolution (VSR) models hinder their deployment on resource-limited devices, e.g., smartphones and drones. Existing VSR models contain considerable redundant filters, which drag down the inference efficiency. To prune these unimportant filters, we develop a structured pruning scheme called Structured Sparsity Learning (SSL) according to the properties of VSR. In SSL, we design pruning schemes for several key components in VSR models, including residual blocks, recurrent networks, and upsampling networks. Specifically, we develop a Residual Sparsity Connection (RSC) scheme for residual blocks of recurrent networks to liberate pruning restrictions and preserve the restoration information. For upsampling networks, we design a pixel-shuffle pruning scheme to guarantee the accuracy of feature channel-space conversion. In addition, we observe that pruning error would be amplified as the hidden states propagate along with recurrent networks. To alleviate the issue, we design Temporal Finetuning (TF). Extensive experiments show that SSL can significantly outperform recent methods quantitatively and qualitatively. The code is available at https://github.com/Zj-BinXia/SSL.

        ----

        ## [2156] DisCo-CLIP: A Distributed Contrastive Loss for Memory Efficient CLIP Training

        **Authors**: *Yihao Chen, Xianbiao Qi, Jianan Wang, Lei Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02169](https://doi.org/10.1109/CVPR52729.2023.02169)

        **Abstract**:

        We propose DisCo-CLIP, a distributed memory-efficient CLIP training approach, to reduce the memory consumption of contrastive loss when training contrastive learning models. Our approach decomposes the contrastive loss and its gradient computation into two parts, one to calculate the intra-GPU gradients and the other to compute the inter-GPU gradients. According to our decomposition, only the intra-GPU gradients are computed on the current GPU, while the inter-GPU gradients are collected via all_reduce from other GPUs instead of being repeatedly computed on every GPU. In this way, we can reduce the GPU memory consumption of contrastive loss computation from 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{O}(B^{2})$</tex>
 to 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{O}(\frac{B^{2}}{N})$</tex>
, where 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$B$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$N$</tex>
 are the batch size and the number of GPUs used for training. Such a distributed solution is mathematically equivalent to the original non-distributed contrastive loss computation, without sacrificing any computation accuracy. It is particularly efficient for large-batch CLIP training. For instance, DisCo-CLIP can enable contrastive training of a ViT-B/32 model with a batch size of 32K or 196K using 8 or 64 A100 40GB GPUs, compared with the original CLIP solution which requires 128 A100 40GB GPUs to train a ViT-B/32 model with a batch size of 32K.

        ----

        ## [2157] Boost Vision Transformer with GPU-Friendly Sparsity and Quantization

        **Authors**: *Chong Yu, Tao Chen, Zhongxue Gan, Jiayuan Fan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02170](https://doi.org/10.1109/CVPR52729.2023.02170)

        **Abstract**:

        The transformer extends its success from the language to the vision domain. Because of the stacked self-attention and cross-attention blocks, the acceleration deployment of vision transformer on GPU hardware is challenging and also rarely studied. This paper thoroughly designs a compression scheme to maximally utilize the GPU-friendly 2:4 fine-grained structured sparsity and quantization. Specially, an original large model with dense weight parameters is first pruned into a sparse one by 2:4 structured pruning, which considers the GPU's acceleration of 2:4 structured sparse pattern with FP16 data type, then the floating-point sparse model is further quantized into a fixed-point one by sparse-distillation-aware quantization aware training, which considers GPU can provide an extra speedup of 2:4 sparse calculation with integer tensors. A mixed-strategy knowledge distillation is used during the pruning and quantization process. The proposed compression scheme is flexible to support supervised and unsupervised learning styles. Experiment results show GPUSQ-ViT scheme achieves state-of-the-art compression by reducing vision transformer models 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathbf{6.4}-\mathbf{12.7}\times$</tex>
 on model size and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathbf{30.3}-\mathbf{62} \times$</tex>
 on FLOPs with negligible accuracy degradation on ImageNet classification, COCO detection and ADE20K segmentation benchmarking tasks. Moreover, GPUSQ-ViT can boost actual deployment performance by 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathbf{1.39}-\mathbf{1.79}\times$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathbf{3.22}-\mathbf{3.43}\times$</tex>
 of latency and throughput on A100 GPU, and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathbf{1.57}-\mathbf{1.69}\times$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathbf{2.11}-\mathbf{2.51}\times$</tex>
 improvement of latency and throughput on AGX Orin.

        ----

        ## [2158] All are Worth Words: A ViT Backbone for Diffusion Models

        **Authors**: *Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, Jun Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02171](https://doi.org/10.1109/CVPR52729.2023.02171)

        **Abstract**:

        Vision transformers (ViT) have shown promise in various vision tasks while the U-Net based on a convolutional neural network (CNN) remains dominant in diffusion models. We design a simple and general ViT-based architecture (named U-ViT) for image generation with diffusion models. U-ViT is characterized by treating all inputs including the time, condition and noisy image patches as tokens and employing long skip connections between shallow and deep layers. We evaluate U-ViT in unconditional and classconditional image generation, as well as text-to-image generation tasks, where U-ViT is comparable if not superior to a CNN-based U-Net of a similar size. In particular, latent diffusion models with U-ViT achieve record-breaking FID scores of 2.29 in class-conditional image generation on ImageNet 256×256, and 5.48 in text-to-image generation on MS-COCO, among methods without accessing large external datasets during the training of generative models. Our results suggest that, for diffusion-based image modeling, the long skip connection is crucial while the down-sampling and upsampling operators in CNN-based U-Net are not always necessary. We believe that U-ViT can provide insights for future research on backbones in diffusion models and benefit generative modeling on large scale cross-modality datasets.

        ----

        ## [2159] Sparsifiner: Learning Sparse Instance-Dependent Attention for Efficient Vision Transformers

        **Authors**: *Cong Wei, Brendan Duke, Ruowei Jiang, Parham Aarabi, Graham W. Taylor, Florian Shkurti*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02172](https://doi.org/10.1109/CVPR52729.2023.02172)

        **Abstract**:

        Vision Transformers (ViT) have shown competitive advantages in terms of performance compared to convolutional neural networks (CNNs), though they often come with high computational costs. To this end, previous methods explore different attention patterns by limiting a fixed number of spatially nearby tokens to accelerate the ViT's multi-head self-attention (MHSA) operations. However, such structured attention patterns limit the token-to-token connections to their spatial relevance, which disregards learned semantic connections from a full attention mask. In this work, we propose an approach to learn instance-dependent attention patterns, by devising a lightweight connectivity predictor module that estimates the connectivity score of each pair of tokens. Intuitively, two tokens have high connectivity scores if the features are considered relevant either spatially or semantically. As each token only attends to a small number of other tokens, the binarized connectivity masks are often very sparse by nature and therefore provide the opportunity to reduce network FLOPs via sparse computations. Equipped with the learned unstructured attention pattern, sparse attention ViT (Sparsifiner) produces a superior Pareto frontier between FLOPs and top-1 accuracy on ImageNet compared to token sparsity. Our method reduces 48% ~ 69% FLOPs of MHSA while the accuracy drop is within 0.4%. We also show that combining attention and token sparsity reduces ViT FLOPs by over 60%.

        ----

        ## [2160] DropKey for Vision Transformer

        **Authors**: *Bonan Li, Yinhan Hu, Xuecheng Nie, Congying Han, Xiangjian Jiang, Tiande Guo, Luoqi Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02174](https://doi.org/10.1109/CVPR52729.2023.02174)

        **Abstract**:

        In this paper, we focus on analyzing and improving the dropout technique for self-attention layers of Vision Transformer, which is important while surprisingly ignored by prior works. In particular, we conduct researches on three core questions: First, what to drop in self-attention layers? Different from dropping attention weights in literature, we propose to move dropout operations forward ahead of attention matrix calculation and set the Key as the dropout unit, yielding a novel dropout-before-softmax scheme. We theoretically verify that this scheme helps keep both regularization and probability features of attention weights, alleviating the overfittings problem to specific patterns and enhancing the model to globally capture vital information; Second, how to schedule the drop ratio in consecutive layers? In contrast to exploit a constant drop ratio for all layers, we present a new decreasing schedule that gradually decreases the drop ratio along the stack of self-attention layers. We experimentally validate the proposed schedule can avoid overfittings in low-level features and missing in high-level semantics, thus improving the robustness and stableness of model training; Third, whether need to perform structured dropout operation as CNN? We attempt patch-based block-version of dropout operation and find that this useful trick for CNN is not essential for ViT. Given exploration on the above three questions, we present the novel Drop-Key method that regards Key as the drop unit and exploits decreasing schedule for drop ratio, improving ViTs in a general way. Comprehensive experiments demonstrate the effectiveness of DropKey for various ViT architectures, e.g. T2T, VOLO, CeiT and DeiT, as well as for various vision tasks, e.g., image classification, object detection, human-object interaction detection and human body shape recovery.

        ----

        ## [2161] Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding

        **Authors**: *Zijiao Chen, Jiaxin Qing, Tiange Xiang, Wan Lin Yue, Juan Helen Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02175](https://doi.org/10.1109/CVPR52729.2023.02175)

        **Abstract**:

        Decoding visual stimuli from brain recordings aims to deepen our understanding of the human visual system and build a solid foundation for bridging human and computer vision through the Brain-Computer Interface. However, reconstructing high-quality images with correct semantics from brain recordings is a challenging problem due to the complex underlying representations of brain signals and the scarcity of data annotations. In this work, we present MinD-Vis: Sparse Masked Brain Modeling with Double-Conditioned Latent Diffusion Model for Human Vision Decoding. Firstly, we learn an effective self-supervised representation of fMRI data using mask modeling in a large latent space inspired by the sparse coding of information in the primary visual cortex. Then by augmenting a latent diffusion model with double-conditioning, we show that MinD-Vis can reconstruct highly plausible images with semantically matching details from brain recordings using very few paired annotations. We benchmarked our model qualitatively and quantitatively; the experimental results indicate that our method outperformed state-of-the-art in both semantic mapping (100-way semantic classification) and generation quality (FID) by 66% and 41% respectively. An exhaustive ablation study was also conducted to analyze our framework.

        ----

        ## [2162] ResFormer: Scaling ViTs with Multi-Resolution Training

        **Authors**: *Rui Tian, Zuxuan Wu, Qi Dai, Han Hu, Yu Qiao, Yu-Gang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02176](https://doi.org/10.1109/CVPR52729.2023.02176)

        **Abstract**:

        Vision Transformers (ViTs) have achieved overwhelming success, yet they suffer from vulnerable resolution scalability, i.e., the performance drops drastically when presented with input resolutions that are unseen during training. We introduce, ResFormer, a framework that is built upon the seminal idea of multi-resolution training for improved performance on a wide spectrum of, mostly unseen, testing resolutions. In particular, ResFormer operates on replicated images of different resolutions and enforces a scale consistency loss to engage interactive information across different scales. More importantly, to alternate among varying resolutions effectively, especially novel ones in testing, we propose a global-local positional embedding strategy that changes smoothly conditioned on input sizes. We conduct extensive experiments for image classification on ImageNet. The results provide strong quantitative evidence that ResFormer has promising scaling abilities towards a wide range of resolutions. For instance, ResFormer-B-MR achieves a Top-1 accuracy of 75.86% and 81.72% when evaluated on relatively low and high resolutions respectively (i.e., 96 and 640), which are 48% and 7.49% better than DeiT-B. We also demonstrate, moreover, ResFormer is flexible and can be easily extended to semantic segmentation, object detection and video action recognition.

        ----

        ## [2163] Stare at What You See: Masked Image Modeling without Reconstruction

        **Authors**: *Hongwei Xue, Peng Gao, Hongyang Li, Yu Qiao, Hao Sun, Houqiang Li, Jiebo Luo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02177](https://doi.org/10.1109/CVPR52729.2023.02177)

        **Abstract**:

        Masked Autoencoders (MAE) have been prevailing paradigms for large-scale vision representation pretraining. By reconstructing masked image patches from a small portion of visible image regions, MAE forces the model to infer semantic correlation within an image. Recently, some approaches apply semantic-rich teacher models to extract image features as the reconstruction target, leading to better performance. However, unlike the low-level features such as pixel values, we argue the features extracted by powerful teacher models already encode rich semantic correlation across regions in an intact image. This raises one question: is reconstruction necessary in Masked Image Modeling (MIM) with a teacher model? In this paper, we propose an efficient MIM paradigm named MaskAlign. MaskAlign simply learns the consistency of visible patch features extracted by the student model and intact image features extracted by the teacher model. To further advance the performance and tackle the problem of input inconsistency between the student and teacher model, we propose a Dynamic Alignment (DA) module to apply learnable alignment. Our experimental results demonstrate that masked modeling does not lose effectiveness even without reconstruction on masked regions. Combined with Dynamic Alignment, MaskAlign can achieve state-of-the-art performance with much higher efficiency. Code and models will be available at https://github.com/OpenPerceptionX/maskalign.

        ----

        ## [2164] Mixed Autoencoder for Self-Supervised Visual Representation Learning

        **Authors**: *Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02178](https://doi.org/10.1109/CVPR52729.2023.02178)

        **Abstract**:

        Masked Autoencoder (MAE) has demonstrated superior performance on various vision tasks via randomly masking image patches and reconstruction. However, effective data augmentation strategies for MAE still remain open questions, different from those in contrastive learning that serve as the most important part. This paper studies the prevailing mixing augmentation for MAE. We first demonstrate that naïve mixing will in contrast degenerate model performance due to the increase of mutual information (MI). To address, we propose homologous recognition, an auxiliary pretext task, not only to alleviate the MI increasement by explicitly requiring each patch to recognize homologous patches, but also to perform object-aware self-supervised pre-training for better downstream dense perception performance. With extensive experiments, we demonstrate that our proposed Mixed Autoencoder (MixedAE) achieves the state-of-the-art transfer results among masked image modeling (MIM) augmentations on different downstream tasks with significant efficiency. Specifically, our MixedAE outperforms MAE by +0.3% accuracy, +1.7 mIoU and +0.9 AP on ImageNet-1K, ADE20K and COCO respectively with a standard ViT-Base. Moreover, MixedAE surpasses iBOT, a strong MIM method combined with instance discrimination, while accelerating training by 2×. To our best knowledge, this is the very first work to consider mixing for MIM from the perspective of pretext task design. Code will be made available.

        ----

        ## [2165] Shape-Erased Feature Learning for Visible-Infrared Person Re-Identification

        **Authors**: *Jiawei Feng, Ancong Wu, Wei-Shi Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02179](https://doi.org/10.1109/CVPR52729.2023.02179)

        **Abstract**:

        Due to the modality gap between visible and infrared images with high visual ambiguity, learning diverse modality-shared semantic concepts for visible-infrared person re-identification (VI-ReID) remains a challenging problem. Body shape is one of the significant modality-shared cues for VI-ReID. To dig more diverse modality-shared cues, we expect that erasing body-shape-related semantic concepts in the learned features can force the ReID model to extract more and other modality-shared features for identification. To this end, we propose shape-erased feature learning paradigm that decorrelates modality-shared features in two orthogonal subspaces. Jointly learning shape-related feature in one subspace and shape-erased features in the orthogonal complement achieves a conditional mutual information maximization between shape-erased feature and identity discarding body shape information, thus enhancing the diversity of the learned representation explicitly. Extensive experiments on SYSU-MM01, RegDB, and HITSZ-VCM datasets demonstrate the effectiveness of our method.

        ----

        ## [2166] G-MSM: Unsupervised Multi-Shape Matching with Graph-Based Affinity Priors

        **Authors**: *Marvin Eisenberger, Aysim Toker, Laura Leal-Taixé, Daniel Cremers*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02180](https://doi.org/10.1109/CVPR52729.2023.02180)

        **Abstract**:

        We present G-MSM (Graph-based Multi-Shape Matching), a novel unsupervised learning approach for non-rigid shape correspondence. Rather than treating a collection of input poses as an unordered set of samples, we explicitly model the underlying shape data manifold. To this end, we propose an adaptive multi-shape matching architecture that constructs an affinity graph on a given set of training shapes in a self-supervised manner. The key idea is to combine putative, pairwise correspondences by propagating maps along shortest paths in the underlying shape graph. During training, we enforce cycle-consistency between such optimal paths and the pairwise matches which enables our model to learn topology-aware shape priors. We explore different classes of shape graphs and recover specific settings, like template-based matching (star graph) or learnable ranking/sorting (TSP graph), as special cases in our framework. Finally, we demonstrate state-of-the-art performance on several recent shape correspondence benchmarks, including realworld 3D scan meshes with topological noise and challenging inter-class pairs.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our implementation is available under the following link: https://github.com/marvin-eisenberger/gmsm-matching

        ----

        ## [2167] Efficient Mask Correction for Click-Based Interactive Image Segmentation

        **Authors**: *Fei Du, Jianlong Yuan, Zhibin Wang, Fan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02181](https://doi.org/10.1109/CVPR52729.2023.02181)

        **Abstract**:

        The goal of click-based interactive image segmentation is to extract target masks with the input of positive/negative clicks. Every time a new click is placed, existing methods run the whole segmentation network to obtain a corrected mask, which is inefficient since several clicks may be needed to reach satisfactory accuracy. To this end, we propose an efficient method to correct the mask with a lightweight mask correction network. The whole network remains a low computational cost from the second click, even if we have a large backbone. However, a simple correction network with limited capacity is not likely to achieve comparable performance with a classic segmentation network. Thus, we propose a click-guided self-attention module and a click-guided correlation module to effectively exploits the click information to boost performance. First, several tem-plates are selected based on the semantic similarity with click features. Then the self-attention module propagates the template information to other pixels, while the correlation module directly uses the templates to obtain target out-lines. With the efficient architecture and two click-guided modules, our method shows preferable performance and efficiency compared to existing methods. The code will be released at https://github.com/feiaxyt/EMC-Click.

        ----

        ## [2168] Prototype-Based Embedding Network for Scene Graph Generation

        **Authors**: *Chaofan Zheng, Xinyu Lyu, Lianli Gao, Bo Dai, Jingkuan Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02182](https://doi.org/10.1109/CVPR52729.2023.02182)

        **Abstract**:

        Current Scene Graph Generation (SGG) methods explore contextual information to predict relationships among entity pairs. However, due to the diverse visual appearance of numerous possible subject-object combinations, there is a large intra-class variation within each predicate category, e.g., “man-eating-pizza, giraffe-eating-leaf”, and the severe inter-class similarity between different classes, e.g., “man-holding-plate, man-eating-pizza”, in model's latent space. The above challenges prevent current SGG methods from acquiring robust features for reliable relation prediction. In this paper, we claim that the predicate's category-inherent semantics can serve as class-wise prototypes in the semantic space for relieving the challenges. To the end, we propose the Prototype-based Embedding Network (PE-Net), which models entities/predicates with prototype-aligned compact and distinctive representations and thereby establishes matching between entity pairs and predicates in a common embedding space for relation recognition. Moreover, Prototype-guided Learning (PL) is introduced to help PE-Net efficiently learn such entity-predicate matching, and Prototype Regularization (PR) is devised to relieve the ambiguous entity-predicate matching caused by the predicate's semantic overlap. Extensive experiments demonstrate that our method gains superior relation recognition capability on SGG, achieving new state-of-the-art performances on both Visual Genome and Open Images datasets. The codes are available at https://github.com/VL-Group/PENET.

        ----

        ## [2169] Graph Representation for Order-aware Visual Transformation

        **Authors**: *Yue Qiu, Yanjun Sun, Fumiya Matsuzawa, Kenji Iwata, Hirokatsu Kataoka*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02183](https://doi.org/10.1109/CVPR52729.2023.02183)

        **Abstract**:

        This paper proposes a new visual reasoning formulation that aims at discovering changes between image pairs and their temporal orders. Recognizing scene dynamics and their chronological orders is a fundamental aspect of human cognition. The aforementioned abilities make it possible to follow step-by-step instructions, reason about and analyze events, recognize abnormal dynamics, and restore scenes to their previous states. However, it remains unclear how well current AI systems perform in these capabilities. Although a series of studies have focused on identifying and describing changes from image pairs, they mainly consider those changes that occur synchronously, thus neglecting potential orders within those changes. To address the above issue, we first propose a visual transformation graph structure for conveying order-aware changes. Then, we benchmarked previous methods on our newly generated dataset and identified the issues of existing methods for change order recognition. Finally, we show a significant improvement in order-aware change recognition by introducing a new model that explicitly associates different changes and then identifies changes and their orders in a graph representation.

        ----

        ## [2170] Unbiased Scene Graph Generation in Videos

        **Authors**: *Sayak Nag, Kyle Min, Subarna Tripathi, Amit K. Roy-Chowdhury*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02184](https://doi.org/10.1109/CVPR52729.2023.02184)

        **Abstract**:

        The task of dynamic scene graph generation (SGG) from videos is complicated and challenging due to the inherent dynamics of a scene, temporal fluctuation of model predictions, and the long-tailed distribution of the visual relationships in addition to the already existing challenges in image-based SGG. Existing methods for dynamic SGG have primarily focused on capturing spatio-temporal context using complex architectures without addressing the challenges mentioned above, especially the long-tailed distribution of relationships. This often leads to the generation of biased scene graphs. To address these challenges, we introduce a new framework called TEMPURA: TEmporal consistency and Memory Prototype guided UnceR-tainty Attenuation for unbiased dynamic SGG. TEMPURA employs object-level temporal consistencies via transformer-based sequence modeling, learns to synthesize unbiased relationship representations using memory-guided training, and attenuates the predictive uncertainty of visual relations using a Gaussian Mixture Model (GMM). Extensive experiments demonstrate that our method achieves significant (up to 10% in some cases) performance gain over existing methods highlighting its superiority in generating more unbiased scene graphs. Code: https://github.com/sayaknag/unbiasedSGG.git

        ----

        ## [2171] Recurrence without Recurrence: Stable Video Landmark Detection with Deep Equilibrium Models

        **Authors**: *Paul Micaelli, Arash Vahdat, Hongxu Yin, Jan Kautz, Pavlo Molchanov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02185](https://doi.org/10.1109/CVPR52729.2023.02185)

        **Abstract**:

        Cascaded computation, whereby predictions are recurrently refined over several stages, has been a persistent theme throughout the development of landmark detection models. In this work, we show that the recently proposed Deep Equilibrium Model (DEQ) can be naturally adapted to this form of computation. Our Landmark DEQ (LDEQ) achieves state-of-the-art performance on the challenging WFLW facial landmark dataset, reaching 3.92 NME with fewer parameters and a training memory cost of O(1) in the number of recurrent modules. Furthermore, we show that DEQs are particularly suited for landmark detection in videos. In this setting, it is typical to train on still images due to the lack of labelled videos. This can lead to a “flickering” effect at inference time on video, whereby a model can rapidly oscillate between different plausible solutions across consecutive frames. By rephrasing DEQs as a constrained optimization, we emulate recurrence at inference time, despite not having access to temporal data at training time. This Recurrence without Recurrence (RwR) paradigm helps in reducing landmark flicker, which we demonstrate by introducing a new metric, normalized mean flicker (NMF), and contributing a new facial landmark video dataset (WFLW-V) targeting landmark uncertainty. On the WFLW-V hard subset made up of 500 videos, our LDEQ with RwR improves the NME and NMF by 10 and 13% respectively, compared to the strongest previously published model using a hand-tuned conventional filter.

        ----

        ## [2172] VideoTrack: Learning to Track Objects via Video Transformer

        **Authors**: *Fei Xie, Lei Chu, Jiahao Li, Yan Lu, Chao Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02186](https://doi.org/10.1109/CVPR52729.2023.02186)

        **Abstract**:

        Existing Siamese tracking methods, which are built on pair-wise matching between two single frames, heavily rely on additional sophisticated mechanism to exploit temporal information among successive video frames, hindering them from efficiency and industrial deployments. In this work, we resort to sequence-level target matching that can encode temporal contexts into the spatial features through a neat feedforward video model. Specifically, we adapt the standard video transformer architecture to visual tracking by enabling spatiotemporal feature learning directly from frame-level patch sequences. To better adapt to the tracking task, we carefully blend the spatiotemporal information in the video clips through sequential multi-branch triplet blocks, which formulates a video transformer backbone. Our experimental study compares different model variants, such as tokenization strategies, hierarchical structures, and video attention schemes. Then, we propose a disentangled dual-template mechanism that decouples static and dynamic appearance clues over time, and reduces temporal redundancy in video frames. Extensive experiments show that our method, named as Video Track, achieves state-of-the-art results while running in real-time.

        ----

        ## [2173] Breaking the "Object" in Video Object Segmentation

        **Authors**: *Pavel Tokmakov, Jie Li, Adrien Gaidon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02187](https://doi.org/10.1109/CVPR52729.2023.02187)

        **Abstract**:

        The appearance of an object can be fleeting when it transforms. As eggs are broken or paper is torn, their color, shape and texture can change dramatically, preserving virtually nothing of the original except for the identity itself. Yet, this important phenomenon is largely absent from existing video object segmentation (VOS) benchmarks. In this work, we close the gap by collecting a new dataset for Video Object Segmentation under Transformations (VOST). It consists of more than 700 high-resolution videos, captured in diverse environments, which are 20 seconds long on average and densely labeled with instance masks. We adopt a careful, multi-step approach to ensure that these videos focus on complex object transformations, capturing their full temporal extent. We then extensively evaluate state-of-the-art VOS methods and make a number of important discoveries. In particular, we show that existing methods struggle when applied to this novel task and that their main limitation lies in over-reliance on static appearance cues. This motivates us to propose a few modifications for the top-performing baseline that improve its capabilities by better modeling spatiotemporal information. More broadly, our work highlights the need for further research on learning more robust video object representations. Nothing is lost or created, all things are merely transformed. Antoine Lavoisier

        ----

        ## [2174] Hierarchical Semantic Contrast for Scene-aware Video Anomaly Detection

        **Authors**: *Shengyang Sun, Xiaojin Gong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02188](https://doi.org/10.1109/CVPR52729.2023.02188)

        **Abstract**:

        Increasing scene-awareness is a key challenge in video anomaly detection (VAD). In this work, we propose a hierarchical semantic contrast (HSC) method to learn a scene-aware VAD model from normal videos. We first incorporate foreground object and background scene features with high-level semantics by taking advantage of pre-trained video parsing models. Then, building upon the autoencoder-based reconstruction framework, we introduce both scene-level and object-level contrastive learning to enforce the encoded latent features to be compact within the same semantic classes while being separable across different classes. This hierarchical semantic contrast strategy helps to deal with the diversity of normal patterns and also increases their discrimination ability. Moreover, for the sake of tackling rare normal activities, we design a skeleton-based motion augmentation to increase samples and refine the model further. Extensive experiments on three public datasets and scene-dependent mixture datasets validate the effectiveness of our proposed method.

        ----

        ## [2175] Mask-Free Video Instance Segmentation

        **Authors**: *Lei Ke, Martin Danelljan, Henghui Ding, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02189](https://doi.org/10.1109/CVPR52729.2023.02189)

        **Abstract**:

        The recent advancement in Video Instance Segmentation (VIS) has largely been driven by the use of deeper and increasingly data-hungry transformer-based models. However, video masks are tedious and expensive to annotate, limiting the scale and diversity of existing VIS datasets. In this work, we aim to remove the mask-annotation requirement. We propose MaskFreeVIS, achieving highly competitive VIS performance, while only using bounding box annotations for the object state. We leverage the rich temporal mask consistency constraints in videos by introducing the Temporal KNN-patch Loss (TK-Loss), providing strong mask supervision without any labels. Our TK-Loss finds one-to-many matches across frames, through an efficient patch-matching step followed by a K-nearest neighbor selection. A consistency loss is then enforced on the found matches. Our mask-free objective is simple to implement, has no trainable parameters, is computationally efficient, yet outperforms baselines employing, e.g., state-of-the-art optical flow to enforce temporal mask consistency. We validate MaskFreeVIS on the YouTube-VIS 2019/2021, OVIS and BDD100K MOTS benchmarks. The results clearly demonstrate the efficacy of our method by drastically narrowing the gap between fully and weakly-supervised VIS performance. Our code and trained models are available at http://vis.xyz/pub/maskfreevis.

        ----

        ## [2176] Hierarchical Neural Memory Network for Low Latency Event Processing

        **Authors**: *Ryuhei Hamaguchi, Yasutaka Furukawa, Masaki Onishi, Ken Sakurada*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02190](https://doi.org/10.1109/CVPR52729.2023.02190)

        **Abstract**:

        This paper proposes a low latency neural network architecture for event-based dense prediction tasks. Conventional architectures encode entire scene contents at a fixed rate regardless of their temporal characteristics. Instead, the proposed network encodes contents at a proper temporal scale depending on its movement speed. We achieve this by constructing temporal hierarchy using stacked latent memories that operate at different rates. Given low latency event steams, the multi-level memories gradually extract dynamic to static scene contents by propagating information from the fast to the slow memory modules. The architecture not only reduces the redundancy of conventional architectures but also exploits long-term dependencies. Further-more, an attention-based event representation efficiently encodes sparse event streams into the memory cells. We conduct extensive evaluations on three event-based dense prediction tasks, where the proposed approach outperforms the existing methods on accuracy and latency, while demonstrating effective event and image fusion capabilities. The code is available at https://hamarh.github.io/hmnet/.

        ----

        ## [2177] Unifying Short and Long-Term Tracking with Graph Hierarchies

        **Authors**: *Orcun Cetintas, Guillem Brasó, Laura Leal-Taixé*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02191](https://doi.org/10.1109/CVPR52729.2023.02191)

        **Abstract**:

        Tracking objects over long videos effectively means solving a spectrum of problems, from short-term association for un-occluded objects to long-term association for objects that are occluded and then reappear in the scene. Methods tackling these two tasks are often disjoint and crafted for specific scenarios, and top-performing approaches are often a mix of techniques, which yields engineering-heavy solutions that lack generality. In this work, we question the need for hybrid approaches and introduce SUSHI, a unified and scalable multi-object tracker. Our approach processes long clips by splitting them into a hierarchy of subclips, which enables high scalability. We leverage graph neural networks to process all levels of the hierarchy, which makes our model unified across temporal scales and highly general. As a result, we obtain significant improvements over state-of-the-art on four diverse datasets. Our code and models are available at bit.ly/sushi-mot.

        ----

        ## [2178] Towards End-to-End Generative Modeling of Long Videos with Memory-Efficient Bidirectional Transformers

        **Authors**: *Jaehoon Yoo, Semin Kim, Doyup Lee, Chiheon Kim, Seunghoon Hong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02192](https://doi.org/10.1109/CVPR52729.2023.02192)

        **Abstract**:

        Autoregressive transformers have shown remarkable success in video generation. However, the transformers are prohibited from directly learning the longterm dependency in videos due to the quadratic complexity of self-attention, and inherently suffering from slow inference time and error propagation due to the autoregressive process. In this paper, we propose Memory-efficient Bidirectional Transformer (MeBT) for end-to-end learning of longterm dependency in videos and fast inference. Based on recent advances in bidirectional transformers, our method learns to decode the entire spatio-temporal volume of a video in parallel from partially observed patches. The proposed transformer achieves a linear time complexity in both encoding and decoding, by projecting observable context tokens into a fixed number of latent tokens and conditioning them to decode the masked tokens through the cross-attention. Empowered by linear complexity and bidirectional modeling, our method demonstrates significant improvement over the autoregressive transformers for generating moderately long videos in both quality and speed. Videos and code are available at https://sites.google.com/view/mebt-cvpr2023.

        ----

        ## [2179] An Empirical Study of End-to-End Video-Language Transformers with Masked Visual Modeling

        **Authors**: *Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin, William Yang Wang, Lijuan Wang, Zicheng Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02193](https://doi.org/10.1109/CVPR52729.2023.02193)

        **Abstract**:

        Masked visual modeling (MVM) has been recently proven effective for visual pre-training. While similar reconstructive objectives on video inputs (e.g., masked frame modeling) have been explored in video-language (VidL) pre-training, previous studies fail to find a truly effective MVM strategy that can largely benefit the downstream performance. In this work, we systematically examine the potential of MVM in the context of VidL learning. Specifically, we base our study on a fully end-to-end VIdeO-LanguagE Transformer (VIOLET) [15], where the supervision from MVM training can be backpropogated to the video pixel space. In total, eight different reconstructive targets of MVM are explored, from low-level pixel values and oriented gradients to high-level depth maps, optical flow, discrete visual tokens and latent visual features. We conduct comprehensive experiments and provide insights into the factors leading to effective MVM training, resulting in an enhanced model VIOLETv2. Empirically, we show VIOLETv2 pre-trained with MVM objective achieves notable improvements on 13 VidL benchmarks, ranging from video question answering, video captioning, to text-to-video retrieval.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code has been released at https://github.com/tsujuifu/pytorch_empirical-mvm

        ----

        ## [2180] Egocentric Audio-Visual Object Localization

        **Authors**: *Chao Huang, Yapeng Tian, Anurag Kumar, Chenliang Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02194](https://doi.org/10.1109/CVPR52729.2023.02194)

        **Abstract**:

        Humans naturally perceive surrounding scenes by unifying sound and sight from a first-person view. Likewise, machines are advanced to approach human intelligence by learning with multisensory inputs from an egocentric perspective. In this paper, we explore the challenging egocentric audio-visual object localization task and observe that 1) egomotion commonly exists in first-person recordings, even within a short duration; 2) The out-of-view sound components can be created when wearers shift their attention. To address the first problem, we propose a geometryaware temporal aggregation module that handles the egomotion explicitly. The effect of egomotion is mitigated by estimating the temporal geometry transformation and exploiting it to update visual representations. Moreover, we propose a cascaded feature enhancement module to over-come the second issue. It improves cross-modal localization robustness by disentangling visually-indicated audio representation. During training, we take advantage of the naturally occurring audio-visual temporal synchronization as the “free” self-supervision to avoid costly labeling. We also annotate and create the Epic Sounding Object dataset for evaluation purposes. Extensive experiments show that our method achieves state-of-the-art localization performance in egocentric videos and can be generalized to diverse audio-visual scenes. Code is available at https://github.com/WikiChao/Ego-AV-Loc.

        ----

        ## [2181] AVFormer: Injecting Vision into Frozen Speech Models for Zero-Shot AV-ASR

        **Authors**: *Paul Hongsuck Seo, Arsha Nagrani, Cordelia Schmid*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02195](https://doi.org/10.1109/CVPR52729.2023.02195)

        **Abstract**:

        Audiovisual automatic speech recognition (AV-ASR) aims to improve the robustness of a speech recognition system by incorporating visual information. Training fully supervised multimodal models for this task from scratch, however is limited by the need for large labelled audiovisual datasets (in each downstream domain of interest). We present AVFormer, a simple method for augmenting audio-only models with visual information, at the same time performing lightweight domain adaptation. We do this by (i) injecting visual embeddings into a frozen ASR model using lightweight trainable adaptors. We show that these can be trained on a small amount of weakly labelled video data with minimum additional training time and parameters. (ii) We also introduce a simple curriculum scheme during training which we show is crucial to enable the model to jointly process audio and visual information effectively; and finally (iii) we show that our model achieves state of the art zero-shot results on three different AV-ASR benchmarks (How2, VisSpeech and Ego4D), while also crucially preserving decent performance on traditional audio-only speech recognition benchmarks (LibriSpeech). Qualitative results show that our model effectively leverages visual information for robust speech recognition.

        ----

        ## [2182] A Light Weight Model for Active Speaker Detection

        **Authors**: *Junhua Liao, Haihan Duan, Kanghui Feng, Wanbing Zhao, Yanbing Yang, Liangyin Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02196](https://doi.org/10.1109/CVPR52729.2023.02196)

        **Abstract**:

        Active speaker detection is a challenging task in audiovisual scenarios, with the aim to detect who is speaking in one or more speaker scenarios. This task has received considerable attention because it is crucial in many applications. Existing studies have attempted to improve the performance by inputting multiple candidate information and designing complex models. Although these methods have achieved excellent performance, their high memory and computational power consumption render their application to resource-limited scenarios difficult. Therefore, in this study, a lightweight active speaker detection architecture is constructed by reducing the number of input candidates, splitting 2D and 3D convolutions for audio-visual feature extraction, and applying gated recurrent units with low computational complexity for cross-modal modeling. Experimental results on the AVA-ActiveSpeaker dataset reveal that the proposed framework achieves competitive mAP performance (94.1% vs. 94.2%), while the resource costs are significantly lower than the state-of-the-art method, particularly in model parameters (1.0M vs. 22.5M, approximately 23×) and FLOPs (0.6G vs. 2.6G, approximately 4×). Additionally, the proposed framework also performs well on the Columbia dataset, thus demonstrating good robustness. The code and model weights are available at https://github.com/Junhua-Liao/Light-ASD.

        ----

        ## [2183] Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline

        **Authors**: *Tiantian Geng, Teng Wang, Jinming Duan, Runmin Cong, Feng Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02197](https://doi.org/10.1109/CVPR52729.2023.02197)

        **Abstract**:

        Existing audio-visual event localization (AVE) handles manually trimmed videos with only a single instance in each of them. However, this setting is unrealistic as natural videos often contain numerous audio-visual events with different categories. To better adapt to real-life applications, in this paper we focus on the task of dense-localizing audio-visual events, which aims to jointly localize and recognize all audio-visual events occurring in an untrimmed video. The problem is challenging as it requires fine-grained audio-visual scene and context understanding. To tackle this problem, we introduce the first Untrimmed Audio-Visual (UnAV-J 00) dataset, which contains 10K untrimmed videos with over 30K audio-visual events. Each video has 2.8 audio-visual events on average, and the events are usually related to each other and might co-occur as in real-life scenes. Next, we formulate the task using a new learning-based framework, which is capable of fully integrating audio and visual modalities to localize audio-visual events with various lengths and capture dependencies between them in a single pass. Extensive experiments demonstrate the effectiveness of our method as well as the significance of multi-scale cross-modal perception and dependency modeling for this task. The dataset and code are available at https://unav100.github.io.

        ----

        ## [2184] Video Test-Time Adaptation for Action Recognition

        **Authors**: *Wei Lin, Muhammad Jehanzeb Mirza, Mateusz Kozinski, Horst Possegger, Hilde Kuehne, Horst Bischof*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02198](https://doi.org/10.1109/CVPR52729.2023.02198)

        **Abstract**:

        Although action recognition systems can achieve top performance when evaluated on in-distribution test points, they are vulnerable to unanticipated distribution shifts in test data. However, test-time adaptation of video action recognition models against common distribution shifts has so far not been demonstrated. We propose to address this problem with an approach tailored to spatio-temporal models that is capable of adaptation on a single video sample at a step. It consists in a feature distribution alignment technique that aligns online estimates of test set statistics towards the training statistics. We further enforce prediction consistency over temporally augmented views of the same test video sample. Evaluations on three benchmark action recognition datasets show that our proposed technique is architecture-agnostic and able to significantly boost the performance on both, the state of the art convolutional architecture TANet and the Video Swin Transformer. Our proposed method demonstrates a substantial performance gain over existing test-time adaptation approaches in both evaluations of a single distribution shift and the challenging case of random distribution shifts. Code will be available at https://github.com/wlin-at/ViTTA.

        ----

        ## [2185] Unified Keypoint-Based Action Recognition Framework via Structured Keypoint Pooling

        **Authors**: *Ryo Hachiuma, Fumiaki Sato, Taiki Sekii*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02199](https://doi.org/10.1109/CVPR52729.2023.02199)

        **Abstract**:

        This paper simultaneously addresses three limitations associated with conventional skeleton-based action recognition; skeleton detection and tracking errors, poor variety of the targeted actions, as well as person-wise and framewise action recognition. A point cloud deep-learning paradigm is introduced to the action recognition, and a unified framework along with a novel deep neural network architecture called Structured Keypoint Pooling is proposed. The proposed method sparsely aggregates keypoint features in a cascaded manner based on prior knowledge of the data structure (which is inherent in skeletons), such as the instances and frames to which each keypoint belongs, and achieves robustness against input errors. Its less constrained and tracking-free architecture enables time-series keypoints consisting of human skeletons and nonhuman object contours to be efficiently treated as an input 3D point cloud and extends the variety of the targeted action. Furthermore, we propose a Pooling-Switching Trick inspired by Structured Keypoint Pooling. This trick switches the pooling kernels between the training and inference phases to detect person-wise and framewise actions in a weakly supervised manner using only video-level action labels. This trick enables our training scheme to naturally introduce novel data augmentation, which mixes multiple point clouds extracted from different videos. In the experiments, we comprehensively verify the effectiveness of the proposed method against the limitations, and the method outperforms state-of-the-art skeleton-based action recognition and spatio-temporal action localization methods.

        ----

        ## [2186] Object Discovery from Motion-Guided Tokens

        **Authors**: *Zhipeng Bao, Pavel Tokmakov, Yu-Xiong Wang, Adrien Gaidon, Martial Hebert*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02200](https://doi.org/10.1109/CVPR52729.2023.02200)

        **Abstract**:

        Object discovery – separating objects from the background without manual labels – is a fundamental open challenge in computer vision. Previous methods struggle to go beyond clustering of low-level cues, whether handcrafted (e.g., color, texture) or learned (e.g., from auto-encoders). In this work, we augment the auto-encoder representation learning framework with two key components: motion-guidance and mid-level feature tokenization. Although both have been separately investigated, we introduce a new transformer decoder showing that their benefits can compound thanks to motion-guided vector quantization. We show that our architecture effectively leverages the synergy between motion and tokenization, improving upon the state of the art on both synthetic and real datasets. Our approach enables the emergence of interpretable object-specific mid-level features, demonstrating the benefits of motion-guidance (no labeling) and quantization (interpretability, memory efficiency).

        ----

        ## [2187] Open Set Action Recognition via Multi-Label Evidential Learning

        **Authors**: *Chen Zhao, Dawei Du, Anthony Hoogs, Christopher Funk*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02201](https://doi.org/10.1109/CVPR52729.2023.02201)

        **Abstract**:

        Existing methods for open set action recognition focus on novelty detection that assumes video clips show a single action, which is unrealistic in the real world. We propose a new method for open set action recognition and novelty detection via MUlti-Label Evidential learning (MULE), that goes beyond previous novel action detection methods by addressing the more general problems of single or multiple actors in the same scene, with simultaneous action(s) by any actor. Our Beta Evidential Neural Network estimates multi-action uncertainty with Beta densities based on actor-context-object relation representations. An evidence debiasing constraint is added to the objective function for optimization to reduce the static bias of video representations, which can incorrectly correlate predictions and static cues. We develop a primal-dual average scheme update-based learning algorithm to optimize the proposed problem and provide corresponding theoretical analysis. Besides, uncertainty and belief-based novelty estimation mechanisms are formulated to detect novel actions. Extensive experiments on two real-world video datasets show that our proposed approach achieves promising performance in single/multi-actor, single/multi-action settings. Our code and models are released at https://github.com/charliezhaoyinpeng/mule.

        ----

        ## [2188] PivoTAL: Prior-Driven Supervision for Weakly-Supervised Temporal Action Localization

        **Authors**: *Mamshad Nayeem Rizve, Gaurav Mittal, Ye Yu, Matthew Hall, Sandra Sajeev, Mubarak Shah, Mei Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02202](https://doi.org/10.1109/CVPR52729.2023.02202)

        **Abstract**:

        Weakly-supervised Temporal Action Localization (WTAL) attempts to localize the actions in untrimmed videos using only video-level supervision. Most recent works approach WTAL from a localization-by-classification perspective where these methods try to classify each video frame followed by a manually-designed post-processing pipeline to aggregate these per-frame action predictions into action snippets. Due to this perspective, the model lacks any explicit understanding of action boundaries and tends to focus only on the most discriminative parts of the video resulting in incomplete action localization. To address this, we present PivoTAL, Prior-driven Supervision for Weakly-supervised Temporal Action Localization, to approach WTAL from a localization-by-localization perspective by learning to localize the action snippets directly. To this end, PivoTAL leverages the underlying spatio-temporal regularities in videos in the form of action-specific scene prior, action snippet generation prior, and learnable Gaussian prior to supervise the localization-based training. PivoTAL shows significant improvement (of at least 3% avg mAP) over all existing methods on the benchmark datasets, THUMOS-14 and ActivitNet-v1.3.

        ----

        ## [2189] Improving Weakly Supervised Temporal Action Localization by Bridging Train-Test Gap in Pseudo Labels

        **Authors**: *Jingqiu Zhou, Linjiang Huang, Liang Wang, Si Liu, Hongsheng Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02203](https://doi.org/10.1109/CVPR52729.2023.02203)

        **Abstract**:

        The task of weakly supervised temporal action localization targets at generating temporal boundaries for actions of interest, meanwhile the action category should also be classified. Pseudo-label-based methods, which serve as an effective solution, have been widely studied recently. However, existing methods generate pseudo labels during training and make predictions during testing under different pipelines or settings, resulting in a gap between training and testing. In this paper, we propose to generate high-quality pseudo labels from the predicted action boundaries. Nevertheless, we note that existing post-processing, like NMS, would lead to information loss, which is insufficient to generate high-quality action boundaries. More importantly, transforming action boundaries into pseudo labels is quite challenging, since the predicted action instances are generally overlapped and have different confidence scores. Besides, the generated pseudo-labels can be fluctuating and inaccurate at the early stage of training. It might repeatedly strengthen the false predictions if there is no mechanism to conduct self-correction. To tackle these issues, we come up with an effective pipeline for learning better pseudo labels. Firstly, we propose a Gaussian weighted fusion module to preserve information of action instances and obtain high-quality action boundaries. Second, we formulate the pseudo-label generation as an optimization problem under the constraints in terms of the confidence scores of action instances. Finally, we introduce the idea of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\Delta$</tex>
 pseudo labels, which enables the model with the ability of self-correction. Our method achieves superior performance to existing methods on two benchmarks, THUMOS14 and ActivityNet1.3, achieving gains of 1.9% on THUMOS14 and 3.7% on ActivityNet1.3 in terms of average mAP. Our code is available at https://github.com/zhou745/GauFuse_WSTAL.git.

        ----

        ## [2190] Are Binary Annotations Sufficient? Video Moment Retrieval via Hierarchical Uncertainty-based Active Learning

        **Authors**: *Wei Ji, Renjie Liang, Zhedong Zheng, Wenqiao Zhang, Shengyu Zhang, Juncheng Li, Mengze Li, Tat-Seng Chua*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02204](https://doi.org/10.1109/CVPR52729.2023.02204)

        **Abstract**:

        Recent research on video moment retrieval has mostly focused on enhancing the performance of accuracy, efficiency, and robustness, all of which largely rely on the abundance of high-quality annotations. While the precise frame-level annotations are time-consuming and cost-expensive, few attentions have been paid to the labeling process. In this work, we explore a new interactive manner to stimulate the process of human-in-the-loop annotation in video moment retrieval task. The key challenge is to select “ambiguous” frames and videos for binary annotations to facilitate the network training. To be specific, we propose a new hierarchical uncertainty-based modeling that explicitly considers modeling the uncertainty of each frame within the entire video sequence corresponding to the query description, and selecting the frame with the highest uncertainty. Only selected frame will be annotated by the human experts, which can largely reduce the workload. After obtaining a small number of labels provided by the expert, we show that it is sufficient to learn a competitive video moment retrieval model in such a harsh environment. Moreover, we treat the uncertainty score of frames in a video as a whole, and estimate the difficulty of each video, which can further relieve the burden of video selection. In general, our active learning strategy for video moment retrieval works not only at the frame level but also at the sequence level. Experiments on two public datasets validate the effectiveness of our proposed method. Our code is released at https://github.com/renjie-liang/HUAL.

        ----

        ## [2191] Query - Dependent Video Representation for Moment Retrieval and Highlight Detection

        **Authors**: *WonJun Moon, Sangeek Hyun, Sanguk Park, Dongchan Park, Jae-Pil Heo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02205](https://doi.org/10.1109/CVPR52729.2023.02205)

        **Abstract**:

        Recently, video moment retrieval and highlight detection (MR/HD) are being spotlighted as the demand for video understanding is drastically increased. The key objective of MR/HD is to localize the moment and estimate clip-wise accordance level, i.e., saliency score, to the given text query. Although the recent transformer-based models brought some advances, we found that these methods do not fully exploit the information of a given query. For example, the relevance between text query and video contents is sometimes neglected when predicting the moment and its saliency. To tackle this issue, we introduce Query-Dependent DETR (QD-DETR), a detection transformer tailored for MR/HD. As we observe the insignificant role of a given query in transformer architectures, our encoding module starts with cross-attention layers to explicitly inject the context of text query into video representation. Then, to enhance the model's capability of exploiting the query information, we manipulate the video-query pairs to produce irrelevant pairs. Such negative (irrelevant) video-query pairs are trained to yield low saliency scores, which in turn, encourages the model to estimate precise accordance between query-video pairs. Lastly, we present an input-adaptive saliency predictor which adaptively defines the criterion of saliency scores for the given video-query pairs. Our extensive studies verify the importance of building the query-dependent representation for MR/HD. Specifically, QD-DETR outperforms state-of-the-art methods on QVHighlights, TVSum, and Charades-STA datasets. Codes are available at github.com/wjun0830IQD-DETR.

        ----

        ## [2192] Vita-CLIP: Video and text adaptive CLIP via Multimodal Prompting

        **Authors**: *Syed Talal Wasim, Muzammal Naseer, Salman H. Khan, Fahad Shahbaz Khan, Mubarak Shah*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02206](https://doi.org/10.1109/CVPR52729.2023.02206)

        **Abstract**:

        Adopting contrastive image-text pretrained models like CLIP towards video classification has gained attention due to its cost-effectiveness and competitive performance. However, recent works in this area face a trade-off. Finetuning the pretrained model to achieve strong supervised performance results in low zero-shot generalization. Similarly, freezing the backbone to retain zero-shot capability causes significant drop in supervised accuracy. Because of this, recent works in literature typically train separate models for supervised and zero-shot action recognition. In this work, we propose a multimodal prompt learning scheme that works to balance the supervised and zero-shot performance under a single unified training. Our prompting approach on the vision side caters for three aspects: 1) Global video-level prompts to model the data distribution; 2) Local frame-level prompts to provide per-frame discriminative conditioning; and 3) a summary prompt to extract a condensed video representation. Additionally, we define a prompting scheme on the text side to augment the textual context. Through this prompting scheme, we can achieve state-of-the-art zero-shot performance on Kinetics-600, HMDB51 and UCF101 while remaining competitive in the supervised setting. By keeping the pretrained backbone frozen, we optimize a much lower number of parameters and retain the existing general representation which helps achieve the strong zero-shot performance. Our codes/models will be released at https://github.com/TalalWasim/Vita-Clip..

        ----

        ## [2193] Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training

        **Authors**: *Dezhao Luo, Jiabo Huang, Shaogang Gong, Hailin Jin, Yang Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02207](https://doi.org/10.1109/CVPR52729.2023.02207)

        **Abstract**:

        The correlation between the vision and text is essential for video moment retrieval (VMR), however, existing methods heavily rely on separate pre-training feature extractors for visual and textual understanding. Without sufficient temporal boundary annotations, it is non-trivial to learn universal video-text alignments. In this work, we explore multi-modal correlations derived from large-scale image-text data to facilitate generalisable VMR. To address the limitations of image-text pre-training models on capturing the video changes, we propose a generic method, referred to as Visual-Dynamic Injection (VDI), to empower the model's understanding of video moments. Whilst existing VMR methods are focusing on building temporalaware video features, being aware of the text descriptions about the temporal changes is also critical but originally overlooked in pre-training by matching static images with sentences. Therefore, we extract visual context and spatial dynamic information from video frames and explicitly enforce their alignments with the phrases describing video changes (e.g. verb). By doing so, the potentially relevant visual and motion patterns in videos are encoded in the corresponding text embeddings (injected) so to enable more accurate video-text alignments. We conduct extensive experiments on two VMR benchmark datasets (Charades-STA and ActivityNet-Captions) and achieve state-of-the-art performances. Especially, VDI yields notable advantages when being tested on the out-of-distribution splits where the testing samples involve novel scenes and vocabulary.

        ----

        ## [2194] Hierarchical Video-Moment Retrieval and Step-Captioning

        **Authors**: *Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, Mohit Bansal*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02208](https://doi.org/10.1109/CVPR52729.2023.02208)

        **Abstract**:

        There is growing interest in searching for information from large video corpora. Prior works have studied relevant tasks, such as text-based video retrieval, moment retrieval, video summarization, and video captioning in isolation, without an end-to-end setup that can jointly search from video corpora and generate summaries. Such an end-to-end setup would allow for many interesting applications, e.g., a text-based search that finds a relevant video from a video corpus, extracts the most relevant moment from that video, and segments the moment into important steps with captions. To address this, we present the HIREST (HIerarchical REtrieval and STep-captioning) dataset and propose a new benchmark that covers hierarchical information retrieval and visual/textual stepwise summarization from an instructional video corpus. Hirest consists of 3.4K text-video pairs from an instructional video dataset, where 1.1 K videos have annotations of moment spans relevant to text query and breakdown of each moment into key instruction steps with caption and timestamps (totaling 8.6K step captions). Our hierarchical benchmark consists of video retrieval, moment retrieval, and two novel moment segmentation and step captioning tasks. In moment segmentation, models break down a video moment into instruction steps and identify start-end boundaries. In step captioning, models generate a textual summary for each step. We also present starting point task-specific and end-to-end joint baseline models for our new benchmark. While the baseline models show some promising results, there still exists large room for future improvement by the community.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
code and data: https://github.com/j-min/HiREST

        ----

        ## [2195] HierVL: Learning Hierarchical Video-Language Embeddings

        **Authors**: *Kumar Ashutosh, Rohit Girdhar, Lorenzo Torresani, Kristen Grauman*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02209](https://doi.org/10.1109/CVPR52729.2023.02209)

        **Abstract**:

        Video-language embeddings are a promising avenue for injecting semantics into visual representations, but existing methods capture only short-term associations between seconds-long video clips and their accompanying text. We propose HierVL, a novel hierarchical video-language em-bedding that simultaneously accounts for both long-term and short-term associations. As training data, we take videos accompanied by timestamped text descriptions of human actions, together with a high-level text summary of the activity throughout the long video (as are available in Ego4D). We introduce a hierarchical contrastive training objective that encourages text-visual alignment at both the clip level and video level. While the clip-level constraints use the step-by-step descriptions to capture what is happening in that instant, the video-level constraints use the summary text to capture why it is happening, i.e., the broader context for the activity and the intent of the actor. Our hierarchical scheme yields a clip representation that outperforms its single-level counterpart as well as a long-term video representation that achieves SotA results on tasks requiring long-term video modeling. HierVL success-fully transfers to multiple challenging downstream tasks (in EPIC-KITCHENS-100, Charades-Ego, HowTo100M) in both zero-shot and fine-tuned settings.

        ----

        ## [2196] Learning Transferable Spatiotemporal Representations from Natural Script Knowledge

        **Authors**: *Ziyun Zeng, Yuying Ge, Xihui Liu, Bin Chen, Ping Luo, Shu-Tao Xia, Yixiao Ge*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02210](https://doi.org/10.1109/CVPR52729.2023.02210)

        **Abstract**:

        Pre-training on large-scale video data has become a common recipe for learning transferable spatiotemporal representations in recent years. Despite some progress, existing methods are mostly limited to highly curated datasets (e.g., K400) and exhibit unsatisfactory out-of-the-box representations. We argue that it is due to the fact that they only capture pixel-level knowledge rather than spatiotemporal semantics, which hinders further progress in video understanding. Inspired by the great success of image-text pre-training (e.g., CLIP), we take the first step to exploit language semantics to boost transferable spatiotemporal representation learning. We introduce a new pre-text task, Turning to Video for Transcript Sorting (TVTS), which sorts shuffled ASR scripts by attending to learned video representations. We do not rely on descriptive captions and learn purely from video, i.e., leveraging the natural transcribed speech knowledge to provide noisy but useful semantics over time. Our method enforces the vision model to contextualize what is happening over time so that it can re-organize the narrative transcripts, and can seamlessly apply to large-scale uncurated video data in the real world. Our method demonstrates strong out-of-the-box spatiotemporal representations on diverse benchmarks, e.g., +13.6% gains over VideoMAE on SSV2 via linear probing. The code is available at https://github.com/TencentARC/TVTS.

        ----

        ## [2197] WINNER: Weakly-supervised hIerarchical decompositioN and aligNment for spatio-tEmporal video gRounding

        **Authors**: *Mengze Li, Han Wang, Wenqiao Zhang, Jiaxu Miao, Zhou Zhao, Shengyu Zhang, Wei Ji, Fei Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02211](https://doi.org/10.1109/CVPR52729.2023.02211)

        **Abstract**:

        Spatio-temporal video grounding aims to localize the aligned visual tube corresponding to a language query. Existing techniques achieve such alignment by exploiting dense boundary and bounding box annotations, which can be prohibitively expensive. To bridge the gap, we investigate the weakly-supervised setting, where models learn from easily accessible video-language data without annotations. We identify that intra-sample spurious correlations among video-language components can be alleviated if the model captures the decomposed structures of video and language data. In this light, we propose a novel framework, namely WINNER, for hierarchical video-text understanding. WINNER first builds the language decomposition tree in a bottom-up manner, upon which the structural attention mechanism and top-down feature backtracking jointly build a multi-modal decomposition tree, permitting a hierarchical understanding of unstructured videos. The multi-modal decomposition tree serves as the basis for multi-hierarchy language-tube matching. A hierarchical contrastive learning objective is proposed to learn the multi-hierarchy correspondence and distinguishment with intra-sample and inter-sample video-text decomposition structures, achieving video-language decomposition structure alignment. Extensive experiments demonstrate the rationality of our design and its effectiveness beyond state-of-the-art weakly supervised methods, even some supervised methods.

        ----

        ## [2198] Collaborative Static and Dynamic Vision-Language Streams for Spatio-Temporal Video Grounding

        **Authors**: *Zihang Lin, Chaolei Tan, Jian-Fang Hu, Zhi Jin, Tiancai Ye, Wei-Shi Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02212](https://doi.org/10.1109/CVPR52729.2023.02212)

        **Abstract**:

        Spatio-Temporal Video Grounding (STVG) aims to localize the target object spatially and temporally according to the given language query. It is a challenging task in which the model should well understand dynamic visual cues (e.g., motions) and static visual cues (e.g., object appearances) in the language description, which requires effective joint modeling of spatiotemporal visuallinguistic dependencies. In this work, we propose a novel framework in which a static vision-language stream and a dynamic vision-language stream are developed to collaboratively reason the target tube. The static stream performs cross-modal understanding in a single frame and learns to attend to the target object spatially according to intraframe visual cues like object appearances. The dynamic stream models visual-linguistic dependencies across multiple consecutive frames to capture dynamic cues like motions. We further design a novel cross-stream collaborative block between the two streams, which enables the static and dynamic streams to transfer useful and complementary information from each other to achieve collaborative reasoning. Experimental results show the effectiveness of the collaboration of the two streams and our overall frame-work achieves new state-of-the-art performance on both HCSTVG and VidSTG datasets.

        ----

        ## [2199] Learning Action Changes by Measuring Verb-Adverb Textual Relationships

        **Authors**: *Davide Moltisanti, Frank Keller, Hakan Bilen, Laura Sevilla-Lara*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02213](https://doi.org/10.1109/CVPR52729.2023.02213)

        **Abstract**:

        The goal of this work is to understand the way actions are performed in videos. That is, given a video, we aim to predict an adverb indicating a modification applied to the action (e.g. cut “finely”). We cast this problem as a regression task. We measure textual relationships between verbs and adverbs to generate a regression target representing the action change we aim to learn. We test our approach on a range of datasets and achieve state-of-the-art results on both adverb prediction and antonym classification. Furthermore, we outperform previous work when we lift two commonly assumed conditions: the availability of action labels during testing and the pairing of adverbs as antonyms. Existing datasets for adverb recognition are either noisy, which makes learning difficult, or contain actions whose appearance is not influenced by adverbs, which makes evaluation less reliable. To address this, we collect a new high quality dataset: Adverbs in Recipes (AIR). We focus on instructional recipes videos, curating a set of actions that exhibit meaningful visual changes when performed differently. Videos in AIR are more tightly trimmed and were manually reviewed by multiple annotators to ensure high labelling quality. Results show that models learn better from AIR given its cleaner videos. At the same time, adverb prediction on AIR is challenging, demonstrating that there is considerable room for improvement.

        ----

        

[Go to the previous page](CVPR-2023-list10.md)

[Go to the next page](CVPR-2023-list12.md)

[Go to the catalog section](README.md)