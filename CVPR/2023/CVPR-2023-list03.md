## [400] SPARF: Neural Radiance Fields from Sparse and Noisy Poses

**Authors**: *Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, Federico Tombari*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00408](https://doi.org/10.1109/CVPR52729.2023.00408)

**Abstract**:

Neural Radiance Field (NeRF) has recently emerged as a powerful representation to synthesize photorealistic novel views. While showing impressive performance, it relies on the availability of dense input views with highly accurate camera poses, thus limiting its application in real-world scenarios. In this work, we introduce Sparse Pose Adjusting Radiance Field (SPARF), to address the challenge of novel-view synthesis given only few wide-baseline input images (as low as 3) with noisy camera poses. Our approach exploits multi-view geometry constraints in order to jointly learn the NeRF and refine the camera poses. By relying on pixel matches extracted between the input views, our multiview correspondence objective enforces the optimized scene and camera poses to converge to a global and geometrically accurate solution. Our depth consistency loss further encourages the reconstructed scene to be consistent from any viewpoint. Our approach sets a new state of the art in the sparse-view regime on multiple challenging datasets.

----

## [401] Interactive Segmentation of Radiance Fields

**Authors**: *Rahul Goel, Dhawal Sirikonda, Saurabh Saini, P. J. Narayanan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00409](https://doi.org/10.1109/CVPR52729.2023.00409)

**Abstract**:

Radiance Fields (RF) are popular to represent casually-captured scenes for new view synthesis and several applications beyond it. Mixed reality on personal spaces needs understanding and manipulating scenes represented as RFs, with semantic segmentation of objects as an important step. Prior segmentation efforts show promise but don't scale to complex objects with diverse appearance. We present the ISRF method to interactively segment objects with fine structure and appearance. Nearest neighbor feature matching using distilled semantic features identifies high-confidence seed regions. Bilateral search in a joint spatio-semantic space grows the region to recover accurate segmentation. We show state-of-the-art results of segmenting objects from RFs and compositing them to another scene, changing appearance, etc., and an interactive segmentation tool that others can use.

----

## [402] Temporal Interpolation is all You Need for Dynamic Neural Radiance Fields

**Authors**: *Sungheon Park, Minjung Son, Seokhwan Jang, Young Chun Ahn, Ji-Yeon Kim, Nahyup Kang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00410](https://doi.org/10.1109/CVPR52729.2023.00410)

**Abstract**:

Temporal interpolation often plays a crucial role to learn meaningful representations in dynamic scenes. In this paper, we propose a novel method to train spatiotemporal neural radiance fields of dynamic scenes based on temporal interpolation of feature vectors. Two feature interpolation methods are suggested depending on underlying representations, neural networks or grids. In the neural representation, we extract features from space-time inputs via multiple neural network modules and interpolate them based on time frames. The proposed multi-level feature interpolation network effectively captures features of both short-term and long-term time ranges. In the grid representation, space-time features are learned via four-dimensional hash grids, which remarkably reduces training time. The grid representation shows more than 100x faster training speed than the previous neural-net-based methods while maintaining the rendering quality. Concatenating static and dynamic features and adding a simple smoothness term further improve the performance of our proposed models. Despite the simplicity of the model architectures, our method achieved state-of-the-art performance both in rendering quality for the neural representation and in training speed for the grid representation.

----

## [403] Compressing Volumetric Radiance Fields to 1 MB

**Authors**: *Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, Liefeng Bo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00411](https://doi.org/10.1109/CVPR52729.2023.00411)

**Abstract**:

Approximating radiance fields with discretized volumetric grids is one of promising directions for improving NeRFs, represented by methods like DVGO, Plenoxels and TensoRF, which achieve super-fast training convergence and real-time rendering. However, these methods typically require a tremendous storage overhead, costing up to hundreds of megabytes of disk space and runtime memory for a single scene. We address this issue in this paper by introducing a simple yet effective framework, called vector quantized radiance fields (VQRF), for compressing these volume-grid-based radiance fields. We first present a robust and adaptive metric for estimating redundancy in grid models and performing voxel pruning by better exploring intermediate outputs of volumetric rendering. A trainable vector quantization is further proposed to improve the compactness of grid models. In combination with an efficient joint tuning strategy and post-processing, our method can achieve a compression ratio of 100× by reducing the overall model size to 1 MB with negligible loss on visual quality. Extensive experiments demonstrate that the proposed framework is capable of achieving unrivaled performance and well generalization across multiple methods with distinct volumetric structures, facilitating the wide use of volumetric radiance fields methods in real-world applications. Code is available at https://github.com/AlgoHunt/VQRF.

----

## [404] Multiscale Tensor Decomposition and Rendering Equation Encoding for View Synthesis

**Authors**: *Kang Han, Wei Xiang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00412](https://doi.org/10.1109/CVPR52729.2023.00412)

**Abstract**:

Rendering novel views from captured multi-view images has made considerable progress since the emergence of the neural radiance field. This paper aims to further advance the quality of view synthesis by proposing a novel approach dubbed the neural radiance feature field (NRFF). We first propose a multiscale tensor decomposition scheme to organize learnable features so as to represent scenes from coarse to fine scales. We demonstrate many benefits of the proposed multiscale representation, including more accurate scene shape and appearance reconstruction, and faster convergence compared with the single-scale representation. Instead of encoding view directions to model view-dependent effects, we further propose to encode the rendering equation in the feature space by employing the anisotropic spherical Gaussian mixture predicted from the proposed multiscale representation. The proposed NRFF improves state-of-the-art rendering results by over 1 dB in PSNR on both the NeRF and NSVF synthetic datasets. A significant improvement has also been observed on the real-world Tanks & Temples dataset. Code can be found at https://github.com/imkanghan/nrff.

----

## [405] Ref-NPR: Reference-Based Non-Photorealistic Radiance Fields for Controllable Scene Stylization

**Authors**: *Yuechen Zhang, Zexin He, Jinbo Xing, Xufeng Yao, Jiaya Jia*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00413](https://doi.org/10.1109/CVPR52729.2023.00413)

**Abstract**:

Current 3D scene stylization methods transfer textures and colors as styles using arbitrary style references, lacking meaningful semantic correspondences. We introduce Reference-Based Non-Photorealistic Radiance Fields (Ref-NP R) to address this limitation. This controllable method stylizes a 3D scene using radiance fields with a single stylized 2D view as a reference. We propose a ray registration process based on the stylized reference view to obtain pseudo-ray supervision in novel views. Then we exploit semantic correspondences in content images to fill occluded regions with perceptually similar styles, resulting in non-photorealistic and continuous novel view sequences. Our experimental results demonstrate that Ref-NPR out-performs existing scene and video stylization methods regarding visual quality and semantic correspondence. The code and data are publicly available on the project page at https://ref-npr.github.io.

----

## [406] Representing Volumetric Videos as Dynamic MLP Maps

**Authors**: *Sida Peng, Yunzhi Yan, Qing Shuai, Hujun Bao, Xiaowei Zhou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00414](https://doi.org/10.1109/CVPR52729.2023.00414)

**Abstract**:

This paper introduces a novel representation of volumetric videos for real-time view synthesis of dynamic scenes. Recent advances in neural scene representations demonstrate their remarkable capability to model and render complex static scenes, but extending them to represent dynamic scenes is not straightforward due to their slow rendering speed or high storage cost. To solve this problem, our key idea is to represent the radiance field of each frame as a set of shallow MLP networks whose parameters are stored in 2D grids, called MLP maps, and dynamically predicted by a 2D CNN decoder shared by all frames. Representing 3D scenes with shallow MLPs significantly improves the rendering speed, while dynamically predicting MLP parameters with a shared 2D CNN instead of explicitly storing them leads to low storage cost. Experiments show that the proposed approach achieves state-of-the-art rendering quality on the NHR and ZJU-MoCap datasets, while being efficient for real-time rendering with a speed of 41.7 fps for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$512\times 512$</tex>
 images on an RTX 3090 GPU. The code is available at https://zju3dv.github.io/mlp_maps/.

----

## [407] Fast Monocular Scene Reconstruction with Global-Sparse Local-Dense Grids

**Authors**: *Wei Dong, Christopher B. Choy, Charles Loop, Or Litany, Yuke Zhu, Anima Anandkumar*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00415](https://doi.org/10.1109/CVPR52729.2023.00415)

**Abstract**:

Indoor scene reconstruction from monocular images has long been sought after by augmented reality and robotics developers. Recent advances in neural field representations and monocular priors have led to remarkable results in scene-level surface reconstructions. The reliance on Multilayer Perceptrons (MLP), however, significantly limits speed in training and rendering. In this work, we propose to directly use signed distance function (SDF) in sparse voxel block grids for fast and accurate scene reconstruction without MLPs. Our globally sparse and locally dense data structure exploits surfaces' spatial sparsity, enables cache-friendly queries, and allows direct extensions to multi-modal data such as color and semantic labels. To apply this representation to monocular scene reconstruction, we develop a scale calibration algorithm for fast geometric initialization from monocular depth priors. We apply differentiable volume rendering from this initialization to refine details with fast convergence. We also introduce efficient high-dimensional Continuous Random Fields (CRFs) to further exploit the semantic-geometry consistency between scene objects. Experiments show that our approach is 10× faster in training and 100× faster in rendering while achieving comparable accuracy to state-of-the-art neural implicit methods.

----

## [408] DynIBaR: Neural Dynamic Image-Based Rendering

**Authors**: *Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, Noah Snavely*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00416](https://doi.org/10.1109/CVPR52729.2023.00416)

**Abstract**:

We address the problem of synthesizing novel views from a monocular video depicting a complex dynamic scene. State-of-the-art methods based on temporally varying Neural Radiance Fields (aka dynamic NeRFs) have shown impressive results on this task. However, for long videos with complex object motions and uncontrolled camera trajectories, these methods can produce blurry or inaccurate renderings, hampering their use in real-world applications. Instead of encoding the entire dynamic scene within the weights of MLPs, we present a new approach that addresses these limitations by adopting a volumetric image-based rendering framework that synthesizes new viewpoints by aggregating features from nearby views in a scene motion-aware manner. Our system retains the advantages of prior methods in its ability to model complex scenes and view-dependent effects, but also enables synthesizing photo-realistic novel views from long videos featuring complex scene dynamics with unconstrained camera trajectories. We demonstrate significant improvements over state-of-the-art methods on dynamic scene datasets, and also apply our approach to in-the-wild videos with challenging camera and object motion, where prior methods fail to produce high-quality renderings.

----

## [409] Plateau-Reduced Differentiable Path Tracing

**Authors**: *Michael Fischer, Tobias Ritschel*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00417](https://doi.org/10.1109/CVPR52729.2023.00417)

**Abstract**:

Current differentiable renderers provide light transport gradients with respect to arbitrary scene parameters. However, the mere existence of these gradients does not guarantee useful update steps in an optimization. Instead, inverse rendering might not converge due to inherent plateaus, i.e., regions of zero gradient, in the objective function. We propose to alleviate this by convolving the high-dimensional rendering function, that maps scene parameters to images, with an additional kernel that blurs the parameter space. We describe two Monte Carlo estimators to compute plateau-reduced gradients efficiently, i.e., with low variance, and show that these translate into net-gains in optimization error and runtime performance. Our approach is a straightforward extension to both black-box and differentiable renderers and enables optimization of problems with intricate light transport, such as caustics or global illumination, that existing differentiable renderers do not converge on. Our code is at github.com/mfischerucl/prdpt.

----

## [410] NeFII: Inverse Rendering for Reflectance Decomposition with Near-Field Indirect Illumination

**Authors**: *Haoqian Wu, Zhipeng Hu, Lincheng Li, Yongqiang Zhang, Changjie Fan, Xin Yu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00418](https://doi.org/10.1109/CVPR52729.2023.00418)

**Abstract**:

Inverse rendering methods aim to estimate geometry, materials and illumination from multi-view RGB images. In order to achieve better decomposition, recent approaches attempt to model indirect illuminations reflected from different materials via Spherical Gaussians (SG), which, however, tends to blur the high-frequency reflection details. In this paper, we propose an end-to-end inverse rendering pipeline that decomposes materials and illumination from multi-view images, while considering near-field indirect illumination. In a nutshell, we introduce the Monte Carlo sampling based path tracing and cache the indirect illumination as neural radiance, enabling a physics-faithful and easy-to-optimize inverse rendering method. To enhance efficiency and practicality, we leverage SG to represent the smooth environment illuminations and apply importance sampling techniques. To supervise indirect illuminations from unobserved directions, we develop a novel radiance consistency constraint between implicit neural radiance and path tracing results of unobserved rays along with the joint optimization of materials and illuminations, thus significantly improving the decomposition performance. Extensive experiments demonstrate that our method outperforms the state-of-the-art on multiple synthetic and real datasets, especially in terms of inter-reflection decomposition.

----

## [411] WildLight: In-the-wild Inverse Rendering with a Flashlight

**Authors**: *Ziang Cheng, Junxuan Li, Hongdong Li*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00419](https://doi.org/10.1109/CVPR52729.2023.00419)

**Abstract**:

This paper proposes a practical photometric solution for the challenging problem of in-the-wild inverse rendering under unknown ambient lighting. Our system recovers scene geometry and reflectance using only multi-view images captured by a smartphone. The key idea is to exploit smartphone's built-in flashlight as a minimally controlled light source, and decompose image intensities into two photometric components – a static appearance corresponds to ambient flux, plus a dynamic reflection induced by the moving flashlight. Our method does not require flash/non-flash images to be captured in pairs. Building on the success of neural light fields, we use an off-the-shelf method to capture the ambient reflections, while the flashlight component enables physically accurate photometric constraints to decouple reflectance and illumination. Compared to existing inverse rendering methods, our setup is applicable to non-darkroom environments yet sidesteps the inherent difficulties of explicit solving ambient reflections. We demonstrate by extensive experiments that our method is easy to implement, casual to set up, and consistently outperforms existing in-the-wild inverse rendering techniques. Finally, our neural reconstruction can be easily exported to PBR textured triangle mesh ready for industrial renderers. Our source code and data are released to https://github.com/za-cheng/WildLight.

----

## [412] Relightable Neural Human Assets from Multi-view Gradient Illuminations

**Authors**: *Taotao Zhou, Kai He, Di Wu, Teng Xu, Qixuan Zhang, Kuixiang Shao, Wenzheng Chen, Lan Xu, Jingyi Yu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00420](https://doi.org/10.1109/CVPR52729.2023.00420)

**Abstract**:

Human modeling and relighting are two fundamental problems in computer vision and graphics, where high-quality datasets can largely facilitate related research. However, most existing human datasets only provide multi-view human images captured under the same illumination. Although valuable for modeling tasks, they are not read-ily used in relighting problems. To promote research in both fields, in this paper, we present UltraStage, a new 3D human dataset that contains more than 2, 000 high-quality human assets captured under both multi-view and multi-illumination settings. Specifically, for each example, we provide 32 surrounding views illuminated with one white light and two gradient illuminations. In addition to regular multi-view images, gradient illuminations help recover de-tailed surface normal and spatially-varying material maps, enabling various relighting applications. Inspired by recent advances in neural representation, we further interpret each example into a neural human asset which allows novel view synthesis under arbitrary lighting conditions. We show our neural human assets can achieve extremely high capture performance and are capable of representing fine details such as facial wrinkles and cloth folds. We also validate UltraStage in single image relighting tasks, training neural networks with virtual relighted data from neural assets and demonstrating realistic rendering improvements over prior arts. UltraStage will be publicly available to the community to stimulate significant future developments in various human modeling and rendering tasks. The dataset is available at https://miaoing.github.io/RNHA.

----

## [413] DiffRF: Rendering-Guided 3D Radiance Field Diffusion

**Authors**: *Norman Müller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulò, Peter Kontschieder, Matthias Nießner*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00421](https://doi.org/10.1109/CVPR52729.2023.00421)

**Abstract**:

We introduce DiffRF, a novel approach for 3D radiance field synthesis based on denoising diffusion probabilistic models. While existing diffusion-based methods operate on images, latent codes, or point cloud data, we are the first to directly generate volumetric radiance fields. To this end, we propose a 3D denoising model which directly operates on an explicit voxel grid representation. However, as radiance fields generated from a set of posed images can be ambiguous and contain artifacts, obtaining ground truth radiance field samples is non-trivial. We address this challenge by pairing the denoising formulation with a rendering loss, enabling our model to learn a deviated prior that favours good image quality instead of trying to replicate fitting errors like floating artifacts. In contrast to 2D-diffusion models, our model learns multi-view consistent priors, enabling free-view synthesis and accurate shape generation. Compared to 3D GANs, our diffusion-based approach naturally enables conditional generation such as masked completion or single-view 3D synthesis at inference time.

----

## [414] Analyzing Physical Impacts Using Transient Surface Wave Imaging

**Authors**: *Tianyuan Zhang, Mark Sheinin, Dorian Chan, Mark Rau, Matthew O'Toole, Srinivasa G. Narasimhan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00422](https://doi.org/10.1109/CVPR52729.2023.00422)

**Abstract**:

The subtle vibrations on an object's surface contain information about the object's physical properties and its interaction with the environment. Prior works imaged surface vibration to recover the object's material properties via modal analysis, which discards the transient vibrations propagating immediately after the object is disturbed. Conversely, prior works that captured transient vibrations focused on recovering localized signals (e.g., recording nearby sound sources), neglecting the spatiotemporal relationship between vibrations at different object points. In this paper, we extract information from the transient surface vibrations simultaneously measured at a sparse set of object points using the dual-shutter camera described by Sheinin et al. [37]. We model the geometry of an elastic wave generated at the moment an object's surface is disturbed (e.g., a knock or a footstep) and use the model to localize the disturbance source for various materials (e.g., wood, plastic, tile). We also show that transient object vibrations contain additional cues about the impact force and the impacting object's material properties. We demonstrate our approach in applications like localizing the strikes of a ping-pong ball on a table mid-play and recovering the footsteps' locations by imaging the floor vibrations they create.

----

## [415] Neural Kaleidoscopic Space Sculpting

**Authors**: *Byeongjoo Ahn, Michael DeZeeuw, Ioannis Gkioulekas, Aswin C. Sankaranarayanan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00423](https://doi.org/10.1109/CVPR52729.2023.00423)

**Abstract**:

We introduce a method that recovers full-surround 3D reconstructions from a single kaleidoscopic image using a neural surface representation. Full-surround 3D reconstruction is critical for many applications, such as augmented and virtual reality. A kaleidoscope, which uses a single camera and multiple mirrors, is a convenient way of achieving full-surround coverage, as it redistributes light directions and thus captures multiple viewpoints in a single image. This enables single-shot and dynamic full-surround 3D reconstruction. However, using a kaleidoscopic image for multiview stereo is challenging, as we need to decompose the image into multi-view images by identifying which pixel corresponds to which virtual camera, a process we call labeling. To address this challenge, pur approach avoids the need to explicitly estimate labels, but instead “sculpts” a neural surface representation through the careful use of silhouette, background, foreground, and texture information present in the kaleidoscopic image. We demonstrate the advantages of our method in a range of simulated and real experiments, on both static and dynamic scenes.

----

## [416] Towards Unbiased Volume Rendering of Neural Implicit Surfaces with Geometry Priors

**Authors**: *Yongqiang Zhang, Zhipeng Hu, Haoqian Wu, Minda Zhao, Lincheng Li, Zhengxia Zou, Changjie Fan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00424](https://doi.org/10.1109/CVPR52729.2023.00424)

**Abstract**:

Learning surface by neural implicit rendering has been a promising way for multi-view reconstruction in recent years. Existing neural surface reconstruction methods, such as NeuS [24] and VolSDF [32], can produce reliable meshes from multi-view posed images. Although they build a bridge between volume rendering and Signed Distance Function (SDF), the accuracy is still limited. In this paper, we argue that this limited accuracy is due to the bias of their volume rendering strategies, especially when the viewing direction is close to be tangent to the surface. We revise and provide an additional condition for the unbiased volume rendering. Following this analysis, we propose a new rendering method by scaling the SDF field with the angle between the viewing direction and the surface normal vector. Experiments on simulated data indicate that our rendering method reduces the bias of SDF-based volume rendering. Moreover, there still exists non-negligible bias when the learnable standard deviation of SDF is large at early stage, which means that it is hard to supervise the rendered depth with depth priors. Alternatively we supervise zero-level set with surface points obtained from a pre-trained Multi-View Stereo network. We evaluate our method on the DTU dataset and show that it outperforms the state-of-the-arts neural implicit surface methods without mask supervision.

----

## [417] Neural Kernel Surface Reconstruction

**Authors**: *Jiahui Huang, Zan Gojcic, Matan Atzmon, Or Litany, Sanja Fidler, Francis Williams*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00425](https://doi.org/10.1109/CVPR52729.2023.00425)

**Abstract**:

We present a novel method for reconstructing a 3D implicit surface from a large-scale, sparse, and noisy point cloud. Our approach builds upon the recently introduced Neural Kernel Fields (NKF) [58] representation. It enjoys similar generalization capabilities to NKF, while simultaneously addressing its main limitations: (a) We can scale to large scenes through compactly supported kernel functions, which enable the use of memory-efficient sparse linear solvers. (b) We are robust to noise, through a gradient fitting solve. (c) We minimize training requirements, enabling us to learn from any dataset of dense oriented points, and even mix training data consisting of objects and scenes at different scales. Our method is capable of reconstructing millions of points in a few seconds, and handling very large scenes in an out-of-core fashion. We achieve state-of-the-art results on reconstruction benchmarks consisting of single objects (ShapeNet [5], ABC [33]), indoor scenes (ScanNet [11], Matterport3D [4]), and outdoor scenes (CARLA [16], Waymo [49]).

----

## [418] MM-3DScene: 3D Scene Understanding by Customizing Masked Modeling with Informative-Preserved Reconstruction and Self-Distilled Consistency

**Authors**: *Mingye Xu, Mutian Xu, Tong He, Wanli Ouyang, Yali Wang, Xiaoguang Han, Yu Qiao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00426](https://doi.org/10.1109/CVPR52729.2023.00426)

**Abstract**:

Masked Modeling (MM) has demonstrated widespread success in various vision challenges, by reconstructing masked visual patches. Yet, applying MM for large-scale 3D scenes remains an open problem due to the data sparsity and scene complexity. The conventional random masking paradigm used in 2D images often causes a high risk of ambiguity when recovering the masked region of 3D scenes. To this end, we propose a novel informative-preserved reconstruction, which explores local statistics to discover and preserve the representative structured points, effectively enhancing the pretext masking task for 3D scene understanding. Integrated with a progressive reconstruction manner, our method can concentrate on modeling regional geometry and enjoy less ambiguity for masked reconstruction. Besides, such scenes with progressive masking ratios can also serve to self-distill their intrinsic spatial consistency, requiring to learn the consistent representations from unmasked areas. By elegantly combining informative-preserved reconstruction on masked areas and consistency self-distillation from unmasked areas, a unified framework called MM-3DScene is yielded. We conduct comprehensive experiments on a host of downstream tasks. The consistent improvement (e.g., +6.1% mAP@0.5 on object detection and +2.2% mIoU on semantic segmentation) demonstrates the superiority of our approach.

----

## [419] Shape, Pose, and Appearance from a Single Image via Bootstrapped Radiance Field Inversion

**Authors**: *Dario Pavllo, David Joseph Tan, Marie-Julie Rakotosaona, Federico Tombari*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00427](https://doi.org/10.1109/CVPR52729.2023.00427)

**Abstract**:

Neural Radiance Fields (NeRF) coupled with CANs represent a promising direction in the area of 3D reconstruction from a single view, owing to their ability to efficiently model arbitrary topologies. Recent work in this area, however, has mostly focused on synthetic datasets where exact ground-truth poses are known, and has overlooked pose estimation, which is important for certain down-stream applications such as augmented reality (AR) and robotics. We introduce a principled end-to-end reconstructionframeworkfor natural images, where accurate ground-truth poses are not available. Our approach recovers an SDF-parameterized 3D shape, pose, and appearance from a single image of an object, without exploiting multiple views during training. More specifically, we leverage an unconditional 3D-aware generator, to which we apply a hybrid inversion scheme where a model produces a first guess of the solution which is then refined via optimization. Our frame-work can de-render an image in as few as 10 steps, enabling its use in practical scenarios. We demonstrate state-of-the-art results on a variety of real and synthetic benchmarks.

----

## [420] DisCoScene: Spatially Disentangled Generative Radiance Fields for Controllable 3D-aware Scene Synthesis

**Authors**: *Yinghao Xu, Menglei Chai, Zifan Shi, Sida Peng, Ivan Skorokhodov, Aliaksandr Siarohin, Ceyuan Yang, Yujun Shen, Hsin-Ying Lee, Bolei Zhou, Sergey Tulyakov*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00428](https://doi.org/10.1109/CVPR52729.2023.00428)

**Abstract**:

Existing 3D-aware image synthesis approaches mainly focus on generating a single canonical object and show limited capacity in composing a complex scene containing a variety of objects. This work presents DisCoScene: a 3D-aware generative model for high-quality and controllable scene synthesis. The key ingredient of our method is a very abstract object-level representation (i.e., 3D bounding boxes without semantic annotation) as the scene layout prior, which is simple to obtain, general to describe various scene contents, and yet informative to disentangle objects and background. Moreover, it serves as an intuitive user control for scene editing. Based on such a prior, the proposed model spatially disentangles the whole scene into object-centric generative radiance fields by learning on only 2D images with the global-local discrimination. Our model obtains the generation fidelity and editing flexibility of individual objects while being able to efficiently compose objects and the background into a complete scene. We demonstrate state-of-the-art performance on many scene datasets, including the challenging Waymo outdoor dataset. Project page can be found here.

----

## [421] Heat Diffusion Based Multi-Scale and Geometric Structure-Aware Transformer for Mesh Segmentation

**Authors**: *Chi-Chong Wong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00429](https://doi.org/10.1109/CVPR52729.2023.00429)

**Abstract**:

Triangle mesh segmentation is an important task in 3D shape analysis, especially in applications such as digital humans and AR/VR. Transformer model is inherently permutation-invariant to input, which makes it a suitable candidate model for 3D mesh processing. However, two main challenges involved in adapting Transformer from natural languages to 3D mesh are yet to be solved, such as i) extracting the multi-scale information of mesh data in an adaptive manner; ii) capturing geometric structures of mesh data as the discriminative characteristics of the shape. Current point based Transformer models fail to tackle such challenges and thus provide inferior performance for discretized surface segmentation. In this work, heat diffusion based method is exploited to tackle these problems. A novel Transformer model called MeshFormer is proposed, which i) integrates Heat Diffusion method into Multi-head Self-Attention operation (HDMSA) to adaptively capture the features from local neighborhood to global contexts; ii) applies a novel Heat Kernel Signature based Structure Encoding (HKSSE) to embed the intrinsic geometric structures of mesh instances into Transformer for structure-aware processing. Extensive experiments on triangle mesh segmentation validate the effectiveness of the proposed MeshFormer model and show significant improvements over current state-of-the-art methods.

----

## [422] Learning Detailed Radiance Manifolds for High-Fidelity and 3D-Consistent Portrait Synthesis from Monocular Image

**Authors**: *Yu Deng, Baoyuan Wang, Heung-Yeung Shum*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00430](https://doi.org/10.1109/CVPR52729.2023.00430)

**Abstract**:

A key challenge for novel view synthesis of monocular portrait images is 3D consistency under continuous pose variations. Most existing methods rely on 2D generative models which often leads to obvious 3D inconsistency artifacts. We present a 3D-consistent novel view synthesis approach for monocular portrait images based on a recent proposed 3D-aware GAN, namely Generative Radiance Manifolds (GRAM) [13], which has shown strong 3D consistency at multiview image generation of virtual subjects via the radiance manifolds representation. However, simply learning an encoder to map a real image into the latent space of GRAM can only reconstruct coarse radiance manifolds without faithful fine details, while improving the reconstruction fidelity via instance-specific optimization is time-consuming. We introduce a novel detail manifolds reconstructor to learn 3D-consistent fine details on the radiance manifolds from monocular images, and combine them with the coarse radiance manifolds for high-fidelity reconstruction. The 3D priors derived from the coarse radiance manifolds are used to regulate the learned details to ensure reasonable synthesized results at novel views. Trained on in-the-wild 2D images, our method achieves high-fidelity and 3D-consistent portrait synthesis largely outperforming the prior art. Project page: https://yudeng.github.io/GRAMInverter/

----

## [423] 3D-aware Conditional Image Synthesis

**Authors**: *Kangle Deng, Gengshan Yang, Deva Ramanan, Jun-Yan Zhu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00431](https://doi.org/10.1109/CVPR52729.2023.00431)

**Abstract**:

We propose pix2pix3D, a 3D-aware conditional generative model for controllable photorealistic image synthesis. Given a 2D label map, such as a segmentation or edge map, our model learns to synthesize a corresponding image from different viewpoints. To enable explicit 3D user control, we extend conditional generative models with neural radiance fields. Given widely-available posed monocular image and label map pairs, our model learns to assign a label to every 3D point in addition to color and density, which enables it to render the image and pixel-aligned label map simultaneously. Finally, we build an interactive system that allows users to edit the label map from different viewpoints and generate outputs accordingly.

----

## [424] VIVE3D: Viewpoint-Independent Video Editing using 3D-Aware GANs

**Authors**: *Anna Frühstück, Nikolaos Sarafianos, Yuanlu Xu, Peter Wonka, Tony Tung*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00432](https://doi.org/10.1109/CVPR52729.2023.00432)

**Abstract**:

We introduce VIVE3D, a novel approach that extends the capabilities of image-based 3D GANs to video editing and is able to represent the input video in an identity-preserving and temporally consistent way. We propose two new building blocks. First, we introduce a novel GAN inversion technique specifically tailored to 3D GANs by jointly embedding multiple frames and optimizing for the camera parameters. Second, besides traditional semantic face edits (e.g. for age and expression), we are the first to demonstrate edits that show novel views of the head enabled by the inherent prop-erties of 3D GANs and our optical flow-guided compositing technique to combine the head with the background video. Our experiments demonstrate that VIVE3D generates high-fidelity face edits at consistent quality from a range of camera viewpoints which are composited with the original video in a temporally and spatially consistent manner.

----

## [425] SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation

**Authors**: *Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G. Schwing, Liangyan Gui*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00433](https://doi.org/10.1109/CVPR52729.2023.00433)

**Abstract**:

In this work, we present a novel framework built to sim-plify 3D asset generation for amateur users. To enable interactive generation, our method supports a variety of input modalities that can be easily provided by a human, in-cluding images, text, partially observed shapes and combinations of these, further allowing to adjust the strength of each input. At the core of our approach is an encoder-decoder, compressing 3D shapes into a compact latent representation, upon which a diffusion model is learned. To enable a variety of multimodal inputs, we employ task-specific encoders with dropout followed by a cross-attention mechanism. Due to its flexibility, our model naturally supports a variety of tasks, outperforming prior works on shape completion, image-based 3D reconstruction, and text-to-3D. Most interestingly, our model can combine all these tasks into one swiss-army-knife tool, enabling the user to perform shape generation using incomplete shapes, images, and textual descriptions at the same time, providing the relative weights for each input and facilitating interactivity. Despite our approach being shape-only, we further show an efficient method to texture the generated shape using large-scale text-to-image models.

----

## [426] Generating Part-Aware Editable 3D Shapes without 3D Supervision

**Authors**: *Konstantinos Tertikas, Despoina Paschalidou, Boxiao Pan, Jeong Joon Park, Mikaela Angelina Uy, Ioannis Z. Emiris, Yannis Avrithis, Leonidas J. Guibas*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00434](https://doi.org/10.1109/CVPR52729.2023.00434)

**Abstract**:

Impressive progress in generative models and implicit representations gave rise to methods that can generate 3D shapes of high quality. However, being able to locally con-trol and edit shapes is another essential property that can unlock several content creation applications. Local control can be achieved with part-aware models, but existing meth-ods require 3D supervision and cannot produce textures. In this work, we devise PartNeRF, a novel part-aware gener-ative model for editable 3D shape synthesis that does not require any explicit 3D supervision. Our model generates objects as a set of locally defined NeRFs, augmented with an affine transformation. This enables several editing op-erations such as applying transformations on parts, mixing parts from different objects etc. To ensure distinct, manip-ulable parts we enforce a hard assignment of rays to parts that makes sure that the color of each ray is only determined by a single NeRF. As a result, altering one part does not af-fect the appearance of the others. Evaluations on various ShapeNet categories demonstrate the ability of our model to generate editable 3D objects of improved fidelity, compared to previous part-based generative approaches that require 3D supervision or models relying on NeRFs.

----

## [427] NeuralLift-360: Lifting an in-the-Wild 2D Photo to A 3D Object with 360° Views

**Authors**: *Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Yi Wang, Zhangyang Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00435](https://doi.org/10.1109/CVPR52729.2023.00435)

**Abstract**:

Virtual reality and augmented reality (XR) bring increasing demand for 3D content generation. However, creating high-quality 3D content requires tedious work from a human expert. In this work, we study the challenging task of lifting a single image to a 3D object and, for the first time, demonstrate the ability to generate a plausible 3D object with 360° views that corresponds well with the given reference image. By conditioning on the reference image, our model can fulfill the everlasting curiosity for synthesizing novel views of objects from images. Our technique sheds light on a promising direction of easing the workflows for 3D artists and XR designers. We propose a novel framework, dubbed NeuralLift-360, that utilizes a depth-aware neural radiance representation (NeRF) and learns to craft the scene guided by denoising diffusion models. By introducing a ranking loss, our NeuralLift-360 can be guided with rough depth estimation in the wild. We also adopt a CLIP-guided sampling strategy for the diffusion prior to provide coherent guidance. Extensive experiments demonstrate that our NeuralLift-360 significantly outperforms existing state-of-the-art baselines. Project page: https://vita-group.github.io/NeuralLift-360/

----

## [428] Implicit Identity Driven Deepfake Face Swapping Detection

**Authors**: *Baojin Huang, Zhongyuan Wang, Jifan Yang, Jiaxin Ai, Qin Zou, Qian Wang, Dengpan Ye*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00436](https://doi.org/10.1109/CVPR52729.2023.00436)

**Abstract**:

In this paper, we consider the face swapping detection from the perspective of face identity. Face swapping aims to replace the target face with the source face and generate the fake face that the human cannot distinguish between real and fake. We argue that the fake face contains the explicit identity and implicit identity, which respectively corresponds to the identity of the source face and target face during face swapping. Note that the explicit identities of faces can be extracted by regular face recognizers. Particularly, the implicit identity of real face is consistent with the its explicit identity. Thus the difference between explicit and implicit identity of face facilitates face swapping detection. Following this idea, we propose a novel implicit identity driven framework for face swapping detection. Specifically, we design an explicit identity contrast (EIC) loss and an implicit identity exploration (IIE) loss, which supervises a CNN backbone to embed face images into the implicit identity space. Under the guidance of EIC, real samples are pulled closer to their explicit identities, while fake samples are pushed away from their explicit identities. More-over, IIE is derived from the margin-based classification loss function, which encourages the fake faces with known target identities to enjoy intra-class compactness and inter-class diversity. Extensive experiments and visualizations on several datasets demonstrate the generalization of our method against the state-of-the-art counterparts.

----

## [429] Canonical Fields: Self-Supervised Learning of Pose-Canonicalized Neural Fields

**Authors**: *Rohith Agaram, Shaurya Dewan, Rahul Sajnani, Adrien Poulenard, K. Madhava Krishna, Srinath Sridhar*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00437](https://doi.org/10.1109/CVPR52729.2023.00437)

**Abstract**:

Coordinate-based implicit neural networks, or neural fields, have emerged as useful representations of shape and appearance in 3D computer vision. Despite advances, however, it remains challenging to build neural fields for categories of objects without datasets like ShapeNet that provide “canonicalized” object instances that are consistently aligned for their 3D position and orientation (pose). We present Canonical Field Network (CaFi-Net), a self-supervised method to canonicalize the 3D pose of instances from an object category represented as neural fields, specifically neural radiance fields (NeRFs). CaFi-Net directly learns from continuous and noisy radiance fields using a Siamese network architecture that is designed to extract equivariant field features for category-level canonicalization. During inference, our method takes pre-trained neural radiance fields of novel object instances at arbitrary 3D pose and estimates a canonical field with consistent 3D pose across the entire category. Extensive experiments on a new dataset of 1300 NeRF models across 13 object categories show that our method matches or exceeds the performance of 3D point cloud-based methods.

----

## [430] Improving Fairness in Facial Albedo Estimation via Visual-Textual Cues

**Authors**: *Xingyu Ren, Jiankang Deng, Chao Ma, Yichao Yan, Xiaokang Yang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00438](https://doi.org/10.1109/CVPR52729.2023.00438)

**Abstract**:

Recent 3D face reconstruction methods have made significant advances in geometry prediction, yet further cosmetic improvements are limited by lagged albedo because inferring albedo from appearance is an ill-posed problem. Although some existing methods consider prior knowledge from illumination to improve albedo estimation, they still produce a light-skin bias due to racially biased albedo models and limited light constraints. In this paper, we reconsider the relationship between albedo and face attributes and propose a ID2Albedo to directly estimate albedo without constraining illumination. Our key insight is that intrinsic semantic attributes such as race, skin color, and age can be used to constrain the albedo map. We first introduce visual-textual cues and design a semantic loss to supervise facial albedo estimation. Specifically, we pre-define text labels such as race, skin color, age, and wrinkles. Then, we employ the text-image model (CLIP) to compute the similarity between the text and the input image, and assign a pseudo-label to each facial image. We constrain generated albedos in the training phase to have the same attributes as the inputs. In addition, we train a high-quality, unbiased facial albedo generator and utilize the semantic loss to learn the mapping from illumination-robust identity features to the albedo latent codes. Finally, our ID2Albedo is trained in a self-supervised way and outperforms state-of-the-art albedo estimation methods in terms of accuracy and fidelity. It is worth mentioning that our approach has excellent generalizability and fairness, especially on in-the- wild data.

----

## [431] High-fidelity 3D Face Generation from Natural Language Descriptions

**Authors**: *Menghua Wu, Hao Zhu, Linjia Huang, Yiyu Zhuang, Yuanxun Lu, Xun Cao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00439](https://doi.org/10.1109/CVPR52729.2023.00439)

**Abstract**:

Synthesizing high-quality 3D face models from natural language descriptions is very valuable for many applications, including avatar creation, virtual reality, and telepresence. However, little research ever tapped into this task. We argue the major obstacle lies in 1) the lack of high-quality 3D face data with descriptive text annotation, and 2) the complex mapping relationship between descriptive language space and shape/appearance space. To solve these problems, we build Describe3D dataset, the first large-scale dataset with fine-grained text descriptions for text-to-3D face generation task. Then we propose a two-stage framework to first generate a 3D face that matches the concrete descriptions, then optimize the parameters in the 3D shape and texture space with abstract description to refine the 3D face model. Extensive experimental results show that our method can produce a faithful 3D face that conforms to the input descriptions with higher accuracy and quality than previous methods. The code and Describe3D dataset are released at https://github.com/zhuhao-nju/describe3D.

----

## [432] DSFNet: Dual Space Fusion Network for Occlusion-Robust 3D Dense Face Alignment

**Authors**: *Heyuan Li, Bo Wang, Yu Cheng, Mohan S. Kankanhalli, Robby T. Tan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00440](https://doi.org/10.1109/CVPR52729.2023.00440)

**Abstract**:

Sensitivity to severe occlusion and large view angles limits the usage scenarios of the existing monocular 3D dense face alignment methods. The state-of-the-art 3DMM-based method, directly regresses the model's coefficients, underutilizing the low-level 2D spatial and semantic information, which can actually offer cues for face shape and orientation. In this work, we demonstrate how modeling 3D facial geometry in image and model space jointly can solve the occlusion and view angle problems. Instead of predicting the whole face directly, we regress image space features in the visible facial region by dense prediction first. Subsequently, we predict our model's coefficients based on the regressed feature of the visible regions, leveraging the prior knowledge of whole face geometry from the morphable models to complete the invisible regions. We further propose a fusion network that combines the advantages of both the image and model space predictions to achieve high robustness and accuracy in unconstrained scenarios. Thanks to the proposed fusion module, our method is robust not only to occlusion and large pitch and roll view angles, which is the bene- fit of our image space approach, but also to noise and large yaw angles, which is the benefit of our model space method. Comprehensive evaluations demonstrate the superior performance of our method compared with the state-of-the-art methods. On the 3D dense face alignment task, we achieve 3.80% NME on the AFLW2000-3D dataset, which outperforms the state-of-the-art method by 5.5%. Code is available at https://github.com/1hyfst/DSFNet.

----

## [433] High-fidelity Facial Avatar Reconstruction from Monocular Video with Generative Priors

**Authors**: *Yunpeng Bai, Yanbo Fan, Xuan Wang, Yong Zhang, Jingxiang Sun, Chun Yuan, Ying Shan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00441](https://doi.org/10.1109/CVPR52729.2023.00441)

**Abstract**:

High-fidelity facial avatar reconstruction from a monocular video is a significant research problem in computer graphics and computer vision. Recently, Neural Radiance Field (NeRF) has shown impressive novel view rendering results and has been considered for facial avatar reconstruction. However, the complex facial dynamics and missing 3D information in monocular videos raise significant challenges for faithful facial reconstruction. In this work, we propose a new method for NeRF-based facial avatar reconstruction that utilizes 3D-aware generative prior. Different from existing works that depend on a conditional deformation field for dynamic modeling, we propose to learn a personalized generative prior, which is formulated as a local and low dimensional subspace in the latent space of 3D-GAN. We propose an efficient method to construct the personalized generative prior based on a small set of facial images of a given individual. After learning, it allows for photo-realistic rendering with novel views, and the face reenactment can be realized by performing navigation in the latent space. Our proposed method is applicable for different driven signals, including RGB images, 3DMM coefficients, and audio. Compared with existing works, we obtain superior novel view synthesis results and faithfully face reenactment performance. The code is available here https://github.com/bbaaii/HFA-GP.

----

## [434] 3DAvatarGAN: Bridging Domains for Personalized Editable Avatars

**Authors**: *Rameen Abdal, Hsin-Ying Lee, Peihao Zhu, Menglei Chai, Aliaksandr Siarohin, Peter Wonka, Sergey Tulyakov*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00442](https://doi.org/10.1109/CVPR52729.2023.00442)

**Abstract**:

Modern 3D-GANs synthesize geometry and texture by training on large-scale datasets with a consistent structure. Training such models on stylized, artistic data, with often unknown, highly variable geometry, and camera information has not yet been shown possible. Can we train a 3D GAN on such artistic data, while maintaining multi-view consistency and texture quality? To this end, we propose an adaptation framework, where the source domain is a pre-trained 3D-GAN, while the target domain is a 2D-GAN trained on artistic datasets. We, then, distill the knowledge from a 2D generator to the source 3D generator. To do that, we first propose an optimization-based method to align the distributions of camera parameters across domains. Second, we propose regularizations necessary to learn high-quality texture, while avoiding degenerate geometric solutions, such as flat shapes. Third, we show a deformation-based technique for modeling exaggerated geometry of artistic domains, enabling-as a byproduct- personalized geometric editing. Finally, we propose a novel inversion method for 3D-GANs linking the latent spaces of the source and the target domains. Our contributions-for the first time-allow for the generation, editing, and animation of personalized artistic 3D avatars on artistic datasets. Project Page: https:/rameenabdal.github.io/3DAvatarGAN

----

## [435] RODIN: A Generative Model for Sculpting 3D Digital Avatars Using Diffusion

**Authors**: *Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, Baining Guo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00443](https://doi.org/10.1109/CVPR52729.2023.00443)

**Abstract**:

This paper presents a 3D diffusion model that automatically generates 3D digital avatars represented as neural radiance fields (NeRFs). A significant challenge for 3D diffusion is that the memory and processing costs are prohibitive for producing high-quality results with rich details. To tackle this problem, we propose the roll-out diffusion network (RODIN), which takes a 3D NeRF model represented as multiple 2D feature maps and rolls out them onto a single 2D feature plane within which we perform 3D-aware diffusion. The RODIN model brings much-needed computational efficiency while preserving the integrity of 3D diffusion by using 3D-aware convolution that attends to projected features in the 2D plane according to their original relationships in 3D. We also use latent conditioning to orchestrate the feature generation with global coherence, leading to high-fidelity avatars and enabling semantic editing based on text prompts. Finally, we use hierarchical synthesis to further enhance details. The 3D avatars generated by our model compare favorably with those produced by existing techniques. We can generate highly detailed avatars with realistic hairstyles and facial hair. We also demonstrate 3D avatar generation from image or text, as well as text-guided editability.

----

## [436] Instant Volumetric Head Avatars

**Authors**: *Wojciech Zielonka, Timo Bolkart, Justus Thies*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00444](https://doi.org/10.1109/CVPR52729.2023.00444)

**Abstract**:

We present Instant Volumetric Head Avatars (INSTA), a novel approach for reconstructing photo-realistic digital avatars instantaneously. INSTA models a dynamic neural radiance field based on neural graphics primitives embedded around a parametric face model. Our pipeline is trained on a single monocular RGB portrait video that observes the subject under different expressions and views. While state-of-the-art methods take up to several days to train an avatar, our method can reconstruct a digital avatar in less than 10 minutes on modern GPU hardware, which is orders of magnitude faster than previous solutions. In addition, it allows for the interactive rendering of novel poses and expressions. By leveraging the geometry prior of the underlying parametric face model, we demonstrate that INSTA extrapolates to unseen poses. In quantitative and qualitative studies on various subjects, INSTA outperforms state-of-the-art methods regarding rendering quality and training time. Project website: https://zielon.github.io/insta/

----

## [437] Synthesizing Photorealistic Virtual Humans Through Cross-Modal Disentanglement

**Authors**: *Siddarth Ravichandran, Ondrej Texler, Dimitar Dinev, Hyun Jae Kang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00445](https://doi.org/10.1109/CVPR52729.2023.00445)

**Abstract**:

Over the last few decades, many aspects of human life have been enhanced with virtual domains, from the advent of digital assistants such as Amazon's Alexa and Apple's Siri to the latest metaverse efforts of the rebranded Meta. These trends underscore the importance of generating photorealistic visual depictions of humans. This has led to the rapid growth of so-called deepfake and talking-head generation methods in recent years. Despite their impressive results and popularity, they usually lack certain qualitative aspects such as texture quality, lips synchronization, or resolution, and practical aspects such as the ability to run in real-time. To allow for virtual human avatars to be used in practical scenarios, we propose an end-to-end framework for synthesizing high-quality virtual human faces capable of speaking with accurate lip motion with a special emphasis on performance. We introduce a novel network utilizing visemes as an intermediate audio representation and a novel data augmentation strategy employing a hierarchical image synthesis approach that allows disentanglement of the different modalities used to control the global head motion. Our method runs in real-time, and is able to deliver superior results compared to the current state-of-the-art.

----

## [438] 3D Cinemagraphy from a Single Image

**Authors**: *Xingyi Li, Zhiguo Cao, Huiqiang Sun, Jianming Zhang, Ke Xian, Guosheng Lin*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00446](https://doi.org/10.1109/CVPR52729.2023.00446)

**Abstract**:

We present 3D Cinemagraphy, a new technique that mar-ries 2D image animation with 3D photography. Given a single still image as input, our goal is to generate a video that contains both visual content animation and camera motion. We empirically find that naively combining existing 2D image animation and 3D photography methods leads to obvious artifacts or inconsistent animation. Our key insight is that representing and animating the scene in 3D space offers a natural solution to this task. To this end, we first convert the input image into feature-based layered depth images using predicted depth values, followed by unprojecting them to a feature point cloud. To animate the scene, we perform motion estimation and lift the 2D motion into the 3D scene flow. Finally, to resolve the problem of hole emer-gence as points move forward, we propose to bidirectionally displace the point cloud as per the scene flow and synthe-size novel views by separately projecting them into target image planes and blending the results. Extensive experiments demonstrate the effectiveness of our method. A user study is also conducted to validate the compelling rendering results of our method.

----

## [439] TryOnDiffusion: A Tale of Two UNets

**Authors**: *Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, Ira Kemelmacher-Shlizerman*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00447](https://doi.org/10.1109/CVPR52729.2023.00447)

**Abstract**:

Given two images depicting a person and a garment worn by another person, our goal is to generate a visualization of how the garment might look on the input person. A key challenge is to synthesize a photorealistic detail-preserving visualization of the garment, while warping the garment to accommodate a significant body pose and shape change across the subjects. Previous methods either focus on garment detail preservation without effective pose and shape variation, or allow tryon with the desired shape and pose but lack garment details. In this paper, we propose a diffusion-based architecture that unifies two UN ets (referred to as Parallel-UNet), which allows us to preserve garment details and warp the garment for significant pose and body change in a single network. The key ideas behind Parallel-UNet include: 1) garment is warped implicitly via a cross attention mechanism, 2) garment warp and person blend happen as part of a unified process as opposed to a sequence of two separate tasks. Experimental results indicate that TryOnDiffusion achieves state-of-the-art performance both qualitatively and quantitatively.

----

## [440] Diverse 3D Hand Gesture Prediction from Body Dynamics by Bilateral Hand Disentanglement

**Authors**: *Xingqun Qi, Chen Liu, Muyi Sun, Lincheng Li, Changjie Fan, Xin Yu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00448](https://doi.org/10.1109/CVPR52729.2023.00448)

**Abstract**:

Predicting natural and diverse 3D hand gestures from the upper body dynamics is a practical yet challenging task in virtual avatar creation. Previous works usually overlook the asymmetric motions between two hands and generate two hands in a holistic manner, leading to unnatural results. In this work, we introduce a novel bilateral hand disentanglement based two-stage 3D hand generation method to achieve natural and diverse 3D hand prediction from body dynamics. In the first stage, we intend to generate natural hand gestures by two hand-disentanglement branches. Considering the asymmetric gestures and motions of two hands, we introduce a Spatial-Residual Memory (SRM) module to model spatial interaction between the body and each hand by residual learning. To enhance the coordination of two hand motions wrt. body dynamics holistically, we then present a Temporal-Motion Memory (TMM) module. TMM can effectively model the temporal association between body dynamics and two hand motions. The second stage is built upon the insight that 3D hand predictions should be non-deterministic given the sequential body postures. Thus, we further diversify our 3D hand predictions based on the initial output from the stage one. Concretely, we propose a Prototypical-Memory Sampling Strategy (PSS) to generate the non-deterministic hand gestures by gradient-based Markov Chain Monte Carlo (MCMC) sampling. Extensive experiments demonstrate that our method outperforms the state-of-the-art models on the B2H dataset and our newly collected TED Hands dataset. The dataset and code are available at: https://github.com/XingqunQilab/Diverse-3D-Hand-Gesture-Prediction.

----

## [441] Normal-guided Garment UV Prediction for Human Re-texturing

**Authors**: *Yasamin Jafarian, Tuanfeng Y. Wang, Duygu Ceylan, Jimei Yang, Nathan Carr, Yi Zhou, Hyun Soo Park*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00449](https://doi.org/10.1109/CVPR52729.2023.00449)

**Abstract**:

Clothes undergo complex geometric deformations, which lead to appearance changes. To edit human videos in a physically plausible way, a texture map must take into account not only the garment transformation induced by the body movements and clothes fitting, but also its 3D fine-grained surface geometry. This poses, however, a new challenge of 3D reconstruction of dynamic clothes from an image or a video. In this paper, we show that it is possible to edit dressed human images and videos without 3D reconstruction. We estimate a geometry aware texture map between the garment region in an image and the texture space, a.k.a, UV map. Our UV map is designed to preserve isometry with respect to the underlying 3D surface by making use of the 3D surface normals predicted from the image. Our approach captures the underlying geometry of the garment in a self-supervised way, requiring no ground truth annotation of UV maps and can be readily extended to predict temporally coherent UV maps. We demonstrate that our method outperforms the state-of-the-art human UV map estimation approaches on both real and synthetic data.

----

## [442] REC-MV: REconstructing 3D Dynamic Cloth from Monocular Videos

**Authors**: *Lingteng Qiu, Guanying Chen, Jiapeng Zhou, Mutian Xu, Junle Wang, Xiaoguang Han*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00450](https://doi.org/10.1109/CVPR52729.2023.00450)

**Abstract**:

Reconstructing dynamic 3D garment surfaces with open boundaries from monocular videos is an important problem as it provides a practical and low-cost solution for clothes digitization. Recent neural rendering methods achieve high-quality dynamic clothed human reconstruction results from monocular video, but these methods cannot separate the garment surface from the body. Moreover, despite existing garment reconstruction methods based on feature curve representation demonstrating impressive results for garment reconstruction from a single image, they struggle to generate temporally consistent surfaces for the video input. To address the above limitations, in this paper, we formulate this task as an optimization problem of 3D garment feature curves and surface reconstruction from monocular video. We introduce a novel approach, called REC-MV, to jointly optimize the explicit feature curves and the implicit signed distance field (SDF) of the garments. Then the open garment meshes can be extracted via garment template registration in the canonical space. Experiments on multiple casually captured datasets show that our approach outperforms existing methods and can produce high-quality dynamic garment surfaces. The source code is available at https://github.com/GAP-LAB-CUHK-SZ/REC-MV.

----

## [443] SeSDF: Self-Evolved Signed Distance Field for Implicit 3D Clothed Human Reconstruction

**Authors**: *Yukang Cao, Kai Han, Kwan-Yee K. Wong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00451](https://doi.org/10.1109/CVPR52729.2023.00451)

**Abstract**:

We address the problem of clothed human reconstruction from a single image or uncalibrated multiview images. existing methods struggle with reconstructing detailed geometry of a clothed human and often require a calibrated setting for multiview reconstruction. We propose a flexible framework which, by leveraging the parametric SMPL-X model, can take an arbitrary number of input images to reconstruct a clothed human model under an uncalibrated setting. At the core of our framework is our novel self-evolved signed distance field (SeSDF) module which allows the framework to learn to deform the signed distance field (SDF) derived from the fitted SMPL-X model, such that detailed geometry reflecting the actual clothed human can be encoded for better reconstruction. Besides, we propose a simple method for self-calibration of multiview images via the fitted SMPL-X parameters. This lifts the requirement of tedious manual calibration and largely increases the flexibility of our method. Further, we introduce an effective occlusion-aware feature fusion strategy to account for the most useful features to reconstruct the human model. We thoroughly evaluate our framework on public benchmarks, demonstrating significant superiority over the state-of-the-arts both qualitatively and quantitatively.

----

## [444] Handy: Towards a High Fidelity 3D Hand Shape and Appearance Model

**Authors**: *Rolandos Alexandros Potamias, Stylianos Ploumpis, Stylianos Moschoglou, Vasileios Triantafyllou, Stefanos Zafeiriou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00453](https://doi.org/10.1109/CVPR52729.2023.00453)

**Abstract**:

Over the last few years, with the advent of virtual and augmented reality, an enormous amount of research has been focused on modeling, tracking and reconstructing human hands. Given their power to express human behavior, hands have been a very important, but challenging component of the human body. Currently, most of the state-of-the-art reconstruction and pose estimation methods rely on the low polygon MANO model. Apart from its low polygon count, MANO model was trained with only 31 adult subjects, which not only limits its expressive power but also imposes unnecessary shape reconstruction constraints on pose estimation methods. Moreover, hand appearance remains almost unexplored and neglected from the majority of hand reconstruction methods. In this work, we propose “Handy”, a large-scale model of the human hand, modeling both shape and appearance composed of over 1200 subjects which we make publicly available for the benefit of the research community. In contrast to current models, our proposed hand model was trained on a dataset with large diversity in age, gender, and ethnicity, which tackles the limitations of MANO and accurately reconstructs out-of-distribution samples. In order to create a high quality texture model, we trained a powerful GAN, which preserves high frequency details and is able to generate high resolution hand textures. To showcase the capabilities of the proposed model, we built a synthetic dataset of textured hands and trained a hand pose estimation network to reconstruct both the shape and appearance from single images. As it is demonstrated in an extensive series of quantitative as well as qualitative experiments, our model proves to be robust against the state-of-the-art and realistically captures the 3D hand shape and pose along with a high frequency detailed texture even in adverse “in-the-wild” conditions.

----

## [445] Fantastic Breaks: A Dataset of Paired 3D Scans of Real-World Broken Objects and Their Complete Counterparts

**Authors**: *Nikolas Lamb, Cameron Palmer, Benjamin Molloy, Sean Banerjee, Natasha Kholgade Banerjee*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00454](https://doi.org/10.1109/CVPR52729.2023.00454)

**Abstract**:

Automated shape repair approaches currently lack access to datasets that describe real-world damaged geometry. We present Fantastic Breaks (and Where to Find Them: https://terascale-all-sensing-research-studio.github.io/FantasticBreaks), a dataset containing scanned, waterproofed, and cleaned 3D meshes for 150 broken objects, paired and geometrically aligned with complete counterparts. Fantastic Breaks contains class and material labels, proxy repair parts that join to broken meshes to generate complete meshes, and manually annotated fracture boundaries. Through a detailed analysis of fracture geometry, we reveal differences between Fantastic Breaks and synthetic fracture datasets generated using geometric and physics-based methods. We show experimental shape repair evaluation with Fantastic Breaks using multiple learning-based approaches pre-trained with synthetic datasets and re-trained with subset of Fantastic Breaks.

----

## [446] Distilling Neural Fields for Real-Time Articulated Shape Reconstruction

**Authors**: *Jeff Tan, Gengshan Yang, Deva Ramanan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00455](https://doi.org/10.1109/CVPR52729.2023.00455)

**Abstract**:

We present a method for reconstructing articulated 3D models from videos in real-time, without test-time optimization or manual 3D supervision at training time. Prior work often relies on pre-built deformable models (e.g. SMAL/SMPL), or slow perscene optimization through differentiable rendering (e.g. dynamic NeRFs). Such methods fail to support arbitrary object categories, or are unsuitable for real-time applications. To address the challenge of collecting large-scale 3D training data for arbitrary deformable object categories, our key insight is to use off-the-shelf video-based dynamic NeRFs as 3D supervision to train a fast feed-forward network, turning 3D shape and motion prediction into a supervised distillation task. Our temporal-aware network uses articulated bones and blend skinning to represent arbitrary deformations, and is self-supervised on video datasets without requiring 3D shapes or viewpoints as input. Through distillation, our network learns to 3D-reconstruct unseen articulated objects at interactive frame rates. Our method yields higher-fidelity 3D reconstructions than prior real-time methods for animals, with the ability to render realistic images at novel viewpoints and poses.

----

## [447] GANmouflage: 3D Object Nondetection with Texture Fields

**Authors**: *Rui Guo, Jasmine Collins, Oscar de Lima, Andrew Owens*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00456](https://doi.org/10.1109/CVPR52729.2023.00456)

**Abstract**:

We propose a method that learns to camouflage 3D objects within scenes. Given an object's shape and a distribution of viewpoints from which it will be seen, we estimate a texture that will make it difficult to detect. Successfully solving this task requires a model that can accurately reproduce textures from the scene, while simultaneously dealing with the highly conflicting constraints imposed by each viewpoint. We address these challenges with a model based on texture fields and adversarial learning. Our model learns to camouflage a variety of object shapes from randomly sampled locations and viewpoints within the input scene, and is the first to address the problem of hiding complex object shapes. Using a human visual search study, we find that our estimated textures conceal objects significantly better than previous methods.

----

## [448] 3D Human Pose Estimation via Intuitive Physics

**Authors**: *Shashank Tripathi, Lea Müller, Chun-Hao P. Huang, Omid Taheri, Michael J. Black, Dimitrios Tzionas*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00457](https://doi.org/10.1109/CVPR52729.2023.00457)

**Abstract**:

Estimating 3D humans from images often produces implausible bodies that lean, float, or penetrate the floor. Such methods ignore the fact that bodies are typically supported by the scene. A physics engine can be used to enforce physical plausibility, but these are not differentiable, rely on unrealistic proxy bodies, and are difficult to integrate into existing optimization and learning frameworks. In contrast, we exploit novel intuitive-physics (IP) terms that can be inferred from a 3D SMPL body interacting with the scene. Inspired by biomechanics, we infer the pressure heatmap on the body, the Center of Pressure (CoP) from the heatmap, and the SMPL body's Center of Mass (CoM). With these, we develop IPMAN, to estimate a 3D body from a color image in a “stable” configuration by encouraging plausible floor contact and overlapping CoP and CoM. Our IP terms are intuitive, easy to implement, fast to compute, differentiable, and can be integrated into existing optimization and regression methods. We evaluate IPMAN on standard datasets and MoYo, a new dataset with synchronized multi-view images, ground-truth 3D bodies with complex poses, body-floor contact, CoM and pressure. IPMAN produces more plausible results than the state of the art, improving accuracy for static poses, while not hurting dynamic ones. Code and data are available for research at https://ipman.is.tue.mpg.de.

----

## [449] Object pop-up: Can we infer 3D objects and their poses from human interactions alone?

**Authors**: *Ilya A. Petrov, Riccardo Marin, Julian Chibane, Gerard Pons-Moll*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00458](https://doi.org/10.1109/CVPR52729.2023.00458)

**Abstract**:

The intimate entanglement between objects affordances and human poses is of large interest, among others, for behavioural sciences, cognitive psychology, and Computer Vision communities. In recent years, the latter has developed several object-centric approaches: starting from items, learning pipelines synthesizing human poses and dynamics in a realistic way, satisfying both geometrical and functional expectations. However, the inverse perspective is significantly less explored: Can we infer 3D objects and their poses from human interactions alone? Our investigation follows this direction, showing that a generic 3D human point cloud is enough to pop up an unobserved object, even when the user is just imitating a functionality (e.g., looking through a binocular) without involving a tangible counterpart. We validate our method qualitatively and quantitatively, with synthetic data and sequences acquired for the task, showing applicability for XR/VR.

----

## [450] UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy

**Authors**: *Yinzhen Xu, Weikang Wan, Jialiang Zhang, Haoran Liu, Zikang Shan, Hao Shen, Ruicheng Wang, Haoran Geng, Yijia Weng, Jiayi Chen, Tengyu Liu, Li Yi, He Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00459](https://doi.org/10.1109/CVPR52729.2023.00459)

**Abstract**:

In this work, we tackle the problem of learning universal robotic dexterous grasping from a point cloud observation under a table-top setting. The goal is to grasp and lift up objects in high-quality and diverse ways and generalize across hundreds of categories and even the unseen. Inspired by successful pipelines used in parallel gripper grasping, we split the task into two stages: 1) grasp proposal (pose) generation and 2) goal-conditioned grasp execution. For the first stage, we propose a novel probabilistic model of grasp pose conditioned on the point cloud observation that factorizes rotation from translation and articulation. Trained on our synthesized large-scale dexterous grasp dataset, this model enables us to sample diverse and high-quality dexterous grasp poses for the object point cloud. For the second stage, we propose to replace the motion planning used in parallel gripper grasping with a goal-conditioned grasp policy, due to the complexity involved in dexterous grasping execution. Note that it is very challenging to learn this highly generalizable grasp policy that only takes realistic inputs without oracle states. We thus propose several important innovations, including state canonicalization, object curriculum, and teacher-student distillation. In-tegrating the two stages, our final pipeline becomes the first to achieve universal generalization for dexterous grasping, demonstrating an average success rate of more than 60% on thousands of object instances, which significantly out-performs all baselines, meanwhile showing only a minimal generalization gap.

----

## [451] Constrained Evolutionary Diffusion Filter for Monocular Endoscope Tracking

**Authors**: *Xiongbiao Luo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00460](https://doi.org/10.1109/CVPR52729.2023.00460)

**Abstract**:

Stochastic filtering is widely used to deal with nonlinear optimization problems such as 3-D and visual tracking in various computer vision and augmented reality applications. Many current methods suffer from an imbalance between exploration and exploitation due to their particle degeneracy and impoverishment, resulting in local optimums. To address this imbalance, this work proposes a new constrained evolutionary diffusion filter for nonlinear optimization. Specifically, this filter develops spatial state constraints and adaptive history-recall differential evolution embedded evolutionary stochastic diffusion instead of sequential resampling to resolve the degeneracy and impoverishment problem. With application to monocular endoscope 3-D tracking, the experimental results show that the proposed filtering significantly improves the balance between exploration and exploitation and certainly works better than recent 3-D tracking methods. Particularly, the surgical tracking error was reduced from 4.03 mm to 2.59 mm.

----

## [452] Visibility Aware Human-Object Interaction Tracking from Single RGB Camera

**Authors**: *Xianghui Xie, Bharat Lal Bhatnagar, Gerard Pons-Moll*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00461](https://doi.org/10.1109/CVPR52729.2023.00461)

**Abstract**:

Capturing the interactions between humans and their environment in 3D is important for many applications in robotics, graphics, and vision. Recent works to reconstruct the 3D human and object from a single RGB image do not have consistent relative translation across frames because they assume a fixed depth. Moreover, their performance drops significantly when the object is occluded. In this work, we propose a novel method to track the 3D human, object, contacts, and relative translation across frames from a single RGB camera, while being robust to heavy occlusions. Our method is built on two key insights. First, we condition our neural field reconstructions for human and object on per-frame SMPL model estimates obtained by pre-fitting SMPL to a video sequence. This improves neural reconstruction accuracy and produces coherent relative translation across frames. Second, human and object motion from visible frames provides valuable information to infer the occluded object. We propose a novel transformer-based neural network that explicitly uses object visibility and human motion to leverage neighboring frames to make predictions for the occluded frames. Building on these insights, our method is able to track both human and object robustly even under occlusions. Experiments on two datasets show that our method significantly improves over the state-of-the-art methods. Our code and pretrained models are available at: https://virtualhumans.mpi-inf.mpg.de/VisTracker.

----

## [453] Transformer-based Unified Recognition of Two Hands Manipulating Objects

**Authors**: *Hoseong Cho, Chanwoo Kim, Jihyeon Kim, Seongyeong Lee, Elkhan Ismayilzada, Seungryul Baek*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00462](https://doi.org/10.1109/CVPR52729.2023.00462)

**Abstract**:

Understanding the hand-object interactions from an egocentric video has received a great attention recently. So far, most approaches are based on the convolutional neural network (CNN) features combined with the temporal encoding via the long shortterm memory (LSTM) or graph convolution network (GCN) to provide the unified understanding of two hands, an object and their interactions. In this paper, we propose the Transformer-based unified framework that provides better understanding of two hands manipulating objects. In our framework, we insert the whole image depicting two hands, an object and their interactions as input and jointly estimate 3 information from each frame: poses of two hands, pose of an object and object types. Afterwards, the action class defined by the hand-object interactions is predicted from the entire video based on the estimated information combined with the contact map that encodes the interaction between two hands and an object. Experiments are conducted on H2O and FPHA benchmark datasets and we demonstrated the superiority of our method achieving the state-of-the-art accuracy. Ablative studies further demonstrate the effectiveness of each proposed module.

----

## [454] HuManiFlow: Ancestor-Conditioned Normalising Flows on SO(3) Manifolds for Human Pose and Shape Distribution Estimation

**Authors**: *Akash Sengupta, Ignas Budvytis, Roberto Cipolla*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00463](https://doi.org/10.1109/CVPR52729.2023.00463)

**Abstract**:

Monocular 3D human pose and shape estimation is an illposed problem since multiple 3D solutions can explain a 2D image of a subject. Recent approaches predict a probability distribution over plausible 3D pose and shape parameters conditioned on the image. We show that these approaches exhibit a trade-off between three key properties: (i) accuracy - the likelihood of the ground-truth 3D solution under the predicted distribution, (ii) sample-input consistency - the extent to which 3D samples from the predicted distribution match the visible 2D image evidence, and (iii) sample diversity - the range of plausible 3D solutions modelled by the predicted distribution. Our method, HuManiFlow, predicts simultaneously accurate, consistent and diverse distributions. We use the human kinematic tree to factorise full body pose into ancestor-conditioned per-body-part pose distributions in an autoregressive manner. Per-body-part distributions are implemented using normalising flows that respect the manifold structure of SO(3), the Lie group of per-body-part poses. We show that ill-posed, but ubiquitous, 3D point estimate losses reduce sample diversity, and employ only probabilistic training losses. HuManiFlow outperforms state-of-the-art probabilistic approaches on the 3DPW and SSP-3D datasets.

----

## [455] 3D Human Pose Estimation with Spatio-Temporal Criss-Cross Attention

**Authors**: *Zhenhua Tang, Zhaofan Qiu, Yanbin Hao, Richang Hong, Ting Yao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00464](https://doi.org/10.1109/CVPR52729.2023.00464)

**Abstract**:

Recent transformer-based solutions have shown great success in 3D human pose estimation. Nevertheless, to calculate the joint-to-joint affinity matrix, the computational cost has a quadratic growth with the increasing number of joints. Such drawback becomes even worse especially for pose estimation in a video sequence, which necessitates spatio-temporal correlation spanning over the entire video. In this paper, we facilitate the issue by decomposing correlation learning into space and time, and present a novel Spatio-Temporal Criss-cross attention (STC) block. Technically, STC first slices its input feature into two partitions evenly along the channel dimension, followed by performing spatial and temporal attention respectively on each partition. STC then models the interactions between joints in an identical frame and joints in an identical trajectory simultaneously by concatenating the outputs from attention layers. On this basis, we devise STCFormer by stacking multiple STC blocks and further integrate a new Structure-enhanced Positional Embedding (SPE) into STCFormer to take the structure of human body into consideration. The embedding function consists of two components: spatio-temporal convolution around neighboring joints to capture local structure, and part-aware embedding to indicate which part each joint belongs to. Extensive experiments are conducted on Human3.6M and MPI-INF-3DHP benchmarks, and superior results are reported when comparing to the state-of-the-art approaches. More remarkably, STCFormer achieves to-date the best published performance: 40.5mm P1 error on the challenging Human3.6M dataset.

----

## [456] GFPose: Learning 3D Human Pose Prior with Gradient Fields

**Authors**: *Hai Ci, Mingdong Wu, Wentao Zhu, Xiaoxuan Ma, Hao Dong, Fangwei Zhong, Yizhou Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00465](https://doi.org/10.1109/CVPR52729.2023.00465)

**Abstract**:

Learning 3D human pose prior is essential to human-centered AI. Here, we present GFPose, a versatile framework to model plausible 3D human poses for various applications. At the core of GFPose is a time-dependent score network, which estimates the gradient on each body joint and progressively denoises the perturbed 3D human pose to match a given task specification. During the denoising process, GFPose implicitly incorporates pose priors in gradients and unifies various discriminative and generative tasks in an elegant framework. Despite the simplicity, GFPose demonstrates great potential in several downstream tasks. Our experiments empirically show that 1) as a multi-hypothesis pose estimator, GFPose outperforms existing SOTAs by 20% on Human3.6M dataset. 2) as a single-hypothesis pose estimator, GFPose achieves comparable results to deterministic SOTAs, even with a vanilla backbone. 3) GFPose is able to produce diverse and realistic samples in pose denoising, completion and generation tasks.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page https://sites.google.com/view/gfpose/

----

## [457] JRDB-Pose: A Large-Scale Dataset for Multi-Person Pose Estimation and Tracking

**Authors**: *Edward Vendrow, Duy-Tho Le, Jianfei Cai, Hamid Rezatofighi*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00466](https://doi.org/10.1109/CVPR52729.2023.00466)

**Abstract**:

Autonomous robotic systems operating in human environments must understand their surroundings to make accurate and safe decisions. In crowded human scenes with close-up human-robot interaction and robot navigation, a deep understanding of surrounding people requires reasoning about human motion and body dynamics over time with human body pose estimation and tracking. However, existing datasets captured from robot platforms either do not provide pose annotations or do not reflect the scene distribution of social robots. In this paper, we introduce JRDB-Pose, a large-scale dataset and benchmark for multi-person pose estimation and tracking. JRDB-Pose extends the existing JRDB which includes videos captured from a social navigation robot in a university campus environment, containing challenging scenes with crowded indoor and outdoor locations and a diverse range of scales and occlusion types. JRDB-Pose provides human pose annotations with per-keypoint occlusion labels and track IDs consistent across the scene and with existing annotations in JRDB. We conduct a thorough experimental study of state-of-the-art multi-person pose estimation and tracking methods on JRDB-Pose, showing that our dataset imposes new challenges for the existing methods. JRDB-Pose is available at https://jrdb.erc.monash.edu/.

----

## [458] Analyzing and Diagnosing Pose Estimation with Attributions

**Authors**: *Qiyuan He, Linlin Yang, Kerui Gu, Qiuxia Lin, Angela Yao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00467](https://doi.org/10.1109/CVPR52729.2023.00467)

**Abstract**:

We present Pose Integrated Gradient (PoseIG), the first interpretability technique designed for pose estimation. We extend the concept of integrated gradients for pose estimation to generate pixel-level attribution maps. To enable comparison across different pose frameworks, we unify different pose outputs into a common output space, along with a likelihood approximation function for gradient back-propagation. To complement the qualitative insight from the attribution maps, we propose three indices for quantitative analysis. With these tools, we systematically compare different pose estimation frameworks to understand the impacts of network design, backbone and auxiliary tasks. Our analysis reveals an interesting shortcut of the knuckles (MCP joints) for hand pose estimation and an under-explored inversion error for keypoints in body pose estimation. Project page and code: https://qy-h00.github.io/poseig/.

----

## [459] Shape-Constraint Recurrent Flow for 6D Object Pose Estimation

**Authors**: *Yang Hai, Rui Song, Jiaojiao Li, Yinlin Hu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00468](https://doi.org/10.1109/CVPR52729.2023.00468)

**Abstract**:

Most recent 6D object pose methods use 2D optical flow to refine their results. However, the general optical flow methods typically do not consider the target's 3D shape information during matching, making them less effective in 6D object pose estimation. In this work, we propose a shape-constraint recurrent matching framework for 6D object pose estimation. We first compute a pose-induced flow based on the displacement of 2D reprojection between the initial pose and the currently estimated pose, which embeds the target's 3D shape implicitly. Then we use this pose-induced flow to construct the correlation map for the following matching iterations, which reduces the matching space significantly and is much easier to learn. Further-more, we use networks to learn the object pose based on the current estimated flow, which facilitates the computation of the pose-induced flow for the next iteration and yields an end-to-end system for object pose. Finally, we optimize the optical flow and object pose simultaneously in a recurrent manner. We evaluate our method on three challenging 6D object pose datasets and show that it outperforms the state of the art significantly in both accuracy and efficiency.

----

## [460] TexPose: Neural Texture Learning for Self-Supervised 6D Object Pose Estimation

**Authors**: *Hanzhi Chen, Fabian Manhardt, Nassir Navab, Benjamin Busam*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00469](https://doi.org/10.1109/CVPR52729.2023.00469)

**Abstract**:

In this paper, we introduce neural texture learning for 6D object pose estimation from synthetic data and a few unlabelled real images. Our major contribution is a novel learning scheme which removes the drawbacks of previous works, namely the strong dependency on co-modalities or additional refinement. These have been previously necessary to provide training signals for convergence. We formulate such a scheme as two sub-optimisation problems on texture learning and pose learning. We separately learn to predict realistic texture of objects from real image collections and learn pose estimation from pixel-perfect synthetic data. Combining these two capabilities allows then to synthesise photorealistic novel views to supervise the pose estimator with accurate geometry. To alleviate pose noise and segmentation imperfection present during the texture learning phase, we propose a surfel-based adversarial training loss together with texture regularisation from synthetic data. We demonstrate that the proposed approach significantly outperforms the recent state-of-the-art methods without ground-truth pose annotations and demonstrates substantial generalisation improvements towards unseen scenes. Remarkably, our scheme improves the adopted pose estimators substantially even when initialised with much inferior performance.

----

## [461] Hi-LASSIE: High-Fidelity Articulated Shape and Skeleton Discovery from Sparse Image Ensemble

**Authors**: *Chun-Han Yao, Wei-Chih Hung, Yuanzhen Li, Michael Rubinstein, Ming-Hsuan Yang, Varun Jampani*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00470](https://doi.org/10.1109/CVPR52729.2023.00470)

**Abstract**:

Automatically estimating 3D skeleton, shape, camera viewpoints, and part articulation from sparse in-the-wild image ensembles is a severely under-constrained and challenging problem. Most prior methods rely on large-scale image datasets, dense temporal correspondence, or human annotations like camera pose, 2D keypoints, and shape templates. We propose Hi-LASSIE, which performs 3D articulated reconstruction from only 20–30 online images in the wild without any user-defined shape or skeleton templates. We follow the recent work of LASSIE that tackles a similar problem setting and make two significant advances. First, instead of relying on a manually annotated 3D skeleton, we automatically estimate a class-specific skeleton from the selected reference image. Second, we improve the shape reconstructions with novel instance-specific optimization strategies that allow reconstructions to faithful fit on each instance while preserving the class-specific priors learned across all images. Experiments on in-the-wild image ensembles show that Hi-LASSIE obtains higher fidelity state-of-the-art 3D reconstructions despite requiring minimum user input. Project page: chhankyao.github.io/hi-lassie/

----

## [462] Revisiting Rolling Shutter Bundle Adjustment: Toward Accurate and Fast Solution

**Authors**: *Bangyan Liao, Delin Qu, Yifei Xue, Huiqing Zhang, Yizhen Lao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00471](https://doi.org/10.1109/CVPR52729.2023.00471)

**Abstract**:

We propose an accurate and fast bundle adjustment (BA) solution that estimates the 6-DoF pose with an independent RS model of the camera and the geometry of the environment based on measurements from a rolling shutter (RS) camera. This tackles the challenges in the existing works, namely, relying on high frame rate video as input, restrictive assumptions on camera motion and poor efficiency. To this end, we first verify the positive influence of the image point normalization to RSBA. Then we present a novel visual residual covariance model to standardize the reprojection error during RSBA, which consequently improves the overall accuracy. Besides, we demonstrate the combination of Normalization and covariance standardization Weighting in RSBA (NW-RSBA) can avoid common planar degeneracy without the need to constrain the filming manner. Finally, we propose an acceleration strategy for NW-RSBA based on the sparsity of its Jacobian matrix and Schur complement. The extensive synthetic and real data experiments verify the effectiveness and efficiency of the proposed solution over the state-of-the-art works.

----

## [463] Revisiting the P3P Problem

**Authors**: *Yaqing Ding, Jian Yang, Viktor Larsson, Carl Olsson, Kalle Åström*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00472](https://doi.org/10.1109/CVPR52729.2023.00472)

**Abstract**:

One of the classical multi-view geometry problems is the so called P3P problem, where the absolute pose of a calibrated camera is determined from three 2D-to-3D correspondences. Since these solvers form a critical component of many vision systems (e.g. in localization and Structure-from-Motion), there have been significant effort in developing faster and more stable algorithms. While the current state-of-the-art solvers are both extremely fast and stable, there still exist configurations where they break down. In this paper we algebraically formulate the problem as finding the intersection of two conics. With this formulation we are able to analytically characterize the real roots of the polynomial system and employ a tailored solution strategy for each problem instance. The result is a fast and stable solver, that is able to correctly solve cases where competing methods might fail. Our experimental evaluation shows that we outperform the current state-of-the-art methods both in terms of speed and success rate.

----

## [464] Common Pets in 3D: Dynamic New-View Synthesis of Real-Life Deformable Categories

**Authors**: *Samarth Sinha, Roman Shapovalov, Jeremy Reizenstein, Ignacio Rocco, Natalia Neverova, Andrea Vedaldi, David Novotný*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00473](https://doi.org/10.1109/CVPR52729.2023.00473)

**Abstract**:

Obtaining photorealistic reconstructions of objects from sparse views is inherently ambiguous and can only be achieved by learning suitable reconstruction priors. Earlier works on sparse rigid object reconstruction successfully learned such priors from large datasets such as CO3D. In this paper, we extend this approach to dynamic objects. We use cats and dogs as a representative example and introduce Common Pets in 3D (CoP3D), a collection of crowd-sourced videos of approximately 4,200 distinct pets. CoP3D is one of the first large-scale datasets for benchmarking non-rigid 3D reconstruction “in the wild”. We also propose Tracker-NeRF, a method for learning 4D reconstruction from our dataset. At test time, given a small number of video frames of an unseen object, Tracker-NeRF predicts the trajectories of its 3D points and generates new views, interpolating viewpoint and time. Results on CoP3D reveal significantly better non-rigid new-view synthesis performance than existing baselines. The data will be available on the project webpage: https://cop3d.github.io/.

----

## [465] MobileBrick: Building LEGO for 3D Reconstruction on Mobile Devices

**Authors**: *Kejie Li, Jia-Wang Bian, Robert Castle, Philip H. S. Torr, Victor Adrian Prisacariu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00474](https://doi.org/10.1109/CVPR52729.2023.00474)

**Abstract**:

High-quality 3D ground-truth shapes are critical for 3D object reconstruction evaluation. However, it is difficult to create a replica of an object in reality, and even 3D reconstructions generated by 3D scanners have artefacts that cause biases in evaluation. To address this issue, we introduce a novel multi-view RGBD dataset captured using a mobile device, which includes highly precise 3D ground-truth annotations for 153 object models featuring a diverse set of 3D structures. We obtain precise 3D ground-truth shape without relying on high-end 3D scanners by utilising LEGO models with known geometry as the 3D structures for image capture. The distinct data modality offered by high-resolution RGB images and low-resolution depth maps captured on a mobile device, when combined with precise 3D geometry annotations, presents a unique opportunity for future research on high-fidelity 3D reconstruction. Furthermore, we evaluate a range of 3D reconstruction algorithms on the proposed dataset.

----

## [466] EFEM: Equivariant Neural Field Expectation Maximization for 3D Object Segmentation Without Scene Supervision

**Authors**: *Jiahui Lei, Congyue Deng, Karl Schmeckpeper, Leonidas J. Guibas, Kostas Daniilidis*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00475](https://doi.org/10.1109/CVPR52729.2023.00475)

**Abstract**:

We introduce Equivariant Neural Field Expectation Maximization (EFEM), a simple, effective, and robust geometric algorithm that can segment objects in 3D scenes without annotations or training on scenes. We achieve such unsupervised segmentation by exploiting single object shape priors. We make two novel steps in that direction. First, we introduce equivariant shape representations to this problem to eliminate the complexity induced by the variation in object configuration. Second, we propose a novel EM algorithm that can iteratively refine segmentation masks using the equivariant shape prior. We collect a novel real dataset Chairs and Mugs that contains various object configurations and novel scenes in order to verify the effectiveness and robustness of our method. Experimental results demonstrate that our method achieves consistent and robust performance across different scenes where the (weakly) supervised methods may fail. Code and data available at https://www.cis.upenn.edu/~leijh/projects/EFEM

----

## [467] GINA-3D: Learning to Generate Implicit Neural Assets in the Wild

**Authors**: *Bokui Shen, Xinchen Yan, Charles R. Qi, Mahyar Najibi, Boyang Deng, Leonidas J. Guibas, Yin Zhou, Dragomir Anguelov*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00476](https://doi.org/10.1109/CVPR52729.2023.00476)

**Abstract**:

Modeling the 3D world from sensor data for simulation is a scalable way of developing testing and validation environments for robotic learning problems such as autonomous driving. However, manually creating or recreating real-world-like environments is difficult, expensive, and not scalable. Recent generative model techniques have shown promising progress to address such challenges by learning 3D assets using only plentiful 2D images - but still suffer limitations as they leverage either human-curated image datasets or renderings from manually-created synthetic 3D environments. In this paper, we introduce GINA-3D, a generative model that uses real-world driving data from camera and LiDAR sensors to create realistic 3D implicit neural assets of diverse vehicles and pedestrians. Compared to the existing image datasets, the real-world driving setting poses new challenges due to occlusions, lighting-variations and longtail distributions. GINA-3D tackles these challenges by decoupling representation learning and generative modeling into two stages with a learned triplane latent structure, inspired by recent advances in generative modeling of images. To evaluate our approach, we construct a large-scale object-centric dataset containing over 520K images of vehicles and pedestrians from the Waymo Open Dataset, and a new set of 80K images of longtail instances such as construction equipment, garbage trucks, and cable cars. We compare our model with existing approaches and demonstrate that it achieves state-of-the-art performance in quality and diversity for both generated images and geometries.

----

## [468] Habitat-Matterport 3D Semantics Dataset

**Authors**: *Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Théophile Gervet, John M. Turner, Aaron Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, Alexander William Clegg, Devendra Singh Chaplot*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00477](https://doi.org/10.1109/CVPR52729.2023.00477)

**Abstract**:

We present the Habitat-Matterport 3D Semantics (HM3DSem) dataset. HM3DSem is the largest dataset of 3D real-world spaces with densely annotated semantics that is currently available to the academic community. It consists of 142,646 object instance annotations across 216 3D spaces and 3,100 rooms within those spaces. The scale, quality, and diversity of object annotations far exceed those of prior datasets. A key difference setting apart HM3DSem from other datasets is the use of texture information to annotate pixel-accurate object boundaries. We demonstrate the effectiveness of HM3DSem dataset for the Object Goal Navigation task using different methods. Policies trained using HM3DSem perform outperform those trained on prior datasets. Introduction of HM3DSem in the Habitat ObjectNav Challenge lead to an increase in participation from 400 submissions in 2021 to 1022 submissions in 2022. Project page: https://aihabitat.org/datasets/hm3d-semantics/

----

## [469] BUOL: A Bottom-Up Framework with Occupancy-Aware Lifting for Panoptic 3D Scene Reconstruction From a Single Image

**Authors**: *Tao Chu, Pan Zhang, Qiong Liu, Jiaqi Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00478](https://doi.org/10.1109/CVPR52729.2023.00478)

**Abstract**:

Understanding and modeling the 3D scene from a single image is a practical problem. A recent advance proposes a panoptic 3D scene reconstruction task that performs both 3D reconstruction and 3D panoptic segmentation from a single image. Although having made substantial progress, recent works only focus on top-down approaches that fill 2D instances into 3D voxels according to estimated depth, which hinders their performance by two ambiguities. (1) instance-channel ambiguity: The variable ids of instances in each scene lead to ambiguity during filling voxel channels with 2D information, confusing the following 3D refinement. (2) voxel-reconstruction ambiguity: 2D-to-3D lifting with estimated single view depth only propagates 2D information onto the surface of 3D regions, leading to ambiguity during the reconstruction of regions behind the frontal view surface. In this paper, we propose BUOL, a Bottom-Up framework with Occupancy-aware Lifting to address the two issues for panoptic 3D scene reconstruction from a single image. For instance-channel ambiguity, a bottom-up framework lifts 2D information to 3D voxels based on deterministic semantic assignments rather than arbitrary instance id assignments. The 3D voxels are then refined and grouped into 3D instances according to the predicted 2D instance centers. For voxel-reconstruction ambiguity, the estimated multi-plane occupancy is leveraged together with depth to fill the whole regions of things and stuff. Our method shows a tremendous performance advantage over state-of-the-art methods on synthetic dataset 3D-Front and real-world dataset Matterport3D. Code and models will be released.

----

## [470] Panoptic Compositional Feature Field for Editable Scene Rendering with Network-Inferred Labels via Metric Learning

**Authors**: *Xinhua Cheng, Yanmin Wu, Mengxi Jia, Qian Wang, Jian Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00479](https://doi.org/10.1109/CVPR52729.2023.00479)

**Abstract**:

Despite neural implicit representations demonstrating impressive high-quality view synthesis capacity, decom-posing such representations into objects for instance-level editing is still challenging. Recent works learn object-compositional representations supervised by ground truth instance annotations and produce promising scene editing results. However, ground truth annotations are manually labeled and expensive in practice, which limits their usage in real-world scenes. In this work, we attempt to learn an object-compositional neural implicit representation for editable scene rendering by leveraging labels inferred from the off-the-shelf 2D panoptic segmentation networks instead of the ground truth annotations. We propose a novel framework named Panoptic Compositional Feature Field (PCFF), which introduces an instance quadruplet metric learning to build a discriminating panoptic feature space for reliable scene editing. In addition, we propose semantic-related strategies to further exploit the correlations between semantic and appearance attributes for achieving better rendering results. Experiments on multiple scene datasets including ScanNet, Replica, and ToyDesk demonstrate that our proposed method achieves superior performance for novel view synthesis and produces convincing real-world scene editing results.

----

## [471] A Light Touch Approach to Teaching Transformers Multi-view Geometry

**Authors**: *Yash Bhalgat, João F. Henriques, Andrew Zisserman*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00480](https://doi.org/10.1109/CVPR52729.2023.00480)

**Abstract**:

Transformers are powerful visual learners, in large part due to their conspicuous lack of manually-specified priors. This flexibility can be problematic in tasks that involve multiple-view geometry, due to the near-infinite possible variations in 3D shapes and viewpoints (requiring flexibility), and the precise nature of projective geometry (obeying rigid laws). To resolve this conundrum, we propose a “light touch” approach, guiding visual Transformers to learn multiple-view geometry but allowing them to break free when needed. We achieve this by using epipolar lines to guide the Transformer's cross-attention maps during training, penalizing attention values outside the epipolar lines and encouraging higher attention along these lines since they contain geometrically plausible matches. Unlike previous methods, our proposal does not require any camera pose information at test-time. We focus on pose-invariant object instance retrieval, where standard Transformer networks struggle, due to the large differences in viewpoint between query and retrieved images. Experimentally, our method outperforms state-of-the-art approaches at object retrieval, without needing pose information at test-time.

----

## [472] Learning to Render Novel Views from Wide-Baseline Stereo Pairs

**Authors**: *Yilun Du, Cameron Smith, Ayush Tewari, Vincent Sitzmann*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00481](https://doi.org/10.1109/CVPR52729.2023.00481)

**Abstract**:

We introduce a method for novel view synthesis given only a single wide-baseline stereo image pair. In this challenging regime, 3D scene points are regularly observed only once, requiring prior-based reconstruction of scene geometry and appearance. We find that existing approaches to novel view synthesis from sparse observations fail due to recovering incorrect 3D geometry and due to the high cost of differentiable rendering that precludes their scaling to large-scale training. We take a step towards resolving these shortcomings by formulating a multi-view transformer encoder, proposing an efficient, image-space epipolar line sampling scheme to assemble image features for a target ray, and a lightweight cross-attention-based renderer. Our contributions enable training of our method on a large-scale real-world dataset of indoor and outdoor scenes. We demonstrate that our method learns powerful multi-view geometry priors while reducing the rendering time. We conduct extensive comparisons on held-out test scenes across two real-world datasets, significantly outperforming prior work on novel view synthesis from sparse image observations and achieving multi-viewconsistent novel view synthesis.

----

## [473] Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo

**Authors**: *Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, Andrés Bruhn*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00482](https://doi.org/10.1109/CVPR52729.2023.00482)

**Abstract**:

While recent methods for motion and stereo estimation recover an unprecedented amount of details, such highly detailed structures are neither adequately reflected in the data of existing benchmarks nor their evaluation methodology. Hence, we introduce Spring - a large, high-resolution, high-detail, computer-generated benchmark for scene flow, optical flow, and stereo. Based on rendered scenes from the open-source Blender movie “Spring”, it provides photo-realistic HD datasets with state-of-the-art visual effects and ground truth training data. Furthermore, we provide a website to upload, analyze and compare results. Using a novel evaluation methodology based on a super-resolved UHD ground truth, our Spring benchmark can assess the quality of fine structures and provides further detailed performance statistics on different image regions. Regarding the number of ground truth frames, Spring is 60× larger than the only scene flow benchmark, KITTI 2015, and 15× larger than the well-established MPI Sintel optical flow benchmark. Initial results for recent methods on our benchmark show that estimating fine details is indeed challenging, as their accuracy leaves significant room for improvement. The Spring benchmark and the corresponding datasets are available at http://spring-benchmark.org.

----

## [474] EventNeRF: Neural Radiance Fields from a Single Colour Event Camera

**Authors**: *Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, Vladislav Golyanik*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00483](https://doi.org/10.1109/CVPR52729.2023.00483)

**Abstract**:

Asynchronously operating event cameras find many applications due to their high dynamic range, vanishingly low motion blur, low latency and low data bandwidth. The field saw remarkable progress during the last few years, and existing event-based 3D reconstruction approaches recover sparse point clouds of the scene. However, such sparsity is a limiting factor in many cases, especially in computer vision and graphics, that has not been addressed satisfactorily so far. Accordingly, this paper proposes the first approach for 3D-consistent, dense and photorealistic novel view synthesis using just a single colour event stream as input. At its core is a neural radiance field trained entirely in a self-supervised manner from events while preserving the original resolution of the colour event channels. Next, our ray sampling strategy is tailored to events and allows for data-efficient training. At test, our method produces results in the RGB space at unprecedented quality. We evaluate our method qualitatively and numerically on several challenging synthetic and real scenes and show that it produces significantly denser and more visually appealing renderings than the existing methods. We also demonstrate robustness in challenging scenarios with fast motion and under low lighting conditions. We release the newly recorded dataset and our source code to facilitate the research field, see https://4dqv.mpi-inf.mpg.de/EventNeRF.

----

## [475] LightedDepth: Video Depth Estimation in Light of Limited Inference View Angles

**Authors**: *Shengjie Zhu, Xiaoming Liu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00484](https://doi.org/10.1109/CVPR52729.2023.00484)

**Abstract**:

Video depth estimation infers the dense scene depth from immediate neighboring video frames. While recent works consider it a simplified structure-from-motion (SfM) problem, it still differs from the SfM in that significantly fewer view angels are available in inference. This setting, however, suits the mono-depth and optical flow estimation. This observation motivates us to decouple the video depth estimation into two components, a normalized pose estimation over af lowmap and a logged residual depth estimation over a mono-depth map. The two parts are unified with an efficient off-the-shelf scale alignment algorithm. Additionally, we stabilize the indoor two-view pose estimation by including additional projection constraints and ensuring sufficient camera translation. Though a two-view algorithm, we validate the benefit of the decoupling with the substantial performance improvement over multi-view iterative prior works on indoor and outdoor datasets. Codes and models are available at https://github.com/ShngJZ/LightedDepth.

----

## [476] Generating Aligned Pseudo-Supervision from Non-Aligned Data for Image Restoration in Under-Display Camera

**Authors**: *Ruicheng Feng, Chongyi Li, Huaijin G. Chen, Shuai Li, Jinwei Gu, Chen Change Loy*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00485](https://doi.org/10.1109/CVPR52729.2023.00485)

**Abstract**:

Due to the difficulty in collecting large-scale and perfectly aligned paired training data for Under-Display Camera (UDC) image restoration, previous methods resort to monitor-based image systems or simulation-based methods, sacrificing the realness of the data and introducing domain gaps. In this work, we revisit the classic stereo setup for training data collection – capturing two images of the same scene with one UDC and one standard camera. The key idea is to “copy” details from a high-quality reference image and “paste” them on the UDC image. While being able to generate real training pairs, this setting is susceptible to spatial misalignment due to perspective and depth of field changes. The problem is further compounded by the large domain discrepancy between the UDC and normal images, which is unique to UDC restoration. In this paper, we mitigate the non-trivial domain discrepancy and spatial misalignment through a novel Transformer-based framework that generates well-aligned yet high-quality target data for the corresponding UDC input. This is made possible through two carefully designed components, namely, the Domain Alignment Module (DAM) and Geometric Alignment Module (GAM), which encourage robust and accurate discovery of correspondence between the UDC and normal views. Extensive experiments show that high-quality and well-aligned pseudo UDC training pairs are beneficial for training a robust restoration network. Code and the dataset are available at https://github.com/jnjaby/AlignFormer.

----

## [477] Spatio-Focal Bidirectional Disparity Estimation from a Dual-Pixel Image

**Authors**: *Donggun Kim, Hyeonjoong Jang, Inchul Kim, Min H. Kim*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00486](https://doi.org/10.1109/CVPR52729.2023.00486)

**Abstract**:

Dual-pixel photography is monocular RGB-D photography with an ultra-high resolution, enabling many applications in computational photography. However, there are still several challenges to fully utilizing dual-pixel photography. Unlike the conventional stereo pair, the dual pixel exhibits a bidirectional disparity that includes positive and negative values, depending on the focus plane depth in an image. Furthermore, capturing a wide range of dual-pixel disparity requires a shallow depth of field, resulting in a severely blurred image, degrading depth estimation performance. Recently, several data-driven approaches have been proposed to mitigate these two challenges. However, due to the lack of the ground-truth dataset of the dual-pixel disparity, existing data-driven methods estimate either inverse depth or blurriness map. In this work, we propose a self-supervised learning method that learns bidirectional disparity by utilizing the nature of anisotropic blur kernels in dual-pixel photography. We observe that the dual-pixel left/right images have reflective-symmetric anisotropic kernels, so their sum is equivalent to that of a conventional image. We take a self-supervised training approach with the novel kernel-split symmetry loss accounting for the phenomenon. Our method does not rely on a training dataset of dual-pixel disparity that does not exist yet. Our method can estimate a complete disparity map with respect to the focus-plane depth from a dual-pixel image, outperforming the baseline dual-pixel methods.

----

## [478] Trap Attention: Monocular Depth Estimation with Manual Traps

**Authors**: *Chao Ning, Hongping Gan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00487](https://doi.org/10.1109/CVPR52729.2023.00487)

**Abstract**:

Predicting a high quality depth map from a single image is a challenging task, because it exists infinite possibility to project a 2D scene to the corresponding 3D scene. Recently, some studies introduced multi-head attention (MHA) modules to perform long-range interaction, which have shown significant progress in regressing the depth maps. The main functions of MHA can be loosely summarized to capture long-distance information and report the attention map by the relationship between pixels. However, due to the quadratic complexity of MHA, these methods can not leverage MHA to compute depth features in high resolution with an appropriate computational complexity. In this paper, we exploit a depth-wise convolution to obtain long-range information, and propose a novel trap attention, which sets some traps on the extended space for each pixel, and forms the attention mechanism by the feature retention ratio of convolution window, resulting in that the quadratic computational complexity can be converted to linear form. Then we build an encoder-decoder trap depth estimation network, which introduces a vision transformer as the encoder, and uses the trap attention to estimate the depth from single image in the decoder. Extensive experimental results demonstrate that our proposed network can outperform the state-of-the-art methods in monocular depth estimation on datasets NYU Depth-v2 and KITTI, with significantly reduced number of parameters. Code is available at: https://github.com/ICSResearch/TrapAttention.

----

## [479] Accelerated Coordinate Encoding: Learning to Relocalize in Minutes Using RGB and Poses

**Authors**: *Eric Brachmann, Tommaso Cavallari, Victor Adrian Prisacariu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00488](https://doi.org/10.1109/CVPR52729.2023.00488)

**Abstract**:

Learning-based visual relocalizers exhibit leading pose accuracy, but require hours or days of training. Since training needs to happen on each new scene again, long training times make learning-based relocalization impractical for most applications, despite its promise of high accuracy. In this paper we show how such a system can actually achieve the same accuracy in less than 5 minutes. We start from the obvious: a relocalization network can be split in a scene-agnostic feature backbone, and a scene-specific prediction head. Less obvious: using an MLP prediction head allows us to optimize across thousands of view points simultaneously in each single training iteration. This leads to stable and extremely fast convergence. Furthermore, we substitute effective but slow end-to-end training using a robust pose solver with a curriculum over a reprojection loss. Our approach does not require privileged knowledge, such a depth maps or a 3D model, for speedy training. Overall, our approach is up to 300x faster in mapping than state-of-the-art scene coordinate regression, while keeping accuracy on par. Code is available: https://nianticlabs.github.io/ace

----

## [480] Energy-Efficient Adaptive 3D Sensing

**Authors**: *Brevin Tilmon, Zhanghao Sun, Sanjeev J. Koppal, Yicheng Wu, Georgios Evangelidis, Ramzi Zahreddine, Gurunandan Krishnan, Sizhuo Ma, Jian Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00489](https://doi.org/10.1109/CVPR52729.2023.00489)

**Abstract**:

Active depth sensing achieves robust depth estimation but is usually limited by the sensing range. Naively increasing the optical power can improve sensing range but induces eye-safety concerns for many applications, including autonomous robots and augmented reality. In this paper, we propose an adaptive active depth sensor that jointly optimizes range, power consumption, and eye-safety. The main observation is that we need not project light patterns to the entire scene but only to small regions of interest where depth is necessary for the application and passive stereo depth estimation fails. We theoretically compare this adaptive sensing scheme with other sensing strategies, such as full-frame projection, line scanning, and point scanning. We show that, to achieve the same maximum sensing distance, the proposed method consumes the least power while having the shortest (best) eye-safety distance. We implement this adaptive sensing scheme with two hardware prototypes, one with a phase-only spatial light modulator (SLM) and the other with a micro-electro-mechanical (MEMS) mirror and diffractive optical elements (DOE). Experimental results validate the advantage of our method and demonstrate its capability of acquiring higher quality geometry adaptively. Please see our project website for video results and code: https://btilmon.github.io/e3d.html.

----

## [481] Incremental 3D Semantic Scene Graph Prediction from RGB Sequences

**Authors**: *Shun-Cheng Wu, Keisuke Tateno, Nassir Navab, Federico Tombari*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00490](https://doi.org/10.1109/CVPR52729.2023.00490)

**Abstract**:

3D semantic scene graphs are a powerful holistic representation as they describe the individual objects and depict the relation between them. They are compact high-level graphs that enable many tasks requiring scene reasoning. In real-world settings, existing 3D estimation methods produce robust predictions that mostly rely on dense inputs. In this work, we propose a real-time framework that incrementally builds a consistent 3D semantic scene graph of a scene given an RGB image sequence. Our method consists of a novel incremental entity estimation pipeline and a scene graph prediction network. The proposed pipeline simultaneously reconstructs a sparse point map and fuses entity estimation from the input images. The proposed network estimates 3D semantic scene graphs with iterative message passing using multi-view and geometric features extracted from the scene entities. Extensive experiments on the 3RScan dataset show the effectiveness of the proposed method in this challenging task, outperforming state-of-the-art approaches. Our implementation is available at https://shunchengwu.github.io/MonoSSG.

----

## [482] Consistent Direct Time-of-Flight Video Depth Super-Resolution

**Authors**: *Zhanghao Sun, Wei Ye, Jinhui Xiong, Gyeongmin Choe, Jialiang Wang, Shuochen Su, Rakesh Ranjan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00491](https://doi.org/10.1109/CVPR52729.2023.00491)

**Abstract**:

Direct time-of-flight (dToF) sensors are promising for next-generation on-device 3D sensing. However, limited by manufacturing capabilities in a compact module, the dToF data has a low spatial resolution (e.g. 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\sim 20\times 30$</tex>
 for iPhone dToF), and it requires a super-resolution step before being passed to downstream tasks. In this paper, we solve this super-resolution problem by fusing the low-resolution dToF data with the corresponding high-resolution RGB guidance. Unlike the conventional RGB-guided depth enhancement approaches, which perform the fusion in a per-frame manner, we propose the first multi-frame fusion scheme to mitigate the spatial ambiguity resulting from the low-resolution dToF imaging. In addition, dToF sensors provide unique depth histogram information for each local patch, and we incorporate this dToF-specific feature in our network design to further alleviate spatial ambiguity. To evaluate our models on complex dynamic indoor environments and to provide a large-scale dToF sensor dataset, we introduce Dy-DToF, the first synthetic RGB-dToF video dataset that features dynamic objects and a realistic dToF simulator following the physical imaging process. We believe the methods and dataset are beneficial to a broad community as dToF depth sensing is becoming mainstream on mobile devices. Our code and data are publicly available. https://github.com/facebookresearch/DVSR/

----

## [483] Learning to Zoom and Unzoom

**Authors**: *Chittesh Thavamani, Mengtian Li, Francesco Ferroni, Deva Ramanan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00492](https://doi.org/10.1109/CVPR52729.2023.00492)

**Abstract**:

Many perception systems in mobile computing, autonomous navigation, and AR/VR face strict compute constraints that are particularly challenging for high-resolution input images. Previous works propose nonuniform downsamplers that “learn to zoom” on salient image regions, reducing compute while retaining task-relevant image information. However, for tasks with spatial labels (such as 2D/3D object detection and semantic segmentation), such distortions may harm performance. In this work (LZU), we “learn to zoom” in on the input image, compute spatial features, and then “unzoom” to revert any deformations. To enable efficient and differentiable unzooming, we approximate the zooming warp with a piecewise bilinear mapping that is invertible. LZU can be applied to any task with 2D spatial input and any model with 2D spatial features, and we demonstrate this versatility by evaluating on a variety of tasks and datasets: object detection on Argoverse-HD, semantic segmentation on Cityscapes, and monocular 3D object detection on nuScenes. Interestingly, we observe boosts in performance even when high-resolution sensor data is unavailable, implying that LZU can be used to “learn to upsample” as well. Code and additional visuals are available at https://tchittesh.github.io/lzu/.

----

## [484] FrustumFormer: Adaptive Instance-aware Resampling for Multi-view 3D Detection

**Authors**: *Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00493](https://doi.org/10.1109/CVPR52729.2023.00493)

**Abstract**:

The transformation of features from 2D perspective space to 3D space is essential to multi-view 3D object detection. Recent approaches mainly focus on the design of view transformation, either pixel-wisely lifting perspective view features into 3D space with estimated depth or grid-wisely constructing BEV features via 3D projection, treating all pixels or grids equally. However, choosing what to transform is also important but has rarely been discussed before. The pixels of a moving car are more informative than the pixels of the sky. To fully utilize the information contained in images, the view transformation should be able to adapt to different image regions according to their contents. In this paper, we propose a novel framework named FrustumFormer, which pays more attention to the features in instance regions via adaptive instance-aware resampling. Specifically, the model obtains instance frustums on the bird's eye view by leveraging image view object proposals. An adaptive occupancy mask within the instance frustum is learned to refine the instance location. Moreover, the temporal frustum intersection could further reduce the localization uncertainty of objects. Comprehensive experiments on the nuScenes dataset demonstrate the effectiveness of FrustumFormer, and we achieve a new state-of-the-art performance on the benchmark. Codes and models will be made available at https://github.com/Robertwyq/Frustum.

----

## [485] 3D Video Object Detection with Learnable Object-Centric Global Optimization

**Authors**: *Jiawei He, Yuntao Chen, Naiyan Wang, Zhaoxiang Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00494](https://doi.org/10.1109/CVPR52729.2023.00494)

**Abstract**:

We explore long-term temporal visual correspondence-based optimization for 3D video object detection in this work. Visual correspondence refers to one-to-one mappings for pixels across multiple images. Correspondence-based optimization is the cornerstone for 3D scene reconstruction but is less studied in 3D video object detection, because moving objects violate multi-view geometry constraints and are treated as outliers during scene reconstruction. We address this issue by treating objects as first-class citizens during correspondence-based optimization. In this work, we propose BA-Det, an end-to-end optimizable object detector with object-centric temporal correspondence learning and featuremetric object bundle adjustment. Empirically, we verify the effectiveness and efficiency of BA-Det for multiple baseline 3D detectors under various setups. Our BA-Det achieves SOTA performance on the large-scale Waymo Open Dataset (WOD) with only marginal computation cost. Our code is available at https://github.com/jiaweihe1996/BA-Det.

----

## [486] UniDistill: A Universal Cross-Modality Knowledge Distillation Framework for 3D Object Detection in Bird's-Eye View

**Authors**: *Shengchao Zhou, Weizhou Liu, Chen Hu, Shuchang Zhou, Chao Ma*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00495](https://doi.org/10.1109/CVPR52729.2023.00495)

**Abstract**:

In the field of 3D object detection for autonomous driving, the sensor portfolio including multi-modality and single-modality is diverse and complex. Since the multi-modal methods have system complexity while the accuracy of single-modal ones is relatively low, how to make a tradeoff between them is difficult. In this work, we propose a universal cross-modality knowledge distillation framework (UniDistill) to improve the performance of single-modality detectors. Specifically, during training, UniDistill projects the features of both the teacher and the student detector into Bird's-Eye-View (BEV), which is a friendly representation for different modalities. Then, three distillation losses are calculated to sparsely align the foreground features, helping the student learn from the teacher without introducing additional cost during inference. Taking advantage of the similar detection paradigm of different detectors in BEV, UniDistill easily supports LiDAR-to-camera, camera-to-LiDAR, fusion-to-LiDAR and fusion-to-camera distillation paths. Furthermore, the three distillation losses can filter the effect of misaligned background information and balance between objects of different sizes, improving the distillation effectiveness. Extensive experiments on nuScenes demonstrate that UniDistill effectively improves the mAP and NDS of student detectors by 2.0%~3.2%.

----

## [487] ARKitTrack: A New Diverse Dataset for Tracking Using Mobile RGB-D Data

**Authors**: *Haojie Zhao, Junsong Chen, Lijun Wang, Huchuan Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00496](https://doi.org/10.1109/CVPR52729.2023.00496)

**Abstract**:

Compared with traditional RGB-only visual tracking, few datasets have been constructed for RGB-D tracking. In this paper, we propose ARKitTrack, a new RGB-D tracking dataset for both static and dynamic scenes captured by consumer-grade LiDAR scanners equipped on Apple's iPhone and iPad. ARKitTrack contains 300 RGB-D sequences, 455 targets, and 229.7K video frames in total. Along with the bounding box annotations and frame-level attributes, we also annotate this dataset with 123.9K pixel-level target masks. Besides, the camera intrinsic and camera pose of each frame are provided for future developments. To demonstrate the potential usefulness of this dataset, we further present a unified baseline for both box-level and pixel-level tracking, which integrates RGB features with bird's-eye-view representations to better explore cross-modality 3D geometry. In-depth empirical analysis has verified that the ARKitTrack dataset can significantly facilitate RGB-D tracking and that the proposed baseline method compares favorably against the state of the arts. The code and dataset is available at https://arkittrack.github.io.

----

## [488] Deep Dive into Gradients: Better Optimization for 3D Object Detection with Gradient-Corrected IoU Supervision

**Authors**: *Qi Ming, Lingjuan Miao, Zhe Ma, Lin Zhao, Zhiqiang Zhou, Xuhui Huang, Yuanpei Chen, Yufei Guo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00497](https://doi.org/10.1109/CVPR52729.2023.00497)

**Abstract**:

Intersection-over-Union (IoU) is the most popular metric to evaluate regression performance in 3D object detection. Recently, there are also some methods applying IoU to the optimization of 3D bounding box regression. However, we demonstrate through experiments and mathematical proof that the 3D IoU loss suffers from abnormal gradient w.r.t. angular error and object scale, which further leads to slow convergence and suboptimal regression process, respectively. In this paper, we propose a Gradient-Corrected IoU (GCIoU) loss to achieve fast and accurate 3D bounding box regression. Specifically, a gradient correction strategy is designed to endow 3D IoU loss with a reasonable gradient. It ensures that the model converges quickly in the early stage of training, and helps to achieve fine-grained refinement of bounding boxes in the later stage. To solve suboptimal regression of 3D IoU loss for objects at different scales, we introduce a gradient rescaling strategy to adaptively optimize the step size. Finally, we integrate GCIoU Loss into multiple models to achieve stable performance gains and faster model convergence. Experiments on KITTI dataset demonstrate superiority of the proposed method. The code is available at https://github.com/ming71/GCIoU-loss.

----

## [489] SlowLiDAR: Increasing the Latency of LiDAR-Based Detection Using Adversarial Examples

**Authors**: *Han Liu, Yuhao Wu, Zhiyuan Yu, Yevgeniy Vorobeychik, Ning Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00498](https://doi.org/10.1109/CVPR52729.2023.00498)

**Abstract**:

LiDAR-based perception is a central component of autonomous driving, playing a key role in tasks such as vehicle localization and obstacle detection. Since the safety of LiDAR-based perceptual pipelines is critical to safe autonomous driving, a number of past efforts have investigated its vulnerability under adversarial perturbations of raw point cloud inputs. However, most such efforts have focused on investigating the impact of such perturbations on predictions (integrity), and little has been done to understand the impact on latency (availability), a critical concern for real-time cyber-physical systems. We present the first systematic investigation of the availability of LiDAR detection pipelines, and SlowLiDAR, an adversarial perturbation attack that maximizes LiDAR detection runtime. The attack overcomes the technical challenges posed by the non-differentiable parts of the LiDAR detection pipelines by using differentiable proxies and uses a novel loss function that effectively captures the impact of adversarial perturbations on the execution time of the pipeline. Extensive experimental results show that SlowLiDAR can significantly increase the latency of the six most popular LiDAR detection pipelines while maintaining imperceptibility 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at: https://github.com/WUSTL-CSPL/SlowLiDAR.

----

## [490] Normalizing Flow based Feature Synthesis for Outlier-Aware Object Detection

**Authors**: *Nishant Kumar, Sinisa Segvic, Abouzar Eslami, Stefan Gumhold*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00499](https://doi.org/10.1109/CVPR52729.2023.00499)

**Abstract**:

Real-world deployment of reliable object detectors is crucial for applications such as autonomous driving. However, general-purpose object detectors like Faster R-CNN are prone to providing overconfident predictions for outlier objects. Recent outlier-aware object detection approaches estimate the density of instance-wide features with class-conditional Gaussians and train on synthesized outlier features from their low-likelihood regions. However, this strategy does not guarantee that the synthesized outlier features will have a low likelihood according to the other class-conditional Gaussians. We propose a novel outlier-aware object detection framework that distinguishes outliers from inlier objects by learning the joint data distribution of all inlier classes with an invertible normalizing flow. The appropriate sampling of the flow model ensures that the synthesized outliers have a lower likelihood than inliers of all object classes, thereby modeling a better decision boundary between inlier and outlier objects. Our approach significantly outperforms the state-of-the-art for outlier-aware object detection on both image and video datasets.

----

## [491] OcTr: Octree-Based Transformer for 3D Object Detection

**Authors**: *Chao Zhou, Yanan Zhang, Jiaxin Chen, Di Huang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00500](https://doi.org/10.1109/CVPR52729.2023.00500)

**Abstract**:

A key challenge for LiDAR-based 3D object detection is to capture sufficient features from large scale 3D scenes especially for distant or/and occluded objects. Albeit recent efforts made by Transformers with the long sequence modeling capability, they fail to properly balance the accuracy and efficiency, suffering from inadequate receptive fields or coarse-grained holistic correlations. In this paper, we propose an Octree-based Transformer, named OcTr, to address this issue. It first constructs a dynamic octree on the hierarchical feature pyramid through conducting self-attention on the top level and then recursively propagates to the level below restricted by the octants, which captures rich global context in a coarse-to-fine manner while maintaining the computational complexity under control. Furthermore, for enhanced foreground perception, we propose a hybrid positional embedding, composed of the semantic-aware positional embedding and attention mask, to fully exploit semantic and geometry clues. Extensive experiments are conducted on the Waymo Open Dataset and KITTI Dataset, and OcTr reaches newly state-of-the-art results.

----

## [492] HypLiLoc: Towards Effective LiDAR Pose Regression with Hyperbolic Fusion

**Authors**: *Sijie Wang, Qiyu Kang, Rui She, Wei Wang, Kai Zhao, Yang Song, Wee Peng Tay*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00501](https://doi.org/10.1109/CVPR52729.2023.00501)

**Abstract**:

LiDAR relocalization plays a crucial role in many fields, including robotics, autonomous driving, and computer vision. LiDAR-based retrieval from a database typically incurs high computation storage costs and can lead to globally inaccurate pose estimations if the database is too sparse. On the other hand, pose regression methods take images or point clouds as inputs and directly regress global poses in an end-to-end manner. They do not perform database matching and are more computationally efficient than retrieval techniques. We propose HypLiLoc, a new model for LiDAR pose regression. We use two branched back-bones to extract 3D features and 2D projection features, respectively. We consider multi-modal feature fusion in both Euclidean and hyperbolic spaces to obtain more effective feature representations. Experimental results indicate that HypLiLoc achieves state-of-the-art performance in both outdoor and indoor datasets. We also conduct extensive ablation studies on the framework design, which demonstrate the effectiveness of multi-modal feature extraction and multi-space embedding. Our code is released at: https://github.com/sijieaaa/HypLiLoc

----

## [493] LiDAR2Map: In Defense of LiDAR-Based Semantic Map Construction Using Online Camera Distillation

**Authors**: *Song Wang, Wentong Li, Wenyu Liu, Xiaolu Liu, Jianke Zhu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00502](https://doi.org/10.1109/CVPR52729.2023.00502)

**Abstract**:

Semantic map construction under bird's-eye view (BEV) plays an essential role in autonomous driving. In contrast to camera image, LiDAR provides the accurate 3D observations to project the captured 3D features onto BEV space inherently. However, the vanilla LiDAR-based BEV feature often contains many indefinite noises, where the spatial features have little texture and semantic cues. In this paper, we propose an effective LiDAR-based method to build semantic map. Specifically, we introduce a BEV pyramid feature decoder that learns the robust multi-scale BEV features for semantic map construction, which greatly boosts the accuracy of the LiDAR-based method. To mitigate the defects caused by lacking semantic cues in LiDAR data, we present an online Camera-to-LiDAR distillation scheme to facilitate the semantic learning from image to point cloud. Our distillation scheme consists of feature-level and log it-level distillation to absorb the semantic information from camera in BEV. The experimental results on challenging nuScenes dataset demonstrate the efficacy of our proposed LiDAR2Map on semantic map construction, which significantly outperforms the previous LiDAR-based methods over 27.9% mIoU and even performs better than the state-of-the-art camera-based approaches. Source code is available at: https://github.com/songw-zjuILiDAR2Map.

----

## [494] MSF: Motion-guided Sequential Fusion for Efficient 3D Object Detection from Point Cloud Sequences

**Authors**: *Chenhang He, Ruihuang Li, Yabin Zhang, Shuai Li, Lei Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00503](https://doi.org/10.1109/CVPR52729.2023.00503)

**Abstract**:

Point cloud sequences are commonly used to accurately detect 3D objects in applications such as autonomous driving. Current top-performing multi-frame detectors mostly follow a Detect-and-Fuse framework, which extracts features from each frame of the sequence and fuses them to detect the objects in the current frame. However, this inevitably leads to redundant computation since adjacent frames are highly correlated. In this paper, we propose an efficient Motion-guided Sequential Fusion (MSF) method, which exploits the continuity of object motion to mine useful sequential contexts for object detection in the current frame. We first generate 3D proposals on the current frame and propagate them to preceding frames based on the estimated velocities. The points-of-interest are then pooled from the sequence and encoded as proposal features. A novel Bidi-rectional Feature Aggregation (BiFA) module is further proposed to facilitate the interactions of proposal features across frames. Besides, we optimize the point cloud pooling by a voxel-based sampling technique so that millions of points can be processed in several milliseconds. The proposed MSF method achieves not only better efficiency than other multi-frame detectors but also leading accuracy, with 83.12% and 78.30% mAP on the LEVEL1 and LEVEL2 test sets of Waymo Open Dataset, respectively. Codes can be found at https://github.com/skyhehe123/MSF.

----

## [495] SFD2: Semantic-Guided Feature Detection and Description

**Authors**: *Fei Xue, Ignas Budvytis, Roberto Cipolla*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00504](https://doi.org/10.1109/CVPR52729.2023.00504)

**Abstract**:

Visual localization is a fundamental task for various applications including autonomous driving and robotics. Prior methods focus on extracting large amounts of often redundant locally reliable features, resulting in limited efficiency and accuracy, especially in large-scale environments under challenging conditions. Instead, we propose to extract globally reliable features by implicitly embedding high-level semantics into both the detection and description processes. Specifically, our semantic-aware detector is able to detect keypoints from reliable regions (e.g. building, traffic lane) and suppress unreliable areas (e.g. sky, car) implicitly instead of relying on explicit semantic labels. This boosts the accuracy of keypoint matching by reducing the number of features sensitive to appearance changes and avoiding the need of additional segmentation networks at test time. Moreover, our descriptors are augmented with semantics and have stronger discriminative ability, providing more inliers at test time. Particularly, experiments on long-term large-scale visual localization Aachen Day Night and RobotCar-Seasons datasets demonstrate that our model outperforms previous local features and gives competitive accuracy to advanced matchers but is about 2 and 3 times faster when using 2k and 4k keypoints, respectively. Code is available at https://github.com/feixue94/sfd2.

----

## [496] Temporal Consistent 3D LiDAR Representation Learning for Semantic Perception in Autonomous Driving

**Authors**: *Lucas Nunes, Louis Wiesmann, Rodrigo Marcuzzi, Xieyuanli Chen, Jens Behley, Cyrill Stachniss*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00505](https://doi.org/10.1109/CVPR52729.2023.00505)

**Abstract**:

Semantic perception is a core building block in autonomous driving, since it provides information about the drivable space and location of other traffic participants. For learning-based perception, often a large amount of diverse training data is necessary to achieve high performance. Data labeling is usually a bottleneck for developing such methods, especially for dense prediction tasks, e.g., semantic segmentation or panoptic segmentation. For 3D Li-DAR data, the annotation process demands even more effort than for images. Especially in autonomous driving, point clouds are sparse, and objects appearance depends on its distance from the sensor, making it harder to acquire large amounts of labeled training data. This paper aims at taking an alternative path proposing a self-supervised representation learning method for 3D LiDAR data. Our approach exploits the vehicle motion to match objects across time viewed in different scans. We then train a model to maximize the point-wise feature similarities from points of the associated object in different scans, which enables to learn a consistent representation across time. The experimental results show that our approach performs better than previous state-of-the-art self-supervised representation learning methods when fine-tuning to different downstream tasks. We furthermore show that with only 10% of labeled data, a network pre-trained with our approach can achieve better performance than the same network trained from scratch with all labels for semantic segmentation on SemanticKITTI. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/PRBonn/TARL

----

## [497] Unsupervised 3D Point Cloud Representation Learning by Triangle Constrained Contrast for Autonomous Driving

**Authors**: *Bo Pang, Hongchi Xia, Cewu Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00506](https://doi.org/10.1109/CVPR52729.2023.00506)

**Abstract**:

Due to the difficulty of annotating the 3D LiDAR data of autonomous driving, an efficient unsupervised 3D representation learning method is important. In this paper, we design the Triangle Constrained Contrast (TriCC) framework tailored for autonomous driving scenes which learns 3D unsupervised representations through both the multimodal information and dynamic of temporal sequences. We treat one camera image and two LiDAR point clouds with different timestamps as a triplet. And our key design is the consistent constraint that automatically finds matching relationships among the triplet through “self-cycle” and learns representations from it. With the matching relations across the temporal dimension and modalities, we can further conduct a triplet contrast to improve learning efficiency. To the best of our knowledge, TriCC is the first framework that unifies both the temporal and multimodal semantics, which means it utilizes almost all the information in autonomous driving scenes. And compared with previous contrastive methods, it can automatically dig out contrasting pairs with higher difficulty, instead of relying on handcrafted ones. Extensive experiments are conducted with Minkowski-UNet and VoxelNet on several semantic segmentation and 3D detection datasets. Results show that TriCC learns effective representations with much fewer training iterations and improves the SOTA results greatly on all the downstream tasks. Code and models can be found at https://bopang1996.github.io/.

----

## [498] RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving

**Authors**: *Angelika Ando, Spyros Gidaris, Andrei Bursuc, Gilles Puy, Alexandre Boulch, Renaud Marlet*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00507](https://doi.org/10.1109/CVPR52729.2023.00507)

**Abstract**:

Casting semantic segmentation of outdoor LiDAR point clouds as a 2D problem, e.g., via range projection, is an effective and popular approach. These projection-based methods usually benefit from fast computations and, when combined with techniques which use other point cloud representations, achieve state-of-the-art results. Today, projection-based methods leverage 2D CNNs but recent advances in computer vision show that vision transformers (ViTs) have achieved state-of-the-art results in many image- based benchmarks. In this work, we question if projection- based methods for 3D semantic segmentation can benefit from these latest improvements on ViTs. We answer positively but only after combining them with three key ingredients: (a) ViTs are notoriously hard to train and require a lot of training data to learn powerful representations. By preserving the same backbone architecture as for RGB images, we can exploit the knowledge from long training on large image collections that are much cheaper to acquire and annotate than point clouds. We reach our best results with pre-trained ViTs on large image datasets. (b) We compensate ViTs' lack of inductive bias by substituting a tailored convolutional stem for the classical linear embedding layer. (c) We refine pixel-wise predictions with a convolutional decoder and a skip connection from the convolutional stem to combine low-level but fine-grained features of the the convolutional stem with the high-level but coarse predictions of the ViT encoder. With these ingredients, we show that our method, called RangeViT, outperforms existing projection-based methods on nuScenes and SemanticKITTI. The code is available at https://github.com/valeoai/rangevit.

----

## [499] Spatiotemporal Self-Supervised Learning for Point Clouds in the Wild

**Authors**: *Yanhao Wu, Tong Zhang, Wei Ke, Sabine Süsstrunk, Mathieu Salzmann*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00508](https://doi.org/10.1109/CVPR52729.2023.00508)

**Abstract**:

Self-supervised learning (SSL) has the potential to benefit many applications, particularly those where manually annotating data is cumbersome. One such situation is the semantic segmentation of point clouds. In this context, existing methods employ contrastive learning strategies and define positive pairs by performing various augmentation of point clusters in a single frame. As such, these methods do not exploit the temporal nature of LiDAR data. In this paper, we introduce an SSL strategy that leverages positive pairs in both the spatial and temporal domain. To this end, we design (i) a point-to-cluster learning strategy that aggregates spatial information to distinguish objects; and (ii) a cluster-to-cluster learning strategy based on unsupervised object tracking that exploits temporal correspondences. We demonstrate the benefits of our approach via extensive experiments performed by self-supervised training on two large-scale LiDAR datasets and transferring the resulting models to other point cloud segmentation benchmarks. Our results evidence that our method outperforms the state-of-the-art point cloud SSL methods.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code and pretrained models will be found at https://github.com/YanhaoWu/STSSL. Correspondence to Ke Wei.

----

## [500] Change-Aware Sampling and Contrastive Learning for Satellite Images

**Authors**: *Utkarsh Mall, Bharath Hariharan, Kavita Bala*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00509](https://doi.org/10.1109/CVPR52729.2023.00509)

**Abstract**:

Automatic remote sensing tools can help inform many large-scale challenges such as disaster management, climate change, etc. While a vast amount of spatio-temporal satellite image data is readily available, most of it remains unlabelled. Without labels, this data is not very useful for supervised learning algorithms. Self-supervised learning instead provides a way to learn effective representations for various downstream tasks without labels. In this work, we leverage characteristics unique to satellite images to learn better self-supervised features. Specifically, we use the temporal signal to contrast images with long-term and short-term differences, and we leverage the fact that satellite images do not change frequently. Using these characteristics, we formulate a new loss contrastive loss called Change-Aware Contrastive (CACo) Loss. Further, we also present a novel method of sampling different geographical regions. We show that leveraging these properties leads to better performance on diverse downstream tasks. For example, we see a 6.5% relative improvement for semantic segmentation and an 8.5% relative improvement for change detection over the best-performing baseline with our method.

----

## [501] Self-Supervised 3D Scene Flow Estimation Guided by Superpoints

**Authors**: *Yaqi Shen, Le Hui, Jin Xie, Jian Yang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00510](https://doi.org/10.1109/CVPR52729.2023.00510)

**Abstract**:

3D scene flow estimation aims to estimate point-wise motions between two consecutive frames of point clouds. Superpoints, i.e., points with similar geometric features, are usually employed to capture similar motions of local regions in 3D scenes for scene flow estimation. However, in existing methods, superpoints are generated with the offline clustering methods, which cannot characterize local regions with similar motions for complex 3D scenes well, leading to inaccurate scene flow estimation. To this end, we propose an iterative end-to-end super point based scene flow estimation framework, where the superpoints can be dynamically updated to guide the point-level flow prediction. Specifically, our framework consists of a flow guided superpoint generation module and a superpoint guided flow refinement module. In our superpoint generation module, we utilize the bidirectional flow information at the previous iteration to obtain the matching points of points and superpoint centers for soft point-to-superpoint association construction, in which the superpoints are generated for pairwise point clouds. With the generated superpoints, we first reconstruct the flow for each point by adaptively aggregating the superpoint-level flow, and then encode the consistency between the reconstructed flow of pairwise point clouds. Finally, we feed the consistency encoding along with the reconstructed flow into GRU to refine point-level flow. Extensive experiments on several different datasets show that our method can achieve promising performance. Code is available at https://github.com/supersyq/SPFlowNet.

----

## [502] SCOOP: Self-Supervised Correspondence and Optimization-Based Scene Flow

**Authors**: *Itai Lang, Dror Aiger, Forrester Cole, Shai Avidan, Michael Rubinstein*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00511](https://doi.org/10.1109/CVPR52729.2023.00511)

**Abstract**:

Scene flow estimation is a long-standing problem in computer vision, where the goal is to find the 3D motion of a scene from its consecutive observations. Recently, there have been efforts to compute the scene flow from 3D point clouds. A common approach is to train a regression model that consumes source and target point clouds and outputs the per-point translation vector. An alternative is to learn point matches between the point clouds concurrently with regressing a refinement of the initial correspondence flow. In both cases, the learning task is very challenging since the flow regression is done in the free 3D space, and a typical solution is to resort to a large annotated synthetic dataset. We introduce SCOOP, a new method for scene flow estimation that can be learned on a small amount of data without employing ground-truth flow supervision. In contrast to previous work, we train a pure correspondence model focused on learning point feature representation and initialize the flow as the difference between a source point and its softly corresponding target point. Then, in the run-time phase, we directly optimize a flow refinement component with a self-supervised objective, which leads to a coherent and accurate flow field between the point clouds. Experiments on widespread datasets demonstrate the performance gains achieved by our method compared to existing leading techniques while using a fraction of the training data. Our code is publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/itailang/SCOOP .

----

## [503] PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection

**Authors**: *Anthony Chen, Kevin Zhang, Renrui Zhang, Zihan Wang, Yuheng Lu, Yandong Guo, Shanghang Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00512](https://doi.org/10.1109/CVPR52729.2023.00512)

**Abstract**:

Masked Autoencoders learn strong visual representations and achieve state-of-the-art results in several independent modalities, yet very few works have addressed their capabilities in multi-modality settings. In this work, we focus on point cloud and RGB image data, two modalities that are often presented together in the real world, and explore their meaningful interactions. To improve upon the cross-modal synergy in existing works, we propose Pi-MAE, a self-supervised pre-training framework that promotes 3D and 2D interaction through three aspects. Specifically, we first notice the importance of masking strategies between the two sources and utilize a projection module to complementarily align the mask and visible tokens of the two modalities. Then, we utilize a well-crafted two-branch MAE pipeline with a novel shared decoder to promote cross-modality interaction in the mask tokens. Finally, we design a unique cross-modal reconstruction module to enhance representation learning for both modalities. Through extensive experiments performed on large-scale RGB-D scene understanding benchmarks (SUN RGB-D and ScannetV2), we discover it is nontrivial to interactively learn point-image features, where we greatly improve multiple 3D detectors, 2D detectors, and few-shot classifiers by 2.9%, 6.7%, and 2.4%, respectively. Code is available at https://github.com/BLVLab/PiMAE.

----

## [504] CP3: Channel Pruning Plug-in for Point-Based Networks

**Authors**: *Yaomin Huang, Ning Liu, Zhengping Che, Zhiyuan Xu, Chaomin Shen, Yaxin Peng, Guixu Zhang, Xinmei Liu, Feifei Feng, Jian Tang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00513](https://doi.org/10.1109/CVPR52729.2023.00513)

**Abstract**:

Channel pruning can effectively reduce both computational cost and memory footprint of the original network while keeping a comparable accuracy performance. Though great success has been achieved in channel pruning for 2D image-based convolutional networks (CNNs), existing works seldom extend the channel pruning methods to 3D point-based neural networks (PNNs). Directly implementing the 2D CNN channel pruning methods to PNNs undermine the performance of PNNs because of the different representations of 2D images and 3D point clouds as well as the network architecture disparity. In this paper, we proposed CP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
, which is a Channel Pruning Plugin for Point-based network. CP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
 is elaborately designed to leverage the characteristics of point clouds and PNNs in order to enable 2D channel pruning methods for PNNs. Specifically, it presents a coordinate-enhanced channel importance metric to reflect the correlation between dimensional information and individual channel features, and it recycles the discarded points in PNN's sampling process and reconsiders their potentially-exclusive information to enhance the robustness of channel pruning. Experiments on various PNN architectures show that CP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
 constantly improves state-of-the-art 2D CNN pruning approaches on different point cloud tasks. For instance, our compressed PointNeXt-S on ScanObjectNN achieves an accuracy of 88.52% with a pruning rate of 57.8%, outperforming the baseline pruning methods with an accuracy gain of 1.94%.

----

## [505] Binarizing Sparse Convolutional Networks for Efficient Point Cloud Analysis

**Authors**: *Xiuwei Xu, Ziwei Wang, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00514](https://doi.org/10.1109/CVPR52729.2023.00514)

**Abstract**:

In this paper, we propose binary sparse convolutional networks called BSC-Net for efficient point cloud analysis. We empirically observe that sparse convolution operation causes larger quantization errors than standard convolution. However, conventional network quantization methods directly binarize the weights and activations in sparse convolution, resulting in performance drop due to the significant quantization loss. On the contrary, we search the optimal subset of convolution operation that activates the sparse convolution at various locations for quantization error alleviation, and the performance gap between real-valued and binary sparse convolutional networks is closed without complexity overhead. Specifically, we first present the shifted sparse convolution that fuses the information in the receptive field for the active sites that match the pre-defined positions. Then we employ the differentiable search strategies to discover the optimal opsitions for active site matching in the shifted sparse convolution, and the quantization errors are significantly alleviated for efficient point cloud analysis. For fair evaluation of the proposed method, we empirically select the recently advances that are beneficial for sparse convolution network binarization to construct a strong baseline. The experimental results on ScanNet and NYU Depth v2 show that our BSC-Net achieves significant improvement upon our srtong baseline and outperforms the state-of-the-art network binarization methods by a remarkable margin without additional computation overhead for binarizing sparse convolutional networks.

----

## [506] Hyperspherical Embedding for Point Cloud Completion

**Authors**: *Junming Zhang, Haomeng Zhang, Ram Vasudevan, Matthew Johnson-Roberson*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00515](https://doi.org/10.1109/CVPR52729.2023.00515)

**Abstract**:

Most real-world 3D measurements from depth sensors are incomplete, and to address this issue the point cloud completion task aims to predict the complete shapes of objects from partial observations. Previous works often adapt an encoder-decoder architecture, where the encoder is trained to extract embeddings that are used as inputs to generate predictions from the decoder. However, the learned embeddings have sparse distribution in the feature space, which leads to worse generalization results during testing. To address these problems, this paper proposes a hyperspherical module, which transforms and normalizes embeddings from the encoder to be on a unit hypersphere. With the proposed module, the magnitude and direction of the output hyperspherical embedding are decoupled and only the directional information is optimized. We theoretically analyze the hyperspherical embedding and show that it enables more stable training with a wider range of learning rates and more compact embedding distributions. Experiment results show consistent improvement of point cloud completion in both single-task and multi-task learning, which demonstrates the effectiveness of the proposed method.

----

## [507] Attention-Based Point Cloud Edge Sampling

**Authors**: *Chengzhi Wu, Junwei Zheng, Julius Pfrommer, Jürgen Beyerer*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00516](https://doi.org/10.1109/CVPR52729.2023.00516)

**Abstract**:

Point cloud sampling is a less explored research topic for this data representation. The most commonly used sampling methods are still classical random sampling and farthest point sampling. With the development of neural networks, various methods have been proposed to sample point clouds in a task-based learning manner. However, these methods are mostly generative-based, rather than selecting points directly using mathematical statistics. Inspired by the Canny edge detection algorithm for images and with the help of the attention mechanism, this paper proposes a non-generative Attention-based Point cloud Edge Sampling method (APES), which captures salient points in the point cloud outline. Both qualitative and quantitative experimental results show the superior performance of our sampling method on common benchmark tasks.

----

## [508] Starting from Non-Parametric Networks for 3D Point Cloud Analysis

**Authors**: *Renrui Zhang, Liuhui Wang, Yali Wang, Peng Gao, Hongsheng Li, Jianbo Shi*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00517](https://doi.org/10.1109/CVPR52729.2023.00517)

**Abstract**:

We present a Non-parametric Network for 3D point cloud analysis, Point-NN, which consists of purely non-learnable components: farthest point sampling (FPS), k-nearest neighbors (k-NN), and pooling operations, with trigonometric functions. Surprisingly, it performs well on various 3D tasks, requiring no parameters or training, and even surpasses existing fully trained models. Starting from this basic non-parametric model, we propose two extensions. First, Point-NN can serve as a base architectural framework to construct Parametric Networks by simply inserting linear layers on top. Given the superior non-parametric foundation, the derived Point-PN exhibits a high performance-efficiency tradeoff with only a few learnable parameters. Second, Point-NN can be regarded as a plug-and-play module for the already trained 3D models during inference. Point-NN captures the complementary geometric knowledge and enhances existing methods for different 3D benchmarks without retraining. We hope our work may cast a light on the community for understanding 3D point clouds with non-parametric methods. Code is available at https://github.com/ZrrSkywalker/Point-NN.

----

## [509] Grad-PU: Arbitrary-Scale Point Cloud Upsampling via Gradient Descent with Learned Distance Functions

**Authors**: *Yun He, Danhang Tang, Yinda Zhang, Xiangyang Xue, Yanwei Fu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00518](https://doi.org/10.1109/CVPR52729.2023.00518)

**Abstract**:

Most existing point cloud upsampling methods have roughly three steps: feature extraction, feature expansion and 3D coordinate prediction. However, they usually suffer from two critical issues: (1) fixed upsampling rate after one-time training, since the feature expansion unit is customized for each upsampling rate; (2) outliers or shrinkage artifact caused by the difficulty of precisely predicting 3D coordinates or residuals of upsampled points. To adress them, we propose a new framework for accurate point cloud upsampling that supports arbitrary upsampling rates. Our method first interpolates the low-res point cloud according to a given upsampling rate. And then refine the positions of the interpolated points with an iterative optimization process, guided by a trained model estimating the difference between the current point cloud and the high-res target. Extensive quantitative and qualitative results on benchmarks and downstream tasks demonstrate that our method achieves the state-of-the-art accuracy and efficiency.

----

## [510] SE-ORNet: Self-Ensembling Orientation-Aware Network for Unsupervised Point Cloud Shape Correspondence

**Authors**: *Jiacheng Deng, Chuxin Wang, Jiahao Lu, Jianfeng He, Tianzhu Zhang, Jiyang Yu, Zhe Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00519](https://doi.org/10.1109/CVPR52729.2023.00519)

**Abstract**:

Unsupervised point cloud shape correspondence aims to obtain dense point-to-point correspondences between point clouds without manually annotated pairs. However, humans and some animals have bilateral symmetry and various orientations, which lead to severe mispredictions of symmetrical parts. Besides, point cloud noise disrupts consistent representations for point cloud and thus degrades the shape correspondence accuracy. To address the above issues, we propose a Self-Ensembling ORientation-aware Network termed SE-ORNet. The key of our approach is to exploit an orientation estimation module with a domain adaptive discriminator to align the orientations of point cloud pairs, which significantly alleviates the mispredictions of symmetrical parts. Additionally, we design a self-ensembling framework for unsupervised point cloud shape correspondence. In this framework, the disturbances of point cloud noise are overcome by perturbing the inputs of the student and teacher networks with different data augmentations and constraining the consistency of predictions. Extensive experiments on both human and animal datasets show that our SE-ORNet can surpass state-of-the-art unsupervised point cloud shape correspondence methods.

----

## [511] Robust 3D Shape Classification via Non-local Graph Attention Network

**Authors**: *Shengwei Qin, Zhong Li, Ligang Liu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00520](https://doi.org/10.1109/CVPR52729.2023.00520)

**Abstract**:

We introduce a non-local graph attention network (NLGAT), which generates a novel global descriptor through two sub-networks for robust 3D shape classification. In the first sub-network, we capture the global relationships between points (i.e., point-point features) by designing a global relationship network (GRN). In the second sub-network, we enhance the local features with a geometric shape attention map obtained from a global structure network (GSN). To keep rotation invariant and extract more information from sparse point clouds, all sub-networks use the Gram matrices with different dimensions as input for working with robust classification. Additionally, GRN effectively preserves the low-frequency features and improves the classification results. Experimental results on various datasets exhibit that the classification effect of the NLGAT model is better than other state-of-the-art models. Especially, in the case of sparse point clouds (64 points) with noise under arbitrary SO(3) rotation, the classification result (85.4%) of NLGAT is improved by 39.4% compared with the best development of other methods.

----

## [512] Rotation-Invariant Transformer for Point Cloud Matching

**Authors**: *Hao Yu, Zheng Qin, Ji Hou, Mahdi Saleh, Dongsheng Li, Benjamin Busam, Slobodan Ilic*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00521](https://doi.org/10.1109/CVPR52729.2023.00521)

**Abstract**:

The intrinsic rotation invariance lies at the core of matching point clouds with handcrafted descriptors. However, it is widely despised by recent deep matchers that obtain the rotation invariance extrinsically via data augmentation. As the finite number of augmented rotations can never span the continuous 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$SO(3)$</tex>
 space, these methods usually show instability when facing rotations that are rarely seen. To this end, we introduce RoITr, a Rotation-Invariant Transformer to cope with the pose variations in the point cloud matching task. We contribute both on the local and global levels. Starting from the local level, we introduce an attention mechanism embedded with Point Pair Feature (PPF)-based coordinates to describe the pose-invariant geometry, upon which a novel attention-based encoder-decoder architecture is constructed. We further propose a global transformer with rotation-invariant cross-frame spatial awareness learned by the self-attention mechanism, which significantly improves the feature distinctiveness and makes the model robust with respect to the low overlap. Experiments are conducted on both the rigid and non-rigid public benchmarks, where RoITr outperforms all the state-of-the-art models by a considerable margin in the low-overlapping scenarios. Especially when the rotations are enlarged on the challenging 3DLoMatch benchmark, RoITr surpasses the existing methods by at least 13 and 5 percentage points in terms of Inlier Ratio and Registration Recall, respectively. Code is publicly available 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/haoyu94/RoITr.

----

## [513] Deep Graph-based Spatial Consistency for Robust Non-rigid Point Cloud Registration

**Authors**: *Zheng Qin, Hao Yu, Changjian Wang, Yuxing Peng, Kai Xu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00522](https://doi.org/10.1109/CVPR52729.2023.00522)

**Abstract**:

We study the problem of outlier correspondence pruning for non-rigid point cloud registration. In rigid registration, spatial consistency has been a commonly used criterion to discriminate outliers from inliers. It measures the compatibility of two correspondences by the discrepancy between the respective distances in two point clouds. However, spatial consistency no longer holds in non-rigid cases and outlier rejection for non-rigid registration has not been well studied. In this work, we propose Graph-based Spatial Consistency Network (GraphSCNet) to filter outliers for non-rigid registration. Our method is based on the fact that non-rigid deformations are usually locally rigid, or local shape preserving. We first design a local spatial consistency measure over the deformation graph of the point cloud, which evaluates the spatial compatibility only between the correspondences in the vicinity of a graph node. An attention-based non-rigid correspondence embedding module is then devised to learn a robust representation of non-rigid correspondences from local spatial consistency. Despite its simplicity, GraphSCNet effectively improves the quality of the putative correspondences and attains state-of-the-art performance on three challenging benchmarks. Our code and models are available at https://github.com/qinzheng93/GraphSCNet.

----

## [514] Efficient RGB-T Tracking via Cross-Modality Distillation

**Authors**: *Tianlu Zhang, Hongyuan Guo, Qiang Jiao, Qiang Zhang, Jungong Han*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00523](https://doi.org/10.1109/CVPR52729.2023.00523)

**Abstract**:

Most current RGB-T trackers adopt a two-stream structure to extract unimodal RGB and thermal features and complex fusion strategies to achieve multi-modal feature fusion, which require a huge number of parameters, thus hindering their real-life applications. On the other hand, a compact RGB-T tracker may be computationally efficient but encounter non-negligible performance degradation, due to the weakening of feature representation ability. To remedy this situation, a cross-modality distillation framework is presented to bridge the performance gap between a compact tracker and a powerful tracker. Specifically, a specific-common feature distillation module is proposed to transform the modality-common information as well as the modality-specific information from a deeper two-stream network to a shallower single-stream network. In addition, a multi-path selection distillation module is proposed to instruct a simple fusion module to learn more accurate multi-modal information from a well-designed fusion mechanism by using multiple paths. We validate the effectiveness of our method with extensive experiments on three RGB-T benchmarks, which achieves state-of-the-art performance but consumes much less computational resources.

----

## [515] Finding Geometric Models by Clustering in the Consensus Space

**Authors**: *Daniel Barath, Denys Rozumnyi, Ivan Eichhardt, Levente Hajder, Jiri Matas*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00524](https://doi.org/10.1109/CVPR52729.2023.00524)

**Abstract**:

We propose a new algorithm for finding an unknown number of geometric models, e.g., homographies. The problem is formalized as finding dominant model instances progressively without forming crisp point-to-model assignments. Dominant instances are found via a RANSAC-like sampling and a consolidation process driven by a model quality function considering previously proposed instances. New ones are found by clustering in the consensus space. This new formulation leads to a simple iterative algorithm with state-of-the-art accuracy while running in real-time on a number of vision problems - at least two orders of magnitude faster than the competitors on two-view motion estimation. Also, we propose a deterministic sampler reflecting the fact that real-world data tend to form spatially coherent structures. The sampler returns connected components in a progressively densified neighborhood-graph. We present a number of applications where the use of multiple geometric models improves accuracy. These include pose estimation from multiple generalized homographies; trajectory estimation of fast-moving objects; and we also propose a way of using multiple homographies in global SfM algorithms. Source code: https://github.com/danini/clustering-in-consensus-space.

----

## [516] Adaptive Assignment for Geometry Aware Local Feature Matching

**Authors**: *Dihe Huang, Ying Chen, Yong Liu, Jianlin Liu, Shang Xu, Wenlong Wu, Yikang Ding, Fan Tang, Chengjie Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00525](https://doi.org/10.1109/CVPR52729.2023.00525)

**Abstract**:

The detector-free feature matching approaches are currently attracting great attention thanks to their excellent performance. However, these methods still struggle at large-scale and viewpoint variations, due to the geometric inconsistency resulting from the application of the mutual nearest neighbour criterion (i.e., one-to-one assignment) in patch-level matching. Accordingly, we in-troduce AdaMatcher, which first accomplishes the feature correlation and co-visible area estimation through an elaborate feature interaction module, then performs adaptive assignment on patch-level matching while es-timating the scales between images, and finally refines the co-visible matches through scale alignment and sub-pixel regression module. Extensive experiments show that AdaMatcher outperforms solid baselines and achieves state-of-the-art results on many downstream tasks. Ad-ditionally, the adaptive assignment and sub-pixel refinement module can be used as a refinement network for other matching methods, such as SuperGlue, to boost their performance further. The code will be publicly available at https://github.com/AbyssGaze/AdaMatcher.

----

## [517] Masked Representation Learning for Domain Generalized Stereo Matching

**Authors**: *Zhibo Rao, Bangshu Xiong, Mingyi He, Yuchao Dai, Renjie He, Zhelun Shen, Xing Li*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00526](https://doi.org/10.1109/CVPR52729.2023.00526)

**Abstract**:

Recently, many deep stereo matching methods have begun to focus on cross-domain performance, achieving impressive achievements. However, these methods did not deal with the significant volatility of generalization performance among different training epochs. Inspired by masked representation learning and multi-task learning, this paper designs a simple and effective masked representation for domain generalized stereo matching. First, we feed the masked left and complete right images as input into the models. Then, we add a lightweight and simple decoder following the feature extraction module to recover the original left image. Finally, we train the models with two tasks (stereo matching and image reconstruction) as a pseudo-multi-task learning framework, promoting models to learn structure information and to improve generalization performance. We implement our method on two well-known architectures (CFNet and LacGwcNet) to demonstrate its effectiveness. Experimental results on multi-datasets show that: (1) our method can be easily plugged into the current various stereo matching models to improve generalization performance; (2) our method can reduce the significant volatility of generalization performance among different training epochs; (3) we find that the current methods prefer to choose the best results among different training epochs as generalization performance, but it is impossible to select the best performance by ground truth in practice.

----

## [518] Learning Optical Expansion from Scale Matching

**Authors**: *Han Ling, Yinghui Sun, Quansen Sun, Zhenwen Ren*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00527](https://doi.org/10.1109/CVPR52729.2023.00527)

**Abstract**:

This paper address the problem of optical expansion (OE). OE describes the object scale change between two frames, widely used in monocular 3D vision tasks. Previous methods estimate optical expansion mainly from optical flow results, but this two-stage architecture makes their results limited by the accuracy of optical flow and less robust. To solve these problems, we propose the concept of 3D optical flow by integrating optical expansion into the 2D optical flow, which is implemented by a plug-and-play module, namely TPCV. TPCV implements matching features at the correct location and scale, thus allowing the simultaneous optimization of optical flow and optical expansion tasks. Experimentally, we apply TPCV to the RAFT optical flow baseline. Experimental results show that the baseline optical flow performance is substantially improved. Moreover, we apply the optical flow and optical expansion results to various dynamic 3D vision tasks, including motion-in-depth, time-to-collision, and scene flow, often achieving significant improvement over the prior SOTA. Code is available at https://github.com/HanLingsgjk/TPCV.

----

## [519] AnyFlow: Arbitrary Scale Optical Flow with Implicit Neural Representation

**Authors**: *Hyunyoung Jung, Zhuo Hui, Lei Luo, Haitao Yang, Feng Liu, Sungjoo Yoo, Rakesh Ranjan, Denis Demandolx*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00528](https://doi.org/10.1109/CVPR52729.2023.00528)

**Abstract**:

To apply optical flow in practice, it is often necessary to resize the input to smaller dimensions in order to reduce computational costs. However, downsizing inputs makes the estimation more challenging because objects and motion ranges become smaller. Even though recent approaches have demonstrated high-quality flow estimation, they tend to fail to accurately model small objects and precise boundaries when the input resolution is lowered, restricting their applicability to high-resolution inputs. In this paper, we introduce AnyFlow, a robust network that estimates accurate flow from images of various resolutions. By representing optical flow as a continuous coordinate-based representation, AnyFlow generates outputs at arbitrary scales from low-resolution inputs, demonstrating superior performance over prior works in capturing tiny objects with detail preservation on a wide range of scenes. We establish a new state-of-the-art performance of cross-dataset generalization on the KITTI dataset, while achieving comparable accuracy on the online benchmarks to other SOTA methods.

----

## [520] HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising

**Authors**: *Mohammad Amin Shabani, Sepidehsadat Hosseini, Yasutaka Furukawa*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00529](https://doi.org/10.1109/CVPR52729.2023.00529)

**Abstract**:

The paper presents a novel approach for vector-floorplan generation via a diffusion model, which denoises 2D coordinates of room/door corners with two inference objectives: 1) a single-step noise as the continuous quantity to precisely invert the continuous forward process; and 2) the final 2D coordinate as the discrete quantity to establish geometric incident relationships such as parallelism, orthogonality, and corner-sharing. Our task is graph-conditioned floorplan generation, a common workflow in floorplan design. We represent a floorplan as 1D polygonal loops, each of which corresponds to a room or a door. Our diffusion model employs a Transformer architecture at the core, which controls the attention masks based on the input graph-constraint and directly generates vector-graphics floorplans via a discrete and continuous denoising process. We have evaluated our approach on RPLAN dataset. The proposed approach makes significant improvements in all the metrics against the state-of-the-art with significant margins, while being capable of generating non-Manhattan structures and controlling the exact number of corners per room. A project website with supplementary video and document is here https://aminshabani.github.io/housediffusion.

----

## [521] Localized Semantic Feature Mixers for Efficient Pedestrian Detection in Autonomous Driving

**Authors**: *Abdul Hannan Khan, Mohammed Shariq Nawaz, Andreas Dengel*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00530](https://doi.org/10.1109/CVPR52729.2023.00530)

**Abstract**:

Autonomous driving systems rely heavily on the underlying perception module which needs to be both performant and efficient to allow precise decisions in realtime. Avoiding collisions with pedestrians is of topmost priority in any autonomous driving system. Therefore, pedestrian detection is one of the core parts of such systems' perception modules. Current state-of-the-art pedestrian detectors have two major issues. Firstly, they have long inference times which affect the efficiency of the whole perception module, and secondly, their performance in the case of small and heavily occluded pedestrians is poor: We propose Local-ized Semantic Feature Mixers (LSFM), a novel, anchor-free pedestrian detection architecture. It uses our novel Super Pixel Pyramid Pooling module instead of the, computation-ally costly, Feature Pyramid Networks for feature encoding. Moreover, our MLPMixer-based Dense Focal Detection Network is used as a light detection head, reducing computational effort and inference time compared to existing approaches. To boost the performance of the proposed architecture, we adapt and use mixup augmentation which improves the performance, especially in small and heavily occluded cases. We benchmark LSFM against the state-of-the-art on well-established traffic scene pedestrian datasets. The proposed LSFM achieves state-of-the-art performance in Caltech, City Persons, Euro City Persons, and TJU-Traffic-Pedestrian datasets while reducing the inference time on average by 55%. Further, LSFM beats the human baseline for the first time in the history of pedestrian detection. Finally, we conducted a cross-dataset evaluation which proved that our proposed LSFM generalizes well to unseen data.

----

## [522] V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting

**Authors**: *Haibao Yu, Wenxian Yang, Hongzhi Ruan, Zhenwei Yang, Yingjuan Tang, Xu Gao, Xin Hao, Yifeng Shi, Yifeng Pan, Ning Sun, Juan Song, Jirui Yuan, Ping Luo, Zaiqing Nie*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00531](https://doi.org/10.1109/CVPR52729.2023.00531)

**Abstract**:

Utilizing infrastructure and vehicle-side information to track and forecast the behaviors of surrounding traffic participants can significantly improve decision-making and safety in autonomous driving. However, the lack of real-world sequential datasets limits research in this area. To address this issue, we introduce V2X-Seq, the first large-scale sequential V2X dataset, which includes data frames, trajectories, vector maps, and traffic lights captured from natural scenery. V2X-Seq comprises two parts: the sequential perception dataset, which includes more than 15,000 frames captured from 95 scenarios, and the trajectory forecasting dataset, which contains about 80,000 infrastructure-view scenarios, 80,000 vehicle-view scenarios, and 50,000 cooperative-view scenarios captured from 28 intersections' areas, covering 672 hours of data. Based on V2X-Seq, we introduce three new tasks for vehicle-infrastructure cooperative (VIC) autonomous driving: VIC3D Tracking, Online-VIC Forecasting, and Offline-VIC Forecasting. We also provide benchmarks for the introduced tasks. Find data, code, and more up-to-date information at https://github.com/AIR-THU/DAIR-V2X-Seq.

----

## [523] ViP3D: End-to-End Visual Trajectory Prediction via 3D Agent Queries

**Authors**: *Junru Gu, Chenxu Hu, Tianyuan Zhang, Xuanyao Chen, Yilun Wang, Yue Wang, Hang Zhao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00532](https://doi.org/10.1109/CVPR52729.2023.00532)

**Abstract**:

Perception and prediction are two separate modules in the existing autonomous driving systems. They interact with each other via hand-picked features such as agent bounding boxes and trajectories. Due to this separation, prediction, as a downstream module, only receives limited information from the perception module. To make matters worse, errors from the perception modules can propagate and accumulate, adversely affecting the prediction results. In this work, we propose ViP 3D, a query-based visual trajectory prediction pipeline that exploits rich information from raw videos to directly predict future trajectories of agents in a scene. ViP3D employs sparse agent queries to detect, track, and predict throughout the pipeline, making it the first fully differentiable vision-based trajectory prediction approach. Instead of using historical feature maps and trajectories, useful information from previous timestamps is encoded in agent queries, which makes ViP3D a concise streaming prediction method. Furthermore, extensive experimental results on the nuScenes dataset show the strong vision-based prediction performance of ViP 3D over traditional pipelines and previous end-to-end models.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and demos are available on the project page: https://tsinghua-mars-lab.github.io/ViP3D

----

## [524] IPCC-TP: Utilizing Incremental Pearson Correlation Coefficient for Joint Multi-Agent Trajectory Prediction

**Authors**: *Dekai Zhu, Guangyao Zhai, Yan Di, Fabian Manhardt, Hendrik Berkemeyer, Tuan Tran, Nassir Navab, Federico Tombari, Benjamin Busam*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00533](https://doi.org/10.1109/CVPR52729.2023.00533)

**Abstract**:

Reliable multi-agent trajectory prediction is crucial for the safe planning and control of autonomous systems. Compared with single-agent cases, the major challenge in simultaneously processing multiple agents lies in modeling complex social interactions caused by various driving intentions and road conditions. Previous methods typically leverage graph-based message propagation or attention mechanism to encapsulate such interactions in the format of marginal probabilistic distributions. However, it is inherently sub-optimal. In this paper, we propose IPCC-TP, a novel relevance-aware module based on Incremental Pearson Correlation Coefficient to improve multi-agent interaction modeling. IPCC-TP learns pairwise joint Gaussian Distributions through the tightly-coupled estimation of the means and covariances according to interactive incremental movements. Our module can be conveniently embedded into existing multi-agent prediction methods to extend original motion distribution decoders. Extensive experiments on nuScenes and Argoverse 2 datasets demonstrate that IPCC-TP improves the performance of baselines by a large margin.

----

## [525] Leapfrog Diffusion Model for Stochastic Trajectory Prediction

**Authors**: *Weibo Mao, Chenxin Xu, Qi Zhu, Siheng Chen, Yanfeng Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00534](https://doi.org/10.1109/CVPR52729.2023.00534)

**Abstract**:

To model the indeterminacy of human behaviors, stochastic trajectory prediction requires a sophisticated multi-modal distribution of future trajectories. Emerging diffusion models have revealed their tremendous representation capacities in numerous generation tasks, showing potential for stochastic trajectory prediction. However, expensive time consumption prevents diffusion models from real-time prediction, since a large number of denoising steps are required to assure sufficient representation ability. To resolve the dilemma, we present LEapfrog Diffusion model (LED), a novel diffusion-based trajectory prediction model, which provides real-time, precise, and diverse predictions. The core of the proposed LED is to leverage a trainable leapfrog initializer to directly learn an expressive multi-modal distribution of future trajectories, which skips a large number of denoising steps, significantly accelerating inference speed. Moreover, the leapfrog initializer is trained to appropriately allocate correlated samples to provide a diversity of predicted future trajectories, significantly improving prediction performances. Extensive experiments on four real-world datasets, including NBA/NFL/SDD/ETH-UCY, show that LED consistently improves performance and achieves 23.7%/21.9% ADE/FDE improvement on NFL. The proposed LED also speeds up the inference 19.3/30.8/24.3/25.1 times compared to the standard diffusion model on NBA/NFL/SDD/ETH-UCY, satisfying real-time inference needs. Code is available at https://github.com/MediaBrain-SJTU/LED.

----

## [526] DeFeeNet: Consecutive 3D Human Motion Prediction with Deviation Feedback

**Authors**: *Xiaoning Sun, Huaijiang Sun, Bin Li, Dong Wei, Weiqing Li, Jianfeng Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00535](https://doi.org/10.1109/CVPR52729.2023.00535)

**Abstract**:

Let us rethink the real-world scenarios that require human motion prediction techniques, such as human-robot collaboration. Current works simplify the task of predicting human motions into a one-off process of forecasting a short future sequence (usually no longer than 1 second) based on a historical observed one. However, such simplification may fail to meet practical needs due to the neglect of the fact that motion prediction in real applications is not an isolated “observe then predict” unit, but a consecutive process composed of many rounds of such unit, semi-overlapped along the entire sequence. As time goes on, the predicted part of previous round has its corresponding ground truth observable in the new round, but their deviation in-between is neither exploited nor able to be captured by existing isolated learning fashion. In this paper, we propose DeFeeNet, a simple yet effective network that can be added on existing one-off prediction models to realize deviation perception and feedback when applied to consecutive motion prediction task. At each prediction round, the deviation generated by previous unit is first encoded by our DeFeeNet, and then incorporated into the existing predictor to enable a deviation-aware prediction manner, which, for the first time, allows for information transmit across adjacent prediction units. We design two versions of DeFeeNet as MLP-based and GRU-based, respectively. On Human3.6M and more complicated BABEL, experimental results indicate that our proposed network improves consecutive human motion prediction performance regardless of the basic model.

----

## [527] Self-Correctable and Adaptable Inference for Generalizable Human Pose Estimation

**Authors**: *Zhehan Kan, Shuoshuo Chen, Ce Zhang, Yushun Tang, Zhihai He*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00536](https://doi.org/10.1109/CVPR52729.2023.00536)

**Abstract**:

A central challenge in human pose estimation, as well as in many other machine learning and prediction tasks, is the generalization problem. The learned network does not have the capability to characterize the prediction error, generate feedback information from the test sample, and correct the prediction error on the fly for each individual test sample, which results in degraded performance in generalization. In this work, we introduce a self-correctable and adaptable inference (SCAI) method to address the generalization challenge of network prediction and use human pose estimation as an example to demonstrate its effectiveness and performance. We learn a correction network to correct the prediction result conditioned by a fitness feedback error. This feedback error is generated by a learned fitness feedback network which maps the prediction result to the original input domain and compares it against the original input. Interestingly, we find that this self-referential feedback error is highly correlated with the actual prediction error. This strong correlation suggests that we can use this error as feedback to guide the correction process. It can be also used as a loss function to quickly adapt and optimize the correction network during the inference process. Our extensive experimental results on human pose estimation demonstrate that the proposed SCAI method is able to significantly improve the generalization capability and performance of human pose estimation.

----

## [528] ReDirTrans: Latent-to-Latent Translation for Gaze and Head Redirection

**Authors**: *Shiwei Jin, Zhen Wang, Lei Wang, Ning Bi, Truong Q. Nguyen*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00537](https://doi.org/10.1109/CVPR52729.2023.00537)

**Abstract**:

Learning-based gaze estimation methods require large amounts of training data with accurate gaze annotations. Facing such demanding requirements of gaze data collection and annotation, several image synthesis methods were proposed, which successfully redirected gaze directions pre-cisely given the assigned conditions. However, these methods focused on changing gaze directions of the images that only include eyes or restricted ranges of faces with low res-olution (less than 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$128\times 128$</tex>
) to largely reduce interference from other attributes such as hairs, which limits application scenarios. To cope with this limitation, we proposed a portable network, called ReDirTrans, achieving latent-to-latent translation for redirecting gaze directions and head orientations in an interpretable manner. ReDirTrans projects input latent vectors into aimed-attribute embed-dings only and redirects these embeddings with assigned pitch and yaw values. Then both the initial and edited embeddings are projected back (deprojected) to the initial latent space as residuals to modify the input latent vec-tors by subtraction and addition, representing old status re-moval and new status addition. The projection of aimed at-tributes only and subtraction-addition operations for status replacement essentially mitigate impacts on other attributes and the distribution of latent vectors. Thus, by combining ReDirTrans with a pretrained fixed e4e-StyleGAN pair, we created ReDirTrans-GAN, which enables accurately redi-recting gaze in full-face images with 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$1024\times 1024$</tex>
 resolution while preserving other attributes such as identity, expres-sion, and hairstyle. Furthermore, we presented improvements for the downstream learning-based gaze estimation task, using redirected samples as dataset augmentation.

----

## [529] Feature Shrinkage Pyramid for Camouflaged Object Detection with Transformers

**Authors**: *Zhou Huang, Hang Dai, Tian-Zhu Xiang, Shuo Wang, Huai-Xin Chen, Jie Qin, Huan Xiong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00538](https://doi.org/10.1109/CVPR52729.2023.00538)

**Abstract**:

Vision transformers have recently shown strong global context modeling capabilities in camouflaged object detection. However, they suffer from two major limitations: less effective locality modeling and insufficient feature aggregation in decoders, which are not conducive to camou-flaged object detection that explores subtle cues from indistinguishable backgrounds. To address these issues, in this paper, we propose a novel transformer-based Feature Shrinkage Pyramid Network (FSPNet), which aims to hierarchically decode locality-enhanced neighboring transformer features through progressive shrinking for camou-flaged object detection. Specifically, we propose a non-local token enhancement module (NL-TEM) that employs the non-local mechanism to interact neighboring tokens and explore graph-based high-order relations within tokens to enhance local representations of transformers. Moreover, we design a feature shrinkage decoder (FSD) with adjacent interaction modules (AIM), which progressively aggregates adjacent transformer features through a layer-by-layer shrinkage pyramid to accumulate imperceptible but effective cues as much as possible for object information decoding. Extensive quantitative and qualitative experiments demonstrate that the proposed model significantly outperforms the existing 24 competitors on three challenging COD benchmark datasets under six widely-used evaluation metrics. Our code is publicly available at https://github.com/ZhouHuang23/FSPNet.

----

## [530] OVTrack: Open-Vocabulary Multiple Object Tracking

**Authors**: *Siyuan Li, Tobias Fischer, Lei Ke, Henghui Ding, Martin Danelljan, Fisher Yu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00539](https://doi.org/10.1109/CVPR52729.2023.00539)

**Abstract**:

The ability to recognize, localize and track dynamic objects in a scene is fundamental to many real-world applications, such as self-driving and robotic systems. Yet, traditional multiple object tracking (MOT) benchmarks rely only on a few object categories that hardly represent the multitude of possible objects that are encountered in the real world. This leaves contemporary MOT methods limited to a small set of pre-defined object categories. In this paper, we address this limitation by tackling a novel task, open-vocabulary MOT, that aims to evaluate tracking beyond pre-defined training categories. We further develop OVTrack, an open-vocabulary tracker that is capable of tracking arbitrary object classes. Its design is based on two key ingredients: First, leveraging vision-language models for both classification and association via knowledge distillation; second, a data hallucination strategy for robust appearance feature learning from denoising diffusion probabilistic models. The result is an extremely data-efficient open-vocabulary tracker that sets a new state-of-the-art on the large-scale, large-vocabulary TAO benchmark, while being trained solely on static images.

----

## [531] GaitGCI: Generative Counterfactual Intervention for Gait Recognition

**Authors**: *Huanzhang Dou, Pengyi Zhang, Wei Su, Yunlong Yu, Yining Lin, Xi Li*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00540](https://doi.org/10.1109/CVPR52729.2023.00540)

**Abstract**:

Gait is one of the most promising biometrics that aims to identify pedestrians from their walking patterns. However, prevailing methods are susceptible to confounders, resulting in the networks hardly focusing on the regions that re-flect effective walking patterns. To address this fundamen-tal problem in gait recognition, we propose a Generative Counterfactual Intervention framework, dubbed GaitGCI, consisting of Counterfactual Intervention Learning (CIL) and Diversity-Constrained Dynamic Convolution (DCDC). CIL eliminates the impacts of confounders by maximizing the likelihood difference between factual/counterfactual attention while DCDC adaptively generates sample-wise factual/counterfactual attention to efficiently perceive the sample-wise properties. With matrix decomposition and diversity constraint, DCDC guarantees the model to be efficient and effective. Extensive experiments indicate that proposed GaitGCI.· 1) could effectively focus on the discrimi-native and interpretable regions that reflect gait pattern; 2) is model-agnostic and could be plugged into existing models to improve performance with nearly no extra cost; 3) efficiently achieves state-of-the-art performance on arbitrary scenarios (in-the-lab and in-the-wild).

----

## [532] Multi-Label Compound Expression Recognition: C-EXPR Database & Network

**Authors**: *Dimitrios Kollias*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00541](https://doi.org/10.1109/CVPR52729.2023.00541)

**Abstract**:

Research in automatic analysis of facial expressions mainly focuses on recognising the seven basic ones. How-ever, compound expressions are more diverse and represent the complexity and subtlety of our daily affective displays more accurately. Limited research has been conducted for compound expression recognition (CER), because only a few databases exist, which are small, lab controlled, imbalanced and static. In this paper we present an in-the-wild A/V database, C-EXPR-DB, consisting of 400 videos of200Kframes, annotated in terms of 13 compound expressions, valence-arousal emotion descriptors, action units, speech, facial landmarks and attributes. We also propose C-EXPR-NET, a multi-task learning (MTL) methodfor CER and AU detection (AU-D); the latter task is introduced to en-hance CER performance. For AU-D we incorporate AU se-mantic description along with visual information. For CER we use a multi-label formulation and the KL-divergence loss. We also propose a distribution matching loss for cou-pling CER and AU-D tasks to boost their performance and alleviate negative transfer (i.e., when MT model's performance is worse than that of at least one single-task model). An extensive experimental study has been conducted illus-trating the excellent performance of C-EXPR-NET, vali-dating the theoretical claims. Finally, C-EXPR-NET is shown to effectively generalize its knowledge in new emotion recognition contexts, in a zero-shot manner.

----

## [533] Blemish-aware and Progressive Face Retouching with Limited Paired Data

**Authors**: *Lianxin Xie, Wen Xue, Zhen Xu, Si Wu, Zhiwen Yu, Hau-San Wong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00542](https://doi.org/10.1109/CVPR52729.2023.00542)

**Abstract**:

Face retouching aims to remove facial blemishes, while at the same time maintaining the textual details of a given input image. The main challenge lies in distinguishing blemishes from the facial characteristics, such as moles. Training an image-to-image translation network with pixel-wise supervision suffers from the problem of expensive paired training data, since professional retouching needs specialized experience and is time-consuming. In this paper, we propose a Blemish-aware and Progressive Face Retouching model, which is referred to as BPFRe. Our framework can be partitioned into two manageable stages to perform progressive blemish removal. Specifically, an encoder-decoder-based module learns to coarsely remove the blemishes at the first stage, and the resulting intermediate features are injected into a generator to enrich local detail at the second stage. We find that explicitly suppressing the blemishes can contribute to an effective collaboration among the components. Toward this end, we incorporate an attention module, which learns to infer a blemish-aware map and further determine the corresponding weights, which are then used to refine the intermediate features transferred from the encoder to the decoder, and from the decoder to the generator. Therefore, BPFRe is able to deliver significant performance gains on a wide range of face retouching tasks. It is worth noting that we reduce the dependence of BPFRe on paired training samples by imposing effective regularization on unpaired ones.

----

## [534] High-Fidelity and Freely Controllable Talking Head Video Generation

**Authors**: *Yue Gao, Yuan Zhou, Jinglu Wang, Xiao Li, Xiang Ming, Yan Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00543](https://doi.org/10.1109/CVPR52729.2023.00543)

**Abstract**:

Talking head generation is to generate video based on a given source identity and target motion. However, current methods face several challenges that limit the quality and controllability of the generated videos. First, the generated face often has unexpected deformation and severe distortions. Second, the driving image does not explicitly disentangle movement-relevant information, such as poses and expressions, which restricts the manipulation of different attributes during generation. Third, the generated videos tend to have flickering artifacts due to the inconsistency of the extracted landmarks between adjacent frames. In this paper, we propose a novel model that produces high-fidelity talking head videos with free control over head pose and expression. Our method leverages both self-supervised learned landmarks and 3D face model-based landmarks to model the motion. We also introduce a novel motion-aware multi-scale feature alignment module to effectively transfer the motion without face distortion. Furthermore, we enhance the smoothness of the synthesized talking head videos with a feature context adaptation and propagation module. We evaluate our model on challenging datasets and demonstrate its state-of-the-art performance. More information is available at https://yuegao.me/PECHead.

----

## [535] 3Mformer: Multi-order Multi-mode Transformer for Skeletal Action Recognition

**Authors**: *Lei Wang, Piotr Koniusz*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00544](https://doi.org/10.1109/CVPR52729.2023.00544)

**Abstract**:

Many skeletal action recognition models use GCNs to represent the human body by 3D body joints connected body parts. GCNs aggregate one- or few-hop graph neighbourhoods, and ignore the dependency between not linked body joints. We propose to form hypergraph to model hyperedges between graph nodes (e.g., third- and fourth-order hyper-edges capture three and four nodes) which help capture higher-order motion patterns of groups of body joints. We split action sequences into temporal blocks, Higher-order Transformer (HoT) produces embeddings of each temporal block based on (i) the body joints, (ii) pairwise links of body joints and (iii) higher-order hyper-edges of skeleton body joints. We combine such HoT embeddings of hyper-edges of orders 1,…, r by a novel Multi-order Multi-mode Transformer (3Mformer) with two modules whose order can be exchanged to achieve coupled-mode attention on coupled-mode tokens based on ‘channel-temporal block’, ‘order-channel-body joint’, ‘channel-hyper-edge (any order)’ and ‘channel-only’ pairs. The first module, called Multi-order Pooling (MP), additionally learns weighted aggregation along the hyper-edge mode, whereas the second module, Temporal block Pooling (TP), aggregates along the temporal block
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
For brevity, we write τ temporal blocks per sequence but τ varies. mode. Our end-to-end trainable network yields state-of-the-art results compared to GCN-, transformer- and hypergraph-based counterparts.

----

## [536] UDE: A Unified Driving Engine for Human Motion Generation

**Authors**: *Zixiang Zhou, Baoyuan Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00545](https://doi.org/10.1109/CVPR52729.2023.00545)

**Abstract**:

Generating controllable and editable human motion sequences is a key challenge in 3D Avatar generation. It has been labor-intensive to generate and animate human motion for a long time until learning-based approaches have been developed and applied recently. However, these approaches are still task-specific or modality-specific [1][6] [5][18]. In this paper, we propose “UDE”, the first unified driving engine that enables generating human motion sequences from natural language or audio sequences (see Fig. 1). Specifically, UDE consists of the following key components: 1) a motion quantization module based on VQVAE that represents continuous motion sequence as discrete latent code [33], 2) a modality-agnostic transformer encoder [34] that learns to map modality-aware driving signals to a joint space, and 3) a unified token transformer (GPT-like[24])network to predict the quantized latent code index in an auto-regressive manner. 4) a diffusion motion decoder that takes as input the motion tokens and decodes them into motion sequence swith high diversity. Weevaluate our method on HumanML3D [8] and AIST+ [19] benchmarks, and the experiment results demonstrate our method achieves state-of-the-art performance. Project website: https://zixiangzhou916.github.io/UDE/

----

## [537] Data-Driven Feature Tracking for Event Cameras

**Authors**: *Nico Messikommer, Carter Fang, Mathias Gehrig, Davide Scaramuzza*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00546](https://doi.org/10.1109/CVPR52729.2023.00546)

**Abstract**:

Because of their high temporal resolution, increased resilience to motion blur, and very sparse output, event cameras have been shown to be ideal for low-latency and low-bandwidth feature tracking, even in challenging scenarios. Existing feature tracking methods for event cameras are either handcrafted or derived from first principles but require extensive parameter tuning, are sensitive to noise, and do not generalize to different scenarios due to unmodeled effects. To tackle these deficiencies, we introduce the first data-driven feature tracker for event cameras, which leverages low-latency events to track features detected in a grayscale frame. We achieve robust performance via a novel frame attention module, which shares information across feature tracks. By directly transferring zero-shot from synthetic to real data, our data-driven tracker outperforms existing approaches in relative feature age by up to 120 % while also achieving the lowest latency. This performance gap is further increased to 130 % by adapting our tracker to real data with a novel self-supervision strategy. Multimedia Material A video is available at https://youtu.be/dtkXvNXcWRY and code at https://github.com/uzh-rpg/deep_ev_tracker

----

## [538] MoStGAN-V: Video Generation with Temporal Motion Styles

**Authors**: *Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00547](https://doi.org/10.1109/CVPR52729.2023.00547)

**Abstract**:

Video generation remains a challenging task due to spatiotemporal complexity and the requirement of synthesizing diverse motions with temporal consistency. Previous works attempt to generate videos in arbitrary lengths either in an autoregressive manner or regarding time as a continuous signal. However, they struggle to synthesize detailed and diverse motions with temporal coherence and tend to generate repetitive scenes after a few time steps. In this work, we argue that a single time-agnostic latent vector of style-based generator is insufficient to model various and temporally-consistent motions. Hence, we introduce additional time-dependent motion styles to model diverse motion patterns. In addition, a Motion Style Attention modulation mechanism, dubbed as MoStAtt, is proposed to augment frames with vivid dynamics for each specific scale (i.e., layer), which assigns attention score for each motion style w.r.t deconvolution filter weights in the target synthesis layer and softly attends different motion styles for weight modulation. Experimental results show our model achieves state-of-the-art performance on four unconditional 256
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
video synthesis benchmarks trained with only 3 frames per clip and produces better qualitative results with respect to dynamic motions. Code and videos have been made available at https:/github.com/xiaoqian-shen/MoStGAN-V

----

## [539] Two-stage Co-segmentation Network Based on Discriminative Representation for Recovering Human Mesh from Videos

**Authors**: *Boyang Zhang, Kehua Ma, Suping Wu, Zhixiang Yuan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00548](https://doi.org/10.1109/CVPR52729.2023.00548)

**Abstract**:

Recovering 3D human mesh from videos has recently made significant progress. However, most of the existing methods focus on the temporal consistency of videos, while ignoring the spatial representation in complex scenes, thus failing to recover a reasonable and smooth human mesh sequence under extreme illumination and chaotic backgrounds. To alleviate this problem, we propose a two-stage co-segmentation network based on discriminative representationfor recovering human body meshes from videos. Specifically, the first stage of the network segments the video spatial domain to spotlight spatially fine-grained information, and then learns and enhances the intra-frame discriminative representation through a dual-excitation mechanism and a frequency domain enhancement module, while sup-pressing irrelevant information (e.g., background). The second stage focuses on temporal context by segmenting the video temporal domain, and models inter-frame discriminative representation via a dynamic integration strategy. Further, to efficiently generate reasonable human discriminative actions, we carefully elaborate a landmark anchor area loss to constrain the variation of the human motion area. Extensive experimental results on large publicly available datasets indicate superiority in comparison with most state-of-the-art. The Code will be made public.

----

## [540] Joint Appearance and Motion Learning for Efficient Rolling Shutter Correction

**Authors**: *Bin Fan, Yuxin Mao, Yuchao Dai, Zhexiong Wan, Qi Liu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00549](https://doi.org/10.1109/CVPR52729.2023.00549)

**Abstract**:

Rolling shutter correction (RSC) is becoming increasingly popular for RS cameras that are widely used in commercial and industrial applications. Despite the promising performance, existing RSC methods typically employ a two-stage network structure that ignores intrinsic infor-mation interactions and hinders fast inference. In this pa-per, we propose a single-stage encoder-decoder-based network, named JAMNet, for efficient RSC. It first extracts pyramid features from consecutive RS inputs, and then simultaneously refines the two complementary information (i.e., global shutter appearance and undistortion motion field) to achieve mutual promotion in a joint learning de-coder. To inject sufficient motion cues for guiding joint learning, we introduce a transformer-based motion embed-ding module and propose to pass hidden states across pyra-mid levels. Moreover, we present a new data augmentation strategy “vertical flip + inverse order” to release the potential of the RSC datasets. Experiments on various benchmarks show that our approach surpasses the state-of-the-art methods by a large margin, especially with a 4.7 dB PSNR leap on real-world RSC. Code is available at https://github.com/GitCVfb/JAMNet.

----

## [541] Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation

**Authors**: *Guozhen Zhang, Yuhan Zhu, Haonan Wang, Youxin Chen, Gangshan Wu, Limin Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00550](https://doi.org/10.1109/CVPR52729.2023.00550)

**Abstract**:

Effectively extracting inter-frame motion and appearance information is important for video frame interpolation (VFI). Previous works either extract both types of information in a mixed way or devise separate modules for each type of information, which lead to representation ambiguity and low efficiency. In this paper, we propose a new module to explicitly extract motion and appearance information via a unified operation. Specifically, we rethink the information process in inter-frame attention and reuse its attention map for both appearance feature enhancement and motion information extraction. Furthermore, for efficient VFI, our proposed module could be seamlessly integrated into a hybrid CNN and Transformer architecture. This hybrid pipeline can alleviate the computational complexity of inter-frame attention as well as preserve detailed low-level structure information. Experimental results demonstrate that, for both fixed- and arbitrary-timestep interpolation, our method achieves state-of-the-art performance on various datasets. Meanwhile, our approach enjoys a lighter computation overhead over models with close performance. The source code and models are available at https://github.com/MCG-NJU/EMA-VFI.

----

## [542] Deep Stereo Video Inpainting

**Authors**: *Zhiliang Wu, Changchang Sun, Hanyu Xuan, Yan Yan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00551](https://doi.org/10.1109/CVPR52729.2023.00551)

**Abstract**:

Stereo video inpainting aims to fill the missing regions on the left and right views of the stereo video with plausible content simultaneously. Compared with the single video inpainting that has achieved promising results using deep convolutional neural networks, inpainting the missing regions of stereo video has not been thoroughly explored. In essence, apart from the spatial and temporal consistency that single video inpainting needs to achieve, another key challenge for stereo video inpainting is to maintain the stereo consistency between left and right views and hence alleviate the 3D fatigue for viewers. In this paper, we propose a novel deep stereo video inpainting network named SVINet, which is the first attempt for stereo video inpainting task utilizing deep convolutional neural networks. SVINet first utilizes a self-supervised flow-guided deformable temporal alignment module to align the features on the left and right view branches, respectively. Then, the aligned features are fed into a shared adaptive feature aggregation module to generate missing contents of their respective branches. Finally, the parallax attention module (PAM) that uses the cross-view information to consider the significant stereo correlation is introduced to fuse the completed features of left and right views. Furthermore, we develop a stereo consistency loss to regularize the trained parameters, so that our model is able to yield high-quality stereo video inpainting results with better stereo consistency. Experimental results demonstrate that our SVINet outperforms state-of-the-art single video inpainting models.

----

## [543] Burstormer: Burst Image Restoration and Enhancement Transformer

**Authors**: *Akshay Dudhane, Syed Waqas Zamir, Salman Khan, Fahad Shahbaz Khan, Ming-Hsuan Yang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00552](https://doi.org/10.1109/CVPR52729.2023.00552)

**Abstract**:

On a shutter press, modern handheld cameras capture multiple images in rapid succession and merge them to gen-erate a single image. However, individual frames in a burst are misaligned due to inevitable motions and contain mul-tiple degradations. The challenge is to properly align the successive image shots and merge their complimentary in-formation to achieve high-quality outputs. Towards this direction, we propose Burstormer: a novel transformer-based architecture for burst image restoration and enhancement. In comparison to existing works, our approach exploits multi-scale local and non-local features to achieve improved alignment and feature fusion. Our key idea is to enable inter-frame communication in the burst neighborhoods for information aggregation and progressive fusion while modeling the burst-wide context. However, the input burst frames need to be properly aligned before fusing their information. Therefore, we propose an enhanced de-formable alignment module for aligning burst features with regards to the reference frame. Unlike existing methods, the proposed alignment module not only aligns burst features but also exchanges fea-ture information and maintains focused communication with the reference frame through the proposed reference-based feature enrichment mechanism, which facilitates handling complex motions. After multi-level alignment and enrichment, we re-emphasize on inter-frame communication within burst using a cyclic burst sampling module. Finally, the inter-frame information is aggre-gated using the proposed burst feature fusion module followed by progressive upsampling. Our Burstormer outperforms state-of-the-art methods on burst super-resolution, burst denoising and burst low-light enhance-ment. Our codes and pre-trained models are available at https://github.com/akshaydudhane16/Burstormer.

----

## [544] Blur Interpolation Transformer for Real-World Motion from Blur

**Authors**: *Zhihang Zhong, Mingdeng Cao, Xiang Ji, Yinqiang Zheng, Imari Sato*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00553](https://doi.org/10.1109/CVPR52729.2023.00553)

**Abstract**:

This paper studies the challenging problem of recovering motion from blur, also known as joint deblurring and interpolation or blur temporal super-resolution. The challenges are twofold: 1) the current methods still leave considerable room for improvement in terms of visual quality even on the synthetic dataset, and 2) poor generalization to real-world data. To this end, we propose a blur interpolation transformer (BiT) to effectively unravel the underlying temporal correlation encoded in blur. Based on multi-scale residual Swin transformer blocks, we introduce dual-end temporal supervision and temporally symmetric ensembling strategies to generate effective features for time-varying motion rendering. In addition, we design a hybrid camera system to collect the first real-world dataset of one-to-many blur-sharp video pairs. Experimental results show that BiT has a significant gain over the state-of-the-art methods on the public dataset Adobe240. Besides, the proposed real-world dataset effectively helps the model generalize well to real blurry scenarios. Code and data are available at https://github.com/zzh-tech/Bi'T.

----

## [545] HDR Imaging with Spatially Varying Signal-to-Noise Ratios

**Authors**: *Yiheng Chi, Xingguang Zhang, Stanley H. Chan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00554](https://doi.org/10.1109/CVPR52729.2023.00554)

**Abstract**:

While today's high dynamic range (HDR) image fusion algorithms are capable of blending multiple exposures, the acquisition is often controlled so that the dynamic range within one exposure is narrow. For HDR imaging in photon-limited situations, the dynamic range can be enormous and the noise within one exposure is spatially varying. Existing image denoising algorithms and HDR fusion algorithms both fail to handle this situation, leading to severe limitations in low-light HDR imaging. This paper presents two contributions. Firstly, we identify the source of the problem. We find that the issue is associated with the co-existence of (1) spatially varying signal-to-noise ratio, especially the excessive noise due to very dark regions, and (2) a wide luminance range within each exposure. We show that while the issue can be handled by a bank of denoisers, the complexity is high. Secondly, we propose a new method called the spatially varying high dynamic range (SV-HDR) fusion network to simultaneously denoise and fuse images. We introduce a new exposure-shared block within our custom-designed multi-scale transformer framework. In a variety of testing conditions, the performance of the proposed SV-HDR is better than the existing methods.

----

## [546] Light Source Separation and Intrinsic Image Decomposition under AC Illumination

**Authors**: *Yusaku Yoshida, Ryo Kawahara, Takahiro Okabe*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00555](https://doi.org/10.1109/CVPR52729.2023.00555)

**Abstract**:

Artificial light sources are often powered by an electric grid, and then their intensities rapidly oscillate in response to the grid's alternating current (AC). Interestingly, the flickers of scene radiance values due to AC illumination are useful for extracting rich information on a scene of interest. In this paper, we show that the flickers due to AC illumination is useful for intrinsic image decomposition (IID). Our proposed method conducts the light source separation (LSS) followed by the IID under AC illumination. In particular, we reveal the ambiguity in the blind LSS via matrix factorization and the ambiguity in the IID assuming the diffuse reflection model, and then show why and how those ambiguities can be resolved via a physics-based approach. We experimentally confirmed that our method can recover the colors of the light sources, the diffuse reflectance values, and the diffuse and specular intensities (shadings) under each of the light sources, and that the IID under AC illumination is effective for application to auto white balancing.

----

## [547] Physics-Guided ISO-Dependent Sensor Noise Modeling for Extreme Low-Light Photography

**Authors**: *Yue Cao, Ming Liu, Shuai Liu, Xiaotao Wang, Lei Lei, Wangmeng Zuo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00556](https://doi.org/10.1109/CVPR52729.2023.00556)

**Abstract**:

Although deep neural networks have achieved astonishing performance in many vision tasks, existing learningbased methods are far inferior to the physical model-based solutions in extreme low-light sensor noise modeling. To tap the potential of learning-based sensor noise modeling, we investigate the noise formation in a typical imaging process and propose a novel physics-guided ISO-dependent sensor noise modeling approach. Specifically, we build a normalizing flow-based framework to represent the complex noise characteristics of CMOS camera sensors. Each component of the noise model is dedicated to a particular kind of noise under the guidance of physical models. Moreover, we take into consideration of the ISO dependence in the noise model, which is not completely considered by the existing learning-based methods. For training the proposed noise model, a new dataset is further collected with paired noisy-clean images, as well as flat-field and bias frames covering a wide range of ISO settings. Compared to existing methods, the proposed noise model is equipped with a flexible structure and accurate modeling capabilities, which is beneficial for better denoising performance in extreme low-light scenes. The dataset and code are available at https://github.com/happycaoyue/LLD.

----

## [548] Neumann Network with Recursive Kernels for Single Image Defocus Deblurring

**Authors**: *Yuhui Quan, Zicong Wu, Hui Ji*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00557](https://doi.org/10.1109/CVPR52729.2023.00557)

**Abstract**:

Single image defocus deblurring (SIDD) refers to recovering an all-in-focus image from a defocused blurry one. It is a challenging recovery task due to the spatially-varying defocus blurring effects with significant size variation. Motivated by the strong correlation among defocus kernels of different sizes and the blob-type structure of defocus kernels, we propose a learnable recursive kernel representation (RKR) for defocus kernels that expresses a defocus kernel by a linear combination of recursive, separable and positive atom kernels, leading to a compact yet effective and physics-encoded parametrization of the spatially-varying defocus blurring process. Afterwards, a physics-driven and efficient deep model with a cross-scale fusion structure is presented for SIDD, with inspirations from the truncated Neumann series for approximating the matrix inversion of the RKR-based blurring operator. In addition, a reblurring loss is proposed to regularize the RKR learning. Extensive experiments show that, our proposed approach significantly outperforms existing ones, with a model size comparable to that of the top methods.

----

## [549] UMat: Uncertainty-Aware Single Image High Resolution Material Capture

**Authors**: *Carlos Rodríguez-Pardo, Henar Dominguez-Elvira, David Pascual-Hernández, Elena Garces*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00558](https://doi.org/10.1109/CVPR52729.2023.00558)

**Abstract**:

We propose a learning-based method to recover normals, specularity, and roughness from a single diffuse image of a material, using microgeometry appearance as our primary cue. Previous methods that work on single images tend to produce over-smooth outputs with artifacts, operate at limited resolution, or train one model per class with little room for generalization. In contrast, in this work, we propose a novel capture approach that leverages a generative network with attention and a U-Net discriminator, which shows outstanding performance integrating global information at reduced computational complexity. We showcase the performance of our method with a real dataset of digitized textile materials and show that a commodity flatbed scanner can produce the type of diffuse illumination required as input to our method. Additionally, because the problem might be ill-posed –more than a single diffuse image might be needed to disambiguate the specular reflection– or because the training dataset is not representative enough of the real distribution, we propose a novel framework to quantify the model's confidence about its prediction at test time. Our method is the first one to deal with the problem of modeling uncertainty in material digitization, increasing the trustworthiness of the process and enabling more intelligent strategies for dataset creation, as we demonstrate with an active learning experiment.

----

## [550] SMAE: Few-shot Learning for HDR Deghosting with Saturation-Aware Masked Autoencoders

**Authors**: *Qingsen Yan, Song Zhang, Weiye Chen, Hao Tang, Yu Zhu, Jinqiu Sun, Luc Van Gool, Yanning Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00559](https://doi.org/10.1109/CVPR52729.2023.00559)

**Abstract**:

Generating a high-quality High Dynamic Range (HDR) image from dynamic scenes has recently been extensively studied by exploiting Deep Neural Networks (DNNs). Most DNNs-based methods require a large amount of training data with ground truth, requiring tedious and time-consuming work. Few-shot HDR imaging aims to generate satisfactory images with limited data. However, it is difficult for modern DNNs to avoid overfitting when trained on only a few images. In this work, we propose a novel semi-supervised approach to realize few-shot HDR imaging via two stages of training, called SSHDR. Unlikely previous methods, directly recovering content and removing ghosts simultaneously, which is hard to achieve optimum, we first generate content of saturated regions with a self-supervised mechanism and then address ghosts via an iterative semi-supervised learning framework. Concretely, considering that saturated regions can be regarded as masking Low Dynamic Range (LDR) input regions, we design a Saturated Mask AutoEncoder (SMAE) to learn a robust feature representation and reconstruct a non-saturated HDR image. We also propose an adaptive pseudo-label selection strategy to pick high-quality HDR pseudo-labels in the second stage to avoid the effect of mislabeled samples. Experiments demonstrate that SSHDR outperforms state-of-the-art methods quantitatively and qualitatively within and across different datasets, achieving appealing HDR visualization with few labeled samples.

----

## [551] Curricular Contrastive Regularization for Physics-Aware Single Image Dehazing

**Authors**: *Yu Zheng, Jiahui Zhan, Shengfeng He, Junyu Dong, Yong Du*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00560](https://doi.org/10.1109/CVPR52729.2023.00560)

**Abstract**:

Considering the ill-posed nature, contrastive regularization has been developed for single image dehazing, introducing the information from negative images as a lower bound. However, the contrastive samples are non-consensual, as the negatives are usually represented distantly from the clear (i.e., positive) image, leaving the solution space still under-constricted. Moreover, the interpretability of deep dehazing models is underexplored towards the physics of the hazing process. In this paper, we propose a novel curricular contrastive regularization targeted at a consensual contrastive space as opposed to a non-consensual one. Our negatives, which provide better lower-bound constraints, can be assembled from 1) the hazy image, and 2) corresponding restorations by other existing methods. Further, due to the different similarities between the embeddings of the clear image and negatives, the learning difficulty of the multiple components is intrinsically imbalanced. To tackle this issue, we customize a curriculum learning strategy to reweight the importance of different negatives. In addition, to improve the interpretability in the feature space, we build a physics-aware dual-branch unit according to the atmospheric scattering model. With the unit, as well as curricular contrastive regularization, we establish our dehazing network, named C
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
PNet. Extensive experiments demonstrate that our C
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
PNet significantly outperforms state-of-the-art methods, with extreme PSNR boosts of 3.94dB and 1.50dB, respectively, on SOTS-indoor and SOTS-outdoor datasets. Code is available at https://github.com/YuZheng9/C2PNet.

----

## [552] PatchCraft Self-Supervised Training for Correlated Image Denoising

**Authors**: *Gregory Vaksman, Michael Elad*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00561](https://doi.org/10.1109/CVPR52729.2023.00561)

**Abstract**:

Supervised neural networks are known to achieve excellent results in various image restoration tasks. However, such training requires datasets composed of pairs of corrupted images and their corresponding ground truth targets. Unfortunately, such data is not available in many applications. For the task of image denoising in which the noise statistics is unknown, several self-supervised training methods have been proposed for overcoming this difficulty. Some of these require knowledge of the noise model, while others assume that the contaminating noise is uncorrelated, both assumptions are too limiting for many practical needs. This work proposes a novel self-supervised training technique suitable for the removal of unknown correlated noise. The proposed approach neither requires knowledge of the noise model nor access to ground truth targets. The input to our algorithm consists of easily captured bursts of noisy shots. Our algorithm constructs artificial patch-craft images from these bursts by patch matching and stitching, and the obtained crafted images are used as targets for the training. Our method does not require registration of the images within the burst. We evaluate the proposed framework through extensive experiments with synthetic and real image noise.

----

## [553] Spectral Enhanced Rectangle Transformer for Hyperspectral Image Denoising

**Authors**: *Miaoyu Li, Ji Liu, Ying Fu, Yulun Zhang, Dejing Dou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00562](https://doi.org/10.1109/CVPR52729.2023.00562)

**Abstract**:

Denoising is a crucial step for hyperspectral image (HSI) applications. Though witnessing the great power of deep learning, existing HSI denoising methods suffer from limitations in capturing the nonlocal self-similarity. Trans-formers have shown potential in capturing longrange de-pendencies, but few attempts have been made with specifically designed Transformer to model the spatial and spec-tral correlation in HSIs. In this paper, we address these issues by proposing a spectral enhanced rectangle Trans-former, driving it to explore the nonlocal spatial similarity and global spectral low-rank property of HSIs. For the former, we exploit the rectangle self-attention horizontally and vertically to capture the nonlocal similarity in the spatial domain. For the latter, we design a spectral enhancement module that is capable of extracting global underlying low-rank property of spatial-spectral cubes to suppress noise, while enabling the interactions among non-overlapping spatial rectangles. Extensive experiments have been conducted on both synthetic noisy HSIs and real noisy HSIs, showing the effectiveness of our proposed method in terms of both objective metric and subjective visual quality. The code is available at https://github.com/MyuLi/SERT.

----

## [554] All-in-One Image Restoration for Unknown Degradations Using Adaptive Discriminative Filters for Specific Degradations

**Authors**: *Dongwon Park, Byung Hyun Lee, Se Young Chun*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00563](https://doi.org/10.1109/CVPR52729.2023.00563)

**Abstract**:

Image restorations for single degradations have been widely studied, demonstrating excellent performance for each degradation, but can not reflect unpredictable realistic environments with unknown multiple degradations, which may change over time. To mitigate this issue, image restorations for known and unknown multiple degradations have recently been investigated, showing promising results, but require large networks or have sub-optimal architectures for potential interference among different degradations. Here, inspired by the filter attribution integrated gradients (FAIG), we propose an adaptive discriminative filter-based model for specific degradations (ADMS) to restore images with unknown degradations. Our method allows the network to contain degradation-dedicated filters only for about 3% of all network parameters per each degradation and to apply them adaptively via degradation classification (DC) to explicitly disentangle the network for multiple degradations. Our proposed method has demonstrated its effectiveness in comparison studies and achieved state-of-the-art performance in all-in-one image restoration benchmark datasets of both Rain-Noise-Blur and Rain-Snow-Haze.

----

## [555] Ingredient-oriented Multi-Degradation Learning for Image Restoration

**Authors**: *Jinghao Zhang, Jie Huang, Mingde Yao, Zizheng Yang, Hu Yu, Man Zhou, Feng Zhao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00564](https://doi.org/10.1109/CVPR52729.2023.00564)

**Abstract**:

Learning to leverage the relationship among diverse image restoration tasks is quite beneficial for unraveling the intrinsicingredients behind the degradation. Recent years have witnessed the flourish of various All-in-one methods, which handle multiple image degradations within a single model. In practice, however, few attempts have been made to excavate task correlations in that exploring the underlying fundamentalingredients of various image degradations, resulting in poor scalability as more tasks are involved. In this paper, we propose a novel perspective to delve into the degradation via aningredients-oriented rather than previous task-oriented manner for scalable learning. Specifically, our method, named Ingredients-oriented Degradation Reformulation framework (IDR), consists of two stages, namely task-oriented knowledge collection and ingredients-oriented knowledge integration. In the first stage, we conduct ad hoc operations on different degradations according to the underlying physics principles, and establish the corresponding prior hubs for each type of degradation. While the second stage progressively reformulates the preceding task-oriented hubs into single ingredients-oriented hub via learnable Principal Component Analysis (PCA), and employs a dynamic routing mechanism for probabilistic unknown degradation removal. Extensive experiments on various image restoration tasks demonstrate the effectiveness and scalability of our method. More importantly, our IDR exhibits the favorable generalization ability to unknown downstream tasks.

----

## [556] CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability

**Authors**: *Fadi Boutros, Meiling Fang, Marcel Klemt, Biying Fu, Naser Damer*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00565](https://doi.org/10.1109/CVPR52729.2023.00565)

**Abstract**:

Face image quality assessment (FIQA) estimates the utility of the captured image in achieving reliable and accurate recognition performance. This work proposes a novel FIQA method, CR-FIQA, that estimates the face image quality of a sample by learning to predict its relative classifiability. This classifiability is measured based on the allocation of the training sample feature representation in angular space with respect to its class center and the nearest negative class center. We experimentally illustrate the correlation between the face image quality and the sample relative classifiability. As such property is only observable for the training dataset, we propose to learn this property by probing internal network observations during the training process and utilizing it to predict the quality of unseen samples. Through extensive evaluation experiments on eight benchmarks and four face recognition models, we demonstrate the superiority of our proposed CR-FIQA over state-of-the-art (SOTA) FIQA algorithms.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/fdbtrs/CR-FIQA

----

## [557] Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild

**Authors**: *Avinab Saha, Sandeep Mishra, Alan C. Bovik*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00566](https://doi.org/10.1109/CVPR52729.2023.00566)

**Abstract**:

Automatic Perceptual Image Quality Assessment is a challenging problem that impacts billions of internet, and social media users daily. To advance research in this field, we propose a Mixture of Experts approach to train two separate encoders to learn high-level content and low-level image quality features in an unsupervised setting. The unique novelty of our approach is its ability to generate low-level representations of image quality that are complementary to high-level features representing image content. We refer to the framework used to train the two encoders as Re-IQA. For Image Quality Assessment in the Wild, we deploy the complementary low and high-level image representations obtained from the Re-IQA framework to train a linear regression model, which is used to map the image representations to the ground truth quality scores, refer Figure 1. Our method achieves state-of-the-art performance on multiple large-scale image quality assessment databases containing both real and synthetic distortions, demonstrating how deep neural networks can be trained in an unsupervised setting to produce perceptually relevant representations. We conclude from our experiments that the low and high-level features obtained are indeed complementary and positively impact the performance of the linear regressor. A public release of all the codes associated with this work will be made available on GitHub.

----

## [558] Toward Accurate Post-Training Quantization for Image Super Resolution

**Authors**: *Zhijun Tu, Jie Hu, Hanting Chen, Yunhe Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00567](https://doi.org/10.1109/CVPR52729.2023.00567)

**Abstract**:

Model quantization is a crucial step for deploying super resolution (SR) networks on mobile devices. However, existing works focus on quantization-aware training, which requires complete dataset and expensive computational overhead. In this paper, we study post-training quantization (PTQ) for image super resolution using only a few unlabeled calibration images. As the SR model aims to maintain the texture and color information of input images, the distribution of activations are long-tailed, asymmetric and highly dynamic compared with classification models. To this end, we introduce the density-based dual clipping to cut off the outliers based on analyzing the asymmetric bounds of activations. Moreover, we present a novel pixel aware calibration method with the supervision of the full-precision model to accommodate the highly dynamic range of different samples. Extensive experiments demonstrate that the proposed method significantly outperforms existing PTQ algorithms on various models and datasets. For instance, we get a 2.091 dB increase on Urban100 benchmark when quantizing 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$EDSR\times4$</tex>
 to 4-bit with 100 unlabeled images. Our code is available at both PyTorch and MindSpore.

----

## [559] Learning Steerable Function for Efficient Image Resampling

**Authors**: *Jiacheng Li, Chang Chen, Wei Huang, Zhiqiang Lang, Fenglong Song, Youliang Yan, Zhiwei Xiong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00568](https://doi.org/10.1109/CVPR52729.2023.00568)

**Abstract**:

Image resampling is a basic technique that is widely employed in daily applications. Existing deep neural networks (DNNs) have made impressive progress in resampling performance. Yet these methods are still not the perfect substitute for interpolation, due to the issues of efficiency and continuous resampling. In this work, we propose a novel method of Learning Resampling Function (termed LeRF), which takes advantage of both the structural priors learned by DNNs and the locally continuous assumption of interpolation methods. Specifically, LeRF assigns spatially-varying steerable resampling functions to input image pixels and learns to predict the hyper-parameters that determine the orientations of these resampling functions with a neural network. To achieve highly efficient inference, we adopt look-up tables (LUTs) to accelerate the inference of the learned neural network. Furthermore, we design a directional ensemble strategy and edge-sensitive indexing patterns to better capture local structures. Extensive experiments show that our method runs as fast as interpolation, generalizes well to arbitrary transformations, and outperforms interpolation significantly, e.g., up to 3dB PSNR gain over bicubic for x 2 upsampling on Manga109.

----

## [560] ABCD : Arbitrary Bitwise Coefficient for De-Quantization

**Authors**: *Woo Kyoung Han, Byeonghun Lee, Sang Hyun Park, Kyong Hwan Jin*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00569](https://doi.org/10.1109/CVPR52729.2023.00569)

**Abstract**:

Modern displays and contents support more than 8bits image and video. However, bit-starving situations such as compression codecs make low bit-depth (LBD) images (<8bits), occurring banding and blurry artifacts. Previous bit depth expansion (BDE) methods still produce unsatisfactory high bit-depth (HBD) images. To this end, we propose an implicit neural function with a bit query to recover de-quantized images from arbitrarily quantized inputs. We develop a phasor estimator to exploit the information of the nearest pixels. Our method shows superior performance against prior BDE methods on natural and animation images. We also demonstrate our model on YouTube UGC datasets for de-banding. Our source code is available at https://github.com/WooKyoungHan/ABCD

----

## [561] Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring

**Authors**: *Lingshun Kong, Jiangxin Dong, Jianjun Ge, Mingqiang Li, Jinshan Pan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00570](https://doi.org/10.1109/CVPR52729.2023.00570)

**Abstract**:

We present an effective and efficient method that explores the properties of Transformers in the frequency domain for high-quality image deblurring. Our method is motivated by the convolution theorem that the correlation or convolution of two signals in the spatial domain is equivalent to an element-wise product of them in the frequency domain. This inspires us to develop an efficient frequency domain-based self-attention solver (FSAS) to estimate the scaled dot-product attention by an element-wise product operation instead of the matrix multiplication in the spatial domain. In addition, we note that simply using the naive feed-forward network (FFN) in Transformers does not generate good deblurred results. To overcome this problem, we propose a simple yet effective discriminative frequency domain-based FFN (DFFN), where we introduce a gated mechanism in the FFN based on the Joint Photographic Experts Group (JPEG) compression algorithm to discriminatively determine which low- and high-frequency information of the features should be preserved for latent clear image restoration. We formulate the proposed FSAS and DFFN into an asymmetrical network based on an encoder and decoder architecture, where the FSAS is only used in the decoder module for better image deblurring. Experimental results show that the proposed method performs favorably against the state-of-the-art approaches.

----

## [562] Learning A Sparse Transformer Network for Effective Image Deraining

**Authors**: *Xiang Chen, Hao Li, Mingqiang Li, Jinshan Pan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00571](https://doi.org/10.1109/CVPR52729.2023.00571)

**Abstract**:

Transformers-based methods have achieved significant performance in image deraining as they can model the non-local information which is vital for high-quality image reconstruction. In this paper, we find that most existing Transformers usually use all similarities of the tokens from the query-key pairs for the feature aggregation. However, if the tokens from the query are different from those of the key, the self-attention values estimated from these tokens also involve in feature aggregation, which accordingly interferes with the clear image restoration. To overcome this problem, we propose an effective DeRaining network, Sparse Transformer (DRSformer) that can adaptively keep the most useful self-attention values for feature aggregation so that the aggregated features better facilitate high-quality image reconstruction. Specifically, we develop a learnable top-k selection operator to adaptively retain the most crucial attention scores from the keys for each query for better feature aggregation. Simultaneously, as the naive feed-forward network in Transformers does not model the multi-scale information that is important for latent clear image restoration, we develop an effective mixed-scale feed-forward network to generate better features for image deraining. To learn an enriched set of hybrid features, which combines local context from CNN operators, we equip our model with mixture of experts feature compensator to present a cooperation refinement deraining scheme. Extensive experimental results on the commonly used benchmarks demonstrate that the proposed method achieves favorable performance against state-of-the-art approaches. The source code and trained models are available at https://github.com/cschenxiang/DRSformer.

----

## [563] CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion

**Authors**: *Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Shuang Xu, Zudi Lin, Radu Timofte, Luc Van Gool*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00572](https://doi.org/10.1109/CVPR52729.2023.00572)

**Abstract**:

Multi-modality (MM) image fusion aims to render fused images that maintain the merits of different modalities, e.g., functional highlight and detailed textures. To tackle the challenge in modeling cross-modality features and decomposing desirable modality-specific and modality-shared features, we propose a novel Correlation-Driven feature Decomposition Fusion (CDDFuse) network. Firstly, CDDFuse uses Restormer blocks to extract cross-modality shallow features. We then introduce a dual-branch Transformer-CNN feature extractor with Lite Transformer (LT) blocks leveraging long-range attention to handle low-frequency global features and Invertible Neural Networks (INN) blocks focusing on extracting high-frequency local information. A correlation-driven loss is further proposed to make the low-frequency features correlated while the high-frequency features uncorrelated based on the embedded information. Then, the LT-based global fusion and INN-based local fusion layers output the fused image. Extensive experiments demonstrate that our CDDFuse achieves promising results in multiple fusion tasks, including infrared-visible image fusion and medical image fusion. We also show that CDDFuse can boost the performance in downstream infrared-visible semantic segmentation and object detection in a unified benchmark. The code is available at https://github.om/haozixiang1228/MMIF-CDDFuse.

----

## [564] PCT-Net: Full Resolution Image Harmonization Using Pixel-Wise Color Transformations

**Authors**: *Julian Jorge Andrade Guerreiro, Mitsuru Nakazawa, Björn Stenger*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00573](https://doi.org/10.1109/CVPR52729.2023.00573)

**Abstract**:

In this paper, we present PCT-Net, a simple and general image harmonization method that can be easily applied to images at full-resolution. The key idea is to learn a parameter network that uses downsampled input images to predict the parameters for pixel-wise color transforms (PCTs) which are applied to each pixel in the full-resolution image. We show that affine color transforms are both efficient and effective, resulting in state-of-the-art harmonization results. Moreover, we explore both CNNs and Transformers as the parameter network, and show that Transformers lead to better results. We evaluate the proposed method on the public full-resolution iHarmony4 dataset, which is comprised of four datasets, and show a reduction of the foreground MSE (fMSE) and MSE values by more than 20% and an increase of the PSNR value by 1.4dB, while keeping the architecture light-weight. In a user study with 20 people, we show that the method achieves a higher B-T score than two other recent methods.

----

## [565] Semi-Supervised Parametric Real-World Image Harmonization

**Authors**: *Ke Wang, Michaël Gharbi, He Zhang, Zhihao Xia, Eli Shechtman*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00574](https://doi.org/10.1109/CVPR52729.2023.00574)

**Abstract**:

Learning-based image harmonization techniques are usually trained to undo synthetic random global transformations applied to a masked foreground in a single ground truth photo. This simulated data does not model many of the important appearance mismatches (illumination, object boundaries, etc.) between foreground and background in real composites, leading to models that do not generalize well and cannot model complex local changes. We propose a new semi-supervised training strategy that addresses this problem and lets us learn complex local appearance harmonization from unpaired real composites, where foreground and background come from different images. Our model is fully parametric. It uses RGB curves to correct the global colors and tone and a shading map to model local variations. Our method outperforms previous work on established benchmarks and real composites, as shown in a user study, and processes high-resolution images interactively. Code, and project page available at: https://kewang0622.github.io/sprih/

----

## [566] Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution

**Authors**: *Chenfan Qu, Chongyu Liu, Yuliang Liu, Xinhong Chen, Dezhi Peng, Fengjun Guo, Lianwen Jin*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00575](https://doi.org/10.1109/CVPR52729.2023.00575)

**Abstract**:

Recently, tampered text detection in document image has attracted increasingly attention due to its essential role on information security. However, detecting visually consistent tampered text in photographed document images is still a main challenge. In this paper, we propose a novel framework to capture more fine-grained clues in complex scenarios for tampered text detection, termed as Document Tampering Detector (DTD), which consists of a Frequency Perception Head (FPH) to compensate the deficiencies caused by the inconspicuous visual features, and a Multi-view Iterative Decoder (MID) for fully utilizing the information of features in different scales. In addition, we design a new training paradigm, termed as Curriculum Learning for Tampering Detection (CLTD), which can address the confusion during the training procedure and thus to improve the robustness for image compression and the ability to generalize. To further facilitate the tampered text detection in document images, we construct a large-scale document image dataset, termed as DocTamper, which contains 170,000 document images of various types. Experiments demonstrate that our proposed DTD outperforms previous state-of-the-art by 9.2%, 26.3% and 12.3% in terms of F-measure on the DocTamper testing set, and the crossdomain testing sets of DocTamper-FCD and DocTamper-SCD, respectively. Codes and dataset will be available at https://github.com/qcf-568/DocTamper.

----

## [567] QuantArt: Quantizing Image Style Transfer Towards High Visual Fidelity

**Authors**: *Siyu Huang, Jie An, Donglai Wei, Jiebo Luo, Hanspeter Pfister*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00576](https://doi.org/10.1109/CVPR52729.2023.00576)

**Abstract**:

The mechanism of existing style transfer algorithms is by minimizing a hybrid loss function to push the generated image toward high similarities in both content and style. However, this type of approach cannot guarantee visual fidelity, i.e., the generated artworks should be indistinguishable from real ones. In this paper, we devise a new style transfer framework called QunntArt for high visual-fidelity stylization. QuantArt pushes the latent representation of the generated artwork toward the centroids of the real artwork distribution with vector quantization. By fusing the quantized and continuous latent representations, QuantArt allows flexible control over the generated artworks in terms of content preservation, style similarity, and visual fidelity. Experiments on various style transfer settings show that our QuantArt framework achieves significantly higher visual fidelity compared with the existing style transfer methods.

----

## [568] Deep Curvilinear Editing: Commutative and Nonlinear Image Manipulation for Pretrained Deep Generative Model

**Authors**: *Takehiro Aoshima, Takashi Matsubara*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00577](https://doi.org/10.1109/CVPR52729.2023.00577)

**Abstract**:

Semantic editing of images is the fundamental goal of computer vision. Although deep learning methods, such as generative adversarial networks (GANs), are capable of producing high-quality images, they often do not have an inherent way of editing generated images semantically. Recent studies have investigated a way of manipulating the latent variable to determine the images to be generated. However, methods that assume linear semantic arithmetic have certain limitations in terms of the quality of image editing, whereas methods that discover nonlinear semantic pathways provide non-commutative editing, which is inconsistent when applied in different orders. This study proposes a novel method called deep curvilinear editing (DeCurvEd) to determine semantic commuting vector fields on the latent space. We theoretically demonstrate that owing to commutativity, the editing of multiple attributes depends only on the quantities and not on the order. Furthermore, we experimentally demonstrate that compared to previous methods, the nonlinear and commutative nature of DeCurvEdfacilitates the disentanglement of image attributes and provides higher-quality editing.

----

## [569] Person Image Synthesis via Denoising Diffusion Model

**Authors**: *Ankan Kumar Bhunia, Salman H. Khan, Hisham Cholakkal, Rao Muhammad Anwer, Jorma Laaksonen, Mubarak Shah, Fahad Shahbaz Khan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00578](https://doi.org/10.1109/CVPR52729.2023.00578)

**Abstract**:

The pose-guided person image generation task requires synthesizing photorealistic images of humans in arbitrary poses. The existing approaches use generative adversarial networks that do not necessarily maintain realistic textures or need dense correspondences that struggle to handle complex deformations and severe occlusions. In this work, we show how denoising diffusion models can be applied for high-fidelity person image synthesis with strong sample diversity and enhanced mode coverage of the learnt data distribution. Our proposed Person Image Diffusion Model (PIDM) disintegrates the complex transfer problem into a series of simpler forward-backward denoising steps. This helps in learning plausible source-to-target transformation trajectories that result in faithful textures and undistorted appearance details. We introduce a ‘texture diffusion module’ based on cross-attention to accurately model the correspondences between appearance and pose information available in source and target images. Further, we propose ‘disentangled classifier-free guidance’ to ensure close resemblance between the conditional inputs and the synthesized output in terms of both pose and appearance information. Our extensive results on two large-scale benchmarks and a user study demonstrate the photorealism of our proposed approach under challenging scenarios. We also show how our generated images can help in downstream tasks. Code is available at https://github.com/ankanbhunia/PIDM.

----

## [570] Disentangling Writer and Character Styles for Handwriting Generation

**Authors**: *Gang Dai, Yifan Zhang, Qingfeng Wang, Qing Du, Zhuliang Yu, Zhuoman Liu, Shuangping Huang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00579](https://doi.org/10.1109/CVPR52729.2023.00579)

**Abstract**:

Training machines to synthesize diverse handwritings is an intriguing task. Recently, RNN-based methods have been proposed to generate stylized online Chinese characters. However, these methods mainly focus on capturing a person's overall writing style, neglecting subtle style inconsistencies between characters written by the same person. For example, while a person's handwriting typically exhibits general uniformity (e.g., glyph slant and aspect ratios), there are still small style variations in finer details (e.g., stroke length and curvature) of characters. In light of this, we propose to disentangle the style representations at both writer and character levels from individual handwritings to synthesize realistic stylized online handwritten characters. Specifically, we present the style-disentangled Transformer (SDT), which employs two complementary contrastive objectives to extract the style commonalities of reference samples and capture the detailed style patterns of each sample, respectively. Extensive experiments on various language scripts demonstrate the effectiveness of SDT. Notably, our empirical findings reveal that the two learned style representations provide information at different frequency magnitudes, underscoring the importance of separate style extraction. Our source code is public at: https://github.com/dailenson/SDT.

----

## [571] NoisyTwins: Class-Consistent and Diverse Image Generation Through StyleGANs

**Authors**: *Harsh Rangwani, Lavish Bansal, Kartik Sharma, Tejan Karmali, Varun Jampani, R. Venkatesh Babu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00580](https://doi.org/10.1109/CVPR52729.2023.00580)

**Abstract**:

StyleGANs are at the forefront of controllable image generation as they produce a latent space that is semantically disentangled, making it suitable for image editing and manipulation. However, the performance of StyleGANs severely degrades when trained via class-conditioning on large-scale long-tailed datasets. We find that one reason for degradation is the collapse of latents for each class in the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{W}$</tex>
 latent space. With NoisyTwins, we first introduce an effective and inexpensive augmentation strategy for class embeddings, which then decorrelates the latents based on self-supervision in the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{W}$</tex>
 space. This decorrelation mitigates collapse, ensuring that our method preserves intra-class diversity with class-consistency in image generation. We show the effectiveness of our approach on large-scale real-world long-tailed datasets of ImageNet-LT and iNaturalist 2019, where our method outperforms other methods by ∼ 19% on FID, establishing a new state-of-the-art.

----

## [572] High-Fidelity Guided Image Synthesis with Latent Diffusion Models

**Authors**: *Jaskirat Singh, Stephen Gould, Liang Zheng*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00581](https://doi.org/10.1109/CVPR52729.2023.00581)

**Abstract**:

Controllable image synthesis with user scribbles has gained huge public interest with the recent advent of text-conditioned latent diffusion models. The user scribbles control the color composition while the text prompt provides control over the overall image semantics. However, we note that prior works in this direction suffer from an intrinsic domain shift problem wherein the generated outputs often lack details and resemble simplistic representations of the target domain. In this paper, we propose a novel guided image synthesis framework, which addresses this problem by modelling the output image as the solution of a constrained optimization problem. We show that while computing an exact solution to the optimization is infeasible, an approximation of the same can be achieved while just requiring a single pass of the reverse diffusion process. Additionally, we show that by simply defining a cross-attention based correspondence between the input text tokens and the user stroke-painting, the user is also able to control the semantics of different painted regions without requiring any conditional training or finetuning. Human user study results show that the proposed approach outperforms the previous state-of-the-art by over 85.32% on the overall user satisfaction scores. Project page for our paper is available at https://1jsingh.github.io/gradop.

----

## [573] Imagic: Text-Based Real Image Editing with Diffusion Models

**Authors**: *Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, Michal Irani*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00582](https://doi.org/10.1109/CVPR52729.2023.00582)

**Abstract**:

Text-conditioned image editing has recently attracted considerable interest. However, most methods are currently limited to one of the following: specific editing types (e.g., object overlay, style transfer), synthetically generated images, or requiring multiple input images of a common object. In this paper we demonstrate, for the very first time, the ability to apply complex (e.g., non-rigid) text-based semantic edits to a single real image. For example, we can change the posture and composition of one or multiple objects inside an image, while preserving its original characteristics. Our method can make a standing dog sit down, cause a bird to spread its wings, etc. – each within its single high-resolution user-provided natural image. Contrary to previous work, our proposed method requires only a single input image and a target text (the desired edit). It operates on real images, and does not require any additional inputs (such as image masks or additional views of the object). Our method, called Imagic, leverages a pre-trained text-to-image diffusion model for this task. It produces a text embedding that aligns with both the input image and the target text, while fine-tuning the diffusion model to capture the image-specific appearance. We demonstrate the quality and versatility of Imagic on numerous inputs from various domains, showcasing a plethora of high quality complex semantic image edits, all within a single unified framework. To better assess performance, we introduce TEdBench, a highly challenging image editing benchmark. We conduct a user study, whose findings show that human raters prefer Imagic to previous leading editing methods on TEdBench.

----

## [574] PosterLayout: A New Benchmark and Approach for Content-Aware Visual-Textual Presentation Layout

**Authors**: *HsiaoYuan Hsu, Xiangteng He, Yuxin Peng, Hao Kong, Qing Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00583](https://doi.org/10.1109/CVPR52729.2023.00583)

**Abstract**:

Content-aware visual-textual presentation layout aims at arranging spatial space on the given canvas for pre-defined elements, including text, logo, and underlay, which is a key to automatic template-free creative graphic design. In practical applications, e.g., poster designs, the canvas is originally non-empty, and both inter-element relationships as well as inter-layer relationships should be concerned when generating a proper layout. A few recent works deal with them simultaneously, but they still suffer from poor graphic performance, such as a lack of layout variety or spatial non-alignment. Since content-aware visual-textual presentation layout is a novel task, we first construct a new dataset named PKU PosterLayout, which consists of 9,974 poster- layout pairs and 905 images, i.e., non-empty canvases. It is more challenging and useful for greater layout variety, domain diversity, and content diversity. Then, we propose design sequence formation (DSF) that reorganizes elements in layouts to imitate the design processes of human designers, and a novel CNN-LSTM-based conditional generative adversarial network (GAN) is presented to generate proper layouts. Specifically, the discriminator is design-sequence- aware and will supervise the “design” process of the generator. Experimental results verify the usefulness of the new benchmark and the effectiveness of the proposed approach, which achieves the best performance by generating suitable layouts for diverse canvases. The dataset and the source code are available at https://github.com/PKU-ICST-MIPL/PosterLayout-CVPR2023.

----

## [575] SINE: SINgle Image Editing with Text-to-Image Diffusion Models

**Authors**: *Zhixing Zhang, Ligong Han, Arnab Ghosh, Dimitris N. Metaxas, Jian Ren*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00584](https://doi.org/10.1109/CVPR52729.2023.00584)

**Abstract**:

Recent works on diffusion models have demonstrated a strong capability for conditioning image generation, e.g., text-guided image synthesis. Such success inspires many efforts trying to use large-scale pre-trained diffusion models for tackling a challenging problem-real image editing. Works conducted in this area learn a unique textual token corresponding to several images containing the same object. However, under many circumstances, only one image is available, such as the painting of the Girl with a Pearl Earring. Using existing works on fine-tuning the pre-trained diffusion models with a single image causes severe overfitting issues. The information leakage from the pre-trained diffusion models makes editing can not keep the same content as the given image while creating new features depicted by the language guidance. This work aims to address the problem of single-image editing. We propose a novel model-based guidance built upon the classifier-free guidance so that the knowledge from the model trained on a single image can be distilled into the pre-trained diffusion model, enabling content creation even with one given image. Additionally, we propose a patch-based fine-tuning that can effectively help the model generate images of arbitrary resolution. We provide extensive experiments to validate the design choices of our approach and show promising editing capabilities, including changing style, content addition, and object manipulation. Our code is made publicly available here.

----

## [576] Null-text Inversion for Editing Real Images using Guided Diffusion Models

**Authors**: *Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, Daniel Cohen-Or*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00585](https://doi.org/10.1109/CVPR52729.2023.00585)

**Abstract**:

Recent large-scale text-guided diffusion models provide powerful image generation capabilities. Currently, a massive effort is given to enable the modification of these images using text only as means to offer intuitive and versatile editing tools. To edit a real image using these state-of-the-art tools, one must first invert the image with a meaningful text prompt into the pretrained model's domain. In this paper, we introduce an accurate inversion technique and thus facilitate an intuitive text-based modification of the image. Our proposed inversion consists of two key novel components: (i) Pivotal inversion for diffusion models. While current methods aim at mapping random noise samples to a single input image, we use a single pivotal noise vector for each timestamp and optimize around it. We demonstrate that a direct DDIM inversion is inadequate on its own, but does provide a rather good anchor for our optimization. (ii) Null-text optimization, where we only modify the unconditional textual embedding that is used for classifier-free guidance, rather than the input text embedding. This allows for keeping both the model weights and the conditional embedding intact and hence enables applying prompt-based editing while avoiding the cumbersome tuning of the model's weights. Our null-text inversion, based on the publicly available Stable Diffusion model, is extensively evaluated on a variety of images and various prompt editing, showing high-fidelity editing of real images.

----

## [577] Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models

**Authors**: *Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00586](https://doi.org/10.1109/CVPR52729.2023.00586)

**Abstract**:

Cutting-edge diffusion models produce images with high quality and customizability, enabling them to be used for commercial art and graphic design purposes. But do diffusion models create unique works of art, or are they replicating content directly from their training sets? In this work, we study image retrieval frameworks that enable us to compare generated images with training samples and detect when content has been replicated. Applying our frameworks to diffusion models trained on multiple datasets including Oxford flowers, Celeb-A, ImageNet, and LAION, we discuss how factors such as training set size impact rates of content replication. We also identify cases where diffusion models, including the popular Stable Diffusion model, blatantly copy from their training data. Project page: https://somepago.github.io/diffrep.html

----

## [578] Parallel Diffusion Models of Operator and Image for Blind Inverse Problems

**Authors**: *Hyungjin Chung, Jeongsol Kim, Sehui Kim, Jong Chul Ye*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00587](https://doi.org/10.1109/CVPR52729.2023.00587)

**Abstract**:

Diffusion model-based inverse problem solvers have demonstrated state-of-the-art performance in cases where the forward operator is known (i.e. non-blind). However, the applicability of the method to blind inverse problems has yet to be explored. In this work, we show that we can indeed solve a family of blind inverse problems by constructing another diffusion prior for the forward operator. Specifically, parallel reverse diffusion guided by gradients from the intermediate stages enables joint optimization of both the forward operator parameters as well as the image, such that both are jointly estimated at the end of the parallel reverse diffusion procedure. We show the efficacy of our method on two representative tasks - blind deblurring, and imaging through turbulence - and show that our method yields state-of-the-art performance, while also being flexible to be applicable to general blind inverse problems when we know the functional forms. Code available: https://github.com/BlindDPS/blind-dps

----

## [579] Unite and Conquer: Plug & Play Multi-Modal Synthesis Using Diffusion Models

**Authors**: *Nithin Gopalakrishnan Nair, Wele Gedara Chaminda Bandara, Vishal M. Patel*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00588](https://doi.org/10.1109/CVPR52729.2023.00588)

**Abstract**:

Generating photos satisfying multiple constraints finds broad utility in the content creation industry. A key hurdle to accomplishing this task is the need for paired data consisting of all modalities (i.e., constraints) and their corresponding output. Moreover, existing methods need retraining using paired data across all modalities to introduce a new condition. This paper proposes a solution to this problem based on denoising diffusion probabilistic models (DDPMs). Our motivation for choosing diffusion models over other generative models comes from the flexible internal structure of diffusion models. Since each sampling step in the DDPM follows a Gaussian distribution, we show that there exists a closed-form solution for generating an image given various constraints. Our method can unite multiple diffusion models trained on multiple sub-tasks and conquer the combined task through our proposed sampling strategy. We also introduce a novel reliability parameter that allows using different off-the-shelf diffusion models trained across various datasets during sampling time alone to guide it to the desired outcome satisfying multiple constraints. We perform experiments on various standard multimodal tasks to demonstrate the effectiveness of our approach. More details can be found at: https://nithin-gk.github.io/projectpages/Multidiff

----

## [580] Collaborative Diffusion for Multi-Modal Face Generation and Editing

**Authors**: *Ziqi Huang, Kelvin C. K. Chan, Yuming Jiang, Ziwei Liu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00589](https://doi.org/10.1109/CVPR52729.2023.00589)

**Abstract**:

Diffusion models arise as a powerful generative tool recently. Despite the great progress, existing diffusion models mainly focus on uni-modal control, i.e., the diffusion process is driven by only one modality of condition. To further unleash the users' creativity, it is desirable for the model to be controllable by multiple modalities simultaneously, e.g. generating and editing faces by describing the age (text-driven) while drawing the face shape (mask-driven). In this work, we present Collaborative Diffusion, where pre-trained uni-modal diffusion models collaborate to achieve multi-modal face generation and editing without re-training. Our key insight is that diffusion models driven by different modalities are inherently complementary regarding the latent denoising steps, where bilateral connections can be established upon. Specifically, we propose dynamic diffuser, a meta-network that adaptively hallucinates multimodal denoising steps by predicting the spatial-temporal influence functions for each pre-trained uni-modal model. Collaborative Diffusion not only collaborates generation capabilities from uni-modal diffusion models, but also integrates multiple uni-modal manipulations to perform multi-modal editing. Extensive qualitative and quantitative experiments demonstrate the superiority of our framework in both image quality and condition consistency.

----

## [581] Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding

**Authors**: *Gyeongman Kim, Hajin Shim, Hyunsu Kim, Yunjey Choi, Junho Kim, Eunho Yang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00590](https://doi.org/10.1109/CVPR52729.2023.00590)

**Abstract**:

Inspired by the impressive performance of recent face image editing methods, several studies have been naturally proposed to extend these methods to the face video editing task. One of the main challenges here is temporal consistency among edited frames, which is still unresolved. To this end, we propose a novel face video editing framework based on diffusion autoencoders that can successfully extract the decomposed features - for the first time as a face video editing model - of identity and motion from a given video. This modeling allows us to edit the video by simply manipulating the temporally invariant feature to the desired direction for the consistency. Another unique strength of our model is that, since our model is based on diffusion models, it can satisfy both reconstruction and edit capabilities at the same time, and is robust to corner cases in wild face videos (e.g. occluded faces) unlike the existing GAN-based methods.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://diff-video-ae.github.io

----

## [582] NVTC: Nonlinear Vector Transform Coding

**Authors**: *Runsen Feng, Zongyu Guo, Weiping Li, Zhibo Chen*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00591](https://doi.org/10.1109/CVPR52729.2023.00591)

**Abstract**:

In theory, vector quantization (VQ) is always better than scalar quantization (SQ) in terms of rate-distortion (RD) performance [33]. Recent state-of-the-art methods for neural image compression are mainly based on nonlinear transform coding (NTC) with uniform scalar quantization, overlooking the benefits of VQ due to its exponentially increased complexity. In this paper, we first investigate on some toy sources, demonstrating that even if modern neural networks considerably enhance the compression performance of SQ with nonlinear transform, there is still an insurmountable chasm between SQ and VQ. Therefore, revolving around VQ, we propose a novel framework for neural image compression named Nonlinear Vector Transform Coding (NVTC). NVTC solves the critical complexity issue of VQ through (1) a multi-stage quantization strategy and (2) nonlinear vector transforms. In addition, we apply entropy-constrained VQ in latent space to adaptively determine the quantization boundaries for joint rate-distortion optimization, which improves the performance both theoretically and experimentally. Compared to previous NTC approaches, NVTC demonstrates superior rate-distortion performance, faster decoding speed, and smaller model size. Our code is available at https://github.com/USTC-IMCL/NVTC.

----

## [583] Motion Information Propagation for Neural Video Compression

**Authors**: *Linfeng Qi, Jiahao Li, Bin Li, Houqiang Li, Yan Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00592](https://doi.org/10.1109/CVPR52729.2023.00592)

**Abstract**:

In most existing neural video codecs, the information flow therein is uni-directional, where only motion coding provides motion vectors for frame coding. In this paper, we argue that, through information interactions, the synergy between motion coding and frame coding can be achieved. We effectively introduce bi-directional information interactions between motion coding and frame coding via our Motion Information Propagation. When generating the temporal contexts for frame coding, the high-dimension motion feature from the motion decoder serves as motion guidance to mitigate the alignment errors. Meanwhile, besides assisting frame coding at the current time step, the feature from context generation will be propagated as motion condition when coding the subsequent motion latent. Through the cycle of such interactions, feature propagation on motion coding is built, strengthening the capacity of exploiting long-range temporal correlation. In addition, we propose hybrid context generation to exploit the multiscale context features and provide better motion condition. Experiments show that our method can achieve 12.9% bit rate saving over the previous SOTA neural video codec.

----

## [584] A Dynamic Multi-Scale Voxel Flow Network for Video Prediction

**Authors**: *Xiaotao Hu, Zhewei Huang, Ailin Huang, Jun Xu, Shuchang Zhou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00593](https://doi.org/10.1109/CVPR52729.2023.00593)

**Abstract**:

The performance of video prediction has been greatly boosted by advanced deep neural networks. However, most of the current methods suffer from large model sizes and require extra inputs, e.g., semantic/depth maps, for promising performance. For efficiency consideration, in this paper, we propose a Dynamic Multi-scale Voxel Flow Network (DMVFN) to achieve better video prediction performance at lower computational costs with only RGB images, than previous methods. The core of our DMVFN is a differentiable routing module that can effectively perceive the motion scales of video frames. Once trained, our DMVFN selects adaptive sub-networks for different inputs at the inference stage. Experiments on several benchmarks demonstrate that our DMVFN is an order of magnitude faster than Deep Voxel Flow [35] and surpasses the state-of-the-art iterative-based OPT [63] on generated image quality.

----

## [585] Towards Scalable Neural Representation for Diverse Videos

**Authors**: *Bo He, Xitong Yang, Hanyu Wang, Zuxuan Wu, Hao Chen, Shuaiyi Huang, Yixuan Ren, Ser-Nam Lim, Abhinav Shrivastava*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00594](https://doi.org/10.1109/CVPR52729.2023.00594)

**Abstract**:

Implicit neural representations (INR) have gained increasing attention in representing 3D scenes and images, and have been recently applied to encode videos (e.g., NeRV [1], E-NeRV [2]). While achieving promising results, existing INR-based methods are limited to encoding a handful of short videos (e.g., seven 5-second videos in the UVG dataset) with redundant visual content, leading to a model design that fits individual video frames independently and is not efficiently scalable to a large number of diverse videos. This paper focuses on developing neural representations for a more practical setup - encoding long and/or a large number of videos with diverse visual content. We first show that instead of dividing videos into small subsets and encoding them with separate models, encoding long and diverse videos jointly with a unified model achieves better compression results. Based on this observation, we propose D-NeRV, a novel neural representation framework designed to encode diverse videos by (i) decoupling clip-specific visual content from motion information, (ii) introducing temporal reasoning into the implicit neural network, and (iii) employing the task-oriented flow as intermediate output to reduce spatial redundancies. Our new model largely surpasses NeRV and traditional video compression techniques on UCF101 and UVG datasets on the video compression task. Moreover, when used as an efficient data-loader, D-NeRV achieves 3%-10% higher accuracy than NeRV on action recognition tasks on the UCF101 dataset under the same compression ratios.

----

## [586] DINER: Disorder-Invariant Implicit Neural Representation

**Authors**: *Shaowen Xie, Hao Zhu, Zhen Liu, Qi Zhang, You Zhou, Xun Cao, Zhan Ma*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00595](https://doi.org/10.1109/CVPR52729.2023.00595)

**Abstract**:

Implicit neural representation (INR) characterizes the attributes of a signal as a function of corresponding coordinates which emerges as a sharp weapon for solving inverse problems. However, the capacity of INR is limited by the spectral bias in the network training. In this paper, we find that such a frequency-related problem could be largely solved by re-arranging the coordinates of the input signal, for which we propose the disorder-invariant implicit neural representation (DINER) by augmenting a hash-table to a traditional INR backbone. Given discrete signals sharing the same histogram of attributes and different arrangement orders, the hash-table could project the coordinates into the same distribution for which the mapped signal can be better modeled using the subsequent INR network, leading to significantly alleviated spectral bias. Experiments not only reveal the generalization of the DINER for different INR backbones (MLP vs. SIREN) and various tasks (image/video representation, phase retrieval, and refractive index recovery) but also show the superiority over the state-of-the-art algorithms both in quality and speed. Project page: https://ezio77.github.io/DINER-website/

----

## [587] SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy

**Authors**: *Jiafeng Li, Ying Wen, Lianghua He*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00596](https://doi.org/10.1109/CVPR52729.2023.00596)

**Abstract**:

Convolutional Neural Networks (CNNs) have achieved remarkable performance in various computer vision tasks but this comes at the cost of tremendous computational resources, partly due to convolutional layers extracting redundant features. Recent works either compress well-trained large-scale models or explore well-designed lightweight models. In this paper, we make an attempt to exploit spatial and channel redundancy among features for CNN compression and propose an efficient convolution module, called SCConv (Spatial and Channel reconstruction Convolution), to decrease redundant computing and facilitate representative feature learning. The proposed SCConv consists of two units: spatial reconstruction unit (SRU) and channel reconstruction unit (CRU). SRU utilizes a separate-and-reconstruct method to suppress the spatial redundancy while CRU uses a split-transform-and-fuse strategy to diminish the channel redundancy. In addition, SCConv is a plug-and-play architectural unit that can be used to replace standard convolution in various convolutional neural networks directly. Experimental results show that SCConv-embedded models are able to achieve better performance by reducing redundant features with significantly lower complexity and computational costs.

----

## [588] DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network

**Authors**: *Xuan Shen, Yaohua Wang, Ming Lin, Yilun Huang, Hao Tang, Xiuyu Sun, Yanzhi Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00597](https://doi.org/10.1109/CVPR52729.2023.00597)

**Abstract**:

The rapid advances in Vision Transformer (ViT) refresh the state-of-the-art performances in various vision tasks, overshadowing the conventional CNN-based models. This ignites a few recent striking-back research in the CNN world showing that pure CNN models can achieve as good performance as ViT models when carefully tuned. While encouraging, designing such high-performance CNN models is challenging, requiring non-trivial prior knowledge of network design. To this end, a novel framework termed Mathematical Architecture Design for Deep CNN (Deep-MAD
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Source codes are available at https://github.com/alibaba/lightweight-neural-architecture-search) is proposed to design high-performance CNN models in a principled way. In DeepMAD, a CNN network is modeled as an information processing system whose expressiveness and effectiveness can be analytically formulated by their structural parameters. Then a constrained mathematical programming (MP) problem is proposed to optimize these structural parameters. The MP problem can be easily solved by off-the-shelf MP solvers on CPUs with a small memory footprint. In addition, DeepMAD is a pure mathematical framework: no GPU or training data is required during network design. The superiority of DeepMAD is validated on multiple large-scale computer vision benchmark datasets. Notably on ImageNet-1k, only using conventional convolutional layers, DeepMAD achieves 0.7% and 1.5% higher top-1 accuracy than ConvNeXt and Swin on Tiny level, and 0.8% and 0.9% higher on Small level.

----

## [589] Optimization-Inspired Cross-Attention Transformer for Compressive Sensing

**Authors**: *Jiechong Song, Chong Mou, Shiqi Wang, Siwei Ma, Jian Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00598](https://doi.org/10.1109/CVPR52729.2023.00598)

**Abstract**:

By integrating certain optimization solvers with deep neural networks, deep unfolding network (DUN) with good interpretability and high performance has attracted growing attention in compressive sensing (CS). However, existing DUNs often improve the visual quality at the price of a large number of parameters and have the problem of feature information loss during iteration. In this paper, we propose an Optimization-inspired Cross-attention Transformer (OCT) module as an iterative process, leading to a lightweight OCT-based Unfolding Framework (OCTUF) for image CS. Specifically, we design a novel Dual Cross Attention (Dual-CA) sub-module, which consists of an Inertia-Supplied Cross Attention (ISCA) block and a Projection-Guided Cross Attention (PGCA) block. ISCA block introduces multi-channel inertia forces and increases the memory effect by a cross attention mechanism between adjacent iterations. And, PGCA block achieves an enhanced information interaction, which introduces the inertia force into the gradient descent step through a cross attention block. Extensive CS experiments manifest that our OCTUF achieves superior performance compared to state-of-the-art methods while training lower complexity. Codes are available at https://github.com/songjiechong/OCTUF.

----

## [590] Neighborhood Attention Transformer

**Authors**: *Ali Hassani, Steven Walton, Jiachen Li, Shen Li, Humphrey Shi*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00599](https://doi.org/10.1109/CVPR52729.2023.00599)

**Abstract**:

We present Neighborhood Attention (NA), the first efficient and scalable sliding window attention mechanism for vision. NA is a pixel-wise operation, localizing self attention (SA) to the nearest neighboring pixels, and therefore enjoys a linear time and space complexity compared to the quadratic complexity of SA. The sliding window pattern allows NA's receptive field to grow without needing extra pixel shifts, and preserves translational equivariance, unlike Swin Transformer's Window Self Attention (WSA). We develop NATTEN (Neighborhood Attention Extension), a Python package with efficient C++ and CUDA kernels, which allows NA to run up to 40% faster than Swin's WSA while using up to 25% less memory. We further present Neighborhood Attention Transformer (NAT), a new hierarchical transformer design based on NA that boosts image classification and downstream vision performance. Experimental results on NAT are competitive; NAT-Tiny reaches 83.2% top-1 accuracy on ImageNet, 51.4% mAP on MSCOCO and 48.4% mIoU on ADE20K, which is 1.9% ImageNet accuracy, 1.0% COCO mAP, and 2.6% ADE20K mIoU improvement over a Swin model with similar size. To support more research based on sliding window attention, we open source our project and release our checkpoints.

----

## [591] Making Vision Transformers Efficient from A Token Sparsification View

**Authors**: *Shuning Chang, Pichao Wang, Ming Lin, Fan Wang, David Junhao Zhang, Rong Jin, Mike Zheng Shou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00600](https://doi.org/10.1109/CVPR52729.2023.00600)

**Abstract**:

The quadratic computational complexity to the number of tokens limits the practical applications of Vision Transformers (ViTs). Several works propose to prune redundant tokens to achieve efficient ViTs. However, these methods generally suffer from (i) dramatic accuracy drops, (ii) application difficulty in the local vision transformer, and (iii) non-general-purpose networks for downstream tasks. In this work, we propose a novel Semantic Token ViT (STViT), for efficient global and local vision transformers, which can also be revised to serve as backbone for downstream tasks. The semantic tokens represent cluster centers, and they are initialized by pooling image tokens in space and recovered by attention, which can adaptively represent global or local semantic information. Due to the cluster properties, a few semantic tokens can attain the same effect as vast image tokens, for both global and local vision transformers. For instance, only 16 semantic tokens on DeiT-(Tiny,Small,Base) can achieve the same accuracy with more than 100% inference speed improvement and nearly 60% FLOPs reduction; on Swin-(Tiny,Small,Base), we can employ 16 semantic tokens in each window to further speed it up by around 20% with slight accuracy increase. Besides great success in image classification, we also extend our method to video recognition. In addition, we design a STViT-R(ecovery) network to restore the detailed spatial information based on the STViT, making it work for downstream tasks, which is powerless for previous token sparsification methods. Experiments demonstrate that our method can achieve competitive results compared to the original networks in object detection and instance segmentation, with over 30% FLOPs reduction for backbone.

----

## [592] Towards Efficient Use of Multi-Scale Features in Transformer-Based Object Detectors

**Authors**: *Gongjie Zhang, Zhipeng Luo, Zichen Tian, Jingyi Zhang, Xiaoqin Zhang, Shijian Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00601](https://doi.org/10.1109/CVPR52729.2023.00601)

**Abstract**:

Multi-scale features have been proven highly effective for object detection but often come with huge and even prohibitive extra computation costs, especially for the recent Transformer-based detectors. In this paper, we propose Iterative Multi-scale Feature Aggregation (IMFA) - a generic paradigm that enables efficient use of multi-scale features in Transformer-based object detectors. The core idea is to exploit sparse multi-scale features from just a few crucial locations, and it is achieved with two novel designs. First, IMFA rearranges the Transformer encoder-decoder pipeline so that the encoded features can be iteratively updated based on the detection predictions. Second, IMFA sparsely samples scale-adaptive features for refined detection from just a few keypoint locations under the guidance of prior detection predictions. As a result, the sampled multi-scale features are sparse yet still highly beneficial for object detection. Extensive experiments show that the proposed IMFA boosts the performance of multiple Transformer-based object detectors significantly yet with only slight computational overhead.

----

## [593] Neuralizer: General Neuroimage Analysis without Re-Training

**Authors**: *Steffen Czolbe, Adrian V. Dalca*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00602](https://doi.org/10.1109/CVPR52729.2023.00602)

**Abstract**:

Neuroimage processing tasks like segmentation, reconstruction, and registration are central to the study of neuroscience. Robust deep learning strategies and architectures used to solve these tasks are often similar. Yet, when presented with a new task or a dataset with different visual characteristics, practitioners most often need to train a new model, or fine-tune an existing one. This is a time-consuming process that poses a substantial barrier for the thousands of neuroscientists and clinical researchers who often lack the resources or machine-learning expertise to train deep learning models. In practice, this leads to a lack of adoption of deep learning, and neuroscience tools being dominated by classical frameworks. We introduce Neuralizer, a single model that generalizes to previously unseen neuroimaging tasks and modalities without the need for retraining or fine-tuning. Tasks do not have to be known a priori, and generalization happens in a single forward pass during inference. The model can solve processing tasks across multiple image modalities, acquisition methods, and datasets, and generalize to tasks and modalities it has not been trained on. Our experiments on coronal slices show that when few annotated subjects are available, our multi-task network outperforms task-specific baselines without training on the task.

----

## [594] Learning Partial Correlation based Deep Visual Representation for Image Classification

**Authors**: *Saimunur Rahman, Piotr Koniusz, Lei Wang, Luping Zhou, Peyman Moghadam, Changming Sun*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00603](https://doi.org/10.1109/CVPR52729.2023.00603)

**Abstract**:

Visual representation based on covariance matrix has demonstrates its efficacy for image classification by characterising the pairwise correlation of different channels in convolutional feature maps. However, pairwise correlation will become misleading once there is another channel correlating with both channels of interest, resulting in the “confounding” effect. For this case, “partial correlation” which removes the confounding effect shall be estimated instead. Nevertheless, reliably estimating partial correlation requires to solve a symmetric positive definite matrix optimisation, known as sparse inverse covariance estimation (SICE). How to incorporate this process into CNN remains an open issue. In this work, we formulate SICE as a novel structured layer of CNN. To ensure end-to-end trainability, we develop an iterative method to solve the above matrix optimisation during forward and backward propagation steps. Our work obtains a partial correlation based deep visual representation and mitigates the small sample problem often encountered by covariance matrix estimation in CNN. Computationally, our model can be effectively trained with GPU and works well with a large number of channels of advanced CNNs. Experiments show the efficacy and superior classification performance of our deep visual representation compared to covariance matrix based counterparts.

----

## [595] Understanding Masked Image Modeling via Learning Occlusion Invariant Feature

**Authors**: *Xiangwen Kong, Xiangyu Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00604](https://doi.org/10.1109/CVPR52729.2023.00604)

**Abstract**:

Recently, Masked Image Modeling (MIM) achieves great success in self-supervised visual recognition. However, as a reconstruction-based framework, it is still an open question to understand how MIM works, since MIM appears very different from previous well-studied siamese approaches such as contrastive learning. In this paper, we propose a new viewpoint: MIM implicitly learns occlusion-invariant features, which is analogous to other siamese methods while the latter learns other invariance. By relaxing MIM formulation into an equivalent siamese form, MIM methods can be interpreted in a unified framework with conventional methods, among which only a) data transformations, i.e. what invariance to learn, and b) similarity measurements are different. Furthermore, taking MAE [29] as a representative example of MIM, we empirically find the success of MIM models relates a little to the choice of similarity functions, but the learned occlusion invariant feature introduced by masked image - it turns out to be a favored initialization for vision transformers, even though the learned feature could be less semantic. We hope our findings could inspire researchers to develop more powerful self-supervised methods in computer vision community.

----

## [596] MixMAE: Mixed and Masked Autoencoder for Efficient Pretraining of Hierarchical Vision Transformers

**Authors**: *Jihao Liu, Xin Huang, Jinliang Zheng, Yu Liu, Hongsheng Li*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00605](https://doi.org/10.1109/CVPR52729.2023.00605)

**Abstract**:

In this paper, we propose Mixed and Masked AutoEncoder (MixMAE), a simple but efficient pretraining method that is applicable to various hierarchical Vision Transformers. Existing masked image modeling (MIM) methods for hierarchical Vision Transformers replace a random subset of input tokens with a special [MASK] symbol and aim at reconstructing original image tokens from the corrupted image. However, we find that using the [MASK] symbol greatly slows down the training and causes pretraining-finetuning inconsistency, due to the large masking ratio (e.g., 60% in SimMIM). On the other hand, MAE does not introduce [MASK] tokens at its encoder at all but is not applicable for hierarchical Vision Transformers. To solve the issue and accelerate the pretraining of hierarchical models, we replace the masked tokens of one image with visible tokens of another image, i.e., creating a mixed image. We then conduct dual reconstruction to reconstruct the two original images from the mixed input, which significantly improves efficiency. While MixMAE can be applied to various hierarchical Transformers, this paper explores using Swin Transformer with a large window size and scales up to huge model size (to reach 600M parameters). Empirical results demonstrate that MixMAE can learn high-quality visual representations efficiently. Notably, MixMAE with Swin-B/W14 achieves 85.1% top-1 accuracy on ImageNet-1K by pretraining for 600 epochs. Besides, its transfer performances on the other 6 datasets show that MixMAE has better FLOPs / performance tradeoff than previous popular MIM methods.

----

## [597] Adaptive Graph Convolutional Subspace Clustering

**Authors**: *Lai Wei, Zhengwei Chen, Jun Yin, Changming Zhu, Rigui Zhou, Jin Liu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00606](https://doi.org/10.1109/CVPR52729.2023.00606)

**Abstract**:

Spectral-type subspace clustering algorithms have shown excellent performance in many subspace clustering applications. The existing spectral-type subspace clustering algorithms either focus on designing constraints for the reconstruction coefficient matrix or feature extraction methods for finding latent features of original data samples. In this paper, inspired by graph convolutional networks, we use the graph convolution technique to develop a feature extraction method and a coefficient matrix constraint simultaneously. And the graph-convolutional operator is updated iteratively and adaptively in our proposed algorithm. Hence, we call the proposed method adaptive graph convolutional subspace clustering (AGCSC). We claim that, by using AGCSC, the aggregated feature representation of original data samples is suitable for subspace clustering, and the coefficient matrix could reveal the subspace structure of the original data set more faithfully. Finally, plenty of subspace clustering experiments prove our conclusions and show that AGCSC
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
We present the codes of AGCSC and the evaluated algorithms on https://github.com/weilyshmtu/AGCSC. outperforms some related methods as well as some deep models.

----

## [598] Deep Learning of Partial Graph Matching via Differentiable Top-K

**Authors**: *Runzhong Wang, Ziao Guo, Shaofei Jiang, Xiaokang Yang, Junchi Yan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00607](https://doi.org/10.1109/CVPR52729.2023.00607)

**Abstract**:

Graph matching (GM) aims at discovering node matching between graphs, by maximizing the node-and edgewise affinities between the matched elements. As an NP-hard problem, its challenge is further pronounced in the existence of outlier nodes in both graphs which is ubiquitous in practice, especially for vision problems. However, popular affinity-maximization-based paradigms often lack a principled scheme to suppress the false matching and resort to handcrafted thresholding to dismiss the outliers. This limitation is also inherited by the neural GM solvers though they have shown superior performance in the ideal no-outlier setting. In this paper, we propose to formulate the partial GM problem as the top-k selection task with a given/estimated number of inliers k. Specifically, we devise a differentiable top-k module that enables effective gradient descent over the optimal-transport layer, which can be readily plugged into SOTA deep GM pipelines including the quadratic matching network NGMv2 as well as the linear matching network GCAN. Meanwhile, the attention-fused aggregation layers are developed to estimate k to enable automatic outlier-robust matching in the wild. Last but not least, we remake and release a new benchmark called IMC-PT-SparseGM, originating from the IMC-PT stereomatching dataset. The new benchmark involves more scale-varying graphs and partial matching instances from the real world. Experiments show that our methods outperform other partial matching schemes on popular benchmarks.

----

## [599] DynamicDet: A Unified Dynamic Architecture for Object Detection

**Authors**: *Zhihao Lin, Yongtao Wang, Jinhe Zhang, Xiaojie Chu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00608](https://doi.org/10.1109/CVPR52729.2023.00608)

**Abstract**:

Dynamic neural network is an emerging research topic in deep learning. With adaptive inference, dynamic models can achieve remarkable accuracy and computational efficiency. However, it is challenging to design a powerful dynamic detector, because of no suitable dynamic architecture and exiting criterion for object detection. To tackle these difficulties, we propose a dynamic framework for object detection, named DynamicDet. Firstly, we carefully design a dynamic architecture based on the nature of the object detection task. Then, we propose an adaptive router to analyze the multi-scale information and to decide the inference route automatically. We also present a novel optimization strategy with an exiting criterion based on the detection losses for our dynamic detectors. Last, we present a variable-speed inference strategy, which helps to realize a wide range of accuracy-speed trade-offs with only one dynamic detector. Extensive experiments conducted on the COCO benchmark demonstrate that the proposed DynamicDet achieves new state-of-the-art accuracy-speed trade-offs. For instance, with comparable accuracy, the inference speed of our dynamic detector Dy-YOLOv7-W6 surpasses YOLOv7-E6 by 12%, YOLOv7-D6 by 17%, and YOLOv7-E6E by 39%. The code is available at https://github.com/VDIGPKU/DynamicDet.

----



[Go to the previous page](CVPR-2023-list02.md)

[Go to the next page](CVPR-2023-list04.md)

[Go to the catalog section](README.md)