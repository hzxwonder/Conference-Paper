## [600] Multi-View Mesh Reconstruction with Neural Deferred Shading

**Authors**: *Markus Worchel, Rodrigo Diaz, Weiwen Hu, Oliver Schreer, Ingo Feldmann, Peter Eisert*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00609](https://doi.org/10.1109/CVPR52688.2022.00609)

**Abstract**:

We propose an analysis-by-synthesis method for fast multi-view 3D reconstruction of opaque objects with arbitrary materials and illumination. State-of-the-art methods use both neural surface representations and neural rendering. While flexible, neural surface representations are a significant bottleneck in optimization runtime. Instead, we represent surfaces as triangle meshes and build a differentiable rendering pipeline around triangle rasterization and neural shading. The renderer is used in a gradient descent optimization where both a triangle mesh and a neural shader are jointly optimized to reproduce the multi-view images. We evaluate our method on a public 3D reconstruction dataset and show that it can match the reconstruction accuracy of traditional baselines and neural approaches while surpassing them in optimization runtime. Additionally, we investigate the shader and find that it learns an interpretable representation of appearance, enabling applications such as 3D material editing.

----

## [601] StyleMesh: Style Transfer for Indoor 3D Scene Reconstructions

**Authors**: *Lukas Höllein, Justin Johnson, Matthias Nießner*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00610](https://doi.org/10.1109/CVPR52688.2022.00610)

**Abstract**:

We apply style transfer on mesh reconstructions of indoor scenes. This enables VR applications like experiencing 3D environments painted in the style of a favorite artist. Style transfer typically operates on 2D images, making stylization of a mesh challenging. When optimized over a variety of poses, stylization patterns become stretched out and inconsistent in size. On the other hand, model-based 3D style transfer methods exist that allow stylization from a sparse set of images, but they require a network at inference time. To this end, we optimize an explicit texture for the reconstructed mesh of a scene and stylize it jointly from all available input images. Our depth- and angle-aware optimization leverages surface normal and depth data of the underlying mesh to create a uniform and consistent stylization for the whole scene. Our experiments show that our method creates sharp and detailed results for the complete scene without view-dependent artifacts. Through extensive ablation studies, we show that the proposed 3D awareness enables style transfer to be applied to the 3D domain of a mesh. Our method
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://lukashoel.github.io/stylemesh/ can be used to render a stylized mesh in real-time with traditional rendering pipelines.

----

## [602] RGB-Depth Fusion GAN for Indoor Depth Completion

**Authors**: *Haowen Wang, Mingyuan Wang, Zhengping Che, Zhiyuan Xu, Xiuquan Qiao, Mengshi Qi, Feifei Feng, Jian Tang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00611](https://doi.org/10.1109/CVPR52688.2022.00611)

**Abstract**:

The raw depth image captured by the indoor depth sen-sor usually has an extensive range of missing depth values due to inherent limitations such as the inability to perceive transparent objects and limited distance range. The incomplete depth map burdens many downstream vision tasks, and a rising number of depth completion methods have been proposed to alleviate this issue. While most existing meth-ods can generate accurate dense depth maps from sparse and uniformly sampled depth maps, they are not suitable for complementing the large contiguous regions of missing depth values, which is common and critical. In this paper, we design a novel two-branch end-to-end fusion network, which takes a pair of RGB and incomplete depth images as input to predict a dense and completed depth map. The first branch employs an encoder-decoder structure to regress the local dense depth values from the raw depth map, with the help of local guidance information extracted from the RGB image. In the other branch, we propose an RGB-depth fusion GAN to transfer the RGB image to the fine-grained textured depth map. We adopt adaptive fusion modules named W-AdaIN to propagate the features across the two branches, and we append a confidence fusion head to fuse the two out-puts of the branches for the final depth map. Extensive ex-periments on NYU-Depth V2 and SUN RGB-D demonstrate that our proposed method clearly improves the depth completion performance, especially in a more realistic setting of indoor environments with the help of the pseudo depth map.

----

## [603] PlanarRecon: Realtime 3D Plane Detection and Reconstruction from Posed Monocular Videos

**Authors**: *Yiming Xie, Matheus Gadelha, Fengting Yang, Xiaowei Zhou, Huaizu Jiang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00612](https://doi.org/10.1109/CVPR52688.2022.00612)

**Abstract**:

We present PlanarRecon - a novel framework for globally coherent detection and reconstruction of 3D planes from a posed monocular video. Unlike previous works that detect planes in 2D from a single image, PlanarRecon incrementally detects planes in 3D for each video fragment, which consists of a set of key frames, from a volumetric representation of the scene using neural networks. A learning-based tracking and fusion module is designed to merge planes from previous fragments to form a coherent global plane reconstruction. Such design allows Planar-Recon to integrate observations from multiple views within each fragment and temporal information across different ones, resulting in an accurate and coherent reconstruction of the scene abstraction with low-polygonal geometry. Experiments show that the proposed approach achieves state-of-the-art performances on the ScanNet dataset while being real-time. Code is available at the project page: https://neu-vi.github.io/planarrecon/.

----

## [604] Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations

**Authors**: *Mehdi S. M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Lucic, Daniel Duckworth, Alexey Dosovitskiy, Jakob Uszkoreit, Thomas A. Funkhouser, Andrea Tagliasacchi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00613](https://doi.org/10.1109/CVPR52688.2022.00613)

**Abstract**:

A classical problem in computer vision is to infer a 3D scene representation from few images that can be used to render novel views at interactive rates. Previous work focuses on reconstructing pre-defined 3D representations, e.g. textured meshes, or implicit representations, e.g. radiance fields, and often requires input images with precise camera poses and long processing times for each novel scene. In this work, we propose the Scene Representation Transformer (SRT), a method which processes posed or unposed RGB images of a new area, infers a “set-latent scene representation ”, and synthesises novel views, all in a single feed-forward pass. To calculate the scene representation, we propose a generalization of the Vision Transformer to sets of images, enabling global information integration, and hence 3D reasoning. An efficient decoder transformer parameterizes the light field by attending into the scene representation to render novel views. Learning is supervised end-to-end by minimizing a novel-view reconstruction error. We show that this method outperforms recent baselines in terms of PSNR and speed on synthetic datasets, including a new dataset created for the paper. Further, we demonstrate that SRT scales to support interactive visualization and semantic segmentation of real-world outdoor environments using Street View imagery.

----

## [605] ShapeFormer: Transformer-based Shape Completion via Sparse Representation

**Authors**: *Xingguang Yan, Liqiang Lin, Niloy J. Mitra, Dani Lischinski, Daniel Cohen-Or, Hui Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00614](https://doi.org/10.1109/CVPR52688.2022.00614)

**Abstract**:

We present ShapeFormer, a transformer-based network that produces a distribution of object completions, conditioned on incomplete, and possibly noisy, point clouds. The resultant distribution can then be sampled to generate likely completions, each exhibiting plausible shape details while being faithful to the input. To facilitate the use of transformers for 3D, we introduce a compact 3D representation, vector quantized deep implicit function (VQDIF), that utilizes spatial sparsity to represent a close approximation of a 3D shape by a short sequence of discrete variables. Experiments demonstrate that ShapeFormer outperforms prior art for shape completion from ambiguous partial inputs in terms of both completion quality and diversity. We also show that our approach effectively handles a variety of shape types, incomplete patterns, and real-world scans.

----

## [606] GuideFormer: Transformers for Image Guided Depth Completion

**Authors**: *Kyeongha Rho, Jinsung Ha, Youngjung Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00615](https://doi.org/10.1109/CVPR52688.2022.00615)

**Abstract**:

Depth completion has been widely studied to predict a dense depth image from its sparse measurement and a single color image. However, most state-of-the-art methods rely on static convolutional neural networks (CNNs) which are not flexible enough for capturing the dynamic nature of input contexts. In this paper, we propose GuideFormer, a fully transformer-based architecture for dense depth completion. We first process sparse depth and color guidance images with separate transformer branches to extract hierarchical and complementary token representations. Each branch consists of a stack of self-attention blocks and has key design features to make our model suitable for the task. We also devise an effective token fusion method based on guided-attention mechanism. It explicitly models information flow between the two branches and captures inter-modal dependencies that cannot be obtained from depth or color image alone. These properties allow GuideFormer to enjoy various visual dependencies and recover precise depth values while preserving fine details. We evaluate GuideFormer on the KITTI dataset containing realworld driving scenes and provide extensive ablation studies. Experimental results demonstrate that our approach significantly outperforms the state-of-the-art methods.

----

## [607] Improving neural implicit surfaces geometry with patch warping

**Authors**: *François Darmon, Bénédicte Bascle, Jean-Clément Devaux, Pascal Monasse, Mathieu Aubry*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00616](https://doi.org/10.1109/CVPR52688.2022.00616)

**Abstract**:

Neural implicit surfaces have become an important technique for multi-view 3D reconstruction but their accuracy remains limited. In this paper, we argue that this comes from the difficulty to learn and render high frequency textures with neural networks. We thus propose to add to the standard neural rendering optimization a direct photo-consistency term across the different views. Intuitively, we optimize the implicit geometry so that it warps views on each other in a consistent way. We demonstrate that two elements are key to the success of such an approach: (i) warping entire patches, using the predicted occupancy and normals of the 3D points along each ray, and measuring their similarity with a robust structural similarity (SSIM); (ii) handling visibility and occlusion in such a way that incorrect warps are not given too much importance while encouraging a reconstruction as complete as possible. We evaluate our approach, dubbed NeuralWarp, on the standard DTU and EPFL benchmarks and show it outperforms state of the art unsupervised implicit surfaces reconstructions by over 20% on both datasets. Our code is available at https://github.com/fdarmon/NeuralWarp

----

## [608] Critical Regularizations for Neural Surface Reconstruction in the Wild

**Authors**: *Jingyang Zhang, Yao Yao, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00617](https://doi.org/10.1109/CVPR52688.2022.00617)

**Abstract**:

Neural implicit functions have recently shown promising results on surface reconstructions from multiple views. However, current methods still suffer from excessive time complexity and poor robustness when reconstructing unbounded or complex scenes. In this paper, we present RegSDF, which shows that proper point cloud supervisions and geometry regularizations are sufficient to produce high-quality and robust reconstruction results. Specifically, RegSDF takes an additional oriented point cloud as input, and optimizes a signed distance field and a surface light field within a differentiable rendering framework. We also introduce the two critical regularizations for this optimization. The first one is the Hessian regularization that smoothly diffuses the signed distance values to the entire distance field given noisy and incomplete input. And the second one is the minimal surface regularization that compactly interpolates and extrapolates the missing geometry. Extensive experiments are conducted on DTU, Blended-MVS, and Tanks and Temples datasets. Compared with recent neural surface reconstruction approaches, RegSDF is able to reconstruct surfaces with fine details even for open scenes with complex topologies and unstructured camera trajectories.

----

## [609] Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction

**Authors**: *Christiane Sommer, Lu Sang, David Schubert, Daniel Cremers*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00618](https://doi.org/10.1109/CVPR52688.2022.00618)

**Abstract**:

We present Gradient-SDF, a novel representation for 3D geometry that combines the advantages of implict and explicit representations. By storing at every voxel both the signed distance field as well as its gradient vector field, we enhance the capability of implicit representations with approaches originally formulated for explicit surfaces. As concrete examples, we show that (1) the Gradient-SDF allows us to perform direct SDF tracking from depth images, using efficient storage schemes like hash maps, and that (2) the Gradient-SDF representation enables us to perform photometric bundle adjustment directly in a voxel representation (without transforming into a point cloud or mesh), naturally a fully implicit optimization of geometry and camera poses and easy geometry upsampling. Experimental results confirm that this leads to significantly sharper reconstructions. Since the overall SDF voxel structure is still respected, the proposed Gradient-SDF is equally suited for (GPU) parallelization as related approaches.

----

## [610] Neural RGB-D Surface Reconstruction

**Authors**: *Dejan Azinovic, Ricardo Martin-Brualla, Dan B. Goldman, Matthias Nießner, Justus Thies*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00619](https://doi.org/10.1109/CVPR52688.2022.00619)

**Abstract**:

Obtaining high-quality 3D reconstructions of room-scale scenes is of paramount importance for upcoming applications in AR or VR. These range from mixed reality applications for teleconferencing, virtual measuring, virtual room planing, to robotic applications. While current volume-based view synthesis methods that use neural radiance fields (NeRFs) show promising results in reproducing the appearance of an object or scene, they do not reconstruct an actual surface. The volumetric representation of the surface based on densities leads to artifacts when a surface is extracted using Marching Cubes, since during optimization, densities are accumulated along the ray and are not used at a single sample point in isolation. Instead of this volumetric representation of the surface, we propose to represent the surface using an implicit function (truncated signed distance function). We show how to incorporate this representation in the NeRF framework, and extend it to use depth measurements from a commodity RGB-D sensor, such as a Kinect. In addition, we propose a pose and camera re-finement technique which improves the overall reconstruction quality. In contrast to concurrent work on integrating depth priors in NeRF which concentrates on novel view synthesis, our approach is able to reconstruct high-quality, metrical 3D reconstructions.

----

## [611] POCO: Point Convolution for Surface Reconstruction

**Authors**: *Alexandre Boulch, Renaud Marlet*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00620](https://doi.org/10.1109/CVPR52688.2022.00620)

**Abstract**:

Implicit neural networks have been successfully used for surface reconstruction from point clouds. However, many of them face scalability issues as they encode the isosurface function of a whole object or scene into a single latent vector. To overcome this limitation, a few approaches infer latent vectors on a coarse regular 3D grid or on 3D patches, and interpolate them to answer occupancy queries. In doing so, they lose the direct connection with the input points sampled on the surface of objects, and they attach information uniformly in space rather than where it matters the most, i.e., near the surface. Besides, relying on fixed patch sizes may require discretization tuning. To address these issues, we propose to use point cloud convolutions and compute latent vectors at each input point. We then perform a learning-based interpolation on nearest neighbors using inferred weights. Experiments on both object and scene datasets show that our approach significantly outperforms other methods on most classical metrics, producing finer details and better reconstructing thinner volumes. The code is available at https://github.com/valeoai/POCO.

----

## [612] Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors

**Authors**: *Baorui Ma, Yu-Shen Liu, Zhizhong Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00621](https://doi.org/10.1109/CVPR52688.2022.00621)

**Abstract**:

It is an important task to reconstruct surfaces from 3D point clouds. Current methods are able to reconstruct surfaces by learning Signed Distance Functions (SDFs) from single point clouds without ground truth signed distances or point normals. However, they require the point clouds to be dense, which dramatically limits their performance in real applications. To resolve this issue, we propose to reconstruct highly accurate surfaces from sparse point clouds with an on-surface prior. We train a neural network to learn SDFs via projecting queries onto the surface represented by the sparse point cloud. Our key idea is to infer signed distances by pushing both the query projections to be on the surface and the projection distance to be the minimum. To achieve this, we train a neural network to capture the on-surface prior to determine whether a point is on a sparse point cloud or not, and then leverage it as a differentiable function to learn SDFs from unseen sparse point cloud. Our method can learn SDFs from a single s parse point cloud without ground truth signed distances or point normals. Our numerical evaluation under widely used benchmarks demonstrates that our method achieves state-of-the-art reconstruction accuracy, especially for sparse point clouds. Code and data are available at https://github.com/mabaorui/OnSurfacePrior.

----

## [613] Surface Reconstruction from Point Clouds by Learning Predictive Context Priors

**Authors**: *Baorui Ma, Yu-Shen Liu, Matthias Zwicker, Zhizhong Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00622](https://doi.org/10.1109/CVPR52688.2022.00622)

**Abstract**:

Surface reconstruction from point clouds is vital for 3D computer vision. State-of-the-art methods leverage large datasets to first learn local context priors that are represented as neural network-based signed distance functions (SDFs) with some parameters encoding the local contexts. To reconstruct a surface at a specific query location at inference time, these methods then match the local reconstruction target by searching for the best match in the local prior space (by optimizing the parameters encoding the local context) at the given query location. However, this requires the local context prior to generalize to a wide variety of unseen target regions, which is hard to achieve. To resolve this issue, we introduce Predictive Context Priors by learning Predictive Queries for each specific point cloud at inference time. Specifically, we first train a local context prior using a large point cloud dataset similar to previous techniques. For surface reconstruction at inference time, however, we specialize the local context prior into our Predictive Context Prior by learning Predictive Queries, which predict adjusted spatial query locations as displacements of the original locations. This leads to a global SDF that fits the specific point cloud the best. Intuitively, the query prediction enables us to flexibly search the learned local context prior over the entire prior space, rather than being restricted to the fixed query locations, and this improves the generalizability. Our method does not require ground truth signed distances, normals, or any additional procedure of signed distance fusion across overlapping regions. Our experimental results in surface reconstruction for single shapes or complex scenes show significant improvements over the state-of-the-art under widely used benchmark-s. Code and data are available at https://github.com/mabaorui/PredictableContextPrior.

----

## [614] IDEA-Net: Dynamic 3D Point Cloud Interpolation via Deep Embedding Alignment

**Authors**: *Yiming Zeng, Yue Qian, Qijian Zhang, Junhui Hou, Yixuan Yuan, Ying He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00623](https://doi.org/10.1109/CVPR52688.2022.00623)

**Abstract**:

This paper investigates the problem of temporally interpolating dynamic 3D point clouds with large non-rigid deformation. We formulate the problem as estimation of point-wise trajectories (i.e., smooth curves) and further reason that temporal irregularity and under-sampling are two major challenges. To tackle the challenges, we propose IDEA-Net, an end-to-end deep learning framework, which disentangles the problem under the assistance of the explicitly learned temporal consistency. Specifically, we propose a temporal consistency learning module to align two consecutive point cloud frames point-wisely, based on which we can employ linear interpolation to obtain coarse trajectories/in-between frames. To compensate the high-order nonlinear components of trajectories, we apply aligned feature embeddings that encode local geometry properties to regress point-wise increments, which are combined with the coarse estimations. We demonstrate the effectiveness of our method on various point cloud sequences and observe large improvement over state-of-the-art methods both quantitatively and visually. Our framework can bring benefits to 3D motion data acquisition. The source code is publicly available at https://github.com/ZENGYIMING-EAMON/IDEANet.git.

----

## [615] Deterministic Point Cloud Registration via Novel Transformation Decomposition

**Authors**: *Wen Chen, Haoang Li, Qiang Nie, Yun-Hui Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00624](https://doi.org/10.1109/CVPR52688.2022.00624)

**Abstract**:

Given a set of putative 3D-3D point correspondences, we aim to remove outliers and estimate rigid transformation with 6 degrees of freedom (DOF). Simultaneously estimating these 6 DOF is time-consuming due to high-dimensional parameter space. To solve this problem, it is common to decompose 6 DOF, i.e. independently compute 3-DOF rotation and 3-DOF translation. However, high non-linearity of 3-DOF rotation still limits the algorithm efficiency, especially when the number of correspondences is large. In contrast, we propose to decompose 6 DOF into 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(2+1)$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(1+2)\ DOF$</tex>
. Specifically, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(2+1)DOF$</tex>
 represent 2-DOF rotation axis and 1-DOF displacement along this rotation axis. 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(1+2)\ DOF$</tex>
 indicate 1-DOF rotation angle and 2-DOF displacement orthogonal to the above rotation axis. To compute these DOF, we design a novel two-stage strategy based on inlier set maximization. By leveraging branch and bound, we first search for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(2+1)\ DOF$</tex>
, and then the remaining 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(1+2)\ DOF$</tex>
. Thanks to the proposed transformation decomposition and two-stage search strategy, our method is deterministic and leads to low computational complexity. We extensively compare our method with state-of-the-art approaches. Our method is more accurate and robust than the approaches that provide similar efficiency to ours. Our method is more efficient than the approaches whose accuracy and robustness are comparable to ours.

----

## [616] Global-Aware Registration of Less-Overlap RGB-D Scans

**Authors**: *Che Sun, Yunde Jia, Yi Guo, Yuwei Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00625](https://doi.org/10.1109/CVPR52688.2022.00625)

**Abstract**:

We propose a novel method of registering less-overlap RGB-D scans. Our method learns global information of a scene to construct a panorama, and aligns RGB-D scans to the panorama to perform registration. Different from existing methods that use local feature points to register less-overlap RGB-D scans and mismatch too much, we use global information to guide the registration, thereby allevi-ating the mismatching problem by preserving global consis-tency of alignments. To this end, we build a scene inference network to construct the panorama representing global in-formation. We introduce a reinforcement learning strategy to iteratively align RGB-D scans with the panorama and re-fine the panorama representation, which reduces the noise of global information and preserves global consistency of both geometric and photometric alignments. Experimental results on benchmark datasets including SUNCG, Matterport, and ScanNet show the superiority of our method.

----

## [617] Finding Good Configurations of Planar Primitives in Unorganized Point Clouds

**Authors**: *Mulin Yu, Florent Lafarge*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00626](https://doi.org/10.1109/CVPR52688.2022.00626)

**Abstract**:

We present an algorithm for detecting planar primitives from unorganized 3D point clouds. Departing from an initial configuration, the algorithm refines both the continuous plane parameters and the discrete assignment of input points to them by seeking high fidelity, high simplicity and high completeness. Our key contribution relies upon the design of an exploration mechanism guided by a multi-objective energy function. The transitions within the large solution space are handled by five geometric operators that create, remove and modify primitives. We demonstrate the potential of our method on a variety of scenes, from organic shapes to man-made objects, and sensors, from multiview stereo to laser. We show its efficacy with respect to existing primitive fitting approaches and illustrate its applicative interest in compact mesh reconstruction, when combined with a plane assembly method.

----

## [618] Self-Supervised Global-Local Structure Modeling for Point Cloud Domain Adaptation with Reliable Voted Pseudo Labels

**Authors**: *Hehe Fan, Xiaojun Chang, Wanyue Zhang, Yi Cheng, Ying Sun, Mohan S. Kankanhalli*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00627](https://doi.org/10.1109/CVPR52688.2022.00627)

**Abstract**:

In this paper, we propose an unsupervised domain adaptation method for deep point cloud representation learning. To model the internal structures in target point clouds, we first propose to learn the global representations of unla-beled data by scaling up or down point clouds and then predicting the scales. Second, to capture the local structure in a self-supervised manner, we propose to project a 3D local area onto a 2D plane and then learn to reconstruct the squeezed region. Moreover, to effectively transfer the knowledge from source domain, we propose to vote pseudo labels for target samples based on the labels of their nearest source neighbors in the shared feature space. To avoid the noise caused by incorrect pseudo labels, we only select re-liable target samples, whose voting consistencies are high enough, for enhancing adaptation. The voting method is able to adaptively select more and more target samples during training, which in return facilitates adaptation because the amount of labeled target data increases. Experiments on PointDA (ModelNet-10, ShapeNet-10 and ScanNet-10) and Sim-to-Real (ModelNet-11, ScanObjectNN-11, ShapeNet-9 and ScanObjectNN-9) demonstrate the effectiveness of our method.

----

## [619] AziNorm: Exploiting the Radial Symmetry of Point Cloud for Azimuth-Normalized 3D Perception

**Authors**: *Shaoyu Chen, Xinggang Wang, Tianheng Cheng, Wenqiang Zhang, Qian Zhang, Chang Huang, Wenyu Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00628](https://doi.org/10.1109/CVPR52688.2022.00628)

**Abstract**:

Studying the inherent symmetry of data is of great importance in machine learning. Point cloud, the most important data format for 3D environmental perception, is naturally endowed with strong radial symmetry. In this work, we exploit this radial symmetry via a divide-and-conquer strategy to boost 3D perception performance and ease optimization. We propose Azimuth Normalization (AziNorm), which normalizes the point clouds along the radial direction and eliminates the variability brought by the difference of azimuth. AziNorm can be flexibly incorporated into most LiDAR-based perception methods. To validate its effectiveness and generalization ability, we apply AziNorm in both object detection and semantic segmentation. For detection, we integrate AziNorm into two representative detection methods, the one-stage SECOND detector and the state-of-the-art two-stage PV-RCNN detector. Experiments on Waymo Open Dataset demonstrate that AziNorm improves SECOND and PV-RCNN by 7.03 mAPH and 3.01 mAPH respectively. For segmentation, we integrate AziNorm into KPConv. On SemanticKitti dataset, AziNorm improves KPConv by 1.6/1.1 mIoU on val/test set. Besides, AziNorm remarkably improves data efficiency and accelerates convergence, reducing the requirement of data amounts or training epochs by an order of magnitude. SECOND w/ AziNorm can significantly outperform fully trained vanilla SECOND, even trained with only 10% data or 10% epochs. Code and models are available at https://github.com/hustvl/AziNorm.

----

## [620] WarpingGAN: Warping Multiple Uniform Priors for Adversarial 3D Point Cloud Generation

**Authors**: *Yingzhi Tang, Yue Qian, Qijian Zhang, Yiming Zeng, Junhui Hou, Xuefei Zhe*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00629](https://doi.org/10.1109/CVPR52688.2022.00629)

**Abstract**:

We propose WarpingGAN, an effective and efficient 3D point cloud generation network. Unlike existing methods that generate point clouds by directly learning the mapping functions between latent codes and 3D shapes, Warping-GAN learns a unified local-warping function to warp multiple identical pre-defined priors (i.e., sets of points uniformly distributed on regular 3D grids) into 3D shapes driven by local structure-aware semantics. In addition, we also in-geniously utilize the principle of the discriminator and tai-lor a stitching loss to eliminate the gaps between different partitions of a generated shape corresponding to different priors for boosting quality. Owing to the novel gen-erating mechanism, WarpingGAN, a single lightweight network after one-time training, is capable of efficiently gen-erating uniformly distributed 3D point clouds with various resolutions. Extensive experimental results demonstrate the superiority of our WarpingGAN over state-of-the-art methods in terms of quantitative metrics, visual quality, and efficiency. The source code is publicly available at https://github.com/yztang4/WarpingGAN.git.

----

## [621] Forward Propagation, Backward Regression, and Pose Association for Hand Tracking in the Wild

**Authors**: *Mingzhen Huang, Supreeth Narasimhaswamy, Saif Vazir, Haibin Ling, Minh Hoai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00630](https://doi.org/10.1109/CVPR52688.2022.00630)

**Abstract**:

We propose HandLer, a novel convolutional architecture that can jointly detect and track hands online in unconstrained videos. HandLer is based on Cascade-RCNN with additional three novel stages. The first stage is Forward Propagation, where the features from frame t −1 are propagated to frame t based on previously detected hands and their estimated motion. The second stage is the Detection and Backward Regression, which uses outputs from the forward propagation to detect hands for frame t and their relative offset in frame t −1. The third stage uses an off-the-shelf human pose method to link any fragmented hand tracklets. We train the forward propagation and backward regression and detection stages end-to-end together with the other Cascade-RCNN components. To train and evaluate HandLer, we also contribute YouTube-Hand, the first challenging large-scale dataset of unconstrained videos annotated with hand locations and their trajectories. Experiments on this dataset and other benchmarks show that HandLer outperforms the existing state-of-the-art tracking algorithms by a large margin. Code and data are available at https://vision.cs.stonybrook.edu/~mingzhen/handler/.

----

## [622] Neural MoCon: Neural Motion Control for Physically Plausible Human Motion Capture

**Authors**: *Buzhen Huang, Liang Pan, Yuan Yang, Jingyi Ju, Yangang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00631](https://doi.org/10.1109/CVPR52688.2022.00631)

**Abstract**:

Due to the visual ambiguity, purely kinematic formulations on monocular human motion capture are often physically incorrect, biomechanically implausible, and can not reconstruct accurate interactions. In this work, we focus on exploiting the high-precision and non-differentiable physics simulator to incorporate dynamical constraints in motion capture. Our key-idea is to use real physical supervisions to train a target pose distribution prior for sampling-based motion control to capture physically plausible human motion. To obtain accurate reference motion with terrain interactions for the sampling, we first introduce an interaction constraint based on SDF (Signed Distance Field) to enforce appropriate ground contact modeling. We then design a novel two-branch decoder to avoid stochastic error from pseudo ground-truth and train a distribution prior with the non-differentiable physics simulator. Finally, we regress the sampling distribution from the current state of the physical character with the trained prior and sample satisfied target poses to track the estimated reference motion. Qualitative and quantitative results show that we can obtain physically plausible human motion with complex terrain interactions, human shape variations, and diverse behaviors. More information can be found ar https://www.yangangwang.com/papers/HBZ-NM-2022-03.html

----

## [623] MotionAug: Augmentation with Physical Correction for Human Motion Prediction

**Authors**: *Takahiro Maeda, Norimichi Ukita*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00632](https://doi.org/10.1109/CVPR52688.2022.00632)

**Abstract**:

This paper presents a motion data augmentation scheme incorporating motion synthesis encouraging diversity and motion correction imposing physical plausibility. This motion synthesis consists of our modified Variational AutoEncoder (VAE) and Inverse Kinematics (IK). In this VAE, our proposed sampling-near-samples method generates various valid motions even with insufficient training motion data. Our IK-based motion synthesis method allows us to generate a variety of motions semi-automatically. Since these two schemes generate unrealistic artifacts in the synthesized motions, our motion correction rectifies them. This motion correction scheme consists of imitation learning with physics simulation and subsequent motion debiasing. For this imitation learning, we propose the PD-residual force that significantly accelerates the training process. Furthermore, our motion debiasing successfully offsets the motion bias induced by imitation learning to maximize the effect of augmentation. As a result, our method outperforms previous noise-based motion augmentation methods by a large margin on both Recurrent Neural Network-based and Graph Convolutional Network-based human motion prediction models. The code is available at https://github.com/meaten/MotionAug.

----

## [624] Progressively Generating Better Initial Guesses Towards Next Stages for High-Quality Human Motion Prediction

**Authors**: *Tiezheng Ma, Yongwei Nie, Chengjiang Long, Qing Zhang, Guiqing Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00633](https://doi.org/10.1109/CVPR52688.2022.00633)

**Abstract**:

This paper presents a high-quality human motion pre-diction method that accurately predicts future human poses given observed ones. Our method is based on the observation that a good “initial guess” of the future poses is very helpful in improving the forecasting accuracy. This mo-tivates us to propose a novel two-stage prediction frame-work, including an init-prediction network that just computes the good guess and then a formal-prediction network that predicts the target future poses based on the guess. More importantly, we extend this idea further and design a multi-stage prediction framework where each stage pre-dicts initial guess for the next stage, which brings more performance gain. To fulfill the prediction task at each stage, we propose a network comprising Spatial Dense Graph Convolutional Networks (S-DGCN) and Temporal Dense Graph Convolutional Networks (T-DGCN). Alternatively executing the two networks helps extract spatiotem-poral features over the global receptive field of the whole pose sequence. All the above design choices cooperating together make our method outperform previous approaches by large margins: 6%-7% on Human3.6M, 5%-10% on CMU-MoCap, and 13%-16% on 3DPW. Code is available at https://github.com/705062791/PGBIG.

----

## [625] Spatio-Temporal Gating-Adjacency GCN for Human Motion Prediction

**Authors**: *Chongyang Zhong, Lei Hu, Zihao Zhang, Yongjing Ye, Shihong Xia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00634](https://doi.org/10.1109/CVPR52688.2022.00634)

**Abstract**:

Predicting future motion based on historical motion sequence is a fundamental problem in computer vision, and it has wide applications in autonomous driving and robotics. Some recent works have shown that Graph Convolutional Networks(GCN) are instrumental in modeling the relationship between different joints. However, considering the variants and diverse action types in human motion data, the cross-dependency of the spatio-temporal relationships will be difficult to depict due to the decoupled modeling strategy, which may also exacerbate the problem of insufficient generalization. Therefore, we propose the Spatio-Temporal Gating-Adjacency GCN(GAGCN) to learn the complex spatio-temporal dependencies over diverse action types. Specifically, we adopt gating networks to enhance the generalization of GCN via the trainable adaptive adjacency matrix obtained by blending the candidate spatio-temporal adjacency matrices. Moreover, GAGCN addresses the cross-dependency of space and time by balancing the weights of spatio-temporal modeling and fusing the decoupled spatio-temporal features. Extensive experiments on Human 3.6M, AMASS, and 3DPW demonstrate that GAGCN achieves state-of-the-art performance in both short-term and long-term predictions.

----

## [626] Motron: Multimodal Probabilistic Human Motion Forecasting

**Authors**: *Tim Salzmann, Marco Pavone, Markus Ryll*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00635](https://doi.org/10.1109/CVPR52688.2022.00635)

**Abstract**:

Autonomous systems and humans are increasingly sharing the same space. Robots work side by side or even hand in hand with humans to balance each other's limitations. Such cooperative interactions are ever more sophisticated. Thus, the ability to reason not just about a human's center of gravity position, but also its granular motion is an important prerequisite for human-robot interaction. Though, many algorithms ignore the multimodal nature of humans or neglect uncertainty in their motion forecasts. We present Motron, a multimodal, probabilistic, graph-structured model, that captures human's multimodality using probabilistic methods while being able to output deterministic maximum-likelihood motions and corresponding confidence values for each mode. Our model aims to be tightly integrated with the robotic planning-control-interaction loop; outputting physically feasible human motions and being computationally efficient. We demonstrate the performance of our model on several challenging real-world motion forecasting datasets, outperforming a wide array of generative/variational methods while providing state-of-the-art single-output motions if required. Both using significantly less computational power than state-of-the art algorithms.

----

## [627] Human Trajectory Prediction with Momentary Observation

**Authors**: *Jianhua Sun, Yuxuan Li, Liang Chai, Haoshu Fang, Yong-Lu Li, Cewu Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00636](https://doi.org/10.1109/CVPR52688.2022.00636)

**Abstract**:

Human trajectory prediction task aims to analyze human future movements given their past status, which is a crucial step for many autonomous systems such as self-driving cars and social robots. In real-world scenarios, it is unlikely to obtain sufficiently long observations at all times for prediction, considering inevitable factors such as tracking losses and sudden events. However, the problem of trajectory pre-diction with limited observations has not drawn much at-tention in previous work. In this paper, we study a task named momentary trajectory prediction, which reduces the observed history from a long time sequence to an extreme situation of two frames, one frame for social and scene contexts and both frames for the velocity of agents. We perform a rigorous study of existing state-of-the-art approaches in this challenging setting on two widely used benchmarks. We further propose a unified feature extractor, along with a novel pre-training mechanism, to capture effective infor-mation within the momentary observation. Our extractor can be adopted in existing prediction models and substan-tially boost their performance of momentary trajectory pre-diction. We hope our work will pave the way for more re-sponsive, precise and robust prediction approaches, an important step toward real-world autonomous systems.

----

## [628] Non-Probability Sampling Network for Stochastic Human Trajectory Prediction

**Authors**: *Inhwan Bae, Jin-Hwi Park, Hae-Gon Jeon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00637](https://doi.org/10.1109/CVPR52688.2022.00637)

**Abstract**:

Capturing multimodal natures is essential for stochastic pedestrian trajectory prediction, to infer a finite set of future trajectories. The inferred trajectories are based on observation paths and the latent vectors of potential decisions of pedestrians in the inference step. However, stochastic approaches provide varying results for the same data and parameter settings, due to the random sampling of the latent vector. In this paper, we analyze the problem by reconstructing and comparing probabilistic distributions from prediction samples and socially-acceptable paths, respectively. Through this analysis, we observe that the inferences of all stochastic models are biased toward the random sampling, and fail to generate a set of realistic paths from finite samples. The problem cannot be resolved unless an infinite number of samples is available, which is infeasible in practice. We introduce that the Quasi-Monte Carlo (QMC) method, ensuring uniform coverage on the sampling space, as an alternative to the conventional random sampling. With the same finite number of samples, the QMC improves all the multimodal prediction results. We take an additional step ahead by incorporating a learnable sampling network into the existing networks for trajectory prediction. For this purpose, we propose the Non-Probability Sampling Network (NPSN), a very small network (~5K parameters) that generates purposive sample sequences using the past paths of pedestrians and their social interactions. Extensive experiments confirm that NPSN can significantly improve both the prediction accuracy (up to 60%) and reliability of the public pedestrian trajectory prediction benchmark. Code is publicly available at https://github.com/inhwanbae/NPSN.

----

## [629] Remember Intentions: Retrospective-Memory-based Trajectory Prediction

**Authors**: *Chenxin Xu, Weibo Mao, Wenjun Zhang, Siheng Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00638](https://doi.org/10.1109/CVPR52688.2022.00638)

**Abstract**:

To realize trajectory prediction, most previous methods adopt the parameter-based approach, which encodes all the seen past-future instance pairs into model parameters. However, in this way, the model parameters come from all seen instances, which means a huge amount of irrelevant seen instances might also involve in predicting the current situation, disturbing the performance. To provide a more explicit link between the current situation and the seen instances, we imitate the mechanism of retrospective memory in neuropsychology and propose MemoNet, an instance-based approach that predicts the movement intentions of agents by looking for similar scenarios in the training data. In MemoNet, we design a pair of memory banks to explicitly store representative instances in the training set, acting as prefrontal cortex in the neural system, and a trainable memory addresser to adaptively search a current situation with similar instances in the memory bank, acting like basal ganglia. During prediction, MemoNet recalls previous memory by using the memory addresser to index related instances in the memory bank. We further propose a two-step trajectory prediction system, where the first step is to leverage MemoNet to predict the destination and the second step is to fulfill the whole trajectory according to the predicted destinations. Experiments show that the proposed MemoNet improves the FDE by 20.3%/10.2%/28.3%from the previous best method on SDD/ETH-UCY/NBA datasets. Experiments also show that our MemoNet has the ability to trace back to specific instances during prediction, promoting more interpretability.

----

## [630] GroupNet: Multiscale Hypergraph Neural Networks for Trajectory Prediction with Relational Reasoning

**Authors**: *Chenxin Xu, Maosen Li, Zhenyang Ni, Ya Zhang, Siheng Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00639](https://doi.org/10.1109/CVPR52688.2022.00639)

**Abstract**:

Demystifying the interactions among multiple agents from their past trajectories is fundamental to precise and interpretable trajectory prediction. However, previous works only consider pair-wise interactions with limited relational reasoning. To promote more comprehensive interaction modeling for relational reasoning, we propose GroupNet, a multiscale hypergraph neural network, which is novel in terms of both interaction capturing and representation learning. From the aspect of interaction capturing, we propose a trainable multiscale hypergraph to capture both pair-wise and group-wise interactions at multiple group sizes. From the aspect of interaction representation learning, we propose a three-element format that can be learnt end-to-end and explicitly reason some relational factors including the interaction strength and category. We apply GroupNet into both CVAE-based prediction system and previous state-of-the-art prediction systems for predicting socially plausible trajectories with relational reasoning. To validate the ability of relational reasoning, we experiment with synthetic physics simulations to reflect the ability to capture group behaviors, reason interaction strength and interaction category. To validate the effectiveness of prediction, we conduct extensive experiments on three real-world trajectory prediction datasets, including NBA, SDD and ETH-UCY; and we show that with GroupNet, the CVAE-based prediction system outperforms state-of-the-art methods. We also show that adding GroupNet will further improve the performance of previous state-of-the-art prediction systems.

----

## [631] Learning Pixel Trajectories with Multiscale Contrastive Random Walks

**Authors**: *Zhangxing Bian, Allan Jabri, Alexei A. Efros, Andrew Owens*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00640](https://doi.org/10.1109/CVPR52688.2022.00640)

**Abstract**:

A range of video modeling tasks, from optical flow to multiple object tracking, share the same fundamental challenge: establishing space-time correspondence. Yet, approaches that dominate each space differ. We take a step to-wards bridging this gap by extending the recent contrastive random walk formulation to much denser, pixel-level spacetime graphs. The main contribution is introducing hierarchy into the search problem by computing the transition matrix between two frames in a coarse-to-fine manner, forming a multiscale contrastive random walk when ex-tended in time. This establishes a unified technique for self-supervised learning of optical flow, keypoint tracking, and video object segmentation. Experiments demonstrate that, for each of these tasks, the unified model achieves performance competitive with strong self-supervised approaches specific to that task.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page at https://jasonbian97.github.io/flowwalk

----

## [632] Adaptive Trajectory Prediction via Transferable GNN

**Authors**: *Yi Xu, Lichen Wang, Yizhou Wang, Yun Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00641](https://doi.org/10.1109/CVPR52688.2022.00641)

**Abstract**:

Pedestrian trajectory prediction is an essential component in a wide range of AI applications such as autonomous driving and robotics. Existing methods usually assume the training and testing motions follow the same pattern while ignoring the potential distribution differences (e.g., shopping mall and street). This issue results in inevitable performance decrease. To address this issue, we propose a novel Transferable Graph Neural Network (TGNN) frame-work, which jointly conducts trajectory prediction as well as domain alignment in a unified framework. Specifically, a domain-invariant GNN is proposed to explore the structural motion knowledge where the domain-specific knowledge is reduced. Moreover, an attention-based adaptive knowledge learning module is further proposed to explore fine-grained individual-level feature representations for knowledge transfer. By this way, disparities across different trajectory domains will be better alleviated. More challenging while practical trajectory prediction experiments are designed, and the experimental results verify the superior performance of our proposed model. To the best of our knowledge, our work is the pioneer which fills the gap in benchmarks and techniques for practical pedestrian trajectory prediction across different domains.

----

## [633] Neural Prior for Trajectory Estimation

**Authors**: *Chaoyang Wang, Xueqian Li, Jhony Kaesemodel Pontes, Simon Lucey*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00642](https://doi.org/10.1109/CVPR52688.2022.00642)

**Abstract**:

Neural priors are a promising direction to capture low-level vision statistics without relying on handcrafted regularizers. Recent works have successfully shown the use of neural architecture biases to implicitly regularize image denoising, super-resolution, inpainting, synthesis, scene flow, among others. They do not rely on large-scale datasets to capture prior statistics and thus generalize well to out-of-the-distribution data. Inspired by such advances, we investigate neural priors for trajectory representation. Traditionally, trajectories have been represented by a set of handcrafted bases that have limited expressibility. Here, we propose a neural trajectory prior to capture continuous spatio-temporal information without the need for offline data. We demonstrate how our proposed objective is optimized during runtime to estimate trajectories for two important tasks: Non-Rigid Structure from Motion (NRSfM) and lidar scene flow integration for self-driving scenes. Our results are competitive to many state-of-the-art methods for both tasks.

----

## [634] M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction

**Authors**: *Qiao Sun, Xin Huang, Junru Gu, Brian C. Williams, Hang Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00643](https://doi.org/10.1109/CVPR52688.2022.00643)

**Abstract**:

Predicting future motions of road participants is an important task for driving autonomously in urban scenes. Existing models excel at predicting marginal trajectories for single agents, yet it remains an open question to jointly predict scene compliant trajectories over multiple agents. The challenge is due to exponentially increasing prediction space as a function of the number of agents. In this work, we exploit the underlying relations between interacting agents and decouple the joint prediction problem into marginal prediction problems. Our proposed approach M2I first classifies interacting agents as pairs of influencers and reactors, and then leverages a marginal prediction model and a conditional prediction model to predict trajectories for the influencers and reactors, respectively. The predictions from interacting agents are combined and selected according to their joint likelihoods. Experiments show that our simple but effective approach achieves state-of-the-art performance on the Waymo Open Motion Dataset interactive prediction benchmark.

----

## [635] How many Observations are Enough? Knowledge Distillation for Trajectory Forecasting

**Authors**: *Alessio Monti, Angelo Porrello, Simone Calderara, Pasquale Coscia, Lamberto Ballan, Rita Cucchiara*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00644](https://doi.org/10.1109/CVPR52688.2022.00644)

**Abstract**:

Accurate prediction of future human positions is an essential task for modern video-surveillance systems. Current state-of-the-art models usually rely on a “history” of past tracked locations (e.g., 3 to 5 seconds) to predict a plausible sequence of future locations (e.g., up to the next 5 seconds). We feel that this common schema neglects critical traits of realistic applications: as the collection of input trajectories involves machine perception (i.e., detection and tracking), incorrect detection and fragmentation errors may accumulate in crowded scenes, leading to tracking drifts. On this account, the model would be fed with corrupted and noisy input data, thus fatally affecting its prediction performance. In this regard, we focus on delivering accurate predictions when only few input observations are used, thus potentially lowering the risks associated with automatic perception. To this end, we conceive a novel distillation strategy that allows a knowledge transfer from a teacher network to a student one, the latter fed with fewer observations (just two ones). We show that a properly defined teacher super-vision allows a student network to perform comparably to state-of-the-art approaches that demand more observations. Besides, extensive experiments on common trajectory forecasting datasets highlight that our student network better generalizes to unseen scenarios.

----

## [636] ATPFL: Automatic Trajectory Prediction Model Design under Federated Learning Framework

**Authors**: *Chunnan Wang, Xiang Chen, Junzhe Wang, Hongzhi Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00645](https://doi.org/10.1109/CVPR52688.2022.00645)

**Abstract**:

Although the Trajectory Prediction (TP) model has achieved great success in computer vision and robotics fields, its architecture and training scheme design rely on heavy manual work and domain knowledge, which is not friendly to common users. Besides, the existing works ignore Federated Learning (FL) scenarios, failing to make full use of distributed multi-source datasets with rich actual scenes to learn more a powerful TP model. In this paper, we make up for the above defects and propose ATPFL to help users federate multi-source trajectory datasets to automatically design and train a powerful TP model. In ATPFL, we build an effective TP search space by analyzing and summarizing the existing works. Then, based on the characters of this search space, we design a relation-sequence-aware search strategy, realizing the automatic design of the TP model. Finally, we find appropriate federated training methods to respectively support the TP model search and final model training under the FL framework, ensuring both the search efficiency and the final model performance. Extensive experimental results show that ATPFL can help users gain well-performed TP models, achieving better results than the existing TP models trained on the single-source dataset.

----

## [637] Whose Track Is It Anyway? Improving Robustness to Tracking Errors with Affinity-based Trajectory Prediction

**Authors**: *Xinshuo Weng, Boris Ivanovic, Kris Kitani, Marco Pavone*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00646](https://doi.org/10.1109/CVPR52688.2022.00646)

**Abstract**:

Multi-agent trajectory prediction is critical for planning and decision-making in human-interactive autonomous systems, such as self-driving cars. However, most prediction models are developed separately from their upstream perception (detection and tracking) modules, assuming ground truth past trajectories as inputs. As a result, their performance degrades significantly when using real-world noisy tracking results as inputs. This is typically caused by the propagation of errors from tracking to prediction, such as noisy tracks, fragments and identity switches. To alleviate this propagation of errors, we propose a new prediction paradigm that uses detections and their affinity matrices across frames as inputs, removing the need for error- prone data association during tracking. Since affinity matrices contain “soft” information about the similarity and identity of detections across frames, making prediction directly from affinity matrices retains strictly more information than making prediction from the tracklets generated by data association. Experiments on large-scale, real-world autonomous driving datasets show that our affinity-based prediction scheme 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our project website is at https://www.xinshuoweng.com/projects/Affinipred. reduces overall prediction errors by up to 57.9%, in comparison to standard prediction pipelines that use tracklets as inputs, with even more significant error reduction (up to 88.6%) if restricting the evaluation to challenging scenarios with tracking errors.

----

## [638] Convolutions for Spatial Interaction Modeling

**Authors**: *Zhaoen Su, Chao Wang, David Bradley, Carlos Vallespi-Gonzalez, Carl Wellington, Nemanja Djuric*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00647](https://doi.org/10.1109/CVPR52688.2022.00647)

**Abstract**:

In many different fields interactions between objects play a critical role in determining their behavior. Graph neural networks (GNNs) have emerged as a powerful tool for modeling interactions, although often at the cost of adding considerable complexity and latency. In this paper, we consider the problem of spatial interaction modeling in the context of predicting the motion of actors around autonomous vehicles, and investigate alternatives to GNNs. We revisit 2D convolutions and show that they can demonstrate comparable performance to graph networks in modeling spatial interactions with lower latency, thus providing an effective and efficient alternative in time-critical systems. Moreover, we propose a novel interaction loss to further improve the interaction modeling of the considered methods.

----

## [639] Style-ERD: Responsive and Coherent Online Motion Style Transfer

**Authors**: *Tianxin Tao, Xiaohang Zhan, Zhongquan Chen, Michiel van de Panne*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00648](https://doi.org/10.1109/CVPR52688.2022.00648)

**Abstract**:

Motion style transfer is a common method for enriching character animation. Motion style transfer algorithms are often designed for offline settings where motions are processed in segments. However, for online animation applications, such as real-time avatar animation from motion capture, motions need to be processed as a stream with minimal latency. In this work, we realize a flexible, high-quality motion style transfer method for this setting. We propose a novel style transfer model, Style-ERD, to stylize motions in an online manner with an Encoder-Recurrent-Decoder structure, along with a novel discriminator that combines feature attention and temporal attention. Our method stylizes motions into multiple target styles with a unified model. Although our method targets online settings, it outperforms previous offline methods in motion realism and style expressiveness and provides significant gains in runtime efficiency.

----

## [640] Neural Inertial Localization

**Authors**: *Sachini Herath, David Caruso, Chen Liu, Yufan Chen, Yasutaka Furukawa*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00649](https://doi.org/10.1109/CVPR52688.2022.00649)

**Abstract**:

This paper proposes the inertial localization problem, the task of estimating the absolute location from a sequence of inertial sensor measurements. This is an exciting and unexplored area of indoor localization research, where we present a rich dataset with 53 hours of inertial sensor data and the associated ground truth locations. We developed a solution, dubbed neural inertial localization (NILoc) which 1) uses a neural inertial navigation technique to turn inertial sensor history to a sequence of velocity vectors; then 2) employs a transformer-based neural architecture to find the device location from the sequence of velocities. We only use an IMU sensor, which is energy efficient and privacy preserving compared to WiFi, cameras, and other data sources. Our approach is significantly faster and achieves competitive results even compared with state-of-the-art methods that require a floorplan and run 20 to 30 times slower. We share our code, model and data at https://sachini.github.io/niloc.

----

## [641] RIO: Rotation-equivariance supervised learning of robust inertial odometry

**Authors**: *Xiya Cao, Caifa Zhou, Dandan Zeng, Yongliang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00650](https://doi.org/10.1109/CVPR52688.2022.00650)

**Abstract**:

This paper introduces rotation-equivariance as a self-supervisor to train inertial odometry models. We demonstrate that the self-supervised scheme provides a powerful supervisory signal at training phase as well as at inference stage. It reduces the reliance on massive amounts of labeled data for training a robust model and makes it possible to update the model using various unlabeled data. Further, we propose adaptive Test-Time Training (TTT) based on uncertainty estimations in order to enhance the generalizability of the inertial odometry to various unseen data. We show in experiments that the Rotation-equivariance-supervised Inertial Odometry (RIO) trained with 30% data achieves on par performance with a model trained with the whole dataset. Adaptive TTT improves models' performance in all cases and makes more than 25% improvements under several scenarios. We release our code and dataset at this website.

----

## [642] CaDeX: Learning Canonical Deformation Coordinate Space for Dynamic Surface Representation via Neural Homeomorphism

**Authors**: *Jiahui Lei, Kostas Daniilidis*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00651](https://doi.org/10.1109/CVPR52688.2022.00651)

**Abstract**:

While neural representations for static 3D shapes are widely studied, representations for deformable surfaces are limited to be template-dependent or to lack efficiency. We introduce Canonical Deformation Coordinate Space (CaDeX), a unified representation of both shape and nonrigid motion. Our key insight is the factorization of the deformation between frames by continuous bijective canonical maps (homeomorphisms) and their inverses that go through a learned canonical shape. Our novel deformation representation and its implementation are simple, efficient, and guarantee cycle consistency, topology preservation, and, if needed, volume conservation. Our modelling of the learned canonical shapes provides a flexible and stable space for shape prior learning. We demonstrate state-of-the-art performance in modelling a wide range of deformable geometries: human bodies, animal bodies, and articulated objects. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://www.cis.upenn.edu/-leijh/projects/cadex

----

## [643] ElePose: Unsupervised 3D Human Pose Estimation by Predicting Camera Elevation and Learning Normalizing Flows on 2D Poses

**Authors**: *Bastian Wandt, James J. Little, Helge Rhodin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00652](https://doi.org/10.1109/CVPR52688.2022.00652)

**Abstract**:

Human pose estimation from single images is a challenging problem that is typically solved by supervised learning. Unfortunately, labeled training data does not yet exist for many human activities since 3D annotation requires dedicated motion capture systems. Therefore, we propose an unsupervised approach that learns to predict a 3D human pose from a single image while only being trained with 2D pose data, which can be crowd-sourced and is already widely available. To this end, we estimate the 3D pose that is most likely over random projections, with the likelihood estimated using normalizing flows on 2D poses. While previous work requires strong priors on camera rotations in the training data set, we learn the distribution of camera angles which significantly improves the performance. Another part of our contribution is to stabilize training with normalizing flows on high-dimensional 3D pose data by first projecting the 2D poses to a linear subspace. We outperform the state-of-the-art unsupervised human pose estimation methods on the benchmark datasets Human3.6M and MPI-INF-3DHP in many metrics.

----

## [644] Projective Manifold Gradient Layer for Deep Rotation Regression

**Authors**: *Jiayi Chen, Yingda Yin, Tolga Birdal, Baoquan Chen, Leonidas J. Guibas, He Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00653](https://doi.org/10.1109/CVPR52688.2022.00653)

**Abstract**:

Regressing rotations on SO(3) manifold using deep neural networks is an important yet unsolved problem. The gap between the Euclidean network output space and the non-Euclidean SO(3) manifold imposes a severe challenge for neural network learning in both forward and backward passes. While several works have proposed different regression-friendly rotation representations, very few works have been devoted to improving the gradient back-propagating in the backward pass. In this paper, we propose a manifold-aware gradient that directly backpropagates into deep network weights. Leveraging Riemannian optimization to construct a novel projective gradient, our proposed regularized projective manifold gradient (RPMG) method helps networks achieve new state-of-the-art performance in a variety of rotation estimation tasks. Our proposed gradient layer can also be applied to other smooth manifolds such as the unit sphere. Our project page is at https://jychen18.github.io/RPMG.

----

## [645] Multimodal Colored Point Cloud to Image Alignment

**Authors**: *Noam Rotstein, Amit Bracha, Ron Kimmel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00654](https://doi.org/10.1109/CVPR52688.2022.00654)

**Abstract**:

Reconstruction of geometric structures from images using supervised learning suffers from limited available amount of accurate data. One type of such data is accurate real-world RGB-D images. A major challenge in acquiring such ground truth data is the accurate alignment between RGB images and the point cloud measured by a depth scanner. To overcome this difficulty, we consider a differential optimization method that aligns a colored point cloud with a given color image through iterative geometric and color matching. In the proposed framework, the optimization minimizes the photometric difference between the colors of the point cloud and the corresponding colors of the image pixels. Unlike other methods that try to reduce this photometric error, we analyze the computation of the gradient on the image plane and propose a different direct scheme. We assume that the colors produced by the geometric scanner camera and the color camera sensor are different and therefore characterized by different chromatic acquisition properties. Under these multimodal conditions, we find the transformation between the camera image and the point cloud colors. We alternately optimize for aligning the position of the point cloud and matching the different color spaces. The alignments produced by the proposed method are demonstrated on both synthetic data with quantitative evaluation and real scenes with qualitative results.

----

## [646] Multi-instance Point Cloud Registration by Efficient Correspondence Clustering

**Authors**: *Weixuan Tang, Danping Zou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00655](https://doi.org/10.1109/CVPR52688.2022.00655)

**Abstract**:

We address the problem of estimating the poses of multiple instances of the source point cloud within a target point cloud. Existing solutions require sampling a lot of hypotheses to detect possible instances and reject the outliers, whose robustness and efficiency degrade notably when the number of instances and outliers increase. We propose to directly group the set of noisy correspondences into different clusters based on a distance invariance matrix. The instances and outliers are automatically identified through clustering. Our method is robust and fast. We evaluated our method on both synthetic and real-world datasets. The results show that our approach can correctly register up to 20 instances with an F1 score of 90.46% in the presence of 70% outliers, which performs significantly better and at least 10x faster than existing methods. (Source code: https://github.com/SITU-ViSYSlmulti-instant-reg).

----

## [647] REGTR: End-to-end Point Cloud Correspondences with Transformers

**Authors**: *Zi Jian Yew, Gim Hee Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00656](https://doi.org/10.1109/CVPR52688.2022.00656)

**Abstract**:

Despite recent success in incorporating learning into point cloud registration, many works focus on learning feature descriptors and continue to rely on nearest-neighbor feature matching and outlier filtering through RANSAC to obtain the final set of correspondences for pose estimation. In this work, we conjecture that attention mechanisms can replace the role of explicit feature matching and RANSAC, and thus propose an end-to-end framework to directly predict the final set of correspondences. We use a network architecture consisting primarily of transformer layers containing self and cross attentions, and train it to predict the probability each point lies in the overlapping region and its corresponding position in the other point cloud. The required rigid transformation can then be estimated directly from the predicted correspondences without further post-processing. Despite its simplicity, our approach achieves state-of-the-art performance on 3DMatch and ModelNet benchmarks. Our source code can be found at https://github.com/yewzijian/RegTR.

----

## [648] Text2Pos: Text-to-Point-Cloud Cross-Modal Localization

**Authors**: *Manuel Kolmet, Qunjie Zhou, Aljosa Osep, Laura Leal-Taixé*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00657](https://doi.org/10.1109/CVPR52688.2022.00657)

**Abstract**:

Natural language-based communication with mobile devices and home appliances is becoming increasingly popular and has the potential to become natural for communicating with mobile robots in the future. Towards this goal, we investigate cross-modal text-to-point-cloud localization that will allow us to specify, for example, a vehicle pick-up or goods delivery location. In particular, we propose Text2Pos, a cross-modal localization module that learns to align textual descriptions with localization cues in a coarse-to-fine manner. Given a point cloud of the environment, Text2Pos locates a position that is specified via a natural language-based description of the immediate surroundings. To train Text2Pos and study its performance, we construct KITTI360Pose, the first dataset for this task based on the recently introduced KITTI360 dataset. Our experiments show that we can localize 65% of textual queries within 15m distance to query locations for top-10 retrieved locations. This is a starting point that we hope will spark future developments towards language-based navigation. “Alexa, hand me over my special delivery at the sidewalk in front of the yellow building next to the blue bus stop.”

----

## [649] BCOT: A Markerless High-Precision 3D Object Tracking Benchmark

**Authors**: *Jiachen Li, Bin Wang, Shiqiang Zhu, Xin Cao, Fan Zhong, Wenxuan Chen, Te Li, Jason Gu, Xueying Qin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00658](https://doi.org/10.1109/CVPR52688.2022.00658)

**Abstract**:

Template-based 3D object tracking still lacks a high-precision benchmark of real scenes due to the difficulty of annotating the accurate 3D poses of real moving video objects without using markers. In this paper, we present a multi-view approach to estimate the accurate 3D poses of real moving objects, and then use binocular data to construct a new benchmark for monocular textureless 3D object tracking. The proposed method requires no markers, and the cameras only need to be synchronous, relatively fixed as cross-view and calibrated. Based on our object-centered model, we jointly optimize the object pose by minimizing shape reprojection constraints in all views, which greatly improves the accuracy compared with the single-view approach, and is even more accurate than the depth-based method. Our new benchmark dataset contains 20 textureless objects, 22 scenes, 404 video sequences and 126K images captured in real scenes. The annotation error is guaranteed to be less than 2mm, according to both theoretical analysis and validation experiments. We reevaluate the state-of-the-art 3D object tracking methods with our dataset, reporting their performance ranking in real scenes. Our BCOT benchmark and code can be found at https://ar3dv.github.io/BCOT-Benchmark/.

----

## [650] SAR-Net: Shape Alignment and Recovery Network for Category-level 6D Object Pose and Size Estimation

**Authors**: *Haitao Lin, Zichang Liu, Chilam Cheang, Yanwei Fu, Guodong Guo, Xiangyang Xue*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00659](https://doi.org/10.1109/CVPR52688.2022.00659)

**Abstract**:

Given a single scene image, this paper proposes a method of Category-level 6D Object Pose and Size Estimation (COPSE) from the point cloud of the target object, without external real pose-annotated training data. Specifically, beyond the visual cues in RGB images, we rely on the shape information predominately from the depth (D) channel. The key idea is to explore the shape alignment of each instance against its corresponding category-level template shape, and the symmetric correspondence of each object category for estimating a coarse 3D object shape. Our framework deforms the point cloud of the category-level template shape to align the observed instance point cloud for implicitly representing its 3D rotation. Then we model the symmetric correspondence by predicting symmetric point cloud from the partially observed point cloud. The concatenation of the observed point cloud and symmetric one reconstructs a coarse object shape, thus facilitating object center (3D translation) and 3D size estimation. Extensive experiments on the category-level NOCS benchmark demonstrate that our lightweight model still competes with state-of-the-art approaches that require labeled real-world images. We also deploy our approach to a physical Baxter robot to perform grasping tasks on unseen but category-known instances, and the results further validate the efficacy of our proposed model. Code and pre-trained models are available on the project webpage 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project webpage. https://hetolin.github.io/SAR-Net.

----

## [651] ES6D: A Computation Efficient and Symmetry-Aware 6D Pose Regression Framework

**Authors**: *Ningkai Mo, Wanshui Gan, Naoto Yokoya, Shifeng Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00660](https://doi.org/10.1109/CVPR52688.2022.00660)

**Abstract**:

In this paper, a computation efficient regression framework is presented for estimating the 6D pose of rigid objects from a single RGB-D image, which is applicable to handling symmetric objects. This framework is designed in a simple architecture that efficiently extracts point-wise features from RGB-D data using a fully convolutional network, called XYZNet, and directly regresses the 6D pose without any post refinement. In the case of symmetric object, one object has multiple ground-truth poses, and this one-to-many relationship may lead to estimation ambiguity. In order to solve this ambiguity problem, we design a symmetry-invariant pose distance metric, called average (maximum) grouped primitives distance or A(M)GPD. The proposed A(M)GPD loss can make the regression network converge to the correct state, i.e., all minima in the A(M)GPD loss surface are mapped to the correct poses. Extensive experiments on YCB-Video and TLESS datasets demonstrate the proposed framework's substantially superior performance in top accuracy and low computational cost. The relevant code is available in https://github.com/GANWANSHUI/ES6D.git.

----

## [652] Coupled Iterative Refinement for 6D Multi-Object Pose Estimation

**Authors**: *Lahav Lipson, Zachary Teed, Ankit Goyal, Jia Deng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00661](https://doi.org/10.1109/CVPR52688.2022.00661)

**Abstract**:

We address the task of 6D multi-object pose: given a set of known 3D objects and an RGB or RGB-D input image, we detect and estimate the 6D pose of each object. We propose a new approach to 6D object pose estimation which consists of an end-to-end differentiable architecture that makes use of geometric knowledge. Our approach iteratively refines both pose and correspondence in a tightly coupled manner, allowing us to dynamically remove outliers to improve accuracy. We use a novel differentiable layer to perform pose refinement by solving an optimization problem we refer to as Bidirectional Depth-Augmented Perspective-N-Point (BD-PnP). Our method achieves state-of-the-art accuracy on standard 6D Object Pose benchmarks. Code is available at https://github.com/princeton-vl/Coupled-Iterative-Refinement.

----

## [653] ZebraPose: Coarse to Fine Surface Encoding for 6DoF Object Pose Estimation

**Authors**: *Yongzhi Su, Mahdi Saleh, Torben Fetzer, Jason R. Rambach, Nassir Navab, Benjamin Busam, Didier Stricker, Federico Tombari*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00662](https://doi.org/10.1109/CVPR52688.2022.00662)

**Abstract**:

Establishing correspondences from image to 3D has been a key task of 6DoF object pose estimation for a long time. To predict pose more accurately, deeply learned dense maps replaced sparse templates. Dense methods also improved pose estimation in the presence of occlusion. More recently researchers have shown improvements by learning object fragments as segmentation. In this work, we present a discrete descriptor, which can represent the object surface densely. By incorporating a hierarchical binary grouping, we can encode the object surface very efficiently. Moreover, we propose a coarse to fine training strategy, which enables fine-grained correspondence prediction. Finally, by matching predicted codes with object surface and using a PnP solver, we estimate the 6DoF pose. Results on the public LM-O and YCB-V datasets show major improvement over the state of the art w.r.t. ADD(-S) metric, even surpassing RGB-D based methods in some cases.

----

## [654] SurfEmb: Dense and Continuous Correspondence Distributions for Object Pose Estimation with Learnt Surface Embeddings

**Authors**: *Rasmus Laurvig Haugaard, Anders Glent Buch*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00663](https://doi.org/10.1109/CVPR52688.2022.00663)

**Abstract**:

We present an approach to learn dense, continuous 2D-3D correspondence distributions over the surface of objects from data with no prior knowledge of visual ambiguities like symmetry. We also present a new method for 6D pose estimation of rigid objects using the learnt distributions to sample, score and refine pose hypotheses. The correspondence distributions are learnt with a contrastive loss, represented in object-specific latent spaces by an encoder-decoder query model and a small fully connected key model. Our method is unsupervised with respect to visual ambiguities, yet we show that the query- and key models learn to represent accurate multi-modal surface distributions. Our pose estimation method improves the state-of-the-art significantly on the comprehensive BOP Challenge, trained purely on synthetic data, even compared with methods trained on real data. The project site is at surfemb.github.io.

----

## [655] MetaPose: Fast 3D Pose from Multiple Views without 3D Supervision

**Authors**: *Ben Usman, Andrea Tagliasacchi, Kate Saenko, Avneesh Sud*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00664](https://doi.org/10.1109/CVPR52688.2022.00664)

**Abstract**:

In the era of deep learning, human pose estimation from multiple cameras with unknown calibration has received little attention to date. We show how to train a neural model to perform this task with high precision and minimal latency overhead. The proposed model takes into account joint location uncertainty due to occlusion from multiple views, and requires only 2D keypoint data for training. Our method outperforms both classical bundle adjustment and weakly-supervised monocular 3D baselines on the well-established Human3.6M dataset, as well as the more challenging in-the-wild Ski-Pose PTZ dataset.

----

## [656] Templates for 3D Object Pose Estimation Revisited: Generalization to New Objects and Robustness to Occlusions

**Authors**: *Van Nguyen Nguyen, Yinlin Hu, Yang Xiao, Mathieu Salzmann, Vincent Lepetit*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00665](https://doi.org/10.1109/CVPR52688.2022.00665)

**Abstract**:

We present a method that can recognize new objects and estimate their 3D pose in RGB images even under partial occlusions. Our method requires neither a training phase on these objects nor real images depicting them, only their CAD models. It relies on a small set of training objects to learn local object representations, which allow us to locally match the input image to a set of “templates”, rendered images of the CAD models for the new objects. In contrast with the state-of-the-art methods, the new objects on which our method is applied can be very different from the training objects. As a result, we are the first to show generalization without retraining on the LINEMOD and Occlusion-LINEMOD datasets. Our analysis of the failure modes of previous template-based approaches further confirms the benefits of local features for template matching. We outperform the state-of-the-art template matching methods on the LINEMOD, Occlusion-LINEMOD and T-LESS datasets. Our source code and data are publicly available at https://github.com/nv-nguyen/template-pose.

----

## [657] GPV-Pose: Category-level Object Pose Estimation via Geometry-guided Point-wise Voting

**Authors**: *Yan Di, Ruida Zhang, Zhiqiang Lou, Fabian Manhardt, Xiangyang Ji, Nassir Navab, Federico Tombari*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00666](https://doi.org/10.1109/CVPR52688.2022.00666)

**Abstract**:

While 6D object pose estimation has recently made a huge leap forward, most methods can still only handle a single or a handful of different objects, which limits their applications. To circumvent this problem, category-level object pose estimation has recently been revamped, which aims at predicting the 6D pose as well as the 3D metric size for previously unseen instances from a given set of object classes. This is, however, a much more challenging task due to severe intra-class shape variations. To address this issue, we propose GPV-Pose, a novel framework for robust category-level pose estimation, harnessing geometric insights to enhance the learning of category-level pose-sensitive features. First, we introduce a decoupled confidence-driven rotation representation, which allows geometry-aware recovery of the associated rotation matrix. Second, we propose a novel geometry-guided point-wise voting paradigm for robust retrieval of the 3D object bounding box. Finally, leveraging these different output streams, we can enforce several geometric consistency terms, further increasing performance, especially for non-symmetric categories. GPV-Pose produces superior results to state-of-the-art competitors on common public benchmarks, whilst almost achieving real-time inference speed at 20 FPS.

----

## [658] HSC4D: Human-centered 4D Scene Capture in Large-scale Indoor-outdoor Space Using Wearable IMUs and LiDAR

**Authors**: *Yudi Dai, Yitai Lin, Chenglu Wen, Siqi Shen, Lan Xu, Jingyi Yu, Yuexin Ma, Cheng Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00667](https://doi.org/10.1109/CVPR52688.2022.00667)

**Abstract**:

We propose Human-centered 4D Scene Capture (HSC4D) to accurately and efficiently create a dynamic digital world, containing large-scale indoor-outdoor scenes, diverse human motions, and rich interactions between humans and environments. Using only body-mounted IMUs and LiDAR, HSC4D is space-free without any external devices' constraints and map-free without pre-built maps. Considering that IMUs can capture human poses but always drift for long-period use, while LiDAR is stable for global localization but rough for local positions and orientations, HSC4D makes both sensors complement each other by a joint optimization and achieves promising results for long-term capture. Relationships between humans and environments are also explored to make their interaction more realistic. To facilitate many down-stream tasks, like AR, VR, robots, autonomous driving, etc., we propose a dataset containing three large scenes (1k-5k m
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
) with accurate dynamic human motions and locations. Diverse scenarios (climbing gym, multi-story building, slope, etc.) and challenging human activities (exercising, walking up/down stairs, climbing, etc.) demonstrate the effectiveness and the generalization ability of HSC4D. The dataset and code is available at lidarhumanmotion.net/hsc4d.

----

## [659] OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation

**Authors**: *Dingding Cai, Janne Heikkilä, Esa Rahtu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00668](https://doi.org/10.1109/CVPR52688.2022.00668)

**Abstract**:

This paper proposes a universal framework, called OVE6D, for model-based 6D object pose estimation from a single depth image and a target object mask. Our model is trained using purely synthetic data rendered from ShapeNet, and, unlike most of the existing methods, it generalizes well on new real-world objects without any fine-tuning. We achieve this by decomposing the 6D pose into viewpoint, in-plane rotation around the camera optical axis and translation, and introducing novel lightweight modules for estimating each component in a cascaded manner. The resulting network contains less than 4M parameters while demon-strating excellent performance on the challenging T-LESS and Occluded LINEMOD datasets without any dataset-specific training. We show that OVE6D outperforms some contemporary deep learning-based pose estimation methods specifically trained for individual objects or datasets with real-world training data. The implementation is available at https://github.com/dingdingcai/OVE6D-pose.

----

## [660] FS6D: Few-Shot 6D Pose Estimation of Novel Objects

**Authors**: *Yisheng He, Yao Wang, Haoqiang Fan, Jian Sun, Qifeng Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00669](https://doi.org/10.1109/CVPR52688.2022.00669)

**Abstract**:

6D object pose estimation networks are limited in their capability to scale to large numbers of object instances due to the close-set assumption and their reliance on high-fidelity object CAD models. In this work, we study a new open set problem; the few-shot 6D object poses estimation: estimating the 6D pose of an unknown object by a few support views without extra training. To tackle the problem, we point out the importance of fully exploring the appearance and geometric relationship between the given support views and query scene patches and propose a dense prototypes matching framework by extracting and matching dense RGBD prototypes with transformers. Moreover, we show that the priors from diverse appearances and shapes are crucial to the generalization capability under the problem setting and thus propose a large-scale RGBD photorealistic dataset (ShapeNet6D) for network pre-training. A simple and effective online texture blending approach is also introduced to eliminate the domain gap from the synthesis dataset, which enriches appearance diversity at a low cost. Finally, we discuss possible solutions to this problem and establish benchmarks on popular datasets to facilitate future research. [project page]

----

## [661] OnePose: One-Shot Object Pose Estimation without CAD Models

**Authors**: *Jiaming Sun, Zihao Wang, Siyu Zhang, Xingyi He, Hongcheng Zhao, Guofeng Zhang, Xiaowei Zhou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00670](https://doi.org/10.1109/CVPR52688.2022.00670)

**Abstract**:

We propose a new method named OnePose for object pose estimation. Unlike existing instance-level or category-level methods, OnePose does not rely on CAD models and can handle objects in arbitrary categories without instance-or category-specific network training. OnePose draws the idea from visual localization and only requires a simple RGB video scan of the object to build a sparse SfM model of the object. Then, this model is registered to new query images with a generic feature matching network. To mitigate the slow runtime of existing visual localization methods, we propose a new graph attention network that directly matches 2D interest points in the query image with the 3D points in the SfM model, resulting in efficient and robust pose estimation. Combined with a feature-based pose tracker, OnePose is able to stably detect and track 6D poses of everyday household objects in real-time. We also collected a large-scale dataset that consists of 450 sequences of 150 objects. Code and data are available at the project page: https://zju3dv.github.io/onepose/.

----

## [662] OSOP: A Multi-Stage One Shot Object Pose Estimation Framework

**Authors**: *Ivan Shugurov, Fu Li, Benjamin Busam, Slobodan Ilic*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00671](https://doi.org/10.1109/CVPR52688.2022.00671)

**Abstract**:

We present a novel one-shot method for object detection and 6 DoF pose estimation, that does not require training on target objects. At test time, it takes as input a target image and a textured 3D query model. The core idea is to represent a 3D model with a number of 2D templates rendered from different viewpoints. This enables CNN-based direct dense feature extraction and matching. The object is first localized in 2D, then its approximate viewpoint is estimated, followed by dense 2D-3D correspondence prediction. The final pose is computed with PnP. We evaluate the method on LineMOD, Occlusion, Homebrewed, YCB-V and TLESS datasets and report very competitive performance in comparison to the state-of-the-art methods trained on synthetic data, even though our method is not trained on the object models used for testing.

----

## [663] DiffPoseNet: Direct Differentiable Camera Pose Estimation

**Authors**: *Chethan M. Parameshwara, Gokul Hari, Cornelia Fermüller, Nitin J. Sanket, Yiannis Aloimonos*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00672](https://doi.org/10.1109/CVPR52688.2022.00672)

**Abstract**:

Current deep neural network approaches for camera pose estimation rely on scene structure for 3D motion estimation, but this decreases the robustness and thereby makes cross-dataset generalization difficult. In contrast, classical approaches to structure from motion estimate 3D motion utilizing optical flow and then compute depth. Their accuracy, however, depends strongly on the quality of the optical flow. To avoid this issue, direct methods have been proposed, which separate 3D motion from depth estimation, but compute 3D motion using only image gradients in the form of normal flow. In this paper, we introduce a network NFlowNet, for normal flow estimation which is used to enforce robust and direct constraints. In particular, normal flow is used to estimate relative camera pose based on the cheirality (depth positivity) constraint. We achieve this by formulating the optimization problem as a differentiable cheirality layer, which allows for end-to-end learning of camera pose. We perform extensive qualitative and quantitative evaluation of the proposed DiffPoseNet's sensitivity to noise and its generalization across datasets. We compare our approach to existing state-of-the-art methods on KITTI, TartanAir, and TUM-RGBD datasets.

----

## [664] Iterative Corresponding Geometry: Fusing Region and Depth for Highly Efficient 3D Tracking of Textureless Objects

**Authors**: *Manuel Stoiber, Martin Sundermeyer, Rudolph Triebel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00673](https://doi.org/10.1109/CVPR52688.2022.00673)

**Abstract**:

Tracking objects in 3D space and predicting their 6DoF pose is an essential task in computer vision. State-of-the-art approaches often rely on object texture to tackle this problem. However, while they achieve impressive results, many objects do not contain sufficient texture, violating the main underlying assumption. In the following, we thus propose ICG, a novel probabilistic tracker that fuses region and depth information and only requires the object geometry. Our method deploys correspondence lines and points to iteratively refine the pose. We also implement robust occlusion handling to improve performance in real-world settings. Experiments on the YCB-Video, OPT, and Choi datasets demonstrate that, even for textured objects, our approach outperforms the current state of the art with respect to accuracy and robustness. At the same time, ICG shows fast convergence and outstanding efficiency, requiring only 1.3 ms per frame on a single CPU core. Finally, we analyze the influence of individual components and discuss our performance compared to deep learning-based methods. The source code of our tracker is publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/DLR-RM/3DObjectTracking.

----

## [665] CPPF: Towards Robust Category-Level 9D Pose Estimation in the Wild

**Authors**: *Yang You, Ruoxi Shi, Weiming Wang, Cewu Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00674](https://doi.org/10.1109/CVPR52688.2022.00674)

**Abstract**:

In this paper, we tackle the problem of category-level 9D pose estimation in the wild, given a single RGB-D frame. Using supervised data of real-world 9D poses is tedious and erroneous, and also fails to generalize to unseen scenarios. Besides, category-level pose estimation requires a method to be able to generalize to unseen objects at test time, which is also challenging. Drawing inspirations from traditional point pair features (PPFs), in this paper, we design a novel Category-level PPF (CPPF) voting method to achieve accurate, robust and generalizable 9D pose estimation in the wild. To obtain robust pose estimation, we sample numerous point pairs on an object, and for each pair our model predicts necessary SE(3)-invariant voting statistics on object centers, orientations and scales. A novel coarse-to-fine voting algorithm is proposed to eliminate noisy point pair samples and generate final predictions from the population. To get rid of false positives in the orientation voting process, an auxiliary binary disambiguating classification task is introduced for each sampled point pair. In order to detect objects in the wild, we carefully design our sim-to-real pipeline by training on synthetic point clouds only, unless objects have ambiguous poses in geometry. Under this cir-cumstance, color information is leveraged to disambiguate these poses. Results on standard benchmarks show that our method is on par with current state of the arts with real-world training data. Extensive experiments further show that our method is robust to noise and gives promising results under extremely challenging scenarios. Our code is available on https://github.com/qq456cvb/CPPF.

----

## [666] Leveraging Equivariant Features for Absolute Pose Regression

**Authors**: *Mohamed Adel Musallam, Vincent Gaudillière, Miguel Ortiz del Castillo, Kassem Al Ismaeil, Djamila Aouada*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00675](https://doi.org/10.1109/CVPR52688.2022.00675)

**Abstract**:

While end-to-end approaches have achieved state-of-the-art performance in many perception tasks, they are not yet able to compete with 3D geometry-based methods in pose estimation. Moreover, absolute pose regression has been shown to be more related to image retrieval. As a result, we hypothesize that the statistical features learned by classical Convolutional Neural Networks do not carry enough geometric information to reliably solve this inherently geometric task. In this paper, we demonstrate how a translation and rotation equivariant Convolutional Neural Network directly induces representations of camera motions into the feature space. We then show that this geometric property allows for implicitly augmenting the training data under a whole group of image plane-preserving transformations. Therefore, we argue that directly learning equivariant features is preferable than learning data-intensive intermediate representations. Comprehensive experimental validation demonstrates that our lightweight model outperforms existing ones on standard datasets. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>

----

## [667] The Majority Can Help the Minority: Context-rich Minority Oversampling for Long-tailed Classification

**Authors**: *Seulki Park, Youngkyu Hong, Byeongho Heo, Sangdoo Yun, Jin Young Choi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00676](https://doi.org/10.1109/CVPR52688.2022.00676)

**Abstract**:

The problem of class imbalanced data is that the gener-alization performance of the classifier deteriorates due to the lack of data from minority classes. In this paper, we pro-pose a novel minority over-sampling method to augment di-versified minority samples by leveraging the rich context of the majority classes as background images. To diversify the minority samples, our key idea is to paste an image from a minority class onto rich-context images from a majority class, using them as background images. Our method is simple and can be easily combined with the existing long-tailed recognition methods. We empirically prove the effectiveness of the proposed oversampling method through extensive experiments and ablation studies. Without any architectural changes or complex algorithms, our method achieves state-of-the-art performance on various long-tailed classification benchmarks. Our code is made available at https://github.com/naver-ai/cmo.

----

## [668] Long- Tailed Recognition via Weight Balancing

**Authors**: *Shaden Alshammari, Yu-Xiong Wang, Deva Ramanan, Shu Kong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00677](https://doi.org/10.1109/CVPR52688.2022.00677)

**Abstract**:

In the real open world, data tends to follow long-tailed class distributions, motivating the well-studied long-tailed recognition (LTR) problem. Naive training produces models that are biased toward common classes in terms of higher accuracy. The key to addressing LTR is to balance various aspects including data distribution, training losses, and gradients in learning. We explore an orthogonal direction, weight balancing, motivated by the empirical observation that the naively trained classifier has “artificially” larger weights in norm for common classes (because there exists abundant data to train them, unlike the rare classes). We investigate three techniques to balance weights, L2-normalization, weight decay, and MaxNorm. We first point out that L2-normalization “perfectly” balances per-class weights to be unit norm, but such a hard constraint might prevent classes from learning better classifiers. In contrast, weight decay penalizes larger weights more heavily and so learns small balanced weights; the MaxNorm constraint encourages growing small weights within a norm ball but caps all the weights by the radius. Our extensive study shows that both help learn balanced weights and greatly improve the LTR accuracy. Surprisingly, weight decay, although underexplored in LTR, significantly improves over prior work. Therefore, we adopt a two-stage training paradigm and propose a simple approach to LTR: (1) learning features using the cross-entropy loss by tuning weight decay, and (2) learning classifiers using class-balanced loss by tuning weight decay and MaxNorm. Our approach achieves the state-of-the-art accuracy on five standard benchmarks, serving as a future baseline for long-tailed recognition.

----

## [669] Balanced Contrastive Learning for Long-Tailed Visual Recognition

**Authors**: *Jianggang Zhu, Zheng Wang, Jingjing Chen, Yi-Ping Phoebe Chen, Yu-Gang Jiang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00678](https://doi.org/10.1109/CVPR52688.2022.00678)

**Abstract**:

Real-world data typically follow a long-tailed distribution, where a few majority categories occupy most of the data while most minority categories contain a limited number of samples. Classification models minimizing crossentropy struggle to represent and classify the tail classes. Although the problem of learning unbiased classifiers has been well studied, methods for representing imbalanced data are under-explored. In this paper, we focus on representation learning for imbalanced data. Recently, supervised contrastive learning has shown promising performance on balanced data recently. However, through our theoretical analysis, we find that for long-tailed data, it fails to form a regular simplex which is an ideal geometric configuration for representation learning. To correct the optimization behavior of SCL and further improve the performance of long-tailed visual recognition, we propose a novel loss for balanced contrastive learning (BCL). Compared with SCL, we have two improvements in BCL: classaveraging, which balances the gradient contribution of negative classes; class-complement, which allows all classes to appear in every mini-batch. The proposed balanced contrastive learning (BCL) method satisfies the condition of forming a regular simplex and assists the optimization of cross-entropy. Equipped with BCL, the proposed two-branch framework can obtain a stronger feature representation and achieve competitive performance on long-tailed benchmark datasets such as CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT, and iNaturalist2018.

----

## [670] Targeted Supervised Contrastive Learning for Long-Tailed Recognition

**Authors**: *Tianhong Li, Peng Cao, Yuan Yuan, Lijie Fan, Yuzhe Yang, Rogério Feris, Piotr Indyk, Dina Katabi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00679](https://doi.org/10.1109/CVPR52688.2022.00679)

**Abstract**:

Real-world data often exhibits long tail distributions with heavy class imbalance, where the majority classes can dominate the training process and alter the decision bound-aries of the minority classes. Recently, researchers have in-vestigated the potential of supervised contrastive learning for long-tailed recognition, and demonstrated that it provides a strong performance gain. In this paper, we show that while supervised contrastive learning can help improve performance, past baselines suffer from poor uniformity brought in by imbalanced data distribution. This poor uni-formity manifests in samples from the minority class having poor separability in the feature space. To address this problem, we propose targeted supervised contrastive learning (TSC), which improves the uniformity of the feature distribution on the hypersphere. TSC first generates a set of targets uniformly distributed on a hypersphere. It then makes the features of different classes converge to these distinct and uniformly distributed targets during training. This forces all classes, including minority classes, to main-tain a uniform distribution in the feature space, improves class boundaries, and provides better generalization even in the presence of long-tail data. Experiments on multi-ple datasets show that TSC achieves state-of-the-art performance on long-tailed recognition tasks.

----

## [671] Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment

**Authors**: *Mengke Li, Yiu-Ming Cheung, Yang Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00680](https://doi.org/10.1109/CVPR52688.2022.00680)

**Abstract**:

Long-tailed data is still a big challenge for deep neural networks, even though they have achieved great success on balanced data. We observe that vanilla training on longtailed data with crossentropy loss makes the instance-rich head classes severely squeeze the spatial distribution of the tail classes, which leads to difficulty in classifying tail class samples. Furthermore, the original crossentropy loss can only propagate gradient short-lively because the gradient in softmax form rapidly approaches zero as the logit difference increases. This phenomenon is called softmax saturation. It is unfavorable for training on balanced data, but can be utilized to adjust the validity of the samples in long-tailed data, thereby solving the distorted embedding space of long-tailed problems. To this end, this paper proposes the Gaussian clouded logit adjustment by Gaussian perturbation of different class logits with varied amplitude. We define the amplitude of perturbation as cloud size and set relatively large cloud sizes to tail classes. The large cloud size can reduce the softmax saturation and thereby making tail class samples more active as well as enlarging the embedding space. To alleviate the bias in a classifier, we therefore propose the class-based effective number sampling strategy with classifier retraining. Extensive experiments on benchmark datasets validate the superior performance of the proposed method. Source code is available at https://github.com/Keke921/GCLLoss.

----

## [672] Long-tail Recognition via Compositional Knowledge Transfer

**Authors**: *Sarah Parisot, Pedro M. Esperança, Steven McDonagh, Tamas J. Madarasz, Yongxin Yang, Zhenguo Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00681](https://doi.org/10.1109/CVPR52688.2022.00681)

**Abstract**:

In this work, we introduce a novel strategy for long-tail recognition that addresses the tail classes’ few-shot problem via training-free knowledge transfer. Our objective is to transfer knowledge acquired from information-rich common classes to semantically similar, and yet data-hungry, rare classes in order to obtain stronger tail class representations. We leverage the fact that class prototypes and learned cosine classifiers provide two different, complementary representations of class cluster centres in feature space, and use an attention mechanism to select and recompose learned classifier features from common classes to obtain higher quality rare class representations. Our knowledge transfer process is training free, reducing overfitting risks, and can afford continual extension of classifiers to new classes. Experiments show that our approach can achieve significant performance boosts on rare classes while maintaining robust common class performance, outperforming directly comparable state-of-the-art models

----

## [673] Nested Collaborative Learning for Long-Tailed Visual Recognition

**Authors**: *Jun Li, Zichang Tan, Jun Wan, Zhen Lei, Guodong Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00682](https://doi.org/10.1109/CVPR52688.2022.00682)

**Abstract**:

The networks trained on the long-tailed dataset vary remarkably, despite the same training settings, which shows the great uncertainty in long-tailed learning. To alleviate the uncertainty, we propose a Nested Collaborative Learning (NCL), which tackles the problem by collaboratively learning multiple experts together. NCL consists of two core components, namely Nested Individual Learning (NIL) and Nested Balanced Online Distillation (NBOD), which focus on the individual supervised learning for each single expert and the knowledge transferring among multiple experts, respectively. To learn representations more thoroughly, both NIL and NBOD are formulated in a nested way, in which the learning is conducted on not just all categories from a full perspective but some hard categories from a partial perspective. Regarding the learning in the partial perspective, we specifically select the negative categories with high predicted scores as the hard categories by using a proposed Hard Category Mining (HCM). In the NCL, the learning from two perspectives is nested, highly related and complementary, and helps the network to capture not only global and robust features but also meticulous distinguishing ability. Moreover, self-supervision is further utilized for feature enhancement. Extensive experiments manifest the superiority of our method with outperforming the state-of-the-art whether by using a single model or an ensemble. Code is available at https://github.com/Bazinga699/NCL

----

## [674] Retrieval Augmented Classification for Long-Tail Visual Recognition

**Authors**: *Alexander Long, Wei Yin, Thalaiyasingam Ajanthan, Vu Nguyen, Pulak Purkait, Ravi Garg, Alan Blair, Chunhua Shen, Anton van den Hengel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00683](https://doi.org/10.1109/CVPR52688.2022.00683)

**Abstract**:

We introduce Retrieval Augmented Classification (RAC), a generic approach to augmenting standard image classification pipelines with an explicit retrieval module. RAC consists of a standard base image encoder fused with a parallel retrieval branch that queries a non-parametric external memory of pre-encoded images and associated text snippets. We apply RAC to the problem of long-tail classification and demonstrate a significant improvement over previous state-of-the-art on Places365-LT and iNaturalist-2018 (14.5% and 6.7% respectively), despite using only the training datasets themselves as the external information source. We demonstrate that RAC's retrieval module, without prompting, learns a high level of accuracy on tail classes. This, in turn, frees the base encoder to focus on common classes, and improve its performance thereon. RAC represents an alternative approach to utilizing large, pretrained models without requiring fine-tuning, as well as a first step towards more effectively making use of external memory within common computer vision architectures.

----

## [675] Trustworthy Long-Tailed Classification

**Authors**: *Bolian Li, Zongbo Han, Haining Li, Huazhu Fu, Changqing Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00684](https://doi.org/10.1109/CVPR52688.2022.00684)

**Abstract**:

Classification on long-tailed distributed data is a challenging problem, which suffers from serious class-imbalance and accordingly unpromising performance es-pecially on tail classes. Recently, the ensembling based methods achieve the state-of-the-art performance and show great potential. However, there are two limitations for cur-rent methods. First, their predictions are not trustworthy for failure-sensitive applications. This is especially harmful for the tail classes where the wrong predictions is basically fre-quent. Second, they assign unified numbers of experts to all samples, which is redundant for easy samples with excessive computational cost. To address these issues, we propose a Trustworthy Long-tailed Classification (TLC) method to jointly conduct classification and uncertainty estimation to identify hard samples in a multi-expert framework. Our TLC obtains the evidence-based uncertainty (EvU) and ev-idence for each expert, and then combines these uncer-tainties and evidences under the Dempster-Shafer Evidence Theory (DST). Moreover, we propose a dynamic expert en-gagement to reduce the number of engaged experts for easy samples and achieve efficiency while maintaining promising performances. Finally, we conduct comprehensive ex-periments on the tasks of classification, tail detection, OOD detection and failure prediction. The experimental results show that the proposed TLC outperforms existing methods and is trustworthy with reliable uncertainty.

----

## [676] C2AM Loss: Chasing a Better Decision Boundary for Long-Tail Object Detection

**Authors**: *Tong Wang, Yousong Zhu, Yingying Chen, Chaoyang Zhao, Bin Yu, Jinqiao Wang, Ming Tang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00685](https://doi.org/10.1109/CVPR52688.2022.00685)

**Abstract**:

Long-tail object detection suffers from poor performance on tail categories. We reveal that the real culprit lies in the extremely imbalanced distribution of the classifier's weight norm. For conventional softmax cross-entropy loss, such imbalanced weight norm distribution yields ill conditioned decision boundary for categories which have small weight norms. To get rid of this situation, we choose to maxi-mize the cosine similarity between the learned feature and the weight vector of target category rather than the inner-product of them. The decision boundary between any two categories is the angular bisector of their weight vectors. Whereas, the absolutely equal decision boundary is sub-optimal because it reduces the model's sensitivity to vari-ous categories. Intuitively, categories with rich data diver-sity should occupy a larger area in the classification space while categories with limited data diversity should occupy a slightly small space. Hence, we devise a Category-Aware Angular Margin Loss (C2AM Loss) to introduce an adaptive angular margin between any two categories. Specif-ically, the margin between two categories is proportional to the ratio of their classifiers' weight norms. As a result, the decision boundary is slightly pushed towards the cat-egory which has a smaller weight norm. We conduct comprehensive experiments on LVIS dataset. C2AM Loss brings 4.9~5.2 AP improvements on different detectors and back-bones compared with baseline.

----

## [677] Equalized Focal Loss for Dense Long-Tailed Object Detection

**Authors**: *Bo Li, Yongqiang Yao, Jingru Tan, Gang Zhang, Fengwei Yu, Jianwei Lu, Ye Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00686](https://doi.org/10.1109/CVPR52688.2022.00686)

**Abstract**:

Despite the recent success of long-tailed object detection, almost all long-tailed object detectors are developed based on the two-stage paradigm. In practice, one-stage detectors are more prevalent in the industry because they have a simple and fast pipeline that is easy to deploy. However, in the long-tailed scenario, this line of work has not been explored so far. In this paper, we investigate whether one-stage detectors can perform well in this case. We discover the primary obstacle that prevents one-stage detectors from achieving excellent performance is: categories suffer from different degrees of positive-negative imbalance problems under the long-tailed data distribution. The conventional focal loss balances the training process with the same modulating factor for all categories, thus failing to handle the long-tailed problem. To address this issue, we propose the Equalized Focal Loss (EFL) that rebalances the loss contribution of positive and negative samples of different categories independently according to their imbalance degrees. Specifically, EFL adopts a category-relevant modulating factor which can be adjusted dynamically by the training status of different categories. Extensive experiments conducted on the challenging LVIS v1 benchmark demonstrate the effectiveness of our proposed method. With an end-to-end training pipeline, EFL achieves 29.2% in terms of overall AP and obtains significant performance improvements on rare categories, surpassing all existing state-of-the-art methods. The code is available at https: //github.com/ModelTC/EOD.

----

## [678] Relieving Long-tailed Instance Segmentation via Pairwise Class Balance

**Authors**: *Yin-Yin He, Peizhen Zhang, Xiu-Shen Wei, Xiangyu Zhang, Jian Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00687](https://doi.org/10.1109/CVPR52688.2022.00687)

**Abstract**:

Long-tailed instance segmentation is a challenging task due to the extreme imbalance of training samples among classes. It causes severe biases of the head classes (with majority samples) against the tailed ones. This renders “how to appropriately define and alleviate the bias” one of the most important issues. Prior works mainly use label distribution or mean score information to indicate a coarse-grained bias. In this paper, we explore to excavate the confusion matrix, which carries the fine-grained misclassification details, to relieve the pairwise biases, generalizing the coarse one. To this end, we propose a novel Pairwise Class Balance (PCB) method, built upon a confusion matrix which is updated during training to accumulate the ongoing prediction preferences. PCB generates fightback soft labels for regularization during training. Besides, an iterative learning paradigm is developed to support a progressive and smooth regularization in such debiasing. PCB can be plugged and played to any existing method as a complement. Experimental results on LVIS demonstrate that our method achieves state-of-the-art performance without bells and whistles. Superior results across various architectures show the generalization ability. The code and trained models are available at https://github.com/megvii-research/PCB.

----

## [679] iFS-RCNN: An Incremental Few-shot Instance Segmenter

**Authors**: *Khoi Nguyen, Sinisa Todorovic*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00688](https://doi.org/10.1109/CVPR52688.2022.00688)

**Abstract**:

This paper addresses incremental few-shot instance seg-mentation, where a few examples of new object classes ar-rive when access to training examples of old classes is not available anymore, and the goal is to perform well on both old and new classes. We make two contributions by extending the common Mask-RCNN framework in its second stage - namely, we specify a new object class classifier based on the probit function and a new uncertainty-guided bounding-box predictor. The former leverages Bayesian learning to address a paucity of training examples of new classes. The latter learns not only to predict object bounding boxes but also to estimate the uncertainty of the prediction as a guid-ance for bounding box refinement. We also specify two new loss functions in terms of the estimated object-class distribution and bounding-box uncertainty. Our contributions produce significant performance gains on the COCO dataset over the state of the art - specifically, the gain of +6 on the new classes and +16 on the old classes in the AP instance segmentation metric. Furthermore, we are the first to evaluate the incremental few-shot setting on the more challenging LVIS dataset.

----

## [680] Open-Vocabulary Instance Segmentation via Robust Cross-Modal Pseudo-Labeling

**Authors**: *Dat Huynh, Jason Kuen, Zhe Lin, Jiuxiang Gu, Ehsan Elhamifar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00689](https://doi.org/10.1109/CVPR52688.2022.00689)

**Abstract**:

Open-vocabulary instance segmentation aims at segmenting novel classes without mask annotations. It is an important step toward reducing laborious human supervision. Most existing works first pretrain a model on captioned images covering many novel classes and then finetune it on limited base classes with mask annotations. However, the high-level textual information learned from caption pretraining alone cannot effectively encode the details required for pixelwise segmentation. To address this, we propose a cross-modal pseudo-labeling framework, which generates training pseudo masks by aligning word semantics in captions with visual features of object masks in images. Thus, our framework is capable of labeling novel classes in captions via their word semantics to self-train a student model. To account for noises in pseudo masks, we design a robust student model that selectively distills mask knowledge by estimating the mask noise levels, hence mitigating the adverse impact of noisy pseudo masks. By extensive experiments, we show the effectiveness of our framework, where we significantly improve mAP score by 4.5% on MS-COCO and 5.1 % on the large-scale Open Images & Conceptual Captions datasets compared to the state-of-the-art.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/hbdat/cvpr22_cross_modal_pseudo_labeling.

----

## [681] SimT: Handling Open-set Noise for Domain Adaptive Semantic Segmentation

**Authors**: *Xiaoqing Guo, Jie Liu, Tongliang Liu, Yixuan Yuan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00690](https://doi.org/10.1109/CVPR52688.2022.00690)

**Abstract**:

This paper studies a practical domain adaptive (DA) semantic segmentation problem where only pseudo-labeled target data is accessible through a black-box model. Due to the domain gap and label shift between two domains, pseudo-labeled target data contains mixed closed-set and open-set label noises. In this paper, we propose a simplex noise transition matrix (SimT) to model the mixed noise distributions in DA semantic segmentation and formulate the problem as estimation of SimT. By exploiting computational geometry analysis and properties of segmentation, we design three complementary regularizers, i.e. volume regularization, anchor guidance, convex guarantee, to approximate the true SimT. Specifically, volume regularization minimizes the volume of simplex formed by rows of the non-square SimT, which ensures outputs of segmentation model to fit into the ground truth label distribution. To compensate for the lack of open-set knowledge, anchor guidance and convex guarantee are devised to facilitate the modeling of open-set noise distribution and enhance the discriminative feature learning among closed-set and open-set classes. The estimated SimT is further utilized to correct noise issues in pseudo labels and promote the generalization ability of segmentation model on target domain data. Extensive experimental results demonstrate that the proposed SimT can be flexibly plugged into existing DA methods to boost the performance. The source code is available at https://github.com/CityU-AIM-Group/SimT.

----

## [682] Undoing the Damage of Label Shift for Cross-domain Semantic Segmentation

**Authors**: *Yahao Liu, Jinhong Deng, Jiale Tao, Tong Chu, Lixin Duan, Wen Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00691](https://doi.org/10.1109/CVPR52688.2022.00691)

**Abstract**:

Existing works typically treat cross-domain semantic segmentation (CDSS) as a data distribution mismatch prob-lem and focus on aligning the marginal distribution or con-ditional distribution. However, the label shift issue is un-fortunately overlooked, which actually commonly exists in the CDSS task, and often causes a classifier bias in the learnt model. In this paper, we give an in-depth analysis and show that the damage of label shift can be overcome by aligning the data conditional distribution and correcting the posterior probability. To this end, we propose a novel approach to undo the damage of the label shift problem in CDSS. In implementation, we adopt class-level feature alignment for conditional distribution alignment, as well as two simple yet effective methods to rectify the classifier bias from source to target by remolding the classifier predictions. We conduct extensive experiments on the benchmark datasets of urban scenes, including GTA5 to Cityscapes and SYNTHIA to Cityscapes, where our proposed approach outperforms previous methods by a large margin. For instance, our model equipped with a self-training strat-egy reaches 59.3% mIoU on GTA5 to Cityscapes, pushing to a new state-of-the-art. The code will be available at https://github.com/manmanjun/Undoing_UDA.

----

## [683] Representation Compensation Networks for Continual Semantic Segmentation

**Authors**: *Chang-Bin Zhang, Jia-Wen Xiao, Xialei Liu, Ying-Cong Chen, Ming-Ming Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00692](https://doi.org/10.1109/CVPR52688.2022.00692)

**Abstract**:

In this work, we study the continual semantic segmentation problem, where the deep neural networks are required to incorporate new classes continually without catastrophic forgetting. We propose to use a structural re-parameterization mechanism, named representation compensation (RC) module, to decouple the representation learning of both old and new knowledge. The RC module consists of two dynamically evolved branches with one frozen and one trainable. Besides, we design a pooled cube knowledge distillation strategy on both spatial and channel dimensions to further enhance the plasticity and stability of the model. We conduct experiments on two challenging continual semantic segmentation scenarios, continual class segmentation and continual domain segmentation. Without any extra computational overhead and parameters during inference, our method outperforms state-of-the-art performance. The code is available at https://github.com/zhangchbin/RCIL.

----

## [684] Remember the Difference: Cross-Domain Few-Shot Semantic Segmentation via Meta-Memory Transfer

**Authors**: *Wenjian Wang, Lijuan Duan, Yuxi Wang, Qing En, Junsong Fan, Zhaoxiang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00693](https://doi.org/10.1109/CVPR52688.2022.00693)

**Abstract**:

Few-shot semantic segmentation intends to predict pixel-level categories using only a few labeled samples. Existing few-shot methods focus primarily on the categories sampled from the same distribution. Nevertheless, this assumption cannot always be ensured. The actual domain shift problem significantly reduces the performance of few-shot learning. To remedy this problem, we propose an interesting and challenging cross-domain few-shot semantic segmentation task, where the training and test tasks perform on different domains. Specifically, we first propose a meta-memory bank to improve the generalization of the segmentation network by bridging the domain gap between source and target domains. The meta-memory stores the intra-domain style information from source domain instances and transfers it to target samples. Subsequently, we adopt a new contrastive learning strategy to explore the knowledge of different categories during the training stage. The negative and positive pairs are obtained from the proposed memory-based style augmentation. Comprehensive experiments demon-strate that our proposed method achieves promising results on cross-domain few-shot semantic segmentation tasks on COCO-20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
, PASCAL-S
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
, FSS-1000, and SUIM datasets.

----

## [685] Domain-Agnostic Prior for Transfer Semantic Segmentation

**Authors**: *Xinyue Huo, Lingxi Xie, Hengtong Hu, Wengang Zhou, Houqiang Li, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00694](https://doi.org/10.1109/CVPR52688.2022.00694)

**Abstract**:

Unsupervised domain adaptation (UDA) is an important topic in the computer vision community. The key difficulty lies in defining a common property between the source and target domains so that the source-domain features can align with the target-domain semantics. In this paper, we present a simple and effective mechanism that regularizes cross-domain representation learning with a domain-agnostic prior (DAP) that constrains the features extracted from source and target domains to align with a domain-agnostic space. In practice, this is easily implemented as an extra loss term that requires a little extra costs. In the standard evaluation protocol of transferring synthesized data to real data, we validate the effectiveness of different types of DAP, especially that borrowed from a text embedding model that shows favorable performance beyond the state-of-the-art UDA approaches in terms of segmentation accuracy. Our research reveals that UDA benefits much from better proxies, possibly from other data modalities.

----

## [686] Image Segmentation Using Text and Image Prompts

**Authors**: *Timo Lüddecke, Alexander S. Ecker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00695](https://doi.org/10.1109/CVPR52688.2022.00695)

**Abstract**:

Image segmentation is usually addressed by training a model for a fixed set of object classes. Incorporating additional classes or more complex queries later is expensive as it requires re-training the model on a dataset that encompasses these expressions. Here we propose a system that can generate image segmentations based on arbitrary prompts at test time. A prompt can be either a text or an image. This approach enables us to create a unified model (trained once) for three common segmentation tasks, which come with distinct challenges: referring expression segmentation, zero-shot segmentation and one-shot segmentation. We build upon the CLIP model as a backbone which we extend with a transformer-based decoder that enables dense prediction. After training on an extended version of the PhraseCut dataset, our system generates a binary segmentation map for an image based on a free-text prompt or on an additional image expressing the query. We analyze different variants of the latter image-based prompts in detail. This novel hybrid input allows for dynamic adaptation not only to the three segmentation tasks mentioned above, but to any binary segmentation task where a text or image query can be formulated. Finally, we find our system to adapt well to generalized queries involving affordances or properties. Code is available at https://eckerlab.org/code/CLIPSeg

----

## [687] PCL: Proxy-based Contrastive Learning for Domain Generalization

**Authors**: *Xufeng Yao, Yang Bai, Xinyun Zhang, Yuechen Zhang, Qi Sun, Ran Chen, Ruiyu Li, Bei Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00696](https://doi.org/10.1109/CVPR52688.2022.00696)

**Abstract**:

Domain generalization refers to the problem of training a model from a collection of different source domains that can directly generalize to the unseen target domains. A promising solution is contrastive learning, which attempts to learn domain-invariant representations by exploiting rich semantic relations among sample-to-sample pairs from different domains. A simple approach is to pull positive sample pairs from different domains closer while pushing other negative pairs further apart. In this paper, we find that directly applying contrastive-based methods (e.g., supervised contrastive learning) are not effective in domain generalization. We argue that aligning positive sample-to-sample pairs tends to hinder the model generalization due to the significant distribution gaps between different domains. To address this issue, we propose a novel proxy-based contrastive learning method, which replaces the original sample-to-sample relations with proxy-to-sample relations, significantly alleviating the positive alignment issue. Experiments on the four standard benchmarks demonstrate the effectiveness of the proposed method. Furthermore, we also consider a more complex scenario where no ImageNet pre-trained models are provided. Our method consistently shows better performance.

----

## [688] Localized Adversarial Domain Generalization

**Authors**: *Wei Zhu, Le Lu, Jing Xiao, Mei Han, Jiebo Luo, Adam P. Harrison*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00697](https://doi.org/10.1109/CVPR52688.2022.00697)

**Abstract**:

Deep learning methods can struggle to handle domain shifts not seen in training data, which can cause them to not generalize well to unseen domains. This has led to research attention on domain generalization (DG), which aims to the model's generalization ability to out-of-distribution. Adversarial domain generalization is a popular approach to DG, but conventional approaches (1) struggle to sufficiently align features so that local neighborhoods are mixed across domains; and (2) can suffer from feature space over collapse which can threaten generalization performance. To address these limitations, we propose localized adversarial domain generalization with space compactness maintenance (LADG) which constitutes two major contributions. First, we propose an adversarial localized classifier as the domain discriminator, along with a principled primary branch. This constructs a min-max game whereby the aim of the featurizer is to produce locally mixed domains. Second, we propose to use a coding-rate loss to alleviate feature space over collapse. We conduct comprehensive experiments on the Wilds DG benchmark to validate our approach, where LADG outperforms leading competitors on most datasets.

----

## [689] Compound Domain Generalization via Meta-Knowledge Encoding

**Authors**: *Chaoqi Chen, Jiongcheng Li, Xiaoguang Han, Xiaoqing Liu, Yizhou Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00698](https://doi.org/10.1109/CVPR52688.2022.00698)

**Abstract**:

Domain generalization (DG) aims to improve the generalization performance for an unseen target domain by using the knowledge of multiple seen source domains. Mainstream DG methods typically assume that the domain label of each source sample is known a priori, which is challenged to be satisfied in many real-world applications. In this paper, we study a practical problem of compound DG, which relaxes the discrete domain assumption to the mixed source domains setting. On the other hand, current DG algorithms prioritize the focus on semantic invariance across domains (one-vs-one), while paying less attention to the holistic semantic structure (many-vs-many). Such holistic semantic structure, referred to as meta-knowledge here, is crucial for learning generalizable representations. To this end, we present COmpound domain generalization via Meta-knowledge ENcoding (COMEN), a general approach to automatically discover and model latent domains in two steps. Firstly, we introduce Style-induced Domain-specific Normalization (SDNorm) to re-normalize the multi-modal underlying distributions, thereby dividing the mixture of source domains into latent clusters. Secondly, we harness the prototype representations, the centroids of classes, to perform relational modeling in the embedding space with two parallel and complementary modules, which explicitly encode the semantic structure for the out-of-distribution generalization. Experiments on four standard DG benchmarks reveal that COMEN exceeds the state-of-the-art performance without the need of domain supervision.

----

## [690] Style Neophile: Constantly Seeking Novel Styles for Domain Generalization

**Authors**: *Juwon Kang, Sohyun Lee, Namyup Kim, Suha Kwak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00699](https://doi.org/10.1109/CVPR52688.2022.00699)

**Abstract**:

This paper studies domain generalization via domain-invariant representation learning. Existing methods in this direction suppose that a domain can be characterized by styles of its images, and train a network using style-augmented data so that the network is not biased to particular style distributions. However, these methods are restricted to a finite set of styles since they obtain styles for augmentation from a fixed set of external images or by in-terpolating those of training data. To address this limitation and maximize the benefit of style augmentation, we propose a new method that synthesizes novel styles constantly during training. Our method manages multiple queues to store styles that have been observed so far, and synthesizes novel styles whose distribution is distinct from the distribution of styles in the queues. The style synthesis process is formu-lated as a monotone submodular optimization, thus can be conducted efficiently by a greedy algorithm. Extensive ex-periments on four public benchmarks demonstrate that the proposed method is capable of achieving state-of-the-art domain generalization performance.

----

## [691] Slimmable Domain Adaptation

**Authors**: *Rang Meng, Weijie Chen, Shicai Yang, Jie Song, Luojun Lin, Di Xie, Shiliang Pu, Xinchao Wang, Mingli Song, Yueting Zhuang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00700](https://doi.org/10.1109/CVPR52688.2022.00700)

**Abstract**:

Vanilla unsupervised domain adaptation methods tend to optimize the model with fixed neural architecture, which is not very practical in real-world scenarios since the target data is usually processed by different resource-limited devices. It is therefore of great necessity to facilitate architecture adaptation across various devices. In this paper, we introduce a simple framework, Slimmable Domain Adaptation, to improve cross-domain generalization with a weight-sharing model bank, from which models of different capacities can be sampled to accommodate different accuracy-efficiency trade-offs. The main challenge in this frame-work lies in simultaneously boosting the adaptation performance of numerous models in the model bank. To tackle this problem, we develop a Stochastic EnsEmble Distillation method to fully exploit the complementary knowledge in the model bank for inter-model interaction. Nevertheless, considering the optimization conflict between inter-model interaction and intra-model adaptation, we augment the existing bi-classifier domain confusion architecture into an Optimization-Separated Tri-Classifier counterpart. After optimizing the model bank, architecture adaptation is leveraged via our proposed Unsupervised Performance Evaluation Metric. Under various resource constraints, our framework surpasses other competing approaches by a very large margin on multiple benchmarks. It is also worth emphasizing that our framework can preserve the performance improvement against the source-only model even when the computing complexity is reduced to 1/64. Code will be available at https://github.com/HIK-LAB/SlimDA.

----

## [692] Exploring Domain-Invariant Parameters for Source Free Domain Adaptation

**Authors**: *Fan Wang, Zhongyi Han, Yongshun Gong, Yilong Yin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00701](https://doi.org/10.1109/CVPR52688.2022.00701)

**Abstract**:

Source-free domain adaptation (SFDA) newly emerges to transfer the relevant knowledge of a well-trained source model to an unlabeled target domain, which is critical in various privacy-preserving scenarios. Most existing methods focus on learning the domain-invariant representations depending solely on the target data, leading to the obtained representations are target-specific. In this way, they cannot fully address the distribution shift problem across domains. In contrast, we provide a fascinating insight: rather than attempting to learn domain-invariant representations, it is better to explore the domain-invariant parameters of the source model. The motivation behind this insight is clear: the domain-invariant representations are dominated by only partial parameters of an available deep source model. We devise the Domain-Invariant Parameter Exploring (DIPE) approach to capture such domain-invariant parameters in the source model to generate domain-invariant representations. A distinguishing method is developed correspondingly for two types of parameters, i.e., domain-invariant and domain-specific parameters, as well as an effective update strategy based on the clustering correction technique and a target hypothesis is proposed. Extensive experiments verify that DIPE successfully exceeds the current state-of-the-art models on many domain adaptation datasets.

----

## [693] Cross-domain Few-shot Learning with Task-specific Adapters

**Authors**: *Wei-Hong Li, Xialei Liu, Hakan Bilen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00702](https://doi.org/10.1109/CVPR52688.2022.00702)

**Abstract**:

In this paper, we look at the problem of cross-domain few-shot classification that aims to learn a classifier from previously unseen classes and domains withfew labeled samples. Recent approaches broadly solve this problem by pa-rameterizing their few-shot classifiers with task-agnostic and task-specific weights where the former is typically learned on a large training set and the latter is dynamically predicted through an auxiliary network conditioned on a small support set. In this work, we focus on the estimation of the latter, and propose to learn task-specific weights from scratch directly on a small support set, in contrast to dynamically estimating them. In particular, through systematic analysis, we show that task-specific weights through parametric adapters in matrix form with residual connections to multiple intermediate layers of a backbone network significantly improves the per-formance of the state-of-the-art models in the Meta-Dataset benchmark with minor additional cost.

----

## [694] Task-Adaptive Negative Envision for Few-Shot Open-Set Recognition

**Authors**: *Shiyuan Huang, Jiawei Ma, Guangxing Han, Shih-Fu Chang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00703](https://doi.org/10.1109/CVPR52688.2022.00703)

**Abstract**:

We study the problem of few-shot open-set recognition (FSOR), which learns a recognition system capable of both fast adaptation to new classes with limited labeled exam-ples and rejection of unknown negative samples. Traditional large-scale open-set methods have been shown in-effective for FSOR problem due to data limitation. Current FSOR methods typically calibrate few-shot closed-set clas-sifiers to be sensitive to negative samples so that they can be rejected via thresholding. However, threshold tuning is a challenging process as different FSOR tasks may require different rejection powers. In this paper, we instead propose task-adaptive negative class envision for FSOR to integrate threshold tuning into the learning process. Specifically, we augment the few-shot closed-set classifier with additional negative prototypes generated from few-shot examples. By incorporating few-shot class correlations in the negative generation process, we are able to learn dynamic rejection boundaries for FSOR tasks. Besides, we extend our method to generalized few-shot open-set recognition (GF-SOR), which requires classification on both many-shot and few-shot classes as well as rejection of negative samples. Extensive experiments on public benchmarks validate our methods on both problems.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://github.com/shiyuanh/TANE

----

## [695] Reusing the Task-specific Classifier as a Discriminator: Discriminator-free Adversarial Domain Adaptation

**Authors**: *Lin Chen, Huaian Chen, Zhixiang Wei, Xin Jin, Xiao Tan, Yi Jin, Enhong Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00704](https://doi.org/10.1109/CVPR52688.2022.00704)

**Abstract**:

Adversarial learning has achieved remarkable performances for unsupervised domain adaptation (UDA). Existing adversarial UDA methods typically adopt an additional discriminator to play the min-max game with a feature extractor. However, most of these methods failed to effectively leverage the predicted discriminative information, and thus cause mode collapse for generator. In this work, we address this problem from a different perspective and design a simple yet effective adversarial paradigm in the form of a discriminator-free adversarial learning network (DALN), wherein the category classifier is reused as a discriminator, which achieves explicit domain alignment and category distinguishment through a unified objective, enabling the DALN to leverage the predicted discriminative information for sufficient feature alignment. Basically, we introduce a Nuclear-norm Wasserstein discrepancy (NWD) that has definite guidance meaning for performing discrimination. Such NWD can be coupled with the classifier to serve as a discriminator satisfying the K-Lipschitz constraint without the requirements of additional weight clipping or gradient penalty strategy. Without bells and whistles, DALN compares favorably against the existing state-of-the-art (SOTA) methods on a variety of public datasets. Moreover, as a plug-and-play technique, NWD can be directly used as a generic regularizer to benefit existing UDA algorithms. Code is available at https://github.com/xiaoachen98/DALN.

----

## [696] Safe Self-Refinement for Transformer-based Domain Adaptation

**Authors**: *Tao Sun, Cheng Lu, Tianshuo Zhang, Haibin Ling*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00705](https://doi.org/10.1109/CVPR52688.2022.00705)

**Abstract**:

Unsupervised Domain Adaptation (UDA) aims to leverage a label-rich source domain to solve tasks on a related unlabeled target domain. It is a challenging problem especially when a large domain gap lies between the source and target domains. In this paper we propose a novel solution named SSRT (Safe Self-Refinement for Transformer-based domain adaptation), which brings improvement from two aspects. First, encouraged by the success of vision transformers in various vision tasks, we arm SSRT with a transformer backbone. We find that the combination of vision transformer with simple adversarial adaptation surpasses best reported Convolutional Neural Network (CNN)-based results on the challenging DomainNet benchmark, showing its strong transferable feature representation. Second, to reduce the risk of model collapse and improve the effectiveness of knowledge transfer between domains with large gaps, we propose a Safe Self-Refinement strategy. Specifically, SSRT utilizes predictions of perturbed target domain data to refine the model. Since the model capacity of vision transformer is large and predictions in such challenging tasks can be noisy, a safe training mechanism is designed to adaptively adjust learning configuration. Extensive evaluations are conducted on several widely tested UDA benchmarks and SSRT achieves consistently the best performances, including 85.43% on Office-Home, 88.76% on VisDA-2017 and 45.2% on DomainNet.

----

## [697] Continual Test-Time Domain Adaptation

**Authors**: *Qin Wang, Olga Fink, Luc Van Gool, Dengxin Dai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00706](https://doi.org/10.1109/CVPR52688.2022.00706)

**Abstract**:

Test-time domain adaptation aims to adapt a source pre-trained model to a target domain without using any source data. Existing works mainly consider the case where the target domain is static. However, real-world machine perception systems are running in non-stationary and continually changing environments where the target domain distribution can change over time. Existing methods, which are mostly based on self-training and entropy regularization, can suffer from these non-stationary environments. Due to the distribution shift over time in the target domain, pseudo-labels become unreliable. The noisy pseudo-labels can further lead to error accumulation and catastrophic forgetting. To tackle these issues, we propose a continual test-time adaptation approach (CoTTA) which comprises two parts. Firstly, we propose to reduce the error accumulation by using weight-averaged and augmentation-averaged predictions which are often more accurate. On the other hand, to avoid catastrophic forgetting, we propose to stochastically restore a small part of the neurons to the source pre-trained weights during each iteration to help preserve source knowledge in the longterm. The proposed method enables the longterm adaptation for all parameters in the network. CoTTA is easy to implement and can be readily incorporated in off-the-shelf pre-trained models. We demonstrate the effectiveness of our approach on four classification tasks and a segmentation task for continual test-time adaptation, on which we outperform existing methods. Our code is available at https://gin.ee/cotta.

----

## [698] Source-Free Domain Adaptation via Distribution Estimation

**Authors**: *Ning Ding, Yixing Xu, Yehui Tang, Chao Xu, Yunhe Wang, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00707](https://doi.org/10.1109/CVPR52688.2022.00707)

**Abstract**:

Domain Adaptation aims to transfer the knowledge learned from a labeled source domain to an unlabeled target domain whose data distributions are different. However, the training data in source domain required by most of the existing methods is usually unavailable in real-world applications due to privacy preserving policies. Recently, Source-Free Domain Adaptation (SFDA) has drawn much attention, which tries to tackle domain adaptation problem without using source data. In this work, we propose a novel framework called SFDA-DE to address SFDA task via source Distribution Estimation. Firstly, we produce robust pseudo-labels for target data with spherical k-means clustering, whose initial class centers are the weight vectors (anchors) learned by the classifier of pretrained model. Furthermore, we propose to estimate the class-conditioned feature distribution of source domain by exploiting target data and corresponding anchors. Finally, we sample surrogate features from the estimated distribution, which are then utilized to align two domains by minimizing a contrastive adaptation loss function. Extensive experiments show that the proposed method achieves state-of-the-art performance on multiple DA benchmarks, and even outperforms traditional DA methods which require plenty of source data.

----

## [699] Domain Adaptation on Point Clouds via Geometry-Aware Implicits

**Authors**: *Yuefan Shen, Yanchao Yang, Mi Yan, He Wang, Youyi Zheng, Leonidas J. Guibas*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00708](https://doi.org/10.1109/CVPR52688.2022.00708)

**Abstract**:

As a popular geometric representation, point clouds have attracted much attention in 3D vision, leading to many applications in autonomous driving and robotics. One important yet unsolved issue for learning on point cloud is that point clouds of the same object can have significant geometric variations if generated using different procedures or captured using different sensors. These inconsistencies induce domain gaps such that neural networks trained on one domain may fail to generalize on others. A typical technique to reduce the domain gap is to perform adversarial training so that point clouds in the feature space can align. However, adversarial training is easy to fall into degenerated local minima, resulting in negative adaptation gains. Here we propose a simple yet effective method for unsupervised domain adaptation on point clouds by employing a self-supervised task of learning geometry-aware implicits, which plays two critical roles in one shot. First, the geometric information in the point clouds is preserved through the implicit representations for downstream tasks. More importantly, the domain-specific variations can be effectively learned away in the implicit space. We also propose an adaptive strategy to compute unsigned distance fields for arbitrary point clouds due to the lack of shape models in practice. When combined with a task loss, the proposed outperforms state-of-the-art unsupervised domain adaptation methods that rely on adversarial domain alignment and more complicated self-supervised tasks. Our method is evaluated on both PointDA-10 and GraspNet datasets. Code and data are available at: https://github.com/Jhonve/ImplicitPCDA.

----

## [700] Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds

**Authors**: *Zhao Jin, Yinjie Lei, Naveed Akhtar, Haifeng Li, Munawar Hayat*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00709](https://doi.org/10.1109/CVPR52688.2022.00709)

**Abstract**:

Point cloud scene flow estimation is of practical importance for dynamic scene navigation in autonomous driving. Since scene flow labels are hard to obtain, current methods train their models on synthetic data and transfer them to real scenes. However, large disparities between existing synthetic datasets and real scenes lead to poor model transfer. We make two major contributions to address that. First, we develop a point cloud collector and scene flow annotator for GTA-V engine to automatically obtain diverse realistic training samples without human intervention. With that, we develop a large-scale synthetic scene flow dataset GTA-SF. Second, we propose a mean-teacher-based domain adaptation framework that leverages self-generated pseudo-labels of the target domain. It also explicitly incorporates shape deformation regularization and surface correspondence refinement to address distortions and misalignments in domain transfer. Through extensive experiments, we show that our GTA-SF dataset leads to a consistent boost in model generalization to three real datasets (i.e., Waymo, Lyft and KITTI) as compared to the most widely used FT3D dataset. Moreover, our framework achieves superior adaptation performance on six source-target dataset pairs, remarkably closing the average domain gap by 60%. Data and codes are available at https://github.com/leolyj/DCA-SRSFE

----

## [701] Hyperspherical Consistency Regularization

**Authors**: *Cheng Tan, Zhangyang Gao, Lirong Wu, Siyuan Li, Stan Z. Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00710](https://doi.org/10.1109/CVPR52688.2022.00710)

**Abstract**:

Recent advances in contrastive learning have enlightened diverse applications across various semi-supervised fields. Jointly training supervised learning and unsupervised learning with a shared feature encoder becomes a common scheme. Though it benefits from taking advantage of both feature-dependent information from self-supervised learning and label-dependent information from supervised learning, this scheme remains suffering from bias of the classifier. In this work, we systematically explore the relationship between self-supervised learning and supervised learning, and study how self-supervised learning helps robust data-efficient deep learning. We propose hyperspherical consistency regularization (HCR), a simple yet effective plug-and-play method, to regularize the classifier using feature-dependent information and thus avoid bias from labels. Specifically, HCR first project logits from the classifier and feature projections from the projection head on the respective hypersphere, then it enforces data points on hyperspheres to have similar structures by minimizing binary cross entropy of pairwise distances' similarity metrics. Extensive experiments on semi-supervised and weakly-supervised learning demonstrate the effectiveness of our method, by showing superior performance with HCR.

----

## [702] BatchFormer: Learning to Explore Sample Relationships for Robust Representation Learning

**Authors**: *Zhi Hou, Baosheng Yu, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00711](https://doi.org/10.1109/CVPR52688.2022.00711)

**Abstract**:

Despite the success of deep neural networks, there are still many challenges in deep representation learning due to the data scarcity issues such as data imbalance, unseen distribution, and domain shift. To address the above-mentioned issues, a variety of methods have been devised to explore the sample relationships in a vanilla way (i.e., from the perspectives of either the input or the loss function), failing to explore the internal structure of deep neural networks for learning with sample relationships. Inspired by this, we propose to enable deep neural networks themselves with the ability to learn the sample relationships from each mini-batch. Specifically, we introduce a batch transformer module or BatchFormer, which is then applied into the batch dimension of each mini-batch to implicitly explore sample relationships during training. By doing this, the proposed method enables the collaboration of different samples, e.g., the head-class samples can also contribute to the learning of the tail classes for long-tailed recognition. Furthermore, to mitigate the gap between training and testing, we share the classifier between with or without the BatchFormer during training, which can thus be removed during testing. We perform extensive experiments on over ten datasets and the proposed method achieves significant improvements on different data scarcity applications without any bells and whistles, including the tasks of long-tailed recognition, compositional zero-shot learning, domain generalization, and contrastive learning. Code is made publicly available at https://github.com/zhihou7/BatchFormer.

----

## [703] Cascade Transformers for End-to-End Person Search

**Authors**: *Rui Yu, Dawei Du, Rodney LaLonde, Daniel Davila, Christopher Funk, Anthony Hoogs, Brian Clipp*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00712](https://doi.org/10.1109/CVPR52688.2022.00712)

**Abstract**:

The goal of person search is to localize a target person from a gallery set of scene images, which is extremely challenging due to large scale variations, pose/viewpoint changes, and occlusions. In this paper, we propose the Cascade Occluded Attention Transformer (COAT) for end-to-end person search. Our three-stage cascade design focuses on detecting people in the first stage, while later stages simultaneously and progressively refine the representation for person detection and re-identification. At each stage the occluded attention transformer applies tighter intersection over union thresholds, forcing the network to learn coarse-to-fine pose/scale invariant features. Meanwhile, we calculate each detection's occluded attention to differentiate a person's tokens from other people or the background. In this way, we simulate the effect of other objects occluding a person of interest at the token-level. Through comprehensive experiments, we demonstrate the benefits of our method by achieving state-of-the-art performance on two benchmark datasets.

----

## [704] Delving Deep into the Generalization of Vision Transformers under Distribution Shifts

**Authors**: *Chongzhi Zhang, Mingyuan Zhang, Shanghang Zhang, Daisheng Jin, Qiang Zhou, Zhongang Cai, Haiyu Zhao, Xianglong Liu, Ziwei Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00713](https://doi.org/10.1109/CVPR52688.2022.00713)

**Abstract**:

Vision Transformers (ViTs) have achieved impressive performance on various vision tasks, yet their generalization under distribution shifts (DS) is rarely understood. In this work, we comprehensively study the out-of-distribution (OOD) generalization of ViTs. For systematic investigation, we first present a taxonomy of DS. We then perform extensive evaluations of ViT variants under different DS and compare their generalization with Convolutional Neural Network (CNN) models. Important observations are obtained: 1) ViTs learn weaker biases on backgrounds and textures, while they are equipped with stronger inductive biases towards shapes and structures, which is more consistent with human cognitive traits. Therefore, ViTs generalize better than CNNs under DS. With the same or less amount of parameters, ViTs are ahead of corresponding CNNs by more than 5% in top-1 accuracy under most types of DS. 2) As the model scale increases, ViTs strengthen these biases and thus gradually narrow the in-distribution and OOD performance gap. To further improve the generalization of ViTs, we design the Generalization-Enhanced ViTs (GE-ViTs) from the perspectives of adversarial learning, information theory, and self-supervised learning. By comprehensively investigating these GE-ViTs and comparing with their corresponding CNN models, we observe: 1) For the enhanced model, larger ViTs still benefit more for the OOD generalization. 2) GE-ViTs are more sensitive to the hyper-parameters than their corresponding CNN models. We design a smoother learning strategy to achieve a stable training process and obtain performance improvements on OOD data by 4% from vanilla ViTs. We hope our comprehensive study could shed light on the design of more generalizable learning architectures. Codes and datasets are released in https://github.com/Phoenix1153/ViT_OOD_generalization.

----

## [705] MPViT: Multi-Path Vision Transformer for Dense Prediction

**Authors**: *Youngwan Lee, Jonghee Kim, Jeffrey Willette, Sung Ju Hwang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00714](https://doi.org/10.1109/CVPR52688.2022.00714)

**Abstract**:

Dense computer vision tasks such as object detection and segmentation require effective multi-scale feature representation for detecting or classifying objects or regions with varying sizes. While Convolutional Neural Networks (CNNs) have been the dominant architectures for such tasks, recently introduced Vision Transformers (ViTs) aim to replace them as a backbone. Similar to CNNs, ViTs build a simple multi-stage structure (i.e., fine-to-coarse) for multi-scale representation with single-scale patches. In this work, with a different perspective from existing Transformers, we explore multi-scale patch embedding and multi-path structure, constructing the Multi-Path Vision Transformer (MPViT). MPViT embeds features of the same size (i.e., sequence length) with patches of different scales simultaneously by using overlapping convolutional patch embedding. Tokens of different scales are then independently fed into the Transformer encoders via multiple paths and the resulting features are aggregated, enabling both fine and coarse feature representations at the same feature level. Thanks to the diverse, multi-scale feature representations, our MPViTs scaling from tiny (5M) to base (73M) consistently achieve superior performance over state-of-the-art Vision Transformers on ImageNet classification, object detection, instance segmentation, and semantic segmentation.

----

## [706] NFormer: Robust Person Re-identification with Neighbor Transformer

**Authors**: *Haochen Wang, Jiayi Shen, Yongtuo Liu, Yan Gao, Efstratios Gavves*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00715](https://doi.org/10.1109/CVPR52688.2022.00715)

**Abstract**:

Person re-identification aims to retrieve persons in highly varying settings across different cameras and scenarios, in which robust and discriminative representation learning is crucial. Most research considers learning representations from single images, ignoring any potential interactions between them. However, due to the high intraidentity variations, ignoring such interactions typically leads to outlier features. To tackle this issue, we propose a Neighbor Transformer Network, or NFormer, which explicitly models interactions across all input images, thus suppressing outlier features and leading to more robust representations overall. As modelling interactions between enormous amount of images is a massive task with lots of distractors, NFormer introduces two novel modules, the Landmark Agent Attention, and the Reciprocal Neighbor Softmax. Specifically, the Landmark Agent Attention efficiently models the relation map between images by a low-rank factorization with a few landmarks in feature space. Moreover, the Reciprocal Neighbor Softmax achieves sparse attention to relevant -rather than all- neighbors only, which alleviates interference of irrelevant representations and further relieves the computational burden. In experiments on four large-scale datasets, NFormer achieves a new state-of-the-art. The code is released at https://github.com/haochenheheda/NFormer.

----

## [707] Part-based Pseudo Label Refinement for Unsupervised Person Re-identification

**Authors**: *Yoonki Cho, Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00716](https://doi.org/10.1109/CVPR52688.2022.00716)

**Abstract**:

Unsupervised person re-identification (re-ID) aims at learning discriminative representations for person retrieval from unlabeled data. Recent techniques accomplish this task by using pseudo-labels, but these labels are inherently noisy and deteriorate the accuracy. To overcome this problem, several pseudo-label refinement methods have been proposed, but they neglect the fine-grained local context essential for person re-ID. In this paper, we propose a novel Part-based Pseudo Label Refinement (PPLR) framework that reduces the label noise by employing the complementary relationship between global and part features. Specifically, we design a cross agreement score as the similarity of k-nearest neighbors between feature spaces to exploit the reliable complementary relationship. Based on the cross agreement, we refine pseudo-labels of global features by ensembling the predictions of part features, which collectively alleviate the noise in global feature clustering. We further refine pseudo-labels of part features by applying label smoothing according to the suitability of given labels for each part. Thanks to the reliable complementary information provided by the cross agreement score, our PPLR effectively reduces the influence of noisy labels and learns discriminative representations with rich local contexts. Extensive experimental results on Market-1501 and MSMT17 demonstrate the effectiveness of the proposed method over the state-of-the-art performance. The code is available at https://github.com/yoonkicho/PPLR.

----

## [708] Temporal Complementarity-Guided Reinforcement Learning for Image-to-Video Person Re-Identification

**Authors**: *Wei Wu, Jiawei Liu, Kecheng Zheng, Qibin Sun, Zhengjun Zha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00717](https://doi.org/10.1109/CVPR52688.2022.00717)

**Abstract**:

Image-to-video person re-identification aims to retrieve the same pedestrian as the image-based query from a video-based gallery set. Existing methods treat it as a cross-modality retrieval task and learn the common latent embeddings from image and video modalities, which are both less effective and efficient due to large modality gap and redundant feature learning by utilizing all video frames. In this work, we first regard this task as point-to-set matching problem identical to human decision process, and propose a novel Temporal Complementarity-Guided Reinforcement Learning (TCRL) approach for image-to-video person re-identification. TCRL employs deep reinforcement learning to make sequential judgments on dynamically selecting suitable amount of frames from gallery videos, and accumulate adequate temporal complementary information among these frames by the guidance of the query image, towards balancing efficiency and accuracy. Specifically, TCRL formulates point-to-set matching procedure as Markov decision process, where a sequential judgement agent measures the uncertainty between the query image and all historical frames at each time step, and verifies that sufficient complementary clues are accumulated for judgment (same or different) or one more frames are requested to assist judgment. Moreover, TCRL maintains a sequential feature extraction module with complementary residual detectors to dynamically suppress redundant salient regions and thoroughly mine diverse complementary clues among these selected frames for enhancing frame-level representation. Extensive experiments demonstrate the superiority of our method.

----

## [709] Augmented Geometric Distillation for Data-Free Incremental Person ReID

**Authors**: *Yichen Lu, Mei Wang, Weihong Deng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00718](https://doi.org/10.1109/CVPR52688.2022.00718)

**Abstract**:

Incremental learning (IL) remains an open issue for Person Re-identification (ReID), where a ReID system is expected to preserve preceding knowledge while learning incrementally. However, due to the strict privacy licenses and the open-set retrieval setting, it is intractable to adapt existing class IL methods to ReID. In this work, we propose an Augmented Geometric Distillation (AGD) framework to tackle these issues. First, a general data-free incremental framework with dreaming memory is constructed to avoid privacy disclosure. On this basis, we reveal a “noisy distillation” problem stemming from the noise in dreaming memory, and further propose to augment distillation in a pairwise and cross-wise pattern over different views of memory to mitigate it. Second, for the open-set retrieval property, we propose to maintain feature space structure during evolving via a novel geometric way and preserve relationships between exemplars when representations drift. Extensive experiments demonstrate the superiority of our AGD to baseline with a margin of 6.0% mAP/7.9% R@1 and it could be generalized to class IL. Code is available here
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
†https://github.com/eddielyc/Augmented-Geometric-Distillation.

----

## [710] Salient-to-Broad Transition for Video Person Re-identification

**Authors**: *Shutao Bai, Bingpeng Ma, Hong Chang, Rui Huang, Xilin Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00719](https://doi.org/10.1109/CVPR52688.2022.00719)

**Abstract**:

Due to the limited utilization of temporal relations in video re-id, the frame-level attention regions of mainstream methods are partial and highly similar. To address this problem, we propose a Salient-to-Broad Module (SBM) to enlarge the attention regions gradually. Specifically, in SBM, while the previous frames have focused on the most salient regions, the later frames tend to focus on broader regions. In this way, the additional information in broad regions can supplement salient regions, incurring more powerful video-level representations. To further improve SBM, an Integration-and-Distribution Module (IDM) is introduced to enhance frame-level representations. IDM first integrates features from the entire feature space and then distributes the integrated features to each spatial location. SBM and IDM are mutually beneficial since they enhance the representations from video-level and frame-level, respectively. Extensive experiments on four prevalent benchmarks demonstrate the effectiveness and superiority of our method. The source code is available at https://github.com/baist/SINet.

----

## [711] FMCNet: Feature-Level Modality Compensation for Visible-Infrared Person Re-Identification

**Authors**: *Qiang Zhang, Changzhou Lai, Jianan Liu, Nianchang Huang, Jungong Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00720](https://doi.org/10.1109/CVPR52688.2022.00720)

**Abstract**:

For Visible-Infrared person ReIDentification (VI-ReID), existing modality-specific information compensation based models try to generate the images of missing modality from existing ones for reducing cross-modality discrepancy. However, because of the large modality discrepancy between visible and infrared images, the generated images usually have low qualities and introduce much more interfering information (e.g., color inconsistency). This greatly degrades the subsequent VI-ReID performance. Alternatively, we present a novel Feature-level Modality Compensation Network (FMCNet) for VI-ReID in this paper, which aims to compensate the missing modality-specific information in the feature level rather than in the image level, i.e., directly generating those missing modality-specific features of one modality from existing modality-shared features of the other modality. This will enable our model to mainly generate some discriminative person related modality-specific features and discard those non-discriminative ones for benefiting VI-ReID. For that, a single-modality feature decomposition module is first designed to decompose single-modality features into modality-specific ones and modality-shared ones. Then, a feature-level modality compensation module is present to generate those missing modality-specific features from existing modality-shared ones. Finally, a shared-specific feature fusion module is proposed to combine the existing and generated features for VI-ReID. The effectiveness of our proposed model is verified on two benchmark datasets.

----

## [712] Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification

**Authors**: *Shengcai Liao, Ling Shao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00721](https://doi.org/10.1109/CVPR52688.2022.00721)

**Abstract**:

Recent studies show that, both explicit deep feature matching as well as large-scale and diverse training data can significantly improve the generalization of person reidentification. However, the efficiency of learning deep matchers on large-scale data has not yet been adequately studied. Though learning with classification parameters or class memory is a popular way, it incurs large memory and computational costs. In contrast, pairwise deep metric learning within mini batches would be a better choice. However, the most popular random sampling method, the well-known PKsampler, is not informative and efficient for deep metric learning. Though online hard example mining has improved the learning efficiency to some extent, the mining in mini batches after random sampling is still limited. This inspires us to explore the use of hard example mining earlier, in the data sampling stage. To do so, in this paper, we propose an efficient mini-batch sampling method, called graph sampling (GS), for large-scale deep metric learning. The basic idea is to build a nearest neighbor relationship graph for all classes at the beginning of each epoch. Then, each mini batch is composed of a randomly selected class and its nearest neighboring classes so as to provide informative and challenging examples for learning. Together with an adapted competitive baseline, we improve the state of the art in generalizable person re-identification significantly, by 25.1% in Rank-1 on MSMT17 when trained on RandPerson. Besides, the proposed method also outperforms the competitive baseline, by 6.8% in Rank-1 on CUHK03-NP when trained on MSMT17. Meanwhile, the training time is significantly reduced, from 25.4 hours to 2 hours when trained on RandPerson with 8,000 identities. Code is available at https://github.com/ShengcaiLiao/QAConv.

----

## [713] Implicit Sample Extension for Unsupervised Person Re-Identification

**Authors**: *Xinyu Zhang, Dongdong Li, Zhigang Wang, Jian Wang, Errui Ding, Javen Qinfeng Shi, Zhaoxiang Zhang, Jingdong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00722](https://doi.org/10.1109/CVPR52688.2022.00722)

**Abstract**:

Most existing unsupervised person re-identification (Re-ID) methods use clustering to generate pseudo labels for model training. Unfortunately, clustering sometimes mixes different true identities together or splits the same identity into two or more sub clusters. Training on these noisy clusters substantially hampers the Re-ID accuracy. Due to the limited samples in each identity, we suppose there may lack some underlying information to well reveal the accurate clusters. To discover these information, we propose an Implicit Sample Extension (ISE) method to generate what we call support samples around the cluster boundaries. Specifically, we generate support samples from actual samples and their neighbouring clusters in the embedding space through a progressive linear interpolation (PLI) strategy. PLI controls the generation with two critical factors, i.e., 1) the direction from the actual sample towards its K-nearest clusters and 2) the degree for mixing up the context information from the K-nearest clusters. Meanwhile, given the support samples, ISE further uses a label-preserving loss to pull them towards their corresponding actual samples, so as to compact each cluster. Consequently, ISE reduces the “sub and mixed” clustering errors, thus improving the Re-ID performance. Extensive experiments demonstrate that the proposed method is effective and achieves state-of-the-art performance for unsupervised person Re-ID. Code is available at: https://github.com/PaddlePaddle/PaddleClas.

----

## [714] Rethinking Reconstruction Autoencoder-Based Out-of-Distribution Detection

**Authors**: *Yibo Zhou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00723](https://doi.org/10.1109/CVPR52688.2022.00723)

**Abstract**:

In some scenarios, classifier requires detecting out-of-distribution samples far from its training data. With desirable characteristics, reconstruction autoencoder-based methods deal with this problem by using input reconstruction error as a metric of novelty vs. normality. We formulate the essence of such approach as a quadruplet domain translation with an intrinsic bias to only query for a proxy of conditional data uncertainty. Accordingly, an improvement direction is formalized as maximumly compressing the autoencoder's latent space while ensuring its reconstructive power for acting as a described domain translator. From it, strategies are introduced including semantic reconstruction, data certainty decomposition and normalized L2 distance to substantially improve original methods, which together establish state-of-the-art performance on various benchmarks, e.g., the FPR@95%TPR of CIFAR-100 vs. TinyImagenet-crop on Wide-ResNet is 0.2%. Importantly, our method works without any additional data, hard-to-implement structure, time-consuming pipeline, and even harming the classification accuracy of known classes.

----

## [715] Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection

**Authors**: *Choubo Ding, Guansong Pang, Chunhua Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00724](https://doi.org/10.1109/CVPR52688.2022.00724)

**Abstract**:

Despite most existing anomaly detection studies assume the availability of normal training samples only, a few labeled anomaly examples are often available in many real-world applications, such as defect samples identified during random quality inspection, lesion images confirmed by radiologists in daily medical screening, etc. These anomaly examples provide valuable knowledge about the application-specific abnormality, enabling significantly improved detection of similar anomalies in some recent models. However, those anomalies seen during training often do not illustrate every possible class of anomaly, rendering these models ineffective in generalizing to unseen anomaly classes. This paper tackles open-set supervised anomaly detection, in which we learn detection models using the anomaly examples with the objective to detect both seen anomalies (‘gray swans’) and unseen anomalies (‘black swans’). We propose a novel approach that learns disentangled representations of abnormalities illustrated by seen anomalies, pseudo anomalies, and latent residual anomalies (i.e., samples that have unusual residuals compared to the normal data in a latent space), with the last two abnormalities designed to detect unseen anomalies. Extensive experiments on nine real-world anomaly detection datasets show superior performance of our model in detecting seen and unseen anomalies under diverse settings. Code and data are available at: https://github.com/choubo/DRA

----

## [716] Fine-Grained Object Classification via Self-Supervised Pose Alignment

**Authors**: *Xuhui Yang, Yaowei Wang, Ke Chen, Yong Xu, Yonghong Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00725](https://doi.org/10.1109/CVPR52688.2022.00725)

**Abstract**:

Semantic patterns offine-grained objects are determined by subtle appearance difference of local parts, which thus inspires a number of part-based methods. However, due to uncontrollable object poses in images, distinctive de-tails carried by local regions can be spatially distributed or even self-occluded, leading to a large variation on ob-ject representation. For discounting pose variations, this paper proposes to learn a novel graph based object rep-resentation to reveal a global configuration of local parts for self-supervised pose alignment across classes, which is employed as an auxiliary feature regularization on a deep representation learning network. Moreover, a coarse-to-fine supervision together with the proposed pose-insensitive constraint on shallow-to-deep sub-networks encourages discriminative features in a curriculum learning manner. We evaluate our method on three popular fine-grained ob-ject classification benchmarks, consistently achieving the state-of-the-art performance. Source codes are available at https://github.com/yangxhll/P2P-Net.

----

## [717] Hyperbolic Vision Transformers: Combining Improvements in Metric Learning

**Authors**: *Aleksandr Ermolov, Leyla Mirvakhabova, Valentin Khrulkov, Nicu Sebe, Ivan V. Oseledets*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00726](https://doi.org/10.1109/CVPR52688.2022.00726)

**Abstract**:

Metric learning aims to learn a highly discriminative model encouraging the embeddings of similar classes to be close in the chosen metrics and pushed apart for dissimilar ones. The common recipe is to use an encoder to extract embeddings and a distance-based loss function to match the representations - usually, the Euclidean distance is utilized. An emerging interest in learning hyperbolic data embeddings suggests that hyperbolic geometry can be beneficial for natural data. Following this line of work, we propose a new hyperbolic-based model for metric learning. At the core of our method is a vision transformer with output embeddings mapped to hyperbolic space. These embeddings are directly optimized using modified pairwise cross-entropy loss. We evaluate the proposed model with six different formulations on four datasets achieving the new state-of-the-art performance. The source code is available at https://github.com/htdt/hyp_metric.

----

## [718] Non-isotropy Regularization for Proxy-based Deep Metric Learning

**Authors**: *Karsten Roth, Oriol Vinyals, Zeynep Akata*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00727](https://doi.org/10.1109/CVPR52688.2022.00727)

**Abstract**:

Deep Metric Learning (DML) aims to learn representation spaces on which semantic relations can simply be expressed through predefined distance metrics. Best performing approaches commonly leverage class proxies as sample stand-ins for better convergence and generalization. However, these proxy-methods solely optimize for sample-proxy distances. Given the inherent non-bijectiveness of used distance functions, this can induce locally isotropic sample distributions, leading to crucial semantic context being missed due to difficulties resolving local structures and intraclass relations between samples. To alleviate this problem, we propose non-isotropy regularization 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(\mathbb{NIR})$</tex>
 for proxy-based Deep Metric Learning. By leveraging Normalizing Flows, we enforce unique translatability of samples from their respective class proxies. This allows us to explicitly induce a non-isotropic distribution of samples around a proxy to optimize for. In doing so, we equip proxy-based objectives to better learn local structures. Extensive experiments highlight consistent generalization benefits of NIR while achieving competitive and state-of-the-art performance on the standard benchmarks CUB200-2011, Cars196 and Stanford Online Products. In addition, we find the superior convergence properties of proxy-based methods to still be retained or even improved, making NIR very attractive for practical usage. Code available at github.com/ExplainableML/NonIsotropicProxyDML.

----

## [719] Self-Taught Metric Learning without Labels

**Authors**: *Sungyeon Kim, Dongwon Kim, Minsu Cho, Suha Kwak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00728](https://doi.org/10.1109/CVPR52688.2022.00728)

**Abstract**:

We present a novel self-taught framework for unsuper-vised metric learning, which alternates between predicting class-equivalence relations between data through a moving average of an embedding model and learning the model with the predicted relations as pseudo labels. At the heart of our framework lies an algorithm that investigates contexts of data on the embedding space to predict their class-equivalence relations as pseudo labels. The algorithm enables efficient end-to-end training since it demands no off-the-shelf module for pseudo labeling. Also, the class-equivalence relations provide rich supervisory signals for learning an embedding space. On standard benchmarks for metric learning, it clearly outperforms existing unsupervised learning methods and sometimes even beats supervised learning models using the same backbone network. It is also applied to semi-supervised metric learning as a way of exploiting additional unlabeled data, and achieves the state of the art by boosting performance of supervised learning substantially.

----

## [720] Not Just Selection, but Exploration: Online Class-Incremental Continual Learning via Dual View Consistency

**Authors**: *Yanan Gu, Xu Yang, Kun Wei, Cheng Deng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00729](https://doi.org/10.1109/CVPR52688.2022.00729)

**Abstract**:

Online class-incremental continual learning aims to learn new classes continually from a never-ending and single-pass data stream, while not forgetting the learned knowledge of old classes. Existing replay-based methods have shown promising performance by storing a subset of old class data. Unfortunately, these methods only focus on selecting samples from the memory bank for replay and ignore the adequate exploration of semantic information in the single-pass data stream, leading to poor classification accuracy. In this paper, we propose a novel yet effective framework for online class-incremental continual learning, which considers not only the selection of stored samples, but also the full exploration of the data stream. Specifically, we propose a gradient-based sample selection strategy, which selects the stored samples whose gradients generated in the network are most interfered by the new incoming samples. We believe such samples are beneficial for updating the neural network based on back gradient propagation. More importantly, we seek to explore the semantic information between two different views of training images by maximizing their mutual information, which is conducive to the improvement of classification accuracy. Extensive experimental results demonstrate that our method achieves state-of-the-art performance on a variety of benchmark datasets. Our code is available on https://github.com/YananGu/DVC.

----

## [721] Energy-based Latent Aligner for Incremental Learning

**Authors**: *K. J. Joseph, Salman Khan, Fahad Shahbaz Khan, Rao Muhammad Anwer, Vineeth N. Balasubramanian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00730](https://doi.org/10.1109/CVPR52688.2022.00730)

**Abstract**:

Deep learning models tend to forget their earlier knowledge while incrementally learning new tasks. This behavior emerges because the parameter updates optimized for the new tasks may not align well with the updates suitable for older tasks. The resulting latent representation mismatch causes forgetting. In this work, we propose ELI: Energy-based Latent Aligner for Incremental Learning, which first learns an energy manifold for the latent representations such that previous task latents will have low energy and the current task latents have high energy values. This learned manifold is used to counter the representational shift that happens during incremental learning. The implicit regularization that is offered by our proposed methodology can be used as a plug-and-play module in existing incremental learning methodologies. We validate this through extensive evaluation on CIFAR-100, ImageNet subset, ImageNet1k and Pascal VOC datasets. We observe consistent improvement when ELI is added to three prominent methodologies in class-incremental learning, across multiple incremental settings. Further, when added to the state-of-the-art incremental object detector, ELI provides over 5% improvement in detection accuracy, corroborating its effectiveness and complementary advantage to the existing art. Code is available at: https://github.com/JosephKJ/ELI.

----

## [722] Sketch3T: Test-Time Training for Zero-Shot SBIR

**Authors**: *Aneeshan Sain, Ayan Kumar Bhunia, Vaishnav Potlapalli, Pinaki Nath Chowdhury, Tao Xiang, Yi-Zhe Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00731](https://doi.org/10.1109/CVPR52688.2022.00731)

**Abstract**:

Zero-shot sketch-based image retrieval typically asks for a trained model to be applied as is to unseen categories. In this paper, we question to argue that this setup by definition is not compatible with the inherent abstract and subjective nature of sketches – the model might transfer well to new categories, but will not understand sketches existing in different test-time distribution as a result. We thus extend ZS-SBIR asking it to transfer to both categories and sketch distributions. Our key contribution is a test-time training paradigm that can adapt using just one sketch. Since there is no paired photo, we make use of a sketch raster-vector reconstruction module as a self-supervised auxiliary task. To maintain the fidelity of the trained cross-modal joint embedding during test-time update, we design a novel meta-learning based training paradigm to learn a separation between model updates incurred by this auxiliary task from those off the primary objective of discriminative learning. Extensive experiments show our model to outperform state-of-the-arts, thanks to the proposed test-time adaption that not only transfers to new categories but also accommodates to new sketching styles.

----

## [723] The Devil is in the Pose: Ambiguity-free 3D Rotation-invariant Learning via Pose-aware Convolution

**Authors**: *Ronghan Chen, Yang Cong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00732](https://doi.org/10.1109/CVPR52688.2022.00732)

**Abstract**:

Recent progress in introducing rotation invariance (RI) to 3D deep learning methods is mainly made by designing RI features to replace 3D coordinates as input. The key to this strategy lies in how to restore the global information that is lost by the input RI features. Most state-of-the-arts achieve this by incurring additional blocks or complex global representations, which is time-consuming and ineffective. In this paper, we real that the global information loss stems from an unexplored pose information loss problem, i.e., common convolution layers cannot capture the relative poses between RI features, thus hindering the global information to be hierarchically aggregated in the deep networks. To address this problem, we develop a Poseaware Rotation Invariant Convolution (i.e., PaRI-Conv), which dynamically adapts its kernels based on the relative poses. Specifically, in each PaRI-Conv layer, a lightweight Augmented Point Pair Feature (APPF) is designed to fully encode the RI relative pose information. Then, we propose to synthesize a factorized dynamic kernel, which reduces the computational cost and memory burden by decomposing it into a shared basis matrix and a pose-aware diagonal matrix that can be learned from the APPF. Extensive experiments on shape classification and part segmentation tasks show that our PaRI-Conv surpasses the state-of-the-art RI methods while being more compact and efficient.

----

## [724] Finding Badly Drawn Bunnies

**Authors**: *Lan Yang, Kaiyue Pang, Honggang Zhang, Yi-Zhe Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00733](https://doi.org/10.1109/CVPR52688.2022.00733)

**Abstract**:

As lovely as bunnies are, your sketched version would probably not do it justice (Fig. 1). This paper recognises this very problem and studies sketch quality measurement for the first time - letting you find these badly drawn ones. Our key discovery lies in exploiting the magnitude (
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$L$</tex>
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</inf>
 norm) of a sketch feature as a quantitative quality metric. We propose Geometry-Aware Classification Layer (GACL), a generic method that makes feature-magnitude-as-quality-metric possible and importantly does it without the need for specific quality annotations from humans. GACL sees feature magnitude and recognisability learning as a dual task, which can be simultaneously optimised under a neat crossentropy classification loss. GACL is lightweight with theoretic guarantees and enjoys a nice geometric interpretation to reason its success. We confirm consistent quality agreements between our GACL-induced metric and human perception through a carefully designed human study. Notably, we demonstrate three practical sketch applications enabled for the first time using our quantitative quality metric.

----

## [725] Generalized Category Discovery

**Authors**: *Sagar Vaze, Kai Han, Andrea Vedaldi, Andrew Zisserman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00734](https://doi.org/10.1109/CVPR52688.2022.00734)

**Abstract**:

In this paper, we consider a highly general image recognition setting wherein, given a labelled and unlabelled set of images, the task is to categorize all images in the unlabelled set. Here, the unlabelled images may come from labelled classes or from novel ones. Existing recognition methods are not able to deal with this setting, because they make several restrictive assumptions, such as the unlabelled instances only coming from known – or unknown – classes, and the number of unknown classes being known a-priori. We address the more unconstrained setting, naming it ‘Generalized Category Discovery’, and challenge all these assumptions. We first establish strong baselines by taking state-of-the-art algorithms from novel category discovery and adapting them for this task. Next, we propose the use of vision transformers with contrastive representation learning for this open-world setting. We then introduce a simple yet effective semi-supervised k-means method to cluster the unlabelled data into seen and unseen classes automatically, substantially outperforming the baselines. Finally, we also propose a new approach to estimate the number of classes in the unlabelled data. We thoroughly evaluate our approach on public datasets for generic object classification and on fine-grained datasets, leveraging the recent Semantic Shift Benchmark suite. Code: https://www.robots.ox.ac.uk/~vgg/research/gcd

----

## [726] Recall@k Surrogate Loss with Large Batches and Similarity Mixup

**Authors**: *Yash Patel, Giorgos Tolias, Jirí Matas*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00735](https://doi.org/10.1109/CVPR52688.2022.00735)

**Abstract**:

This work focuses on learning deep visual representation models for retrieval by exploring the interplay between a new loss function, the batch size, and a new regularization approach. Direct optimization, by gradient descent, of an evaluation metric, is not possible when it is nondifferentiable, which is the case for recall in retrieval. A differentiable surrogate loss for the recall is proposed in this work. Using an implementation that sidesteps the hardware constraints of the GPU memory, the method trains with a very large batch size, which is essential for metrics computed on the entire retrieval database. It is assisted by an efficient mixup regularization approach that operates on pairwise scalar similarities and virtually increases the batch size further. The suggested method achieves state-of-the-art performance in several image retrieval benchmarks when used for deep metric learning. For instance-level recognition, the method outperforms similar approaches that train using an approximation of average precision.

----

## [727] Modeling 3D Layout For Group Re-Identification

**Authors**: *Quan Zhang, Kaiheng Dang, Jian-Huang Lai, Zhan-Xiang Feng, Xiaohua Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00736](https://doi.org/10.1109/CVPR52688.2022.00736)

**Abstract**:

Group re-identification (GReID) attempts to correctly associate groups with the same members under different cameras. The main challenge is how to resist the membership and layout variations. Existing works attempt to incorporate layout modeling on the basis of appearance features to achieve robust group representations. However, layout ambiguity is introduced because these methods only consider the 2D layout on the imaging plane. In this paper, we overcome the above limitations by 3D layout modeling. Specifically, we propose a novel 3D transformer (3DT) that reconstructs the relative 3D layout relationship among members, then applies sampling and quantification to preset a series of layout tokens along three dimensions, and selects the corresponding tokens as layout features for each member. Furthermore, we build a synthetic GReID dataset, City1M, including 1.84M images, 45K persons and 11.5K groups with 3D annotations to alleviate data shortages and poor annotations. To the best of our knowledge, 3DT is the first work to address GReID with 3D perspective, and the City1M is the currently largest dataset. Several experiments show the superiority of our 3DT and City1M. Our project has been released on https://github.com/LinlyAC/City1M-dataset.

----

## [728] Causal Transportability for Visual Recognition

**Authors**: *Chengzhi Mao, Kevin Xia, James Wang, Hao Wang, Junfeng Yang, Elias Bareinboim, Carl Vondrick*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00737](https://doi.org/10.1109/CVPR52688.2022.00737)

**Abstract**:

Visual representations underlie object recognition tasks, but they often contain both robust and non-robust features. Our main observation is that image classifiers may perform poorly on out-of-distribution samples because spurious correlations between non-robust features and labels can be changed in a new environment. By analyzing procedures for out-of-distribution generalization with a causal graph, we show that standard classifiers fail because the association between images and labels is not transportable across settings. However, we then show that the causal effect, which severs all sources of confounding, remains invariant across domains. This motivates us to develop an algorithm to estimate the causal effect for image classification, which is transportable (i.e., invariant) across source and target environments. Without observing additional variables, we show that we can derive an estimand for the causal effect under empirical assumptions using representations in deep models as proxies. Theoretical analysis, empirical results, and visualizations show that our approach captures causal invariances and improves overall generalization.

----

## [729] Attributable Visual Similarity Learning

**Authors**: *Borui Zhang, Wenzhao Zheng, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00738](https://doi.org/10.1109/CVPR52688.2022.00738)

**Abstract**:

This paper proposes an attributable visual similarity learning (AVSL) framework for a more accurate and ex-plainable similarity measure between images. Most existing similarity learning methods exacerbate the unexplain-ability by mapping each sample to a single point in the em-bedding space with a distance metric (e.g., Mahalanobis distance, Euclidean distance). Motivated by the human se-mantic similarity cognition, we propose a generalized simi-larity learning paradigm to represent the similarity between two images with a graph and then infer the overall simi-larity accordingly. Furthermore, we establish a bottom-up similarity construction and top-down similarity inference framework to infer the similarity based on semantic hier-archy consistency. We first identify unreliable higher-level similarity nodes and then correct them using the most co-herent adjacent lower-level similarity nodes, which simulta-neously preserve traces for similarity attribution. Extensive experiments on the CUB-200-2011, Cars196, and Stanford Online Products datasets demonstrate significant improve-ments over existing deep similarity learning methods and verify the interpretability of our framework.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/zbr17/AVSL.

----

## [730] Bi-level Alignment for Cross-Domain Crowd Counting

**Authors**: *Shenjian Gong, Shanshan Zhang, Jian Yang, Dengxin Dai, Bernt Schiele*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00739](https://doi.org/10.1109/CVPR52688.2022.00739)

**Abstract**:

Recently, crowd density estimation has received increasing attention. The main challenge for this task is to achieve high-quality manual annotations on a large amount of training data. To avoid reliance on such annotations, previous works apply unsupervised domain adaptation (UDA) techniques by transferring knowledge learned from easily accessible synthetic data to real-world datasets. However, current state-of-the-art methods either rely on external data for training an auxiliary task or apply an expensive coarse-to-fine estimation. In this work, we aim to develop a new adversarial learning based method, which is simple and efficient to apply. To reduce the domain gap between the synthetic and real data, we design a bi-level alignment framework (BLA) consisting of (1) task-driven data alignment and (2) fine-grained feature alignment. In contrast to previous domain augmentation methods, we introduce AutoML to search for an optimal transform on source, which well serves for the downstream task. On the other hand, we do fine-grained alignment for foreground and background separately to alleviate the alignment difficulty. We evaluate our approach on five real-world crowd counting benchmarks, where we outperform existing approaches by a large margin. Also, our approach is simple, easy to implement and efficient to apply. The code is publicly available at https://github.com/Yankeegsj/BLA.

----

## [731] Mutual Quantization for Cross-Modal Search with Noisy Labels

**Authors**: *Erkun Yang, Dongren Yao, Tongliang Liu, Cheng Deng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00740](https://doi.org/10.1109/CVPR52688.2022.00740)

**Abstract**:

Deep cross-modal hashing has become an essential tool for supervised multimodal search. These models tend to be optimized with large, curated multimodal datasets, where most labels have been manually verified. Unfortunately, in many scenarios, such accurate labeling may not be avail-able. In contrast, datasets with low-quality annotations may be acquired, which inevitably introduce numerous mis-takes or label noise and therefore degrade the search per-formance. To address the challenge, we present a general robust cross-modal hashing framework to correlate distinct modalities and combat noisy labels simultaneously. More specifically, we propose a proxy-based contrastive (PC) loss to mitigate the gap between different modalities and train networks for different modalities Jointly with small-loss samples that are selected with the PC loss and a mu-tual quantization loss. The small-loss sample selection from such Joint loss can help choose confident examples to guide the model training, and the mutual quantization loss can maximize the agreement between different modalities and is beneficial to improve the effectiveness of sample selection. Experiments on three widely-used multimodal datasets show that our method significantly outperforms existing state-of-the-arts.

----

## [732] Task Adaptive Parameter Sharing for Multi-Task Learning

**Authors**: *Matthew Wallingford, Hao Li, Alessandro Achille, Avinash Ravichandran, Charless C. Fowlkes, Rahul Bhotika, Stefano Soatto*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00741](https://doi.org/10.1109/CVPR52688.2022.00741)

**Abstract**:

Adapting pre-trained models with broad capabilities has become standard practice for learning a wide range of downstream tasks. The typical approach of fine-tuning different models for each task is performant, but incurs a substantial memory cost. To efficiently learn multiple down-stream tasks we introduce Task Adaptive Parameter Sharing (TAPS), a simple method for tuning a base model to a new task by adaptively modifying a small, task-specific subset of layers. This enables multi-task learning while minimizing the resources used and avoids catastrophic forgetting and competition between tasks. TAPS solves a joint optimization problem which determines both the layers that are shared with the base model and the value of the task-specific weights. Further, a sparsity penalty on the number of active layers promotes weight sharing with the base model. Compared to other methods, TAPS retains a high accuracy on the target tasks while still introducing only a small number of task-specific parameters. Moreover, TAPS is agnostic to the particular architecture used and requires only minor changes to the training scheme. We evaluate our method on a suite of fine-tuning tasks and architectures (ResNet, DenseNet, ViT) and show that it achieves state-of-the-art performance while being simple to implement.

----

## [733] Simple Multi-dataset Detection

**Authors**: *Xingyi Zhou, Vladlen Koltun, Philipp Krähenbühl*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00742](https://doi.org/10.1109/CVPR52688.2022.00742)

**Abstract**:

How do we build a general and broad object detection system? We use all labels of all concepts ever annotated. These labels span diverse datasets with potentially inconsistent taxonomies. In this paper, we present a simple method for training a unified detector on multiple large-scale datasets. We use dataset-specific training protocols and losses, but share a common detection architecture with dataset-specific outputs. We show how to automatically integrate these dataset-specific outputs into a common semantic taxonomy. In contrast to prior work, our approach does not require manual taxonomy reconciliation. Experiments show our learned taxonomy outperforms a expert-designed taxonomy in all datasets. Our multi-dataset detector performs as well as dataset-specific models on each training domain, and can generalize to new unseen dataset without fine-tuning on them. Code is available at https://github.com/xingyizhou/UniDet.

----

## [734] Cross-Domain Adaptive Teacher for Object Detection

**Authors**: *Yu-Jhe Li, Xiaoliang Dai, Chih-Yao Ma, Yen-Cheng Liu, Kan Chen, Bichen Wu, Zijian He, Kris Kitani, Peter Vajda*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00743](https://doi.org/10.1109/CVPR52688.2022.00743)

**Abstract**:

We address the task of domain adaptation in object detection, where there is an obvious domain gap between a domain with annotations (source) and a domain of interest without annotations (target). As a popular semi-supervised learning method, the teacher-student framework (a student model is supervised by the pseudo labels from a teacher model) has also yielded a large accuracy gain in cross-domain object detection. However, it suffers from the domain shift and generates many low-quality pseudo labels (e.g., false positives), which leads to sub-optimal performance. To mitigate this problem, we propose a teacher-student framework named Adaptive Teacher (AT) which leverages domain adversarial learning and weak-strong data augmentation to address the domain gap. Specifically, we employ feature-level adversarial training in the student model, allowing features derived from the source and target domains to share similar distributions. This process ensures the student model produces domain-invariant features. Furthermore, we apply weak-strong augmentation and mutual learning between the teacher model (taking data from the target domain) and the student model (taking data from both domains). This enables the teacher model to learn the knowledge from the student model without being biased to the source domain. We show that AT demonstrates superiority over existing approaches and even Oracle (fully-supervised) models by a large margin. For example, we achieve 50.9% (49.3%) mAP on Foggy Cityscape (Cli-part1K), which is 9.2% (5.2%) and 8.2% (11.0%) higher than previous state-of-the-art and Oracle, respectively.

----

## [735] Balanced and Hierarchical Relation Learning for One-shot Object Detection

**Authors**: *Hanqing Yang, Sijia Cai, Hualian Sheng, Bing Deng, Jianqiang Huang, Xian-Sheng Hua, Yong Tang, Yu Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00744](https://doi.org/10.1109/CVPR52688.2022.00744)

**Abstract**:

Instance-level feature matching is significantly important to the success of modern one-shot object detectors. Re-cently, the methods based on the metric-learning paradigm have achieved an impressive process. Most of these works only measure the relations between query and target objects on a single level, resulting in suboptimal performance overall. In this paper, we introduce the balanced and hierarchical learning for our detector. The contributions are two-fold: firstly, a novel Instance-level Hierarchical Relation (IHR) module is proposed to encode the contrastive-level, salient-level, and attention-level relations simultane-ously to enhance the query-relevant similarity representation. Secondly, we notice that the batch training of the IHR module is substantially hindered by the positive-negative sample imbalance in the one-shot scenario. We then in-troduce a simple but effective Ratio-Preserving Loss (RPL) to protect the learning of rare positive samples and sup-press the effects of negative samples. Our loss can adjust the weight for each sample adaptively, ensuring the desired positive-negative ratio consistency and boosting query-related IHR learning. Extensive experiments show that our method outperforms the state-of-the-art method by 1.6% and 1.3% on PASCAL VOC and MS COCO datasets for unseen classes, respectively. The code will be available at https://github.com/hero-y/BHRL.

----

## [736] Semantic-aligned Fusion Transformer for One-shot Object Detection

**Authors**: *Yizhou Zhao, Xun Guo, Yan Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00745](https://doi.org/10.1109/CVPR52688.2022.00745)

**Abstract**:

One-shot object detection aims at detecting novel objects according to merely one given instance. With extreme data scarcity, current approaches explore various feature fusions to obtain directly transferable meta-knowledge. Yet, their performances are often unsatisfactory. In this paper, we attribute this to inappropriate correlation methods that misalign query-support semantics by overlooking spatial structures and scale variances. Upon analysis, we leverage the attention mechanism and propose a simple but effective architecture named Semantic-aligned Fusion Transformer (SaFT) to resolve these issues. Specifically, we equip SaFT with a vertical fusion module (VFM) for cross-scale semantic enhancement and a horizontal fusion module (HFM) for cross-sample feature fusion. Together, they broaden the vision for each feature point from the support to a whole augmented feature pyramid from the query, facilitating semantic-aligned associations. Extensive experiments on multiple benchmarks demonstrate the superiority of our framework. Without fine-tuning on novel classes, it brings significant performance gains to one-stage baselines, lifting state-of-the-art results to a higher level.

----

## [737] MSDN: Mutually Semantic Distillation Network for Zero-Shot Learning

**Authors**: *Shiming Chen, Ziming Hong, Guo-Sen Xie, Wenhan Yang, Qinmu Peng, Kai Wang, Jian Zhao, Xinge You*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00746](https://doi.org/10.1109/CVPR52688.2022.00746)

**Abstract**:

The key challenge of zero-shot learning (ZSL) is how to infer the latent semantic knowledge between visual and attribute features on seen classes, and thus achieving a desirable knowledge transfer to unseen classes. Prior works either simply align the global features of an image with its associated class semantic vector or utilize unidirectional attention to learn the limited latent semantic representations, which could not effectively discover the intrinsic semantic knowledge (e.g., attribute semantics) between visual and attribute features. To solve the above dilemma, we propose a Mutually Semantic Distillation Network (MSDN), which progressively distills the intrinsic semantic representations between visual and attribute features for ZSL. MSDN incorporates an attribute→visual attention sub-net that learns attribute-based visual features, and a visual→attribute attention sub-net that learns visual-based attribute features. By further introducing a semantic distillation loss, the two mutual attention sub-nets are capable of learning collaboratively and teaching each other throughout the training process. The proposed MSDN yields significant improvements over the strong baselines, leading to new state-of-the-art performances on three popular challenging benchmarks. Our codes have been available at: https://github.com/shiming-chen/MSDN.

----

## [738] Robust Region Feature Synthesizer for Zero-Shot Object Detection

**Authors**: *Peiliang Huang, Junwei Han, De Cheng, Dingwen Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00747](https://doi.org/10.1109/CVPR52688.2022.00747)

**Abstract**:

Zero-shot object detection aims at incorporating class semantic vectors to realize the detection of (both seen and) unseen classes given an unconstrained test image. In this study, we reveal the core challenges in this research area: how to synthesize robust region features (for unseen objects) that are as intra-class diverse and inter-class separable as the real samples, so that strong unseen object detectors can be trained upon them. To address these challenges, we build a novel zero-shot object detection framework that contains an Intra-class Semantic Diverging component and an Inter-class Structure Preserving component. The former is used to realize the one-to-more mapping to obtain diverse visual features from each class semantic vector, preventing miss-classifying the real unseen objects as image backgrounds. While the latter is used to avoid the synthesized features too scattered to mix up the inter-class and foreground-background relationship. To demonstrate the effectiveness of the proposed approach, comprehensive experiments on PASCAL VOC, COCO, and DIOR datasets are conducted. Notably, our approach achieves the new state-of-the-art performance on PASCAL VOC and COCO and it is the first study to carry out zero-shot object detection in remote sensing imagery.

----

## [739] Region-Aware Face Swapping

**Authors**: *Chao Xu, Jiangning Zhang, Miao Hua, Qian He, Zili Yi, Yong Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00748](https://doi.org/10.1109/CVPR52688.2022.00748)

**Abstract**:

This paper presents a novel Region-Aware Face Swapping (RAFSwap) network to achieve identity-consistent harmonious high-resolution face generation in a local-global manner: 1) Local Facial Region-Aware (FRA) branch augments local identity-relevant features by introducing the Transformer to effectively model misaligned crossscale semantic interaction. 2) Global Source Feature-Adaptive (SFA) branch further complements global identity-relevant cues for generating identity-consistent swapped faces. Besides, we propose a Face Mask Predictor (FMP) module incorporated with StyleGAN2 to predict identity-relevant soft facial masks in an unsupervised manner that is more practical for generating harmonious high-resolution faces. Abundant experiments qualitatively and quantitatively demonstrate the superiority of our method for generating more identity-consistent high-resolution swapped faces over SOTA methods, e.g., obtaining 96.70 ID retrieval that outperforms SOTA MegaFS by 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$5.87\uparrow$</tex>
.

----

## [740] High-resolution Face Swapping via Latent Semantics Disentanglement

**Authors**: *Yangyang Xu, Bailin Deng, Junle Wang, Yanqing Jing, Jia Pan, Shengfeng He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00749](https://doi.org/10.1109/CVPR52688.2022.00749)

**Abstract**:

We present a novel high-resolution face swapping method using the inherent prior knowledge of a pre-trained GAN model. Although previous research can leverage generative priors to produce high-resolution results, their quality can suffer from the entangled semantics of the latent space. We explicitly disentangle the latent semantics by utilizing the progressive nature of the generator, deriving structure at-tributes from the shallow layers and appearance attributes from the deeper ones. Identity and pose information within the structure attributes are further separated by introducing a landmark-driven structure transfer latent direction. The disentangled latent code produces rich generative features that incorporate feature blending to produce a plausible swapping result. We further extend our method to video face swapping by enforcing two spatio-temporal constraints on the latent space and the image space. Extensive experiments demonstrate that the proposed method outperforms state-of-the-art image/video face swapping methods in terms of hallucination quality and consistency. Code can be found at: https://github.com/cnnlstm/FSLSD_HiRes.

----

## [741] Rethinking Deep Face Restoration

**Authors**: *Yang Zhao, Yu-Chuan Su, Chun-Te Chu, Yandong Li, Marius Renn, Yukun Zhu, Changyou Chen, Xuhui Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00750](https://doi.org/10.1109/CVPR52688.2022.00750)

**Abstract**:

A model that can authentically restore a low-quality face image to a high-quality one can benefit many applications. While existing approaches for face restoration make significant progress in generating high-quality faces, they often fail to preserve facial features that compromise the authenticity of reconstructed faces. Because the human visual system is very sensitive to faces, even minor changes may significantly degrade the perceptual quality. In this work, we argue that the problems of existing models can be traced down to the two sub-tasks of the face restoration problem, i.e. face generation and face reconstruction, and the fragile balance between them. Based on the observation, we propose a new face restoration model that improves both generation and reconstruction. Besides the model improvement, we also introduce a new evaluation metric for measuring models' ability to preserve the identity in the restored faces. Extensive experiments demonstrate that our model achieves state-of-the-art performance on multiple face restoration benchmarks, and the proposed metric has a higher correlation with user preference. The user study shows that our model produces higher quality faces while better preserving the identity 86.4% of the time compared with state-of-the-art methods.

----

## [742] Blind Face Restoration via Integrating Face Shape and Generative Priors

**Authors**: *Feida Zhu, Junwei Zhu, Wenqing Chu, Xinyi Zhang, Xiaozhong Ji, Chengjie Wang, Ying Tai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00751](https://doi.org/10.1109/CVPR52688.2022.00751)

**Abstract**:

Blind face restoration, which aims to reconstruct high-quality images from low-quality inputs, can benefit many applications. Although existing generative-based methods achieve significant progress in producing high-quality images, they often fail to restore natural face shapes and high-fidelity facial details from severely-degraded inputs. In this work, we propose to integrate shape and generative priors to guide the challenging blind face restoration. Firstly, we set up a shape restoration module to recover reason-able facial geometry with 3D reconstruction. Secondly, a pretrained facial generator is adopted as decoder to generate photo-realistic high-resolution images. To ensure high-fidelity, hierarchical spatial features extracted from the low-quality inputs and rendered 3D images are inserted into the decoder with our proposed Adaptive Feature Fusion Block (AFFB). Moreover, we introduce hybrid-level losses to Jointly train the shape and generative priors together with other network parts such that these two priors better adapt to our blind face restoration task. The proposed Shape and Generative Prior integrated Network (SGPN) can re-store high-quality images with clear face shapes and real-istic facial details. Experimental results on synthetic and real-world datasets demonstrate SGPN performs favorably against state-of-the-art blind face restoration methods.

----

## [743] FENeRF: Face Editing in Neural Radiance Fields

**Authors**: *Jingxiang Sun, Xuan Wang, Yong Zhang, Xiaoyu Li, Qi Zhang, Yebin Liu, Jue Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00752](https://doi.org/10.1109/CVPR52688.2022.00752)

**Abstract**:

Previous portrait image generation methods roughly fall into two categories: 2D GANs and 3D-aware GANs. 2D GANs can generate high fidelity portraits but with low view consistency. 3D-aware GAN methods can maintain view consistency but their generated images are not locally editable. To overcome these limitations, we propose FENeRF, a 3D-aware generator that can produce view-consistent and locally-editable portrait images. Our method uses two decoupled latent codes to generate corresponding facial semantics and texture in a spatial-aligned 3D volume with shared geometry. Benefiting from such underlying 3D representation, FENeRF can Jointly render the boundary-aligned image and semantic mask and use the semantic mask to edit the 3D volume via GAN inversion. We further show such 3D representation can be learned from widely available monocular image and semantic mask pairs. Moreover, we reveal that Joint learning semantics and texture helps to generate finer geometry. Our experiments demonstrate that FENeRF outperforms state-of-the-art methods in various face editing tasks. Code is available at https://github.com/MrTornado24/FENeRF.

----

## [744] TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing

**Authors**: *Yanbo Xu, Yueqin Yin, Liming Jiang, Qianyi Wu, Chengyao Zheng, Chen Change Loy, Bo Dai, Wayne Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00753](https://doi.org/10.1109/CVPR52688.2022.00753)

**Abstract**:

Recent advances like StyleGAN have promoted the growth of controllable facial editing. To address its core challenge of attribute decoupling in a single latent space, attempts have been made to adopt dual-space GAN for better disentanglement of style and content representations. Nonetheless, these methods are still incompetent to obtain plausible editing results with high controllability, especially for complicated attributes. In this study, we highlight the importance of interaction in a dual-space GAN for more controllable editing. We propose TransEditor, a novel Transformer-based framework to enhance such interaction. Besides, we develop a new dual-space editing and inversion strategy to provide additional editing flexibility. Extensive experiments demonstrate the superiority of the proposed framework in image quality and editing capability, suggesting the effectiveness of TransEditor for highly controllable facial editing. Code and models are publicly available at https://github.com/BillyXYB/TransEditor.

----

## [745] Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer

**Authors**: *Shuai Yang, Liming Jiang, Ziwei Liu, Chen Change Loy*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00754](https://doi.org/10.1109/CVPR52688.2022.00754)

**Abstract**:

Recent studies on StyleGAN show high performance on artistic portrait generation by transfer learning with limited data. In this paper, we explore more challenging exemplar-based high-resolution portrait style transfer by introducing a novel DualStyleGAN with flexible control of dual styles of the original face domain and the extended artistic portrait domain. Different from StyleGAN, DualStyleGAN provides a natural way of style transfer by characterizing the content and style of a portrait with an intrinsic style path and a new extrinsic style path, respectively. The del-icately designed extrinsic style path enables our model to modulate both the color and complex structural styles hierarchically to precisely pastiche the style example. Furthermore, a novel progressive fine-tuning scheme is introduced to smoothly transform the generative space of the model to the target domain, even with the above modifications on the network architecture. Experiments demonstrate the superiority of DualStyleGAN over state-of-the-art methods in high-quality portrait style transfer and flexible stylecontrol. Code is available at https://github.com/williamyang1991/DualStyleGAN.

----

## [746] Self-supervised Correlation Mining Network for Person Image Generation

**Authors**: *Zijian Wang, Xingqun Qi, Kun Yuan, Muyi Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00755](https://doi.org/10.1109/CVPR52688.2022.00755)

**Abstract**:

Person image generation aims to perform non-rigid deformation on source images, which generally requires unaligned data pairs for training. Recently, self-supervised methods express great prospects in this task by merging the disentangled representations for self-reconstruction. However, such methods fail to exploit the spatial correlation between the disentangled features. In this paper, we propose a Self-supervised Correlation Mining Network (SCM-Net) to rearrange the source images in the feature space, in which two collaborative modules are integrated, Decomposed Style Encoder (DSE) and Correlation Mining Module (CMM). Specifically, the DSE first creates unaligned pairs at the feature level. Then, the CMM establishes the spatial correlation field for feature rearrangement. Eventually, a translation module transforms the rearranged features to realistic results. Meanwhile, for improving the fidelity of cross-scale pose transformation, we propose a graph based Body Structure Retaining Loss (BSR Loss) to preserve reasonable body structures on half body to full body generation. Extensive experiments conducted on DeepFashion dataset demonstrate the superiority of our method compared with other supervised and unsupervised approaches. Furthermore, satisfactory results on face generation show the versatility of our method in other deformation tasks.

----

## [747] Exploring Dual-task Correlation for Pose Guided Person Image Generation

**Authors**: *Pengze Zhang, Lingxiao Yang, Jianhuang Lai, Xiaohua Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00756](https://doi.org/10.1109/CVPR52688.2022.00756)

**Abstract**:

Pose Guided Person Image Generation (PGPIG) is the task of transforming a person image from the source pose to a given target pose. Most of the existing methods only focus on the ill-posed source-to-target task and fail to capture reasonable texture mapping. To address this problem, we propose a novel Dual-task Pose Transformer Network (DPTN), which introduces an auxiliary task (i.e., source-to-source task) and exploits the dual-task correlation to promote the performance of PGPIG. The DPTN is of a Siamese structure, containing a source-to-source self-reconstruction branch, and a transformation branch for source-to-target generation. By sharing partial weights between them, the knowledge learned by the source-to-source task can effectively assist the source-to-target learning. Furthermore, we bridge the two branches with a proposed Pose Transformer Module (PTM) to adaptively explore the correlation between features from dual tasks. Such correlation can establish the fine-grained mapping of all the pixels between the sources and the targets, and promote the source texture transmission to enhance the details of the generated target images. Extensive experiments show that our DPTN outperforms state-of-the-arts in terms of both PSNR and LPIPS. In addition, our DPTN only contains 9.79 million parameters, which is significantly smaller than other approaches. Our code is available at: https://github.com/PangzeCheung/Dual-task-Pose-Transformer-Network.

----

## [748] InsetGAN for Full-Body Image Generation

**Authors**: *Anna Frühstück, Krishna Kumar Singh, Eli Shechtman, Niloy J. Mitra, Peter Wonka, Jingwan Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00757](https://doi.org/10.1109/CVPR52688.2022.00757)

**Abstract**:

While GANs can produce photo-realistic images in ideal conditions for certain domains, the generation of full-body human images remains difficult due to the diversity of identities, hairstyles, clothing, and the variance in pose. In-stead of modeling this complex domain with a single GAN, we propose a novel method to combine multiple pretrained GANs, where one GAN generates a global canvas (e.g., human body) and a set of specialized GANs, or insets, focus on different parts (e.g., faces, shoes) that can be seamlessly inserted onto the global canvas. We model the problem as jointly exploring the respective latent spaces such that the generated images can be combined, by inserting the parts from the specialized generators onto the global canvas, without introducing seams. We demonstrate the setup by combining a full body GAN with a dedicated high-quality face GAN to produce plausible-looking humans. We evalu-ate our results with quantitative metrics and user studies.

----

## [749] BodyGAN: General-purpose Controllable Neural Human Body Generation

**Authors**: *Chaojie Yang, Hanhui Li, Shengjie Wu, Shengkai Zhang, Haonan Yan, Nianhong Jiao, Jie Tang, Runnan Zhou, Xiaodan Liang, Tianxiang Zheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00758](https://doi.org/10.1109/CVPR52688.2022.00758)

**Abstract**:

Recent advances in generative adversarial networks (GANs) have provided potential solutions for photo-realistic human image synthesis. However, the explicit and individual control of synthesis over multiple factors, such as poses, body shapes, and skin colors, remains difficult for existing methods. This is because current methods mainly rely on a single pose/appearance model, which is limited in dis-entangling various poses and appearance in human images. In addition, such a unimodal strategy is prone to causing severe artifacts in the generated images like color distortions and unrealistic textures. To tackle these issues, this paper proposes a multi-factor conditioned method dubbed BodyGAN. Specifically, given a source image, our Body-GAN aims at capturing the characteristics of the human body from multiple aspects: (i) A pose encoding branch consisting of three hybrid subnetworks is adopted, to generate the semantic segmentation based representation, the 3D surface based representation, and the key point based rep-resentation of the human body, respectively. (ii) Based on the segmentation results, an appearance encoding branch is used to obtain the appearance information of the human body parts. (iii) The outputs of these two branches are represented by user-editable condition maps, which are then processed by a generator to predict the synthesized image. In this way, our BodyGAN can achieve the fine-grained dis-entanglement of pose, body shape, and appearance, and consequently enable the explicit and effective control of syn-thesis with diverse conditions. Extensive experiments on multiple datasets and a comprehensive user study show that our BodyGAN achieves the state-of-the-art performance.

----

## [750] HumanNeRF: Efficiently Generated Human Radiance Field from Sparse Inputs

**Authors**: *Fuqiang Zhao, Wei Yang, Jiakai Zhang, Pei Lin, Yingliang Zhang, Jingyi Yu, Lan Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00759](https://doi.org/10.1109/CVPR52688.2022.00759)

**Abstract**:

Recent neural human representations can produce high-quality multi-view rendering but require using dense multi-view inputs and costly training. They are hence largely limited to static models as training each frame is infeasible. We present HumanNeRF - a neural representation with efficient generalization ability - for high-fidelity free-view synthesis of dynamic humans. Analogous to how IBRNet assists NeRF by avoiding perscene training, HumanNeRF employs an aggregated pixel-alignment feature across multi-view inputs along with a pose embedded non-rigid deformation field for tackling dynamic motions. The raw Human-NeRF can already produce reasonable rendering on sparse video inputs of unseen subjects and camera settings. To further improve the rendering quality, we augment our solution with in-hour scene-specific fine-tuning, and an appearance blending module for combining the benefits of both neural volumetric rendering and neural texture blending. Extensive experiments on various multi-view dynamic hu-man datasets demonstrate effectiveness of our approach in synthesizing photo-realistic free-view humans under challenging motions and with very sparse camera view inputs.

----

## [751] Structure-Aware Flow Generation for Human Body Reshaping

**Authors**: *Jianqiang Ren, Yuan Yao, Biwen Lei, Miaomiao Cui, Xuansong Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00760](https://doi.org/10.1109/CVPR52688.2022.00760)

**Abstract**:

Body reshaping is an important procedure in portrait photo retouching. Due to the complicated structure and multifarious appearance of human bodies, existing methods either fall back on the 3D domain via body morphable model or resort to keypoint-based image deformation, leading to inefficiency and unsatisfied visual quality. In this paper, we address these limitations by formulating an end-to-end flow generation architecture under the guidance of body structural priors, including skeletons and Part Affinity Fields, and achieve unprecedentedly controllable performance under arbitrary poses and garments. A compositional attention mechanism is introduced for capturing both visual perceptual correlations and structural associations of the human body to reinforce the manipulation consistency among related parts. For a comprehensive evaluation, we construct the first large-scale body reshaping dataset, namely BR-5K, which contains 5,000 portrait photos as well as professionally retouched targets. Extensive experiments demonstrate that our approach significantly outperforms existing state-of-the-art methods in terms of visual performance, controllability, and efficiency. The dataset is available at our website: https://github.com/JianqiangRen/FlowBasedBodyReshaping.

----

## [752] Modeling Image Composition for Complex Scene Generation

**Authors**: *Zuopeng Yang, Daqing Liu, Chaoyue Wang, Jie Yang, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00761](https://doi.org/10.1109/CVPR52688.2022.00761)

**Abstract**:

We present a method that achieves state-of-the-art results on challenging (few-shot) layout-to-image generation tasks by accurately modeling textures, structures and relationships contained in a complex scene. After compressing RGB images into patch tokens, we propose the Transformer with Focal Attention (TwFA) for exploring dependencies of object-to-object, object-to-patch and patch-to-patch. Compared to existing CNN-based and Transformer-based generation models that entangled modeling on pixel-level&patch-level and object-level&patch-level respectively, the proposed focal attention predicts the current patch token by only focusing on its highly-related tokens that specified by the spatial layout, thereby achieving disambiguation during training. Furthermore, the proposed TwFA largely increases the data efficiency during training, therefore we propose the first few-shot complex scene generation strategy based on the well-trained TwFA. Comprehensive experiments show the superiority of our method, which significantly increases both quantitative metrics and qualitative visual realism with respect to state-of-the-art CNN-based and transformer-based methods. Code is available at https://github.com/JohnDreamer/TwFA.

----

## [753] Local Attention Pyramid for Scene Image Generation

**Authors**: *Sang-Heon Shim, Sangeek Hyun, Dae Hyun Bae, Jae-Pil Heo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00762](https://doi.org/10.1109/CVPR52688.2022.00762)

**Abstract**:

In this paper, we first investigate the class-wise visual quality imbalance problem of scene images generated by GANs. The tendency is empirically found that the class-wise visual qualities are highly correlated with the dominance of object classes in the training data in terms of their scales and appearance frequencies. Specifically, the synthesized qualities of small and less frequent object classes tend to be low. To address this, we propose a novel attention module, Local Attention Pyramid (LAP) module tailored for scene image synthesis, that encourages GANs to generate diverse object classes in a high quality by explicit spread of high attention scores to local regions, since objects in scene images are scattered over the entire images. Moreover, our LAP assigns attention scores in a multiple scale to reflect the scale diversity of various objects. The experimental evaluations on three different datasets show consistent improvements in Frechet Inception Distance (FID) and Frechet Segmentation Distance (FSD) over the state-of-the-art baselines. Furthermore, we apply our LAP module to various GANs methods to demonstrate a wide applicability of our LAP module.

----

## [754] Interactive Image Synthesis with Panoptic Layout Generation

**Authors**: *Bo Wang, Tao Wu, Minfeng Zhu, Peng Du*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00763](https://doi.org/10.1109/CVPR52688.2022.00763)

**Abstract**:

Interactive image synthesis from user-guided input is a challenging task when users wish to control the scene structure of a generated image with ease. Although remarkable progress has been made on layout-based image synthesis approaches, existing methods require high-precision inputs such as accurately placed bounding boxes, which might be constantly violated in an interactive setting. When placement of bounding boxes is subject to perturbation, layout-based models suffer from “missing regions” in the constructed semantic layouts and hence undesirable artifacts in the generated images. In this work, we propose Panoptic Layout Generative Adversarial Network (PLGAN) to address this challenge. The PLGAN employs panoptic theory which distinguishes object categories between “stuff” with amorphous boundaries and “things” with well-defined shapes, such that stuff and instance layouts are constructed through separate branches and later fused into panoptic layouts. In particular, the stuff layouts can take amorphous shapes and fill up the missing regions left out by the instance layouts. We experimentally compare our PLGAN with state-of-the-art layout-based models on the COCO-Stuff, Visual Genome, and Landscape datasets. The advantages of PLGAN are not only visually demonstrated but quantitatively verified in terms of inception score, Fréchet inception distance, classification accuracy score, and coverage. The code is available at https://github.com/wb-finalking/PLGAN.

----

## [755] iPLAN: Interactive and Procedural Layout Planning

**Authors**: *Feixiang He, Yanlong Huang, He Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00764](https://doi.org/10.1109/CVPR52688.2022.00764)

**Abstract**:

Layout design is ubiquitous in many applications, e.g. architecture/urban planning, etc, which involves a lengthy iterative design process. Recently, deep learning has been leveraged to automatically generate layouts via image generation, showing a huge potential to free designers from laborious routines. While automatic generation can greatly boost productivity, designer input is undoubtedly crucial. An ideal AI-aided design tool should automate repetitive routines, and meanwhile accept human guidance and provide smart/proactive suggestions. However, the capability of involving humans into the loop has been largely ignored in existing methods which are mostly end-to-end approaches. To this end, we propose a new human-in-the-loop generative model, iPLAN, which is capable of automatically generating layouts, but also interacting with designers throughout the whole procedure, enabling humans and AI to co-evolve a sketchy idea gradually into the final design. iPLAN is evaluated on diverse datasets and compared with existing methods. The results show that iPLAN has high fidelity in producing similar layouts to those from human designers, great flexibility in accepting designer inputs and providing design suggestions accordingly, and strong generalizability when facing unseen design tasks and limited training data.

----

## [756] E-CIR: Event-Enhanced Continuous Intensity Recovery

**Authors**: *Chen Song, Qixing Huang, Chandrajit Bajaj*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00765](https://doi.org/10.1109/CVPR52688.2022.00765)

**Abstract**:

A camera begins to sense light the moment we press the shutter button. During the exposure interval, relative motion between the scene and the camera causes motion blur, a common undesirable visual artifact. This paper presents E-CIR, which converts a blurry image into a sharp video represented as a parametric function from time to intensity. E-CIR leverages events as an auxiliary input. We discuss how to exploit the temporal event structure to construct the parametric bases. We demonstrate how to train a deep learning model to predict the function coefficients. To improve the appearance consistency, we further introduce a refinement module to propagate visual features among consecutive frames. Compared to state-of-the-art event-enhanced de-blurring approaches, E-CIR generates smoother and more realistic results. The implementation of E-CIR is available at https://github.com/chensong1995/E-CIR.

----

## [757] Learning Robust Image-Based Rendering on Sparse Scene Geometry via Depth Completion

**Authors**: *Yuqi Sun, Shili Zhou, Ri Cheng, Weimin Tan, Bo Yan, Lang Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00766](https://doi.org/10.1109/CVPR52688.2022.00766)

**Abstract**:

Recent image-based rendering (IBR) methods usually adopt plenty of views to reconstruct dense scene geometry. However, the number of available views is limited in prac-tice. When only few views are provided, the performance of these methods drops off significantly, as the scene geometry becomes sparse as well. Therefore, in this paper, we propose Sparse-IBRNet (SIBRNet) to perform robust IBR on sparse scene geometry by depth completion. The SIBR-Net has two stages, geometry recovery (GR) stage and light blending (LB) stage. Specifically, GR stage takes sparse depth map and RGB as input to predict dense depth map by exploiting the correlation between two modals. As in-accuracy of the complete depth map may cause projection biases in the warping process, LB stage first uses a bias-corrected module (BCM) to rectify deviations, and then ag-gregates modified features from different views to render a novel view. Extensive experimental results demonstrate that our method performs best on sparse scene geometry than re-cent IBR methods, and it can generate better or comparable results as well when the geometric information is dense.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>

----

## [758] Neural Rays for Occlusion-aware Image-based Rendering

**Authors**: *Yuan Liu, Sida Peng, Lingjie Liu, Qianqian Wang, Peng Wang, Christian Theobalt, Xiaowei Zhou, Wenping Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00767](https://doi.org/10.1109/CVPR52688.2022.00767)

**Abstract**:

We present a new neural representation, called Neural Ray (NeuRay), for the novel view synthesis task. Recent works construct radiance fields from image features of input views to render novel view images, which enables the generalization to new scenes. However, due to occlusions, a 3D point may be invisible to some input views. On such a 3D point, these generalization methods will include inconsistent image features from invisible views, which interfere with the radiance field construction. To solve this problem, we predict the visibility of 3D points to input views within our NeuRay representation. This visibility enables the radiance field construction to focus on visible image features, which significantly improves its rendering quality. Meanwhile, a novel consistency loss is proposed to refine the visibility in NeuRay when finetuning on a specific scene. Experiments demonstrate that our approach achieves state-of-the-art performance on the novel view synthesis task when generalizing to unseen scenes and outperforms perscene optimization methods after finetuning. Project page:https://liuyuan-pal.github.io/NeuRay/

----

## [759] Industrial Style Transfer with Large-scale Geometric Warping and Content Preservation

**Authors**: *Jinchao Yang, Fei Guo, Shuo Chen, Jun Li, Jian Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00768](https://doi.org/10.1109/CVPR52688.2022.00768)

**Abstract**:

We propose a novel style transfer method to quickly create a new visual product with a nice appearance for industrial designers reference. Given a source product, a target product, and an art style image, our method produces a neural warping field that warps the source shape to imitate the geometric style of the target and a neural texture transformation network that transfers the artistic style to the warped source product. Our model, Industrial Style Transfer (InST), consists of large-scale geometric warping (LGW) and interest-consistency texture transfer (ICTT). LGW aims to explore an unsupervised transformation between the shape masks of the source and target products for fitting large-scale shape warping. Furthermore, we introduce a mask smoothness regularization term to prevent the abrupt changes of the details of the source product. ICTT introduces an interest regularization term to maintain important contents of the warped product when it is stylized by using the art style image. Extensive experimental results demonstrate that InST achieves state-of-the-art performance on multiple visual product design tasks, e.g., companies' snail logos and classical bottles (please see Fig. 1). To the best of our knowledge, we are the first to extend the neural style transfer method to create industrial product appearances. Code is available at https://jcyang98.github.io/InST/home.html

----

## [760] PCA-Based Knowledge Distillation Towards Lightweight and Content-Style Balanced Photorealistic Style Transfer Models

**Authors**: *Tai-Yin Chiu, Danna Gurari*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00769](https://doi.org/10.1109/CVPR52688.2022.00769)

**Abstract**:

Photorealistic style transfer entails transferring the style of a reference image to another image so the result seems like a plausible photo. Our work is inspired by the ob-servation that existing models are slow due to their large sizes. We introduce PCA-based knowledge distillation to distill lightweight models and show it is motivated by the-ory. To our knowledge, this is the first knowledge dis-tillation method for photorealistic style transfer. Our ex-periments demonstrate its versatility for use with differ-ent backbone architectures, VGG and MobileNet, across six image resolutions. Compared to existing models, our top-performing model runs at speeds 5-20x faster using at most 1% of the parameters. Additionally, our dis-tilled models achieve a better balance between stylization strength and content preservation than existing models. To support reproducing our method and models, we share the code at https://github.com/chiutaiyin/PCA-Knowledge-Distillation.

----

## [761] Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data

**Authors**: *Kyungjune Baek, Hyunjung Shim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00770](https://doi.org/10.1109/CVPR52688.2022.00770)

**Abstract**:

Transfer learning for GANs successfully improves generation performance under low-shot regimes. However, existing studies show that the pretrained model using a single benchmark dataset is not generalized to various target datasets. More importantly, the pretrained model can be vulnerable to copyright or privacy risks as membership inference attack advances. To resolve both issues, we propose an effective and unbiased data synthesizer, namely Primitives - PS, inspired by the generic characteristics of natural images. Specifically, we utilize 1) the generic statistics on the frequency magnitude spectrum, 2) the elementary shape (i.e., image composition via elementary shapes) for representing the structure information, and 3) the existence of saliency as prior. Since our synthesizer only considers the generic properties of natural images, the single model pretrained on our dataset can be consistently transferred to various target datasets, and even outperforms the previous methods pretrained with the natural images in terms of Fréchet inception distance. Extensive analysis, ablation study, and evaluations demonstrate that each component of our data synthesizer is effective, and provide insights on the desirable nature of the pretrained model for the transferability of GANs.

----

## [762] Think Twice Before Detecting GAN-generated Fake Images from their Spectral Domain Imprints

**Authors**: *Chengdong Dong, Ajay Kumar, Eryun Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00771](https://doi.org/10.1109/CVPR52688.2022.00771)

**Abstract**:

Accurate detection of the fake but photorealistic images is one of the most challenging tasks to address social, biometrics security and privacy related concerns in our community. Earlier research has underlined the existence of spectral domain artifacts in fake images generated by powerful generative adversarial network (GAN) based methods. Therefore, a number of highly accurate frequency domain methods to detect such GAN generated images have been proposed in the literature. Our study in this paper introduces a pipeline to mitigate the spectral artifacts. We show from our experiments that the artifacts in frequency spectrum of such fake images can be mitigated by proposed methods, which leads to the sharp decrease of performance of spectrum-based detectors. This paper also presents experimental results using a large database of images that are synthesized using BigGAN, CRN, CycleGAN, IMLE, Pro-GAN, StarGAN, StyleGAN and StyleGAN2 (including synthesized high resolution fingerprint images) to illustrate effectiveness of the proposed methods. Furthermore, we select a spatial-domain based fake image detector and observe a notable decrease in the detection performance when proposed method is incorporated. In summary, our insightful analysis and pipeline presented in this paper cautions the forensic community on the reliability of GAN-generated fake image detectors that are based on the analysis of frequency artifacts as these artifacts can be easily mitigated.

----

## [763] Robust Invertible Image Steganography

**Authors**: *Youmin Xu, Chong Mou, Yujie Hu, Jingfen Xie, Jian Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00772](https://doi.org/10.1109/CVPR52688.2022.00772)

**Abstract**:

Image steganography aims to hide secret images into a container image, where the secret is hidden from human vision and can be restored when necessary. Previous image steganography methods are limited in hiding capacity and robustness, commonly vulnerable to distortion on container images such as Gaussian noise, Poisson noise, and lossy compression. This paper presents a novelflow-basedframe-work for robust invertible image steganography, dubbed as RIIS. A conditional normalizing flow is introduced to model the distribution of the redundant high-frequency component with the condition of the container image. Moreover, a well-designed container enhancement module (CEM) also contributes to the robust reconstruction. To regulate the net-work parameters for different distortion levels, a distortion-guided modulation (DGM) is implemented over flow-based blocks to make it a one-size-fits-all model. In terms of both clean and distorted image steganography, extensive experi-ments reveal that the proposed RIIS efficiently improves the robustness while maintaining imperceptibility and capacity. As far as we know, we are the first to propose a learning-based scheme to enhance the robustness of image steganog-raphy in the literature. The guarantee of steganography ro-bustness significantly broadens the application of steganog-raphy in real-world applications.

----

## [764] Distinguishing Unseen from Seen for Generalized Zero-shot Learning

**Authors**: *Hongzu Su, Jingjing Li, Zhi Chen, Lei Zhu, Ke Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00773](https://doi.org/10.1109/CVPR52688.2022.00773)

**Abstract**:

Generalized zero-shot learning (GZSL) aims to recognize samples whose categories may not have been seen at training. Recognizing unseen classes as seen ones or vice versa often leads to poor performance in GZSL. Therefore, distinguishing seen and unseen domains is naturally an effective yet challenging solution for GZSL. In this paper, we present a novel method which leverages both visual and semantic modalities to distinguish seen and unseen categories. Specifically, our method deploys two variational autoencoders to generate latent representations for visual and semantic modalities in a shared latent space, in which we align latent representations of both modalities by Wasserstein distance and reconstruct two modalities with the representations of each other. In order to learn a clearer boundary between seen and unseen classes, we propose a two-stage training strategy which takes advantage of seen and unseen semantic descriptions and searches a threshold to separate seen and unseen visual samples. At last, a seen expert and an unseen expert are used for final classification. Extensive experiments on five widely used benchmarks verify that the proposed method can significantly improve the results of GZSL. For instance, our method correctly recognizes more than 99% samples when separating domains and improves the final classification accuracy from 72.6% to 82.9% on AWA1.

----

## [765] Few-Shot Font Generation by Learning Fine-Grained Local Styles

**Authors**: *Licheng Tang, Yiyang Cai, Jiaming Liu, Zhibin Hong, Mingming Gong, Minhu Fan, Junyu Han, Jingtuo Liu, Errui Ding, Jingdong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00774](https://doi.org/10.1109/CVPR52688.2022.00774)

**Abstract**:

Few-shot font generation (FFG), which aims to generate a new font with a few examples, is gaining increasing attention due to the significant reduction in labor cost. A typical FFG pipeline considers characters in a standard font library as content glyphs and transfers them to a new target font by extracting style information from the reference glyphs. Most existing solutions explicitly disentangle content and style of reference glyphs globally or component-wisely. However, the style of glyphs mainly lies in the local details, i.e. the styles of radicals, components, and strokes together depict the style of a glyph. Therefore, even a single character can contain different styles distributed over spatial locations. In this paper, we propose a new font generation approach by learning 1) the fine-grained local styles from references, and 2) the spatial correspondence between the content and reference glyphs. Therefore, each spatial location in the content glyph can be assigned with the right fine-grained style. To this end, we adopt cross-attention over the representation of the content glyphs as the queries and the representations of the reference glyphs as the keys and values. Instead of explicitly disentangling global or component-wise modeling, the cross-attention mechanism can attend to the right local styles in the reference glyphs and aggregate the reference styles into a fine-grained style representation for the given content glyphs. The experiments show that the proposed method outperforms the state-of-the-art methods in FFG. In particular, the user studies also demonstrate the style consistency of our approach significantly outperforms previous methods.

----

## [766] XMP-Font: Self-Supervised Cross-Modality Pre-training for Few-Shot Font Generation

**Authors**: *Wei Liu, Fangyue Liu, Fei Ding, Qian He, Zili Yi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00775](https://doi.org/10.1109/CVPR52688.2022.00775)

**Abstract**:

Generating a new font library is a very labor-intensive and time-consuming job for glyph-rich scripts. Few-shot font generation is thus required, as it requires only a few glyph references without fine-tuning during test. Existing methods follow the style-content disentanglement paradigm and expect novel fonts to be produced by combining the style codes of the reference glyphs and the content representations of the source. However, these few-shot font generation methods either fail to capture content-independent style representations, or employ localized component-wise style representations, which is insufficient to model many Chinese font styles that involve hyper-component features such as inter-component spacing and “connected-stroke”. To resolve these drawbacks and make the style representations more reliable, we propose a self-supervised cross-modality pre-training strategy and a cross-modality transformer-based encoder that is conditioned jointly on the glyph image and the corresponding stroke labels. The cross-modality encoder is pre-trained in a self-supervised manner to allow effective capture of cross- and intra-modality correlations, which facilitates the content-style disentanglement and modeling style representations of all scales (strokelevel, component-level and character-level). The pretrained encoder is then applied to the downstream font generation task without fine-tuning. Experimental comparisons of our method with state-of-the-art methods demonstrate our method successfully transfers styles of all scales. In addition, it only requires one reference glyph and achieves the lowest rate of bad cases in the few-shot font generation task (28% lower than the second best).

----

## [767] Learning to generate line drawings that convey geometry and semantics

**Authors**: *Caroline Chan, Frédo Durand, Phillip Isola*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00776](https://doi.org/10.1109/CVPR52688.2022.00776)

**Abstract**:

This paper presents an unpaired method for creating line drawings from photographs. Current methods often rely on high quality paired datasets to generate line drawings. However, these datasets often have limitations due to the subjects of the drawings belonging to a specific domain, or in the amount of data collected. Although recent work in unsupervised image-to-image translation has shown much progress, the latest methods still struggle to generate compelling line drawings. We observe that line drawings are encodings of scene information and seek to convey 3D shape and semantic meaning. We build these observations into a set of objectives and train an image translation to map photographs into line drawings. We introduce a geometry loss which predicts depth information from the image features of a line drawing, and a semantic loss which matches the CLIP features of a line drawing with its corresponding photograph. Our approach outperforms state-of-the-art un-paired image translation and line drawing generation methods on creating line drawings from arbitrary photographs.

----

## [768] Balanced MSE for Imbalanced Visual Regression

**Authors**: *Jiawei Ren, Mingyuan Zhang, Cunjun Yu, Ziwei Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00777](https://doi.org/10.1109/CVPR52688.2022.00777)

**Abstract**:

Data imbalance exists ubiquitously in real-world visual regressions, e.g., age estimation and pose estimation, hurting the model's generalizability and fairness. Thus, im-balanced regression gains increasing research attention recently. Compared to imbalanced classification, imbalanced regression focuses on continuous labels, which can be boundless and high-dimensional and hence more challenging. In this work, we identify that the widely used Mean Square Error (MSE) loss function can be ineffective in imbalanced regression. We revisit MSE from a statistical view and propose a novel loss function, Balanced MSE, to accommodate the imbalanced training label distribution. We further design multiple implementations of Balanced MSE to tackle different real-world scenarios, particularly including the one that requires no prior knowledge about the training label distribution. Moreover, to the best of our knowledge, Balanced MSE is the first general solution to high-dimensional imbalanced regression in modern context. Extensive experiments on both synthetic and three real-world benchmarks demonstrate the effectiveness of Balanced MSE. Code and models are available at github.com/jiawei-ren/BalancedMSE.

----

## [769] Transferability Metrics for Selecting Source Model Ensembles

**Authors**: *Andrea Agostinelli, Jasper R. R. Uijlings, Thomas Mensink, Vittorio Ferrari*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00778](https://doi.org/10.1109/CVPR52688.2022.00778)

**Abstract**:

We address the problem of ensemble selection in transfer learning: Given a large pool of source models we want to select an ensemble of models which, after fine-tuning on the target training set, yields the best performance on the target test set. Since fine-tuning all possible ensembles is computationally prohibitive, we aim at predicting performance on the target dataset using a computationally efficient transferability metric. We propose several new transferability metrics designed for this task and evaluate them in a challenging and realistic transfer learning setup for semantic segmentation: we create a large and diverse pool of source models by considering 17 source datasets covering a wide variety of image domain, two different architectures, and two pre-training schemes. Given this pool, we then automatically select a subset to form an ensemble performing well on a given target dataset. We compare the ensemble selected by our method to two baselines which select a single source model, either (1) from the same pool as our method; or (2) from a pool containing large source models, each with similar capacity as an ensemble. Averaged over 17 target datasets, we outperform these baselines by 6.0% and 2.5% relative mean IoU, respectively.

----

## [770] OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization

**Authors**: *Nanyang Ye, Kaican Li, Haoyue Bai, Runpeng Yu, Lanqing Hong, Fengwei Zhou, Zhenguo Li, Jun Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00779](https://doi.org/10.1109/CVPR52688.2022.00779)

**Abstract**:

Deep learning has achieved tremendous success with independent and identically distributed (i. i.d.) data. However, the performance of neural networks often degenerates drastically when encountering out-of-distribution (OoD) data, i.e., when training and test data are sampled from different distributions. While a plethora of algorithms have been proposed for OoD generalization, our understanding of the data used to train and evaluate these algorithms remains stagnant. In this work, we first identify and measure two distinct kinds of distribution shifts that are ubiquitous in various datasets. Next, through extensive experiments, we compare OoD generalization algorithms across two groups of benchmarks, each dominated by one of the distribution shifts, revealing their strengths on one shift as well as limitations on the other shift. Overall, we position existing datasets and algorithms from different research areas seemingly unconnected into the same coherent picture. It may serve as a foothold that can be resorted to by future OoD generalization research. Our code is available at https://github.com/ynysjtulood_bench.

----

## [771] Robust fine-tuning of zero-shot models

**Authors**: *Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, Ludwig Schmidt*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00780](https://doi.org/10.1109/CVPR52688.2022.00780)

**Abstract**:

Large pre-trained models such as CLIP or ALIGN offer consistent accuracy across a range of data distributions when performing zero-shot inference (i.e., without fine-tuning on a specific dataset). Although existing fine-tuning methods substantially improve accuracy on a given target distribution, they often reduce robustness to distribution shifts. We address this tension by introducing a simple and effective method for improving robustness while fine-tuning: ensembling the weights of the zero-shot and fine-tuned models (WiSE-FT). Compared to standard fine-tuning, WiSE-FT provides large accuracy improvements under distribution shift, while preserving high accuracy on the target distribution. On ImageNet and five derived distribution shifts, WiSE-FT improves accuracy under distribution shift by 4 to 6 percentage points (pp) over prior work while increasing ImageNet accuracy by 1.6 pp. WiSE-FT achieves similarly large robustness gains (2 to 23 pp) on a diverse set of six further distribution shifts, and accuracy gains of 0.8 to 3.3 pp compared to standard fine-tuning on commonly used transfer learning datasets. These improvements come at no additional computational cost during fine-tuning or inference.

----

## [772] Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification

**Authors**: *Jiangtao Xie, Fei Long, Jiaming Lv, Qilong Wang, Peihua Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00781](https://doi.org/10.1109/CVPR52688.2022.00781)

**Abstract**:

Few-shot classification is a challenging problem as only very few training examples are given for each new task. One of the effective research lines to address this challenge focuses on learning deep representations driven by a similarity measure between a query image and few support images of some class. Statistically, this amounts to measure the dependency of image features, viewed as random vectors in a high-dimensional embedding space. Previous methods either only use marginal distributions without considering joint distributions, suffering from limited representation capability, or are computationally expensive though harnessing joint distributions. In this paper, we propose a deep Brownian Distance Covariance (DeepBDC) method for few-shot classification. The central idea of DeepBDC is to learn image representations by measuring the discrepancy between joint characteristic functions of embedded features and product of the marginals. As the BDC metric is decoupled, we formulate it as a highly modular and efficient layer. Furthermore, we instantiate DeepBDC in two different few-shot classification frameworks. We make experiments on six standard few-shot image benchmarks, covering general object recognition, fine-grained categorization and cross-domain classification. Extensive evaluations show our DeepBDC significantly outperforms the counterparts, while establishing new state-of-the-art results. The source code is available at http://www.peihuali.org/DeepBDC.

----

## [773] Learning to Learn and Remember Super Long Multi-Domain Task Sequence

**Authors**: *Zhenyi Wang, Li Shen, Tiehang Duan, Donglin Zhan, Le Fang, Mingchen Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00782](https://doi.org/10.1109/CVPR52688.2022.00782)

**Abstract**:

Catastrophic forgetting (CF) frequently occurs when learning with non-stationary data distribution. The CF issue remains nearly unexplored and is more challenging when meta-learning on a sequence of domains (datasets), called sequential domain meta-learning (SDML). In this work, we propose a simple yet effective learning to learn approach, i.e., meta optimizer, to mitigate the CF problem in SDML. We first apply the proposed meta optimizer to the simplified setting of SDML, domain-aware meta-learning, where the domain labels and boundaries are known during the learning process. We propose dynamically freezing the network and incorporating it with the proposed meta optimizer by considering the domain nature during meta training. In addition, we extend the meta optimizer to the more general setting of SDML, domain-agnostic meta-learning, where domain labels and boundaries are unknown during the learning process. We propose a domain shift detection technique to capture latent domain change and equip the meta optimizer with it to work in this setting. The proposed meta optimizer is versatile and can be easily integrated with several existing meta-learning algorithms. Finally, we construct a challenging and large-scale benchmark consisting of 10 heterogeneous domains with a super long task sequence consisting of 100K tasks. We perform extensive experiments on the proposed benchmark for both settings and demonstrate the effectiveness of our proposed method, outperforming current strong baselines by a large margin.

----

## [774] Learning Distinctive Margin toward Active Domain Adaptation

**Authors**: *Ming Xie, Yuxi Li, Yabiao Wang, Zekun Luo, Zhenye Gan, Zhongyi Sun, Mingmin Chi, Chengjie Wang, Pei Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00783](https://doi.org/10.1109/CVPR52688.2022.00783)

**Abstract**:

Despite plenty of efforts focusing on improving the domain adaptation ability (DA) under unsupervised or few-shot semi-supervised settings, recently the solution of active learning started to attract more attention due to its suitability in transferring model in a more practical way with limited annotation resource on target data. Nevertheless, most active learning methods are not inherently designed to handle domain gap between data distribution, on the other hand, some active domain adaptation methods (ADA) usually requires complicated query functions, which is vulnerable to overfitting. In this work, we propose a concise but effective ADA method called Select-by-Distinctive-Margin (SDM), which consists of a maximum margin loss and a margin sampling algorithm for data selection. We provide theoretical analysis to show that SDM works like a Support Vector Machine, storing hard examples around decision boundaries and exploiting them to find informative and transferable data. In addition, we propose two variants of our method, one is designed to adaptively adjust the gradient from margin loss, the other boosts the selectivity of margin sampling by taking the gradient direction into account. We benchmark SDM with standard active learning setting, demonstrating our algorithm achieves competitive results with good data scalability. Code is available at https://github.com/TencentYoutuResearch/ActiveLearning-SDM

----

## [775] DINE: Domain Adaptation from Single and Multiple Black-box Predictors

**Authors**: *Jian Liang, Dapeng Hu, Jiashi Feng, Ran He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00784](https://doi.org/10.1109/CVPR52688.2022.00784)

**Abstract**:

To ease the burden of labeling, unsupervised domain adaptation (UDA) aims to transfer knowledge in previous and related labeled datasets (sources) to a new unlabeled dataset (target). Despite impressive progress, prior methods always need to access the raw source data and develop data-dependent alignment approaches to recognize the target samples in a transductive learning manner, which may raise privacy concerns from source individuals. Several recent studies resort to an alternative solution by exploiting the well-trained white-box model from the source domain, yet, it may still leak the raw data via generative adversarial learning. This paper studies a practical and interesting setting for UDA, where only black-box source models (i.e., only network predictions are available) are provided during adaptation in the target domain. To solve this problem, we propose a new two-step knowledge adaptation framework called DIstill and fine-tuNE (DINE). Taking into consideration the target data structure, DINE first distills the knowledge from the source predictor to a customized target model, then fine-tunes the distilled model to further fit the target domain. Besides, neural networks are not required to be identical across domains in DINE, even allowing effective adaptation on a low-resource device. Empirical results on three UDA scenarios (i.e., single-source, multisource, and partial-set) confirm that DINE achieves highly competitive performance compared to state-of-the-art data-dependent approaches. Code is available at https://github.com/tim-learn/DINE/.

----

## [776] Source-Free Object Detection by Learning to Overlook Domain Style

**Authors**: *Shuaifeng Li, Mao Ye, Xiatian Zhu, Lihua Zhou, Lin Xiong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00785](https://doi.org/10.1109/CVPR52688.2022.00785)

**Abstract**:

Source-free object detection (SFOD) needs to adapt a detector pre-trained on a labeled source domain to a tar-get domain, with only unlabeled training data from the tar-get domain. Existing SFOD methods typically adopt the pseudo labeling paradigm with model adaption alternating between predicting pseudo labels and fine-tuning the model. This approach suffers from both unsatisfactory accuracy of pseudo labels due to the presence of domain shift and lim-ited use of target domain training data. In this work, we present a novel Learning to Overlook Domain Style (LODS) method with such limitations solved in a principled man-ner. Our idea is to reduce the domain shift effect by en-forcing the model to overlook the target domain style, such that model adaptation is simplified and becomes easier to carry on. To that end, we enhance the style of each tar-get domain image and leverage the style degree difference between the original image and the enhanced image as a self-supervised signal for model adaptation. By treating the enhanced image as an auxiliary view, we exploit a student- teacher architecture for learning to overlook the style de-gree difference against the original image, also character-ized with a novel style enhancement algorithm and graph alignment constraint. Extensive experiments demonstrate that our LODS yields new state-of-the-art performance on four benchmarks.

----

## [777] Towards Principled Disentanglement for Domain Generalization

**Authors**: *Hanlin Zhang, Yi-Fan Zhang, Weiyang Liu, Adrian Weller, Bernhard Schölkopf, Eric P. Xing*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00786](https://doi.org/10.1109/CVPR52688.2022.00786)

**Abstract**:

A fundamental challenge for machine learning models is generalizing to out-of-distribution (OOD) data, in part due to spurious correlations. To tackle this challenge, we first formalize the OOD generalization problem as constrained optimization, called Disentanglement-constrained Domain Generalization (DDG). We relax this non-trivial constrained optimization problem to a tractable form with finite-dimensional parameterization and empirical approxi-mation. Then a theoretical analysis of the extent to which the above transformations deviates from the original problem is provided. Based on the transformation, we propose a primal-dual algorithm for joint representation disentanglement and domain generalization. In contrast to traditional approaches based on domain adversarial training and domain labels, DDG jointly learns semantic and variation encoders for disentanglement, enabling flexible manipulation and augmentation on training data. DDG aims to learn intrinsic representations of semantic concepts that are invariant to nuisance factors and generalizable across domains. Comprehensive experiments on popular benchmarks show that DDG can achieve competitive OOD performance and uncover interpretable salient structures within data.

----

## [778] Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization

**Authors**: *Yabin Zhang, Minghan Li, Ruihuang Li, Kui Jia, Lei Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00787](https://doi.org/10.1109/CVPR52688.2022.00787)

**Abstract**:

Arbitrary style transfer (AST) and domain generalization (DG) are important yet challenging visual learning tasks, which can be cast as a feature distribution matching problem. With the assumption of Gaussian feature distribution, conventional feature distribution matching methods usually match the mean and standard deviation of features. However, the feature distributions of real-world data are usually much more complicated than Gaussian, which cannot be accurately matched by using only the first-order and second-order statistics, while it is computationally prohibitive to use high-order statistics for distribution matching. In this work, we, for the first time to our best knowledge, propose to perform Exact Feature Distribution Matching (EFDM) by exactly matching the empirical Cumulative Distribution Functions (eCDFs) of image features, which could be implemented by applying the Exact Histogram Matching (EHM) in the image feature space. Particularly, a fast EHM algorithm, named Sort-Matching, is employed to perform EFDM in a plug-and-play manner with minimal cost. The effectiveness of our proposed EFDM method is verified on a variety of AST and DG tasks, demonstrating new state-of-the-art results. Codes are available at https://github.com/YBZh/EFDM.

----

## [779] Causality Inspired Representation Learning for Domain Generalization

**Authors**: *Fangrui Lv, Jian Liang, Shuang Li, Bin Zang, Chi Harold Liu, Ziteng Wang, Di Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00788](https://doi.org/10.1109/CVPR52688.2022.00788)

**Abstract**:

Domain generalization (DG) is essentially an out-of-distribution problem, aiming to generalize the knowledge learned from multiple source domains to an unseen target domain. The mainstream is to leverage statistical models to model the dependence between data and labels, intending to learn representations independent of domain. Nevertheless, the statistical models are superficial descriptions of reality since they are only required to model dependence instead of the intrinsic causal mechanism. When the dependence changes with the target distribution, the statistic models may fail to generalize. In this regard, we introduce a general structural causal model to formalize the DG problem. Specifically, we assume that each input is constructed from a mix of causal factors (whose relationship with the label is invariant across domains) and non-causal factors (category-independent), and only the former cause the classification judgments. Our goal is to extract the causal factors from inputs and then reconstruct the invariant causal mechanisms. However, the theoretical idea is far from practical of DG since the required causal/non-causal factors are unobserved. We highlight that ideal causal factors should meet three basic properties: separated from the non-causal ones, jointly independent, and causally sufficient for the classification. Based on that, we propose a Causality Inspired Representation Learning (CIRL) algorithm that enforces the representations to satisfy the above properties and then uses them to simulate the causal factors, which yields improved generalization ability. Extensive experimental results on several widely used datasets verify the effectiveness of our approach. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at “https://github.com/BIT-DA/CIRL”.

----

## [780] Learning What Not to Segment: A New Perspective on Few-Shot Segmentation

**Authors**: *Chunbo Lang, Gong Cheng, Binfei Tu, Junwei Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00789](https://doi.org/10.1109/CVPR52688.2022.00789)

**Abstract**:

Recently few-shot segmentation (FSS) has been extensively developed. Most previous works strive to achieve generalization through the meta-learning framework derived from classification tasks; however, the trained models are biased towards the seen classes instead of being ideally class-agnostic, thus hindering the recognition of new concepts. This paper proposes a fresh and straightforward insight to alleviate the problem. Specifically, we apply an additional branch (base learner) to the conventional FSS model (meta learner) to explicitly identify the targets of base classes, i.e., the regions that do not need to be segmented. Then, the coarse results output by these two learners in parallel are adaptively integrated to yield precise segmentation prediction. Considering the sensitivity of meta learner, we further introduce an adjustment factor to estimate the scene differences between the input image pairs for facilitating the model ensemble forecasting. The substantial performance gains on PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 and COCO-20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 verify the effectiveness, and surprisingly, our versatile scheme sets a new state-of-the-art even with two plain learners. Moreover, in light of the unique nature of the proposed approach, we also extend it to a more realistic but challenging setting, i.e., generalized FSS, where the pixels of both base and novel classes are required to be determined. The source code is available at github.com/chunbolang/BAM.

----

## [781] Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation

**Authors**: *Binhui Xie, Longhui Yuan, Shuang Li, Chi Harold Liu, Xinjing Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00790](https://doi.org/10.1109/CVPR52688.2022.00790)

**Abstract**:

Self-training has greatly facilitated domain adaptive semantic segmentation, which iteratively generates pseudo labels on unlabeled target data and retrains the network. However, realistic segmentation datasets are highly imbalanced, pseudo labels are typically biased to the majority classes and basically noisy, leading to an error-prone and suboptimal model. In this paper, we propose a simple region-based active learning approach for semantic segmentation under a domain shift, aiming to automatically query a small partition of image regions to be labeled while maximizing segmentation performance. Our algorithm, Region Impurity and Prediction Uncertainty (RIPU), introduces a new acquisition strategy characterizing the spatial adjacency of image regions along with the prediction confidence. We show that the proposed region-based selection strategy makes more efficient use of a limited budget than image-based or point-based counterparts. Further, we enforce local prediction consistency between a pixel and its nearest neighbors on a source image. Alongside, we develop a negative learning loss to make the features more discriminative. Extensive experiments demonstrate that our method only requires very few annotations to almost reach the supervised performance and substantially outperforms state-of-the-art methods. The code is available at https://github.com/BIT-DA/RIPU.

----

## [782] ADeLA: Automatic Dense Labeling with Attention for Viewpoint Shift in Semantic Segmentation

**Authors**: *Hanxiang Ren, Yanchao Yang, He Wang, Bokui Shen, Qingnan Fan, Youyi Zheng, C. Karen Liu, Leonidas J. Guibas*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00791](https://doi.org/10.1109/CVPR52688.2022.00791)

**Abstract**:

We describe a method to deal with performance drop in semantic segmentation caused by viewpoint changes within multi-camera systems, where temporally paired images are readily available, but the annotations may only be abundant for a few typical views. Existing methods alleviate performance drop via domain alignment in a shared space and assume that the mapping from the aligned space to the output is transferable. However, the novel content induced by viewpoint changes may nullify such a space for effective alignments, thus resulting in negative adaptation. Our method works without aligning any statistics of the images between the two domains. Instead, it utilizes a novel attention-based view transformation network trained only on color images to hallucinate the semantic images for the target. Despite the lack of supervision, the view transformation network can still generalize to semantic images thanks to the induced “information transport” bias. Furthermore, to resolve ambiguities in converting the semantic images to semantic labels, we treat the view transformation network as a functional representation of an unknown mapping implied by the color images and propose functional label hallucination to generate pseudo-labels with uncertainties in the target domains. Our method surpasses baselines built on state-of-the-art correspondence estimation and view synthesis methods. Moreover, it outperforms the state-of-the-art unsupervised domain adaptation methods that utilize self-training and adversarial domain alignments. Our code and dataset will be made publicly available.

----

## [783] MeMOT: Multi-Object Tracking with Memory

**Authors**: *Jiarui Cai, Mingze Xu, Wei Li, Yuanjun Xiong, Wei Xia, Zhuowen Tu, Stefano Soatto*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00792](https://doi.org/10.1109/CVPR52688.2022.00792)

**Abstract**:

We propose an online tracking algorithm that performs the object detection and data association under a common framework, capable of linking objects after a long time span. This is realized by preserving a large spatio-temporal memory to store the identity embeddings of the tracked objects, and by adaptively referencing and aggregating useful information from the memory as needed. Our model, called MeMOT, consists of three main modules that are all Transformer-based: 1) Hypothesis Generation that produce object proposals in the current video frame; 2) Memory Encoding that extracts the core information from the memory for each tracked object; and 3) Memory Decoding that solves the object detection and data association tasks simultaneously for multi-object tracking. When evaluated on widely adopted MOT benchmark datasets, MeMOT observes very competitive performance.

----

## [784] Unsupervised Learning of Accurate Siamese Tracking

**Authors**: *Qiuhong Shen, Lei Qiao, Jinyang Guo, Peixia Li, Xin Li, Bo Li, Weitao Feng, Weihao Gan, Wei Wu, Wanli Ouyang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00793](https://doi.org/10.1109/CVPR52688.2022.00793)

**Abstract**:

Unsupervised learning has been popular in various computer vision tasks, including visual object tracking. However, prior unsupervised tracking approaches rely heavily on spatial supervision from templatesearch pairs and are still unable to track objects with strong variation over a long time span. As unlimited self-supervision signals can be obtained by tracking a video along a cycle in time, we investigate evolving a Siamese tracker by tracking videos forward-backward. We present a novel unsupervised tracking framework, in which we can learn temporal correspondence both on the classification branch and regression branch. Specifically, to propagate reliable template feature in the forward propagation process so that the tracker can be trained in the cycle, we first propose a consistency propagation transformation. We then identify an ill-posed penalty problem in conventional cycle training in backward propagation process. Thus, a differentiable region mask is proposed to select features as well as to implicitly penalize tracking errors on intermediate frames. Moreover, since noisy labels may degrade training, we propose a mask-guided loss reweighting strategy to assign dynamic weights based on the quality of pseudo labels. In extensive experiments, our tracker outperforms preceding unsupervised methods by a substantial margin, performing on par with supervised methods on large-scale datasets such as TrackingNet and LaSOT. Code is available at https://github.com/FlorinShum/ULAST.

----

## [785] Beyond 3D Siamese Tracking: A Motion-Centric Paradigm for 3D Single Object Tracking in Point Clouds

**Authors**: *Chaoda Zheng, Xu Yan, Haiming Zhang, Baoyuan Wang, Shenghui Cheng, Shuguang Cui, Zhen Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00794](https://doi.org/10.1109/CVPR52688.2022.00794)

**Abstract**:

3D single object tracking (3D SOT) in LiDAR point clouds plays a crucial role in autonomous driving. Current approaches all follow the Siamese paradigm based on appearance matching. However, LiDAR point clouds are usually textureless and incomplete, which hinders effective appearance matching. Besides, previous methods greatly overlook the critical motion clues among targets. In this work, beyond 3D Siamese tracking, we introduce a motion-centric paradigm to handle 3D SOT from a new perspective. Following this paradigm, we propose a matching-free two-stage tracker M2-Track. At the 1
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">st</sup>
-stage, M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
-Track localizes the target within successive frames via motion transformation. Then it refines the target box through motion-assisted shape completion at the 2
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">nd</sup>
-stage. Extensive experiments confirm that M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
-Track significantly outperforms previous state-of-the-arts on three large-scale datasets while running at 57FPS (~ 8%, ~ 17% and ~ 22% precision gains on KITTI, NuScenes, and Waymo Open Dataset respectively). Further analysis verifies each component's effectiveness and shows the motioncentric paradigm's promising potential when combined with appearance matching. Code will be made available at https://github.com/Ghostish/Open3DSOT.

----

## [786] GMFlow: Learning Optical Flow via Global Matching

**Authors**: *Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00795](https://doi.org/10.1109/CVPR52688.2022.00795)

**Abstract**:

Learning-based optical flow estimation has been dominated with the pipeline of cost volume with convolutions for flow regression, which is inherently limited to local correlations and thus is hard to address the long-standing challenge of large displacements. To alleviate this, the state-of-the-art framework RAFT gradually improves its prediction quality by using a large number of iterative refinements, achieving remarkable performance but introducing linearly increasing inference time. To enable both high accuracy and efficiency, we completely revamp the dominant flow regression pipeline by reformulating optical flow as a global matching problem, which identifies the correspondences by directly comparing feature similarities. Specifically, we propose a GMFlow framework, which consists of three main components: a customized Transformer for feature enhancement, a correlation and softmax layer for global feature matching, and a self-attention layer for flow propagation. We further introduce a refinement step that reuses GMFlow at higher feature resolution for residual flow prediction. Our new framework outperforms 31-refinements RAFT on the challenging Sintel benchmark, while using only one refinement and running faster, suggesting a new paradigm for accurate and efficient optical flow estimation. Code is available at https://github.com/haofeixu/gmflow.

----

## [787] GridShift: A Faster Mode-seeking Algorithm for Image Segmentation and Object Tracking

**Authors**: *Abhishek Kumar, Oladayo S. Ajani, Swagatam Das, Rammohan Mallipeddi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00796](https://doi.org/10.1109/CVPR52688.2022.00796)

**Abstract**:

In machine learning and computer vision, mean shift (MS) qualifies as one of the most popular mode-seeking algorithms used for clustering and image segmentation. It iteratively moves each data point to the weighted mean of its neighborhood data points. The computational cost required to find the neighbors of each data point is quadratic to the number of data points. Consequently, the vanilla MS appears to be very slow for large-scale datasets. To address this issue, we propose a mode-seeking algorithm called GridShift, with significant speedup and principally based on MS. To accelerate, GridShift employs a grid-based approach for neighbor search, which is linear in the number of data points. In addition, GridShift moves the active grid cells (grid cells associated with at least one data point) in place of data points towards the higher density, a step that provides more speedup. The runtime of Grid Shift is linear in the number of active grid cells and exponential in the number of features. Therefore, it is ideal for large-scale low-dimensional applications such as object tracking and image segmentation. Through extensive experiments, we showcase the superior performance of GridShift compared to other MS-based as well as state-of-the-art algorithms in terms of accuracy and runtime on benchmark datasets for image segmentation. Finally, we provide a new object-tracking al-gorithm based on GridShift and show promising results for object tracking compared to CamShift and meanshift++.

----

## [788] SNUG: Self-Supervised Neural Dynamic Garments

**Authors**: *Igor Santesteban, Miguel A. Otaduy, Dan Casas*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00797](https://doi.org/10.1109/CVPR52688.2022.00797)

**Abstract**:

We present a self-supervised method to learn dynamic 3D deformations of garments worn by parametric human bodies. State-of-the-art data-driven approaches to model 3D garment deformations are trained using supervised strategies that require large datasets, usually obtained by expensive physics-based simulation methods or professional multi-camera capture setups. In contrast, we propose a new training scheme that removes the need for ground-truth samples, enabling self-supervised training of dynamic 3D garment deformations. Our key contribution is to realize that physics-based deformation models, traditionally solved in a frame-by-frame basis by implicit integrators, can be recasted as an optimization problem. We leverage such optimization-based scheme to formulate a set of physics-based loss terms that can be used to train neural networks without precomputing ground-truth data. This allows us to learn models for interactive garments, including dynamic deformations and fine wrinkles, with a two orders of magnitude speed up in training time compared to state-of-the-art supervised methods.

----

## [789] Weakly-supervised Action Transition Learning for Stochastic Human Motion Prediction

**Authors**: *Wei Mao, Miaomiao Liu, Mathieu Salzmann*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00798](https://doi.org/10.1109/CVPR52688.2022.00798)

**Abstract**:

We introduce the task of action-driven stochastic human motion prediction, which aims to predict multiple plausible future motions given a sequence of action labels and a short motion history. This differs from existing works, which predict motions that either do not respect any specific action category, or follow a single action label. In particular, addressing this task requires tackling two challenges: The transitions between the different actions must be smooth; the length of the predicted motion depends on the action sequence and varies significantly across samples. As we cannot realistically expect training data to cover sufficiently diverse action transitions and motion lengths, we propose an effective training strategy consisting of combining multiple motions from different actions and introducing a weak form of supervision to encourage smooth transitions. We then design a VAE-based model conditioned on both the observed motion and the action label sequence, allowing us to generate multiple plausible future motions of varying length. We illustrate the generality of our approach by exploring its use with two different temporal encoding mod-els, namely RNNs and Transformers. Our approach out-performs baseline models constructed by adapting state-of-the-art single action-conditioned motion generation methods and stochastic human motion prediction approaches to our new task of action-driven stochastic motion prediction. Our code is available at https://github.com/wei-mao-2019/WAT.

----

## [790] Multi-Objective Diverse Human Motion Prediction with Knowledge Distillation

**Authors**: *Hengbo Ma, Jiachen Li, Ramtin Hosseini, Masayoshi Tomizuka, Chiho Choi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00799](https://doi.org/10.1109/CVPR52688.2022.00799)

**Abstract**:

Obtaining accurate and diverse human motion prediction is essential to many industrial applications, especially robotics and autonomous driving. Recent research has ex-plored several techniques to enhance diversity and maintain the accuracy of human motion prediction at the same time. However, most of them need to define a combined loss, such as the weighted sum of accuracy loss and diversity loss, and then decide their weights as hyperparameters before training. In this work, we aim to design a prediction frame-work that can balance the accuracy sampling and diversity sampling during the testing phase. In order to achieve this target, we propose a multi-objective conditional variational inference prediction model. We also propose a short-term oracle to encourage the prediction framework to explore more diverse future motions. We evaluate the performance of our proposed approach on two standard human motion datasets. The experiment results show that our approach is effective and on a par with state-of-the-art performance in terms of accuracy and diversity.

----

## [791] Context-Aware Sequence Alignment using 4D Skeletal Augmentation

**Authors**: *Taein Kwon, Bugra Tekin, Siyu Tang, Marc Pollefeys*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00800](https://doi.org/10.1109/CVPR52688.2022.00800)

**Abstract**:

Temporal alignment of fine-grained human actions in videos is important for numerous applications in computer vision, robotics, and mixed reality. State-of-the-art methods directly learn image-based embedding space by leveraging powerful deep convolutional neural networks. While being straightforward, their results are far from satisfactory, the aligned videos exhibit severe temporal discontinuity without additional post-processing steps. The recent advancements in human body and hand pose estimation in the wild promise new ways of addressing the task of human action alignment in videos. In this work, based on off-the-shelf human pose estimators, we propose a novel context-aware self-supervised learning architecture to align sequences of actions. We name it CASA. Specifically, CASA employs self-attention and cross-attention mechanisms to incorporate the spatial and temporal context of human actions, which can solve the temporal dis-continuity problem. Moreover, we introduce a self-supervised learning scheme that is empowered by novel 4D augmentation techniques for 3D skeleton representations. We systematically evaluate the key components of our method. Our experiments on three public datasets demonstrate CASA significantly improves phase progress and Kendall's Tau scores over the previous state-of-the-art methods.

----

## [792] Enabling Equivariance for Arbitrary Lie Groups

**Authors**: *Lachlan E. MacDonald, Sameera Ramasinghe, Simon Lucey*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00801](https://doi.org/10.1109/CVPR52688.2022.00801)

**Abstract**:

Although provably robust to translational perturbations, convolutional neural networks (CNNs) are known to suffer from extreme performance degradation when presented at test time with more general geometric transformations of inputs. Recently, this limitation has motivated a shift infocus from CNNs to Capsule Networks (CapsNets). However, CapsNets suffer from admitting relatively few theoretical guarantees of invariance. We introduce a rigourous mathematical framework to permit invariance to any Lie group of warps, exclusively using convolutions (over Lie groups), without the need for capsules. Previous work on group convolutions has been hampered by strong assumptions about the group, which precludes the application of such techniques to common warps in computer vision such as affine and homographic. Our framework enables the implementation of group convolutions over any finite-dimensional Lie group. We empirically validate our approach on the benchmark affine-invariant classification task, where we achieve ~30% improvement in accuracy against conventional CNNs while outperforming most CapsNets. As further illustration of the generality of our framework, we train a homography-convolutional model which achieves superior robustness on a homography-perturbed dataset, where CapsNet results degrade.

----

## [793] RAMA: A Rapid Multicut Algorithm on GPU

**Authors**: *Ahmed Abbas, Paul Swoboda*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00802](https://doi.org/10.1109/CVPR52688.2022.00802)

**Abstract**:

We propose a highly parallel primal-dual algorithm for the multicut (a.k.a. correlation clustering) problem, a classical graph clustering problem widely used in machine learning and computer vision. Our algorithm consists of three steps executed recursively: (1) Finding conflicted cycles that correspond to violated inequalities of the underlying multi-cut relaxation, (2) Performing message passing between the edges and cycles to optimize the Lagrange relaxation coming from the found violated cycles producing reduced costs and (3) Contracting edges with high reduced costs through matrix-matrix multiplications. Our algorithm produces primal solutions and lower bounds that estimate the distance to optimum. We implement our algorithm on GPUs and show resulting one to two orders-of-magnitudes improvements in execution speed without sac-rificing solution quality compared to traditional sequential algorithms that run on CPUs. We can solve very large scale benchmark problems with up to O(10
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">8</sup>
) variables in a few seconds with small primal-dual gaps. Our code is available at https://github.com/pawelswoboda/RAMA.

----

## [794] Self-Supervised Material and Texture Representation Learning for Remote Sensing Tasks

**Authors**: *Peri Akiva, Matthew Purri, Matthew J. Leotta*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00803](https://doi.org/10.1109/CVPR52688.2022.00803)

**Abstract**:

Self-supervised learning aims to learn image feature representations without the usage of manually annotated lbabels. It is often used as a precursor step to obtain useful initial network weights which contribute to faster convergence and superior performance of downstream tasks. While self-supervision allows one to reduce the domain gap between supervised and unsupervised learning without the usage of labels, the self-supervised objective still requires a strong inductive bias to downstream tasks for effective transfer learning. In this work, we present our material and texture based self-supervision method named MATTER (MATerial and TExture Representation Learning), which is inspired by classical material and texture methods. Material and texture can effectively describe any surface, including its tactile properties, color, and specularity. By extension, effective representation of material and texture can describe other semantic classes strongly associated with said material and texture. MATTER leverages multitemporal, spatially aligned remote sensing imagery over unchanged regions to learn invariance to illumination and viewing angle as a mechanism to achieve consistency of material and texture representation. We show that our self-supervision pre-training method allows for up to 24.22% and 6.33% performance increase in unsupervised and finetuned setups, and up to 76% faster convergence on change detection, land cover classification, and semantic segmentation tasks. Code and dataset: https://github.com/periakiva/MATTER.

----

## [795] RCP: Recurrent Closest Point for Point Cloud

**Authors**: *Xiaodong Gu, Chengzhou Tang, Weihao Yuan, Zuozhuo Dai, Siyu Zhu, Ping Tan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00804](https://doi.org/10.1109/CVPR52688.2022.00804)

**Abstract**:

3D motion estimation including scene flow and point cloud registration has drawn increasing interest. Inspired by 2D flow estimation, recent methods employ deep neural networks to construct the cost volume for estimating accurate 3D flow. However, these methods are limited by the fact that it is difficult to define a search window on point clouds because of the irregular data structure. In this paper, we avoid this irregularity by a simple yet effective method. We decompose the problem into two interlaced stages, where the 3D flows are optimized point-wisely at the first stage and then globally regularized in a recurrent network at the second stage. Therefore, the recurrent network only receives the regular point-wise information as the input. In the experiments, we evaluate the proposed method on both the 3D scene flow estimation and the point cloud registration task. For 3D scene flow estimation, we make comparisons on the widely used FlyingThings3D [32] and KITTI [33] datasets. For point cloud registration, we follow previous works and evaluate the data pairs with large pose and partially overlapping from ModelNet40 [65]. The results show that our method outperforms the previous method and achieves a new state-of-the-art performance on both 3D scene flow estimation and point cloud registration, which demonstrates the superiority of the proposed zero-order method on irregular point cloud data. Our source code is available at https://github.com/gxd1994/RCP.

----

## [796] Audio-Visual Speech Codecs: Rethinking Audio-Visual Speech Enhancement by Re-Synthesis

**Authors**: *Karren Yang, Dejan Markovic, Steven Krenn, Vasu Agrawal, Alexander Richard*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00805](https://doi.org/10.1109/CVPR52688.2022.00805)

**Abstract**:

Since facial actions such as lip movements contain significant information about speech content, it is not surprising that audio-visual speech enhancement methods are more accurate than their audio-only counterparts. Yet, state-of-the-art approaches still struggle to generate clean, realistic speech without noise artifacts and unnatural distortions in challenging acoustic environments. In this paper, we propose a novel audio-visual speech enhancement framework for high-fidelity telecommunications in AR/VR. Our approach leverages audio-visual speech cues to generate the codes of a neural speech codec, enabling efficient synthesis of clean, realistic speech from noisy signals. Given the importance of speaker-specific cues in speech, we focus on developing personalized models that work well for individual speakers. We demonstrate the efficacy of our approach on a new audio-visual speech dataset collected in an unconstrained, large vocabulary setting, as well as existing audio-visual datasets, outperforming speech enhancement baselines on both quantitative metrics and human evaluation studies. Please see the supplemental video for qualitative results
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/facebookresearch/facestar/releases/download/paper_materials/video.mp4.

----

## [797] Balanced Multimodal Learning via On-the-fly Gradient Modulation

**Authors**: *Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, Di Hu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00806](https://doi.org/10.1109/CVPR52688.2022.00806)

**Abstract**:

Multimodal learning helps to comprehensively understand the world, by integrating different senses. Accordingly, multiple input modalities are expected to boost model performance, but we actually find that they are not fully exploited even when the multimodal model outperforms its uni-modal counterpart. Specifically, in this paper we point out that existing multimodal discriminative models, in which uniform objective is designed for all modalities, could remain under-optimized uni-modal representations, caused by another dominated modality in some scenarios, e.g., sound in blowing wind event, vision in drawing picture event, etc. To alleviate this optimization imbalance, we propose on-the-fly gradient modulation to adaptively control the optimization of each modality, via monitoring the discrepancy of their contribution towards the learning objective. Further, an extra Gaussian noise that changes dynamically is introduced to avoid possible generalization drop caused by gradient modulation. As a result, we achieve considerable improvement over common fusion methods on different multimodal tasks, and this simple strategy can also boost existing multimodal methods, which illustrates its efficacy and versatility. The source code is available at https://github.com/GeWu-Lab/OGM-GE_CVPR2022.

----

## [798] Block-NeRF: Scalable Large Scene Neural View Synthesis

**Authors**: *Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben P. Mildenhall, Pratul P. Srinivasan, Jonathan T. Barron, Henrik Kretzschmar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00807](https://doi.org/10.1109/CVPR52688.2022.00807)

**Abstract**:

We present Block-NeRF, a variant of Neural Radiance Fields that can represent large-scale environments. Specifically, we demonstrate that when scaling NeRF to render city-scale scenes spanning multiple blocks, it is vital to de-compose the scene into individually trained NeRFs. This decomposition decouples rendering time from scene size, enables rendering to scale to arbitrarily large environments, and allows per-block updates of the environment. We adopt several architectural changes to make NeRF robust to data captured over months under different environmental conditions. We add appearance embeddings, learned pose refinement, and controllable exposure to each individual NeRF, and introduce a procedure for aligning appearance between adjacent NeRFs so that they can be seamlessly combined. We build a grid of Block-NeRFs from 2.8 million images to create the largest neural scene representation to date, capable of rendering an entire neighborhood of San Francisco.

----

## [799] SceneSqueezer: Learning to Compress Scene for Camera Relocalization

**Authors**: *Luwei Yang, Rakesh Shrestha, Wenbo Li, Shuaicheng Liu, Guofeng Zhang, Zhaopeng Cui, Ping Tan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00808](https://doi.org/10.1109/CVPR52688.2022.00808)

**Abstract**:

Standard visual localization methods build a priori 3D model of a scene which is used to establish correspondences against the 2D keypoints in a query image. Storing these pre-built 3D scene models can be prohibitively expensive for large-scale environments, especially on mobile devices with limited storage and communication bandwidth. We design a novel framework that compresses a scene while still maintaining localization accuracy. The scene is compressed in three stages: first, the database frames are clustered using pairwise co-visibility information. Then, a learned point selection module prunes the points in each cluster taking into account the final pose estimation accuracy. In the final stage, the features of the selected points are further compressed using learned quantization. Query image registration is done using only the compressed scene points. To the best of our knowledge, we are the first to propose learned scene compression for visual localization. We also demonstrate the effectiveness and efficiency of our method on various outdoor datasets where it can perform accurate localization with low memory consumption.

----



[Go to the previous page](CVPR-2022-list03.md)

[Go to the next page](CVPR-2022-list05.md)

[Go to the catalog section](README.md)