# LandslideNet
Automatic recognition and segmentation methods have become an essential requirement in identifying large-scale earthquake-induced landslides. This used to be conducted through pixel-based or object-oriented methods. However, these methods fail to develop an accurate, rapid, and cross-scene solution for earthquake-induced landslide recognition because of the massive amount of remote sensing data and variations in different earthquake scenarios. To fill this research gap, this paper proposes a robust deep transfer learning scheme for high precision and fast recognition of regional landslides. First, a Multi-scale Feature Fusion regime with an Encoder-decoder Network (MFFENet) is proposed to extract and fuse the multi-scale features of objects in remote sensing images, in which a novel and practical Adaptive Triangle Fork (ATF) Module is designed to integrate the useful features across different scales effectively. Second, an Adversarial Domain Adaptation Network (ADANet) is developed to perform different seismic landslide recognition tasks, and a multi-level output space adaptation scheme is proposed to enhance the adaptability of the segmentation model. Experimental results on standard remote sensing datasets demonstrate the effectiveness of MFFENet and ADANet. Finally, a comprehensive and general scheme is proposed for earthquake-induced landslide recognition, which integrates image features extracted from MFFENet and ADANet with the side information including landslide geologic features, bi-temporal changing features, and spatial analysis. The proposed scheme is applied in two earthquake-induced landslides in Jiuzhaigou (China) and Hokkaido (Japan), using available preand post-earthquake remote sensing images. These experiments show that the proposed scheme presents a state-of-the-art performance in regional landslide identification and performs stably and robustly in different seismic landslide recognition tasks. Our proposed framework demonstrates a competitive performance for high-precision, high-efficiency, and cross-scene recognition of earthquake disasters, which may serve as a new starting point for the application of deep learning and transfer learning methods in earthquake-induced landslide recognition.  
# Overview of LandslideNet

The proposed scheme includes the following four modules to analyze the seismic landslides under two different scenarios (supervised and unsupervised recognition):
