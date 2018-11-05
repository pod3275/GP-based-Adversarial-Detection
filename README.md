# GP-based-Adversarial-Detection

- 현재 module화 및 code 정리 중

- Paper : Detect Adversarial Example Using Gaussian Process-based Detector
- Author : Sangheon Lee, Noo-ri Kim, Jee-hyong Lee

- Abstract

  Adversarial attack is a technique that causes a malfunction of classification models by adding noise that cannot be distinguished by humans, which poses a threat to a deep learning model applied to security-sensitive systems. In this paper, we propose a simple and fast method to detect adversarial image using Gaussian process classification. Existing machine learning-based adversarial detect methods require a large number of adversarial images for the learning of the detector. The proposed method overcomes this problem by performing classification based on the statistical features of adversarial images and clean images that extracted by Gaussian process with small number of images. The proposed method can determine whether the input image is an adversarial image by applying Gaussian process classification based on the predicted value of the classifier model
  
- Using adversarial attack method : FGSM, BIM, JSMA, DeepFool, C&W 

- Model

  ![image](https://user-images.githubusercontent.com/26705935/47216775-0b91b180-d3e1-11e8-90d3-d015f70e02e0.png)

- Result

  ![image](https://user-images.githubusercontent.com/26705935/47216750-f6b51e00-d3e0-11e8-9fac-de9d644afccb.png)
  
  
