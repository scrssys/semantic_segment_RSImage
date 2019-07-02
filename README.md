We use several semantic segmentation models  to deal remote sensing images classifcation.
Semantic segmentation networks:U-net,PSPNet, FPN, LinkNet, DeepLab V3+
Backones: 
Type	Names
VGG:'vgg16' 'vgg19'
ResNet:'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
SE-ResNet:'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'
ResNeXt:'resnext50' 'resnext101'
SE-ResNeXt:'seresnext50' 'seresnext101'
SENet154:'senet154'
DenseNet:'densenet121' 'densenet169' 'densenet201'
Inception:'inceptionv3' 'inceptionresnetv2'
MobileNet:'mobilenet' 'mobilenetv2'
EfficientNet:'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3'

Requirements:
tensorflow-gpu==1.9.0(only test on V1.9.0)
keras>=2.2.4
keras_applications==1.0.7
image-classifiers==0.2.0
efficientnet>=0.0.3
cuda==9.0
qt==5.6
numpy==1.12.0
scipy==0.19.1
tqdm==4.11.2


