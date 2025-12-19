# ViT - Vision Transformer ~ (An Image Worth 16x16 Words)

This work is an approximate replica of Vision Transformer as shown in [AN IMAGE IS WORTH 16X16 WORDS](https://arxiv.org/pdf/2010.11929)

![alt text](<Screenshot 2025-12-19 082538.png>)

## Model Details

Below is the model attributes which is exactly same for ViT-Base with 87M parameters.

|Attribute|Value|
|---------|-----|
|Training Data|ImageNet1K - Kaggle|
|Image Input Size|(224,224,3)|
|Patch|16 in height and width|
|Num of Transformer Layer|12|
|Hidden Dimension|768|
|Num heads|12|
|MLP Size|3072|
|Number of class|1000|
|dropout|0.0|
|Learning Rate|2e-4|
|Scheduler|Exponential Cosine Decay with Warmup|
|Optimizer|Adam|
|$(\beta_1, \beta_2)$|(0.9,0,999)|
|weight decay|0.1|
|Warmup steps| 1000|


***Note: Due to GPU limitation, the training is not completed.***