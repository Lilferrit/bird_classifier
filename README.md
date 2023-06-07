# Welcome to a the bird classifier repo!

The goal of this project is to compare the relative performance of the Vgg19 and Vgg11 CNN
architectures in a transfer learning context. The networks were originally trained on the
ImageNet dataset, and we're retrained on the bird identification dataset available
[here](https://www.kaggle.com/competitions/birds23sp/data).

## Approach: Transfer Learning

The pre-trained Vgg19 and Vgg11 models were retrained on the bird classification dataset. Training
an entire vgg model would be way to computationally expensive. Thankfully, because of the
nature of this bird classification problem, training the entire model is not necessary. 