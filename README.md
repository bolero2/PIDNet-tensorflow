# PIDNet-tf2
PIDNet implementation using tensorflow 2.X

* Original Repository (pytorch ver.) : https://github.com/XuJiacong/PIDNet  
* Paper : https://arxiv.org/abs/2206.02066  
* paperswithcode : https://paperswithcode.com/paper/pidnet-a-real-time-semantic-segmentation

## Abstract
Two-branch network architecture has shown its efficiency and effectiveness for real-time semantic segmentation tasks.  

However, direct fusion of low-level details and high-level semantics will lead to a phenomenon that the detailed features are easily overwhelmed by surrounding contextual infor- mation, namely overshoot in this paper, which limits the improvement of the accuracy of existed two-branch models.  

In this paper, we bridge a connection between Convolu- tional Neural Network (CNN) and **Proportional-Integral-Derivative (PID) controller** and reveal that the two-branch network is nothing but a Proportional-Integral (PI) controller, which inherently suffers from the similar overshoot issue.  

To alleviate this issue, we propose a novel three- branch network architecture: PIDNet, which possesses three branches to parse the detailed, context and boundary infor- mation (derivative of semantics), respectively, and employs boundary attention to guide the fusion of detailed and context branches in final stage. The family of PIDNets achieve the best trade-off between inference speed and accuracy and their test accuracy surpasses all the existed models with similar inference speed on Cityscapes, CamVid and COCO-Stuff datasets.  

Especially, PIDNet-S achieves 78.6% mIOU with inference speed of 93.2 FPS on Cityscapes test set and 80.1% mIOU with speed of 153.7 FPS on CamVid test set.

## Images 

<img width="1221" alt="image" src="https://user-images.githubusercontent.com/41134624/211530339-be479edd-276d-4ebd-960c-950dfe8b51ab.png">

<img width="1221" alt="image" src="https://user-images.githubusercontent.com/41134624/211530539-75dfbb42-8e17-4ac7-82b4-386391d998d5.png">

<img width="1212" alt="image" src="https://user-images.githubusercontent.com/41134624/211530609-17588334-d1b5-448b-aa4b-3351335531aa.png">
