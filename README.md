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

## Model.summary()

```plain
Model: "pid_net"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 re_lu (ReLU)                (None, 168, 96, 32)       0         
                                                                 
 conv1 (Sequential)          (None, 84, 48, 32)        10400     
                                                                 
 made_layer_ (Sequential)    (None, 84, 48, 32)        37376     
                                                                 
 made_layer_ (Sequential)    (None, 42, 24, 64)        132352    
                                                                 
 made_layer_ (Sequential)    (None, 21, 12, 128)       822784    
                                                                 
 made_layer_ (Sequential)    (None, 11, 6, 256)        3283968   
                                                                 
 made_layer_ (Sequential)    (None, 6, 3, 512)         1779712   
                                                                 
 compression3 (Sequential)   (None, 21, 12, 64)        8448      
                                                                 
 compression4 (Sequential)   (None, 11, 6, 64)         16640     
                                                                 
 pag_fm (PagFM)              multiple                  4352      
                                                                 
 pag_fm_1 (PagFM)            multiple                  4352      
                                                                 
 made_layer_layer3_ (Sequent  (None, 42, 24, 64)       148480    
 ial)                                                            
                                                                 
 made_layer_layer4_ (Sequent  (None, 42, 24, 64)       148480    
 ial)                                                            
                                                                 
 made_layer_layer5_ (Sequent  (None, 42, 24, 128)      113152    
 ial)                                                            
                                                                 
 basic_block_14 (BasicBlock)  multiple                 30080     
                                                                 
 made_layer_layer4_d (Sequen  (None, 42, 24, 64)       15104     
 tial)                                                           
                                                                 
 diff3 (Sequential)          (None, 21, 12, 32)        36992     
                                                                 
 diff4 (Sequential)          (None, 11, 6, 64)         147712    
                                                                 
 pappm (PAPPM)               multiple                  720256    
                                                                 
 light__bag (Light_Bag)      multiple                  33792     
                                                                 
 made_layer_layer5_d (Sequen  (None, 42, 24, 128)      58880     
 tial)                                                           
                                                                 
 segmenthead (segmenthead)   multiple                  74883     
                                                                 
 segmenthead_1 (segmenthead)  multiple                 18849     
                                                                 
 segmenthead_2 (segmenthead)  multiple                 148867    
                                                                 
=================================================================
Total params: 7,795,911
Trainable params: 7,770,567
Non-trainable params: 25,344
_________________________________________________________________
```

## Images 

### Model Architecture
<img width="1221" alt="image" src="https://user-images.githubusercontent.com/41134624/211530339-be479edd-276d-4ebd-960c-950dfe8b51ab.png">

### Feature Visualization
<img width="1221" alt="image" src="https://user-images.githubusercontent.com/41134624/211530539-75dfbb42-8e17-4ac7-82b4-386391d998d5.png">

### Segmentation performance of PIDNets on Cityscapes Val set.
<img width="1212" alt="image" src="https://user-images.githubusercontent.com/41134624/211530609-17588334-d1b5-448b-aa4b-3351335531aa.png">
