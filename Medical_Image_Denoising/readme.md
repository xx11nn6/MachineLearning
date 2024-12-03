# Medical Image Denoising
使用传统算法（NLM，BM3D）和监督学习算法（RED-CNN与CycleGAN）进行低剂量CT（LDCT）图像的去噪
使用Mayo的LDCT数据集进行训练，并在Mayo的L506号患者与Piglet数据集上进行测试

### 数据集

#### 文件结构
    Medical_Image_Denoising
    │
    ├── dataset
    │     ├──mayo
    │     └──piglet
    │
    ├── CycleGAN
    │
    │
    ├── RED-CNN
    │
    │
    ├── images
    │
    │
    └── save



##### 其中mayo数据集：
    mayo
    ├── L067
    │   ├── quarter_3mm
    │   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
    │   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
    │   │       └── ...
    │   └── full_3mm
    │           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
    │           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
    │           └── ...
    ├── L096
    │   ├── quarter_3mm
    │   │       └── ...
    │   └── full_3mm
    │           └── ...      
    ...
    │
    └── L506
        ├── quarter_3mm
        │       └── ...
        └── full_3mm
                └── ...    

##### piglet数据集：
    piglet
    ├── DICOM 
    │   └── PA0
    │        └──STO
    │           ├── SE0
    │           │    ├── IM0
    │           │    ├── IM1
    │           │    └── ...
    │           ...
    │           └── SE32
    │               ├── IM0
    │               ├── IM1
    │               └── ...
    ├── Piglet jpg    
    ...
    │
    └── Screenshots

#### Mayo数据集
The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic
https://www.aapm.org/GrandChallenge/LowDoseCT/

#### Piglet数据集
Piglet数据集链接已失效，需要的可以联系我获取

### 算法
#### RED-CNN
Github: https://github.com/SSinyu/RED-CNN
DOI: https://doi.org/10.1109/TMI.2017.2715284

#### CycleGAN
Github: https://github.com/Glanceyes/CT-Denoising-CycleGAN



### 食用方法

#### 预处理数据
1. run `python ./RED-CNN/prep.py`
2. run `python pre_process.py`

#### 训练RED-CNN
3. `python ./RED-CNN/main.py --mode=train --load_mode=0 --batch_size=16 --num_epochs=100 --lr=1e-5 --device='cuda' --num_workers=7`

#### 测试RED-CNN性能并生成统一的Mayo数据集
4. 测试mayo数据集 `python ./RED-CNN/main.py --mode='test' --dataset='mayo' --path_data='../dataset/mayo_npy' --test_iters=100000`
5. 测试piglet数据集 `python ./RED-CNN/main.py --mode='test' --dataset='piglet' --path_data='../dataset/piglet_npy' --test_iters=100000`

#### 训练CycleGAN
6. `python ./CycleGAN/cycleGAN_train.py --model_name='cyclegan_v1' --path_data='../dataset/mayo_npy' --dataset='mayo' --batch_size=16 --num_epoch=50 --lr=1e-4`
   
#### 测试CycleGAN
7. 测试mayo数据集 `python ./CycleGAN/cycleGAN_test.py --path_data='../save' --path_result='../save' --dataset='mayo'`
8. 测试piglet数据集 `python ./CycleGAN/cycleGAN_test.py --path_data='../dataset/piglet_npy' --path_result='../save' --dataset='piglet'`

#### 使用NLM处理图像
9. 处理mayo数据集 `python main.py --function=1 --dataset=mayo --data_dir=./save --save_dir=./save --algorithm=nlm`
10. 处理piglet数据集 `python main.py --function=1 --dataset=piglet --data_dir=./dataset/piglet_npy --save_dir=./save --algorithm=nlm`

#### 使用BM3D处理图像
11. 处理mayo数据集 `python main.py --function=2 --dataset=mayo --data_dir=./save --save_dir=./save --algorithm=bm3d`
12. 处理piglet数据集 `python main.py --function=2 --dataset=piglet --data_dir=./dataset/piglet_npy --save_dir=./save --algorithm=bm3d`

#### 评估不同算法，并生成测试数据
##### 评估Mayo数据集
13. 评估原始LDCT的PSNR与SSIM `python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=LDCT`
14. 评估NLM `python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=nlm`
15. 评估BM3D `python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=bm3d`
16. 评估RED-CNN `python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=redcnn`
17. 评估CycleGAN `python main.py --function=3 --dataset=mayo --save_dir=./save --algorithm=cyclegan`

##### 评估Piglet数据集
18. 评估原始LDCT的PSNR与SSIM `python main.py --function=3 --dataset=piglet --save_dir=./save --algorithm=LDCT`
19. 评估NLM `python main.py --function=3 --dataset=piglet --save_dir=./save --algorithm=nlm`
20. 评估BM3D `python main.py --function=3 --dataset=piglet --save_dir=./save --algorithm=bm3d`
21. 评估RED-CNN `python main.py --function=3 --dataset=piglet --save_dir=./save --algorithm=redcnn`
22. 评估CycleGAN `python main.py --function=3 --dataset=piglet --save_dir=./save --algorithm=cyclegan`

#### 显示不同算法的PSNR与SSIM对比
23. Mayo `python main.py --function=4 --dataset=mayo --save_dir=./save`
24. Piglet `python main.py --function=4 --dataset=piglet --save_dir=./save`

#### 显示局部放大图对比
25. Mayo `python main.py --function=5 --dataset=mayo --save_dir=./save --zoom_coords='200,250,200,250' --index=365`
26. Piglet `python main.py --function=5 --dataset=piglet --save_dir=./save --zoom_coords='200,250,200,250' --index=153`


### 存在的问题
全是问题
SSinyu佬的RED-CNN计算PSNR与SSIM时使用了以下裁切：
$$ Denormalized_image = Normalized_image \times (3072+1024) - 1024 clip to [-160,240] $$
不是很清楚这个裁切是怎么来的，导致在算PSNR与SSIM时结果有很大差异
RED-CNN与CycleGAN实际表现不好，可能也是因为裁切
可能需要修改损失函数重新训练

另外文件系统很混乱，但是不想管了（）
