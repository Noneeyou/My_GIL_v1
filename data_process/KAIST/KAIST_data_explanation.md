# KAIST振动数据解释

## 关于KAIST振动数据命名方式的概况
### 负载_故障类型_转速

## 在这个项目中只是用 0nm 负载条件下故障数据进行训练，因为负载在工作中一般是提前固定的
    所有数据：
            0Nm_BPFI_03.mat
            0Nm_BPFI_10.mat
            0Nm_BPFI_30.mat
            0Nm_BPFO_03.mat
            0Nm_BPFO_10.mat
            0Nm_BPFO_30.mat
            0Nm_Misalign_01.mat
            0Nm_Misalign_03.mat
            0Nm_Misalign_05.mat
            0Nm_Normal.mat
            0Nm_Unbalance_0583mg.mat
            0Nm_Unbalance_1169mg.mat
            0Nm_Unbalance_1751mg.mat
            0Nm_Unbalance_2239mg.mat
            0Nm_Unbalance_3318mg.mat

    映射表：
            | 文件名                   | 工况类别（含转速）                |  数值标签  |
            | :----------------------- | :------------------------        | :----: |
            | 0Nm_Normal.mat           | Normal                           |  **1** |
            | 0Nm_BPFO_03.mat          | BPFO_03（外圈故障-转速03）        |  **2** |
            | 0Nm_BPFO_10.mat          | BPFO_10（外圈故障-转速10）        |  **3** |
            | 0Nm_BPFO_30.mat          | BPFO_30（外圈故障-转速30）        |  **4** |
            | 0Nm_BPFI_03.mat          | BPFI_03（内圈故障-转速03）        |  **5** |
            | 0Nm_BPFI_10.mat          | BPFI_10（内圈故障-转速10）        |  **6** |
            | 0Nm_BPFI_30.mat          | BPFI_30（内圈故障-转速30）        |  **7** | 
            | 0Nm_Misalign_01.mat      | Misalign_01（轴对中不良-低速01）  |  **8** |
            | 0Nm_Misalign_03.mat      | Misalign_03（轴对中不良-中速03）  |  **9** |
            | 0Nm_Misalign_05.mat      | Misalign_05（轴对中不良-高速05）  | **10** |
            | 0Nm_Unbalance_0583mg.mat | Unbalance_0583mg（不平衡-轻度）   | **11** |
            | 0Nm_Unbalance_1169mg.mat | Unbalance_1169mg（不平衡-中度1）  | **12** |
            | 0Nm_Unbalance_1751mg.mat | Unbalance_1751mg（不平衡-中度2）  | **13** |
            | 0Nm_Unbalance_2239mg.mat | Unbalance_2239mg（不平衡-重度1）  | **14** |
            | 0Nm_Unbalance_3318mg.mat | Unbalance_3318mg（不平衡-重度2）  | **15** |

# 数据集的合成
    在这个项目中，建图方式是以时序为基础的建图，在建图后所有的图结构保存为pyg格式
    不同故障混合的时候，采取按百分比混合，考虑到数据要符合真实工况，在混合时会尽力做到故障又轻度到重度在时序上的单向性

## 需要哪些混合数据
### 所有故障的混合
    为了保证pre_model在性能上足以区分上述的所有故障，需要一个包含所有类别的数据集，要先在这个数据集上完成不错的训练效果，也方便后续在增量过程后对比微调模型是否有效。
### 不断类增量的数据
    构建类增量数据的思想：
                        1.相似类的增加：例如外圈故障转速10和外圈故障转速30 故障数据就很相似——可以衡量增量模型对于相似数据的增量效果
                        2.完全新的故障的增加：例如外圈故障和轴对中不齐——可以衡量增量模型对于完全不同的新类别的故障的增量效果
                        3.旧类型回放：检查模型的防遗忘性
#### 一类：
#### 二类：
#### 三类：
#### 四类：
