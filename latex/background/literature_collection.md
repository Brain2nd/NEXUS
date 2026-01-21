# MofNeuroSim论文背景文献汇总

## 1. 神经形态计算综述

### 1.1 综合性综述

**Schuman et al. (2017)** - A Survey of Neuromorphic Computing and Neural Networks in Hardware
- **URL**: https://arxiv.org/abs/1705.06963
- **引用次数**: 1231+
- **关键内容**: 提供了神经形态计算历史和研究动机的全面综述
- **BibTeX**:
```bibtex
@article{schuman2017survey,
  title={A survey of neuromorphic computing and neural networks in hardware},
  author={Schuman, Catherine D and Potok, Thomas E and Patton, Robert M and Birdwell, J Douglas and Dean, Mark E and Rose, Garrett S and Plank, James S},
  journal={arXiv preprint arXiv:1705.06963},
  year={2017}
}
```

**Shrestha & Orchard (2022)** - A Survey on Neuromorphic Computing: Models and Hardware
- **URL**: https://ieeexplore.ieee.org/document/9782767/
- **引用次数**: 148+
- **关键内容**: 综述了现有神经形态计算系统的计算模型和硬件平台,首先介绍神经元和突触模型
- **BibTeX**:
```bibtex
@article{shrestha2022survey,
  title={A survey on neuromorphic computing: Models and hardware},
  author={Shrestha, Amar and Orchard, Garrick},
  journal={IEEE Circuits and Systems Magazine},
  volume={22},
  number={2},
  pages={6--35},
  year={2022},
  publisher={IEEE}
}
```

**Kudithipudi et al. (2025)** - Neuromorphic computing at scale
- **URL**: https://www.nature.com/articles/s41586-024-08253-8
- **引用次数**: 228+
- **关键内容**: 神经形态计算是一种脑启发的硬件和算法设计方法,可高效实现人工神经网络
- **BibTeX**:
```bibtex
@article{kudithipudi2025neuromorphic,
  title={Neuromorphic computing at scale},
  author={Kudithipudi, Dhireesha and others},
  journal={Nature},
  year={2025},
  publisher={Nature Publishing Group}
}
```

### 1.2 脉冲神经网络综述

**Yamazaki et al. (2022)** - Spiking Neural Networks and Their Applications: A Review
- **URL**: https://scholarworks.uark.edu/cscepub/51/
- **引用次数**: 752+
- **关键内容**: SNN旨在弥合神经科学和机器学习之间的差距,使用生物学上真实的神经元模型执行计算
- **BibTeX**:
```bibtex
@article{yamazaki2022spiking,
  title={Spiking neural networks and their applications: A review},
  author={Yamazaki, Kashu and Vo-Ho, Viet-Khoa and Bulsara, Darshan and Le, Ngan},
  journal={Brain Sciences},
  volume={12},
  number={7},
  pages={863},
  year={2022},
  publisher={MDPI}
}
```

**Rathi & Roy (2023)** - Exploring Neuromorphic Computing Based on Spiking Neural Networks: Algorithms to Hardware
- **URL**: https://dl.acm.org/doi/full/10.1145/3571155
- **引用次数**: 297+
- **关键内容**: 概述了神经形态工程下脉冲神经网络领域的最新发展
- **BibTeX**:
```bibtex
@article{rathi2023exploring,
  title={Exploring neuromorphic computing based on spiking neural networks: Algorithms to hardware},
  author={Rathi, Nitin and Roy, Kaushik},
  journal={ACM Computing Surveys},
  volume={55},
  number={12},
  pages={1--49},
  year={2023},
  publisher={ACM}
}
```

**Malcolm et al. (2023)** - A Comprehensive Review of Spiking Neural Networks
- **URL**: https://arxiv.org/abs/2303.10780
- **引用次数**: 40+
- **关键内容**: 对脉冲神经网络在解释、优化、效率和准确性方面的最新发展进行文献综述
- **BibTeX**:
```bibtex
@article{malcolm2023comprehensive,
  title={A comprehensive review of spiking neural networks},
  author={Malcolm, Kaleb and others},
  journal={arXiv preprint arXiv:2303.10780},
  year={2023}
}
```

## 2. 浮点运算在神经形态系统中的实现

### 2.1 IEEE 754浮点运算

**Dubey et al. (2020)** - Floating-Point Multiplication Using Neuromorphic Computing
- **URL**: https://arxiv.org/abs/2008.13245
- **引用次数**: 4+
- **关键内容**: 描述了一个执行IEEE 754兼容浮点乘法的神经形态系统
- **BibTeX**:
```bibtex
@article{dubey2020floating,
  title={Floating-point multiplication using neuromorphic computing},
  author={Dubey, Kshitij and Raj, Bharat and others},
  journal={arXiv preprint arXiv:2008.13245},
  year={2020}
}
```

**George et al. (2019)** - IEEE 754 floating-point addition for neuromorphic architecture
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0925231219308884
- **引用次数**: 12+
- **关键内容**: 考虑了神经形态架构中IEEE 754兼容浮点加法的问题
- **BibTeX**:
```bibtex
@article{george2019ieee,
  title={IEEE 754 floating-point addition for neuromorphic architecture},
  author={George, Anup Mathew and Banerjee, Debayan and Suri, Manan},
  journal={Neurocomputing},
  volume={364},
  pages={139--153},
  year={2019},
  publisher={Elsevier}
}
```

### 2.2 算术原语

**Wurm et al. (2023)** - Arithmetic Primitives for Efficient Neuromorphic Computing
- **URL**: https://ieeexplore.ieee.org/document/10386397/
- **引用次数**: 5+
- **关键内容**: 提出初步结果,使我们能够在神经形态计算机上高效编码数据并执行基本算术运算
- **BibTeX**:
```bibtex
@inproceedings{wurm2023arithmetic,
  title={Arithmetic primitives for efficient neuromorphic computing},
  author={Wurm, Alexander and others},
  booktitle={2023 IEEE International Conference on Rebooting Computing (ICRC)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
```

**Mikaitis (2020)** - Arithmetic Accelerators for a Digital Neuromorphic Processor
- **URL**: https://mmikaitis.github.io/assets/pdf/mikaitis20.pdf
- **引用次数**: 5+
- **关键内容**: 探索了用于计算指数和自然对数的可编程加速器,这些函数在脉冲神经网络中很常见
- **BibTeX**:
```bibtex
@phdthesis{mikaitis2020arithmetic,
  title={Arithmetic accelerators for a digital neuromorphic processor},
  author={Mikaitis, Mantas},
  year={2020},
  school={University of Manchester}
}
```

### 2.3 精度探索

**Kwak et al. (2021)** - Precision Exploration of Floating-Point Arithmetic for Spiking Neural Networks
- **URL**: https://ieeexplore.ieee.org/document/9614005/
- **引用次数**: 5+
- **关键内容**: 探索了各种浮点表示的精度,以实现能效脉冲神经网络
- **BibTeX**:
```bibtex
@inproceedings{kwak2021precision,
  title={Precision exploration of floating-point arithmetic for spiking neural networks},
  author={Kwak, Minhyeok and others},
  booktitle={2021 IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS)},
  pages={1--4},
  year={2021},
  organization={IEEE}
}
```

## 3. LIF和GLIF神经元模型

### 3.1 GLIF模型

**Teeter et al. (2018)** - Generalized leaky integrate-and-fire models classify multiple neuron types
- **URL**: https://www.nature.com/articles/s41467-017-02717-4
- **引用次数**: 328+
- **关键内容**: 构建了一组复杂度递增的广义泄漏积分发火(GLIF)模型,以重现645个记录神经元的脉冲行为
- **BibTeX**:
```bibtex
@article{teeter2018generalized,
  title={Generalized leaky integrate-and-fire models classify multiple neuron types},
  author={Teeter, Corinne and Iyer, Ramakrishnan and Menon, Vilas and Gouwens, Nathan and Feng, David and Berg, Jim and Szafer, Aaron and Cain, Nicholas and Zeng, Hongkui and Hawrylycz, Michael and others},
  journal={Nature communications},
  volume={9},
  number={1},
  pages={709},
  year={2018},
  publisher={Nature Publishing Group}
}
```

**Yao et al. (2022)** - GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks
- **URL**: https://papers.neurips.cc/paper_files/paper/2022/file/cfa8440d500a6a6867157dfd4eaff66e-Paper-Conference.pdf
- **引用次数**: 187+
- **关键内容**: 提出门控LIF(GLIF),一个统一的脉冲神经元,融合三种神经行为中的不同生物特征
- **BibTeX**:
```bibtex
@inproceedings{yao2022glif,
  title={GLIF: A unified gated leaky integrate-and-fire neuron for spiking neural networks},
  author={Yao, Xingting and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={32160--32171},
  year={2022}
}
```

**Marasco et al. (2023)** - An Adaptive Generalized Leaky Integrate-and-Fire Model
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC10550887/
- **引用次数**: 16+
- **关键内容**: 提出了海马CA1神经元和中间神经元的自适应广义泄漏积分发火模型
- **BibTeX**:
```bibtex
@article{marasco2023adaptive,
  title={An adaptive generalized leaky integrate-and-fire model for hippocampal CA1 pyramidal neurons and interneurons},
  author={Marasco, Addolorata and others},
  journal={Cognitive Neurodynamics},
  volume={17},
  number={5},
  pages={1261--1288},
  year={2023},
  publisher={Springer}
}
```

### 3.2 LIF基础模型

**Lu et al. (2022)** - Linear leaky-integrate-and-fire neuron model based spiking neural networks
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9448910/
- **引用次数**: 52+
- **关键内容**: 建立了线性泄漏积分发火模型(LIF)/SNN的生物参数与人工神经网络参数之间的精确数学映射
- **BibTeX**:
```bibtex
@article{lu2022linear,
  title={Linear leaky-integrate-and-fire neuron model based spiking neural networks and its mapping relationship to deep neural networks},
  author={Lu, Sijia and Xu, Feng},
  journal={Frontiers in neuroscience},
  volume={16},
  pages={857513},
  year={2022},
  publisher={Frontiers}
}
```

## 4. 动力系统和拓扑嵌入

### 4.1 动力系统视角

**Zhang et al. (2021)** - Bifurcation Spiking Neural Network
- **URL**: https://jmlr.org/papers/volume22/20-1031/20-1031.pdf
- **引用次数**: 11+
- **关键内容**: 从动力系统的角度研究脉冲神经模型和网络,揭示了脉冲神经元的动力学特性
- **BibTeX**:
```bibtex
@article{zhang2021bifurcation,
  title={Bifurcation spiking neural network},
  author={Zhang, Shaoqing and others},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={184},
  pages={1--50},
  year={2021}
}
```

**Wei et al. (2025)** - Physics-informed spiking neural networks for continuous-time dynamic systems
- **URL**: https://www.sciencedirect.com/science/article/pii/S0925231225028644
- **关键内容**: 物理信息机器学习通过结合先验物理知识使神经网络能够理解动态系统
- **BibTeX**:
```bibtex
@article{wei2025physics,
  title={Physics-informed spiking neural networks for continuous-time dynamic systems},
  author={Wei, Qian and Yang, Qiang and Han, Liang and Zhang, Tao},
  journal={Neurocomputing},
  year={2025},
  publisher={Elsevier}
}
```

### 4.2 拓扑方法

**Papamarkou et al. (2024)** - Position: Topological Deep Learning is the New Frontier for Relational Learning
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11973457/
- **引用次数**: 78+
- **关键内容**: 拓扑深度学习(TDL)是一个快速发展的领域,使用拓扑特征来理解和设计深度学习模型
- **BibTeX**:
```bibtex
@article{papamarkou2024position,
  title={Position: Topological deep learning is the new frontier for relational learning},
  author={Papamarkou, Theodore and others},
  journal={arXiv preprint arXiv:2402.08871},
  year={2024}
}
```

**Suresh et al. (2024)** - On characterizing the evolution of embedding space of neural networks
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0167865524000369
- **引用次数**: 9+
- **关键内容**: 通过Betti数研究特征嵌入空间的拓扑如何在经过训练良好的深度神经网络层时发生变化
- **BibTeX**:
```bibtex
@article{suresh2024characterizing,
  title={On characterizing the evolution of embedding space of neural networks using algebraic topology},
  author={Suresh, Suchismit and others},
  journal={Pattern Recognition Letters},
  volume={180},
  pages={107--114},
  year={2024},
  publisher={Elsevier}
}
```

## 5. SNN训练方法

### 5.1 代理梯度方法

**Neftci et al. (2019)** - Surrogate Gradient Learning in Spiking Neural Networks
- **URL**: https://arxiv.org/abs/1901.09948
- **引用次数**: 2040+
- **关键内容**: 逐步阐明训练脉冲神经网络时通常遇到的问题,并引导读者了解关键概念
- **BibTeX**:
```bibtex
@article{neftci2019surrogate,
  title={Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks},
  author={Neftci, Emre O and Mostafa, Hesham and Zenke, Friedemann},
  journal={IEEE Signal Processing Magazine},
  volume={36},
  number={6},
  pages={51--63},
  year={2019},
  publisher={IEEE}
}
```

**Zenke & Vogels (2021)** - The Remarkable Robustness of Surrogate Gradient Learning
- **URL**: https://direct.mit.edu/neco/article/33/4/899/97482/The-Remarkable-Robustness-of-Surrogate-Gradient
- **引用次数**: 393+
- **关键内容**: 代理梯度为在脉冲网络的人工模型中灌输复杂功能提供了一种有前途的方法
- **BibTeX**:
```bibtex
@article{zenke2021remarkable,
  title={The remarkable robustness of surrogate gradient learning for instilling complex function in spiking neural networks},
  author={Zenke, Friedemann and Vogels, Tim P},
  journal={Neural Computation},
  volume={33},
  number={4},
  pages={899--925},
  year={2021},
  publisher={MIT Press}
}
```

### 5.2 直通估计器

**Yin et al. (2019)** - Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets
- **URL**: https://arxiv.org/abs/1903.05662
- **引用次数**: 470+
- **关键内容**: 训练激活量化神经网络涉及最小化分段常数函数,其梯度几乎处处消失
- **BibTeX**:
```bibtex
@article{yin2019understanding,
  title={Understanding straight-through estimator in training activation quantized neural nets},
  author={Yin, Penghang and Lyu, Jiancheng and Zhang, Shuai and Osher, Stanley and Qi, Yingyong and Xin, Jack},
  journal={arXiv preprint arXiv:1903.05662},
  year={2019}
}
```

**Chen et al. (2024)** - Memristive leaky integrate-and-fire neuron and learnable straight-through estimator
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11564454/
- **引用次数**: 7+
- **关键内容**: 提出了一种基于生物合理性的局部学习方法,如通过STDP的无监督学习训练的SNN
- **BibTeX**:
```bibtex
@article{chen2024memristive,
  title={Memristive leaky integrate-and-fire neuron and learnable straight-through estimator in spiking neural networks},
  author={Chen, Tao and She, Chengdong and Wang, Lidan and Duan, Shukai},
  journal={Cognitive Neurodynamics},
  year={2024},
  publisher={Springer}
}
```

### 5.3 时间反向传播

**Guo et al. (2023)** - Efficient training of spiking neural networks with temporally-truncated backpropagation through time
- **URL**: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1047008/full
- **引用次数**: 28+
- **关键内容**: 著名的时间反向传播(BPTT)算法用于训练SNN,但存在大内存占用问题
- **BibTeX**:
```bibtex
@article{guo2023efficient,
  title={Efficient training of spiking neural networks with temporally-truncated backpropagation through time},
  author={Guo, Wenshuo and others},
  journal={Frontiers in Neuroscience},
  volume={17},
  pages={1047008},
  year={2023},
  publisher={Frontiers}
}
```

**Meng et al. (2023)** - Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks
- **URL**: https://arxiv.org/abs/2302.14311
- **引用次数**: 111+
- **关键内容**: 提出了空间时间学习(SLTT)方法,可以在大幅提高训练效率的同时实现高性能
- **BibTeX**:
```bibtex
@inproceedings{meng2023towards,
  title={Towards memory-and time-efficient backpropagation for training spiking neural networks},
  author={Meng, Qingyan and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6166--6176},
  year={2023}
}
```

## 6. 注意力机制和Transformer架构

### 6.1 SNN中的注意力机制

**Yao et al. (2022)** - Attention Spiking Neural Networks
- **URL**: https://arxiv.org/abs/2209.13929
- **引用次数**: 331+
- **关键内容**: 研究了SNN中注意力机制的效果,首先提出了一个即插即用的注意力工具包,称为多维注意力
- **BibTeX**:
```bibtex
@article{yao2022attention,
  title={Attention spiking neural networks},
  author={Yao, Man and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={8},
  pages={9393--9410},
  year={2023},
  publisher={IEEE}
}
```

**Liu et al. (2022)** - Enhancing spiking neural networks with hybrid top-down attention mechanism
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9443487/
- **引用次数**: 11+
- **关键内容**: 提供了一种通过采用混合自上而下注意力机制来提升SNN性能的新方法
- **BibTeX**:
```bibtex
@article{liu2022enhancing,
  title={Enhancing spiking neural networks with hybrid top-down attention mechanism},
  author={Liu, Fangxin and others},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={15250},
  year={2022},
  publisher={Nature Publishing Group}
}
```

### 6.2 Spiking Transformer

**Yao et al. (2024)** - Spike-driven Transformer V2: Meta Spiking Neural Network Architecture
- **URL**: https://arxiv.org/abs/2404.03663
- **引用次数**: 167+
- **关键内容**: 提出了一种通用的基于Transformer的SNN架构,称为Meta-SpikeFormer
- **BibTeX**:
```bibtex
@article{yao2024spike,
  title={Spike-driven transformer v2: Meta spiking neural network architecture inspiring the design of next-generation neuromorphic chips},
  author={Yao, Man and others},
  journal={arXiv preprint arXiv:2404.03663},
  year={2024}
}
```

**Li et al. (2024)** - Spikeformer: Training high-performance spiking neural network with transformer
- **URL**: https://www.sciencedirect.com/science/article/pii/S092523122400050X
- **引用次数**: 22+
- **关键内容**: 提出了一个基于Transformer的SNN,称为"Spikeformer",在静态数据集和神经形态数据集上都优于其ANN对应物
- **BibTeX**:
```bibtex
@article{li2024spikeformer,
  title={Spikeformer: Training high-performance spiking neural network with transformer},
  author={Li, Yudong and Lei, Yunlin and Yang, Xu},
  journal={Neurocomputing},
  volume={580},
  pages={127499},
  year={2024},
  publisher={Elsevier}
}
```

## 7. 归一化层

**Ba et al. (2016)** - Layer Normalization
- **URL**: https://arxiv.org/abs/1607.06450
- **引用次数**: 17541+
- **关键内容**: 将批量归一化转置为层归一化,通过在每个时间步单独计算归一化统计量来训练循环神经网络
- **BibTeX**:
```bibtex
@article{ba2016layer,
  title={Layer normalization},
  author={Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E},
  journal={arXiv preprint arXiv:1607.06450},
  year={2016}
}
```

**Shao et al. (2020)** - Is normalization indispensable for training deep neural network?
- **URL**: https://proceedings.neurips.cc/paper/2020/hash/9b8619251a19057cff70779273e95aa6-Abstract.html
- **引用次数**: 88+
- **关键内容**: 通过分析从网络中移除归一化层会发生什么,展示了如何在没有归一化层的情况下训练深度神经网络
- **BibTeX**:
```bibtex
@inproceedings{shao2020normalization,
  title={Is normalization indispensable for training deep neural network?},
  author={Shao, Jie and Hu, Kai and Wang, Changhu and Xue, Xiangyang and Raj, Bhiksha},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={13434--13444},
  year={2020}
}
```

## 8. 其他相关主题

### 8.1 神经形态硬件

**Davies et al. (2021)** - Advancing neuromorphic computing with Loihi
- **URL**: https://ieeexplore.ieee.org/abstract/document/9395703/
- **关键内容**: 综述了使用Intel的Loihi神经形态研究处理器获得的结果
- **BibTeX**:
```bibtex
@article{davies2021advancing,
  title={Advancing neuromorphic computing with loihi: A survey of results and outlook},
  author={Davies, Mike and others},
  journal={Proceedings of the IEEE},
  volume={109},
  number={5},
  pages={911--934},
  year={2021},
  publisher={IEEE}
}
```

### 8.2 编码方法

**Date et al. (2023)** - Encoding integers and rationals on neuromorphic computers
- **URL**: https://www.nature.com/articles/s41598-023-35005-x
- **引用次数**: 7+
- **关键内容**: 提出虚拟神经元抽象作为使用脉冲神经网络编码和添加整数和有理数的机制
- **BibTeX**:
```bibtex
@article{date2023encoding,
  title={Encoding integers and rationals on neuromorphic computers using virtual neuron},
  author={Date, Prasanna and others},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={8987},
  year={2023},
  publisher={Nature Publishing Group}
}
```
