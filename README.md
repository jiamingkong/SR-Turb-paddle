# SR-Turb-paddle
Paddle re-implementation for Unsupervised deep learning for super-resolution reconstruction of turbulence

# 飞桨论文复现赛 第七期 Unsupervised deep learning for super-resolution reconstruction of turbulence

**什么是惊喜队**

## 简介

本项目主要是用以复现[Unsupervised deep learning for super-resolution reconstruction of turbulence](https://arxiv.org/abs/2007.15324)文章。原文作者在github中[开放了部分的源代码](https://github.com/HyojinKim-github/SR-Turb-CycleGAN)，但源代码是tensorflow构建的，同时代码中有少许不合理的地方，修复后也无法实现原文中的精度。

本文使用了Paddle重现该文章的工作，附带了一部分[Johns Hopkins Turbulence Databases](http://turbulence.pha.jhu.edu/datasets.aspx)的数据用作该项目的可视化验证。本项目与原始的代码实现的差异主要有如下：

- 扩大了网络的规模；
- 引入了residual connection；
- 对湍流速度场的数据进行了缩放，并让模型最后输出可以tanh激活；
- 重现了原文没有实现的数据IO等训练管线


## 原文介绍

湍流数值模拟结果的超分辨率一直是近年来的研究热点，通过深度学习模型将低分辨率的结果细化成高分辨率的结果可以大幅度地节省传统数值模拟的计算耗时。但是之前的研究都是使用了监督学习的方式，使用成对的高低分辨率数据进行训练，在一些流体力学的应用场景里面（例如Large eddy simulation），这样的成对数据可能不好获得。所以本文作者提出了使用CycleGAN进行无监督的方式来训练湍流的超分模型。

本文提出的超分辨率模型是基于CycleGAN训练的，训练的时候可以输入*不成对的高低分辨率湍流数值模拟结果*，假设我们称低分辨率数据为LR，高分辨率为HR，则该框架同时训练以下四个模型；

- $G(LR) \rightarrow \widehat{HR}$：超分辨率模型
- $F(HR) \rightarrow \widehat{LR}$：降采样模型
- $DX(LR) \rightarrow (0,1)$：低分辨率数据的辨别器
- $DY(HR) \rightarrow (0,1)$：高分辨率数据的辨别器

在训练的时候，我们将会同时输入不成对的低分辨率数据X，高分辨率数据Y：

1. 计算$\widehat{Y} = G(X), \widehat{X} = F(Y)$，
2. 计算辨别器损失：Loss_DX = DX(X) - DX(F(Y))， Loss_DY = DY(Y) - DY(G(X))
3. 计算循环损失: Loss_cycle = (Y - G(F(Y)))^2 + (X - F(G(X))^2
4. 计算生成器的损失：Loss_G = Loss_cycle + DY(G(X))， Loss_F = Loss_cycle + DX(F(Y))
4. 完成梯度的backprop

## 项目架构简单说明

- main.ipynb：本notebook，展示了项目的基础结构和一些流程；
- models/generators.py, models/discriminators.py：生成和辨别模型的定义
- utils/
    - loss.py：定义了损失的计算方法
    - dataloader.py：定义了数据的IO管线
    - functions.py：一些常用的函数，包括2X和0.5X层的定义

## 运行方法

直接点击“运行全部”，将会加载已经训练好的权重进行超分辨率重建；

在“开始训练”章节将两行代码取消注释，将会训练一个全新的模型