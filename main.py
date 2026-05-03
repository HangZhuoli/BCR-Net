from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data
from torch.optim import lr_scheduler
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.utils.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.ctdet_trainer import CtdetTrainer as Trainer
from lib.utils.general import one_cycle, one_flat_cycle

'''
模型对于所有的输入合输出并没有做出任何的调整，选择使用图片原始尺寸进行处理
1、选择了opt.num_epochs=200
2、选择了opt.head_conv=64  // 应该在针对陨石坑数据集合中选择使用头部卷积为32，完成下采样


'''
def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    
    print('Creating model...')
    """
    在构造模型的过程中选择使用的参数，分别是模型的网络框架，模型的输出的结果，在针对陨石坑的数据集
    中，我们其实已经规定了模型的输出是1，因为我们陨石坑是针对单目标进行的识别，同时设置
    """
    model = create_model(opt.arch, opt.heads, opt.head_conv) 
    optimizer = torch.optim.Adam(model.parameters(), opt.lr) # (SGD=1E-2, Adam=1E-3)

    # Scheduler
    lrf = 0.001
    # lf = one_cycle(1, lrf, opt.num_epochs)  # cosine 1->hyp['lrf']
    # lf = one_flat_cycle(1, lrf, opt.num_epochs)  # flat cosine 1->hyp['lrf']        
    # lf = lambda x: 1.0
    lf = lambda x: (1 - x / opt.num_epochs) * (1.0 - lrf) + lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch = 0
    # 在实现验证eval的时候才需要加载模型进行验证和评估
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    trainer = Trainer(opt, model, optimizer, scheduler)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device) #将设备加载进入GPU中

    print('Setting up data...')
    # 创造可以迭代的验证数据集，便于神经网络的验证阶段
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'), 
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
 
    if opt.test:
        _, preds = trainer.val(0, val_loader, logger)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return
    
    # 创造训练迭代数据集，方便进行训练集的模型训练，依据标签读入
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        #默认是保存最后一次训练的权重 last，因为default的默认参数设置为0表示最基础的配置
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader, logger)
        logger.write('epoch: {} |'.format(epoch))

        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        # save the last 10 epoch
        if epoch >= 190 or (epoch%10 == 0 and epoch != 0):
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
            with torch.no_grad(): # 关闭梯度计算，因为处于验证阶段，需要关闭梯度计算，模型网络的权重是在训练阶段进行更新的
                log_dict_val, preds = trainer.val(epoch, val_loader, logger)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            
            """
            1、在验证阶段，如果验证集的loss值小于之前的loss值，则保存当前的模型网络
            在训练单个的陨石坑模型的检测过程中，同样也可以采取使用loss，其中loss的计算公式
            进行区分，前100轮需要关注模糊图像的恢复，需要包含HM、WH、Reg、deblur_loss
            100轮后关注目标检测主要目标，没有delur_loss
            """
            if log_dict_val[opt.metric] < best:  #在模型网络默认的计算中出现metric的值选择为loss，即损失大小，当然是损失值越小说明模型的性能最好
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
        logger.write('\n')


    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)