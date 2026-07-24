import argparse

def get_args():
    parser = argparse.ArgumentParser(description="FlowMatch Training Config")

    # 路径配置
    parser.add_argument("--data-root", type=str, default="/workspace/data/img_align_celeba/img_align_celeba",
                        help="celeba对齐人脸数据集根目录")
    parser.add_argument("--save-dir", type=str, default="./checkpoints",
                        help="模型权重保存目录")
    
    # 数据参数
    parser.add_argument("--image-size", type=int, default=104)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=4, help="控制数据占用的内存不会被交换到磁盘中，提高加载速度")
    parser.add_argument("--edm-train-time", action="store_true", default=True, help="使用EDM的训练期间时间t的抽样方法，即t更偏向1")

    # 训练超参
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=100)

    # 采样相关
    parser.add_argument("--sample-every", type=int, default=1,
                        help="每多少个epoch执行一次采样")
    parser.add_argument("--num-sample-images", type=int, default=9,
                        help="每次采样生成图片数量")
    parser.add_argument("--flow-steps", type=int, default=200,
                        help="采样时步数")
    parser.add_argument("--edm-eval-time", action="store_true", default=True, help="使用EDM的推理期间时间t的取点方法")

    # 模型参数
    parser.add_argument("--base-ch", type=int, default=128,
                        help="模型的基础通道数")
    parser.add_argument("--time-ch", type=int, default=512,
                        help="时间参数的通道数")
    parser.add_argument("--num-res-blocks", type=int, default=2,
                        help="模型的每一层拥有的残差块")
    parser.add_argument("--ema", action="store_true", default=True, help="是否开启EMA模型")
    parser.add_argument("--decay", type=float, default=0.9999, help="EMA模型的衰减系数")
    

    args = parser.parse_args()
    return args