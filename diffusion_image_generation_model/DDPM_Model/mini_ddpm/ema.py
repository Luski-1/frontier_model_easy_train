from transformers import TrainerCallback, Trainer
import torch


class EMACallback(TrainerCallback):
    """
    钩子对象，用于更新EMA模型
    """

    def __init__(self, decay, copy_step, trainer: Trainer):
        self.decay = decay
        self.copy_step = copy_step
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        """
        该钩子方法是train过程的global step执行，不是Mini step
        kwargs包含model，如果是单卡训练，该model就是原版模型 | 如果是多卡训练，该model就被封装，最稳健的做法是使用trainer.accelerator来解包
        """
        model = kwargs.get("model")
        if model is None:
            print(f"model为空:{model}")
            return control
        
        diffusion_model = self.trainer.accelerator.unwrap_model(model)

        train_model = diffusion_model.model
        ema_model = diffusion_model.ema_model

        current_step = state.global_step    # 获取训练状态快照的global step

        # 阶段1：前2000步硬拷贝（预热期）
        if current_step <= self.copy_step:
            # 前2000步直接复制参数
            with torch.no_grad():
                for ema_p, model_p in zip(ema_model.parameters(), train_model.parameters()):
                    ema_p.copy_(model_p)
            if current_step <= 5 or current_step % 500 == 0:
                if args.process_index == 0:         # 这是为了适配多卡训练时避免重复打印日志，也适配单卡训练，也可以选择单卡训练时注释掉改行代码
                    print(f"[Step {current_step}]: EMA模型硬拷贝主模型参数 (预热期)")
        else:
            with torch.no_grad():
                for current_params, ema_params in zip(train_model.parameters(), ema_model.parameters()):
                    ema_params.data = ema_params.data * self.decay + (1 - self.decay) * current_params.data
            if current_step % 500 == 0:
                    if args.process_index == 0:
                        print(f"[Step {current_step}]: 通过指数平均移动方式更新EMA模型")

        return control