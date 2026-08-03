import torch

class EMA:

    def __init__(self, parameters, decay):

        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay    # 0.999
        self.num_updates = 0  # 用于记录当前更新步数
        self.shadow_params = [p.clone().detach()
                                for p in parameters if p.requires_grad]   # 复制并且断开梯度
        self.collected_params = []

    def update(self, parameters):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))   # 如果训练前期，decay会比0.999小
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))   # EMA = 0.9999 EMA + 0.0001 CUR

    def copy_to(self, parameters):
        # 遍历训练模型的参数权重，如果是存在梯度的参数，把EMA的权重复制到训练模型
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        # 保存训练模型的参数权重，稍等用于恢复
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        # 回复训练模型的参数权重
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']