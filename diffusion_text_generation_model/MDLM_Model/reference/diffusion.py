import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
  """
  常见的gumble_max的使用方式
  u = torch.rand_like(p)
  g = -torch.log(-torch.log(u + eps) + eps)
  return (torch.log(p) + g).argmax(dim=-1)
  """

  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor               # ddpm_cache
    self.gen_ppl_eval_model_name_or_path = self.config.eval.gen_ppl_eval_model_name_or_path   # gpt2-large
    self.antithetic_sampling = self.config.training.antithetic_sampling   # True
    self.importance_sampling = self.config.training.importance_sampling   # False
    self.change_of_variables = self.config.training.change_of_variables   # False

    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size                   # 设置mask token
      self.vocab_size += 1                                # vocab siez+1
    else:
      self.mask_index = self.tokenizer.mask_token_id      # 获得mask token的id

    self.parameterization = self.config.parameterization  # subs，MDLM的两种替换方法

    if self.config.backbone == 'dit':                     # 默认该分支
      self.backbone = models.dit.DIT(                     # DIT架构
        self.config, vocab_size=self.vocab_size)          # vocab_size传递，用于设置embedding的大小
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T                                # 0
    self.subs_masking = self.config.subs_masking          # False

    self.softplus = torch.nn.Softplus()     # ln(1 + e^x)
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # 设置eval阶段的tokenizer相关token，用于计算perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype) # 默认LogLinearNoise
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(   # EMA，不再人工注释
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr      # 3e-4
    self.sampling_eps = self.config.training.sampling_eps # 1e-3
    self.time_conditioning = self.config.time_conditioning  # False
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    """
    lighting.trainer会自动调用on_load_checkpoint，根据trainer.fit中ckpt_path参数来加载
    """
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    """
    lighting.trainer开启训练的正式入口on_train_start，就是为了设置EMA和sampler，随后调用on_train_epoch_start
    """

    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)    # 将ema的参数转换到指定device
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    
    # 调整dataloader的抽样器
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler  # 默认该分支，有人工注释
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler

    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      # 根据是否设置shuffle调整创建对象的参数
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      # 如果加载权重后的epoch相关参数不为空，让抽样器加载对应参数
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      # 对dataloader更换抽样器dl_sampler
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    """
    logits: 模型预测的结果[B,S,V]
    xt: 加噪结果[B, S]
    """
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity  # RB2，xt预测为MASK的logit永远为负无穷
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1, # log( exp(x_i) / ∑exp(x...) ) = x_i - log(∑exp(x...)) 直接获取logP
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)    # RB1，xt中非MASK位置永远不变
    logits[unmasked_indices] = self.neg_infinity  # 在xt的[B,S]维度中mask位置直接负无穷
    logits[unmasked_indices, xt[unmasked_indices]] = 0  # 在xt的V维度，指定非mask token的logP=0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:  # MDLM默认分支，返回零向量，因为文献中说明模型无需显式接受时间t作为条件参数，因为时间t已经天然隐含在x_t中，即MASK越多t越大
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    # sigma是加噪比例e^(-σ(t))【即a_t】的σ(t)，[B,1]
    sigma = self._process_sigma(sigma)  # 得到零向量，因为文献中说明模型无需显式接受时间t作为条件参数，因为时间t已经天然隐含在x_t中，即MASK越多t越大
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits, # 进行文献中的RB1和RB2替换logits（logP）
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:               # 默认有
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask) # 真正计算loss
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    """
    lighting.trainer在训练中每个epoch都会调用此方法，目的就是把模型切换为train模式，随后调用training_step
    """
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    """
    lighting.trainer在每个train batch都会调用此方法，计算loss
    batch: dataloader返回的数据
    """
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    """
    lighting.trainer在训练中每个epoch都会调用此方法，目的就是把模型切换为eval模式，随后调用validation_step，不再进行注释了
    """
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    """
    lighting.trainer开始训练前，会先执行configure_optimizers
    """
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558

    # 设置优化器
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,        # 3e-4
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)
    # 利用hydra实例化transformers.get_constant_schedule_with_warmup
    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    # 就是从均匀分布抽样z，如果z比MASK概率数值小，就换成MASK
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    """
    x: 输入的x_t
    t: 当前时间t，[B,]，浮点数
    dt: 步长
    p_x0: None or ?

    参考文献A.2.2，模型的后验公式p(z_s | z_t = m) = Cat(z_s | Q_(t|s) @ M(one-hot) * (Q_s)^T @ x_θ / (M(one-hot)^T) @ (Q_t)^T @ @ x_θ)
    参考文献A.2.2，最后化简得到：
    p_θ(z_s = x | z_t = m) = (a_s - a_t) * <x_θ, x> / (a_t * <x_θ, m> + 1 - a_t)
    p_θ(z_s = m | z_t = m) = (a_s * <x_θ, m> + 1 - a_s) / (a_t * <x_θ, m> + 1 - a_t)

    利用RB2和RB1，那么<x_θ, m> = 0，最终化简得到
    p_θ(z_s = x | z_t = m) = (a_s - a_t) * <x_θ, x> / (1 - a_t)
    p_θ(z_s = m | z_t = m) = (1 - a_s) / (1 - a_t)
    """
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)  # 获取sigma

    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1

    move_chance_t = t[:, None, None]        # 当前时间t，此时也是1 - a_t
    move_chance_s = (t - dt)[:, None, None] # 上一步时间s，此时也是1 - a_s
    assert move_chance_t.ndim == 3, move_chance_t.shape

    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp() # 预测x0的概率情况
    
    assert move_chance_t.ndim == p_x0.ndim

    # move_chance_t - move_chance_s = (1 - a_t) - (1 - a_s) = a_s - a_t
    # q_xs = (a_s - a_t) * <x_θ, x>
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    # 设置mask位置为1 - a_s
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)  # gumble_max采样
    
    # p_x0作为缓存，只要_x没有新解码任何一个token，就可以重复使用
    # copy_flag * x + (1 - copy_flag) * _x，就是维持原有非mask的token
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):

    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)

    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)

    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape

    move_chance_t = 1 - torch.exp(-sigma_t)   # 获取时间t的加噪MASK概率
    move_chance_s = 1 - torch.exp(-sigma_s)   # 获取时间s的加噪MASK概率
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]

    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning) # 传递时间
    assert move_chance_t.ndim == log_p_x0.ndim

    # 以下部分参考_ddpm_caching_update即可
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    # num_steps = 128
    # eps = 1e-5
    batch_size_per_gpu = self.config.loader.eval_batch_size

    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps

    x = self._sample_prior(   # 获得全为MASK token id 的xt
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)  # 获得时间点的列表
    dt = (1 - eps) / num_steps  # 获得时间点的间隔
    p_x0_cache = None

    for i in range(num_steps):

      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)  # 获取时间t
      
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        # 如果x_next和x有差异（浮点数），或者模型是时间参数依赖的
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # 清空缓存
          p_x0_cache = None
        x = x_next
      else:
        # 这个是score-matching的方式，不做注释
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      # 设置时间0
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      # 这个是score-matching的方式，不做注释
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]  # unet_conditioning = 0
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason

    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      
    self.backbone.eval()
    self.noise.eval()

    samples = self._sample(num_steps=num_steps, eps=eps)  # 128

    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    """
    n: batch_ssize
    """
    _eps_t = torch.rand(n, device=device) # 均匀分布抽样时间t
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n # [1, 2/n, ..., (n-1)/n] 作为偏移值
      _eps_t = (_eps_t / n + offset) % 1  # 均匀分布抽样的时间t / n 那么方差缩小N^2，在[0, 0.25)，随后增加偏移值得到每个1/N的小区间都有一个时间t | %1避免超出1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps  # 平移到 [eps, 1]，分布下限很重要
    if self.importance_sampling:  # 默认不走该分支
      return self.noise.importance_sampling_transformation(t) # 逻辑就是通过重要性采样把均匀分布的t转换为其他分布的t
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:     # 1024
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar': # AR分支，就是右移
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:   # 默认该分支，没变化
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):

    t = self._sample_t(x0.shape[0], x0.device)  # 抽样时间t
    if self.T > 0:  # 默认不进该分支
      t = (t * self.T).to(torch.int)   # 离散：乘 T 再取整 → 整数索引
      t = t / self.T                   # 还原到 [0, (T-1)/T] 范围
      t += (1 / self.T)                # 平移到 {1/T, 2/T, ..., 1}

    if self.change_of_variables:  
      unet_conditioning = t[:, None]  
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else: # 默认该分支
      sigma, dsigma = self.noise(t) # 作者定义a_t = exp(-σ(t))，sigma就是σ(t)，dsigma就是σ(t)导数
      unet_conditioning = sigma[:, None] # [B, 1]
      move_chance = 1 - torch.exp(-sigma[:, None])  # 获得1-a_t，就是加噪为mask的概率，因为a_t实际就是1-t，那么1-a_t就是t

    xt = self.q_xt(x0, move_chance) # 加噪得到xt
    model_output = self.forward(xt, unet_conditioning)  # 得到模型预测的logP，已经RB1和RB2替换
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:  # 默认不进该分支
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(         # 获取真实label对应的logP
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:  # 默认不开启
      # 计算结果torch.log1p(- torch.exp(- self.noise.sigma_min) ≈ log(eps)
      # 正如importance_sampling_transformation方法中提到，如果开启重要性采样，会把损失的权重从1/t变成-ln(eps)，所以以下代码又是为了适配各种a_t形式做出的调整
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))  
    
    # dsigma = (1 - self.eps) / (1 - (1 - self.eps) * t)
    # 原本sigma = -log(1 - (1 - eps) * t)
    # torch.expm1(sigma) = exp(-log(1 - (1 - eps) * t)) - 1 = 1 / [1 - (1 - eps) * t] - 1 = (1 - eps) * t/ [1 - (1 - eps) * t]
    # dsigma / torch.expm1(sigma) = 1/t
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,     # output_tokens = None
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    # 自回归就是常见的右移取loss
    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens) # 默认该分支
    
    nlls = loss * attention_mask  # 该attention mask用于挑选非padding
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    # stride_length=1 | num_strides=1
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt) # 1000 + 1 就是总的时间点
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):  # 2
      p_x0_cache = None
      x = self._sample_prior(     # 得到[1, 1024]维度的全mask token id
        n_samples,    # 1
        self.config.model.length).to(self.device) # 1024
      
      # 把上一次预测结果的后半段，替换当前MASK序列的前半段
      if target is not None:
        x[:, : -stride_length] = target

      for i in range(num_steps + 1):                # 遍历1001次
        # p_x0_cache就是模型预测的x0概率，可以用于缓存
        # x_next是抽样后的状态
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        
        # 如果x_next与x非近似相等，或者模型的输入依赖时间条件
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # 清空缓存
          p_x0_cache = None
          sampling_steps += 1

        x = x_next
      # 再执行一次时间t=0的预测，随后直接取max避免还有mask
      x = self.forward(x, 0 * ones).argmax(dim=-1)

      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy()) # 第1轮挤出的1个token  → shape [1, 1] | 第2轮挤出的1个token  → shape [1, 1] ...
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())    # 最后的prefix → shape [1, 1023]
    intermediate_text_samples = []

    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]  # np.concatenate(intermediate_tokens, axis=1)得到[1, 1025]，随后去掉首个token
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)  # == self.tokenizer.eos_token_id 得到属于eos_token_id的布尔数组，cumsum累计就能知道连续为False的片段在哪里
    
    for i in range(2, len(intermediate_tokens) + 1):      # 遍历执行tokenizer解码
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
      
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    
    if self.ema:
      # 把训练模型的权重保存
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      # 把EMA模型的权重转移到训练模型
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      
    self.backbone.eval()
    self.noise.eval()
    # sampling_steps：执行了多少步
    # samples：长度递进的文本片段
    # sequence_lengths：有效的文本长度
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,  # 1
      num_strides=num_strides,      # 1
      dt=dt)
    
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths
