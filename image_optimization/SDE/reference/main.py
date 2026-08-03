# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

FLAGS = flags.FLAGS                                                                       # 设置全局单例对象                

# 类似argparse用于设置命令行参数，可以参看官方README.md的Usage段落
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)                            # 设置配置文件的路径                     
flags.DEFINE_string("workdir", None, "Work directory.")                                   # 设置工作目录
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")         # 设置启动模式
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")                     # 设置评估的目录，默认"eval"
flags.mark_flags_as_required(["workdir", "config", "mode"])                               # 设置必须设置的参数


def main(argv):
  if FLAGS.mode == "train":
    # 创建工作目录
    tf.io.gfile.makedirs(FLAGS.workdir)
    # 设置日志输出到stdout.txt文件中，并设置相关参数
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # 不对evaluate作人工注释，太累了
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
