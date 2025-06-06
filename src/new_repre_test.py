import datetime
import os
import pprint
import time
import threading
from numpy import e
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import json

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import copy


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    remark_str = getattr(args, "remark", "nop")
    unique_token = "{}__{}_{}".format(args.name, remark_str, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    
    training_pop_name = "-".join(args.train_tasks)
    if args.env == "sc2":
        logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "meta_test", args.task, training_pop_name, args.env_args["map_name"], args.name)
    elif args.env in ["grid_mpe", "easy_grid_mpe"]:
        logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "meta_test", args.task, training_pop_name, str(args.env_args["task_id"]), args.name)
    else:
        raise Exception(f"Unsupported env type {args.env}")
    exp_direc = os.path.join(logs_direc, "{}").format(unique_token)
    os.makedirs(exp_direc, exist_ok=True)

    if args.use_tensorboard: # and args.transfer_training:
        logger.setup_tb(exp_direc)
        
    # write config file
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(exp_direc, "config.json"), "w") as f:
        f.write(config_str)

    args.log_dir = exp_direc

    # get unique output file name
    if args.env == "sc2":
        output_dirname = os.path.join(dirname(dirname(abspath(__file__))), "outputs", "meta_test", args.task, training_pop_name, args.env_args["map_name"], args.name)
    elif args.env in ["grid_mpe", "easy_grid_mpe"]:
        output_dirname = os.path.join(dirname(dirname(abspath(__file__))), "outputs", "meta_test", args.task, training_pop_name, str(args.env_args["task_id"]), args.name)
    else:
        raise Exception(f"Unsupported env type {args.env}")
        
    os.makedirs(output_dirname, exist_ok=True)

    # set output dir
    args.output_dir = os.path.join(output_dirname, unique_token)
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(output_dirname, f"{unique_token}.out")

    # sacred is on by default
    logger.setup_sacred(_run, output_file)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    def save_json(dc, path):
        json_str = json.dumps(dc, indent=4)
        with open(path, "w") as json_file:
            json_file.write(json_str)
    
    # Do testing!
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    for i in range(n_test_runs):
        print("already to succeed")
        ret = runner.run(test_mode=True, evaluate_mode=True)
        if i == n_test_runs - 1:
            assert ret is not None, "Bug! ret should not be None"
            # save ret to certain paths
            test_stats, test_returns = ret
            print("-->> arive here!!!")
            save_json(test_stats, os.path.join(args.log_dir, "test_stats.json"))
            save_json(test_returns, os.path.join(args.log_dir, "test_returns.json"))

    if args.save_replay:
        runner.save_replay()
    
    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
    # 根据checkpoint_path加载预训练好的模型
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # 遍历目录下的所有文件
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        if getattr(args, "breakpoint_train", False):
            runner.t_env = timestep_to_load

        # 如果没有few_shot的话，就直接evaluate_sequential测试模型的性能。然后函数返回
        # 就不会有下面的学习了
        if not args.few_shot_adaptation:
            evaluate_sequential(args, runner)
            return
    else:
        raise Exception(f"Should not trained model to do meta_test!")


    # 程序能执行到到这里，说明有few_shot。
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # normal marl algorithm, e.g. QMIX, should not have pretrain phase
    pretrain_phase = getattr(args, "pretrain", False) # 这行代码的意思应该就是从args中获取"pretrain"这个参数，如果没有的话就返回False

    # 在这里进行迁移预训练
    learner.dynamic_train(get_task_return(args, logger))
    
    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False, pretrain_phase=pretrain_phase)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            # balance between parallel and episode run
            # 平衡parallel和episode的运行。因为
            terminated = False
            for _run in range(runner.batch_size): # 注意runner的batch_size其实就是batch_size_run
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)
                
                terminated = learner.train(episode_sample, runner.t_env, episode)
                if terminated:
                    # 说明预训练阶段结束
                    break
            
            # 如果预训练阶段已经结束
            if terminated:
                # 如果没有迁移阶段的话，就直接跳出循环
                if not args.transfer_training:
                    logger.console_logger.info("Only few shot adaptation!")
                    break
                else:
                    # 如果有迁移阶段的话。重新设置一些属性，继续循环学习
                    logger.console_logger.info("Finish pretrain and begin training for {} timesteps".format(args.t_max))
                    # Reset some properties in run.py, not need to modify episode, last_log_T, ...
                    pretrain_phase = False
                    start_time = time.time()
                    last_time = start_time
                    # Reset some properties about buffer and runner
                    buffer.clear()
                    runner.t_env = 0
                    continue
        
        # 这里只有预训练阶段结束后，pretrain_phase设为false，进入rl_train阶段时才会执行
        # Only for rl training phase
        if not pretrain_phase:            
            # # Test the performance of the deterministic version of the agent
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()
                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                local_results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
                if args.env == "sc2":
                    save_path = os.path.join(local_results_path, args.env, args.env_args["map_name"], args.name, "models", args.unique_token, str(runner.t_env))
                elif args.env == "mpe":
                    save_path = os.path.join(local_results_path, args.env, args.env_args["scenario_name"], args.name, "models", args.unique_token, str(runner.t_env))
                else:
                    save_path = os.path.join(local_results_path, args.env, args.name, "models", args.unique_token, str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

            episode += args.batch_size_run

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

    # test model performance
    evaluate_sequential(args, runner)
    logger.console_logger.info("Finished Evaluation")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config

# 加载新任务模型在各个任务上的回报
def get_task_return(args, logger):

    # 收集任务回报
    task_returns = []
    
    for model_path in args.model_paths:
        # 加载模型
        timesteps = []
        if not os.path.isdir(model_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return
        # Go through all files in args.checkpoint_path
        for name in os.listdir(model_path):
            full_name = os.path.join(model_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))
        timestep_to_load = max(timesteps)
        model_path = os.path.join(model_path, str(timestep_to_load))
        
        # 获取回报数组
        returns = []
        task = args.map_name
        
        task_args = copy.deepcopy(args)
        if task_args.env == "sc2":
            task_args.env_args["map_name"] = task
        elif task_args.env == "grid_mpe":
            task_args.env_args["task_id"] = task
            
        runner = r_REGISTRY[args.single_task_runner](args=task_args, logger=logger)
        env_info = runner.get_env_info()  # 获取环境信息
        task_args.n_agents = env_info["n_agents"]  # 智能体数量
        task_args.n_actions = env_info["n_actions"]  # 动作数量
        task_args.state_shape = env_info["state_shape"]  # 状态表示形状
        # scheme(状态表示)
        scheme = {
            "state": {"vshape": env_info["state_shape"]},  # 环境状态
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},  # 智能体私有观测
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},  # 智能体动作
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},  # 智能体可用动作
            "reward": {"vshape": (1,)},  # 奖励
            "terminated": {"vshape": (1,), "dtype": th.uint8},  # 终止标志
        }
        # groups(智能体分组)
        groups = {
            "agents": task_args.n_agents
        }
        # preprocess定义对动作的onehot编码处理
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }
        # 初始化经验回放池buffer
        buffer = ReplayBuffer(scheme, groups, task_args.buffer_size, env_info["episode_limit"] + 1,
                        preprocess=preprocess,
                        device="cpu" if task_args.buffer_cpu_only else task_args.device)
        # 创建MAC
        task_args.agent = task_args.single_task_agent
        task_args.mixer = "attn2_h"

        task_args.mac = task_args.single_task_mac
        
        mac = mac_REGISTRY[task_args.single_task_mac](buffer.scheme, groups, task_args)
        
        if args.use_cuda:
            mac.cuda()      

        mac.load_models(model_path)
        
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
        
        return_mean = runner.run()
        returns.append(return_mean)
        # 关闭runner
        runner.close_env()
        
        returns = th.tensor(returns, dtype=th.float)
        task_returns.append(returns)
    # 在第一维连接得到矩阵
    task_returns = th.stack(task_returns, dim=0)
    # 以为return_mean最大为20左右。所以除以20
    task_returns /= 20.0
    task_returns = task_returns.to(args.device)
    return task_returns