import agent
import torch
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
from common import interface
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse


def my_setup_logging(level):
    logger = logging.getLogger()
    return logger
    logger.setLevel(level)
    log_formatter = logging.Formatter(
        "%(asctime)s (%(levelname)s): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Send INFO to stdout
    handler_out = logging.StreamHandler(sys.stdout)
    handler_out.addFilter(
        SeverityLevelBetween(logging.INFO, logging.WARNING)
    )
    handler_out.setFormatter(log_formatter)
    logger.addHandler(handler_out)

    # Send WARNING (and higher) to stderr
    handler_err = logging.StreamHandler(sys.stderr)
    handler_err.setLevel(logging.WARNING)
    handler_err.setFormatter(log_formatter)
    logger.addHandler(handler_err)
    return logger

class MyTester(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''

    def train(self):
        pass

    def train_test(self, e):
        pass

    def __init__(self, logger, test_scenario, gpu=0, cpu=False, name="tsc"):
        # self.path = os.path.join('configs/sim', Registry.mapping['command_mapping']['setting'].param['network'] + '.cfg')
        self.path = os.path.join('configs/sim', test_scenario + '.cfg')
        self.save_replay = Registry.mapping['world_mapping']['setting'].param['saveReplay']
        if self.save_replay:
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                self.dir = Registry.mapping['world_mapping']['setting'].param['dir']
                self.replay_file_dir = os.path.dirname(
                    Registry.mapping['world_mapping']['setting'].param['roadnetLogFile'])
                if not os.path.exists(os.path.join(self.dir, self.replay_file_dir)):
                    os.makedirs(os.path.join(self.dir, self.replay_file_dir))
        self.seed = Registry.mapping['command_mapping']['setting'].param['seed']
        self.logger = logger
        # self.debug = args['debug']
        self.name = name
        self.cpu = cpu
        self.epoch = 0
        self.step = 0
        self.metric = None
        self.env = None
        self.world = None
        self.agents = None

        if torch.cuda.is_available() and not self.cpu:
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True

        self.load()
        self.create()
        # super().__init__(logger=logger, gpu=gpu, cpu=cpu, name=name)


        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # replay file is only valid in cityflow now.
        # TODO: support SUMO and Openengine later

        # TODO: support other dataset in the future
        # self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
        #     os.path.join(Registry.mapping['logger_mapping']['path'].path,
        #                  Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        # )
        # self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        # self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
        #                              Registry.mapping['logger_mapping']['setting'].param['log_dir'],
        #                              os.path.basename(self.logger.handlers[-1].baseFilename).rstrip(
        #                                  '_BRF.log') + '_DTL.log'
        #                              )

    def create_world(self):
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],
            interface=Registry.mapping['command_mapping']['setting'].param['interface'])
        print(f'Test world: {self.path}')

    def create_metrics(self):
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(
                Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](
                    self.world, i))

        # for magd agents should share information
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        self.env = TSCEnv(self.world, self.agents, self.metric)


    # def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
    #     res = f"{Registry.mapping['model_mapping']['setting'].param['name']:<12}\t{mode:<8}\t{step:<6}\t" \
    #           + f"{travel_time:<20}\t{loss:<20}\t{cur_rwd:<20}\t{cur_queue:<20}\t{cur_delay:<20}\t{cur_throughput:<20}"
    #     log_handle = open(self.log_file, "a")
    #     log_handle.write(res + "\n")
    #     log_handle.close()

    def test(self):
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()

        Registry.mapping['logger_mapping']['path'].path = Registry.mapping['logger_mapping']['path'].path.replace(
            'tsc_test', 'tsc')
        # print(Registry.mapping['logger_mapping']['path'].path);exit()

        load_model = Registry.mapping['model_mapping']['setting'].param.get('load_model')
        if load_model:
            print(f"Load model: {Registry.mapping['logger_mapping']['path'].path}")
        else:
            print(f"No need to load model")
        if load_model and load_model is not False:
            for ag in self.agents:
                ag.load_model(self.episodes)
        attention_mat_list = []
        obs = self.env.reset()
        for a in self.agents:
            a.reset()

        get_time = time.process_time
        pre_env_time = get_time()
        decision_time = 0.0#;self.action_interval = 1
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []

                pre_decision_time = get_time()
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                decision_time += get_time() - pre_decision_time

                actions = np.stack(actions)
                rewards_list = []

                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))

                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
            if all(dones):
                break
        env_time = get_time() - pre_env_time
        print(f'Simulation cost: {decision_time:.4f}/{env_time:.4f}|{decision_time / env_time * 100:.4f}%')
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (
        self.metric.real_average_travel_time(), \
        self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
        return self.metric


def eval_train_test(agent_name, train_scenario, test_scenario, seed):
    args = argparse.Namespace(
        thread_num=2,
        ngpu='1',
        seed=seed,
        debug=True,
        interface='libsumo',
        delay_type='apx',
        task='tsc_test',
        agent=agent_name,
        world='cityflow',
        network=train_scenario,
        dataset='onfly'
    )

    config, _ = build_config(args)

    # rigister configs
    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)  # register logger path
    interface.World_param_Interface(config)
    interface.Logger_path_Interface(config)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)

    if config['model'].get('graphic', False):
        param = Registry.mapping['world_mapping']['setting'].param
        if config['command']['world'] in ['cityflow', 'sumo']:
            roadnet_path = param['dir'] + param['roadnetFile']
        else:
            roadnet_path = param['road_file_addr']
        interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class

    logger = my_setup_logging(logging.INFO)

    tester = MyTester(logger, test_scenario)
    # task = Registry.mapping['task_mapping'][Registry.mapping['command_mapping']['setting'].param['task']](tester)
    ret = tester.test().real_average_travel_time()
    print(f'[{agent_name}]: {train_scenario} => {test_scenario} ({seed}) : {ret}')
    return ret


if __name__ == '__main__':
    SCENARIOS = ['jinan', 'jinan_2000', 'jinan_2500', 'cityflow4x4', 'cityflow4x4_5816', 'cityflow7x28', 'cityflow7x28_double']

    train_s = 'jinan'
    test_s = 'jinan'

    data = []
    for seed in [15]:
        # m = eval_train_test('maxpressure', train_s, test_s, seed)
        # m = eval_train_test('efficient_mp', train_s, test_s, seed)
        m = eval_train_test('advance_mp', train_s, test_s, seed)
        # m = eval_train_test('g2p', train_s, test_s, seed)
        # print(m)
        data.append(m)
    print(data)




