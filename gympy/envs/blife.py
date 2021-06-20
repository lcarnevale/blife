import gym
import time
import random
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .sensors import Sensors
from .scenarios import WhiteNoiseScenario

class BatteryLifetimeEnv(gym.Env):
    """
    Description:
        The blife environment simulates the behavior of a re-chargable
        lithium battery while the controlled device samples white noise
        and delivery it out in streaming or batch processes.
        The controlled device is a Raspberry Pi 3.

    Source:
        This environment and its documentation is available at
        https://github.com/lcarnevale/blife

    Observation:
        Type: Box(4)
        Num     Observation                 Min     Max
        0       Battery Voltage             0 V     5 V
        1       Battery Current             0 mA    2000 mA
        2       Network Outbound Traffic    0       +inf
        3       Buffer Size                 0       100

    Actions:
        Type: Discrete(3)
        Num     Action
        0       do nothing
        1       sample any 0.2s, deliver any 0.2s (streaming)
        2       sample any 1s, deliver any 1s (streaming)
        3       sample any 0.2s, deliver any 5s (batch)
        4       sample ant 1s, deliver any 5s (batch)

    Reward:
        TODO

    Starting State:
        Configure and start the white noise scenario. 

    Episode Termination:
        TODO        
    """

    def __init__(self, battery_capacity=2000, num_discrete_actions=5):
        self.__step_counter = 0
        self.__battery_capacity = battery_capacity

        min_battery_voltage = 0         #V
        max_battery_voltage = 5         #V

        min_battery_current = 0         #mA
        max_battery_current = 2000      #mA

        min_net_outbound = 0            #Bytes
        max_net_outbound = 1000000000   #Bytes

        min_buffer_size = 0             #Units
        max_buffer_size = 100           #Units

        self.__perform_action = [
            self.__action_do_nothing,
            self.__action_streaming_one,
            self.__action_streaming_two,
            self.__action_batch_one,
            self.__action_batch_two
        ]

        self.__sensors = Sensors()
        self.__max_expected_runtime = 0.0

        low = np.array([
            min_battery_voltage, min_battery_current, 
            min_net_outbound, min_buffer_size], 
            dtype = np.float16
        )
        high = np.array([
            max_battery_voltage, max_battery_current,
            max_net_outbound, max_buffer_size],
            dtype = np.float16
        )

        self.action_space = spaces.Discrete(num_discrete_actions)
        self.observation_space = spaces.Box(low, high, dtype = np.float16)


    def reset(self):
        """Reset the state of the environment to an initial state.

        Returns:
            The observations space.
            See the class description for more details.
        """
        self.__configure_experiment()
        self.__reset_experiment()
        return self.__observe()

    def __configure_experiment(self):
        self.__scenario = WhiteNoiseScenario(
            'broker.mqttdashboard.com',
            '/fcrlab/distinsys/lcarnevale',
        )
    
    def __reset_experiment(self):
        self.__scenario.reset()

    def __observe(self):
        """
        Returns:
            The observations space wrapped within a numpy array.
            See the class description for more details.
        """
        observations = self.__next_observation()
        voltage = observations[0]
        current = observations[1]
        self.__power_last_mean = current * voltage
        return np.array(observations)

    def __next_observation(self):
        """Observe the environment.

        Returns:
            The observations space.
            See the class description for more details.
        """
        return [
            self.__sensors.get_bus_voltage(),
            self.__sensors.get_bus_current(),
            self.__sensors.get_net_bytes_sent(),
            self.__scenario.get_queue_size()
        ]


    def step(self, action):
        """Execute one time step within the environment.

        Returns:

        """
        done = False
        
        # perform action
        self.__valuate_action(action)
        self.__perform_action[action]()
        
        # collect observation
        observation = self.__sub_observe()
        voltage_mean = observation[0]
        current_mean = observation[1]
        observation = np.array(observation)

        # calculate reward
        power_mean = voltage_mean * current_mean
        reward = self.__reward_function(power_mean)

        # verify the termination condition
        expected_runtime = self.__battery_capacity / current_mean
        self.__set_max_expected_runtime(expected_runtime)
        done = self.__termination_function(expected_runtime)
        # print("\t\ttermination: %.3fh (runtime) >= 8h is %s" % (expected_runtime, done))
        
        self.__power_last_mean = power_mean
        return observation, reward, done, {}
    
    def __valuate_action(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

    # action 0
    def __action_do_nothing(self):
        # print('do nothing')
        pass

    # action 1
    def __action_streaming_one(self):
        # print("sample any 1s, deliver any 1s (streaming)")
        self.__scenario.set_transmission('streaming')
        self.__scenario.set_rate(0.05)

    # action 2
    def __action_streaming_two(self):
        # print("sample any 10s, deliver any 10s (streaming)")
        self.__scenario.set_transmission('streaming')
        self.__scenario.set_rate(1)

    # action 3
    def __action_batch_one(self):
        self.__scenario.set_transmission('batch')
        self.__scenario.set_rate(0.05)

    # action 4
    def __action_batch_two(self):
        self.__scenario.set_transmission('batch')
        self.__scenario.set_rate(1)

    def __sub_observe(self, observation_rate=1, observation_window=15):
        """Observe the environment under specific conditions.
        """
        voltage_sub_history = list()
        current_sub_history = list()
        net_bytes_sent_sub_history = list()
        queue_size_sub_history = list()
        for _ in range(observation_window):
            observations = self.__next_observation()
            voltage_sub_history.append(observations[0])
            current_sub_history.append(observations[1])
            net_bytes_sent_sub_history.append(observations[2])
            queue_size_sub_history.append(observations[3])
            time.sleep(observation_rate)
        return [
            np.array(voltage_sub_history).mean(),
            np.array(current_sub_history).mean(),
            np.array(net_bytes_sent_sub_history).mean(),
            np.array(queue_size_sub_history).mean()
        ]

    def __reward_function(self, power_mean):
        """Calculate the reward.

        Returns:
            float representing the energy delta in mJ
        """
        energy_delta = - (power_mean - self.__power_last_mean) * 1
        return energy_delta

    def __set_max_expected_runtime(self, expected_runtime):
        if expected_runtime > self.__max_expected_runtime:
            self.__max_expected_runtime = expected_runtime

    def __termination_function(self, expected_runtime, target_expected_runtime=8.5):
        """Calculate the termination

        With the battery capacity and average current consumption,
        you can compute the expected runtime of your project by 
        solving the equation:
        Battery capacity (in mAh) / Average current consumption 
            (in mA) = Hours of expected runtime

        I expect the runtime is at least one day.
        
        Returns:
            bool representing the termination validation
        """
        return expected_runtime >= target_expected_runtime


    def close(self):
        self.__scenario.stop()