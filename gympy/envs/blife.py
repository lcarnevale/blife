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
        2       Network Outbound Traffic    0 B     +inf
        3       Buffer Size                 0       100

    Actions:
        Type: Discrete(3)
        Num     Action
        0       do nothing
        1       sample any 1s, deliver any 1s (streaming)
        2       sample any 10s, deliver any 10s (streaming)

    Reward:
        TODO("Ha senso?")
        Reward of 0 is awarded if the agent observes a battery power consumption
        less than 2W.
        Reward of -1 is awarded if the agent observes a battery power consumption
        grather than 2W.

    Starting State:
        Configure and start the white noise scenario. 

    Episode Termination:
        TODO        
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, battery_capacity=2000, num_discrete_actions=2):
        self.__step_counter = 0
        self.__battery_capacity = battery_capacity

        min_battery_voltage = 0
        max_battery_voltage = 5

        min_battery_current = 0
        max_battery_current = 2

        min_net_outbound = 0
        max_net_outbound = 1000000000

        min_buffer_size = 0
        max_buffer_size = 100

        self.__perform_action = [
            self.__action_do_nothing,
            self.__action_streaming_one_second,
            self.__action_streaming_ten_second
        ]

        self.__sensors = Sensors()
        self.__max_expected_runtime = 0.0

        low = np.array([min_battery_voltage, min_battery_current, min_net_outbound], dtype = np.float16)
        high = np.array([max_battery_voltage, max_battery_current, max_net_outbound], dtype = np.float16)

        self.action_space = spaces.Discrete(num_discrete_actions)
        self.observation_space = spaces.Box(low, high, dtype = np.float16)

    def reset(self):
        """Reset the state of the environment to an initial state.

        Returns:
            The observations space wrapped within a numpy array.
            See the class description for more details.
        """
        self.__step_counter = 0
        self.__configure_experiment()
        self.__reset_experiment()

        self.__observations = self.__next_observation()
        voltage = self.__observations[0]
        current = self.__observations[1]
        self.__power_last = current * voltage
        return np.array(self.__observations)

    def __configure_experiment(self):
        self.__scenario = WhiteNoiseScenario(
        'broker.mqttdashboard.com',
        '/fcrlab/distinsys/lcarnevale',
    )
    
    def __reset_experiment(self):
        self.__scenario.reset()

    def __next_observation(self):
        """Observe the environment.

        Returns:
            The observations space.
            See the class description for more details.
        """
        return [
            self.__sensors.get_bus_voltage(),
            self.__sensors.get_bus_current(),
            self.__sensors.get_net_bytes_sent()
        ]

    def step(self, action):
        """Execute one time step within the environment.

        Returns:

        """
        print("tentative to move action %d during step %d" % (action, self.__step_counter))
        done = False
        
        # perform action
        self.__valuate_action(action)
        self.__perform_action[action]()
        print("waiting action has effect ...")
        time.sleep(10)

        # collect observation
        self.__observations = self.__next_observation()
        voltage = self.__observations[0]
        current = self.__observations[1]
        power = current * voltage
        print("\tpower at step %d: %.3fmA" % (self.__step_counter, power))

        # calculate reward
        reward = self.__reward_function(power)
        print("\treward (energy_delta): %.3fmW" % (reward))

        # verify the termination condition
        expected_runtime = self.__battery_capacity / current
        self.__set_max_expected_runtime(expected_runtime)
        done = self.__termination_function(expected_runtime)
        print("\ttermination: %.3fh (runtime) >= 24h is %s" % (expected_runtime, done))
        print("\tmax runtime is %.3fh" % (self.__max_expected_runtime))
        
        self.__step_counter += 1
        return np.array(self.__observations), reward, done, {}
    
    def __valuate_action(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

    def __action_do_nothing(self):
        # print('do nothing')
        pass

    def __action_streaming_one_second(self):
        # print("sample any 1s, deliver any 1s (streaming)")
        self.__scenario.set_rate(1)

    def __action_streaming_ten_second(self):
        # print("sample any 10s, deliver any 10s (streaming)")
        self.__scenario.set_rate(10)

    def __reward_function(self, power):
        """Calculate the reward.

        Returns:
            float representing the energy delta in mJ
        """
        energy_delta = - (power - self.__power_last) * 1
        return energy_delta

    def __set_max_expected_runtime(self, expected_runtime):
        if expected_runtime > self.__max_expected_runtime:
            self.__max_expected_runtime = expected_runtime

    def __termination_function(self, expected_runtime):
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
        return expected_runtime >= 24