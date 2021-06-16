import gym
import random
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .sensors import Sensors
from .scenarios import WhiteNoiseScenario

class BatteryLifetimeEnv(gym.Env):
    """
    Description:
        This blife environments simulates the behavior of a re-chargable
        lithium battery while the controlled device samples white noise
        and delivery it out in streaming or batch processes.
        The controlled device is a Raspberry Pi 3.

    Source:
        tbd

    Observation:
        Type: Box(1)
        Num     Observation                 Min     Max
        0       Battery Power Consumption   0 W     20 W
        1       Network Outbound Traffic    0 B     +inf
        2       Buffer Size                 0       100

    Actions:
        Type: Discrete(2)
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
        Configure the white noise scenario. 
        It does not start at this time.

    Episode Termination:
        TODO        
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, num_discrete_actions=2):
        self.__step_counter = 0

        # TODO("Make them configurable from file.")
        min_power_consumption = 0
        max_power_consumption = 20
        self.__goal_power_consumption = 2

        min_net_outbound = 0
        max_net_outbound = 1000000000
        self.__goal_net_outbound = 2

        min_buffer_size = 0
        max_buffer_size = 100
        self.__goal_buffer_size = 0

        self.__actions_functions = [
            self.__action_do_nothing,
            self.__action_streaming_one_second,
            self.__action_streaming_ten_second
        ]

        self.__sensors = Sensors()

        low = np.array([min_power_consumption, min_power_consumption], dtype = np.float16)
        high = np.array([max_power_consumption, max_power_consumption], dtype = np.float16)

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
        self.__power_last = self.__observations[0]
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
            self.__sensors.get_power(),
            self.__sensors.get_net_bytes_sent()
        ]

    def step(self, action):
        """Execute one time step within the environment.

        Returns:

        """
        print("Moving action %d during step %d" % (action, self.__step_counter))
        done = False

        self.__assert_action_valid(action)
        self.__actions_functions[action]()
        self.__observations = self.__next_observation()
        self.__power = self.__observations[0]

        reward = self.__reward_function()
        self.__step_counter += 1
        self.__power_last = self.__power
        return np.array(self.__observations), reward, done, {}
    
    def __assert_action_valid(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

    def __action_do_nothing(self):
        # print('do nothing')
        pass

    def __action_streaming_one_second(self):
        # print("sample any 1s, deliver any 1s (streaming)")
        pass

    def __action_streaming_ten_second(self):
        # print("sample any 10s, deliver any 10s (streaming)")
        pass

    def __reward_function(self):
        energy_delta = - (self.__power - self.__power_last) * 1
        return energy_delta