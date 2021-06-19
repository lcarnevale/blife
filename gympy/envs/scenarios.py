# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
This implementation does its best to follow the Robert Martin's Clean code guidelines.
The comments follows the Google Python Style Guide:
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

__copyright__ = 'Copyright 2021, FCRLab at University of Messina'
__author__ = 'Lorenzo Carnevale <lcarnevale@unime.it>'
__credits__ = ''
__description__ = ''

import os
import time
import numpy
import socket
import logging
import argparse
import persistqueue
import numpy as np
from threading import Thread
import paho.mqtt.client as mqtt


class WhiteNoiseScenario:
    # TODO: builder pattern
    def __init__(self, host, topic, port=1883, num_samples=1000, rate=1, verbosity=False):
        # self.__setup_logging(verbosity)

        self.__mean = 0
        self.__std = 1
        self.__num_samples = num_samples
        self.__rate = rate

        self.__host = host
        self.__port = port
        self.__topic = topic

        self.__queue = persistqueue.SQLiteQueue('data', multithreading=True, auto_commit=True)

        self.__transmission = 'streaming'
        self.__client = self.__configure_mqtt_client()

        self.__client.connected_flag = False
        self.__client.loop_start()
        # print("connecting to broker ...")
        self.__client.connect(host, port, keepalive=60)
        while not self.__client.connected_flag: # wait in loop
            # print("waiting for connection ...")
            time.sleep(1)
    
    def __setup_logging(self, verbosity):
        format = "%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s"
        filename='log/population.log'
        datefmt = "%d/%m/%Y %H:%M:%S"
        level = logging.INFO
        if (verbosity):
            level = logging.DEBUG
        logging.basicConfig(filename=filename, filemode='a', format=format, level=level, datefmt=datefmt)

    def __configure_mqtt_client(self):
        # defining the MQTT client and callbacks
        client_name = '%s-sub' % (socket.gethostname())
        client = mqtt.Client(client_name)
        client.on_connect = self.__on_connect
        client.on_disconnect = self.__on_disconnect
        client.on_publish = self.__on_publish
        return client
    
    def __on_connect(self, client, _, __, rc):
        """Connection's callback
        The callback used when the client receives a CONNACK response from the server.
        Subscribing to on_connect() means that the connection is renewed when it is lost.
        Args:
            client (obj:'paho.mqtt.client.Client'): the client instance for this callback
            rc (int): is used for checking that the connection was established
        """
        return_code = {
            0: "Connection successful",
            1: "Connection refused – incorrect protocol version",
            2: "Connection refused – invalid client identifier",
            3: "Connection refused – server unavailable",
            4: "Connection refused – bad username or password",
            5: "Connection refused – not authorised",
        }
        if (rc == 0):
            logging.info(return_code[rc])
            self.__client.connected_flag = True
        else:
            logging.error(return_code[rc])
 
    def __on_disconnect(self, client, _, rc):
        """MQTT Disconnection callback
        Args:
            client (obj:'paho.mqtt.client.Client'): the client instance for this callback
            rc (int): is used for checking that the disconnection was done
        """
        logging.info('Disconnection successful %s' % (rc))

    def __on_publish(self, client, _, result):
        """Publish's callback
        The callback for when a PUBLISH message is sent to the server
		Args:
			client(obj:'paho.mqtt.client.Client'): the client instance for this callback;
			result():.
		"""
        logging.debug('Data published')


    def start(self):
        self.__stop_threads_writer = False
        self.__stop_threads_reader = False
        self.__thread_writer = Thread(
            target = self.__writer, 
            args = (lambda : self.__stop_threads_writer, )
        )
        self.__thread_reader = Thread(
            target = self.__reader,
            args = (lambda : self.__stop_threads_reader, )
        )
        self.__thread_writer.start()
        self.__thread_reader.start()

    def __writer(self, stop):
        try:
            while True:
                if (stop()):
                    # print('\tstop sampling')
                    break
                sample = self.__get_sample()
                # print('\tsampling right now (rate: %s)!' % (self.__rate))
                if (self.__transmission == 'streaming'):
                    # print('\tsending streming right now (rate: %s)!' % (self.__rate))
                    self.__deliver_sample(sample)
                elif (self.__transmission == 'batch'):
                    # print('\tqueueing batch right now (rate: %s)!' % (self.__rate))
                    self.__queue.put(sample)
                time.sleep(self.__rate)
        except KeyboardInterrupt:
            pass

    def __reader(self, stop):
        try:
            while True:
                if (stop()):
                    # print('\tstop dequeueing')
                    break
                while not self.__queue.empty():
                    # print('\tdequeueing batch right now!')
                    sample = self.__queue.get()
                    # print('\tsending batch right now!')
                    self.__deliver_sample(sample)
                time.sleep(5)
        except KeyboardInterrupt:
            pass

    def stop(self):
        self.__stop_threads_writer = True
        self.__stop_threads_reader = True
        self.__client.disconnect()
        time.sleep(0.2)


    def __get_sample(self):
        return numpy.random.normal(
            self.__mean,
            self.__std,
            size = self.__num_samples
        )

    def __deliver_sample(self, sample):
        sample = sample.tobytes()
        self.__client.publish(self.__topic, sample)
        # print("\tsend new data out")


    def reset(self):
        self.stop()
        time.sleep(0.1)
        self.start()

    
    def set_rate(self, rate):
        self.__rate = rate

    def set_transmission(self, transmission):
        self.__transmission = transmission

    def get_queue_size(self):
        return self.__queue.size



if __name__ == "__main__":
    description = ('%s\n%s' % (__author__, __description__))
    epilog = ('%s\n%s' % (__credits__, __copyright__))
    parser = argparse.ArgumentParser(
        description = description,
        epilog = epilog
    )

    parser.add_argument('-v', '--verbosity',
                        dest='verbosity',
                        help='Logging verbosity level',
                        action="store_true")

    options = parser.parse_args()
    verbosity = options.verbosity
    logdir_name = 'log'
    if not os.path.exists(logdir_name):
        os.makedirs(logdir_name)

    white_noise_scenario = WhiteNoiseScenario(
        'broker.mqttdashboard.com',
        '/fcrlab/distinsys/lcarnevale',
        verbosity=verbosity
    )
    white_noise_scenario.start()
    time.sleep(5)
    white_noise_scenario.set_transmission('streaming')
    white_noise_scenario.set_rate(3)
    time.sleep(5)
    white_noise_scenario.set_transmission('batch')
    white_noise_scenario.set_rate(0.2)
    time.sleep(6)
    white_noise_scenario.stop()