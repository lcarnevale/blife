# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
This implementation does its best to follow the Robert Martin's Clean code guidelines.
The comments follows the Google Python Style Guide:
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md
"""

import psutil
from ina219 import INA219
from ina219 import DeviceRangeError

class Sensors:

    def __init__(self):
        SHUNT_OHMS = 0.1
        self.__ina = INA219(SHUNT_OHMS)
        self.__ina.configure()

    def get_pi_model(self):
        with open('/sys/firmware/devicetree/base/model', 'r') as f:
            model = f.read()
        return model.rstrip('\x00')

    def get_MAC(self, interface='wlan0'):
        # Return the MAC address of the specified interface
        try:
            str = open('/sys/class/net/%s/address' %interface).read()
        except:
            str = "00:00:00:00:00:00"
        return str[0:17]


    def get_bus_voltage(self):
        return self.__ina.voltage()

    def get_bus_current(self):
        return self.__ina.current()

    def get_power(self):
        return self.__ina.power() / 1000

    def get_shunt_voltage(self):
        return self.__ina.shunt_voltage() / 1000

    
    def get_cpu_percent(self):
        return psutil.cpu_percent(interval=1, percpu=False)

    def get_cpu_frequency(self):
        return psutil.cpu_freq(percpu=True)


    def get_mem_reading(self):
        mem = self.__get_virtual_memory()
        return {
            "memory_total":  mem.total,
            "memory_available":  mem.available,
            "memory_used":  mem.used,
            "memory_free":  mem.free
        }

    def __get_virtual_memory(self):
        return psutil.virtual_memory()

    
    def get_net_bytes_sent(self):
        return psutil.net_io_counters().bytes_sent

    def get_net_bytes_recv(self):
        return psutil.net_io_counters().bytes_recv

    def get_net_packets_sent(self):
        return psutil.net_io_counters().packets_sent

    def get_net_packets_recv(self):
        return psutil.net_io_counters().packets_recv