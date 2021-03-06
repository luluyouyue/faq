# -*- coding: utf-8 -*-
import os
from configparser import ConfigParser
"""
存储全局变量和配置文件
"""

cur_dir = os.path.dirname(os.path.abspath(__file__))


# 定义全局变量
def _init():
    # type: () -> object
    global _global_dict
    _global_dict = {}
    conf = ConfigParser()
    conf.read(os.path.join(cur_dir, "../", "config", "my.conf"))
    _global_dict['conf'] = conf


def set_value(key, value):
    """ 定义一个全局变量 """
    _global_dict[key] = value


def get_value(key, defValue=None):
    """ 获得一个全局变量,不存在则返回默认值 """
    try:
        return _global_dict[key]
    except KeyError:
        return defValue


_init()

