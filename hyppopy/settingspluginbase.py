# -*- coding: utf-8 -*-
#
# DKFZ
#
#
# Copyright (c) German Cancer Research Center,
# Division of Medical and Biological Informatics.
# All rights reserved.
#
# This software is distributed WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.
#
# See LICENSE.txt or http://www.mitk.org for details.
#
# Author: Sven Wanner (s.wanner@dkfz.de)

import abc

import os
import logging
from hyppopy.globals import DEBUGLEVEL
LOG = logging.getLogger(os.path.basename(__file__))
LOG.setLevel(DEBUGLEVEL)

from hyppopy.globals import SETTINGSPATH

from hyppopy.deepdict.deepdict import DeepDict


class SettingsPluginBase(object):
    _data = None
    _name = None

    def __init__(self):
        self._data = DeepDict()

    @abc.abstractmethod
    def convert_parameter(self):
        raise NotImplementedError('users must define convert_parameter to use this base class')

    def get_hyperparameter(self):
        return self.convert_parameter(self.data["hyperparameter"])

    def set(self, data):
        self.data.clear()
        self.data.data = data

    def read(self, fname):
        self.data.clear()
        self.data.from_file(fname)

    def write(self, fname):
        self.data.to_file(fname)

    def set_attributes(self, cls):
        attrs_sec = self.data[SETTINGSPATH]
        for key, value in attrs_sec.items():
            setattr(cls, key, value)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        return self._data

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            LOG.error(f"Invalid input, str type expected for value, got {type(value)} instead")
            raise IOError(f"Invalid input, str type expected for value, got {type(value)} instead")
        self._name = value
