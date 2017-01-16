# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:12:33 2016

@author: ilkhem
"""
__author__ = 'ilkhem'

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.policies import WhiteListRoundRobinPolicy

from config import config


def create_session(section='read'):
    cassandra_config = config['cassandra'][section]
    auth = PlainTextAuthProvider(username=cassandra_config['auth']['username'],
                                 password=cassandra_config['auth']['password'])
    contact_points = cassandra_config['contact_points']
    lbp = WhiteListRoundRobinPolicy(contact_points)
    cluster = Cluster(contact_points=contact_points, auth_provider=auth,
                      protocol_version=3, load_balancing_policy=lbp)

    session = cluster.connect(cassandra_config['keyspace'])
    # print('connected !')
    return session
