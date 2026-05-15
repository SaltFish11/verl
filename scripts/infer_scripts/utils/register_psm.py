#  python register_psm.py register tiktok.aiic.large_model
import hashlib
import ipaddress
import logging
import os
import random
import socket
import subprocess
import threading
import time
from datetime import datetime
from functools import partial

import requests

# from serve.utils.serve_utils import get_local_ip, get_logger, trace_id_var

import logging

logger = logging.getLogger(__name__)

# logger = get_logger()

def register_service(psm, port):
    result = subprocess.run(['/opt/tiger/consul_deploy/bin/go/sd', 'up', psm, str(port), '--dual-stack',
                             '--tags', '{"env":"prod","weight":"10"}'],
                            capture_output=True, text=True)
    # logger.info(f'register service: {result.stdout}')
    print(f'register service: {result.stdout}')
    print(f'register service: {result.stderr}')

def deregister_service(psm, port):
    result = subprocess.run(['/opt/tiger/consul_deploy/bin/go/sd',
                            'down', psm, str(port)], capture_output=True, text=True)
    print(f'deregister service: {psm} {port}, stdout: {result.stdout}')

def is_ipv6(ip):
    """
    判断给定的IP地址是否为IPv6地址.

    Args:
        ip (str): 要检查的IP地址.

    Returns:
        bool: 如果IP地址是IPv6，返回True；否则返回False.
    """
    try:
        ipaddress.IPv6Address(ip)
        return True
    except ipaddress.AddressValueError:
        return False

def fetch_service_info():
    psm = os.getenv("SERVER_PSM")
    cluster = os.getenv("SERVER_CLUSTER", "default")
    port = os.getenv("ARNOLD_WORKER_0_PORT", 8888)
    return psm, port, cluster

class SDClient:
    BASE_DC = ["lf", "hl", "lq", "yg"]
    BASE_SG_DC = ["alisg", "sg1", "my2"]
    BASE_MALIVA_DC = ["maliva"]
    BASE_BE1A_DC = ["be1a"]
    BASE_I18N_DC = BASE_SG_DC + BASE_MALIVA_DC + BASE_BE1A_DC
    BASE_TTP_DC = ["useast5"]
    BASE_BOE_DC = ['boe']
    DC_DICT = {
        "cn": BASE_DC,
        "sg": BASE_SG_DC,
        "maliva": BASE_MALIVA_DC,
        "boe": BASE_BOE_DC,
        "ttp": BASE_TTP_DC,
        'be1a': BASE_BE1A_DC,
        'i18n': BASE_I18N_DC
    }

    def __init__(self, psm, cluster=None, idc=None, refresh=True, refresh_period=60, **kwargs):
        self.psm_list = psm.split(",")
        self.cluster = cluster
        self.refresh_period = refresh_period
        self.idc = idc

        self.ip_port_list = self._lookup_psm(self.psm_list)
        if len(self.ip_port_list) == 0:
            msg = f"all service for psm {self.psm_list} is not alive or disconnected, please deploy the service and check it."
            logger.error(msg)

        if refresh:
            self.refresh_thread = threading.Thread(target=partial(
                self.refresh_list_periodically, self.refresh_period))
            self.refresh_thread.daemon = True
            self.refresh_thread.start()

    def get_host(self, psm, cluster=None, **kwargs):
        res = []
        idc_list = self.DC_DICT[self.idc] if self.idc else self.DC_DICT[os.environ.get(
            "IDC", "i18n").lower()]
        if cluster is None:
            cluster = self.cluster
        for idc in idc_list:
            try:
                instances = lookup_name(f'{psm}.service.{idc}', cluster)
                for instance in instances:
                    if 'Host' not in instance or 'Port' not in instance:
                        continue
                    res.append((instance['Host'], instance['Port']))
            except Exception as ex:
                logger.error(f'lookup {psm} failed, {ex}')
                continue
        return res

    def refresh_list_periodically(self, periodic_interval_seconds=60):
        while True:
            time.sleep(periodic_interval_seconds)  # 休眠
            self._refresh()

    def _refresh(self):
        self.ip_port_list = self._lookup_psm(self.psm_list)
        cur_time = datetime.now().strftime("%Y:%M:%D %H:%M:%S")
        logger.debug(
            f"psm [{self.psm_list}] refreshed: at {cur_time}, alive instance count: {len(self.ip_port_list)}")

    def _lookup_psm(self, psm_list):
        ip_port_list = []
        for psm in psm_list:
            host_port_instance = self.get_host(psm, self.cluster)
            if len(host_port_instance) == 0:
                continue

            for instance in host_port_instance:
                ip = instance[0]
                port = instance[1]
                if not self.is_ip_accessible(ip, port):
                    logger.error(f"{ip}:{port} is not accessible")
                    continue
                ip_port_list.append((ip, port))
        return ip_port_list

    def get_all(self):
        return self.ip_port_list

    def is_ip_accessible(self, ip, port=80, timeout=2):
        sock = None
        try:
            sock = socket.create_connection((ip, port), timeout)
            return True
        except (socket.timeout, socket.error):
            return False
        finally:
            if sock:
                try:
                    sock.close()
                except:
                    pass

    def get_one(self):
        # should add check here, if the ip is not connected should change to another ip
        copy_ip_port_list = self.ip_port_list.copy()
        random.shuffle(copy_ip_port_list)  # 随机打乱列表顺序
        for ip_port in copy_ip_port_list:
            ip, port = ip_port  # 假设 ip_port 是一个元组 (ip, port)
            if self.is_ip_accessible(ip, port):
                return ip_port  # 返回可访问的 IP 和端口
            else:
                # 如果不可访问，从原列表中移除
                self.ip_port_list.remove(ip_port)
        raise ValueError(f"all service for psm {self.psm_list} is not alive or disconnected, please deploy the service and check it.")

    def _check_service_state(self, ip, port):
        # try:
        #     res = requests.get(f"http://{ip}:{port}/health")
        #     res.raise_for_status()
        #     logger.info(f"http://{ip}:{port} service is alive")
        #     return True
        # except Exception as e:
        #     logger.error(f"check service state failed, {e}")
        #     return False
        return True

# # 设置idc
# os.environ['IDC'] = 'be1a'

# psm = 'tiktok.aiic.large_model'
# s = SDClient(psm)

# # # 获取所有实例
# print(s.get_all())

# # # 获取单个实例（有多个随机获取）
# print(s.get_one())

if __name__ == '__main__':
    import sys
    task = sys.argv[1]
    psm = sys.argv[2]
    port = sys.argv[3]

    if task == 'register':
        register_service(psm, port)
    elif task == 'deregister':
        deregister_service(psm, port)