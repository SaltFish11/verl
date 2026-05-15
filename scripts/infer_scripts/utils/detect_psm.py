#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSM 端口连通性检测脚本：
1. 输入 PSM 列表 → 调用 sd lookup 获取 IP:Port
2. 批量检测每个 IP:Port 的连通性
3. 输出清晰的检测结果
"""
import socket
import subprocess
import argparse
import re
import ipaddress
from typing import Tuple, List, Dict, Optional

def ipv4_2_ipv6(ipv4: str) -> str:
    """
    通过 IPv4 地址获取 IPv6 地址
    :param ipv4: IPv4 地址
    :return: 对应的 IPv6 地址
    """
    ipv4 = ipv4.strip()
    try:
        ipaddress.IPv4Address(ipv4)
    except ValueError as e:
        raise ValueError(f"非法IPv4地址: {ipv4}") from e

    try:
        hostname, _, _ = socket.gethostbyaddr(ipv4)
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_INET6, socket.SOCK_STREAM)

        candidates: list[ipaddress.IPv6Address] = []
        for _family, _socktype, _proto, _canonname, sockaddr in addr_infos:
            ip_str = sockaddr[0]
            try:
                candidates.append(ipaddress.IPv6Address(ip_str))
            except ValueError:
                continue

        for ip6 in candidates:
            if ip6.is_global:
                return str(ip6)
        if candidates:
            return str(candidates[0])
    except Exception:
        pass

    return str(ipaddress.IPv6Address(f"::ffff:{ipv4}"))
def check_ip_alive(ip: str) -> bool:
    """
    检测单个 IP 是否可连通（ICMP）
    :param ip: 目标IP
    :return: 是否连通
    """
    try:
        socket.gethostbyname(ip)
        return True
    except socket.gaierror:
        return False

def check_port_connectivity(
    ip: str,
    port: int,
    timeout: float = 3.0
) -> Tuple[bool, str]:
    """
    检测单个 IP:Port 是否可连通（TCP）
    :param ip: 目标IP
    :param port: 目标端口
    :param timeout: 超时时间（秒）
    :return: (是否连通, 状态描述)
    """
    ip = ipv4_2_ipv6(ip)
    if ip.count(":") > 0: #IPV6
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        # if not (ip[0] == "[" and ip[-1] == "]"):
        #     ip = '['+ ip +']'
    else: #IPV4
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
        
    if not (1 <= port <= 65535):
        return False, f"端口 {port} 超出合法范围（1-65535）"
    try:
        sock.connect((ip, port))
        sock.close()
        return 0, f"✅ {ip}:{port} - 端口可连通(目标端口有服务)"
    except socket.timeout:
        return 2, f"❌ {ip}:{port} - 连接超时（{timeout}秒）"
    except ConnectionRefusedError:
        return 1, f"❌ {ip}:{port} - 端口被拒绝连接（目标端口服务未启动/防火墙拦截）"
    except socket.gaierror:
        return 2, f"❌ {ip}:{port} - IP地址解析失败"
    except Exception as e:
        return 2, f"❌ {ip}:{port} - 连接失败：{str(e)[:50]}"

def execute_sd_lookup(psm: str) -> Optional[List[Dict[str, any]]]:
    """
    执行 sd lookup 命令，解析出 IP:Port 列表
    :param psm: 目标 PSM
    :return: 解析后的列表 [{"ip": "...", "port": ...}, ...]，失败返回 None
    """
    try:
        # 执行 sd lookup 命令，捕获输出
        sd = "/opt/tiger/consul_deploy/bin/go/sd"
        result = subprocess.run(
            [sd, "lookup", psm],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            timeout=10  # sd 命令超时时间
        )

        # 检查命令执行是否失败
        if result.returncode != 0:
            print(f"⚠️ PSM {psm} - sd lookup 执行失败：{result.stderr.strip()}")
            return None

        # 解析输出，提取 IP 和 Port（匹配 "xxx.xxx.xxx.xxx 端口 其他内容" 格式）
        ip_port_pattern = re.compile(r"(\d+\.\d+\.\d+\.\d+)\s+(\d+)\s+.*")
        targets = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            match = ip_port_pattern.match(line)
            if match:
                ip = match.group(1)
                port = int(match.group(2))
                targets.append({"ip": ip, "port": port})

        if not targets:
            print(f"⚠️ PSM {psm} - 未解析到任何 IP:Port 信息")
            return None

        return targets
    except subprocess.TimeoutExpired:
        print(f"⚠️ PSM {psm} - sd lookup 命令超时（10秒）")
        return None
    except Exception as e:
        print(f"⚠️ PSM {psm} - 解析 sd 结果失败：{str(e)[:50]}")
        return None

def batch_check_psm(
    psm_list: List[str],
    timeout: float = 3.0
) -> None:
    """
    批量检测 PSM 对应的所有 IP:Port
    :param psm_list: PSM 列表
    :param timeout: 端口检测超时时间
    """
    print(f"===== 开始检测 PSM 列表（共 {len(psm_list)} 个）=====\n")
    ans = []
    for idx, psm in enumerate(psm_list, 1):
        print(f"=== [{idx}/{len(psm_list)}] 检测 PSM: {psm} ===")
        # 调用 sd lookup 获取 IP:Port
        targets = execute_sd_lookup(psm)
        if not targets:
            print()  # 空行分隔
            continue

        # 批量检测端口
        print(f"解析到 {len(targets)} 个 IP:Port，开始检测（超时 {timeout} 秒）：")
        for target in targets:
            flag_1 = check_ip_alive(target["ip"])
            if not flag_1:
                print(f"❌ ip {target['ip']} 不可连通")
            else:
                print(f"✅ ip {target['ip']} 可连通")

            flag, msg = check_port_connectivity(target["ip"], target["port"], timeout)
            print(f"  {msg}")
            if flag == 0:
                if psm not in ans:
                    ans.append(psm)
            
                
        print()  # 空行分隔
    print("不可使用的PSM:",ans)
    if len(ans) == 0:
        return True
    else :
        return False
    print("===== 所有 PSM 检测完成 =====")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="PSM 端口连通性批量检测工具")
    parser.add_argument(
        "-l", "--psm-list",
        type=str,
        help='PSM 列表，逗号分隔（如 "tiktok.aiic.qwencoder_model_1,tiktok.aiic.qwencoder_model_2"）'
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=3.0,
        help="端口检测超时时间（秒，默认3）"
    )

    args = parser.parse_args()
    psm_list = args.psm_list.split(",") if args.psm_list else []
    # 解析 PSM 列表
    # [psm.strip() for psm in args.psm_list.split(",") if psm.strip()]
    PSM_LIST=[
        "tiktok.aiic.qwencoder_model_1",
        "tiktok.aiic.qwencoder_model_2",
        "tiktok.aiic.qwencoder_model_3",
        "tiktok.aiic.qwencoder_model_4",
        "tiktok.aiic.qwencoder_model_5",
        "tiktok.aiic.qwencoder_model_6",
        "tiktok.aiic.qwencoder_model_7",
        "tiktok.aiic.qwencoder_model_8",
        "tiktok.aiic.qwencoder_model_9",
        "tiktok.aiic.gpt-oss-qwencoder_model_1",
        "tiktok.aiic.gpt-oss-qwencoder_model_2",
        "tiktok.aiic.gpt-oss-qwencoder_model_3",
        "tiktok.aiic.gpt-oss-qwencoder_model_4",
        "tiktok.aiic.gpt-oss-qwencoder_model_5",
        "tiktok.aiic.gpt-oss-qwencoder_model_6",
        "tiktok.aiic.gpt-oss-qwencoder_model_7",
        "tiktok.aiic.gpt-oss-qwencoder_model_8",
        "tiktok.aiic.gpt-oss-qwencoder_model_9",
        "tiktok.aiic.gpt-5-codex-qwencoder_model_0",
        "tiktok.aiic.gpt-5-codex-qwencoder_model_1",
        "tiktok.aiic.gpt-5-codex-qwencoder_model_2"
    # 可继续添加更多 PSM...
    ]
    if len(psm_list) > 0:
        PSM_LIST = psm_list
    if not PSM_LIST:
        print("❌ 未输入有效 PSM 列表！")
        parser.print_help()
        exit(1)

    # 开始批量检测
    flag = batch_check_psm(PSM_LIST, args.timeout)
    if flag:
        exit(0)
    else:
        exit(1)
