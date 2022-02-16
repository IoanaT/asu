#!/bin/sh

#################################################################################
#								            #
# rc.firewall - Initial SIMPLE IP Firewall script for Linux and iptables        #
#									    #
# 02/17/2020  Created by Dijiang Huang ASU SNAC Lab      			    #
# updated 05/12/2021
# updated 10/23/2021 by Tiriac Ioana for ASU CSE 548 Project1
#################################################################################
#                                                                               #
#                                                                               #
# Configuration options, these will speed you up getting this script to         #
# work with your own setup.                                                     #
#                                                                               #
# your LAN's IP range and localhost IP. /24 means to only use the first 24      #
# bits of the 32 bit IP address. the same as netmask 255.255.255.0              #
#                                                                               #
#                                                                               # 
#################################################################################
#
# 1. Configuration options.
# NOTE that you need to change the configuration based on your own network setup.
# The defined alias and variables allow you to manage and update the entire 
# configurations easily, and more readable :-)
#
# Lab Network Topology
#
# ---------                       ----------------               
# |Client |__Client_NET__|Gateway/Server |
# ---------                      ----------------              
#                                      |
#                                      |Internet           
#                                      |                      ________ 
#                                  ----------             /        \
#                                  |Host PC |________|Internet|
#                                 ----------             \_______/ 
#                        
#

####
# 1.1. Internet ip address
# 
#
Internet_IP="10.0.1.1"
Internet_IP_RANGE="10.0.1.0/24"
Internet_BCAST_ADRESS="10.0.1.255"
Internet_IFACE="enp0s8"

####
# 1.2 Client network configuration.
#
#

#
# IP addresses of the client-side network
#
Client_NET_IP="10.0.2.5"
Client_NET_IP_RANGE="10.0.2.0/24"
Client_NET_BCAST_ADRESS="10.0.2.255"
Client_NET_IFACE="enp0s3"


#
# IP aliases for the server (server's IP address)
#
LO_IFACE="lo"
LO_IP="127.0.0.1"
WEB_IP_ADDRESS="127.0.0.1"
#IP aliases for NATed services (this is the GW's ip on client network)
NAT_WEB_IP_ADDRESS="10.0.2.5"

####
# 1.4 IPTables Configuration.
#

IPTABLES="/sbin/iptables"


#######################################################
#                                                     #
# 2. Module loading.                                  #
#                                                     #
#######################################################
#
# Needed to initially load modules
#
/sbin/depmod -a	 

#
# flush iptables
#
$IPTABLES -F 
$IPTABLES -X 
$IPTABLES -F -t nat

#####
# 2.1 Required modules
#

/sbin/modprobe ip_tables
/sbin/modprobe ip_conntrack
/sbin/modprobe iptable_filter
/sbin/modprobe iptable_mangle
/sbin/modprobe iptable_nat
/sbin/modprobe ipt_LOG
/sbin/modprobe ipt_limit
/sbin/modprobe ipt_state

#####
# 2.2 Non-frequently used modules
#

#/sbin/modprobe ipt_owner
#/sbin/modprobe ipt_REJECT
#/sbin/modprobe ipt_MASQUERADE
#/sbin/modprobe ip_conntrack_ftp
#/sbin/modprobe ip_conntrack_irc
#/sbin/modprobe ip_nat_ftp

###########################################################################
#
# 3. /proc set up.
#

#
# 3.1 Required proc configuration
#

#
# Enable ip_forward, this is critical since it is turned off as defaul in 
# Linux.
#
echo "1" > /proc/sys/net/ipv4/ip_forward

#
# 3.2 Non-Required proc configuration
#

#
# Dynamic IP users:
#
#echo "1" > /proc/sys/net/ipv4/ip_dynaddr

###########################################################################
#
# 4. rules set up.
#

# The kernel starts with three lists of rules; these lists are called firewall
# chains or just chains. The three chains are called INPUT, OUTPUT and FORWARD.
#
# The chains are arranged like so:
#
#                     _____
#                    /     \
#  -->[Routing ]--->|FORWARD|------->
#     [Decision]     \_____/        ^
#          |                        |
#          v                       ____
#         ___                     /    \
#        /   \                   |OUTPUT|
#       |INPUT|                   \____/
#        \___/                      ^
#          |                        |
#           ----> Local Process ----
#
# 1. When a packet comes in (say, through the Ethernet card) the kernel first 
#    looks at the destination of the packet: this is called `routing'.
# 2. If it's destined for this box, the packet passes downwards in the diagram, 
#    to the INPUT chain. If it passes this, any processes waiting for that 
#    packet will receive it. 
# 3. Otherwise, if the kernel does not have forwarding enabled, or it doesn't 
#    know how to forward the packet, the packet is dropped. If forwarding is 
#    enabled, and the packet is destined for another network interface (if you 
#    have another one), then the packet goes rightwards on our diagram to the 
#    FORWARD chain. If it is ACCEPTed, it will be sent out. 
# 4. Finally, a program running on the box can send network packets. These 
#    packets pass through the OUTPUT chain immediately: if it says ACCEPT, then 
#    the packet continues out to whatever interface it is destined for. 
#


#####
# 4.1 Filter table
#

#
# 4.1.1 Set policies
#

#
# Set default policies for the INPUT, FORWARD and OUTPUT chains
#

# Whitelist (Whitelist is preferred)
$IPTABLES -P INPUT DROP
$IPTABLES -P OUTPUT DROP
$IPTABLES -P FORWARD DROP

# Blacklist
#$IPTABLES -P INPUT ACCEPT
#$IPTABLES -P OUTPUT ACCEPT
#$IPTABLES -P FORWARD ACCEPT

#
# 4.1.2 Create user-specified chains
#

#
# Example of creating a chain for bad tcp packets
#

#$IPTABLES -N bad_tcp_packets

#
# Create separate chains for allowed (whitelist), ICMP, TCP and UDP to traverse
#

#$IPTABLES -N allowed
#$IPTABLES -N tcp_packets
#$IPTABLES -N udp_packets
#$IPTABLES -N icmp_packets

#
# In the following 4.1.x, you can provide individual user-specified rules


#
# 4.1.3 Example of create content in user-specified chains (bad_tcp_packets)
#

#
# bad_tcp_packets chain
#

#$IPTABLES -A bad_tcp_packets -p tcp --tcp-flags SYN,ACK SYN,ACK -m state --state NEW -j REJECT --reject-with tcp-reset 
#$IPTABLES -A bad_tcp_packets -p tcp ! --syn -m state --state NEW -j LOG --log-prefix "New not syn:"
#$IPTABLES -A bad_tcp_packets -p tcp ! --syn -m state --state NEW -j DROP

#
# 4.1.4 Example of allowed chain (allow packets for initial TCP or already established TCP sessions)
#

#$IPTABLES -A allowed -p TCP --syn -j ACCEPT
#$IPTABLES -A allowed -p TCP -m state --state ESTABLISHED,RELATED -j ACCEPT
#$IPTABLES -A allowed -p TCP -j DROP


#####
# 4.2 FORWARD chain
#

#
# Provide your forwarding rules below
#

# example of checking bad tcp packets
#$IPTABLES -A FORWARD -p tcp -j bad_tcp_packets

# Tiriac : Allow ping from client network to google DNS, and from google DNS back
$IPTABLES -A FORWARD -p icmp -o $Client_NET_IFACE -s 8.8.8.8 -j ACCEPT
$IPTABLES -A FORWARD -p icmp -s 10.0.2.4 -i $Client_NET_IFACE -d 8.8.8.8 -j ACCEPT

# Tiriac : Allow return packets from established connections
$IPTABLES -A FORWARD -p TCP -o $Client_NET_IFACE -i $Internet_IFACE -m state --state ESTABLISHED,RELATED -j ACCEPT

# Tiriac : Allow HTTP/HTTPS internet traffic for client to update OS packages

$IPTABLES -A FORWARD -p TCP -i $Client_NET_IFACE -o $Internet_IFACE --dport 80 -j ACCEPT
$IPTABLES -A FORWARD -p TCP -i $Client_NET_IFACE -o $Internet_IFACE --dport 443 -j ACCEPT


#####
# 4.3 INPUT chain
#

#
# Provide your input rules below to allow web traffic from client
#
#$IPTABLES -A INPUT -p TCP --dport 80 -i $Client_NET_IFACE -s 10.0.2.5 -j ACCEPT

# Tiriac : Allow loopback traffic - 127.0.0.1 or localhost

$IPTABLES -A INPUT -p TCP -i lo -j ACCEPT
$IPTABLES -A INPUT -p TCP -i lo -s 127.0.0.1 -j ACCEPT

# Tiriac: Allow syn and return packets from established connections
$IPTABLES -A INPUT -p TCP --syn -j ACCEPT
$IPTABLES -A INPUT -p TCP -m state --state ESTABLISHED,RELATED -j ACCEPT


# Tiriac: Allow client and loopback to connect to web server on Gateway

$IPTABLES -A INPUT -p TCP --dport 80 -i $Client_NET_IFACE -j ACCEPT
$IPTABLES -A INPUT -p TCP --dport 80 -i $LO_IFACE -d $WEB_IP_ADDRESS -j ACCEPT

# Tiriac: Allow traffic to HTTP/HTTPS for Gateway - for OS updates

$IPTABLES -A INPUT -p TCP --sport 80 -i $Internet_IFACE -j ACCEPT
$IPTABLES -A INPUT -p TCP --sport 443 -i $Internet_IFACE -j ACCEPT 

#
# Example of checking bad TCP packets we don't want.
#

#$IPTABLES -A INPUT -p tcp -j bad_tcp_packets




# Tiriac: ACCEPT ping from client VM's IP to DNS 8.8.8.8

$IPTABLES -A INPUT -p icmp -s 10.0.2.4 -d 8.8.8.8 -j ACCEPT

# Tiriac: REJECT ping to localhost
$IPTABLES -A INPUT -p icmp -i lo -d 127.0.0.1 -j REJECT


#####
# 4.3 OUTPUT chain
#

#
# Provide your output rules below to allow web traffic (port 80) go back to client
#

# Tiriac: example Allowed ping message back to client
$IPTABLES -A OUTPUT -p icmp -s 8.8.8.8 -j ACCEPT

# Tiriac: Allow loopback on output chain
$IPTABLES -A OUTPUT -o $LO_IFACE -j ACCEPT
$IPTABLES -A OUTPUT -o $LO_IFACE -s 127.0.0.1 -j ACCEPT

# Tiriac: Allow syn and return packets from established connections
$IPTABLES -A OUTPUT -p TCP --syn -j ACCEPT
$IPTABLES -A OUTPUT -p TCP -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow TCP traffic back to client
# Tiriac:  Allow gateway server response to reach client and loopback

$IPTABLES -A OUTPUT -p TCP --sport 80 -o $Client_NET_IFACE -j ACCEPT
$IPTABLES -A OUTPUT -p TCP --sport 80 -o $LO_IFACE -j ACCEPT

# Tiriac: Allow traffic to HTTP/HTTPS for Gateway - to update OS

$IPTABLES -A OUTPUT -p TCP --dport 80 -o $Internet_IFACE -j ACCEPT
$IPTABLES -A OUTPUT -p TCP --dport 443 -o $Internet_IFACE -j ACCEPT
#####################################################################
#                                                                   #
# 5. NAT setup                                                      #
#                                                                   #
#####################################################################

#####
# 5.1  PREROUTING chain. (No used in this lab)
#
#
# Provide your NAT PREROUTING rules (packets come into your private domain)
#

#
# Example of enable http to internal web server behind the firewall (port forwarding)
#

# web 
# $IPTABLES -t nat -A PREROUTING -p tcp -d $NAT_WEB_IP_ADDRESS --dport 80 -j DNAT --to $WEB_IP_ADDRESS




#####
# 5.2 POSTROUTING chain.
#
#
# Provide your NAT PREROUTING rules (packets go to the internet domain)
# Add your own rule below to only allow ping from client to 8.8.8.8 on internet

# Tiriac: Only allow client node to ping google DNS using masquerade

$IPTABLES -t nat -A POSTROUTING -p icmp -o $Internet_IFACE -d 8.8.8.8 -j MASQUERADE

# Tiriac: Allow Internet Traffic to HTTP/HTTPS for client to be able to update OS
$IPTABLES -t nat -A POSTROUTING -p tcp -o $Internet_IFACE --dport 80 -j MASQUERADE
$IPTABLES -t nat -A POSTROUTING -p tcp -o $Internet_IFACE --dport 443 -j MASQUERADE


