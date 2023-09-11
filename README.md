# Military Network Intrusion Dataset

## Overview
This dataset was curated by simulating a wide variety of intrusions in a military network environment. The simulation aimed to replicate a typical US Air Force LAN, providing an authentic environment to acquire raw TCP/IP dump data. The LAN was designed to mirror real-world conditions and was subjected to numerous attack scenarios.

## Dataset Description
- **Environment**: The dataset was created by simulating a typical US Air Force LAN.
- **Data Type**: Raw TCP/IP dump data.
- **Connection**: A connection in this dataset refers to a sequence of TCP packets that start and end over a certain time duration. During this duration, data flows from a source IP address to a target IP address under a specific protocol.
- **Labels**: Each connection is labeled as either:
  - **Normal**: Indicating typical, non-malicious traffic.
  - **Anomalous**: Indicating traffic that is indicative of an attack. Each anomalous connection is further labeled with a specific attack type.

## Features
The dataset comprises 41 features for each TCP/IP connection. These features are a mix of quantitative and qualitative data:
- **Quantitative Features**: 38
- **Qualitative Features**: 3

The features provide insights into the nature of the traffic, helping in distinguishing between normal and anomalous connections.

## Class Variable
The class variable in the dataset categorizes the connections into two main categories:
1. **Normal**: Connections that represent regular, non-malicious traffic.
2. **Anomalous**: Connections that represent potential threats or attacks.

## Usage
This dataset can be instrumental for researchers and cybersecurity professionals aiming to develop intrusion detection systems or study network vulnerabilities in a military context.
