from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP
import pandas as pd

# split 500 to 250 / 250
def read_and_split_pcap(file_path, total_packets=500, train_size=250):
    packets = []
    with PcapReader(file_path) as pcap_reader:
        for i, packet in enumerate(pcap_reader):
            if i >= total_packets:  
                break
            packets.append(packet)
    
    training_packets = packets[:train_size]
    testing_packets = packets[train_size:]
    return training_packets, testing_packets

# extract feature function
def extract_features(packets):
    data = []
    for packet in packets:
        if IP in packet:
            pkt_size = len(packet)
            protocol = packet[IP].proto
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            flags = packet[TCP].flags if TCP in packet else None
            data.append({
                "Packet Size": pkt_size,
                "Protocol": protocol,
                "Source IP": src_ip,
                "Destination IP": dst_ip,
                "Flags": flags,
                "Timestamp": packet.time
            })
    return pd.DataFrame(data)

# frequency calculation function
def calculate_frequency(df):
    if len(df) < 2:
        return df
    start_time = df.iloc[0]["Timestamp"]
    end_time = df.iloc[-1]["Timestamp"]
    duration = end_time - start_time
    df['Frequency'] = df.groupby(["Source IP", "Destination IP", "Protocol", "Packet Size"])["Packet Size"].transform('count') / duration
    return df

# Main
if __name__ == "__main__":
    file_paths = {
        "benign": "dataset/benign_500.pcap",
        "DNS": "dataset/DNS_500.pcap",
        "LDAP": "dataset/LDAP_500.pcap",
        "NTP": "dataset/NTP_500.pcap",
        "SNMP": "dataset/SNMP_500.pcap",
        "SSDP": "dataset/SSDP_500.pcap",
        "UDP-lag": "dataset/UDP-lag_500.pcap",
        "UDP": "dataset/UDP_500.pcap"
    }

    for label, file in file_paths.items():
        train_packets, test_packets = read_and_split_pcap(file)

        train_df = extract_features(train_packets)
        test_df = extract_features(test_packets)

        train_df = calculate_frequency(train_df)
        test_df = calculate_frequency(test_df)

        train_df.to_csv(f"{label}_training.csv", index=False)
        test_df.to_csv(f"{label}_testing.csv", index=False)

    print("Data Preparation and Extraction Complete.")
