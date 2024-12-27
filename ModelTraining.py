import pandas as pd
import socket
import struct
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# convert IP to int function
def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]

# load datasets
benign_train = pd.read_csv('benign_training.csv')
benign_test = pd.read_csv('benign_testing.csv')
dns_train = pd.read_csv('DNS_training.csv')
dns_test = pd.read_csv('DNS_testing.csv')
ldap_train = pd.read_csv('LDAP_training.csv')
ldap_test = pd.read_csv('LDAP_testing.csv')
ntp_train = pd.read_csv('NTP_training.csv')
ntp_test = pd.read_csv('NTP_testing.csv')
snmp_train = pd.read_csv('SNMP_training.csv')
snmp_test = pd.read_csv('SNMP_testing.csv')
ssdp_train = pd.read_csv('SSDP_training.csv')
ssdp_test = pd.read_csv('SSDP_testing.csv')
udp_train = pd.read_csv('UDP_training.csv')
udp_test = pd.read_csv('UDP_testing.csv')
udp_lag_train = pd.read_csv('UDP-lag_training.csv')
udp_lag_test = pd.read_csv('UDP-lag_testing.csv')

# assign labels
benign_train['Label'] = 0
dns_train['Label'] = 1
ldap_train['Label'] = 2
ntp_train['Label'] = 3
snmp_train['Label'] = 4
ssdp_train['Label'] = 5
udp_train['Label'] = 6
udp_lag_train['Label'] = 7

benign_test['Label'] = 0
dns_test['Label'] = 1
ldap_test['Label'] = 2
ntp_test['Label'] = 3
snmp_test['Label'] = 4
ssdp_test['Label'] = 5
udp_test['Label'] = 6
udp_lag_test['Label'] = 7

# combine datasets
train_data = pd.concat([benign_train, dns_train, ldap_train, ntp_train, snmp_train, ssdp_train, udp_train, udp_lag_train], ignore_index=True)
test_data = pd.concat([benign_test, dns_test, ldap_test, ntp_test, snmp_test, ssdp_test, udp_test, udp_lag_test], ignore_index=True)

# convert all IPs
train_data['Source IP'] = train_data['Source IP'].apply(ip_to_int)
train_data['Destination IP'] = train_data['Destination IP'].apply(ip_to_int)
test_data['Source IP'] = test_data['Source IP'].apply(ip_to_int)
test_data['Destination IP'] = test_data['Destination IP'].apply(ip_to_int)

# label encode the flags column in datasets
label_encoder = LabelEncoder()
train_data['Flags'] = label_encoder.fit_transform(train_data['Flags'])
test_data['Flags'] = label_encoder.transform(test_data['Flags'])

# separate features and labels
X_train = train_data.drop(columns=['Label'])
y_train = train_data['Label']
X_test = test_data.drop(columns=['Label'])
y_test = test_data['Label']

# set model specs and train
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='gini', random_state=42)
rf_model.fit(X_train, y_train)

# testing
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
