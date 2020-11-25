# Import Modules
import pandas as pd
import numpy as np

# Read Dataset
df = pd.read_csv('data/original.csv')

# Drop Unnecessary Columns
df = df.drop(columns=['Unnamed: 0'])
df = df.drop(columns=['Flow ID', 'Src IP', 'Src Port', 'Dst IP'])

# Delete Missing Values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(axis=0)

# Sampling for each labels
Normal = df[df['Label'].isin(['Benign'])]
Brute_force = df[df['Label'].isin(['FTP-BruteForce', 'SSH-Bruteforce'])]
DoS = df[df['Label'].isin(['DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Hulk'])]
Web_attacks = df[df['Label'].isin(['Brute Force -Web', 'Brute Force -XSS', 'SQL Injection'])]
infiltration = df[df['Label'].isin(['Infilteration'])]
Botnet = df[df['Label'].isin(['Bot'])]
DDoS = df[df['Label'].isin(['DDoS attacks-LOIC-HTTP', 'DDOS attack-LOIC-UDP', 'DDOS attack-HOIC'])]

size = 50000
if len(Normal) >= size:
    Normal = Normal.sample(n=size)
if len(Brute_force) >= size:
    Brute_force = Brute_force.sample(n=size)
if len(DoS) >= size:
    DoS = DoS.sample(n=size)
if len(Web_attacks) >= size:
    Web_attacks = Web_attacks.sample(n=size)
if len(infiltration) >= size:
    infiltration = infiltration.sample(n=size)
if len(Botnet) >= size:
    Botnet = Botnet.sample(n=size)
if len(DDoS) >= size:
    DDoS = DDoS.sample(n=size)

# Merge each DataFrames
df = pd.concat([Normal, Brute_force, DoS, Web_attacks, infiltration, Botnet, DDoS])

# Convert Labels' Names
df['Label'].loc[df['Label'] == 'FTP-BruteForce'] = 'Brute-force'
df['Label'].loc[df['Label'] == 'SSH-Bruteforce'] = 'Brute-force'
df['Label'].loc[df['Label'] == 'DoS attacks-GoldenEye'] = 'DoS'
df['Label'].loc[df['Label'] == 'DoS attacks-Slowloris'] = 'DoS'
df['Label'].loc[df['Label'] == 'DoS attacks-SlowHTTPTest'] = 'DoS'
df['Label'].loc[df['Label'] == 'DoS attacks-Hulk'] = 'DoS'
df['Label'].loc[df['Label'] == 'Brute Force -Web'] = 'Web attacks'
df['Label'].loc[df['Label'] == 'Brute Force -XSS'] = 'Web attacks'
df['Label'].loc[df['Label'] == 'SQL Injection'] = 'Web attacks'
df['Label'].loc[df['Label'] == 'Infilteration'] = 'infiltration'
df['Label'].loc[df['Label'] == 'Bot'] = 'Botnet'
df['Label'].loc[df['Label'] == 'DDoS attacks-LOIC-HTTP'] = 'DDoS'
df['Label'].loc[df['Label'] == 'DDOS attack-LOIC-UDP'] = 'DDoS'
df['Label'].loc[df['Label'] == 'DDOS attack-HOIC'] = 'DDoS'

# Show isnull, groupCount
print(df.isnull().sum())
print(df.groupby('Label').count())
print(df)

# Save as CSV File
df.to_csv('data/sample.csv', index=False)