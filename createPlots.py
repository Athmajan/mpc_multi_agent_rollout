import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('config1_outputs.csv')
df_BP = df[['Step','SimpleTag_v3_BPTrue - Global Steps']].copy()
df_1 = df[['Step','SimpleTag_v3_BPFalse_Runs_0_to_20 - Global Steps']].copy()
df_2 = df[['Step','SimpleTag_v3_BPFalse_Runs_20_to_35 - Global Steps']].copy()
df_3 = df[['Step','SimpleTag_v3_BPFalse_Runs_35_to_50 - Global Steps']].copy()


df_BP.rename(columns={'Step': 'Step', 
                      'SimpleTag_v3_BPTrue - Global Steps': 'BP Steps'}, inplace=True)

df_1.rename(columns={'Step': 'Step', 
                      'SimpleTag_v3_BPFalse_Runs_0_to_20 - Global Steps': 'Seq Steps'}, inplace=True)
df_2.rename(columns={'Step': 'Step', 
                      'SimpleTag_v3_BPFalse_Runs_20_to_35 - Global Steps': 'Seq Steps'}, inplace=True)
df_3.rename(columns={'Step': 'Step', 
                      'SimpleTag_v3_BPFalse_Runs_35_to_50 - Global Steps': 'Seq Steps'}, inplace=True)

df_1 = df_1.dropna()
df_2 = df_2.dropna()
df_3 = df_3.dropna()


# Adjust the Step column for df_2
df_2['Step'] += df_1['Step'].iloc[-1] + 1

# Adjust the Step column for df_3
df_3['Step'] += df_2['Step'].iloc[-1] + 1

# Concatenate the DataFrames
df_seq = pd.concat([df_1, df_2, df_3], ignore_index=True)
merged_df = pd.merge(df_BP, df_seq, on='Step', how='inner')
print(merged_df.mean())
import ipdb; ipdb.set_trace()

# Plotting
plt.figure(figsize=(10, 6))

# Plot BP Steps
plt.plot(merged_df['Step'], merged_df['BP Steps'], label='BP Steps', marker='o')

# Plot Seq Steps
plt.plot(merged_df['Step'], merged_df['Seq Steps'], label='Seq Steps', marker='x')

# Add labels, title, and legend
plt.xlabel('Steps')
plt.ylabel('Global Steps')
plt.title('BP Steps vs Seq Steps')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
