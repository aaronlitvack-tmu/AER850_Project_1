#Step 1: Data Processing
import pandas as pd
data = pd.read_csv("data/Project 1 Data.csv")

#Step 2: Data Visualization
import matplotlib.pyplot as plt
import numpy as np

plt.plot(data['Step'], data['X'], label='X')
plt.plot(data['Step'], data['Y'], label='Y')
plt.plot(data['Step'], data['Z'], label='Z')
plt.xlabel('Step')
plt.ylabel('XYZ Coordinate')
plt.title('XYZ Coordiantes')
plt.legend(loc='best')
plt.show()

#Step 3: Correlation Analysis
import seaborn as sns

corr_matrix = data.corr()
sns.heatmap((corr_matrix))

