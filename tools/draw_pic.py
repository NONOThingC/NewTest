import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Set the seaborn style to "ticks"
file_path="/home/v-chengweihu/code/CRL-change/output/converted/all_training_logs_in_one_file.csv"
sns.set_style("ticks")
data=pd.read_csv(file_path)
data["method"]=data["metric"].apply(lambda x:x.split("---")[0])
data["figure"]=data["metric"].apply(lambda x:x.split("---")[1])
unique_methods=data["method"].unique()
# Define the x and y coordinates of the line
methods_plot=[]
for u_method in unique_methods:
    new_data=data.loc[(data["method"]==u_method) & (data["figure"]=="test/average_proto_history")]
    methods_plot.append(new_data)

# Plot the line using matplotlib
plt.plot(x, y, linewidth=2, color='black')

# Add labels to the axes
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)

# Add a title to the plot
plt.title('Line Plot Example', fontsize=16)

# Adjust the axis limits
plt.xlim(-0.5, 5.5)
plt.ylim(-0.5, 5.5)

# Show the plot
plt.show()