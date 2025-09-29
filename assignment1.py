import matplotlib.pyplot as plt
import numpy as np

# Data
months = ["07/2019", "08/2019", "09/2019", "10/2019", "11/2019"]
searches = [50, 53, 59, 56, 62]
direct = [39, 47, 42, 51, 51]
social_media = [70, 80, 90, 87, 92]

# Bar positions
x = np.arange(len(months))
width = 0.25

# Plotting
plt.figure(figsize=(10,6))
plt.bar(x - width, searches, width, label='Searches', color='royalblue')
plt.bar(x, direct, width, label='Direct', color='orchid')
plt.bar(x + width, social_media, width, label='Social Media', color='orange')

# Labels & Title
plt.ylabel("Visitors (in thousands)")
plt.xlabel("Months")
plt.title("Visitors by Web Traffic Sources")
plt.xticks(x, months)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show
plt.show()
