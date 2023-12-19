import matplotlib.pyplot as plt

category = ["Category 1", "Category 2", "Category 3"]
values = [10, 15, 7]

plt.bar(category, values, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()

