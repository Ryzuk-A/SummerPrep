import matplotlib.pyplot as plt

# Gilt data (from June 20, 2025)
maturities = ['1M', '3M', '6M', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
yields = [4.32, 4.22, 4.18, 3.92, 3.91, 4.04, 4.16, 4.54, 5.16, 5.26]  # in percent

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(maturities, yields, marker='o', linestyle='-', color='royalblue', label='UK Gilt Yields')
plt.title('UK Gilt Yield Curve (as of 20 June 2025)')
plt.xlabel('Maturity')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.ylim(3.5, 5.5)  # Set Y-axis from 3.5% to 5.5%
plt.tight_layout()
plt.legend()
plt.show()