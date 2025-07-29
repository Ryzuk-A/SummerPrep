import matplotlib.pyplot as plt
import numpy as np

# Define CMS rates from 0% to 2%
cms_rates = np.linspace(0, 0.02, 500)
strike = 0.01
calc_amount = 100  # Represent as % of par

# Calculate redemption
redemption = np.where(cms_rates >= strike, calc_amount, calc_amount * (cms_rates / strike))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(cms_rates * 100, redemption, label='Redemption at Maturity', color='blue')
plt.axvline(x=1.0, color='grey', linestyle='--', label='Strike Rate = 1%')
plt.axhline(y=100, color='green', linestyle='--', label='Par Redemption = 100%')
plt.xlabel("Final CMS1 Rate (%)")
plt.ylabel("Redemption (% of Par)")
plt.title("Payout Profile at Maturity (Excluding Coupon)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/payout_profile.png")
plt.show()
