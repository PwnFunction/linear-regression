from linear import fit, line

# linear dataset
X, Y = list(zip(*[
    (500, 2250),
    (1000, 4300),
    (1500, 5650),
    (2000, 7790),
    (2500, 11050),
    (3000, 11600),
    (3500, 15250),
    (4000, 15890),
    (4500, 18240),
    (5000, 19800),
]))

# downscale data to avoid overflow
downscale_factor = 0.001
X = [x * downscale_factor for x in X]
Y = [y * downscale_factor for y in Y]

# fit line to data
m, b = fit(X, Y)

# predict for amount spent on ads = $4200
amount_spent_on_ads = 4200
revenue = line(m, b, amount_spent_on_ads * downscale_factor) / downscale_factor
profit = revenue - amount_spent_on_ads

print(
    f"Amount spent on ads: ${amount_spent_on_ads}\n"
    f"Revenue: ${revenue:.2f}\n"
    f"Profit: ${profit:.2f}"
)