def line(m, b, x) -> float:
    """
    Equation for the line is given by
    y = mx + b
    """
    return m * x + b

def fit(X, Y, learning_rate=0.01, iterations=1000):
    """
    Fit a line to the data using gradient descent.
    Partial derivatives of the error function with respect to m and b are given by

    mse = 1/n * Σ(y' - (m * x + b))^2

    ∂e/∂m = 1/n * Σ 2 * (y' - (m * x + b)) (-x)
    ∂e/∂m = -2/n * Σx(y' - (m * x) - b))

    ∂e/∂b = 2/n * Σ(y' - (m * x + b)) (-1)
    ∂e/∂b = -2/n * Σ(y' - (m * x) - b))
    """

    # initialise m and b
    m, b = 1,1

    n = len(X)

    for _ in range(iterations):
        # ∂e/∂m
        # =====
        # Z = Σx(y' - (m * x) - b))
        Z = 0
        for point in range(n):
            Z += (Y[point] - (m * X[point]) - b) * X[point]
        dedm = -2/n * Z

        # ∂e/∂b
        # =====
        # Z = Σ(y' - (m * x) - b))
        Z = 0
        for point in range(n):
            Z += Y[point] - (m * X[point]) - b
        dedb = -2/n * Z

        # adjust m & b based on learning rates
        m = m - learning_rate * dedm
        b = b - learning_rate * dedb

    return m, b