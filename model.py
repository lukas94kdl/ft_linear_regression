import csv


def compute_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_std(values: list[float], mean: float) -> float:
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def normalize(values: list[float]) -> tuple[list[float], float, float]:
    mean = compute_mean(values)
    std  = compute_std(values, mean)
    normalized = [(x - mean) / std for x in values]
    return normalized, mean, std


def load_data(filepath: str) -> tuple[list[float], list[float]]:
    x_values = []
    y_values = []
    with open(filepath, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_values.append(float(row["km"]))
            y_values.append(float(row["price"]))
    return x_values, y_values


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage


def gradient_descent(
    x: list[float],
    y: list[float],
    learning_rate: float = 0.01,
    iterations:    int   = 1000
) -> tuple[float, float]:

    m      = len(x)
    theta0 = 0.0
    theta1 = 0.0

    for _ in range(iterations):
        errors = [estimate_price(x[i], theta0, theta1) - y[i] for i in range(m)]
        tmp0   = learning_rate * (1/m) * sum(errors)
        tmp1   = learning_rate * (1/m) * sum(errors[i] * x[i] for i in range(m))
        theta0 = theta0 - tmp0
        theta1 = theta1 - tmp1

    return theta0, theta1
 

def denormalize_theta(
    theta0: float, theta1: float,
    x_mean: float, x_std:  float,
    y_mean: float, y_std:  float
) -> tuple[float, float]:

    t1 = theta1 * (y_std / x_std)
    t0 = y_mean + y_std * theta0 - t1 * x_mean
    return t0, t1


def save_params(theta0: float, theta1: float, filepath: str) -> None:
    with open(filepath, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["theta0", "theta1"])
        writer.writerow([theta0, theta1])


def load_params(filepath: str) -> tuple[float, float]:
    try:
        with open(filepath, "r") as file:
            reader = csv.DictReader(file)
            row    = next(reader)
            return float(row["theta0"]), float(row["theta1"])
    except FileNotFoundError:
        return 0.0, 0.0


def compute_r2(
    x: list[float],
    y: list[float],
    theta0: float,
    theta1: float
) -> float:
    y_mean = compute_mean(y)
    ss_res = sum((y[i] - estimate_price(x[i], theta0, theta1)) ** 2 for i in range(len(x)))
    ss_tot = sum((y[i] - y_mean) ** 2                               for i in range(len(x)))
    return 1 - (ss_res / ss_tot)
