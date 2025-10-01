from stats_tools import ft_mean, ft_std_dev

def ft_linear_regression(x, y, learning_rate=0.01, iterations=1000):

	x_mean = ft_mean(x)
	x_std = ft_std_dev(x)
	x_norm = (x - x_mean) / x_std

	theta0 = 0
	theta1 = 0
	n = len(x_norm)

	for _ in range(iterations):
		sum_error_theta0 = 0
		sum_error_theta1 = 0

		for x, y in zip(x_norm, y):
			prediction = theta0 + theta1 * x
			error = prediction - y
			sum_error_theta0 += error
			sum_error_theta1 += error * x
		
		correction_theta0 = (learning_rate * sum_error_theta0) / n
		correction_theta1 = (learning_rate * sum_error_theta1) /n

		theta0 -= correction_theta0
		theta1 -= correction_theta1

	return {"theta0": theta0, "theta1": theta1, "x_mean": x_mean, "x_std": x_std}