def ft_mean(values):
	total = 0
	count = 0
	for v in values:
		if v is not None and (isinstance(v, (int, float)) and not (v != v)):
			total += v
			count += 1
	if count == 0:
		return 0
	return total / count

def ft_variance(values):
	m = ft_mean(values)
	total = 0
	count = 0
	for n in values:
		if n is not None and (isinstance(n, (int, float)) and not (n != n)):
			total += (n - m) ** 2
			count += 1
	if count == 0:
		return 0
	return total / count


def ft_std_dev(values):
	return ft_variance(values) ** 0.5


def ft_count(values):
	count = 0
	for _ in values:
		count += 1
	return count

def ft_min(values):
	m = values[0]
	for i in values:
		if i < m:
			m = i
	return m

def ft_max(values):
	m = values[0]
	for i in values:
		if i > m:
			m = i
	return m


def ft_covariance(x, y):
    mean_x = ft_mean(x)
    mean_y = ft_mean(y)
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / (len(x) - 1)
    return cov


def ft_correlation(x, y):
    cov = ft_covariance(x, y)
    return cov / (ft_std_dev(x) * ft_std_dev(y))


def ft_percentile(values, q):
	sorted_vals = sorted(values)
	n = ft_count(sorted_vals)
	pos = q * (n - 1)
	low = int(pos)
	high = low + 1
	if high >= n:
		return sorted_vals[low]
	# interpolation linéaire correcte: low + (high - low) * fraction
	return sorted_vals[low] + (sorted_vals[high] - sorted_vals[low]) * (pos - low)
