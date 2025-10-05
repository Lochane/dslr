def ft_mean(values):
	total = 0
	for v in values: 
		total += v
	return total / len(values)

def ft_variance(values):
	m = ft_mean(values)
	total = 0
	for n in values:
		total += (n - m) ** 2
	return total / len(values)

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