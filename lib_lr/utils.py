def ft_mean(lst):
	total = 0
	for v in lst: 
		total += v
	return total / len(lst)

def ft_variance(lst):
	m = ft_mean(lst)
	total = 0
	for n in lst:
		total += (n - m) ** 2
	return total / len(lst)

def ft_std_dev(lst):
	return ft_variance(lst) ** 0.5

def ft_count(lst):
	count = 0
	for _ in lst:
		count += 1
	return count

def ft_min(lst):
	m = lst[0]
	for i in lst:
		if i < m:
			m = i
	return m

def ft_max(lst):
	m = lst[0]
	for i in lst:
		if i > m:
			m = i
	return m

def ft_percentile(lst, q):
	sorted_vals = sorted(lst)
	n = ft_count(sorted_vals)
	pos = q * (n - 1)
	low = int(pos)
	high = low + 1
	if high >= n:
		return sorted_vals[low]
	return sorted_vals[low] + (sorted_vals[high] - sorted_vals[low] * (pos -low))