## Calculs statistiques basiques
## Moyenne, variance, écart-type, min, max, covariance, corrélation, percentile

## Calcul de la moyenne
## Somme des valeurs divisée par le nombre de valeurs non nulles
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

## Calcul de la variance
## Moyenne des carrés des écarts à la moyenne, aussi égale à la différence entre la moyenne des carrés des valeurs de la variable et le carré de la moyenne
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


## Calcul de l'écart-type
## Racine carrée de la variance
def ft_std_dev(values):
	return ft_variance(values) ** 0.5

## Calcul du nombre d'éléments
def ft_count(values):
	count = 0
	for _ in values:
		count += 1
	return count


## objet vide par défaut pour les fonctions min et max
sentinel = object()

## Calcul du minimum
def ft_min(*values, default=sentinel, key=None):
    if not values:
        raise TypeError('min expected at least 1 argument, got 0')

    if len(values) == 1: ## Si un seul argument, on iter dessus
        it = iter(values[0])
    else: ## Sinon on les compare directement
        if default is not sentinel:
            raise TypeError('Cannot specify a default for min() with multiple positional arguments')
        it = iter(values)

	## Trouver le maximum
    smallest = next(it, sentinel)
    if smallest is sentinel: ## Séquence vide
        if default is not sentinel: ## Retourner la valeur par défaut si fournie
            return default
        raise ValueError('min() arg is an empty sequence')

    if key is None: ## Si aucune clé de comparaison n'est fournie, on compare directement
        for x in it:
            if x < smallest:
                smallest = x
        return smallest

	## Si une clé de comparaison est fournie
	## key est une fonction que l'on applique aux éléments pour obtenir la valeur de comparaison
    smallest_key = key(smallest)
    for x in it:
        kx = key(x)
        if kx < smallest_key:
            smallest = x
            smallest_key = kx
    return smallest

## Calcul du maximum
def ft_max(*values, default=sentinel, key=None):
    if not values:
        raise TypeError('max expected at least 1 argument, got 0')

    if len(values) == 1: ## Si un seul argument, on iter dessus
        it = iter(values[0])
    else: ## Sinon on les compare directement
        if default is not sentinel:
            raise TypeError('Cannot specify a default for max() with multiple positional arguments')
        it = iter(values)

	## Trouver le maximum
    largest = next(it, sentinel)
    if largest is sentinel: ## Séquence vide
        if default is not sentinel: ## Retourner la valeur par défaut si fournie
            return default
        raise ValueError('max() arg is an empty sequence')

    if key is None: ## Si aucune clé de comparaison n'est fournie, on compare directement
        for x in it:
            if x > largest:
                largest = x
        return largest

	## Si une clé de comparaison est fournie
	## key est une fonction que l'on applique aux éléments pour obtenir la valeur de comparaison
    largest_key = key(largest)
    for x in it:
        kx = key(x)
        if kx > largest_key:
            largest = x
            largest_key = kx
    return largest

## Calcul de la covariance
## Mesure de la façon dont deux variables varient ensemble (écarts par rapport à leurs moyennes)
def ft_covariance(x, y):
    mean_x = ft_mean(x)
    mean_y = ft_mean(y)
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / (len(x) - 1)
    return cov

## Calcul de la corrélation
## Ajustement d’une variable par rapport à l’autre par une relation affine obtenue par régression linéaire, quotiens de la covariance par le produit des écarts-types
def ft_correlation(x, y):
    cov = ft_covariance(x, y)
    return cov / (ft_std_dev(x) * ft_std_dev(y))

## Calcul du pourcentage
def ft_percentile(values, q):
	sorted_vals = sorted(values)
	n = ft_count(sorted_vals)
	pos = q * (n - 1) ## position dans la liste triée
	low = int(pos) ## partie entière de la position
	high = low + 1 ## partie entière + 1

	if high >= n: # Si la position est supérieur au nombre d'éléments, retourne la valeur à cette position
		return sorted_vals[low]
	# interpolation linéaire correcte: low + (high - low) * fraction
	# (pos - low) est la fraction décimale entre low et high, nous dis de combien avancer entre low et hight pour obtenir la valeur au pourcentage demandé
	# (sorted_vals[high] - sorted_vals[low]) l'écart entre les deux valeurs
	# On additionne low pour obtenir la valeur exacte au pourcentage demandé
	return sorted_vals[low] + (sorted_vals[high] - sorted_vals[low]) * (pos - low)
