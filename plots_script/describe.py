import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_tools, stats_tools

def describe(data):
    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    blacklist = {"index", "id", "first name", "last name", "name"}

    numeric_cols = [col for col in data_tools.to_numeric_list(data)
                        if col.lower() not in blacklist]
    if not numeric_cols:
        print("No numeric columns found.")
        return

    numeric_data = {}
    ignored = []
    for col, values in data.items():
        if col in numeric_cols:
            # Nettoyage: retirer None et NaN; ne garder que les valeurs numériques valides
            cleaned = []
            for v in values:
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                # Détecter NaN (NaN est la seule valeur où x != x)
                if fv != fv:
                    continue
                cleaned.append(fv)
            # Ignorer les colonnes ne contenant que des NaN/None
            if cleaned:
                numeric_data[col] = cleaned
            else:
                ignored.append(col)
        else:
            ignored.append(col)

    # Si toutes les colonnes numériques sont vides après nettoyage
    if not numeric_data:
        print("No numeric columns with data.")
        return

    values_str = {}
    for col, vals in numeric_data.items():
        for stat in stats:
            try:
                if stat == "Count":
                    v = float(len(vals))
                elif stat == "Mean":
                    v = stats_tools.ft_mean(vals)
                elif stat == "Std":
                    v = stats_tools.ft_std_dev(vals)
                elif stat == "Min":
                    v = stats_tools.ft_min(vals)
                elif stat == "25%":
                    v = stats_tools.ft_percentile(vals, 0.25)
                elif stat == "50%":
                    v = stats_tools.ft_percentile(vals, 0.50)
                elif stat == "75%":
                    v = stats_tools.ft_percentile(vals, 0.75)
                elif stat == "Max":
                    v = stats_tools.ft_max(vals)
                else:
                    v = None
                    
                if v is None:
                    s = "NaN"
                else:
                    try:
                        s = f"{float(v):.6f}"
                    except Exception:
                        s = "NaN"
            except Exception:
                s = "NaN"
            values_str[(col, stat)] = s

    stat_col_width = 12

# Calculer largeur de chaque colonne
    col_widths = {
        col: stats_tools.ft_max(len(col), stats_tools.ft_max(len(values_str[(col, st)]) for st in stats), 12)
        for col in numeric_data.keys()
    }

# En-tête
    print(f"{'Stat':<{stat_col_width}}", end="|")
    for col in numeric_data.keys():
        print(f"{col:>{col_widths[col]}}", end="|")
    print()

# Lignes de stats
    for stat in stats:
        print(f"{stat:<{stat_col_width}}", end="|")
        for col in numeric_data.keys():
            print(f"{values_str[(col, stat)]:>{col_widths[col]}}", end="|")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <filename>")
        sys.exit(1)

    try :
        dataset = data_tools.load_csv(sys.argv[1])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    describe(dataset)