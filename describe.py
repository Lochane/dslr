import sys
from utils import data_tools, stats_tools

def describe(data, threshold=0.9, min_numeric=2):
    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    blacklist = {"index", "id", "first name", "last name", "name"}

    numeric_data = {}
    ignored = []
    for col, values in data.items():
        if col.lower() in blacklist:
            ignored.append(col)
            continue
        if data_tools.is_numeric_column(values, threshold=threshold, min_numeric=min_numeric):
            numeric_data[col] = data_tools.to_numeric_list(values)
        else:
            ignored.append(col)

    cols = list(numeric_data.keys())
    if not cols:
        print("No numeric columns found.")
        return

    values_str = {}
    for col in cols:
        vals = numeric_data[col]
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
    col_widths = {}
    for col in cols:
        max_val_len = max(len(values_str[(col, st)]) for st in stats)
        col_name_len = len(col)
        col_widths[col] = max(12, max_val_len, col_name_len + 2)

    print(f"{'':<{stat_col_width}}", end="")
    for col in cols:
        print(f"{col:>{col_widths[col]}} ", end="")
    print()

    for stat in stats:
        print(f"{stat:<{stat_col_width}}", end="")
        for col in cols:
            s = values_str[(col, stat)]
            print(f"{s:>{col_widths[col]}} ", end="")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <filename>")
        sys.exit(1)

    dataset = data_tools.load_csv(sys.argv[1])
    describe(dataset)