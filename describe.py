import sys
from utils import data_tools, stats_tools

def describe(data):
    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    blacklist = {"index", "id", "first name", "last name", "name"}

    numeric_cols = [col for col in data_tools.to_numeric_list(data)
                        if col.lower() not in blacklist]
    if not numeric_cols:
        print("No numeric colums found.")
        return

    numeric_data = {}
    ignored = []
    for col, values in data.items():
        if col in numeric_cols:
            numeric_data[col] = values
        else:
            ignored.append(col)

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
    col_widths = {}
    for col in numeric_data.keys():
        max_val_len = max(len(values_str[(col, st)]) for st in stats)
        col_name_len = len(col)
        col_widths[col] = max(12, max_val_len, col_name_len + 2)

    print(f"{'':<{stat_col_width}}", end="")
    for col in numeric_data.keys():
        print(f"{col:>{col_widths[col]}} ", end="")
    print()

    for stat in stats:
        print(f"{stat:<{stat_col_width}}", end="")
        for col in numeric_data.keys():
            s = values_str[(col, stat)]
            print(f"{s:>{col_widths[col]}} ", end="")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <filename>")
        sys.exit(1)

    dataset = data_tools.load_csv(sys.argv[1])
    describe(dataset)