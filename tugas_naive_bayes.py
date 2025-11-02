import pandas as pd

# Data tabel sebagai dataframe
discount_df = pd.DataFrame({
    'Discount': ['Yes', 'No'],
    'Buy_Yes': [19, 5],
    'Buy_No': [5, 5]
})

free_df = pd.DataFrame({
    'FreeDelivery': ['Yes', 'No'],
    'Buy_Yes': [21, 3],
    'Buy_No': [3, 4]
})

day_df = pd.DataFrame({
    'Day': ['Weekday', 'Weekend', 'Holiday'],
    'Buy_Yes': [9, 7, 8],
    'Buy_No': [2, 1, 3]
})

# Probabilitas dasar
total_buy_yes = 24
total_buy_no = 10
total_data = 34

P_buy = total_buy_yes / total_data
P_notbuy = total_buy_no / total_data

# Fungsi menghitung probabilitas dengan pandas
def prob_buy(day, free, disc):
    p = P_buy
    p *= day_df.loc[day_df.Day == day, 'Buy_Yes'].values[0] / total_buy_yes
    p *= free_df.loc[free_df.FreeDelivery == free, 'Buy_Yes'].values[0] / total_buy_yes
    p *= discount_df.loc[discount_df.Discount == disc, 'Buy_Yes'].values[0] / total_buy_yes
    return p

def prob_notbuy(day, free, disc):
    p = P_notbuy
    p *= day_df.loc[day_df.Day == day, 'Buy_No'].values[0] / total_buy_no
    p *= free_df.loc[free_df.FreeDelivery == free, 'Buy_No'].values[0] / total_buy_no
    p *= discount_df.loc[discount_df.Discount == disc, 'Buy_No'].values[0] / total_buy_no
    return p

# Kasus uji A-H
cases = [
    ("Weekday", "Yes", "Yes"),
    ("Weekday", "Yes", "No"),
    ("Weekday", "No", "Yes"),
    ("Weekday", "No", "No"),
    ("Weekend", "Yes", "Yes"),
    ("Weekend", "Yes", "No"),
    ("Weekend", "No", "Yes"),
    ("Weekend", "No", "No"),
]

for i, (day, fd, disc) in enumerate(cases, start=1):
    pb = prob_buy(day, fd, disc)
    pnb = prob_notbuy(day, fd, disc)
    keputusan = "Buy" if pb > pnb else "Not Buy"
    print(f"{chr(96+i)}. {day}, FreeDelivery={fd}, Discount={disc}")
    print(f"   P(Buy|X)     = {pb:.8f}")
    print(f"   P(NotBuy|X)  = {pnb:.8f}")
    print(f"   👉 Prediksi: {keputusan}\n")
