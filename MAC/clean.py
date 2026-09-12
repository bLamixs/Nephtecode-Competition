import pandas as pd

print("1. Загрузка собранной витрины...")
df = pd.read_csv('master_dataset.csv', parse_dates=['date'])
initial_shape = df.shape[0]

print("2. Очистка от простоев и стартовых пустых значений...")
# Считаем суммарный расход сырой нефти (нагрузка на АВТ)
df['total_feed'] = df['F7'] + df['F8'] + df['F9']

# Если расход меньше 50 тонн/час, считаем, что установка стоит (простой/ремонт)
df_clean = df[df['total_feed'] > 50].copy()

# Удаляем строки в начале датасета, где ЛИМС еще не сделал первый замер
df_clean = df_clean.dropna(subset=['lims_sulfur'])

# Удаляем явные сбои датчиков (когда давление или температура падают ниже нуля)
df_clean = df_clean[(df_clean['P8'] > 0) & (df_clean['T1'] > 0)]

print(f"Было строк: {initial_shape}")
print(f"Осталось чистых строк: {df_clean.shape[0]}")

print("3. Временной сплит (Time-based split)...")
# Все данные до 2026 года отдаем на обучение агентам
# Все данные за 2026 год оставляем для тестирования (Holdout)
split_date = '2026-01-01'

train_df = df_clean[df_clean['date'] < split_date].copy()
test_df = df_clean[df_clean['date'] >= split_date].copy()

print(f"Размер Train (обучение): {train_df.shape[0]} строк")
print(f"Размер Test (проверка): {test_df.shape[0]} строк")

# Сохраняем готовые датасеты
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
print("Готово! Данные сохранены.")