import pandas as pd

def build_master_dataset(avt_path, hydro_path, pak_path, lims_path):
    print("1. Загрузка телеметрии...")
    # Читаем телеметрию с правильным разделителем и парсим даты
    df_avt = pd.read_csv(avt_path, parse_dates=['date'])
    df_242 = pd.read_csv(hydro_path, parse_dates=['date'])

    # Фильтруем только нужные колонки (наши MV + несколько важных состояний)
    avt_cols = ['date', 'F7', 'F8', 'F9', 'T1', 'F12', 'F14', 'F30', 'F32']
    hydro_cols = ['date', 'T11', 'F26', 'P8', 'P24', 'F15']
    
    df_avt = df_avt[avt_cols].drop_duplicates(subset=['date']).sort_values('date')
    df_242 = df_242[hydro_cols].drop_duplicates(subset=['date']).sort_values('date')

    # Сливаем телеметрию (у них синхронный шаг в 10 минут)
    df_master = pd.merge(df_avt, df_242, on='date', how='inner')
    
    print("2. Загрузка данных ПАК (сера)...")
    # Исходя из структуры ПАК, берем первые две колонки: дата и значение серы
    df_pak = pd.read_excel(pak_path, skiprows=1, usecols=[0, 1])
    df_pak.columns = ['pak_date', 'pak_sulfur']
    df_pak['pak_date'] = pd.to_datetime(df_pak['pak_date'], errors='coerce')
    df_pak = df_pak.dropna().sort_values('pak_date')
    
    print("3. Загрузка данных ЛИМС (сера)...")
    # ПРИМЕЧАНИЕ: Здесь нужно будет указать точные индексы колонок для даты и серы 
    # из точки отбора "Гидроочистка.. Точка 2. Продукт Дизельное топливо"
    # Для примера предполагаем, что дата это колонка 82, а сера - 90
    df_lims = pd.read_excel(lims_path, skiprows=3, header=None, usecols=[94, 95])
    df_lims.columns = ['lims_date', 'lims_sulfur']
    df_lims['lims_date'] = pd.to_datetime(df_lims['lims_date'], errors='coerce')
    df_lims = df_lims.dropna().sort_values('lims_date')

    print("4. Слияние (merge_asof) без заглядывания в будущее...")
    # Присоединяем ПАК
    df_master = pd.merge_asof(
        df_master, 
        df_pak, 
        left_on='date', 
        right_on='pak_date', 
        direction='backward'  # Берем только прошлые значения!
    )
    
    # Присоединяем ЛИМС
    df_master = pd.merge_asof(
        df_master, 
        df_lims, 
        left_on='date', 
        right_on='lims_date', 
        direction='backward'
    )

    print("5. Расчет возраста данных (Data Freshness)...")
    # Рассчитываем, сколько времени прошло с момента последнего замера
    df_master['pak_age_minutes'] = (df_master['date'] - df_master['pak_date']).dt.total_seconds() / 60.0
    df_master['lims_age_hours'] = (df_master['date'] - df_master['lims_date']).dt.total_seconds() / 3600.0
    
    # Очищаем вспомогательные колонки дат
    df_master = df_master.drop(columns=['pak_date', 'lims_date'])

    return df_master

if __name__ == "__main__":
    # Запуск по умолчанию
    master_df = build_master_dataset(
        'data/raw/avt_tags.csv', 
        'data/raw/242000_tags.csv', 
        'data/raw/ПАК.xlsx', 
        'data/raw/ЛИМСы.xlsx'
    )
    master_df.to_csv('data/processed/master_dataset.csv', index=False)
    print("Датасет сохранён в data/processed/master_dataset.csv")
    print(master_df.head())
