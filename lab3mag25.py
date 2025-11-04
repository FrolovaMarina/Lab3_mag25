import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('Coffee_sales.csv')
#print(f"Размер датасета: {df.shape}")

df_res = df.copy()
encoder_coffee = LabelEncoder()
df_res['coffee_encode'] = encoder_coffee.fit_transform(df['coffee_name'])

features = ['hour_of_day', 'Weekdaysort', 'Monthsort', 'coffee_encode']
x = df_res[features]
y = df_res['money']
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)
scaler = StandardScaler()
x_training_scaled = scaler.fit_transform(x_training_data)
x_test_scaled = scaler.transform(x_test_data)

models = {
    #Случайный лес 
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    #К-ближайших соседей 
    'K-Neighbors': KNeighborsRegressor(n_neighbors = 5),
    #Дерево решений 
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    #Адаптивный бустинг 
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.3),
    #Линейная регрессия 
    'Linear Regression': LinearRegression(), 
    #Опорные векторы
    'SVM': SVR()
}

results = {}

print("\nРезультаты: ")
for name, model in models.items():
    model.fit(x_training_scaled, y_training_data)   
    predictions = model.predict(x_test_scaled) 
    
    mae = mean_absolute_error(y_test_data, predictions)
    mse = mean_squared_error(y_test_data, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_data, predictions)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"\n{name}")
    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"R2 Score: {r2:.5f}")

# График 1. Сравнение моделей по метрике R² 
plt.figure(figsize=(10, 6))
r2_res = [results[name]['R2'] for name in results.keys()]
bars = plt.bar(results.keys(), r2_res, color=['#228B22', '#FF8C00', '#66CDAA', '#DB7093', '#87CEFA', '#000080'])
plt.title('Сравнение моделей по метрике R²', fontsize=14, fontweight='bold')
plt.ylabel('R²')
plt.tight_layout()

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# График 2. Сравнение моделей по метрике RMSE 
plt.figure(figsize=(10, 6))
rmse_res = [results[name]['RMSE'] for name in results.keys()]
bars = plt.bar(results.keys(), rmse_res, color=['#228B22', '#FF8C00', '#66CDAA', '#DB7093', '#87CEFA', '#000080'])
plt.title('Сравнение моделей по метрике RMSE', fontsize=14, fontweight='bold')
plt.ylabel('RMSE')
plt.tight_layout()

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

#График 3. Сравнение моделей по метрике MAE
plt.figure(figsize=(10, 6))
mae_res = [results[name]['MAE'] for name in results.keys()]
bars = plt.bar(results.keys(), mae_res, color=['#228B22', '#FF8C00', '#66CDAA', '#DB7093', '#87CEFA', '#000080'])
plt.title('Сравнение моделей по метрике MAE', fontsize=14, fontweight='bold')
plt.ylabel('MAE')
plt.tight_layout()

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

#График корреляции признаков с целевой переменной 
plt.figure(figsize=(10, 6))
corr_matrix = df_res[features + ['money']].corr()
corr_fixed = corr_matrix['money'].drop('money')
plt.barh(corr_fixed.index, corr_fixed.values, color='#9370D8')
plt.title('Корреляция признаков с целевой переменной money', fontweight='bold')
plt.xlabel('Значение коэффициента корреляции')
plt.xlim(-1, 1)

for index, value in enumerate(corr_fixed.values):
    plt.text(value, index, f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center')
    
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.show()

