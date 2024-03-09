import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Veri setini yükle
veri = pd.read_csv("data.csv")
# Veri setinin ilk birkaç satırını göster
print(veri.head())
# Eksik değerleri kontrol et
print(veri.isnull().sum())
X = veri[['YearsExperience']]  # Bağımsız değişken
y = veri['Salary']    # Bağımlı değişken
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# Eğitim verileri üzerinde tahmin yap
train_tahmin = model.predict(X_train)

# Test verileri üzerinde tahmin yap
test_tahmin = model.predict(X_test)

# Eğitim ve test hatasını hesapla
train_hata = mean_squared_error(y_train, train_tahmin)
test_hata = mean_squared_error(y_test, test_tahmin)

print("Eğitim Hatası:", train_hata)
print("Test Hatası:", test_hata)
# Eğitim verilerini grafiğe çiz
plt.scatter(X_train, y_train, color='blue', label='Eğitim Verileri')

# Eğitim verileri üzerinde yapılan tahminleri grafiğe çiz
plt.plot(X_train, model.predict(X_train), color='red', label='Eğitim Verilerine Göre Tahmin')

# Test verilerini grafiğe çiz
plt.scatter(X_test, y_test, color='green', label='Test Verileri')

# Test verileri üzerinde yapılan tahminleri grafiğe çiz
plt.plot(X_test, model.predict(X_test), color='orange', label='Test Verilerine Göre Tahmin')

plt.xlabel('Tecrübe(Yıl)')
plt.ylabel('Maaş')
plt.title('Lineer Regresyon Tahminleri')
plt.legend()
plt.show()
