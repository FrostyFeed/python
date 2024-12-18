import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Завантаження та підготовка даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 1000  # Зменшуємо розмір даних для швидшого тестування

# Читання даних з файлу
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            y.append(0)  # Клас 0 для <=50K
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            y.append(1)  # Клас 1 для >50K
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape, dtype=object)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

# Залишаємо лише числові дані та мітки
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Створення SVM класифікатора з поліноміальним ядром
classifier = SVC(kernel='poly', degree=3, random_state=0)  # Зменшено degree

# Розбиття на навчальний та тестовий набори (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Навчання класифікатора
print("Training model with polynomial kernel...")
classifier.fit(X_train, y_train)

# Прогнозування для тестового набору
y_test_pred = classifier.predict(X_test)

# Обчислення метрик якості
print("Classification Report for Polynomial Kernel:")
print(classification_report(y_test, y_test_pred))
