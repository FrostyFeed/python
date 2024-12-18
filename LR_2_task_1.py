import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Завантаження та підготовка даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

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

# Створення SVМ-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))

# Розбиття на навчальний та тестовий набори (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Навчання класифікатора
classifier.fit(X_train, y_train)

# Прогнозування для тестового набору
y_test_pred = classifier.predict(X_test)

# Обчислення F1-міри та інших метрик
f1 = classifier.score(X_test, y_test)
print("F1 score:", f1)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

# Перевірка результату для тестової точки
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([item])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded)

# Прогнозування для тестової точки даних
predicted_class = classifier.predict([input_data_encoded])
predicted_class_label = label_encoder[-1].inverse_transform(predicted_class)[0]

print("Predicted class for the input data:", predicted_class_label)



