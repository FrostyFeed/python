from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Розділення датасету на X та y
array = dataset.values
X = array[:, 0:4]  # Перші 4 стовпці
y = array[:, 4]    # 5-й стовпець (мітки)

# Розділення на навчальну та тестову вибірки (80% - навчальні дані, 20% - тестові)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Завантажуємо алгоритми моделі
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Оцінка моделі на кожній ітерації
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)  # 10-кратна крос-валідація
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

# Порівняння алгоритмів
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

model = SVC(gamma='auto')

# Навчаємо модель на тренувальних даних
model.fit(X_train, y_train)

# Прогнозування на контрольній вибірці
predictions = model.predict(X_validation)

# Оцінка якості
print("Точність:", accuracy_score(y_validation, predictions))
print("Матриця помилок:\n", confusion_matrix(y_validation, predictions))
print("Звіт про класифікацію:\n", classification_report(y_validation, predictions))

# Крок 8: Прогнозування для нової квітки
X_new = np.array([[5, 2.9, 1, 0.2]])

# Перевіряємо форму масиву
print("Форма масиву X_new:", X_new.shape)

# Прогнозування для нових даних
prediction = model.predict(X_new)

# Виведення прогнозу
print("Прогноз: ", prediction)

# Прогнозована мітка (якщо prediction містить значення класу без індексації)
predicted_class = prediction[0]  # Отримуємо мітку класу
print("Спрогнозована мітка: ", predicted_class)






