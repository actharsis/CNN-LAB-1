# Обучение нейронной сети,представленной в примере, для решения задачи классификации изображений Food-101
## Архитектура нейронной сети
Файл:
```
CNN-food-101-master/train.py
```
Нейронная сеть включает в себя один свёрточный слой.
```python
x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
```
Операция двумерного максимизирующего пулинга:
```python
x = tf.keras.layers.MaxPool2D()(x)
```
Приведение матрицы признаков к вектору:
```python
x = tf.keras.layers.Flatten()(x)
```
Dense слой с функцией активации softmax
```python
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
## Графики
Метрика качества

![legend1](https://user-images.githubusercontent.com/24518594/114280227-05746d00-9a41-11eb-8748-2033abf44027.png)

![gr1](https://github.com/actharsis/lab1/blob/main/graph/epoch_categorical_accuracy.svg)

Функция потерь

![gr2](https://github.com/actharsis/lab1/blob/main/graph/epoch_loss.svg)
# Создание и обучение сверточной нейронной сети произвольной архитектуры с количеством сверточных слоев >3
## Архитектура нейронной сети
Файл:
```
CNN-food-101-master/train_multiple_layers.py
```
Отличия от прошлой нейронной сети заключаются лишь в добавлении 3 дополнительных свёрточных слоёв.
```python
x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
```
## Графики
Метрика качества

![legend2](https://user-images.githubusercontent.com/24518594/114280451-f3df9500-9a41-11eb-90aa-3f63b223fc93.png)

![gr3](https://github.com/actharsis/lab1/blob/main/graph/epoch_categorical_accuracy_multiple_layers.svg)

Функция потерь

![gr4](https://github.com/actharsis/lab1/blob/main/graph/epoch_loss_multiple_layers.svg)
# 3.Анализ результатов
После добавления к исходной сети дополнительных 3 сверточных слоёв по функции потерь и метрике качества можно сказать, что эффективность обучения сильно снизилась. График функции потерь на validation заметно хуже train, что говорит о переобучении сети.
