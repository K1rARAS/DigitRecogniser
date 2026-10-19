from random import uniform, shuffle
from math import e, sqrt
from FileManager import FileM
from PhotoManager import PhotoM
from pathlib import Path
class AiM:
    def __init__(self):
        self.name = None
    def sigmoid(self, value):
        return 1 / (1 + e ** (-value)) # Значения становятся в пределах 0,1

    def error(self, val, res):
        return sum((p - v) ** 2 for p, v in zip(res, val)) / len(res)  # подсчет расхождения ответов и значений общий

    def choosen(self,target: list) -> list:
        train_base = []
        test_base = []
        mark = float(len(target) / 10) * 0.8
        for pathl in target:
            path = Path(pathl)
            if int(path.stem[2:3]) <= int(mark):
                train_base.append(pathl)
            else:
                test_base.append(pathl)
        return [train_base, test_base]

    def load_data_tn(self):
        fm = FileM()
        pm = PhotoM()
        my_images = fm.scandir(fm.paths["dt"])
        my_train = self.choosen(my_images)[0]
        indx = list(range(len(my_train)))
        shuffle(indx)
        for i in indx:
            path = my_images[i]
            answer = int(Path(path).stem[0])
            result = [0.0] * 10
            img = pm.get_image(path)
            massive = img
            value = massive
            result[answer] = 1.0
            yield value, result

    def load_data_tt(self):
        fm = FileM()
        pm = PhotoM()
        my_images = fm.scandir(fm.paths["dt"])
        my_train = self.choosen(my_images)[1]
        indx = list(range(len(my_train)))
        shuffle(indx)
        for i in indx:
            path = my_images[i]
            answer = int(Path(path).stem[0])
            result = [0.0] * 10
            img = pm.get_image(path)
            massive = img
            value = massive
            result[answer] = 1.0
            yield value, result

class Layer:
    def __init__(self, n_in, n_out):
        scale = sqrt(2.0/n_in) # небольшая оптимизация иначе комп зависнет так много считать надо
        self.matrix = [[uniform(-scale,scale) for _ in range(n_in)] for _ in range(n_out)] # создание слоя матрица строки вход столбцы выход
        self.bias = [uniform(-scale,scale) for _ in range(n_out)] # смещение для контроля

    def forward(self, inp): # сюда попадают данные
        self.last_input = inp[:] # копия входа
        self.last_z = [] # сохранение значений до выравнивания
        output = []
        for row, b in zip(self.matrix, self.bias): # берем строку слоя и смещение
            z = sum(w*m for w, m in zip(row, inp)) + b # выход нейрона(сумматор) сумма всех весов на данные + смещение
            self.last_z.append(z) # сохранение исходных значений сумматора
            output.append(AiM.sigmoid(z)) # приведение выхода к значениям в пределах от 0 до 1
        return output

class Network:
    def __init__(self, size): # допустим 2, 4, 1
        self.layers = []
        for n_in, n_out in zip(size[:-1],size[1:]): # тогда список из [(2,4), (4,1)]
            self.layers.append(Layer(n_in, n_out)) # сохранение слоев

    def forward(self, inp):
        out = inp # входные данные сохранения в выход
        for layer in self.layers: # перебор слоев
            out = layer.forward(out) # выходные значения сохраняются в выход и опять обрабатываются
        return out # конечный выход

    def train_one(self, x, t, lr=0.5):
        out = self.forward(x) # выход сохраняем в переменную
        deltas = [None] * len(self.layers) # создаем цепочку списков по слоям
        last = self.layers[-1]
        output_last = out # сохраняем выход в прошлый выход
        d_last = [] # создаем список векторов прошлый

        for k in range(len(output_last)): # указатель по выходному слою
            y = output_last[k] # значения нейрона из выходного слоя
            d_last.append((t[k] - y) * y * (1 - y)) # величина ошибки нейрона с учетом нелинейности сигмоиды: ошибка * производную f(x)(скорость)
        deltas[-1] = d_last # сохраняем влияние на ошибку каждого нейрона на предпоследнем слое на выход; индекс [4]

        for L in range(len(self.layers) - 2, -1, -1): # допустим кол-во слоев 5 тогда [3, 2, 1, 0] 4- выходной его уже посчитали
            layer_next = self.layers[L + 1] # [4, 3, 2, 1]
            layer_cur = self.layers[L] # [3, 2, 1, 0]
            d_next = deltas[L + 1] # [4, 3, 2, 1]
            a_cur = [AiM.sigmoid(z) for z in layer_cur.last_z] # выравнивание для исходных значений в пределах (0, 1)

            d_cur = [] # искомые влияние ошибки на ошибки

            for i in range(len(layer_cur.matrix)):
                s = 0.0
                for j in range(len(layer_next.matrix)):
                    s += d_next[j] * layer_next.matrix[j][i] # транспонированная матрица(вход - выход, выход - вход) * на матрицу ошибки выхода
                d_cur.append(s * a_cur[i] * (1 - a_cur[i])) # s - взвешенная сумма на производную как в цикле for до L
            deltas[L] = d_cur # сохранение значений по индексам [3, 2, 1, 0]

        for L in range(len(self.layers)): # указатель на слой
            layer = self.layers[L] # слой по индексу в переменную
            d = deltas[L] # смещения этого слоя
            inp = layer.last_input # то что пришло на вход

            for i in range(len(layer.matrix)):  # нейрон выхода слоя
                for k in range(len(layer.matrix[i])):  # вход этого нейрона
                    layer.matrix[i][k] += lr * d[i] * inp[k] # изменение весов
                layer.bias[i] += lr * d[i] # изменение смещения

