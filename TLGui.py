from FileManager import FileM
from PhotoManager import PhotoM
from threading import Thread
from time import sleep
from math import e
from AiManager import AiM, Layer, Network
from pathlib import Path
import os, sys
class GUI:
    def __init__(self):
        self.fm = FileM()
        self.pm = PhotoM()
        self.neyro = None
        self.select = None
        self.start()
    def start(self):
        t1 = Thread(target=self.load)
        print("Запуск программы")
        tmp = input("Начать загрузку модели?(y/n) —")
        if tmp == 'y':
            t1.start()
            sleep(1)
            print("Инициализация модели...")
            t1.join()
            if self.neyro is None:
                print("Модель не найдена...")
                self.exit()
            print("Запуск модели...")
            sleep(2)
        else:
            print("Модель не загружена часть функций недоступна")
        print("Запуск меню...")
        self.advice()
    def load(self):
        self.neyro = self.fm.load_AI()
    def advice(self):
        self.clear()
        print(("-"*14)+"Уведомление"+("-"*15))
        print("Меню:")
        print("Для справки нажмите 0 + enter")
        print("Точность ~95% хорошо прорисованного изображения")
        print("-"*40)
        input("enter...")
        self.main_menu()
    def main_menu(self):
        flag = True
        while flag:
            self.clear()
            print("Меню:")
            com = input(": ")
            if com == "0":
                print("0 — справка\n1 — сканирование папки\n2 — Запуск ИИ\n3 — выход")
                input("enter...")
            if com == "1":
                self.scan()
            if com == "2":
                self.AI()
            if com == "3":
                self.exit()
                flag = False
    def scan(self):
        flag = True
        while flag:
            self.clear()
            print("Сканирование...")
            paths = self.fm.scandir()
            if not paths:
                flag = False
                print('Изображений не обнаружено')
                input("enter...")
                continue
            i = []
            for indx, path in enumerate(paths):
                i.append(indx)
                print(f"{indx+1} — {Path(path).stem}")
            print("Выберите изображение(Введите номер)")
            try:
                answer = int(input("— "))
            except ValueError:
                print("Введите число")
                input("enter...")
                continue
            if 1 > answer > len(paths):
                print("Такого значения нет... Попробуйте снова")
                input("enter...")
            else:
                self.select = paths[int(answer)-1]
                print("Изображение успешно выбрано")
                input("enter...")
                flag = False
    def plato(self, answer: list)->list:
        result = []
        summary = sum(answer)
        for i in answer:
            result.append(i/summary)
        return result
    def printer(self, target: list):
        max = 0
        result = 0
        for indx, a in enumerate(target):
            percent = int(a*100)
            if percent>max:
                max = percent
                result = indx
            print(f"{indx}=>{percent}%:{'/'*(percent//5)}")
        print("-"*50)
        print("Вы ввели: ", result)
    def AI(self):
        self.clear()
        if self.select is None or self.neyro is None:
            print("Функция недоступна")
            input('enter...')
            self.main_menu()
        img = self.pm.get_image(self.select)
        predict = self.neyro.forward(img)
        pp = self.plato(predict)
        print("Анализ изображения...")
        sleep(3)
        self.printer(pp)
        input("enter...")
    def exit(self):
        sys.exit()
    def clear(self):
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
    def softmax(self, target:list)->list:
        result = []
        summary = sum(e**i for i in target)
        for item in target:
            result.append(item/summary)
        return result