from FileManager import FileM
from AiManager import AiM, Layer, Network

AiM = AiM()
neyro = Network([784, 128, 32, 10])# создание нейросети
lr = 0.1# изм при обучении
epochs =20# кол-во проходов
print("Обучение")
for epoch in range(epochs):
    total_loss = 0
    count = 0
    l_data = AiM.load_data_tn()
    if epoch%10==0:
        lr/=2.0# yield value, res
    for val, res in l_data:
        predict = neyro.forward(val)
        loss = AiM.error(predict, res)
        total_loss += loss
        count += 1
        neyro.train_one(val, res, lr=lr)

    mid_loss = total_loss / count if count != 0.0 else 0.0
    print(f"epoch(Эпоха): {epoch} — loss(Ошибка): {mid_loss}")

print("Тесты")
nums = [0,1,2,3,4,5,6,7,8,9]

for numt in range(1):
    tests = AiM.load_data_tt()
    correct = 0
    count = 0
    t_l = 0
    for test, r in tests:
        pred =  neyro.forward(test)
        l = AiM.error(pred, r)
        t_l += l
        count+=1
        if pred.index(max(pred)) == r.index(max(r)):
            correct += 1

    accuracy = correct/count if count>0 else 0.0
    a_l = t_l/count if count > 0 else 0.0
    print(f"Количество предсказаний — {count} из них верно — {accuracy*100}% ошибки — {a_l}")
fm = FileM()
fm.dump_AI(neyro)