# nti-ai-pipeline
## Team: <̷͊s̸̕o̵͒s̶̈́>̶̀, Authors: [Georgii Surkov](https://github.com/GeorgiySurkov) and [Protsenko Arseny](https://github.com/ProtsenkoAI)

Наше решение, которое дало скор 0.8053 на public было blending'ом трёх бертов, обученных на kfold из train-части датасета. 
Основные моменты решения:
1. Конкатенируем ключевое слово (скрытое под @placeholder) с query, в BERT подаём два текста: main_token + query и текст новости
2. Иногда в train правильные ответы есть в разметке, но их нет в entities: мы добавляем их на этапе зарузки данных (весь препроцессинг 
  распределён по модулям rucos_contain, rucos_train_dataset и rucos_processor). Мы также извлекаем NER-вероятности с помощью deeppavlov ner-ru-bert, 
3. Выход NER и CLS-token из BERT конкатенируются и подаются в head - 1 или 2 линейных слоя
4. Для обучения используются lr_sceduler, amp
5. Из каждого текста мы извлекаем entity как отдельный объект и проводит бинарную классификацию. В submission попадает entity с максимальной вероятностью 
  (см реализацию rucos_submitting.py)
6. Код обучения bert можно найти в research/ProtsenkoAISol.ipynb, формирование блендинга - в research/EnsemblingProtsenkoAI.ipynb
