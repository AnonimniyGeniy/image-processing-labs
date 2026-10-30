# Обработка изображений, вариант 2

Три лабораторные: адаптивная бинаризация, трекинг объекта на видео и распознавание цифр.

- [ЛР1 — отчёт](lab1/report.md), [код](lab1/code/main.py), [данные](lab1/data), [результаты](lab1/results).
- [ЛР2 — отчёт](lab2/report.md), [код](lab2/code), [данные](lab2/data), [результаты](lab2/results).
- [ЛР3 — отчёт](lab3/report.md), [код](lab3/code), [данные](lab3/data), [результаты](lab3/results).

В каждой папке лабораторной лежат `code/`, `data/`, `results/`, `report.md` и PDF задания.
Общие функции чтения изображений и сохранения JSON — [common.py](common.py).

## Установка и запуск

Использовал Python 3.9.6. Все команды выполняю из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m lab1.code.main
python -m lab2.code.run_videos
python -m lab3.code.main evaluate-own --json lab3/data/own_digits.json
```

Веса ЛР3 уже сохранены. Повторное обучение: `python -m lab3.code.main train --epochs 5 --device auto`.
Другие команды и результаты описал в отчёте каждой лабораторной.

ЛР1: обе реализации совпали по пикселям. ЛР2: объект найден на 711/711 кадрах моего видео.
ЛР3: MNIST test — 98,74%; мои цифры — 47/50 (94%).
