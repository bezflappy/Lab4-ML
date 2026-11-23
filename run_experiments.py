"""Сравнение ансамблей на german.csv."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ensemble_models import CANDIDATES, SELECTED_MODEL, calculate_metrics, make_model, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

ROOT = Path(__file__).resolve().parent
SEED = 42
METRICS = ("roc_auc", "accuracy", "precision", "recall", "f1")


def threshold_metric(target, probability, metric):
    return calculate_metrics(target, (probability >= 0.5).astype(int), probability)[metric]


def prepare_data():
    frame = pd.read_csv(ROOT / "german.csv", sep=";")
    if frame.columns[0] != "Creditability":
        raise ValueError("Unexpected target column")
    x_train, x_test, y_train, y_test = train_test_split(
        frame.iloc[:, 1:].to_numpy(dtype=float), frame.iloc[:, 0].to_numpy(),
        test_size=0.2, random_state=SEED, stratify=frame.iloc[:, 0],
    )
    return EnsembleInput(x_train, y_train, x_test, y_test, random_seed=SEED), frame


def select_on_train(data):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=data.random_seed)
    rows = []
    for name in CANDIDATES:
        start = time.perf_counter()
        scores = cross_validate(
            make_model(name, data.random_seed), data.train_features, data.train_target,
            cv=cv, scoring={"roc_auc": "roc_auc", **{
                metric: make_scorer(threshold_metric, response_method="predict_proba", metric=metric)
                for metric in METRICS if metric != "roc_auc"
            }}, n_jobs=1, error_score="raise",
        )
        row = {"model": name, "parameters": CANDIDATES[name],
               "cv_mean": {metric: float(scores["test_"+metric].mean()) for metric in METRICS},
               "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
               "fold_roc_auc": scores["test_roc_auc"].tolist(),
               "cv_seconds": time.perf_counter()-start}
        rows.append(row)
        print(f"{name}: CV AUC={row['cv_mean']['roc_auc']:.6f}", flush=True)
    return max(rows, key=lambda row: row["cv_mean"]["roc_auc"])["model"], rows


def run(output_dir=ROOT, show=False, select_only=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data, frame = prepare_data()
    with threadpool_limits(limits=1):
        winner, rows = select_on_train(data)
        if select_only:
            print("TRAIN-ONLY WINNER:", winner)
            (output_dir / "selection.json").write_text(json.dumps({"winner": winner, "experiments": rows}, indent=2), encoding="utf-8")
            return winner
        if winner != SELECTED_MODEL:
            raise RuntimeError(f"Training-CV winner is {winner}, but fixed contract model is {SELECTED_MODEL}. Resolve using training data before evaluating test.")
        selected = run_ensemble_experiment(data)
        probabilities = {}
        # Модель уже выбрана. Теперь считаем результаты на test.
        for row in rows:
            model = make_model(row["model"], SEED)
            model.fit(data.train_features, data.train_target)
            prob = model.predict_proba(data.test_features)[:, list(model.classes_).index(1)]
            probabilities[row["model"]] = prob
            row["test"] = calculate_metrics(data.test_target, (prob >= 0.5).astype(int), prob)
        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
        baseline.fit(data.train_features, data.train_target)
        baseline_prob = baseline.predict_proba(data.test_features)[:, 1]
        baseline_metrics = calculate_metrics(data.test_target, (baseline_prob >= 0.5).astype(int), baseline_prob)
    elapsed = time.perf_counter()-start
    selected_metrics = {metric: float(getattr(selected, metric)) for metric in METRICS}
    payload = {
        "model_name": selected.model_name, **selected_metrics,
        "selected_model": SELECTED_MODEL, "parameters": CANDIDATES[SELECTED_MODEL],
        "selection_criterion": "max mean train-only stratified 5-fold ROC-AUC",
        "random_seed": SEED, "train_rows": len(data.train_target), "test_rows": len(data.test_target),
        "experiments": rows, "logistic_baseline": baseline_metrics,
        "predictions": selected.predictions.tolist(), "probabilities": selected.probabilities.tolist(),
        "runtime_seconds": elapsed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    (output_dir / "experiment_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels = [row["model"] for row in rows]
    positions = np.arange(len(rows))
    axes[0].barh(positions, [row["cv_mean"]["roc_auc"] for row in rows],
                 xerr=[row["cv_roc_auc_std"] for row in rows], color=["#278762" if name==winner else "#6587ac" for name in labels])
    axes[0].set(yticks=positions, yticklabels=labels, xlabel="Mean validation ROC-AUC (+/- fold SD)", xlim=(0.5,1), title="Selection uses TRAIN folds only")
    axes[0].invert_yaxis()
    for label, prob in [(winner, selected.probabilities), ("LogisticRegression baseline", baseline_prob)]:
        fpr, tpr, _ = roc_curve(data.test_target, prob)
        axes[1].plot(fpr, tpr, label=label)
    axes[1].plot([0,1], [0,1], "--", color="gray")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Frozen models: TEST ROC")
    axes[1].legend(fontsize=8)
    figure.savefig(output_dir / "model_comparison.png", dpi=150)
    matrix = confusion_matrix(data.test_target, selected.predictions, labels=[0,1])
    confusion_fig, axis = plt.subplots(figsize=(5,4), constrained_layout=True)
    axis.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i,j]), ha="center", va="center", color="white" if matrix[i,j]>matrix.max()/2 else "black", fontsize=18)
    axis.set(xticks=[0,1], yticks=[0,1], xlabel="Predicted class", ylabel="True class", title="Selected model: TEST confusion matrix")
    confusion_fig.savefig(output_dir / "confusion_matrix.png", dpi=150)

    metric_rows = lambda key: "\n".join("| " + row["model"] + " | " + " | ".join(f"{row[key][metric]:.6f}" for metric in METRICS) + " |" for row in rows)
    selected_cv = next(row for row in rows if row["model"]==winner)
    delta = selected.roc_auc - baseline_metrics["roc_auc"]
    hypotheses = [
        f"Random Forest: leaf=1 → leaf=3 изменил CV AUC с {rows[0]['cv_mean']['roc_auc']:.6f} до {rows[1]['cv_mean']['roc_auc']:.6f}; проверялось сглаживание листьев для снижения переобучения.",
        f"Extra Trees: leaf=1 → leaf=2 изменил CV AUC с {rows[2]['cv_mean']['roc_auc']:.6f} до {rows[3]['cv_mean']['roc_auc']:.6f}; проверялось сочетание случайных порогов и сглаживания.",
        f"Gradient Boosting: depth=1 → depth=2 изменил CV AUC с {rows[4]['cv_mean']['roc_auc']:.6f} до {rows[5]['cv_mean']['roc_auc']:.6f}; проверялась польза парных взаимодействий признаков.",
        f"HistGradientBoosting с 15 листьями и L2=2 получил CV AUC {rows[6]['cv_mean']['roc_auc']:.6f}; проверялась регуляризованная гистограммная модель с ограниченной сложностью.",
    ]
    parameter_lines = "\n".join(f"- `{name}`: `{json.dumps(params)}`." for name, params in CANDIDATES.items())
    report = f'''# Лабораторная №4. Ансамбли

## Данные и подготовка

Используется german.csv: {len(frame)} строк и {frame.shape[1]-1} признаков. Целевая переменная — Creditability, первый столбец. Разделитель — `;`. Пропусков: {int(frame.isna().sum().sum())}, дубликатов строк: {int(frame.duplicated().sum())}.

Данные разделены в отношении 80/20 с сохранением долей классов, random_state=42. На обучение приходится {len(data.train_target)} строк, на тест — {len(data.test_target)}. Численность классов [0, 1]: train — {np.bincount(data.train_target.astype(int)).tolist()}, test — {np.bincount(data.test_target.astype(int)).tolist()}. Разбиение такое же, как в третьей работе.

Используются все признаки, категориальные коды оставлены числовыми. Масштабирование для деревьев не применяется. В pipeline добавлен SimpleImputer: медианы вычисляются только на обучающих данных; полностью пустой столбец заполняется нулём. В данном CSV заполнять пропуски не понадобилось.

Параметры сравниваются на train с помощью StratifiedKFold: 5 частей, shuffle=True, random_state=42. В каждом запуске 640 строк используются для обучения и 160 — для проверки. Лучшая конфигурация выбирается по средней ROC-AUC, затем обучается на всех 800 строках. Тестовая выборка используется только для итоговой оценки.

## Гипотезы и эксперименты

Сравниваются Random Forest, Extra Trees, Gradient Boosting и HistGradientBoosting — всего семь конфигураций. У лесов меняется минимальный размер листа, у Gradient Boosting — глубина дерева. Предположение: ограничение сложности уменьшит переобучение, а более глубокие деревья бустинга смогут учесть взаимодействия признаков.

{parameter_lines}

Остальные параметры оставлены стандартными. Seed — 42, n_jobs=1; вычисления BLAS/OpenMP также ограничены одним потоком. Веса классов не меняются, early stopping у HistGradientBoosting отключён. Время обучения и оценки всех моделей без построения графиков: {elapsed:.2f} с.

{chr(10).join('- '+h for h in hypotheses)}

## Метрики

Средние результаты пятикратной кросс-валидации на train. Критерий выбора — ROC-AUC.

| Конфигурация | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
{metric_rows('cv_mean')}

Результаты на test. Эта таблица рассчитана после выбора модели и на выбор не влияет. Для сравнения добавлена готовая логистическая регрессия из третьей работы.

| Конфигурация | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
{metric_rows('test')}
| LogisticRegression (baseline №3) | {' | '.join(f'{baseline_metrics[m]:.6f}' for m in METRICS)} |

ROC-AUC считается по вероятностям класса 1. Для остальных метрик порог равен 0.5: при равенстве выбирается класс 1. Положительный класс — Creditability=1. Порог заранее фиксирован; для Precision, Recall и F1 задан zero_division=0.

![Сравнение моделей и ROC](model_comparison.png)

![Матрица ошибок](confusion_matrix.png)

## Выбранная модель

Лучший результат на кросс-валидации показал **{selected.model_name}**: средняя ROC-AUC — **{selected_cv['cv_mean']['roc_auc']:.6f}**, стандартное отклонение — {selected_cv['cv_roc_auc_std']:.6f}. Поэтому для функции run_ensemble_experiment выбран этот вариант: 250 деревьев, min_samples_leaf=3, max_features="sqrt".

На test получены ROC-AUC = **{selected.roc_auc:.6f}**, Accuracy = **{selected.accuracy:.6f}**, Precision = {selected.precision:.6f}, Recall = {selected.recall:.6f}, F1 = {selected.f1:.6f}.

Лес обучает деревья на bootstrap-выборках и использует случайные подмножества признаков при разбиении. Итоговая вероятность усредняется по деревьям. Минимум три объекта в листе ограничивает слишком мелкие разбиения; в этом опыте это улучшило среднюю ROC-AUC по сравнению с листом из одного объекта.

## Выводы и ограничения

Выбранный лес превысил результат логистической регрессии на test: разница ROC-AUC составила {delta:+.6f}, Accuracy — {selected.accuracy-baseline_metrics['accuracy']:+.6f}. Разбиение у моделей одинаковое; масштабирование для логистической регрессии обучено только на train.

Увеличение размера листа улучшило результаты обоих лесов. У Gradient Boosting глубина 2 дала более высокую ROC-AUC на кросс-валидации, чем глубина 1, но не обошла выбранный Random Forest. Сам по себе переход к более сложной модели лучшего результата не дал.

Вывод относится к одному разбиению небольшого датасета. Классы несбалансированы, поэтому одной Accuracy недостаточно. Категории закодированы числами, хотя между ними не всегда есть порядок; можно отдельно проверить one-hot кодирование. Высокий Recall класса 1 не означает такого же качества для класса 0 — ошибки видны на матрице. Доверительные интервалы и калибровка вероятностей в этой работе не рассчитывались.

## Воспроизводимость

Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}. Версии библиотек закреплены в requirements.txt. Python 3.12+ для указанного окружения.

```bash
python -m pip install -r requirements.txt
python run_experiments.py
python -m unittest discover -v
```

Для просмотра графиков: `python run_experiments.py --show`. При запуске тестов графики также открываются; нужно закрыть окна, чтобы тесты завершились. Запуск без окон: `MPLBACKEND=Agg python -m unittest discover -v`.

Скрипт сохраняет REPORT.md, experiment_results.json и два графика. В JSON записаны параметры, оценки кросс-валидации, метрики и прогнозы на test. Тесты сверяют метрики с допуском 1e-6. Время выполнения при повторном запуске может отличаться.
'''
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"selected_model": winner, **selected_metrics}, indent=2), flush=True)
    if show:
        plt.show()
    plt.close("all")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    run(output_dir=args.output_dir, show=args.show, select_only=args.select_only)
