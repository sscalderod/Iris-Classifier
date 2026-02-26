import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ---------------------------
# Helpers
# ---------------------------
def load_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)
    return X, y, feature_names, class_names


def get_model_and_pipeline(
    model_name: str,
    use_scaler: bool,
    params: dict,
) -> Pipeline:
    if model_name == "Regresión Logística":
        model = LogisticRegression(
            C=params["C"],
            max_iter=2000,
            solver="lbfgs",
            multi_class="auto",
            random_state=params["random_state"],
        )

    elif model_name == "KNN":
        model = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            metric=params["metric"],
        )

    elif model_name == "SVM (RBF)":
        model = SVC(
            C=params["C"],
            gamma=params["gamma"],
            probability=True,  # para curvas ROC/PR
            random_state=params["random_state"],
        )

    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"],
        )
    else:
        raise ValueError("Modelo no soportado.")

    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def plot_feature_histograms(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(10, 6))
    n = df.shape[1]
    rows = 2
    cols = int(np.ceil(n / rows))
    for i, col in enumerate(df.columns, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.hist(df[col], bins=18)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    corr = df.corr(numeric_only=True)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Matriz de correlación (features)")
    fig.tight_layout()
    return fig


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    return fig


def plot_multiclass_roc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int, class_names: list[str]) -> plt.Figure:
    # One-vs-rest ROC
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC multiclase (One-vs-Rest)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_multiclass_pr(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int, class_names: list[str]) -> plt.Figure:
    Y = label_binarize(y_true, classes=list(range(n_classes)))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(Y[:, i], y_proba[:, i])
        ap = average_precision_score(Y[:, i], y_proba[:, i])
        ax.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall multiclase (One-vs-Rest)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_learning_curve(estimator: Pipeline, X: pd.DataFrame, y: pd.Series, cv_splits: int, seed: int) -> plt.Figure:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=None,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    ax.plot(train_sizes, train_mean, marker="o", label="Train")
    ax.plot(train_sizes, val_mean, marker="o", label="Validación (CV)")

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)

    ax.set_xlabel("Tamaño de entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Clasificador IRIS (Pedagógico)", layout="wide")

st.title("🌸 Clasificador IRIS en Streamlit (dinámico + pedagógico)")
st.write(
    """
Este app te guía por un flujo típico de *Machine Learning* con IRIS:
1) explorar datos, 2) escoger modelo, 3) entrenar y evaluar, 4) probar predicción con entradas nuevas.
"""
)

X, y, feature_names, class_names = load_data()
df = X.copy()
df["target"] = y
df["species"] = df["target"].map(lambda i: class_names[int(i)])

# Sidebar controls
st.sidebar.header("⚙️ Configuración")

model_name = st.sidebar.selectbox(
    "Modelo",
    ["Regresión Logística", "KNN", "SVM (RBF)", "Random Forest"],
    index=0,
)

use_scaler = st.sidebar.checkbox(
    "Estandarizar features (StandardScaler)",
    value=True,
    help="Recomendado para Regresión Logística, KNN y SVM. No suele ser necesario para Random Forest.",
)

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

cv_splits = st.sidebar.slider("CV folds (para curvas y predicciones CV)", 3, 10, 5, 1)

st.sidebar.subheader("🔧 Hiperparámetros")
params = {"random_state": int(random_state)}

if model_name == "Regresión Logística":
    params["C"] = st.sidebar.slider("C (regularización inversa)", 0.01, 10.0, 1.0, 0.01)

elif model_name == "KNN":
    params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 30, 7, 1)
    params["weights"] = st.sidebar.selectbox("weights", ["uniform", "distance"])
    params["metric"] = st.sidebar.selectbox("metric", ["minkowski", "euclidean", "manhattan"])

elif model_name == "SVM (RBF)":
    params["C"] = st.sidebar.slider("C", 0.1, 50.0, 5.0, 0.1)
    params["gamma"] = st.sidebar.selectbox("gamma", ["scale", "auto"])

elif model_name == "Random Forest":
    params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 600, 250, 25)
    max_depth_opt = st.sidebar.selectbox("max_depth", ["None", "2", "3", "4", "5", "6", "8", "10"])
    params["max_depth"] = None if max_depth_opt == "None" else int(max_depth_opt)
    params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 10, 2, 1)

pipeline = get_model_and_pipeline(model_name, use_scaler, params)

# Main layout
tab1, tab2, tab3 = st.tabs(["📚 Dataset & EDA", "🧪 Entrenamiento & Evaluación", "🔮 Predicción (Interactiva / CSV)"])

with tab1:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("Vista del dataset")
        st.dataframe(df, use_container_width=True)
        st.caption("IRIS tiene 150 filas, 4 features numéricas y 3 clases (setosa, versicolor, virginica).")

        st.subheader("Estadísticas rápidas")
        st.dataframe(df[feature_names].describe().T, use_container_width=True)

    with c2:
        st.subheader("Distribución de features")
        fig = plot_feature_histograms(df[feature_names])
        st.pyplot(fig, use_container_width=True)

        st.subheader("Correlación")
        fig = plot_correlation_heatmap(df[feature_names])
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("Entrenamiento y evaluación")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    train_button = st.button("🚀 Entrenar modelo", type="primary")

    if train_button:
        # Fit
        pipeline.fit(X_train, y_train)

        # Test predictions
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

        st.success("Modelo entrenado ✅")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy (test)", f"{acc:.3f}")
        m2.metric("Precision macro", f"{prec:.3f}")
        m3.metric("Recall macro", f"{rec:.3f}")
        m4.metric("F1 macro", f"{f1:.3f}")

        st.write("### Matriz de confusión (test)")
        fig = plot_confusion(y_test.to_numpy(), y_pred, class_names)
        st.pyplot(fig, use_container_width=True)

        # Cross-validated probabilities (for ROC/PR) – uses the full dataset for stable curves
        st.write("### Curvas ROC y Precision-Recall (con predicción Cross-Validation)")
        cv = StratifiedKFold(n_splits=int(cv_splits), shuffle=True, random_state=int(random_state))

        # cross_val_predict for probas needs predict_proba; all our models provide it
        try:
            y_proba_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
            fig = plot_multiclass_roc(y.to_numpy(), y_proba_cv, n_classes=3, class_names=class_names)
            st.pyplot(fig, use_container_width=True)

            fig = plot_multiclass_pr(y.to_numpy(), y_proba_cv, n_classes=3, class_names=class_names)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(
                "No pude calcular curvas ROC/PR con CV para este modelo/configuración. "
                f"Detalle: {e}"
            )

        st.write("### Learning Curve (Accuracy)")
        try:
            fig = plot_learning_curve(pipeline, X, y, cv_splits=int(cv_splits), seed=int(random_state))
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"No pude calcular la learning curve. Detalle: {e}")

        st.info(
            "Tip pedagógico: si Train >> Validación en la learning curve, suele indicar overfitting. "
            "Si ambos son bajos, suele indicar underfitting o falta de señal."
        )
    else:
        st.caption("Configura el modelo a la izquierda y presiona **Entrenar modelo**.")

with tab3:
    st.subheader("Predicción interactiva (ingreso manual)")

    st.write(
        "Ajusta los sliders con medidas de la flor y presiona **Predecir**. "
        "La predicción depende del modelo y configuración actual del panel izquierdo."
    )

    # Default values from dataset ranges
    mins = X.min()
    maxs = X.max()
    means = X.mean()

    ic1, ic2 = st.columns(2)
    with ic1:
        sepal_length = st.slider("sepal length (cm)", float(mins[0]), float(maxs[0]), float(means[0]), 0.1)
        sepal_width = st.slider("sepal width (cm)", float(mins[1]), float(maxs[1]), float(means[1]), 0.1)
    with ic2:
        petal_length = st.slider("petal length (cm)", float(mins[2]), float(maxs[2]), float(means[2]), 0.1)
        petal_width = st.slider("petal width (cm)", float(mins[3]), float(maxs[3]), float(means[3]), 0.1)

    input_df = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=feature_names,
    )

    colp1, colp2 = st.columns([1, 1.2])
    with colp1:
        if st.button("🔮 Predecir", type="primary"):
            # Entrenar con el split actual para que sea consistente con el panel
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
            )
            pipeline.fit(X_train, y_train)

            pred = int(pipeline.predict(input_df)[0])
            st.write(f"**Predicción:** {class_names[pred]}")

            # Probabilidades si el modelo las soporta
            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(input_df)[0]
                proba_df = pd.DataFrame({"Clase": class_names, "Probabilidad": proba})
                st.dataframe(proba_df, use_container_width=True)
            else:
                st.caption("Este modelo no expone probabilidades (predict_proba).")

    with colp2:
        st.write("### Predicción por CSV (batch)")
        st.caption(
            "Sube un CSV con columnas exactamente iguales a las features:\n"
            f"`{', '.join(feature_names)}`"
        )
        file = st.file_uploader("CSV", type=["csv"])
        if file is not None:
            batch = pd.read_csv(file)
            missing = [c for c in feature_names if c not in batch.columns]
            if missing:
                st.error(f"Faltan columnas en tu CSV: {missing}")
            else:
                # Entrenar antes de predecir batch
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
                )
                pipeline.fit(X_train, y_train)

                preds = pipeline.predict(batch[feature_names])
                out = batch.copy()
                out["pred_class_id"] = preds
                out["pred_species"] = out["pred_class_id"].map(lambda i: class_names[int(i)])

                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(batch[feature_names])
                    for i, name in enumerate(class_names):
                        out[f"proba_{name}"] = probs[:, i]

                st.success("Predicción batch lista ✅")
                st.dataframe(out, use_container_width=True)

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Descargar resultados (CSV)",
                    data=csv_bytes,
                    file_name="iris_predictions.csv",
                    mime="text/csv",
                )

st.caption("Hecho para aprendizaje: juega con el modelo, el escalado y los hiperparámetros para ver cómo cambian las métricas y curvas.")
