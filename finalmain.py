import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest, f_classif


# ================= CONFIG =================

DATA_PATH = "processed.cleveland.data"
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV = 5


COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]


# ================= LOAD =================

def load_data():

    df = pd.read_csv(
        DATA_PATH,
        names=COLUMNS,
        na_values="?"
    )

    df = df.dropna()
    df = df.apply(pd.to_numeric)

    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


# ================= EVALUATE =================

def evaluate(model, X_test, y_test, linear=False):

    pred = model.predict(X_test)

    if linear:
        prob = pred
    else:
        prob = model.predict_proba(X_test)[:,1]

    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec  = recall_score(y_test, pred)
    f1   = f1_score(y_test, pred)
    auc  = roc_auc_score(y_test, prob)

    return acc, prec, rec, f1, auc


# ================= MAIN =================

def main():

    start = time.time()

    print("\nLoading Cleveland data...\n")

    X, y = load_data()


    X_train, X_test, y_train, y_test = train_test_split(

        X, y,

        test_size=TEST_SIZE,

        stratify=y,

        random_state=RANDOM_STATE
    )


    cv = StratifiedKFold(
        n_splits=CV,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    results = []


    best_acc = 0
    best_name = ""


    # ================= TUNED LOGISTIC =================

    print("Tuning Logistic Regression...\n")

    lr_pipe = Pipeline([

        ("scaler", StandardScaler()),

        ("select", SelectKBest(f_classif)),

        ("clf", LogisticRegression(
            solver="saga",
            max_iter=5000,
            class_weight="balanced"
        ))
    ])


    lr_params = {

        "select__k": [6,8,10,12],

        "clf__C": np.logspace(-2,2,15),

        "clf__penalty": ["l1","l2"]
    }


    lr_grid = GridSearchCV(

        lr_pipe,

        lr_params,

        scoring="roc_auc",

        cv=cv,

        n_jobs=-1
    )


    lr_grid.fit(X_train, y_train)

    best_lr = lr_grid.best_estimator_


    acc,prec,rec,f1,auc = evaluate(best_lr, X_test, y_test)

    results.append(["Logistic", acc, prec, rec, f1, auc])


    if acc > best_acc:

        best_acc = acc
        best_name = "Logistic"


    # ================= OTHER MODELS =================

    models = {


        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=10)),
            ("clf", SVC(
                probability=True,
                C=5,
                gamma="scale"
            ))
        ]),


        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=10)),
            ("clf", LinearSVC(max_iter=5000))
        ]),


        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            random_state=RANDOM_STATE
        ),


        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            random_state=RANDOM_STATE
        ),


        "Gradient Boost": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE
        ),


        "AdaBoost": AdaBoostClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        ),


        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))
        ]),


        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),


        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE
        )

    }


    print("Training other models...\n")


    for name, model in models.items():

        print("Training:", name)

        model.fit(X_train, y_train)


        if name == "Linear SVM":

            acc,prec,rec,f1,auc = evaluate(model, X_test, y_test, linear=True)

        else:

            acc,prec,rec,f1,auc = evaluate(model, X_test, y_test)


        results.append([name, acc, prec, rec, f1, auc])


        if acc > best_acc:

            best_acc = acc
            best_name = name


    # ================= RESULTS =================

    print("\n========== FINAL RESULTS ==========\n")


    for name, acc, prec, rec, f1, auc in results:

        print(
            f"{name:14s} | "
            f"Acc:{acc:.4f} | "
            f"Prec:{prec:.4f} | "
            f"Rec:{rec:.4f} | "
            f"F1:{f1:.4f} | "
            f"AUC:{auc:.4f}"
        )


    print("\n=============================")

    print("Best Model (By Accuracy):", best_name)

    print("Best Accuracy:", round(best_acc*100,2), "%")

    print("Time Taken:", round((time.time()-start)/60,2), "min")



if __name__ == "__main__":
    main()
