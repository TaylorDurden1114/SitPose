from matplotlib import rcParams
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import dump, load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

df = pd.read_csv('H:\\研究生材料\\大论文\\data\\俯视数据\\data_right_all.csv')
label_mapping = {'healthy': 0, 'lay': 1, 'wrong': 2, 'left': 3, 'right': 4, 'forward': 5, 'stand': 6}
df['label'] = df['label'].map(label_mapping)

# 标签名称
class_labels = ['healthy', 'lay', 'wrong', 'left', 'right', 'forward', 'stand']

X = df.iloc[:, 0: 13]  # 特征
Y = df.iloc[:, 13]  # 标签
print(np.unique(Y))
print(Y.dtype)
# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建分类器实例
svm_clf = SVC(kernel='poly', probability=True, random_state=42)
dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
mlp_clf = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=300, random_state=42)
# xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# 存储每个模型的平均准确率
average_accuracies = {}
average_precisions = {}
average_recalls = {}
average_f1_scores = {}

# 训练基学习器并计算准确率、精准率、召回率和F1分数
models = [svm_clf, dt_clf, mlp_clf]
model_names = ['SVM', 'Decision Tree', 'MLP']
for model, name in zip(models, model_names):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    # 计算平均准确率
    average_accuracy = np.mean(scores)
    # 存储结果
    average_accuracies[name] = average_accuracy
    print(f"{name} Cross-validated average accuracy: {average_accuracy}")

    # 计算平均精准率
    precision_scores = cross_val_score(model, X_train, y_train, cv=5,
                                       scoring=make_scorer(precision_score, average='macro'))
    average_precision = np.mean(precision_scores)
    average_precisions[name] = average_precision

    # 计算平均召回率
    recall_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(recall_score, average='macro'))
    average_recall = np.mean(recall_scores)
    average_recalls[name] = average_recall

    # 计算平均F1分数
    f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(f1_score, average='macro'))
    average_f1 = np.mean(f1_scores)
    average_f1_scores[name] = average_f1

    # 打印结果
    print(f"{name} Cross-validated average precision: {average_precision}")
    print(f"{name} Cross-validated average recall: {average_recall}")
    print(f"{name} Cross-validated average F1 score: {average_f1}\n")

    # 训练模型并计算混淆矩阵
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(label_mapping.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.clear()
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='')
    ax.set_xlabel('Predicted Label', fontsize=25, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=25, fontweight='bold')
    plt.xticks(fontsize=22, rotation=30, fontweight='bold')
    plt.yticks(fontsize=22, fontweight='bold')
    # 清除默认文本标注
    for artist in ax.texts:
        artist.remove()

    # Iterate over data dimensions and create text annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="black", fontsize=22, fontweight='bold')
    plt.subplots_adjust(left=0.2, right=1.0)  # 调整图像的位置
    plt.savefig(f"{name}_confusion_matrix.pdf", format='pdf', dpi=700)
    plt.show()

# 创建包含模型和其名称的元组列表，用于VotingClassifier的estimators参数
estimators = list(zip(model_names, models))

# 从average_accuracies字典中提取权重，保持与estimators相同的顺序
weights = [average_accuracies[name] for name in model_names]

# 创建软投票的集成分类器
voting_clf = VotingClassifier(
    estimators=estimators,
    voting='soft', weights=weights  # 使用软投票
)


# 训练集成模型
voting_clf.fit(X_train, y_train)

# 预测并计算准确率
y_pred = voting_clf.predict(X_test)
print(f"Voting Classifier accuracy: {accuracy_score(y_test, y_pred)}")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 计算并显示混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=list(label_mapping.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(10, 10))
ax.clear()
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='')
# 清除默认文本标注
for artist in ax.texts:
    artist.remove()

ax.set_xlabel('Predicted Label', fontsize=25, fontweight='bold')
ax.set_ylabel('True Label', fontsize=25, fontweight='bold')
plt.xticks(fontsize=22, rotation=30, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')
# Iterate over data dimensions and create text annotations.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="black", fontsize=22, fontweight='bold')
plt.subplots_adjust(left=0.2, right=1.0)
plt.savefig("Voting_Classifier_confusion_matrix.pdf", format='pdf', dpi=700)
plt.show()
dump(voting_clf, 'voting_classifier.joblib')
