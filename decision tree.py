
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Memuat dataset Wine
wine = load_wine()
X = wine.data
y = wine.target

# 2. Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Membuat model baseline
baseline = DecisionTreeClassifier(random_state=42)
baseline.fit(X_train, y_train)

print("=== BASELINE ===")
print("Jumlah node:", baseline.tree_.node_count)
print("Akurasi training:", baseline.score(X_train, y_train))
print("Akurasi testing:", baseline.score(X_test, y_test))

# 4. Membuat model dengan pruning
models = {
    "Max Depth 3": DecisionTreeClassifier(
        max_depth=3, random_state=42
    ),
    "Min Samples Leaf 5": DecisionTreeClassifier(
        min_samples_leaf=5, random_state=42
    ),
    "CCP Alpha 0.01": DecisionTreeClassifier(
        ccp_alpha=0.01, random_state=42
    )
}

print("\n=== HASIL PRUNING ===")

for nama, model in models.items():
    model.fit(X_train, y_train)

    print("\nModel:", nama)
    print("Jumlah node:", model.tree_.node_count)
    print("Akurasi training:", model.score(X_train, y_train))
    print("Akurasi testing:", model.score(X_test, y_test))

# 5. Menggambar pohon hasil pruning
model_pilihan = models["Max Depth 3"]

plt.figure(figsize=(20, 10))
plot_tree(
    model_pilihan,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True
)

plt.title("Decision Tree Wine - Max Depth 3")
plt.tight_layout()
plt.savefig("pohon_decision_tree.png", dpi=150)
plt.show()
