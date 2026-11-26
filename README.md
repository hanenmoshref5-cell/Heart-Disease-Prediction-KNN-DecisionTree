# Heart Disease Prediction — KNN vs Decision Tree 🫀

مشروع تعلم آلة (Machine Learning) للتنبؤ بوجود مرض القلب عند المريض اعتمادًا على بياناته الطبية، باستخدام خوارزميتين وعمل مقارنة شاملة بين أدائهما: **K-Nearest Neighbors (KNN)** و **Decision Tree (ID3 - Entropy)**.

## نظرة عامة

- **Dataset**: [Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) من UCI Machine Learning Repository.
- **عدد العينات**: 303 سجل.
- **عدد الخصائص (Features)**: 13 خاصية (العمر، الجنس، نوع ألم الصدر، ضغط الدم، الكوليسترول...).
- **نوع المشكلة**: تصنيف ثنائي (Binary Classification) — 0 = لا يوجد مرض، 1 = يوجد مرض.

## خطوات العمل (Pipeline)

1. **تحميل البيانات** وتسمية الأعمدة حسب وصف UCI.
2. **استكشاف البيانات (EDA)**: فحص القيم المفقودة، التوزيعات الإحصائية، توزيع الفئات، والعلاقات بين الخصائص (Correlation Heatmap).
3. **معالجة البيانات**: تعويض القيم المفقودة بالـ mode، وتحويل عمود الهدف (target) إلى تصنيف ثنائي.
4. **تقسيم البيانات**: 80% تدريب / 20% اختبار، مع Stratified Split.
5. **تطبيع البيانات (Scaling)** باستخدام StandardScaler (ضروري لخوارزمية KNN).
6. **تدريب النموذج الأول — KNN**: تجربة عدة قيم لـ K لاختيار الأفضل، ومقارنة مقياسي المسافة Euclidean و Manhattan، مع Cross Validation (5-Fold).
7. **تدريب النموذج الثاني — Decision Tree (ID3)**: باستخدام معيار Entropy، مع ضبط Hyperparameter (max_depth) ورسم الشجرة.
8. **تقييم النماذج**: Confusion Matrix, Accuracy, Precision, Recall, F1-Score لكل نموذج.
9. **المقارنة النهائية** بين KNN و Decision Tree على كل المقاييس مع رسوم بيانية توضيحية.

## التقنيات والمكتبات المستخدمة

- Python, Jupyter Notebook
- Pandas, NumPy
- Scikit-learn (KNeighborsClassifier, DecisionTreeClassifier, train_test_split, StandardScaler...)
- Matplotlib, Seaborn (التصورات البيانية)

## محتويات المشروع

| الملف | الوصف |
|---|---|
| `ML_Project_HeartDisease(3).ipynb` | النوتبوك الرئيسي الكامل (EDA + KNN + Decision Tree + المقارنة) |
| `MLpro_ByHanenAndOmar.ipynb` | نوتبوك يركز على نموذج Decision Tree وضبط إعداداته |
| `processed.cleveland.data` | بيانات مرض القلب (UCI Cleveland) |
| `heart-disease.names` | وصف الأعمدة والبيانات |

