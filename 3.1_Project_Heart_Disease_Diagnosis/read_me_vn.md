# Phân loại bệnh tim: Feature Engineering thủ công và OpenFE

[English version](read_me_eng.md)

Dự án tái hiện và mở rộng một pipeline phân loại bệnh tim trên bộ dữ liệu Cleveland. Mục tiêu là so sánh dữ liệu gốc đã tiền xử lý, feature engineering (FE) thủ công và FE tự động bằng OpenFE với cùng cách chia dữ liệu và các mô hình đánh giá.

Kết quả test cao nhất được ghi nhận là **90.32% Accuracy (28/31 mẫu đúng)** với Stacking trên FE thủ công. KNN trên OpenFE đã tuning đạt **87.10% (27/31 mẫu đúng)**. Đây là kết quả của một lần chia dữ liệu cố định, không phải bằng chứng về khả năng tổng quát hóa trên dữ liệu lâm sàng mới.

## 1. Dữ liệu và bài toán

- **303 mẫu**, **13 đặc trưng đầu vào** và một nhãn mục tiêu.
- Nguồn: [Heart Disease — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease), sử dụng phần Cleveland.
- File đầu vào: [data/cleveland.csv](data/cleveland.csv), không có dòng tên cột.
- Dữ liệu gốc có 4 giá trị thiếu ở `ca`, 2 giá trị thiếu ở `thal`; ký hiệu `?` được chuyển thành giá trị thiếu khi xử lý.
- Bài toán cuối cùng là phân loại nhị phân: **0 = không bệnh**, **1 = có bệnh**.

### Bảng mô tả đặc trưng Cleveland Heart Disease

| Đặc trưng | Mô tả và giá trị mã hóa |
| --- | --- |
| `age` | Tuổi của bệnh nhân, tính bằng năm. |
| `sex` | Giới tính: `1` = nam, `0` = nữ. |
| `cp` | Loại đau ngực: `1` = đau thắt ngực điển hình; `2` = không điển hình; `3` = đau ngực không do tim; `4` = không triệu chứng. |
| `trestbps` | Huyết áp lúc nghỉ khi nhập viện, đơn vị mmHg. |
| `chol` | Nồng độ cholesterol huyết thanh, đơn vị mg/dL. |
| `fbs` | Đường huyết lúc đói vượt 120 mg/dL: `1` = đúng, `0` = sai. |
| `restecg` | Điện tâm đồ lúc nghỉ: `0` = bình thường; `1` = bất thường ST–T; `2` = phì đại thất trái theo tiêu chí của bộ dữ liệu. |
| `thalach` | Nhịp tim tối đa đạt được, tính bằng lần/phút. |
| `exang` | Đau thắt ngực do gắng sức: `1` = có, `0` = không. |
| `oldpeak` | Mức độ trầm đoạn ST do gắng sức so với lúc nghỉ. |
| `slope` | Độ dốc đoạn ST khi gắng sức đạt đỉnh: `1` = dốc lên; `2` = bằng phẳng; `3` = dốc xuống. |
| `ca` | Số mạch máu chính được quan sát bằng huỳnh quang: từ `0` đến `3`. |
| `thal` | Mã kết quả thal: `3` = bình thường; `6` = khiếm khuyết cố định; `7` = khiếm khuyết có thể hồi phục. |
| `num` | Nhãn gốc: `0` = không bệnh; `1–4` = có bệnh. Trong code, cột này được đặt tên `target`. |

`thal` được mô tả theo mã kết quả của tài liệu UCI; không được dùng ở đây như nhãn chẩn đoán bệnh Thalassemia. File nguồn giữ nhãn gốc, còn các tập xuất trong `splits` chuyển thành:

```python
target = (num > 0).astype(int)
```

## 2. Những phần đã thực hiện

### Khám phá dữ liệu

- Kiểm tra kiểu dữ liệu, phân bố lớp và giá trị thiếu.
- Vẽ cột chồng cho các biến categorical theo target; histogram và boxplot cho biến numeric.
- Vẽ ma trận tương quan Pearson và scatter plot có đường hồi quy, phân màu theo tình trạng bệnh.

### Chia dữ liệu và tiền xử lý

Chia hai lần bằng `train_test_split`, có stratify theo target và `random_state=42`:

| Tập | Số mẫu | Không bệnh / có bệnh |
| --- | ---: | ---: |
| Train | 242 | 131 / 111 |
| Validation | 30 | 16 / 14 |
| Test | 31 | 17 / 14 |

Median, mode, scaler, encoder và các bước chọn đặc trưng chỉ được fit trên train rồi áp dụng cho validation/test.

| Biểu diễn dữ liệu | Xử lý numeric | Xử lý categorical | Số đặc trưng cuối |
| --- | --- | --- | ---: |
| Baseline `data` | Điền median, StandardScaler | Điền mode, MinMaxScaler | 13 |
| FE thủ công `fe_data` | Điền median, StandardScaler | Điền mode, One-hot | 13 |
| OpenFE đã tuning `openfe_data` | Điền median, StandardScaler | Điền mode, One-hot | 30 |

### Feature Engineering thủ công

Hàm `adding_feature()` tạo `chol/age`, `trestbps/age`, `thalach/age` và nhóm tuổi `age_bin` bằng `pd.cut(..., bins=5)`. Sau tiền xử lý, loại cột hằng và xếp hạng mutual information (MI) trên train để chọn **Top-13**, bằng số đặc trưng đầu vào ban đầu.

Pipeline được đối chiếu với solution tham khảo; sáu CSV baseline/FE đã được kiểm tra trùng giá trị và thứ tự cột với dữ liệu tham khảo.

### Feature Engineering tự động với OpenFE

- Tự động xét **95 ứng viên bậc một** từ các biến numeric; chỉ dùng phép biến đổi theo từng dòng, không dùng thống kê nhóm/tần suất giữa các tập.
- Dùng `stage1_metric="predictive"`, `n_data_blocks=1`, `min_candidate_features=95` và xếp hạng LightGBM gain.
- So sánh `feature_boosting=False` và `True` trên train.
- Loại cột hằng/trùng; giữ 28 cột gốc sau One-hot rồi forward selection trên các đặc trưng sinh thêm, dùng Accuracy của Stratified 5-Fold KNN CV.
- Không đặt Top-K cố định; dừng khi không còn đặc trưng mới cải thiện điểm CV.

Hai cấu hình boosting cùng đạt **85.10% Accuracy CV**; chọn `False` khi hòa. Forward selection tăng điểm từ **82.64% lên 85.10%**, giữ thêm `age/thalach` và `age*oldpeak`, tạo bộ dữ liệu cuối **30 đặc trưng**. MI vẫn được tính để mô tả và visualize, không quyết định số cột xuất.

### Mô hình và đánh giá

| Mô hình | Cấu hình / cách chọn tham số |
| --- | --- |
| KNN | Tìm `k=1..20` bằng Stratified 5-Fold có shuffle, chọn Accuracy trung bình cao nhất. |
| Decision Tree | Tìm `max_depth=2..5` bằng cùng phương pháp CV. |
| KMeans | Hai cụm, `init="random"`; ánh xạ cụm sang lớp đa số bằng nhãn train. |
| Gaussian Naive Bayes | `GaussianNB()` trên từng biểu diễn dữ liệu. |
| Stacking | Cơ sở: KNN `k=5`, DT không giới hạn độ sâu, GaussianNB. Tầng cuối: KNN `k=5`, học từ xác suất out-of-fold của 5-fold **không shuffle**, `passthrough=False`. |

`evaluate_val.py` và `evaluate_test.py` dùng chung phần tính Accuracy, Balanced Accuracy, Precision/Recall/F1 macro và confusion matrix. KNN tối ưu lần lượt là **11 / 15 / 5** cho baseline / FE thủ công / OpenFE; DT chọn **depth=3** cho cả ba. Stacking giữ cấu hình riêng của solution, không lấy tham số tối ưu của các model độc lập.

## 3. Kết quả hiện tại

Accuracy trên **cùng 31 mẫu test**:

| Mô hình | Baseline | FE thủ công | OpenFE đã tuning |
| --- | ---: | ---: | ---: |
| KNN | 83.87% | 83.87% | 87.10% |
| Decision Tree | 80.65% | 80.65% | 80.65% |
| KMeans | 87.10% | 87.10% | 87.10% |
| Gaussian Naive Bayes | 83.87% | 83.87% | 70.97% |
| Stacking | 83.87% | **90.32%** | 80.65% |

| Stacking | Accuracy validation | Accuracy test | F1 macro test |
| --- | ---: | ---: | ---: |
| Baseline | 86.67% | 83.87% | 0.8387 |
| FE thủ công | 93.33% | 90.32% | 0.9032 |
| OpenFE đã tuning | 96.67% | 80.65% | 0.8062 |

Thử nghiệm OpenFE giữ toàn bộ đặc trưng từng tạo 104 cột và cho Stacking Accuracy test 64.52%. Chọn lọc theo CV nâng kết quả lên 80.65%, nhưng chưa vượt FE thủ công. Kết quả cho thấy thêm đặc trưng hoặc dùng ensemble không tự động cải thiện mọi model.

## 4. File chính và cách chạy

| File / thư mục | Vai trò |
| --- | --- |
| [feature_engineering.ipynb](feature_engineering.ipynb) | EDA, baseline, FE thủ công, MI và xuất các tập chia. |
| [heart_disease_diagnosis.ipynb](heart_disease_diagnosis.ipynb) | Huấn luyện và so sánh baseline/FE thủ công. |
| [heart_disease_dignosis_openfe.ipynb](heart_disease_dignosis_openfe.ipynb) | Tuning OpenFE, xuất dữ liệu và so sánh cả ba biểu diễn. |
| [evaluate_val.py](evaluate_val.py), [evaluate_test.py](evaluate_test.py) | CV, huấn luyện và đánh giá tái sử dụng. |
| [stacking_ensemble.py](stacking_ensemble.py) | Xây dựng và fit Stacking trên train. |
| [openfe_features.py](openfe_features.py) | Sinh đặc trưng, tuning boosting và forward selection. |
| [splits/](splits/) | `train/val/test.csv`, `fe_train/val/test.csv`, `openfe_train/val/test.csv`; `openfe_feature_info.json` lưu công thức, cấu hình và chỉ số dòng. |

Chạy từ thư mục project, dùng cùng một môi trường Python cho terminal và kernel Jupyter:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter ipykernel
python -m pip install -r requirements_openfe.txt
jupyter notebook
```

Chạy **Run All** theo thứ tự: `feature_engineering.ipynb` → `heart_disease_diagnosis.ipynb` → `heart_disease_dignosis_openfe.ipynb`. Notebook OpenFE có bước tái sinh ba CSV OpenFE.

Thiết lập seed 42 cho `PYTHONHASHSEED`, NumPy, `random` và các bộ chia/model có hỗ trợ. Môi trường đã chạy: Python 3.14.7, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, OpenFE 0.0.12, LightGBM 4.7.0. Phiên bản thư viện và thứ tự cột có thể ảnh hưởng kết quả.

## 5. Giới hạn và bài học

- Test chỉ có 31 mẫu: một dự đoán tương đương khoảng 3.23 điểm phần trăm Accuracy. Không lấy điểm validation cao làm kết quả test.
- Tiền xử lý và khám phá/chọn đặc trưng được fit trên outer train trước các fold CV; điểm CV là điểm tuning, có thể lạc quan. Nested CV với toàn bộ pipeline fit lại trong từng fold là hướng mở rộng.
- Để tái hiện solution, `age_bin` thủ công dùng `pd.cut` riêng trên từng tập; ranh giới nhóm tuổi vì vậy có thể khác nhau. Phiên bản triển khai tiếp theo nên học ranh giới trên train và tái sử dụng.
- OpenFE chỉ xét các phép biến đổi numeric bậc một trong thí nghiệm này, không đại diện cho toàn bộ không gian cấu hình của thư viện.
- Cấu hình OpenFE được chọn theo train; validation/test chỉ được dùng để báo cáo sau khi đã chọn. Tuy vậy, test đã được xem qua nhiều vòng phát triển, nên cần một holdout mới cho đánh giá cuối độc lập.

## 6. Nguồn và ghi nhận

Pipeline baseline/FE thủ công được tái hiện từ solution bài học AIO2025, sau đó mở rộng bằng module đánh giá tái sử dụng, Stacking riêng, OpenFE tuning và phân tích kết quả. Tài liệu mô tả bài toán nằm trong [references/](references/).

- [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) — dữ liệu và bảng mã thuộc tính; dataset được UCI công bố theo CC BY 4.0.
- [Dataset DOI: 10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
- [OpenFE: mã nguồn và hướng dẫn](https://github.com/IIIS-Li-Group/OpenFE).
- [OpenFE Parameters Tuning](https://openfe-document.readthedocs.io/en/latest/parameter_tuning.html).
