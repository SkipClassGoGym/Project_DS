# Kiến thức từ project House Price Prediction

Tài liệu tổng hợp cả ba notebook: [eda.ipynb](eda.ipynb), [main.ipynb](main.ipynb) và [main_advanced_preprocessing.ipynb](main_advanced_preprocessing.ipynb). Số liệu được đối chiếu với code, output đang lưu và dữ liệu gốc ngày 30/10/2026; các thử nghiệm cũ được ghi riêng, không xem là kết quả của code hiện tại.

| Notebook | Vai trò |
| --- | --- |
| `eda.ipynb` | Khám phá dữ liệu, missing, phân phối mục tiêu, tương quan, histogram và box plot của 10 feature |
| `main.ipynb` | Baseline preprocessing và các mô hình hồi quy, code thử tương tác polynomial |
| `main_advanced_preprocessing.ipynb` | Median, StandardScaler, log feature/target và thực nghiệm polynomial bậc 2 |

## 1. Bài toán và dữ liệu

Đây là **hồi quy có giám sát**: sử dụng thông tin căn nhà để dự đoán `SalePrice`.

- File dữ liệu: `data/train-house-prices-advanced-regression-techniques.csv`.
- Dữ liệu gốc: **1.460 dòng, 81 cột**.
- `Id` là mã định danh, không được dùng làm feature trong hai notebook.
- `SalePrice` là mục tiêu `y`, không được nằm trong tập đầu vào `X`.
- Cả hai notebook bỏ `Id`, `Alley`, `PoolQC`, `Fence`, `MiscFeature`.
- Sau khi bỏ các cột trên và tách mục tiêu: **75 feature gốc**, gồm 36 cột số và 39 cột phân loại theo dtype.

Một số feature quan trọng:

| Feature | Ý nghĩa |
| --- | --- |
| `OverallQual` | Mức chất lượng tổng thể |
| `GrLivArea` | Diện tích sinh hoạt trên mặt đất |
| `GarageCars` | Sức chứa garage theo số xe |
| `GarageArea` | Diện tích garage |
| `TotalBsmtSF` | Tổng diện tích tầng hầm |
| `1stFlrSF` | Diện tích tầng 1 |
| `FullBath` | Số phòng tắm đầy đủ trên mặt đất |
| `TotRmsAbvGrd` | Số phòng trên mặt đất, không gồm phòng tắm |
| `YearBuilt` | Năm xây dựng |
| `YearRemodAdd` | Năm cải tạo/bổ sung |

**Dtype không hoàn toàn phản ánh ý nghĩa của biến.** Ví dụ `MSSubClass` là mã loại nhà dù được lưu bằng số; `OverallQual` là biến thứ bậc; `GarageCars` là biến đếm. Hai notebook hiện vẫn đưa các cột này vào nhóm numeric.

## 2. EDA: kiến thức và insight từ eda.ipynb

EDA là bước khám phá cấu trúc dữ liệu để đặt giả thuyết và chọn những thử nghiệm cần làm. Notebook này chưa huấn luyện mô hình và chưa điền missing, scale hoặc loại outlier.

### 2.1. Đọc dữ liệu và thống kê mô tả

```python
house_df.head()
house_df.describe().transpose()
```

- `head()` giúp kiểm tra tên cột, cách biểu diễn dữ liệu và một số giá trị đầu tiên; không đại diện cho toàn bộ phân phối.
- `describe()` mặc định thống kê các cột số; `.transpose()` đưa tên feature thành hàng để đọc dễ hơn.
- `count` là số quan sát không thiếu; so với 1.460 dòng để biết cột numeric có missing không.
- `mean`, `std` mô tả trung bình và độ phân tán.
- `25%`, `50%`, `75%` là Q1, median và Q3; `IQR = Q3 − Q1`.
- `min`, `max` giúp phát hiện miền giá trị rộng, số 0 hoặc điểm cần kiểm tra.
- Muốn xem thống kê cột phân loại, cần lựa chọn thêm nhóm cột đó; bảng hiện tại không tóm tắt đầy đủ các danh mục.

Ví dụ `LotFrontage` chỉ có 1.201 quan sát không thiếu; `GrLivArea` có median 1.464 nhưng max 5.642, gợi ý cần xem phân phối thay vì chỉ nhìn mean.

### 2.2. Phân phối SalePrice

Notebook vẽ histogram 40 bin, đường KDE và đường đứng tại mean.

| Thống kê | Giá trị trên dữ liệu gốc |
| --- | ---: |
| Mean | 180.921,20 |
| Median | 163.000 |
| Q1 / Q3 | 129.975 / 214.000 |
| Min / Max | 34.900 / 755.000 |
| Skewness | 1,883 |

Mean lớn hơn median và đuôi phải dài cho thấy một số căn giá cao kéo trung bình lên. Histogram tập trung nhiều ở vùng giá thấp đến trung bình; một số ít căn giá rất cao nằm ở đuôi phải.

Điều này gợi ý thử `log1p(SalePrice)`, nhưng **mục tiêu lệch phải không tự chứng minh model tuyến tính vi phạm giả định**. Với hồi quy tuyến tính, còn phải kiểm tra residual, quan hệ với feature và phương sai sai số. Log target là lựa chọn cần đánh giá, không phải bước bắt buộc chỉ vì histogram lệch.

KDE là đường ước lượng mật độ được làm trơn; các gợn trên đường này phụ thuộc cách làm trơn và không tự chứng minh dữ liệu có các nhóm nhà tách biệt.

### 2.3. Missing data by feature

```python
house_df.isnull().sum()
missing = house_df.isna().sum().loc[lambda counts: counts.gt(0)].sort_values()
```

`isnull()` và `isna()` tương đương trong trường hợp này. Tổng theo cột cho số ô thiếu; biểu đồ thanh ngang đặt số missing trên trục X, tên feature trên trục Y. Chỉ những cột có missing được hiển thị.

Với cách đọc CSV hiện tại: **19 cột có missing, tổng cộng 7.829 ô thiếu**. Một số nhóm đáng chú ý:

| Feature hoặc nhóm | Số thiếu mỗi cột | Tỷ lệ xấp xỉ |
| --- | ---: | ---: |
| `PoolQC` | 1.453 | 99,52% |
| `MiscFeature` | 1.406 | 96,30% |
| `Alley` | 1.369 | 93,77% |
| `Fence` | 1.179 | 80,75% |
| `MasVnrType` | 872 | 59,73% |
| `FireplaceQu` | 690 | 47,26% |
| `LotFrontage` | 259 | 17,74% |
| `GarageType`, `GarageYrBlt`, `GarageFinish`, `GarageQual`, `GarageCond` | 81 | 5,55% |
| `BsmtExposure`, `BsmtFinType2` | 38 | 2,60% |
| `BsmtQual`, `BsmtCond`, `BsmtFinType1` | 37 | 2,53% |
| `MasVnrArea` | 8 | 0,55% |
| `Electrical` | 1 | 0,07% |

Các nhóm garage/tầng hầm cùng thiếu ở nhiều cột gợi ý phải đối chiếu sự vắng mặt của tiện ích, không chỉ xử lý mỗi ô như lỗi thu thập dữ liệu độc lập.

Số missing phụ thuộc quy tắc đọc CSV. Cần phân biệt một nhãn như `None` mang nghĩa “không có” với giá trị thiếu thực sự; không suy ra nguyên nhân missing chỉ từ `isna()`.

Hai notebook mô hình đang bỏ một danh sách cột cố định, không thực hiện thuật toán tự động xóa mọi cột thiếu trên 50%. Ví dụ `MasVnrType` vẫn được giữ dù tỷ lệ `isna()` hiện tại vượt 50%.

### 2.4. Ma trận tương quan và chọn 10 feature

```python
correlation = house_df.select_dtypes(include="number").drop(columns="Id").corr()
```

Ma trận gồm **37 cột số**, bao gồm `SalePrice`, loại `Id`. Các cột categorical chưa encode không xuất hiện trong ma trận này. Pearson correlation được tính theo các cặp quan sát có đủ dữ liệu, nên cặp có missing có thể dùng ít mẫu hơn cặp khác.

Heatmap đặt `vmin=-1`, `vmax=1`, `center=0` để màu có cùng ý nghĩa trên toàn biểu đồ. `mask=np.triu(..., k=1)` ẩn nửa trên; ma trận đối xứng nên phần bị ẩn không mang thông tin mới. Đường chéo vẫn giữ giá trị 1.

Notebook chọn `head(11)` vì `SalePrice` có tương quan với chính nó bằng 1, rồi `.index[1:]` để bỏ mục tiêu và giữ 10 feature. `feature_chosen` là pandas Index chứa tên cột, không phải dữ liệu đã được scale hay một model.

| Feature được chọn | Tương quan với SalePrice |
| --- | ---: |
| `OverallQual` | 0,791 |
| `GrLivArea` | 0,709 |
| `GarageCars` | 0,640 |
| `GarageArea` | 0,623 |
| `TotalBsmtSF` | 0,614 |
| `1stFlrSF` | 0,606 |
| `FullBath` | 0,561 |
| `TotRmsAbvGrd` | 0,534 |
| `YearBuilt` | 0,523 |
| `YearRemodAdd` | 0,507 |

Danh sách này là **top tương quan dương** vì code sort theo giá trị, không theo trị tuyệt đối. Nó có thể bỏ qua feature tương quan âm mạnh hoặc feature chỉ có quan hệ phi tuyến.

`feature_chosen` chỉ được dùng để chọn cột vẽ trong EDA. Hai notebook mô hình hiện không giới hạn X vào đúng 10 cột đó; chúng vẫn dùng tập numeric và categorical đã xác định riêng.

### 2.5. Histogram và box plot của feature_chosen

- Histogram cho biết dữ liệu tập trung ở đâu, có lệch phân phối hay nhiều cụm không.
- Box plot biểu diễn median, hộp từ Q1 đến Q3 và các điểm vượt râu. Với cấu hình mặc định, râu vươn đến quan sát xa nhất còn nằm trong ngưỡng 1,5×IQR; không nhất thiết là min/max.
- Các biến diện tích như `GrLivArea`, `TotalBsmtSF`, `1stFlrSF` có đuôi phải dài: đa số nhà có diện tích vừa, một số căn rất lớn.
- Biến đếm như `GarageCars`, `FullBath`, `TotRmsAbvGrd` nên có bin đặt tại số nguyên khi xem từng giá trị.
- Với biến liên tục, quy tắc Freedman–Diaconis chọn độ rộng bin theo IQR và số mẫu, thay vì ép mọi feature dùng cùng 30 bin.

**Điểm nằm ngoài râu box plot không tự động là dữ liệu sai.** Nó có thể là căn nhà hiếm nhưng hợp lệ.

Trong code box plot, `melt(var_name="Feature", value_name="Value")` đổi dữ liệu từ dạng rộng sang dạng dài: một cột chỉ tên feature, một cột chứa giá trị. `col="Feature"` tạo biểu đồ riêng cho từng biến; `col_wrap=3` sắp ba biểu đồ mỗi hàng; `sharey=False` cho mỗi biến một thang Y riêng.

Histogram dùng điều kiện:

```python
discrete = values.nunique() <= 20 and values.mod(1).eq(0).all()
```

Đây là heuristic nhận diện cột có ít giá trị nguyên, không phải định nghĩa đầy đủ của biến rời rạc. Với cột thỏa điều kiện, `discrete=True` dùng bin rộng 1 và `kde=False`. Cột còn lại dùng `bins="fd"` và có KDE. Mỗi subplot tự có thang đo riêng, nên không so chiều cao/độ rộng hộp giữa các biến như thể chúng có cùng đơn vị.

### 2.6. Insight cụ thể của 10 feature

| Feature | Median | Skewness | Nhận xét từ histogram/box plot |
| --- | ---: | ---: | --- |
| `OverallQual` | 6 | 0,22 | Tập trung ở mức 5–7; các mức chất lượng rất thấp/cao ít mẫu |
| `GrLivArea` | 1.464 | 1,37 | Lệch phải rõ, có một số nhà diện tích sinh hoạt rất lớn |
| `GarageCars` | 2 | −0,34 | Chủ yếu 1–2 chỗ; garage 4 xe hiếm nhưng có thể hợp lệ |
| `GarageArea` | 480 | 0,18 | Khá cân đối tổng thể, có cụm bằng 0 và một số garage rất lớn |
| `TotalBsmtSF` | 991,5 | 1,52 | Lệch phải rõ; cần phân biệt không có tầng hầm với missing |
| `1stFlrSF` | 1.087 | 1,38 | Phần lớn diện tích vừa, một số giá trị lớn ở đuôi phải |
| `FullBath` | 2 | 0,04 | Chủ yếu 1–2 phòng tắm đầy đủ; là biến đếm |
| `TotRmsAbvGrd` | 6 | 0,68 | Chủ yếu 5–7 phòng, lệch phải nhẹ; nhà nhiều phòng ít phổ biến |
| `YearBuilt` | 1973 | −0,61 | Đuôi kéo về các năm xây dựng cũ; không phải biến lệch phải |
| `YearRemodAdd` | 1994 | −0,50 | Có cụm lớn ở năm 1950; cần xem định nghĩa trước khi diễn giải |

Diện tích tính theo đơn vị ft² của các cột SF. Skewness là thống kê mô tả phân phối, không đo trực tiếp sức mạnh dự đoán của một feature.

Trên toàn bộ dữ liệu, quy tắc 1,5×IQR đánh dấu 31 điểm ở `GrLivArea`, 61 ở `TotalBsmtSF` và 20 ở `1stFlrSF`. Đây là số điểm vượt ngưỡng tham khảo, **không phải số dòng phải xóa**, và các tập dòng có thể trùng nhau.

### 2.7. Tương quan giữa feature với nhau

Pearson correlation đo mức liên hệ tuyến tính, có giá trị từ −1 đến 1. Giá trị gần 0 không loại trừ quan hệ phi tuyến.

Một số cặp tương quan mạnh trong dữ liệu gốc:

| Cặp feature | Pearson r, xấp xỉ |
| --- | ---: |
| `GarageCars` – `GarageArea` | 0,882 |
| `YearBuilt` – `GarageYrBlt` | 0,826 |
| `GrLivArea` – `TotRmsAbvGrd` | 0,825 |
| `TotalBsmtSF` – `1stFlrSF` | 0,820 |

Các cặp trên có thể mang thông tin trùng lặp. Với mô hình tuyến tính, đa cộng tuyến có thể khiến hệ số thiếu ổn định. Tuy nhiên, không cần tự động xóa một feature chỉ vì tương quan cao.

Khi lọc tương quan giữa **các feature với nhau**, cần bỏ `SalePrice` ở cả hàng và cột; chỉ lấy một nửa ma trận và bỏ đường chéo để tránh cặp trùng và tương quan của cột với chính nó.

Trong code `pairs`, `k=1` giữ nửa trên nhưng loại đường chéo, khác với mask của heatmap dùng để ẩn nửa trên. `.stack()` gom các cặp hợp lệ thành Series có index hai tầng. Điều kiện `pairs.abs() >= 0.5` giữ liên hệ tuyến tính từ trung bình trở lên theo ngưỡng thực nghiệm; đây không phải kiểm định ý nghĩa thống kê. `.sort_values(key=abs, ascending=False).head(10)` chọn 10 cặp có độ lớn tương quan cao nhất.

### 2.8. Liên hệ từ EDA sang mô hình

| Quan sát EDA | Giả thuyết xử lý cần kiểm chứng |
| --- | --- |
| Giá và một số diện tích lệch phải | Thử log target/feature rồi đo metric trên validation |
| Numeric có missing và outlier | So sánh mean với median, không mặc định xóa điểm cực trị |
| Nhiều categorical với số nhóm khác nhau | One-hot, đồng thời theo dõi số chiều sau encode |
| Feature có thang đo rất khác nhau | Scale trước mô hình nhạy với thang đo |
| Diện tích, số phòng, sức chứa garage có liên hệ mạnh | Cân nhắc regularization và kiểm tra thông tin trùng lặp |
| Missing đi cùng sự vắng mặt tiện ích | Xem định nghĩa dữ liệu trước khi gán mean/median hoặc bỏ cột |

Nếu lựa chọn feature theo tương quan với mục tiêu để xây dựng mô hình, việc lựa chọn đó cũng phải học từ tập train, hoặc từ phần train của từng CV fold.

Notebook EDA hiện dùng toàn bộ dữ liệu để mô tả. Nếu dùng các quyết định có tham chiếu mục tiêu trên toàn bộ dữ liệu để liên tục tối ưu model, test không còn hoàn toàn độc lập. Khi làm thực nghiệm nghiêm ngặt, nên thực hiện các lựa chọn đó trên train và giữ test cho đánh giá cuối.

## 3. Chia train/test trước khi học thống kê

Hai notebook dùng:

```python
train_test_split(house_df, test_size=0.25, random_state=42)
```

- Train: **1.095 căn nhà**.
- Test: **365 căn nhà**.
- `random_state=42` giúp tái lập lần chia dữ liệu khi giữ nguyên dữ liệu đầu vào.

`fit` học tham số từ dữ liệu; `transform` áp dụng tham số đã học. `fit_transform` thực hiện cả hai.

```python
train_values = imputer.fit_transform(train_values)
test_values = imputer.transform(test_values)
```

Chỉ train được dùng để học mean, median, min/max, độ lệch chuẩn, danh mục one-hot hoặc ngưỡng xử lý ngoại lai. Không fit riêng trên test và không fit trên toàn bộ dữ liệu trước khi chia; làm như vậy sẽ gây **data leakage**.

## 4. Xử lý missing

### Mean trong main

```python
imputer = SimpleImputer()
```

Mặc định điền trung bình của từng cột số. Cách này đơn giản nhưng mean dễ bị kéo bởi các giá trị cực lớn.

### Median trong advanced

```python
imputer = SimpleImputer(strategy="median")
```

Điền trung vị của từng cột số, ít bị outlier tác động hơn mean. Median không phục hồi được quan hệ giữa các biến, nhưng dễ giải thích và ít chi phí tính toán.

Cột phân loại được điền nhãn `"none"` trước khi encode. Đây là một danh mục để mô hình xử lý, không phải số 0.

### Không nhầm missing với số 0

Trong dữ liệu gốc có 81 căn nhà có `GarageCars = GarageArea = 0` và 37 căn có `TotalBsmtSF = 0`. Số 0 có thể biểu diễn việc không có garage/tầng hầm. Không nên thay mọi số 0 bằng missing hoặc tự động loại các dòng đó.

Các cột thiếu nhiều như `PoolQC` hoặc `Alley` cũng có thể chứa thông tin về sự vắng mặt của tiện ích. Việc bỏ chúng trong project là một lựa chọn đơn giản hóa, không phải quy tắc “thiếu trên 50% thì luôn phải bỏ”.

### Những phương án đã cân nhắc nhưng không còn trong bản advanced đơn giản

| Phương án | Ý tưởng | Đánh đổi |
| --- | --- | --- |
| KNNImputer | Điền dựa trên các mẫu gần nhau ở những feature có quan sát | Nhạy với thang đo, tốn tính toán hơn median; không phải lúc nào cũng tốt hơn |
| IterativeImputer | Lần lượt dự đoán cột thiếu từ các cột còn lại | Phức tạp hơn, có thể chưa hội tụ hoặc tạo giá trị ngoài miền hợp lý |
| IQR clipping | Giới hạn giá trị theo `Q1 − 1,5×IQR` và `Q3 + 1,5×IQR` | Có thể làm mất thông tin của nhà lớn hợp lệ |
| Winsorization | Giới hạn giá trị ở các phân vị chọn trước, ví dụ 1% và 99% | Cần chọn ngưỡng, vẫn thay đổi dữ liệu cực trị |

Clipping không làm phân phối tự động trở thành phân phối chuẩn. Các phương án trên cần học tham số từ train và được kiểm chứng bằng validation/CV. Chúng đã được gỡ khỏi bản đơn giản để giảm độ phức tạp của code.

## 5. Scale feature: MinMaxScaler và StandardScaler

### MinMaxScaler trong main

```text
x_scaled = (x − min_train) / (max_train − min_train)
```

Thông thường đưa giá trị trên train về [0, 1] với cấu hình mặc định. Giá trị test nằm ngoài min/max của train vẫn có thể cho kết quả ngoài [0, 1]. Outlier có thể làm phần lớn quan sát bị dồn vào một khoảng hẹp.

### StandardScaler trong advanced

```text
x_scaled = (x − mean_train) / std_train
```

Đưa trung bình mỗi cột train về gần 0, độ lệch chuẩn về gần 1 nếu cột không hằng. Giúp các feature có thang đo tương đương, hữu ích cho Ridge, Lasso, MLP và các phương pháp dựa trên khoảng cách.

**Scale không đồng nghĩa với sửa phân phối.** StandardScaler vẫn giữ độ lệch và ảnh hưởng tương đối của các điểm cực đoan. RobustScaler dùng median/IQR để giảm độ nhạy khi scale, nhưng cũng không xóa outlier.

Các mô hình cây thường không cần scale. Với mô hình tuyến tính có regularization, đổi scaler cũng thay đổi ý nghĩa thực tế của mức phạt lên hệ số.

## 6. One-hot encoding

One-hot biến một cột danh mục thành các cột chỉ báo 0/1. Nó tránh áp đặt thứ tự giả như `A < B < C` cho danh mục không có thứ bậc.

```python
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

- `handle_unknown="ignore"`: khi gặp danh mục chưa có trong train, nhóm cột tương ứng nhận toàn 0.
- `sparse_output=False`: trả về mảng dense thay vì sparse matrix.

Advanced có thêm `drop="first"` để bỏ danh mục tham chiếu:

| Danh mục gốc | B | C |
| --- | ---: | ---: |
| A | 0 | 0 |
| B | 1 | 0 |
| C | 0 | 1 |

Việc này tránh phụ thuộc tuyến tính chính xác giữa toàn bộ dummy và intercept. Với Ridge/Lasso, bỏ một dummy không bắt buộc và có thể ảnh hưởng cách regularization đối xử với các nhóm.

Khi dùng `drop="first"` cùng `handle_unknown="ignore"`, danh mục lạ và nhóm tham chiếu đều có thể được biểu diễn bằng toàn 0 trong nhóm dummy đó. Cảnh báo unknown category không nhất thiết là lỗi chạy, nhưng cần hiểu giới hạn biểu diễn này.

### Vì sao main không drop cột categorical gốc?

Main thêm one-hot vào DataFrame nhưng chỉ ghép các cột numeric đã scale với `encoded_cols` để tạo `X_train`. Các cột chữ gốc không được chọn vào mô hình nên không bắt buộc phải xóa khỏi DataFrame.

Advanced giữ numeric/categorical trong các mảng riêng rồi ghép chúng; dữ liệu gốc của `train_df` không bị điền missing/encode trực tiếp.

## 7. Các biến dữ liệu trong code có định dạng gì?

| Biến | Định dạng trong project | Vai trò |
| --- | --- | --- |
| `train_df`, `test_df` | pandas DataFrame | Các feature có tên cột và index |
| `num_cols`, `cat_cols`, `encoded_cols` | Danh sách tên cột | Chọn đúng nhóm feature |
| `train_num_features` | NumPy ndarray 2 chiều | Numeric sau imputation/biến đổi/scale |
| `train_cat_features` ở advanced | NumPy ndarray 2 chiều | One-hot categorical |
| `X_train`, `X_test` | NumPy ndarray 2 chiều | Ma trận đầu vào mô hình |
| `y_train`, `y_test` | NumPy ndarray 1 chiều | Giá gốc cần dự đoán |

Mảng numeric không giữ tên cột như DataFrame. Thứ tự cột theo `num_cols`; vì vậy advanced lấy `log_indices = [num_cols.index(col) for col in log_cols]` trước khi log các vị trí tương ứng.

Với lần chia hiện tại:

- Main: **282 feature** sau encode toàn bộ danh mục.
- Advanced: **243 feature** sau encode có `drop="first"`.

`np.hstack([...])` và `np.concatenate([...], axis=1)` đều ghép theo chiều cột khi các mảng đầu vào đều 2 chiều và có cùng số dòng.

## 8. Log feature và log mục tiêu là hai việc khác nhau

### Tại sao dùng log1p?

```text
log1p(x) = ln(1 + x)
```

Nó nén giá trị lớn mạnh hơn giá trị nhỏ, giúp rút ngắn đuôi phải. Ví dụ 500, 1.000, 5.000 lần lượt thành khoảng 6,22; 6,91; 8,52.

`log1p(0) = 0`, nên dùng được với diện tích garage/tầng hầm bằng 0. Trong project, các đại lượng được log đều không âm. Log không tự loại outlier và không đảm bảo mọi phân phối trở thành chuẩn.

### Sáu feature đang được log ở advanced

```python
log_cols = [
    "GrLivArea", "GarageArea", "TotalBsmtSF",
    "1stFlrSF", "TotRmsAbvGrd", "YearBuilt",
]
```

Thứ tự thực hiện:

```text
Numeric gốc → median imputation → log1p 6 cột → StandardScaler
```

Train và test dùng cùng danh sách cột. Log được thực hiện trước scale để tránh log lên các giá trị âm do chuẩn hóa tạo ra.

`YearBuilt` được log trực tiếp theo lựa chọn thực nghiệm. Log năm không đồng nghĩa với tuổi nhà và không nhất thiết có diễn giải tốt. Một phương án khác để kiểm chứng riêng là tạo `YrSold − YearBuilt`; notebook hiện chưa làm bước này.

### Log mục tiêu qua TransformedTargetRegressor

```python
model = TransformedTargetRegressor(
    regressor=estimator,
    func=np.log1p,
    inverse_func=np.expm1,
)
```

- Khi fit: biến đổi giá thành `log1p(SalePrice)` rồi train regressor.
- Khi predict: dùng `expm1` để đổi dự đoán về giá gốc.
- `y_train` và `y_test` bên ngoài wrapper vẫn là giá gốc.
- RMSE và R² trong bảng được tính sau khi đã đổi ngược.

Mô hình học lỗi trên thang log; vì vậy có thể ưu tiên sai số tương đối hơn so với học trực tiếp giá gốc, nhưng không đảm bảo RMSE giá gốc giảm. Ngoài ra, đổi ngược kỳ vọng trên thang log không tự tạo ra kỳ vọng trên thang giá gốc; notebook không thực hiện hiệu chỉnh bias cho phép đổi ngược này.

### MLP scale mục tiêu như thế nào?

| Phiên bản | Mục tiêu được MLP học |
| --- | --- |
| Main | StandardScaler(SalePrice) |
| Advanced | StandardScaler(log1p(SalePrice)) |

Advanced lồng hai wrapper: wrapper ngoài log giá, wrapper trong chuẩn hóa log giá. Khi predict, nó đảo hai bước theo thứ tự ngược lại. Đây là scale **y**, không phải scale lại feature X.

### Box–Cox và Yeo–Johnson

- Box–Cox yêu cầu giá trị đầu vào dương nghiêm ngặt.
- Yeo–Johnson xử lý được số 0 và số âm.
- Các phép biến đổi này học tham số nên phải fit trên train/từng CV fold.
- Chúng đã được cân nhắc trong quá trình thử nghiệm, nhưng không nằm trong bản advanced hiện tại.

## 9. Các mô hình đã dùng

| Mô hình | Kiến thức chính | Điểm cần lưu ý |
| --- | --- | --- |
| Linear Regression | Học tổ hợp tuyến tính của feature bằng bình phương tối thiểu | Nhạy với đa cộng tuyến, điểm ảnh hưởng lớn và số feature quá nhiều |
| Ridge | Thêm phạt L2 để thu nhỏ hệ số | Thường giữ hệ số khác 0; alpha lớn hơn tăng regularization |
| Lasso | Thêm phạt L1, có thể đưa hệ số về 0 | Có tác dụng chọn feature; nhạy với scale và tương quan giữa các feature |
| HuberRegressor | Sai số nhỏ dùng dạng bình phương, sai số lớn dùng dạng tuyến tính | Giảm mức tác động của residual lớn nhưng không miễn nhiễm với điểm leverage lớn ở X |
| MLPRegressor | Mạng neural feedforward học quan hệ phi tuyến | Cần chú ý scale, kiến trúc, regularization, hội tụ và overfitting |

Không so trực tiếp giá trị alpha giữa Ridge, Lasso, Huber và MLP: dạng penalty và hàm mục tiêu của chúng khác nhau. Ngay trong cùng Lasso, alpha khi học giá gốc cũng không tương đương alpha khi học log giá.

### Cấu hình hiện tại

| Thành phần | Main | Advanced |
| --- | --- | --- |
| Numeric imputation | Mean | Median |
| Numeric scale | MinMaxScaler | StandardScaler |
| Log feature | Không | Sáu cột liệt kê ở trên |
| One-hot | Giữ tất cả dummy | `drop="first"` |
| Linear/Ridge/Lasso/Huber target | Giá gốc | Log giá |
| Ridge alpha | Mặc định 1 | 1 |
| Lasso alpha | Mặc định 1 | 0,001 |
| Huber | Tham số mặc định | alpha=0,01; max_iter=5.000; tol=0,0001 |
| MLP hidden layers | (64, 32) | (32, 16) |
| MLP alpha | 0,01 | 1 |
| MLP target | Chuẩn hóa giá gốc | Chuẩn hóa log giá |

Vì nhiều thành phần đổi cùng lúc, chênh lệch kết quả giữa hai notebook không thể quy cho riêng median, StandardScaler hoặc log target.

## 10. Đọc RMSE, R², overfitting và underfitting

```text
RMSE = sqrt(mean((y − y_pred)²))
R²   = 1 − sum((y − y_pred)²) / sum((y − mean(y))²)
```

- RMSE càng thấp càng tốt; có cùng đơn vị với giá và nhạy với sai số lớn.
- RMSE 24.000 không có nghĩa mọi dự đoán lệch đúng 24.000 hoặc MAE bằng 24.000.
- R² càng cao càng tốt trên cùng tập đánh giá; có thể âm, không phải phần trăm dự đoán đúng.
- R² âm nghĩa tổng bình phương sai số vượt baseline dự đoán trung bình của chính tập đánh giá theo định nghĩa chỉ số.

**Overfitting:** train tốt nhưng test kém đáng kể. Main có MLP train R² khoảng 0,989 nhưng test chỉ khoảng 0,855: dấu hiệu mô hình học quá sát train.

**Underfitting:** cả train và test đều chưa tốt. Tăng regularization quá mạnh có thể làm mất cả khả năng học train.

Khoảng cách train/test nhỏ chưa đủ chứng minh model tốt: một model dự đoán kém ở cả hai tập vẫn có thể có khoảng cách nhỏ.

## 11. GridSearchCV: điều đã học từ thử nghiệm

GridSearch thử các tổ hợp hyperparameter và chọn theo điểm trung bình trên CV. Nó không đảm bảo model tốt hơn trên một tập test cụ thể.

Khi dùng CV, cấu trúc đúng là:

```text
Raw X của phần train trong fold
  → imputer → scaler/encoder → model
Đánh giá trên phần validation của fold
```

Pipeline nối các bước theo thứ tự; ColumnTransformer xử lý các nhóm cột khác nhau. Đưa X đã được fit preprocessing từ toàn bộ train vào CV vẫn làm lộ thống kê giữa các fold.

`model__regressor__alpha` là đường dẫn tham số: bước `model` → regressor bên trong TransformedTargetRegressor → alpha của Lasso.

`neg_root_mean_squared_error` trả RMSE âm vì GridSearch chọn điểm lớn nhất. Đổi dấu `best_score_` để đọc RMSE dương. Mặc định `refit=True`, model tốt nhất được fit lại trên toàn bộ tập train sau khi chọn tham số.

Trong một thử nghiệm trước đây, GridSearch Lasso trên log giá chọn alpha=0,1. Train R² giảm còn khoảng 0,629 và test R² khoảng 0,652, kém cấu hình ban đầu. Mức phạt mạnh gây underfitting trên lần chia đó. Để giải thích vì sao CV ưu tiên cấu hình này, cần xem điểm từng fold và độ biến động của `cv_results_`; không được mặc định kết luận nguyên nhân là outlier khi chưa kiểm tra.

Thử nghiệm GridSearch này đã được người dùng xóa khỏi notebook advanced. Không chạy lại hoặc tự bổ sung nó trong lần tổng hợp tài liệu này.

### Nguyên tắc chọn cấu hình

1. Dùng CV trên train để chọn preprocessing/hyperparameter.
2. So sánh các phương án bằng cùng một bộ fold.
3. Xem cả trung bình và mức dao động của metric.
4. Dùng test cho đánh giá cuối cùng, tránh điều chỉnh liên tục chỉ để tăng điểm test.
5. Khi đã thử nhiều phương án dựa trên test, kết quả test có thể lạc quan do quá trình chọn lựa; nên có đánh giá độc lập bổ sung khi cần kết luận cuối cùng.

## 12. PolynomialFeatures và vì sao thử nghiệm hiện tại chưa tốt

### Hai cách tạo polynomial trong project

Với hai biến x₁, x₂:

```text
degree=2, interaction_only=True  → x₁, x₂, x₁x₂
degree=2, interaction_only=False → x₁, x₂, x₁², x₁x₂, x₂²
```

`include_bias=False` không thêm cột hằng 1; các regressor đang dùng vốn đã có intercept.

- Main: polynomial bậc 2 với `interaction_only=True` trên numeric đã impute, sau đó MinMaxScaler. Cell đã được hoàn thiện code nhưng **chưa có output training đang lưu**.
- Advanced, cell thực thi [18]: polynomial bậc 2 đầy đủ trên numeric đã log/scale, rồi StandardScaler lại các cột polynomial; one-hot giữ nguyên.

Scale trước khi nhân giúp các cột gốc ở thang tương đương. Scale sau khi sinh polynomial giúp các cột tích/bình phương mới có thang đo tương đương. Hai bước này phục vụ hai giai đoạn khác nhau.

Với 36 cột số, polynomial đầy đủ bậc 2 không có bias tạo:

```text
36 + 36×37/2 = 702 cột numeric
702 + 207 cột one-hot = 909 feature
```

Trước polynomial, advanced chỉ có 243 feature. Số mẫu train vẫn là 1.095, nên việc tăng mạnh số feature làm mô hình dễ thiếu ổn định và tăng variance. Các biến đếm/nhị phân cũng có thể sinh cột trùng hoặc phụ thuộc.

Trong code, `clone(baseline)` tạo model mới có cùng cấu hình nhưng không mang trạng thái đã fit. Nhờ đó giữ nguyên target transformation/hyperparameter để thử riêng bộ feature mới, không ghi đè các model baseline đã train.

Nếu mô hình dự đoán quá lớn trên thang log, `expm1` khuếch đại rất mạnh khi đổi về giá gốc. Đây là một cơ chế có thể làm RMSE tăng cực lớn; không nên cắt dự đoán tùy ý chỉ để làm đẹp metric.

## 13. Kết quả thực nghiệm

### 13.1. Main: numeric gốc, mean, MinMaxScaler

Nguồn: bảng output của cell thực thi [92] đang lưu trong main.

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 20.029,220 | 28.137,140 | 0,934 | 0,887 |
| Ridge | 22.488,287 | 28.586,776 | 0,917 | 0,883 |
| Lasso | 20.032,669 | **26.362,510** | 0,934 | **0,901** |
| Huber | 27.704,432 | 27.890,849 | 0,874 | 0,889 |
| MLP | 8.339,651 | 31.820,560 | 0,989 | 0,855 |

Lasso tốt nhất theo test RMSE trong bảng này. MLP fit train rất sát nhưng tổng quát hóa kém hơn các mô hình tuyến tính.

### 13.2. Advanced hiện tại: thêm log sáu feature và log target

Nguồn: output cell thực thi [17].

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 17.543,323 | 24.049,104 | 0,949 | 0,917 |
| Ridge | 20.409,390 | **23.948,705** | 0,931 | **0,918** |
| Lasso | 26.408,023 | 25.862,909 | 0,885 | 0,905 |
| Huber | 21.834,157 | 24.053,405 | 0,921 | 0,917 |
| MLP | 15.396,514 | 29.110,860 | 0,961 | 0,879 |

Ridge tốt nhất theo test RMSE trong phiên bản này. So với Lasso tốt nhất ở main, RMSE thấp hơn khoảng **9,2%**, nhưng đây là kết quả của nhiều thay đổi đồng thời, không phải hiệu quả riêng của log feature.

### 13.3. Riêng tác động thực nghiệm của việc thêm log sáu feature

Các số “trước” bên dưới là kết quả đã ghi nhận trước lần thêm log feature; không phải bảng output riêng còn tồn tại trong notebook hiện tại. Khi đó advanced đã dùng median, StandardScaler và log mục tiêu.

| Model | Test RMSE trước log feature | Sau log feature | Nhận xét |
| --- | ---: | ---: | --- |
| Linear Regression | 24.523,589 | 24.049,104 | Giảm khoảng 1,9% |
| Ridge | 24.222,186 | 23.948,705 | Giảm khoảng 1,1% |
| Lasso | 25.408,315 | 25.862,909 | Tăng khoảng 1,8% |
| Huber | 24.161,212 | 24.053,405 | Giảm khoảng 0,4% |
| MLP | 29.002,503 | 29.110,860 | Tăng khoảng 0,4% |

So model tốt nhất mỗi phiên bản, RMSE giảm từ 24.161 xuống 23.949, khoảng **0,9%**. Cải thiện nhỏ, không đồng đều; chưa biết feature nào tạo lợi ích vì cả sáu được log cùng lúc.

### 13.4. Advanced + polynomial bậc 2 đầy đủ

Nguồn: output cell thực thi [18]; giá trị dưới đây làm tròn.

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 8.336 | 18.347.450 | 0,989 | −48.052,395 |
| Ridge | 9.503 | 41.604 | 0,985 | 0,753 |
| Lasso | 15.283 | 26.568 | 0,962 | 0,899 |
| Huber | 12.954 | 24.001.060 | 0,972 | −82.229,601 |
| MLP | 25.538 | 78.790 | 0,893 | 0,114 |

Linear, Ridge và Lasso có train tốt hơn nhưng test xấu đi. Linear/Huber cho dự đoán cực đoan; **Huber còn báo chưa hội tụ sau 5.000 vòng**, nên không xem đây là một nghiệm đã tối ưu ổn định. MLP cũng kém hơn ở cả hai tập với cấu hình đang thử.

Thử nghiệm không chứng minh polynomial luôn kém. Nó cho thấy **polynomial trên toàn bộ numeric với các cấu hình hiện tại không phù hợp**. Hướng thử hợp lý hơn là chỉ sinh một số tương tác có ý nghĩa và chọn regularization bằng CV.

## 14. Trade-off và cách làm thực nghiệm dễ đánh giá

Code tiền xử lý phức tạp hơn đem lại nhiều lựa chọn nhưng cũng tăng chi phí đọc hiểu, debug, kiểm tra leakage và tái lập kết quả. Một pipeline ngắn, có cơ sở và được đánh giá đúng thường hữu ích hơn việc thêm nhiều kỹ thuật cùng lúc.

Đối với project này:

- Median/StandardScaler/one-hot/log target là một baseline advanced tương đối gọn.
- Log thêm sáu feature chỉ cải thiện nhẹ model tốt nhất.
- Polynomial đầy đủ làm số feature tăng mạnh và kết quả test giảm.
- MLP chưa thể hiện lợi thế rõ so với Ridge/Lasso/Huber trên lần chia hiện tại.
- Đừng giữ một bước chỉ vì tên kỹ thuật nghe “advanced”; cần xét lợi ích trên validation, độ ổn định và chi phí triển khai code.

Một chuỗi thử nghiệm dễ diễn giải:

1. Cố định split hoặc CV folds và metric.
2. Xây baseline gọn.
3. Thay mean bằng median, đánh giá riêng.
4. Thay scaler, đánh giá riêng.
5. Log mục tiêu, chọn lại regularization phù hợp với thang mục tiêu mới.
6. Thử log từng feature/nhóm feature, không kết luận từ việc đổi tất cả cùng lúc.
7. Chỉ thêm tương tác có lý do và kiểm chứng bằng CV.
8. Ghi cả code cấu hình, metric train/validation, cảnh báo hội tụ và kết luận.

## 15. Những lưu ý khi chạy lại notebook

- Số trong `In [18]` là số lần thực thi trong kernel, không phải vị trí cell thứ 18 trong file.
- Chạy từ trên xuống để tránh biến cũ trong kernel hoặc `df_results` bị ghi đè bởi thử nghiệm khác.
- Code đổi nhưng chưa chạy lại thì output lưu trong file có thể không phản ánh code mới.
- Giữ nguyên thứ tự cột khi ghép numeric và categorical cho train/test.
- Fit scaler/encoder/imputer trên train; test chỉ transform.
- Nếu đưa preprocessing vào CV, phải đặt nó trong Pipeline của CV.
- Không tắt toàn bộ warning khi cần đánh giá hội tụ: `ConvergenceWarning` ảnh hưởng cách diễn giải kết quả.
- Chưa hội tụ không đồng nghĩa chắc chắn model vô dụng, nhưng cần xử lý hoặc ghi nhận trước khi kết luận.
- Không đổi hyperparameter chỉ để theo đuổi một tập test đã được xem nhiều lần.
- Các bảng ở tài liệu này là kết quả thực nghiệm trên một bộ dữ liệu và một lần chia cụ thể, không phải bảo đảm cho dữ liệu mới.
