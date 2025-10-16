<div align="center">

# **BÁO CÁO BÀI TẬP CÁ NHÂN**  
### MÔN TRÍ TUỆ NHÂN TẠO  
**Đợt 1 - HKI**  
**Năm học 2025 - 2026**  

**TRÒ CHƠI ĐẶT 8 QUÂN XE LÊN BÀN CỜ VUA**  

---

**GV hướng dẫn:** TS. Phan Thị Huyền Trang  
**Lớp học phần:**  ARIN330585-05CLC

**Sinh viên:** Phan Thị Thanh Trà  
**MSSV:** 23110159  

</div>

## 1. Các thư viện sử dụng trong chương trình

| Thư viện | Công dụng |
|-----------|------------|
| `import tkinter as tk` | Tạo giao diện người dùng (cửa sổ chính, khung, nút bấm, canvas để vẽ bàn cờ). |
| `from tkinter import ttk` | Cung cấp các widget nâng cao như Button, Combobox, ProgressBar, Treeview,… |
| `import tkinter.font as tkfont` | Chỉnh font chữ cho các thành phần như Label, Button, Text. |
| `import random` | Sinh giá trị ngẫu nhiên, dùng để tạo bản đồ hoặc dữ liệu thử nghiệm. |
| `from collections import deque` | Cấu trúc hàng đợi (queue) – thường dùng trong BFS và các thuật toán duyệt. |
| `import heapq` | Tạo hàng đợi ưu tiên (priority queue) – thường dùng trong A*, UCS. |
| `import math` | Thực hiện các phép tính toán học (ví dụ: tính heuristics, khoảng cách, cost). |
| `from tkinter import messagebox` | Hiển thị hộp thoại thông báo, cảnh báo, xác nhận cho người dùng. |
| `from PIL import Image, ImageTk` | Xử lý và hiển thị hình ảnh trên giao diện Tkinter.<br/>PIL (Pillow) giúp mở, chỉnh sửa và chuyển đổi ảnh sang định dạng mà Tkinter có thể hiển thị. |
| `import pygame` | Tạo và điều khiển âm thanh (nhạc nền, hiệu ứng khi người chơi di chuyển). |
| `import time` | Đo thời gian thực thi của thuật toán để đánh giá hiệu năng. |
| `from datetime import datetime` | Lấy ngày giờ hiện tại để lưu lịch sử chạy và ghi log. |
| `from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg` | Nhúng biểu đồ Matplotlib trực tiếp vào giao diện Tkinter. |
| `import matplotlib.pyplot as plt` | Vẽ biểu đồ, đồ thị thể hiện hiệu năng hoặc đường đi của các thuật toán. |

---

## 2. Các thuật toán và demo kết quả  
### 2.1. DFS (Depth First Search)  
DFS là thuật toán tìm kiếm theo chiều sâu, dùng stack (LIFO).  
```python
def dfs_timkiem(dich):
    # Lấy danh sách cột đích tương ứng theo thứ tự hàng (0..7)
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # Khởi tạo ngăn xếp (stack) cho DFS
    # Mỗi phần tử trong stack gồm: (trạng thái hiện tại, tập các cột đã dùng)
    # - trạng thái (list): danh sách vị trí quân [(hàng, cột)]
    # - used (set): các cột đã có quân xe
    stack = [([], set())]

    # Danh sách các bước duyệt (để lưu lại quá trình tìm kiếm)
    buoc = []

    # Vòng lặp chính của DFS: lặp cho đến khi stack rỗng
    while stack:
        # Lấy phần tử ở đỉnh stack (LIFO) để duyệt sâu nhất trước
        trangthai, used = stack.pop()

        # Lưu lại trạng thái hiện tại (dùng .copy() để tránh ảnh hưởng khi thay đổi)
        buoc.append(trangthai.copy())

        # Xác định hàng đang xét = số quân đã đặt
        r = len(trangthai)

        # Nếu đã đặt đủ 8 quân (đến hàng cuối cùng)
        if r == SO_HANG:
            # Kiểm tra xem các quân có đúng vị trí đích không
            if all(trangthai[i][1] == target_cols[i] for i in range(SO_HANG)):
                # Nếu đạt trạng thái đích → trả về danh sách các bước đã duyệt
                return buoc
        else:
            # Lấy cột đích cần đạt ở hàng hiện tại để ưu tiên đặt trước
            cot_uu_tien = target_cols[r]

            # Tạo danh sách các cột có thể đặt quân ở hàng này
            candidates = []

            # Nếu cột đích chưa bị chiếm, thêm vào danh sách ưu tiên
            if cot_uu_tien not in used:
                candidates.append(cot_uu_tien)

            # Thêm các cột khác chưa bị chiếm (tránh trùng cột)
            for c in range(SO_HANG):
                if c != cot_uu_tien and c not in used:
                    candidates.append(c)

            # Thêm các trạng thái con vào stack theo thứ tự ngược lại
            # (để cột ưu tiên được pop ra trước → duyệt trước)
            for c in reversed(candidates):
                # Tạo trạng thái mới: thêm 1 quân vào hàng r, cột c
                # Cập nhật tập cột đã dùng (used ∪ {c})
                stack.append((trangthai + [(r, c)], used | {c}))

    # Nếu duyệt hết mà không tìm thấy trạng thái đích → trả về toàn bộ các bước đã duyệt
    return buoc
```
![dfs](https://github.com/user-attachments/assets/dadb418e-453d-416e-96df-4e46b16b6d3f)


### 2.2. BFS  
BFS tìm kiếm theo chiều rộng, dùng queue (FIFO).  
```python
def bfs_timkiem(dich):
    # Lấy danh sách cột đích theo thứ tự hàng (0..7)
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # Khởi tạo hàng đợi BFS: mỗi phần tử gồm (trạng thái, tập cột đã dùng)
    # - trạng thái: danh sách các quân xe đã đặt [(r, c), ...]
    # - used: tập hợp các cột đã bị chiếm
    queue = deque([([], set())])

    # Bắt đầu duyệt hàng đợi
    while queue:
        # Lấy phần tử đầu tiên trong hàng đợi (FIFO)
        state, used = queue.popleft()

        # Số hàng đã được đặt quân = độ dài của trạng thái
        r = len(state)

        # Nếu đã đặt đủ 8 quân, kiểm tra xem có trùng đích không
        if r == SO_HANG:
            # So sánh cột của từng hàng với cột đích tương ứng
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                # Nếu trùng, trả về danh sách các bước để hiển thị
                return [state[:k] for k in range(0, SO_HANG + 1)]
            continue  # nếu chưa trùng, bỏ qua và xét trạng thái khác

        # Danh sách các cột có thể đặt ở hàng hiện tại
        next_cols = []

        # Ưu tiên thử cột đích trước nếu chưa bị chiếm
        if target_cols[r] not in used:
            next_cols.append(target_cols[r])

        # Sau đó thêm các cột còn lại chưa bị chiếm
        for c in range(SO_HANG):
            if c != target_cols[r] and c not in used:
                next_cols.append(c)

        # Tạo các trạng thái con và thêm vào hàng đợi (FIFO)
        for c in next_cols:
            # state + [(r, c)] → thêm quân mới vào hàng r, cột c
            # used | {c} → đánh dấu cột đó đã được dùng
            queue.append((state + [(r, c)], used | {c}))

    # Nếu không tìm thấy trạng thái trùng với đích, trả về rỗng
    return []
```
![bfs](https://github.com/user-attachments/assets/1f1c2c58-8506-493a-870d-7bf4a161b536)

### 2.3. UCS
Thuật toán Uniform Cost Search (UCS) là phiên bản tổng quát của BFS cho các bài toán có chi phí di chuyển khác nhau. 
UCS sử dụng hàng đợi ưu tiên (priority queue) để mở rộng trạng thái có chi phí thấp nhất trước.
Trong bài toán 8 quân xe, chi phí mỗi bước được tính theo khoảng cách giữa cột hiện tại và cột đích.  
```python
# Chi phí mỗi bước = |c - target_cols[r]| (khoảng cách giữa vị trí hiện tại và cột đích)
import heapq  # dùng để tạo hàng đợi ưu tiên (priority queue)

def ucs_timkiem(dich):
    # Lấy danh sách cột đích theo thứ tự hàng 0..7
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # Hàng đợi ưu tiên (priority queue)
    # Mỗi phần tử: (tổng chi phí, trạng thái hiện tại, tập cột đã dùng)
    frontier = [(0, [], set())]  # bắt đầu từ trạng thái rỗng với chi phí = 0

    # Lặp cho đến khi hàng đợi rỗng
    while frontier:
        # Lấy phần tử có chi phí nhỏ nhất ra khỏi hàng đợi
        cost, state, used = heapq.heappop(frontier)

        # Hàng hiện tại cần đặt quân
        r = len(state)

        # Nếu đã đặt đủ 8 quân → kiểm tra trạng thái đích
        if r == SO_HANG:
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                # Tạo danh sách các bước để hiển thị dần trên giao diện
                buoc = [state[:k] for k in range(0, SO_HANG + 1)]
                return buoc
            continue  # nếu không khớp đích → bỏ qua và xét trạng thái khác

        # Duyệt qua tất cả các cột có thể đặt ở hàng hiện tại
        for c in range(SO_HANG):
            if c in used:
                continue  # bỏ qua cột đã bị chiếm

            # Chi phí bước hiện tại = khoảng cách giữa vị trí cột chọn và cột đích
            step_cost = abs(c - target_cols[r])

            # Tạo trạng thái mới bằng cách thêm quân vào (r, c)
            new_state = state + [(r, c)]
            new_used = used | {c}

            # Thêm trạng thái mới vào hàng đợi ưu tiên
            # Tổng chi phí mới = chi phí trước đó + chi phí bước hiện tại
            heapq.heappush(frontier, (cost + step_cost, new_state, new_used))

    # Nếu không tìm được trạng thái đích → trả về rỗng
    return []
```
![UCS](https://github.com/user-attachments/assets/0d5d2d4a-ac8b-4fc1-8608-fa71e0a1e72e)


### 2.4. DLS
Thuật toán Depth-Limited Search là biến thể của DFS, trong đó việc tìm kiếm được giới hạn bởi một độ sâu xác định trước.
DLS giúp tránh việc tìm kiếm vô hạn trong không gian trạng thái lớn nhưng có thể bỏ sót lời giải nếu độ sâu giới hạn quá nhỏ.
Trong bài toán 8 quân xe, DLS hoạt động hiệu quả khi độ sâu giới hạn bằng 8, đảm bảo duyệt đủ để tìm nghiệm.
```python
def dls_timkiem(dich, limit=SO_HANG):
    # Lấy danh sách cột đích tương ứng với từng hàng (0..7)
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # Danh sách lưu lại quá trình duyệt để mô phỏng hoặc hiển thị
    buoc = []

    # Biến đặc biệt 'cutoff' để đánh dấu khi thuật toán dừng do đạt giới hạn độ sâu
    cutoff = object()

    def recursive_dls(state, used, depth):
        """
        Hàm đệ quy thực hiện tìm kiếm theo chiều sâu có giới hạn.
        state: danh sách các vị trí quân đã đặt [(r, c), ...]
        used: tập hợp các cột đã bị chiếm
        depth: độ sâu hiện tại (hàng đang xét)
        """
        # Ghi lại trạng thái hiện tại để hiển thị quá trình tìm kiếm
        buoc.append(state.copy())

        # Nếu đã đặt đủ 8 quân (độ sâu = số hàng)
        if depth == SO_HANG:
            # Kiểm tra xem trạng thái hiện tại có trùng với trạng thái đích không
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                return state  # Trả về trạng thái đích nếu tìm thấy
            else:
                return None  # Không trùng đích → quay lui

        # Nếu đạt tới giới hạn độ sâu mà chưa tới đích → cắt nhánh
        elif depth == limit:
            return cutoff

        # Nếu chưa đạt giới hạn → tiếp tục mở rộng các trạng thái con
        else:
            cutoff_occurred = False  # Cờ đánh dấu xem có nhánh nào bị cắt không

            # Duyệt tất cả các cột có thể đặt quân ở hàng hiện tại
            for c in range(SO_HANG):
                if c not in used:  # Chỉ chọn cột chưa bị chiếm
                    # Tạo trạng thái con mới (thêm quân vào hàng hiện tại)
                    child = state + [(depth, c)]

                    # Gọi đệ quy cho hàng tiếp theo
                    result = recursive_dls(child, used | {c}, depth + 1)

                    # Nếu gặp cắt nhánh, đánh dấu lại
                    if result is cutoff:
                        cutoff_occurred = True
                    # Nếu tìm thấy nghiệm → truyền ngược kết quả về
                    elif result is not None:
                        return result

            # Nếu có cắt nhánh → trả về cutoff; nếu không → None
            return cutoff if cutoff_occurred else None

    # Gọi hàm đệ quy từ trạng thái ban đầu (rỗng)
    result = recursive_dls([], set(), 0)

    # Nếu tìm được nghiệm (và không phải do cắt nhánh)
    if result is not None and result is not cutoff:
        # Trả về danh sách các bước trung gian để hiển thị
        return [result[:k] for k in range(0, SO_HANG + 1)]

    # Nếu không tìm thấy nghiệm, trả về toàn bộ quá trình duyệt
    return buoc
```
![dls](https://github.com/user-attachments/assets/28095be0-0000-4295-b8d2-7021a8fd527d)

### 2.5. IDS
Thuật toán Iterative Deepening Search (IDS) là biến thể của DFS có giới hạn độ sâu tăng dần.
Mỗi lần chạy DFS đến một độ sâu nhất định (DLS), sau đó tăng giới hạn và chạy lại.
Nhờ đó, IDS kết hợp ưu điểm của DFS (ít tốn bộ nhớ) và BFS (tìm được nghiệm gần nhất).
Trong bài toán 8 quân xe, IDS đảm bảo tìm được trạng thái đích mà không cần biết trước độ sâu của nghiệm.
```python
def ids_dfs_timkiem(dich):
    # Lấy danh sách cột đích tương ứng với từng hàng (0..7)
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    def dls_dfs(state, used, depth, limit, buoc):
        """
        Hàm đệ quy thực hiện tìm kiếm theo chiều sâu có giới hạn (DLS).
        state: trạng thái hiện tại (danh sách vị trí quân đã đặt)
        used: tập hợp các cột đã bị chiếm
        depth: độ sâu hiện tại
        limit: giới hạn độ sâu tối đa
        buoc: danh sách lưu lại các trạng thái duyệt để minh họa
        """
        # Lưu trạng thái hiện tại vào danh sách các bước
        buoc.append(state.copy())

        # Nếu đã đặt đủ 8 quân (độ sâu = số hàng của bàn cờ)
        if depth == SO_HANG:
            # Kiểm tra xem trạng thái có trùng với đích hay không
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                return state  # Trả về trạng thái đích nếu khớp
            return None  # Nếu không khớp → quay lui

        # Nếu đạt giới hạn độ sâu mà chưa tới đích → dừng mở rộng
        if depth == limit:
            return None

        # Duyệt tất cả các cột có thể đặt quân ở hàng hiện tại
        for c in range(SO_HANG):
            if c not in used:
                # Tạo trạng thái con mới (đặt thêm 1 quân vào hàng hiện tại)
                child = state + [(depth, c)]

                # Gọi đệ quy cho hàng kế tiếp
                result = dls_dfs(child, used | {c}, depth + 1, limit, buoc)

                # Nếu tìm thấy trạng thái đích → truyền ngược kết quả về
                if result is not None:
                    return result

        # Nếu duyệt hết mà không tìm thấy nghiệm → quay lui
        return None

    # Lặp qua các giới hạn độ sâu từ 1 → số hàng của bàn cờ (8)
    for limit in range(1, SO_HANG + 1):
        buoc = []  # Danh sách lưu các bước duyệt trong lần giới hạn này

        # Gọi tìm kiếm DFS có giới hạn với độ sâu hiện tại
        ket_qua = dls_dfs([], set(), 0, limit, buoc)

        # Nếu tìm thấy nghiệm → trả về danh sách các bước kết quả
        if ket_qua is not None:
            return [ket_qua[:k] for k in range(0, SO_HANG + 1)]

    # Nếu không tìm thấy nghiệm trong tất cả các giới hạn → trả về rỗng
    return []
```
![IDS_DFS](https://github.com/user-attachments/assets/88222060-2efb-48b0-908f-96894ac3ade7)


### 2.6. Greedy
Thuật toán Greedy là phương pháp tìm kiếm có thông tin, sử dụng hàm heuristic h(n) để ước lượng khoảng cách từ trạng thái hiện tại đến đích.
Ở mỗi bước, thuật toán chọn mở rộng trạng thái có giá trị h nhỏ nhất — tức là “ước lượng gần đích nhất”.
Tuy cho tốc độ tìm kiếm nhanh, Greedy Search không đảm bảo nghiệm tối ưu, vì có thể bỏ qua trạng thái có chi phí thực thấp hơn.
```python
 # ========= GREEDY BEST-FIRST SEARCH =========
def heuristic(state, goal_cols):
    """
    Hàm heuristic (h): ước lượng "độ xa" của trạng thái hiện tại so với trạng thái đích.
    Ở đây, sử dụng tổng khoảng cách tuyệt đối giữa cột hiện tại và cột đích của từng quân.
    """
    return sum(abs(col - goal_cols[row]) for row, col in state)


def greedy_timkiem(dich):
    """
        - Chọn mở rộng trạng thái có giá trị heuristic (h) nhỏ nhất.
        - Ưu tiên trạng thái "gần đích" nhất theo hàm ước lượng h(n).
    """

    # Lấy danh sách cột đích tương ứng với từng hàng (0..7)
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # Hàng đợi ưu tiên (priority queue) sắp xếp theo giá trị heuristic nhỏ nhất
    # Mỗi phần tử gồm: (giá trị h, trạng thái, tập các cột đã dùng)
    frontier = [(heuristic([], target_cols), [], set())]

    # Danh sách lưu lại các bước duyệt (dùng để minh họa quá trình tìm kiếm)
    buoc = []

    # Bắt đầu vòng lặp tìm kiếm
    while frontier:
        # Lấy trạng thái có giá trị heuristic nhỏ nhất (h = "ước lượng gần đích nhất")
        h, state, used = heapq.heappop(frontier)

        # Ghi nhận trạng thái hiện tại vào danh sách bước duyệt
        buoc.append(state)

        # Hàng hiện tại (depth) = số quân đã đặt
        r = len(state)

        # Nếu đã đặt đủ 8 quân → kiểm tra xem có đạt đích không
        if r == SO_HANG:
            # So sánh cột của từng hàng với cột đích tương ứng
            if [col for _, col in state] == target_cols:
                # Nếu trùng → trả về danh sách các bước từ đầu đến đích
                return [state[:k] for k in range(SO_HANG + 1)]
            continue  # nếu không trùng → tiếp tục trạng thái khác

        # Ưu tiên thử cột đích của hàng hiện tại trước
        cot_uu_tien = target_cols[r]

        # Tạo danh sách các cột có thể đặt ở hàng này
        # Đầu tiên là cột đích, sau đó là các cột khác chưa dùng
        candidates = [cot_uu_tien] + [
            c for c in range(SO_HANG) if c != cot_uu_tien and c not in used
        ]

        # Sinh các trạng thái con và đưa vào hàng đợi ưu tiên
        for c in candidates:
            new_state = state + [(r, c)]      # thêm quân mới vào hàng r, cột c
            new_used = used | {c}             # đánh dấu cột đó đã bị chiếm
            h_new = heuristic(new_state, target_cols)  # tính heuristic mới
            heapq.heappush(frontier, (h_new, new_state, new_used))

    # Nếu không tìm thấy trạng thái đích → trả về các bước đã duyệt
    return buoc
```
![greedy](https://github.com/user-attachments/assets/e18c41c0-563b-4f19-bad1-01f88196ea7c)

### 2.7. Astar
Thuật toán A* là thuật toán tìm kiếm có thông tin, kết hợp giữa chi phí thực tế g(n) và hàm ước lượng h(n) để xác định trạng thái cần mở rộng tiếp theo.
Trong bài toán 8 quân xe, thuật toán A* mở rộng các trạng thái "có tổng chi phí nhỏ nhất".
```python
 # ========= A* SEARCH =========
def astar_timkiem(dich):
    # Lấy danh sách cột đích theo thứ tự hàng 0..7
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # Hàng đợi ưu tiên (priority queue) lưu các trạng thái theo f = g + h tăng dần
    # Mỗi phần tử trong hàng đợi có dạng:
    # (f, g, state, used)
    #   - f: tổng chi phí (f = g + h)
    #   - g: chi phí thực tế
    #   - state: trạng thái hiện tại [(hàng, cột), ...]
    #   - used: tập các cột đã bị chiếm
    frontier = [(heuristic([], target_cols), 0, [], set())]

    # Danh sách lưu lại các bước duyệt để minh họa hoặc hiển thị
    buoc = []

    # Vòng lặp chính: duyệt đến khi hàng đợi rỗng
    while frontier:
        # Lấy trạng thái có giá trị f nhỏ nhất (ưu tiên nhất)
        f, g, state, used = heapq.heappop(frontier)

        # Lưu lại trạng thái hiện tại
        buoc.append(state)

        # Số hàng đã được đặt quân
        r = len(state)

        # Nếu đã đặt đủ 8 quân → kiểm tra xem có đạt trạng thái đích không
        if r == SO_HANG:
            # So sánh cột của từng hàng với cột đích tương ứng
            if [col for _, col in state] == target_cols:
                # Nếu khớp → trả về danh sách các bước để hiển thị
                return [state[:k] for k in range(SO_HANG + 1)]
            continue  # nếu chưa khớp → bỏ qua, duyệt trạng thái khác

        # Xác định cột đích ưu tiên ở hàng hiện tại
        cot_uu_tien = target_cols[r]

        # Tạo danh sách các cột có thể đặt (ưu tiên cột đích trước)
        candidates = [cot_uu_tien] + [
            c for c in range(SO_HANG)
            if c != cot_uu_tien and c not in used
        ]

        # Sinh các trạng thái con
        for c in candidates:
            # Tạo trạng thái mới: thêm quân vào hàng r, cột c
            new_state = state + [(r, c)]
            new_used = used | {c}

            # Cập nhật chi phí thực tế g (mỗi bước thêm 1)
            g_new = g + 1

            # Tính giá trị heuristic h (khoảng cách đến đích)
            h_new = heuristic(new_state, target_cols)

            # Tổng chi phí f = g + h
            f_new = g_new + h_new

            # Thêm trạng thái mới vào hàng đợi ưu tiên
            heapq.heappush(frontier, (f_new, g_new, new_state, new_used))

    # Nếu duyệt hết mà không tìm thấy nghiệm → trả về toàn bộ các bước duyệt
    return buoc
```
![astar](https://github.com/user-attachments/assets/d6c93d26-10e4-4a8c-90b9-340535703040)

### 2.8. Hill Climbing
Thuật toán Hill Climbing là một kỹ thuật tìm kiếm cục bộ, chỉ quan tâm đến các trạng thái “láng giềng tốt hơn” so với hiện tại.
Nó liên tục “leo lên đồi” (tăng giá trị heuristic) cho đến khi đạt cực đại cục bộ — điểm mà không có trạng thái lân cận nào tốt hơn.
Mặc dù có thể tìm nghiệm gần đúng rất nhanh, Hill Climbing không đảm bảo tìm được nghiệm tối ưu, do có thể bị kẹt ở các cực đại cục bộ hoặc vùng bằng phẳng.
```python
def hill_climbing_timkiem(dich):
    """
        - Bắt đầu từ trạng thái trống (chưa đặt quân nào).
        - Ở mỗi hàng, chọn cột giúp tăng giá trị heuristic nhiều nhất.
        - Nếu không có bước nào tốt hơn, dừng lại (kẹt ở cực đại cục bộ).
    """

    # Lấy danh sách cột đích tương ứng với từng hàng (0..7)
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    def heuristic_hc(state):
        """
        Heuristic h(state):
        Đếm số quân xe đã đặt đúng vị trí (cùng cột với trạng thái đích).
        Giá trị h càng cao → trạng thái càng “tốt”.
        """
        return sum(1 for r, c in state if c == target_cols[r])

    #ban đầu
    state = []       # danh sách các quân xe đã đặt [(r, c)]
    used = set()     # các cột đã bị chiếm
    buoc = [state.copy()]  # lưu lại các bước để minh họa

    # leo đồi
    for r in range(SO_HANG):
        best_move = None  # bước đi tốt nhất hiện tại
        best_h = -1       # giá trị heuristic cao nhất hiện có

        # Thử tất cả các cột có thể đặt ở hàng r
        for c in range(SO_HANG):
            if c not in used:
                # Tạo trạng thái mới sau khi đặt quân ở (r, c)
                new_state = state + [(r, c)]
                # Tính giá trị heuristic cho trạng thái mới
                h_val = heuristic_hc(new_state)

                # Nếu trạng thái này tốt hơn (h lớn hơn) → cập nhật
                if h_val > best_h:
                    best_h = h_val
                    best_move = (r, c)

        # Nếu không có bước nào cải thiện được → dừng lại (cực đại cục bộ)
        if best_move is None:
            break

        # Cập nhật trạng thái với bước đi tốt nhất
        state.append(best_move)
        used.add(best_move[1])
        buoc.append(state.copy())

        # Nếu trạng thái hiện tại đạt đích → trả về danh sách các bước
        if [col for _, col in state] == target_cols:
            return [state[:k] for k in range(len(state) + 1)]

    # Nếu leo đồi không đạt đích → trả về các bước đã duyệt
    return buoc
```
![hill](https://github.com/user-attachments/assets/7d68359f-554f-45fd-85fd-2cbbe0e7d05b)

### 2.9. Simulated Annealing
Thuật toán Simulated Annealing là một mở rộng của Hill Climbing giúp tránh kẹt ở cực đại cục bộ.
Bằng cách cho phép chấp nhận các trạng thái xấu hơn với xác suất giảm dần theo nhiệt độ, thuật toán có thể “thoát khỏi” vùng cục bộ để tìm lời giải tốt hơn.
Khi nhiệt độ giảm về 0, thuật toán dần hội tụ và dừng lại.
Trong bài toán 8 quân xe, Simulated Annealing giúp tìm ra cấu hình đích hiệu quả hơn Hill Climbing, đặc biệt khi không gian tìm kiếm có nhiều cực trị cục bộ.
```python
def simulated_annealing_timkiem(dich):
    """
        - Bắt đầu với trạng thái ngẫu nhiên.
        - Lặp lại việc chọn trạng thái láng giềng và quyết định di chuyển dựa vào:
            + Nếu trạng thái mới tốt hơn → chấp nhận.
            + Nếu xấu hơn → vẫn có thể chấp nhận với xác suất phụ thuộc vào nhiệt độ T.
        - Nhiệt độ T giảm dần theo thời gian → giảm dần khả năng chấp nhận trạng thái xấu.
        - Khi T đủ nhỏ hoặc đạt đích → dừng lại.
    """

    T0 = 10.0         # Nhiệt độ khởi tạo
    alpha = 0.995     # Hệ số giảm nhiệt (0 < alpha < 1)
    Tmin = 1e-4       # Ngưỡng nhiệt độ nhỏ nhất
    max_outer = 5000  # Giới hạn vòng lặp ngoài để tránh lặp vô hạn

    # đích
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    def h(cols):
        """
        Hàm heuristic h(H):
        Đếm số hàng có quân xe KHÁC vị trí đích.
        Giá trị càng nhỏ → trạng thái càng “tốt”.
        """
        return sum(1 for r, c in enumerate(cols) if c != target_cols[r])

    # tt ban đầu
    cols = list(range(SO_HANG))   # Danh sách cột 0..7 (tương ứng với 8 hàng)
    random.shuffle(cols)          # Xáo trộn ngẫu nhiên vị trí ban đầu

    # làm nguộinguội
    t = 0
    while t < max_outer:
        # Nếu đã đạt đích → dừng và trả về các bước
        if cols == target_cols:
            goal_state = [(r, cols[r]) for r in range(SO_HANG)]
            return [goal_state[:k] for k in range(0, SO_HANG + 1)]

        # Tính nhiệt độ hiện tại theo công thức T = T0 * (alpha ^ t)
        T = T0 * (alpha ** t)

        # Nếu nhiệt độ quá thấp → dừng tìm kiếm
        if T < Tmin:
            try:
                messagebox.showinfo("Kết quả", "Không tìm được trạng thái đích bằng Simulated Annealing.")
            except Exception:
                pass
            return []

        # sinh láng giềng
        queue = []
        for i in range(SO_HANG - 1):
            for j in range(i + 1, SO_HANG):
                # Tạo trạng thái mới bằng cách hoán đổi hai cột
                new_cols = cols[:]
                new_cols[i], new_cols[j] = new_cols[j], new_cols[i]
                queue.append(new_cols)

        if not queue:
            try:
                messagebox.showinfo("Kết quả", "Không có trạng thái kế tiếp được sinh ra.")
            except Exception:
                pass
            return []

        # đánh giá và di chuyểnchuyển
        hH = h(cols)               # heuristic của trạng thái hiện tại
        M = min(queue, key=h)      # chọn láng giềng tốt nhất (có h nhỏ nhất)
        delta = h(M) - hH          # thay đổi giá trị heuristic

        if delta < 0:
            # Nếu trạng thái mới tốt hơn → chấp nhận
            cols = M
        else:
            # Nếu trạng thái xấu hơn → chấp nhận với xác suất e^(-Δ/T)
            p = math.exp(-delta / T)
            if random.random() < p:
                cols = M  # "chấp nhận rủi ro" để thoát khỏi cực đại cục bộ

        # Tăng bộ đếm vòng lặp
        t += 1

    # nếu ko thấy đích
    try:
        messagebox.showinfo("Kết quả", "Không tìm được trạng thái đích bằng Simulated Annealing.")
    except Exception:
        pass
    return []
```
![sa](https://github.com/user-attachments/assets/59c51b39-0e3b-43d8-9fc0-c361c534cc2c)

### 2.10. Beam Search
Thuật toán Beam Search là một biến thể của tìm kiếm theo chiều rộng (BFS), trong đó chỉ giữ lại một số lượng giới hạn (beam_width) các trạng thái “tốt nhất” ở mỗi tầng dựa theo giá trị heuristic.
Nhờ vậy, Beam Search giảm mạnh chi phí bộ nhớ và thời gian, nhưng không đảm bảo tìm được nghiệm tối ưu.
```python
 # ========= BEAM SEARCH =========
def beam_search_timkiem(dich, beam_width=3):
    # đích
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    def h(state):
        """
        Hàm heuristic h(state):
        Đếm số quân xe chưa đặt đúng vị trí cột so với trạng thái đích.
        Càng nhỏ → trạng thái càng "gần đích".
        """
        return sum(1 for i, (_, c) in enumerate(state) if c != target_cols[i])

    frontier = [([], set())]  # Danh sách trạng thái ban đầu: (state, used_cols)

    while frontier:
        new_frontier = []  # Danh sách trạng thái mới của tầng kế tiếp

        # Duyệt qua từng trạng thái trong tầng hiện tại
        for state, used in frontier:
            r = len(state)  # Hàng hiện tại (độ sâu trong cây tìm kiếm)

            # Nếu đã đủ 8 quân → kiểm tra xem đạt đích chưa
            if r == SO_HANG:
                if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                    return [state[:k] for k in range(0, SO_HANG + 1)]
                continue  # nếu chưa đạt → bỏ qua

            # sinh concon
            next_cols = []

            # Ưu tiên thử cột đúng của hàng hiện tại trước
            if target_cols[r] not in used:
                next_cols.append(target_cols[r])

            # Sau đó thử các cột khác chưa dùng
            for c in range(SO_HANG):
                if c != target_cols[r] and c not in used:
                    next_cols.append(c)

            # Tạo danh sách trạng thái kế tiếp
            for c in next_cols:
                new_frontier.append((state + [(r, c)], used | {c}))

        # Sắp xếp theo giá trị heuristic tăng dần (càng gần đích càng tốt)
        new_frontier.sort(key=lambda x: h(x[0]))

        # Giữ lại beam_width trạng thái tốt nhất
        frontier = new_frontier[:beam_width]

    # Nếu không tìm thấy trạng thái đích
    return []
```
![beam](https://github.com/user-attachments/assets/eed2c79c-29dd-4238-b48e-8d8de48fcee1)

### 2.11. Genetic
Thuật toán Genetic Algorithm (GA) mô phỏng quá trình chọn lọc tự nhiên để tìm nghiệm tối ưu.
```python
def genetic_algorithm_timkiem(dich, pop_size=30, generations=500, mutation_rate=0.2):
    """
        - chọn lọc, lai ghép, và đột biến để tìm lời giải tốt dần theo thời gian.
        - Mỗi "cá thể" biểu diễn một cách sắp xếp các quân xe trên bàn cờ.
        - Fitness (độ thích nghi): số quân xe đặt đúng vị trí cột so với trạng thái đích.
    """

    # đích
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # hàm fitness
    def fitness(cols):
        """
        Fitness = số quân xe đặt đúng cột so với đích.
        Càng cao → cá thể càng “tốt”.
        """
        return sum(1 for i, c in enumerate(cols) if c == target_cols[i])

    # sinh cá thể ngãu nhiên
    def random_individual():
        """
        Tạo một cá thể ngẫu nhiên (một hoán vị của 8 cột).
        Mỗi vị trí tương ứng với 1 hàng, giá trị là cột đặt quân.
        """
        cols = list(range(SO_HANG))
        random.shuffle(cols)
        return cols

    # lai ghépghép
    def crossover(p1, p2):
        """
        Lai ghép giữa hai cá thể cha mẹ để tạo con.
        Giữ nguyên một đoạn từ cha, và hoàn thành phần còn lại từ mẹ
        theo thứ tự xuất hiện mà không trùng lặp.
        """
        a, b = sorted(random.sample(range(SO_HANG), 2))  # chọn 2 điểm cắt
        child = [-1] * SO_HANG
        child[a:b] = p1[a:b]  # sao chép đoạn giữa từ cha
        fill = [c for c in p2 if c not in child]  # lấy phần còn lại từ mẹ
        j = 0
        for i in range(SO_HANG):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return child

    # đột biến
    def mutate(cols):
        """
        Đột biến: hoán đổi ngẫu nhiên hai vị trí (với xác suất mutation_rate).
        Giúp duy trì đa dạng di truyền và tránh kẹt ở nghiệm cục bộ.
        """
        if random.random() < mutation_rate:
            i, j = random.sample(range(SO_HANG), 2)
            cols[i], cols[j] = cols[j], cols[i]
        return cols

    # khởi tạo quần thể ban đầu
    population = [random_individual() for _ in range(pop_size)]

    # vòng lặp tiến hóa
    for gen in range(generations):
        # sắp xếp quần thể theo fitness giảm dần (tốt nhất lên đầu)
        population.sort(key=lambda ind: fitness(ind), reverse=True)

        #  Kiểm tra xem cá thể tốt nhất đã đạt đích chưa
        if fitness(population[0]) == SO_HANG:
            best = population[0]
            goal_state = [(r, best[r]) for r in range(SO_HANG)]
            return [goal_state[:k] for k in range(0, SO_HANG + 1)]

        #  Tạo quần thể mới
        new_population = population[:2]  # Elitism: giữ lại 2 cá thể tốt nhất

        #  Lai ghép & đột biến để sinh thế hệ mới
        while len(new_population) < pop_size:
            # chọn ngẫu nhiên 2 cha mẹ trong top 10 (chọn lọc tốt nhất)
            p1, p2 = random.sample(population[:10], 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        # Cập nhật quần thể
        population = new_population

    # kết thúc, ko thấy đích
    return []
```
![gen](https://github.com/user-attachments/assets/48670a3b-be75-4491-bd31-ece4ab257394)


### 2.12. And Or Tree Search
Thuật toán kết hợp hai kiểu nút — OR (lựa chọn hành động) và AND (ràng buộc điều kiện đồng thời) 
Trong bài toán 8 quân xe, AND–OR Search có thể được xem là một cách mô hình hóa quy trình ra quyết định tuần tự, nơi mỗi bước lựa chọn cột đặt xe là một hành động (OR), và toàn bộ chuỗi hành động phải dẫn đến trạng thái hợp lệ (AND).
```python
def andor_timkiem(dich):
    """
        - Nút OR: lựa chọn một trong các hành động khả thi.
        - Nút AND: yêu cầu tất cả các kết quả con đều phải thành công.
    """

    # đích
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    buoc = []  # danh sách để ghi lại các bước duyệt

    # NÚT OR: chọn 1 trong nhiều hành động
    def or_search(state, used, path):
        buoc.append(state.copy())  # lưu lại bước hiện tại
        r = len(state)  # hàng đang xét

        # Nếu đã đặt đủ 8 quân → kiểm tra đạt đích chưa
        if r == SO_HANG:
            if [col for _, col in state] == target_cols:
                return state
            return None

        # Phát hiện vòng lặp (tránh lặp lại trạng thái)
        if r in path:
            return None

        # Duyệt qua các hành động có thể (các cột chưa dùng)
        for c in range(SO_HANG):
            if c not in used:
                result = and_search(state + [(r, c)], used | {c}, path | {r})
                if result is not None:
                    return result  # chỉ cần một nhánh thành công
        return None  # không có hành động nào thành công

    #  NÚT AND: tất cả nhánh con phải thành công
    def and_search(state, used, path):
        """
        mỗi hành động không có nhiều trạng thái kế tiếp
        nên ở đây and_search gọi lại or_search.
        """
        result = or_search(state, used, path)
        if result is None:
            return None
        return result

    # gọi hàm chính
    result = or_search([], set(), set())

    # Nếu tìm thấy lời giải, trả về danh sách trạng thái từ đầu → đích
    if result is not None:
        return [result[:k] for k in range(0, SO_HANG + 1)]

    # Nếu không tìm được, trả về toàn bộ các bước đã duyệt
    return buoc
```
![andor](https://github.com/user-attachments/assets/5b3d021e-4816-49cb-997a-1bb09181fd68)


### 2.13. Tìm kiếm với tập trạng thái niềm tin
Thuật toán BFS Belief-State Search mở rộng tìm kiếm theo chiều rộng truyền thống vào không gian niềm tin (Belief Space).
Mỗi nút trong cây tìm kiếm biểu diễn một tập hợp trạng thái có thể xảy ra, thay vì chỉ một trạng thái duy nhất.
```python
def bfs_belief_timkiem(dich):
    # đích
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    buoc = []  # lưu các bước tìm kiếm

    def is_goal(state):
        """
        Kiểm tra xem trạng thái có phải là trạng thái đích không.
        """
        return len(state) == SO_HANG and all(state[i][1] == target_cols[i] for i in range(SO_HANG))

    # khởi tạo niềm tin ban đầu
    initial_belief = frozenset({tuple([])})  # ban đầu chỉ biết mình đang ở trạng thái rỗng
    queue = deque([(initial_belief, [])])    # hàng đợi BFS: (belief_state, đường đi)
    visited = set([initial_belief])          # tập các belief đã thăm

    # BFS
    while queue:
        belief, path = queue.popleft()
        buoc.append([list(s) for s in belief])  # lưu lại các trạng thái hiện tại trong belief

        # Nếu tất cả trạng thái trong belief đều là đích → dừng
        if all(is_goal(list(s)) for s in belief):
            sample = list(belief)[0]
            return [list(sample[:k]) for k in range(len(sample) + 1)]

        # xác định trạng thái hiện tại
        r = len(next(iter(belief)))  # số hàng đã được đặt quân xe
        if r >= SO_HANG:
            continue

        # sinh tt mớimới
        for c in range(SO_HANG):
            next_belief = set()

            # Tạo các trạng thái kế tiếp có thể có khi đặt quân ở cột c
            for s in belief:
                used_cols = {col for _, col in s}
                if c not in used_cols:
                    new_state = list(s) + [(r, c)]
                    next_belief.add(tuple(new_state))

            if not next_belief:
                continue  # nếu không có trạng thái hợp lệ thì bỏ qua

            # Biểu diễn belief mới bằng frozenset (bất biến, dùng để hash)
            next_belief = frozenset(next_belief)

            # Nếu belief mới chưa được duyệt → thêm vào hàng đợi
            if next_belief not in visited:
                visited.add(next_belief)
                queue.append((next_belief, path + [c]))

    # không thấy lời giải
    return buoc
```
![niemtin](https://github.com/user-attachments/assets/4f63a868-5610-43c4-bd53-5923f5dfc9a8)


### 2.14. Tìm kiếm với tập trạng thái niềm tin nhìn thấy một phần
Thuật toán tìm kiếm theo chiều sâu (DFS) cho môi trường quan sát một phần, nơi tác nhân chỉ biết một phần.
Phần trạng thái đã biết được cố định trong quá trình tìm kiếm, còn các phần chưa biết sẽ được mở rộng dần theo chiều sâu.
```python
def dfs_belief_partial_timkiem(dich, da_biet=None, n=SO_HANG):
    if da_biet is None:
        da_biet = []

    buoc = []  # lưu lại các bước duyệt
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]  # cột đích cho từng hàng
    known = {r: c for r, c in da_biet}  # các vị trí đã biết (hàng → cột)

    def is_goal(state):
        return len(state) == n and all(state[i][1] == target_cols[i] for i in range(n))

    stack = [([], set())]   # (state hiện tại, tập các cột đã dùng)
    while stack:
        state, used = stack.pop()  # lấy trạng thái trên đỉnh stack
        buoc.append(state.copy())  # lưu lại bước hiện tại
        r = len(state)             # hàng đang xét

        if r == n:
            if is_goal(state):
                return [state[:k] for k in range(n + 1)]
            continue

        if r in known:
            col = known[r]  # cột đã biết
            if col not in used:
                stack.append((state + [(r, col)], used | {col}))

        else:
            cot_uu_tien = target_cols[r]  # cột đúng theo trạng thái đích
            candidates = []

            # Ưu tiên thử cột đúng trước
            if cot_uu_tien not in used:
                candidates.append(cot_uu_tien)

            # Thêm các cột còn lại (chưa dùng) để thử
            for c in range(n):
                if c != cot_uu_tien and c not in used:
                    candidates.append(c)

            # Duyệt các ứng viên theo thứ tự ngược (để giống DFS)
            for c in reversed(candidates):
                stack.append((state + [(r, c)], used | {c}))

    # Nếu không tìm thấy, trả về tất cả các bước đã duyệt
    return buoc
```
![motphan](https://github.com/user-attachments/assets/82ddc82b-c52b-4b8b-8e98-7ea958a81209)


### 2.15. Backtracking
Thuật toán Backtracking (quay lui) là một phương pháp tìm kiếm có ràng buộc, dựa trên nguyên tắc “thử – sai – quay lại”.
Nó liên tục mở rộng trạng thái bằng cách thêm phần tử hợp lệ, và nếu gặp ngõ cụt, thuật toán quay lại bước trước để thử lựa chọn khác.
```python
def backtracking_timkiem(dich, n=SO_HANG):
    """
        - Duyệt tất cả các cách đặt quân xe theo hàng.
        - Với mỗi hàng, thử đặt 1 quân vào từng cột hợp lệ.
        - Nếu đặt được hết 8 quân thỏa điều kiện → đạt trạng thái đích.
        - Nếu bị kẹt (không có vị trí hợp lệ) → quay lui để thử lại.
    """

    buoc = []  # lưu lại các trạng thái qua từng bước

    def an_toan(trangthai, row, col):
        for r, c in trangthai:
            if c == col:
                return False  # cùng cột → xung đột
        return True

    def thu(row, trangthai):
        """
        Đặt quân xe lần lượt theo từng hàng.
        Nếu đạt đến hàng cuối (row == n) → kiểm tra đạt đích hay chưa.
        Nếu không, thử tất cả các cột hợp lệ cho hàng hiện tại.
        """
        # Nếu đã đặt đủ n quân → kiểm tra xem có trùng đích không
        if row == n:
            if set(trangthai) == set(dich):
                # Lưu toàn bộ quá trình đạt đích
                buoc.extend([trangthai[:k] for k in range(0, n + 1)])
                return True
            return False

        # Thử đặt quân ở từng cột có thể
        for col in range(n):
            if an_toan(trangthai, row, col):
                # Đặt quân
                trangthai.append((row, col))

                # Gọi đệ quy cho hàng tiếp theo
                if thu(row + 1, trangthai):
                    return True  # nếu thành công → dừng

                # Nếu không thành công → quay lui
                trangthai.pop()

        # Nếu không có cách nào hợp lệ → quay lui
        return False

    thu(0, [])
    return buoc
```
![backtrack](https://github.com/user-attachments/assets/5dda405a-d2fb-4456-ac8e-dfd15d151ac4)


### 2.16. Forward Check
Bằng cách duy trì tập giá trị khả dĩ (domain) cho mỗi biến, và loại bỏ sớm các giá trị không hợp lệ sau mỗi bước gán, thuật toán giúp phát hiện xung đột trước khi đi sâu vào cây tìm kiếm.
Nhờ đó, nó giảm đáng kể số lượng nhánh cần duyệt và nâng cao hiệu suất giải bài toán có ràng buộc (CSP).
```python
def backtracking_forward_timkiem(dich, n=SO_HANG):
    buoc = []  # lưu các bước tìm kiếm (để trực quan hóa)

    def thu(row, trangthai, domains):
        # Nếu đã đặt đủ n quân → kiểm tra đạt đích chưa
        if row == n:
            if set(trangthai) == set(dich):
                buoc.extend([trangthai[:k] for k in range(0, n + 1)])
                return True
            return False

        # Duyệt từng giá trị khả dĩ trong miền của hàng hiện tại
        for col in list(domains[row]):
            trangthai.append((row, col))

            # Tạo bản sao miền giá trị để cập nhật (không ảnh hưởng cha)
            new_domains = [set(d) for d in domains]

            # Forward checking:
            # Bỏ cột vừa chọn khỏi miền của tất cả các hàng dưới
            for r in range(row + 1, n):
                if col in new_domains[r]:
                    new_domains[r].remove(col)

            # Nếu sau khi cập nhật, mọi miền vẫn còn ít nhất 1 giá trị
            # thì tiếp tục đệ quy
            if all(new_domains[r] for r in range(row + 1, n)):
                if thu(row + 1, trangthai, new_domains):
                    return True

            # Nếu không hợp lệ → quay lui
            trangthai.pop()

        # Nếu không có lựa chọn nào hợp lệ → quay lui
        return False

    domains = [set(range(n)) for _ in range(n)]  # ban đầu, mỗi hàng có thể chọn mọi cột
    thu(0, [], domains)
    return buoc
```
![FC](https://github.com/user-attachments/assets/37ec57ae-d232-4896-ac86-b648e3c6790b)


### 2.17. AC3
Duy trì tính nhất quán cung (arc consistency) giữa các biến trong bài toán ràng buộc.
```python
def ac3_timkiem(dich, n=SO_HANG):
    """
        - Mỗi biến (hàng) có một miền giá trị (domain) gồm các cột có thể đặt quân xe.
        - Nếu một giá trị trong miền của xi không có giá trị tương thích trong miền của xj
          (theo ràng buộc), thì loại bỏ giá trị đó khỏi miền xi.
        - Lặp lại cho đến khi tất cả các miền đều nhất quán.
        - Sau đó, dùng miền đã rút gọn để tạo nghiệm khả dĩ.
    """

    buoc = []  # Lưu các bước thay đổi miền giá trị (để quan sát tiến trình)
    goal_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    #  Khởi tạo miền giá trị cho mỗi biến (hàng) 
    domains = {r: set(range(n)) for r in range(n)}  # ban đầu: mỗi hàng có thể chọn mọi cột

    # Hàm kiểm tra tính nhất quán giữa hai biến 
    def consistent(xi, xj, vi, vj):
        # Hai biến xi, xj là nhất quán nếu giá trị của chúng không trùng nhau.
        return vi != vj

    #Tạo danh sách các cung (xi, xj) cần kiểm tra
    queue = deque([(xi, xj) for xi in range(n) for xj in range(n) if xi != xj])

    # Duy trì tính nhất quán cung
    while queue:
        xi, xj = queue.popleft()
        revised = False
        to_remove = set()

        # Duyệt qua tất cả giá trị trong miền của xi
        for vi in domains[xi]:
            # Nếu không tồn tại giá trị nào trong miền của xj tương thích → loại bỏ vi
            if not any(consistent(xi, xj, vi, vj) for vj in domains[xj]):
                to_remove.add(vi)

        # Nếu có thay đổi miền của xi
        if to_remove:
            domains[xi] -= to_remove
            revised = True
            buoc.append([{r: sorted(list(domains[r])) for r in range(n)}])  # lưu snapshot

            # Nếu miền xi rỗng → không có nghiệm
            if not domains[xi]:
                return []

            # Thêm các cung liên quan đến xi trở lại hàng đợi
            for xk in range(n):
                if xk != xi and xk != xj:
                    queue.append((xk, xi))

    # Sau khi đạt tính nhất quán, chọn lời giải khả dĩ
    solution = []
    used_cols = set()

    for r in range(n):
        # Chọn giá trị gần đích nhất (ưu tiên vị trí đúng trong goal_cols)
        candidates = sorted(domains[r], key=lambda c: abs(c - goal_cols[r]))
        for c in candidates:
            if c not in used_cols:
                solution.append((r, c))
                used_cols.add(c)
                break

    #  Kiểm tra xem có đạt trạng thái đích không
    if [col for _, col in solution] == goal_cols:
        return [solution[:k] for k in range(n + 1)]
    else:
        return [solution[:k] for k in range(len(solution) + 1)]
```

![ac_3](https://github.com/user-attachments/assets/6047f121-3bcd-41a9-9b93-b2c451eb8f20)

## 3. Giao diện  
### 3.1. StartScreen  
<img width="1005" height="792" alt="Screenshot 2025-10-16 084407" src="https://github.com/user-attachments/assets/5311100c-619d-4393-a4ae-67232a386aa3" />

### 3.2. Giao diện chính  
<img width="1515" height="962" alt="Screenshot 2025-10-16 085242" src="https://github.com/user-attachments/assets/e74855fd-0bf2-4d83-9bad-979ec036ecb2" />

### 3.3. Lịch sử
<img width="806" height="657" alt="Screenshot 2025-10-16 085933" src="https://github.com/user-attachments/assets/40e90141-1161-456a-a0a0-aed0d71159d9" />


<img width="1918" height="1021" alt="Screenshot 2025-10-16 090001" src="https://github.com/user-attachments/assets/755bff33-03f4-4ea3-a0b7-873fc1203803" />






