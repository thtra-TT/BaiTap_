import tkinter as tk
import tkinter.font as tkfont
import random
from collections import deque
import heapq

SO_HANG = 8
O_TRAI = 70
O_PHAI = 60
KHOANG_THOI_GIAN = 300  

root = tk.Tk()
root.title("Trò chơi 8 quân xe")
root.configure(bg="#25324D")

khung_chinh = tk.Frame(root, bg="#25324D")
khung_chinh.pack(padx=20, pady=20)

tieu_de = tk.Label(
    khung_chinh,
    text="TRÒ CHƠI 8 QUÂN XE",
    font=("Arial", 27, "bold"),
    bg="#25324D",
    fg="white"
)
tieu_de.grid(row=0, column=0, columnspan=3, pady=(0, 20))

# bàn cờ trái
khung_trai = tk.Frame(khung_chinh, bg="#344161")
khung_trai.grid(row=1, column=0, padx=10)

lbl_trai = tk.Label(
    khung_trai,
    text="Bàn cờ thuật toán",
    font=("Arial", 15, "bold"),
    bg="#344161",
    fg="white"
)
lbl_trai.pack(pady=(10, 5))

canvas_trai = tk.Canvas(
    khung_trai,
    width=SO_HANG * O_TRAI,
    height=SO_HANG * O_TRAI,
    bg="white",
    highlightthickness=2,
    highlightbackground="#ecf0f1"
)
canvas_trai.pack(padx=10, pady=10)

# các nút
khung_giua = tk.Frame(khung_chinh, bg="#25324D")
khung_giua.grid(row=1, column=1, padx=10, sticky="n")

# bàn cờ trạng thái đích
khung_phai = tk.Frame(khung_chinh, bg="#344161")
khung_phai.grid(row=1, column=2, padx=10)

lbl_phai = tk.Label(
    khung_phai,
    text="Bàn cờ đích",
    font=("Arial", 15, "bold"),
    bg="#344161",
    fg="white"
)
lbl_phai.pack(pady=(10, 5))

canvas_phai = tk.Canvas(
    khung_phai,
    width=SO_HANG * O_PHAI,
    height=SO_HANG * O_PHAI,
    bg="white",
    highlightthickness=2,
    highlightbackground="#ecf0f1"
)
canvas_phai.pack(padx=10, pady=10)

# vẽ bàn cờ 
def ve_banco(canvas, kichthuoc):
    canvas.delete("all")
    for r in range(SO_HANG):
        for c in range(SO_HANG):
            x1 = c * kichthuoc
            y1 = r * kichthuoc
            x2 = x1 + kichthuoc
            y2 = y1 + kichthuoc
            mau = "#b7b7c2" if (r + c) % 2 == 0 else "#000000"
            canvas.create_rectangle(x1, y1, x2, y2, fill=mau, outline="#7F8C8D")

# vẽ quân xe
def ve_quanxe(canvas, hang, cot, kichthuoc, mau="#cb3929"):
    x = cot * kichthuoc + kichthuoc // 2
    y = hang * kichthuoc + kichthuoc // 2
    font_xe = tkfont.Font(family="Arial", size=max(10, int(kichthuoc * 0.5)))
    canvas.create_text(x, y, text="♖", font=font_xe, fill=mau)

# sinh trạng thái đích
def tao_dich():
    ds_cot = list(range(SO_HANG))
    random.shuffle(ds_cot)
    return [(i, ds_cot[i]) for i in range(SO_HANG)]

# DFS
def dfs_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    stack = [([], set())]   # (trạng thái, set(cột đã dùng))
    buoc = []
    while stack:
        trangthai, used = stack.pop()
        buoc.append(trangthai.copy())
        r = len(trangthai)
        if r == SO_HANG:
            if all(trangthai[i][1] == target_cols[i] for i in range(SO_HANG)):
                return buoc
        else:
            cot_uu_tien = target_cols[r]
            candidates = []
            if cot_uu_tien not in used:
                candidates.append(cot_uu_tien)
            for c in range(SO_HANG):
                if c != cot_uu_tien and c not in used:
                    candidates.append(c)
            for c in reversed(candidates):
                stack.append((trangthai + [(r, c)], used | {c}))
    return buoc

# BFS
def bfs_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    queue = deque([([], set())])  # (state, used_cols)

    while queue:
        state, used = queue.popleft()
        r = len(state)

        if r == SO_HANG:
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                return [state[:k] for k in range(0, SO_HANG + 1)]
            continue

        next_cols = []
        if target_cols[r] not in used:
            next_cols.append(target_cols[r])
        for c in range(SO_HANG):
            if c != target_cols[r] and c not in used:
                next_cols.append(c)

        for c in next_cols:
            queue.append((state + [(r, c)], used | {c}))

    return []

# ========= UCS =========
# ========= UCS (Uniform-Cost Search) =========
# Chi phí mỗi bước = |c - target_cols[r]|
def ucs_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    frontier = [(0, [], set())]  # (cost, state, used_cols)
    while frontier:
        cost, state, used = heapq.heappop(frontier)
        r = len(state)

        if r == SO_HANG:
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                buoc = [state[:k] for k in range(0, SO_HANG + 1)]
                return buoc
            continue

        for c in range(SO_HANG):
            if c in used:
                continue
            step_cost = abs(c - target_cols[r])      
            new_state = state + [(r, c)]
            new_used = used | {c}
            heapq.heappush(frontier, (cost + step_cost, new_state, new_used))

    return []

# ========= DLS =========
def dls_timkiem(dich, limit=SO_HANG):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    buoc = []
    cutoff = object()  # sentinel cho cutoff

    def recursive_dls(state, used, depth):
        buoc.append(state.copy())
        if depth == SO_HANG:  
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                return state
            else:
                return None
        elif depth == limit:  # đạt giới hạn
            return cutoff
        else:
            cutoff_occurred = False
            for c in range(SO_HANG):
                if c not in used:
                    child = state + [(depth, c)]
                    result = recursive_dls(child, used | {c}, depth + 1)
                    if result is cutoff:
                        cutoff_occurred = True
                    elif result is not None:
                        return result
            return cutoff if cutoff_occurred else None

    result = recursive_dls([], set(), 0)
    if result is not None and result is not cutoff:
        return [result[:k] for k in range(0, SO_HANG + 1)]
    return buoc

# ========= Quản lý hiển thị/chạy =========
trangthai_dich = tao_dich()
ds_buoc = []
chi_so = [0]
after_id = [None]
che_do = [""]  

def ve_dich():
    ve_banco(canvas_phai, O_PHAI)
    for h, c in trangthai_dich:
        ve_quanxe(canvas_phai, h, c, O_PHAI, "#CB0000")

def ve_trangthai_hien_tai():
    ve_banco(canvas_trai, O_TRAI)
    if 0 <= chi_so[0] < len(ds_buoc):
        for h, c in ds_buoc[chi_so[0]]:
            ve_quanxe(canvas_trai, h, c, O_TRAI, "#27ae60")

def dung_auto():
    if after_id[0] is not None:
        root.after_cancel(after_id[0])
        after_id[0] = None

def phat_tiep():
    if chi_so[0] < len(ds_buoc):
        ve_trangthai_hien_tai()
        chi_so[0] += 1
        if chi_so[0] < len(ds_buoc):
            after_id[0] = root.after(KHOANG_THOI_GIAN, phat_tiep)
        else:
            after_id[0] = None

def chuan_bi_va_chay(thuat_toan):
    dung_auto()
    global ds_buoc, chi_so
    chi_so[0] = 0
    if thuat_toan == "DFS":
        ds_buoc = dfs_timkiem(trangthai_dich)
        che_do[0] = "DFS"
        lbl_trai.config(text="Bàn cờ thuật toán (DFS)")
    elif thuat_toan == "BFS":
        ds_buoc = bfs_timkiem(trangthai_dich)
        che_do[0] = "BFS"
        lbl_trai.config(text="Bàn cờ thuật toán (BFS)")
    elif thuat_toan == "UCS":
        ds_buoc = ucs_timkiem(trangthai_dich)
        che_do[0] = "UCS"
        lbl_trai.config(text="Bàn cờ thuật toán (UCS)")
    elif thuat_toan == "DLS":
        ds_buoc = dls_timkiem(trangthai_dich, limit=SO_HANG)  
        che_do[0] = "DLS"
        lbl_trai.config(text="Bàn cờ thuật toán (DLS)")

    ve_banco(canvas_trai, O_TRAI)
    phat_tiep()


def tao_dich_moi():
    dung_auto()
    global trangthai_dich, ds_buoc, chi_so
    trangthai_dich = tao_dich()
    ve_dich()
    ve_banco(canvas_trai, O_TRAI)
    chi_so[0] = 0
    if che_do[0] == "DFS":
        ds_buoc = dfs_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "BFS":
        ds_buoc = bfs_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "DLS":
        ds_buoc = dls_timkiem(trangthai_dich, limit=SO_HANG)
        phat_tiep()

# ========= Nút điều khiển =========
btn_dfs = tk.Button(
    khung_giua,
    text="Chạy DFS",
    font=("Arial", 14, "bold"),
    bg="#8e44ad",
    fg="white",
    width=14,
    command=lambda: chuan_bi_va_chay("DFS")
)
btn_dfs.pack(pady=(10, 6))

btn_bfs = tk.Button(
    khung_giua,
    text="Chạy BFS",
    font=("Arial", 14, "bold"),
    bg="#27ae60",
    fg="white",
    width=14,
    command=lambda: chuan_bi_va_chay("BFS")
)
btn_bfs.pack(pady=6)

btn_ucs = tk.Button(
    khung_giua,
    text="Chạy UCS",
    font=("Arial", 14, "bold"),
    bg="#f39c12",
    fg="white",
    width=14,
    command=lambda: chuan_bi_va_chay("UCS")
)
btn_ucs.pack(pady=6)

btn_dls = tk.Button(
    khung_giua,
    text="Chạy DLS",
    font=("Arial", 14, "bold"),
    bg="#16a085",
    fg="white",
    width=14,
    command=lambda: chuan_bi_va_chay("DLS")
)
btn_dls.pack(pady=6)

btn_dung = tk.Button(
    khung_giua,
    text="Dừng",
    font=("Arial", 14, "bold"),
    bg="#e74c3c",
    fg="white",
    width=14,
    command=dung_auto
)
btn_dung.pack(pady=6)

btn_dich = tk.Button(
    khung_giua,
    text="Tạo đích mới",
    font=("Arial", 14, "bold"),
    bg="#2980b9",
    fg="white",
    width=14,
    command=tao_dich_moi
)
btn_dich.pack(pady=(6, 12))

ve_dich()
ve_banco(canvas_trai, O_TRAI)

root.mainloop()
