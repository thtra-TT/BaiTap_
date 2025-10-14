import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import random
from collections import deque
import heapq
import math
from tkinter import messagebox
from PIL import Image, ImageTk
import pygame
import time
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


lich_su_chay = []  
pygame.mixer.init()


SO_HANG = 8
O_TRAI = 70
O_PHAI = 35
KHOANG_THOI_GIAN = 300  

root = tk.Tk()
root.title("Trò chơi 8 quân xe")
root.configure(bg="#25324D")

# Tạo frame màn hình bắt đầu
start_frame = tk.Frame(root, bg="#25324D")
start_frame.pack(fill="both", expand=True)

# Tải ảnh startScreen.png
start_img = Image.open("startScreen.png")
start_img = start_img.resize((800, 600))
start_photo = ImageTk.PhotoImage(start_img)

# Label hiển thị ảnh
start_label = tk.Label(start_frame, image=start_photo)
start_label.image = start_photo
start_label.pack(fill="both", expand=True)

# Hàm để chuyển sang giao diện chính
def start_game():
    start_frame.pack_forget()
    khung_chinh.pack(padx=20, pady=20)

# Nút Start
start_button = tk.Button(
    start_frame,
    text="START",
    font=("Arial", 20, "bold"),
    bg="#27ae60",
    fg="white",
    width=10,
    command=start_game
)
start_button.place(relx=0.5, rely=0.8, anchor="center")

khung_chinh = tk.Frame(root, bg="#25324D")

tieu_de = tk.Label(
    khung_chinh,
    text="TRÒ CHƠI 8 QUÂN XE",
    font=("Arial", 27, "bold"),
    bg="#25324D",
    fg="white"
)
tieu_de.grid(row=0, column=0, columnspan=3, pady=(0, 20))

# 
frm_header_buttons = tk.Frame(khung_chinh, bg="#25324D")
frm_header_buttons.grid(row=0, column=2, sticky="ne", padx=(0, 30), pady=(0, 20))

def go_home():
    """Quay về màn hình bắt đầu (Start Screen)"""
    try:
        dung_auto() 
    except Exception:
        pass

    # Reset toàn bộ trạng thái
    global ds_buoc, chi_so, trangthai_dich
    ds_buoc = []
    chi_so = [0]
    trangthai_dich = tao_dich()
    
    # Xóa bàn cờ
    ve_banco(canvas_trai, O_TRAI)
    ve_banco(canvas_phai, O_PHAI)

    # Ẩn khung giao diện chính
    khung_chinh.pack_forget()

    # Hiện lại màn hình Start
    start_frame.pack(fill="both", expand=True)


try:
    pygame.mixer.music.load("chess.mp3")
    pygame.mixer.music.play(-1)
    pygame.mixer.music.set_volume(0.5)
except Exception as e:
    print("Không thể phát nhạc nền:", e)

sound_on = True

def toggle_sound():
    global sound_on 
    if sound_on:
        pygame.mixer.music.pause()
        btn_sound.config(text="🔇", bg="#9E9E9E")
        sound_on = False
    else:
        pygame.mixer.music.unpause()
        btn_sound.config(text="🔊", bg="#4CAF50")
        sound_on = True

def tiep_tuc():
    """Tiếp tục chạy các bước còn lại sau khi dừng"""
    if after_id[0] is None and ds_buoc:
        ghi_trangthai("▶️ Tiếp tục chạy các bước còn lại...")
        phat_tiep()

def reset_banco():
    """Reset về bàn cờ trống ban đầu"""
    dung_auto()
    global ds_buoc, chi_so, trangthai_dich
    ds_buoc = []
    chi_so[0] = 0
    trangthai_dich = []  # không có quân đích nào
    ve_banco(canvas_trai, O_TRAI)
    ve_banco(canvas_phai, O_PHAI)
    lbl_trai.config(text="Bàn cờ thuật toán (Trống)")
    ghi_trangthai("🔁 Đã reset bàn cờ về trạng thái ban đầu.")

btn_home = tk.Button(
    frm_header_buttons,
    text="🏠",
    font=("Arial", 16),
    fg="white",
    bg="#1abc9c",
    activebackground="#16a085",
    width=3,
    height=1,
    bd=0,
    relief="flat",
    cursor="hand2",
    command=go_home
)
btn_home.pack(side="right", padx=5)

btn_sound = tk.Button(
    frm_header_buttons,
    text="🔊",
    font=("Arial", 16),
    fg="white",
    bg="#4CAF50",
    activebackground="#388E3C",
    width=3,
    height=1,
    bd=0,
    relief="flat",
    cursor="hand2",
    command=toggle_sound
)
btn_sound.pack(side="right", padx=5)


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
khung_phai.grid(row=1, column=2, padx=10, pady=(0, 0), sticky="n")


lbl_phai = tk.Label(
    khung_phai,
    text="Bàn cờ đích",
    font=("Arial", 14, "bold"),
    bg="#344161",
    fg="white"
)
lbl_phai.pack(pady=(2, 0))  # đẩy label lên sát top

canvas_phai = tk.Canvas(
    khung_phai,
    width=SO_HANG * O_PHAI,
    height=SO_HANG * O_PHAI,
    bg="white",
    highlightthickness=2,
    highlightbackground="#ecf0f1"
)
canvas_phai.pack(padx=10, pady=(0, 5))


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

#BFS
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
    cutoff = object()  

    def recursive_dls(state, used, depth):
        buoc.append(state.copy())
        if depth == SO_HANG:  
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                return state
            else:
                return None
        elif depth == limit:  
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


#interative Deep search dùng dls
def ids_timkiem(dich):
    for limit in range(1, SO_HANG + 1):
        ket_qua = dls_timkiem(dich, limit=limit)
        if ket_qua and len(ket_qua[-1]) == SO_HANG:
            return ket_qua
    return []

#interative deep search dùng dfs
def ids_dfs_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    def dls_dfs(state, used, depth, limit, buoc):
        buoc.append(state.copy())
        if depth == SO_HANG:
            if all(state[i][1] == target_cols[i] for i in range(SO_HANG)):
                return state
            return None
        if depth == limit:
            return None
        for c in range(SO_HANG):
            if c not in used:
                child = state + [(depth, c)]
                result = dls_dfs(child, used | {c}, depth + 1, limit, buoc)
                if result is not None:
                    return result
        return None

    for limit in range(1, SO_HANG + 1):
        buoc = []
        ket_qua = dls_dfs([], set(), 0, limit, buoc)
        if ket_qua is not None:
            return [ket_qua[:k] for k in range(0, SO_HANG + 1)]
    return []


#greedy search
def heuristic(state, goal_cols):
    #khoảng cách từ tt hiện tại so với tt đích tính theo cột
    return sum(abs(col - goal_cols[row]) for row, col in state)


def greedy_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    frontier = [(heuristic([], target_cols), [], set())]  # (h, state, used)
    buoc = []
    while frontier:
        h, state, used = heapq.heappop(frontier)
        buoc.append(state)
        r = len(state)
        if r == SO_HANG:
            if [col for _, col in state] == target_cols:
                return [state[:k] for k in range(SO_HANG+1)]
            continue
        cot_uu_tien = target_cols[r]
        candidates = [cot_uu_tien] + [c for c in range(SO_HANG) if c != cot_uu_tien and c not in used]
        for c in candidates:
            new_state = state + [(r, c)]
            new_used = used | {c}
            h_new = heuristic(new_state, target_cols)
            heapq.heappush(frontier, (h_new, new_state, new_used))
    return buoc

def astar_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    frontier = [(heuristic([], target_cols), 0, [], set())]  # (f, g, state, used)
    buoc = []
    while frontier:
        f, g, state, used = heapq.heappop(frontier)
        buoc.append(state)
        r = len(state)
        if r == SO_HANG:
            if [col for _, col in state] == target_cols:
                return [state[:k] for k in range(SO_HANG+1)]
            continue
        cot_uu_tien = target_cols[r]
        candidates = [cot_uu_tien] + [c for c in range(SO_HANG) if c != cot_uu_tien and c not in used]
        for c in candidates:
            new_state = state + [(r, c)]
            new_used = used | {c}
            g_new = g + 1
            h_new = heuristic(new_state, target_cols)
            f_new = g_new + h_new
            heapq.heappush(frontier, (f_new, g_new, new_state, new_used))
    return buoc

# ========= Hill Climbing =========
def hill_climbing_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    def heuristic_hc(state):
        return sum(1 for r, c in state if c == target_cols[r])

    state = []
    used = set()
    buoc = [state.copy()]

    for r in range(SO_HANG):
        best_move = None
        best_h = -1

        for c in range(SO_HANG):
            if c not in used:
                new_state = state + [(r, c)]
                h_val = heuristic_hc(new_state)
                if h_val > best_h:
                    best_h = h_val
                    best_move = (r, c)

        if best_move is None:
            break

        state.append(best_move)
        used.add(best_move[1])
        buoc.append(state.copy())

        if [col for _, col in state] == target_cols:
            return [state[:k] for k in range(len(state)+1)]

    return buoc

def simulated_annealing_timkiem(dich):
    T0 = 10.0
    alpha = 0.995
    Tmin = 1e-4
    max_outer = 5000  # giới hạn vòng lặp ngoài để tránh vô hạn

    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # h(H): số hàng đang khác đích
    def h(cols):
        return sum(1 for r, c in enumerate(cols) if c != target_cols[r])

    cols = list(range(SO_HANG))
    random.shuffle(cols)

    t = 0
    while t < max_outer:
        if cols == target_cols:
            goal_state = [(r, cols[r]) for r in range(SO_HANG)]
            return [goal_state[:k] for k in range(0, SO_HANG + 1)]

        T = T0 * (alpha ** t)

        if T < Tmin:
            try:
                messagebox.showinfo("Kết quả", "Không tìm được trạng thái đích bằng Simulated Annealing.")
            except Exception:
                pass
            return []

        queue = []
        for i in range(SO_HANG - 1):
            for j in range(i + 1, SO_HANG):
                new_cols = cols[:]
                new_cols[i], new_cols[j] = new_cols[j], new_cols[i]
                queue.append(new_cols)

        if not queue:
            try:
                messagebox.showinfo("Kết quả", "Không có trạng thái kế tiếp được sinh ra.")
            except Exception:
                pass
            return []
        
        hH = h(cols)
        M = min(queue, key=h)  
        delta = h(M) - hH

        if delta < 0:
            cols = M
        else:
            p = math.exp(-delta / T)
            if random.random() < p:
                cols = M
            

        t += 1

    try:
        messagebox.showinfo("Kết quả", "Không tìm được trạng thái đích bằng Simulated Annealing.")
    except Exception:
        pass
    return []

#Beam search
def beam_search_timkiem(dich, beam_width=3):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    def h(state):
        return sum(1 for i, (_, c) in enumerate(state) if c != target_cols[i])

    frontier = [([], set())]

    while frontier:
        new_frontier = []
        for state, used in frontier:
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
                new_frontier.append((state + [(r, c)], used | {c}))

        new_frontier.sort(key=lambda x: h(x[0]))
        frontier = new_frontier[:beam_width]

    return []

def genetic_algorithm_timkiem(dich, pop_size=30, generations=500, mutation_rate=0.2):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # --- fitness: số quân đúng vị trí ---
    def fitness(cols):
        return sum(1 for i, c in enumerate(cols) if c == target_cols[i])

    # --- sinh cá thể ngẫu nhiên ---
    def random_individual():
        cols = list(range(SO_HANG))
        random.shuffle(cols)
        return cols

    # --- lai ghép (Order Crossover - OX) ---
    def crossover(p1, p2):
        a, b = sorted(random.sample(range(SO_HANG), 2))
        child = [-1] * SO_HANG
        child[a:b] = p1[a:b]
        fill = [c for c in p2 if c not in child]
        j = 0
        for i in range(SO_HANG):
            if child[i] == -1:
                child[i] = fill[j]
                j += 1
        return child

    # --- đột biến ---
    def mutate(cols):
        if random.random() < mutation_rate:
            i, j = random.sample(range(SO_HANG), 2)
            cols[i], cols[j] = cols[j], cols[i]
        return cols

    # --- khởi tạo quần thể ---
    population = [random_individual() for _ in range(pop_size)]

    for gen in range(generations):
        # sắp xếp theo fitness
        population.sort(key=lambda ind: fitness(ind), reverse=True)

        # kiểm tra có lời giải chưa
        if fitness(population[0]) == SO_HANG:
            best = population[0]
            goal_state = [(r, best[r]) for r in range(SO_HANG)]
            return [goal_state[:k] for k in range(0, SO_HANG + 1)]

        # chọn lọc + lai ghép
        new_population = population[:2]  # elitism: giữ lại 2 tốt nhất
        while len(new_population) < pop_size:
            p1, p2 = random.sample(population[:10], 2)  # chọn trong top 10
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return []

# ========= AND-OR Search =========
def andor_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    buoc = []

    def or_search(state, used, path):
        buoc.append(state.copy())
        r = len(state)

        if r == SO_HANG:
            if [col for _, col in state] == target_cols:
                return state
            return None

        if r in path:  # phát hiện vòng lặp
            return None

        for c in range(SO_HANG):
            if c not in used:
                result = and_search(state + [(r, c)], used | {c}, path | {r})
                if result is not None:
                    return result
        return None

    def and_search(state, used, path):
        result = or_search(state, used, path)
        if result is None:
            return None
        return result

    result = or_search([], set(), set())
    if result is not None:
        return [result[:k] for k in range(0, SO_HANG + 1)]
    return buoc

# ========= BFS trên tập trạng thái niềm tin =========

def bfs_belief_timkiem(dich):
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    buoc = []

    def is_goal(state):
        return len(state) == SO_HANG and all(state[i][1] == target_cols[i] for i in range(SO_HANG))

    initial_belief = frozenset({tuple([])})
    queue = deque([(initial_belief, [])])
    visited = set([initial_belief])

    while queue:
        belief, path = queue.popleft()
        buoc.append([list(s) for s in belief])

        if all(is_goal(list(s)) for s in belief):
            sample = list(belief)[0]
            return [list(sample[:k]) for k in range(len(sample)+1)]

        r = len(next(iter(belief)))
        if r >= SO_HANG:
            continue

        for c in range(SO_HANG):
            next_belief = set()
            for s in belief:
                used_cols = {col for _, col in s}
                if c not in used_cols:
                    new_state = list(s) + [(r, c)]
                    next_belief.add(tuple(new_state))
            if not next_belief:
                continue

            next_belief = frozenset(next_belief)
            if next_belief not in visited:
                visited.add(next_belief)
                queue.append((next_belief, path + [c]))

    return buoc

def dfs_belief_partial_timkiem(dich, da_biet=None, n=SO_HANG):
    if da_biet is None:
        da_biet = []

    buoc = []
    target_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]
    known = {r: c for r, c in da_biet}

    def is_goal(state):
        return len(state) == n and all(state[i][1] == target_cols[i] for i in range(n))

    stack = [([], set())]   # (state, used columns)

    while stack:
        state, used = stack.pop()
        buoc.append(state.copy())
        r = len(state)

        if r == n:
            if is_goal(state):
                return [state[:k] for k in range(n+1)]
            continue

        if r in known:
            col = known[r]
            if col not in used:
                stack.append((state + [(r, col)], used | {col}))
        else:
            cot_uu_tien = target_cols[r]
            candidates = []
            if cot_uu_tien not in used:
                candidates.append(cot_uu_tien)
            for c in range(n):
                if c != cot_uu_tien and c not in used:
                    candidates.append(c)

            for c in reversed(candidates):
                stack.append((state + [(r, c)], used | {c}))

    return buoc



def backtracking_timkiem(dich, n=SO_HANG):
    buoc = []

    def an_toan(trangthai, row, col):
        for r, c in trangthai:
            if c == col:
                return False
        return True

    def thu(row, trangthai):
        if row == n:
            if set(trangthai) == set(dich):
                buoc.extend([trangthai[:k] for k in range(0, n+1)])
                return True
            return False

        for col in range(n):
            if an_toan(trangthai, row, col):
                trangthai.append((row, col))
                if thu(row + 1, trangthai):
                    return True
                trangthai.pop()
        return False

    thu(0, [])
    return buoc

def backtracking_forward_timkiem(dich, n=SO_HANG):
    buoc = []

    def thu(row, trangthai, domains):
        if row == n:
            if set(trangthai) == set(dich):
                buoc.extend([trangthai[:k] for k in range(0, n+1)])
                return True
            return False

        for col in list(domains[row]):
            trangthai.append((row, col))

            new_domains = [set(d) for d in domains] #bỏ đi các cột đã dùng
            for r in range(row+1, n):
                if col in new_domains[r]:
                    new_domains[r].remove(col)

            if all(new_domains[r] for r in range(row+1, n)):
                if thu(row+1, trangthai, new_domains):
                    return True

            trangthai.pop()
        return False

    domains = [set(range(n)) for _ in range(n)]
    thu(0, [], domains)
    return buoc

#AC3

def ac3_timkiem(dich, n=SO_HANG):
    buoc = []
    goal_cols = [c for (_, c) in sorted(dich, key=lambda x: x[0])]

    # --- Khởi tạo miền giá trị ---
    domains = {r: set(range(n)) for r in range(n)}

    def consistent(xi, xj, vi, vj):
        return vi != vj

    queue = deque([(xi, xj) for xi in range(n) for xj in range(n) if xi != xj])

    while queue:
        xi, xj = queue.popleft()
        revised = False

        to_remove = set()
        for vi in domains[xi]:
            if not any(consistent(xi, xj, vi, vj) for vj in domains[xj]):
                to_remove.add(vi)

        if to_remove:
            domains[xi] -= to_remove
            revised = True
            buoc.append([{r: sorted(list(domains[r])) for r in range(n)}])
            if not domains[xi]:
                return []

            for xk in range(n):
                if xk != xi and xk != xj:
                    queue.append((xk, xi))

    solution = []
    used_cols = set()
    for r in range(n):
        candidates = sorted(domains[r], key=lambda c: abs(c - goal_cols[r]))
        for c in candidates:
            if c not in used_cols:
                solution.append((r, c))
                used_cols.add(c)
                break

    if [col for _, col in solution] == goal_cols:
        return [solution[:k] for k in range(n + 1)]
    else:
        return [solution[:k] for k in range(len(solution) + 1)]



# ========= Quản lý hiển thị/=========
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
        ghi_trangthai("⏸️ Đã tạm dừng thuật toán.")

def phat_tiep():
    if chi_so[0] < len(ds_buoc):
        ve_trangthai_hien_tai()
        ghi_trangthai(f"🧩 Bước {chi_so[0]+1}/{len(ds_buoc)}: Đã đặt {len(ds_buoc[chi_so[0]])} quân xe.")
        chi_so[0] += 1
        if chi_so[0] < len(ds_buoc):
            after_id[0] = root.after(KHOANG_THOI_GIAN, phat_tiep)
        else:
            after_id[0] = None
            ghi_trangthai("✅ Thuật toán hoàn tất!")
            ghi_trangthai("🎯 Đã đạt trạng thái đích!")


def chuan_bi_va_chay(thuat_toan):
    dung_auto()
    global ds_buoc, chi_so
    chi_so[0] = 0

    txt_trangthai.config(state="normal")
    txt_trangthai.delete("1.0", "end")
    txt_trangthai.insert("end", f"🚀 Đang chạy thuật toán: {thuat_toan}\n")
    txt_trangthai.insert("end", "----------------------------------\n")
    txt_trangthai.insert("end", "🔍 Bắt đầu tìm kiếm lời giải...\n")
    txt_trangthai.config(state="disabled")

    start_time = time.time()

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
    elif thuat_toan == "IDS":
        ds_buoc = ids_timkiem(trangthai_dich)
        che_do[0] = "IDS"
        lbl_trai.config(text="Bàn cờ thuật toán (IDS)")
    elif thuat_toan == "IDS-DFS":
        ds_buoc = ids_dfs_timkiem(trangthai_dich)
        che_do[0] = "IDS-DFS"
        lbl_trai.config(text="Bàn cờ thuật toán (IDS-DFS)")
    elif thuat_toan == "GREEDY":
        ds_buoc = greedy_timkiem(trangthai_dich)
        che_do[0] = "GREEDY"
        lbl_trai.config(text="Bàn cờ thuật toán (Greedy)")
    elif thuat_toan == "ASTAR":
        ds_buoc = astar_timkiem(trangthai_dich)
        che_do[0] = "ASTAR"
        lbl_trai.config(text="Bàn cờ thuật toán (A*)")
    elif thuat_toan == "HILL":
        ds_buoc = hill_climbing_timkiem(trangthai_dich)
        che_do[0] = "HILL"
        lbl_trai.config(text="Bàn cờ thuật toán (Hill Climbing)")
    elif thuat_toan == "SA":
        ds_buoc = simulated_annealing_timkiem(trangthai_dich)
        che_do[0] = "SA"
        lbl_trai.config(text="Bàn cờ thuật toán (Simulated Annealing)")
    elif thuat_toan == "BEAM":
        ds_buoc = beam_search_timkiem(trangthai_dich, beam_width=3)
        che_do[0] = "BEAM"
        lbl_trai.config(text="Bàn cờ thuật toán (Beam Search)")
    elif thuat_toan == "GENETIC":
        ds_buoc = genetic_algorithm_timkiem(trangthai_dich)
        che_do[0] = "GENETIC"
        lbl_trai.config(text="Bàn cờ thuật toán (Genetic Algorithm)")
    elif thuat_toan == "ANDOR":
        ds_buoc = andor_timkiem(trangthai_dich)
        che_do[0] = "ANDOR"
        lbl_trai.config(text="Bàn cờ thuật toán (AND-OR)")
    elif thuat_toan == "BFS-BELIEF":
        ds_buoc = bfs_belief_timkiem(trangthai_dich)
        che_do[0] = "BFS-BELIEF"
        lbl_trai.config(text="Bàn cờ thuật toán (BFS Belief)")
    elif thuat_toan == "DFS-BELIEF-PARTIAL":
        da_biet = [(r, c) for r, c in trangthai_dich[:3]]  # 3 hàng đầu trong đích
        ds_buoc = dfs_belief_partial_timkiem(trangthai_dich, da_biet)
        che_do[0] = "DFS-BELIEF-PARTIAL"
        lbl_trai.config(text="Bàn cờ thuật toán (DFS Belief Partial)")
    elif thuat_toan == "BACKTRACKING":
        ds_buoc = backtracking_timkiem(trangthai_dich)
        che_do[0] = "BACKTRACKING"
        lbl_trai.config(text="Bàn cờ thuật toán (Backtracking)")
    elif thuat_toan == "BACKTRACKING-FC":
        ds_buoc = backtracking_forward_timkiem(trangthai_dich)
        che_do[0] = "BACKTRACKING-FC"
        lbl_trai.config(text="Bàn cờ thuật toán (Backtracking-FC)")
    elif thuat_toan == "AC3":
        ds_buoc = ac3_timkiem(trangthai_dich)
        che_do[0] = "AC3"
        lbl_trai.config(text="Bàn cờ thuật toán (AC-3)")



    end_time = time.time()
    duration = round(end_time - start_time, 4)

    ket_qua = "✅ Thành công" if ds_buoc else "❌ Thất bại"
    lich_su_chay.append({
        "thuat_toan": thuat_toan,
        "thoi_gian": duration,
        "ket_qua": ket_qua,
        "thoi_diem": datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    })

    # --- Ghi ra khung trạng thái ---
    ghi_trangthai(f"🕒 Thời gian thực thi: {duration} giây")
    ghi_trangthai(f"📋 Kết quả: {ket_qua}")

    ve_banco(canvas_trai, O_TRAI)
    phat_tiep()


def tao_dich_moi():
    dung_auto()
    ghi_trangthai("🎯 Đã tạo trạng thái đích mới.")
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
    elif che_do[0] == "IDS":
        ds_buoc = ids_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "IDS-DFS":
        ds_buoc = ids_dfs_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "GREEDY":
        ds_buoc = greedy_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "ASTAR":
        ds_buoc = astar_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "UCS":
        ds_buoc = ucs_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "HILL":
        ds_buoc = hill_climbing_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "SA":
        ds_buoc = simulated_annealing_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "BEAM":
        ds_buoc = beam_search_timkiem(trangthai_dich, beam_width=3)
        phat_tiep()
    elif che_do[0] == "GENETIC":
        ds_buoc = genetic_algorithm_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "ANDOR":
        ds_buoc = andor_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "BFS-BELIEF":
        ds_buoc = bfs_belief_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "BACKTRACKING":
        ds_buoc = backtracking_timkiem(trangthai_dich)
    elif che_do[0] == "BACKTRACKING-FC":
        ds_buoc = backtracking_forward_timkiem(trangthai_dich)
        phat_tiep()
    elif che_do[0] == "DFS-BELIEF-PARTIAL":
        da_biet = [(r, c) for r, c in trangthai_dich[:3]]  # 3 hàng đầu trong đích mới
        ds_buoc = dfs_belief_partial_timkiem(trangthai_dich, da_biet)
        phat_tiep()

def xem_lich_su():
    """Hiển thị cửa sổ lịch sử chạy thuật toán (2 tab: bảng & biểu đồ)"""
    if not lich_su_chay:
        messagebox.showinfo("Lịch sử", "Chưa có lần chạy nào được ghi lại.")
        return

    win = tk.Toplevel(root)
    win.title("📜 Lịch sử chạy thuật toán")
    win.configure(bg="#1B2A41")
    win.geometry("650x500")

    lbl_title = tk.Label(
        win,
        text="📜 LỊCH SỬ CHẠY THUẬT TOÁN",
        font=("Arial", 16, "bold"),
        bg="#1B2A41",
        fg="#FFD700"
    )
    lbl_title.pack(pady=10)

    # ========== Tạo TabControl ==========
    notebook = ttk.Notebook(win)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    # --- TAB 1: Bảng lịch sử ---
    tab_table = tk.Frame(notebook, bg="#1B2A41")
    notebook.add(tab_table, text="📋 Danh sách")

    cols = ("Thuật toán", "Kết quả", "Thời gian (s)", "Thời điểm")
    table = ttk.Treeview(tab_table, columns=cols, show="headings", height=15)
    for col in cols:
        table.heading(col, text=col)
        table.column(col, anchor="center", width=140)
    table.pack(fill="both", expand=True, padx=10, pady=10)

    for entry in lich_su_chay:
        table.insert("", "end", values=(
            entry["thuat_toan"],
            entry["ket_qua"],
            entry["thoi_gian"],
            entry["thoi_diem"]
        ))

    # --- TAB 2: Biểu đồ so sánh thời gian ---
    tab_chart = tk.Frame(notebook, bg="#1B2A41")
    notebook.add(tab_chart, text="📊 Biểu đồ so sánh")

    if lich_su_chay:
        # Gom dữ liệu: chọn thời gian nhỏ nhất cho mỗi thuật toán
        data = {}
        for item in lich_su_chay:
            algo = item["thuat_toan"]
            t = item["thoi_gian"]
            if algo not in data or t < data[algo]:
                data[algo] = t

        algos = list(data.keys())
        times = list(data.values())

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(algos, times)
        ax.set_title("So sánh thời gian chạy các thuật toán", fontsize=12, weight="bold")
        ax.set_xlabel("Thuật toán")
        ax.set_ylabel("Thời gian (giây)")
        plt.xticks(rotation=45, ha="right")

        # Ghi giá trị trên đầu cột
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom")

        canvas = FigureCanvasTkAgg(fig, master=tab_chart)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # --- Nút đóng ---
    ttk.Button(win, text="Đóng", command=win.destroy).pack(pady=10)


btn_history = tk.Button(
    frm_header_buttons,
    text="📜",
    font=("Arial", 16),
    fg="white",
    bg="#7f8c8d",
    activebackground="#616161",
    width=3,
    height=1,
    bd=0,
    relief="flat",
    cursor="hand2",
    command=xem_lich_su
)
btn_history.pack(side="right", padx=5)


# ====== Chọn thuật toán trong combobox ======
lbl_chon = tk.Label(
    khung_giua,
    text="Chọn thuật toán:",
    font=("Arial", 14, "bold"),
    bg="#25324D",
    fg="white"
)
lbl_chon.grid(row=0, column=0, sticky="w", pady=(10, 5))

ds_thuat_toan = [
    "DFS", "BFS", "UCS", "DLS", "IDS", "IDS-DFS",
    "GREEDY", "ASTAR", "HILL", "SA",
    "BEAM", "GENETIC", "ANDOR",
    "BFS-BELIEF", "DFS-BELIEF-PARTIAL",
    "BACKTRACKING", "BACKTRACKING-FC", "AC3"
]

combo_thuat_toan = ttk.Combobox(
    khung_giua,
    values=ds_thuat_toan,
    font=("Arial", 13),
    state="readonly",
    width=20
)
combo_thuat_toan.set("Chọn thuật toán")
combo_thuat_toan.grid(row=1, column=0, sticky="w", pady=(0, 10))

btn_chay = tk.Button(
    khung_giua,
    text="▶️ Chạy thuật toán",
    font=("Arial", 14, "bold"),
    bg="#27ae60",
    fg="white",
    width=18,
    command=lambda: chuan_bi_va_chay(combo_thuat_toan.get())
)
btn_chay.grid(row=2, column=0, sticky="w", pady=(5, 10))

btn_dung = tk.Button(
    khung_giua,
    text="⏸️ Dừng",
    font=("Arial", 14, "bold"),
    bg="#e74c3c",
    fg="white",
    width=18,
    command=dung_auto
)
btn_dung.grid(row=3, column=0, sticky="w", pady=(2, 8))

btn_tiep_tuc = tk.Button(
    khung_giua,
    text="▶️ Tiếp tục",
    font=("Arial", 14, "bold"),
    bg="#16a085",
    fg="white",
    width=18,
    command=tiep_tuc
)
btn_tiep_tuc.grid(row=4, column=0, sticky="w", pady=(2, 8))

btn_reset = tk.Button(
    khung_giua,
    text="🔁 Reset",
    font=("Arial", 14, "bold"),
    bg="#9b59b6",
    fg="white",
    width=18,
    command=reset_banco
)
btn_reset.grid(row=5, column=0, sticky="w", pady=(2, 8))

btn_dich = tk.Button(
    khung_giua,
    text="🎯 Tạo đích mới",
    font=("Arial", 14, "bold"),
    bg="#2980b9",
    fg="white",
    width=18,
    command=tao_dich_moi
)
btn_dich.grid(row=6, column=0, sticky="w", pady=(2, 8))


#
frame_trangthai = tk.Frame(khung_chinh, bg="#1B2A41", bd=3, relief="ridge")
frame_trangthai.grid(
    row=1,
    column=1,
    columnspan=2,
    sticky="nsew",
    padx=(10, 20),
    pady=(350, 5)
)

# Tiêu đề khung
lbl_trangthai_title = tk.Label(
    frame_trangthai,
    text="📋  TRẠNG THÁI",
    font=("Arial", 14, "bold"),
    bg="#1B2A41",
    fg="#FFD700",
    anchor="w",
    padx=10
)
lbl_trangthai_title.pack(fill="x", pady=(5, 0))

# Frame chứa text + scrollbar
content_frame = tk.Frame(frame_trangthai, bg="#0F1A2B")
content_frame.pack(fill="both", expand=True, padx=5, pady=5)

scrollbar = tk.Scrollbar(content_frame)
scrollbar.pack(side="right", fill="y")

txt_trangthai = tk.Text(
    content_frame,
    height=10,
    width=20,
    bg="#0F1A2B",
    fg="white",
    font=("Consolas", 12),
    wrap="word",
    yscrollcommand=scrollbar.set,
    relief="flat"
)
txt_trangthai.pack(fill="both", expand=True, padx=5, pady=5)
scrollbar.config(command=txt_trangthai.yview)

# Thêm nội dung mặc định ban đầu
txt_trangthai.insert("end",
    "👋 Chào mừng đến với 8 Quân Xe!\n\n"
    "🧭  Nhiệm vụ: Tìm cách sắp xếp quân xe\n"
    "🧱  Tránh xung đột theo hàng và cột\n"
    "⚙️  Chọn thuật toán và bắt đầu hành trình!\n\n"
    "🎯 Chúc bạn may mắn!\n"
)
txt_trangthai.config(state="disabled")

def ghi_trangthai(noidung):
    txt_trangthai.config(state="normal")
    txt_trangthai.insert("end", noidung + "\n")
    txt_trangthai.see("end")
    txt_trangthai.config(state="disabled")
    root.update_idletasks()  


ve_dich()
ve_banco(canvas_trai, O_TRAI)

root.mainloop()
