import tkinter as tk
import tkinter.font as tkfont
import random
from collections import deque
import heapq
import math
import random
from tkinter import messagebox

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

khung_giua_trai = tk.Frame(khung_giua, bg="#25324D")
khung_giua_trai.grid(row=0, column=0, padx=5, sticky="n")

khung_giua_phai = tk.Frame(khung_giua, bg="#25324D")
khung_giua_phai.grid(row=0, column=1, padx=5, sticky="n")

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

import random

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
from collections import deque

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




# ========= Nút điều khiển =========
btn_dfs = tk.Button(
    khung_giua_trai,
    text="DFS",
    font=("Arial", 14, "bold"),
    bg="#8e44ad",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("DFS")
)
btn_dfs.pack(pady=(10, 6))

btn_bfs = tk.Button(
    khung_giua_trai,
    text="BFS",
    font=("Arial", 14, "bold"),
    bg="#27ae60",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("BFS")
)
btn_bfs.pack(pady=1)

btn_ucs = tk.Button(
    khung_giua_trai,
    text="UCS",
    font=("Arial", 14, "bold"),
    bg="#f39c12",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("UCS")
)
btn_ucs.pack(pady=1)

btn_dls = tk.Button(
    khung_giua_trai,
    text="DLS",
    font=("Arial", 14, "bold"),
    bg="#16a085",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("DLS")
)
btn_dls.pack(pady=1)

btn_ids = tk.Button(
    khung_giua_trai,
    text="IDS",
    font=("Arial", 14, "bold"),
    bg="#2c3e50",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("IDS")
)
btn_ids.pack(pady=1)

btn_ids_dfs = tk.Button(
    khung_giua_trai,
    text="IDS-DFS",
    font=("Arial", 14, "bold"),
    bg="#9b59b6",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("IDS-DFS")
)
btn_ids_dfs.pack(pady=1)

btn_greedy = tk.Button(
    khung_giua_trai,
    text="Greedy",
    font=("Arial", 14, "bold"),
    bg="#d35400",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("GREEDY")
)
btn_greedy.pack(pady=1)

btn_astar = tk.Button(
    khung_giua_trai,
    text="A*",
    font=("Arial", 14, "bold"),
    bg="#27ae60",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("ASTAR")
)
btn_astar.pack(pady=1)

btn_hill = tk.Button(
    khung_giua_trai,
    text="Hill Climbing",
    font=("Arial", 14, "bold"),
    bg="#6c5ce7",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("HILL")
)
btn_hill.pack(pady=1)

btn_sa = tk.Button(
    khung_giua_phai,
    text="SA",
    font=("Arial", 14, "bold"),
    bg="#00cec9",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("SA")
)
btn_sa.pack(pady=1)

btn_beam = tk.Button(
    khung_giua_phai,
    text="Beam Search",
    font=("Arial", 14, "bold"),
    bg="#fd79a8",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("BEAM")
)
btn_beam.pack(pady=1)

btn_genetic = tk.Button(
    khung_giua_phai,
    text="Genetic",
    font=("Arial", 14, "bold"),
    bg="#e74c3c",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("GENETIC")
)
btn_genetic.pack(pady=1)

btn_andor = tk.Button(
    khung_giua_phai,
    text="AND-OR",
    font=("Arial", 14, "bold"),
    bg="#1abc9c",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("ANDOR")
)
btn_andor.pack(pady=1)

btn_bfs_belief = tk.Button(
    khung_giua_phai,
    text="BFS Belief",
    font=("Arial", 14, "bold"),
    bg="#74b9ff",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("BFS-BELIEF")
)
btn_bfs_belief.pack(pady=1)

btn_dfs_belief_partial = tk.Button(
    khung_giua_phai,
    text="DFS Belief Partial",
    font=("Arial", 14, "bold"),
    bg="#ff7675",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("DFS-BELIEF-PARTIAL")
)
btn_dfs_belief_partial.pack(pady=2)

btn_backtracking = tk.Button(
    khung_giua_phai,
    text="Backtracking",
    font=("Arial", 14, "bold"),
    bg="#27ae60",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("BACKTRACKING")
)
btn_backtracking.pack(pady=(10, 6))

btn_backtracking_fc = tk.Button(
    khung_giua_phai,
    text="Backtracking-FC",
    font=("Arial", 14, "bold"),
    bg="#e67e22",
    fg="white",
    width= 10,
    command=lambda: chuan_bi_va_chay("BACKTRACKING-FC")
)
btn_backtracking_fc.pack(pady=(2, 6))


btn_dung = tk.Button(
    khung_giua_phai,
    text="Dừng",
    font=("Arial", 14, "bold"),
    bg="#e74c3c",
    fg="white",
    width= 10,
    command=dung_auto
)
btn_dung.pack(pady=1)

btn_dich = tk.Button(
    khung_giua_phai,
    text="Tạo đích mới",
    font=("Arial", 14, "bold"),
    bg="#2980b9",
    fg="white",
    width= 10,
    command=tao_dich_moi
)
btn_dich.pack(pady=(6, 12))

ve_dich()
ve_banco(canvas_trai, O_TRAI)

root.mainloop()
