import pandas as pd
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles


print("Загрузка датасета...")
df = pd.read_excel("amazon.xlsx")
print(f"Загружено {len(df)} заказов")

# 1. СТРУКТУРЫ ДАННЫХ (HashMap, SegmentTree, MaxHeap)


class MyHashMap:
    """Хеш-таблица для быстрого поиска значения по ключу.

    Разрешает коллизии методом цепочек (каждая ячейка хранит список пар).
    Автоматически расширяется при превышении максимального коэффициента
    заполнения.
    Ищет полную информацию о заказе по его уникальному номеру (OrderID).
    """

    MAX_LOAD_FACTOR = 0.75

    def __init__(self, capacity: int = 8) -> None:
        """Инициализирует пустую хеш-таблицу заданной вместимости."""

        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _index(self, key: str) -> int:
        """Вычисляет индекс корзины для заданного ключа."""

        return hash(key) % self.capacity

    def _resize(self) -> None:
        """Увеличивает вместимость таблицы в 2 раза и перераспределяет данные."""

        first_buckets = self.buckets
        self.capacity = self.capacity * 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0

        for bucket in first_buckets:
            for key, value in bucket:
                self.put(key, value)

    def put(self, key: str, value: int) -> None:
        """Добавляет новую пару ключ-значение или обновляет существующую."""

        index = self._index(key)
        bucket = self.buckets[index]

        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        if (self.size + 1) / self.capacity > self.MAX_LOAD_FACTOR:
            self._resize()
            bucket = self.buckets[self._index(key)]

        bucket.append((key, value))
        self.size += 1

    def get(self, key: str) -> int | None:
        """Возвращает значение по ключу или None, если ключ не найден."""

        for k, value in self.buckets[self._index(key)]:
            if k == key:
                return value

        return None

    def __len__(self) -> int:
        """Возвращает текущее количество элементов в таблице."""

        return self.size


hash_index = MyHashMap()
for row_id, order_id in enumerate(df["OrderID"]):
    hash_index.put(order_id, row_id) # Потом, когда пользователь введет "ORD0050001", найдем row_id = 50000 и достанем строку из датасета через df.iloc[50000].


class SegmentTree:
    """Дерево отрезков для выполнения динамических запросов.

    Позволяет быстро (за O(log n)) находить сумму на отрезке и обновлять
    значение одного элемента.
    Считает сумму заказов на отрезке [L..R] и позволяет обновлять значение одного заказа (имитация возврата или скидки).
    """

    def __init__(self, a: list) -> None:
        """Инициализирует дерево отрезков на основе исходного массива."""

        self.n = len(a)
        self.a = a[:]
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1)

    def _build(self, v: int, tl: int, tr: int) -> None:
        """Рекурсивно строит дерево отрезков."""

        if tl == tr:
            self.tree[v] = self.a[tl]
        else:
            tm = (tl + tr) // 2
            self._build(2 * v, tl, tm)
            self._build(2 * v + 1, tm + 1, tr)
            self.tree[v] = self.tree[2 * v] + self.tree[2 * v + 1]

    def query(self, l: int, r: int) -> float:
        """Возвращает сумму элементов на отрезке [l, r] и проверяет введенные данные на адекватность."""

        if self.n == 0:
            return 0

        l = max(0, l)
        r = min(self.n - 1, r)

        if l > r:
            return 0

        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, v: int, tl: int, tr: int, l: int, r: int) -> float:
        """Вспомогательный рекурсивный метод для поиска суммы."""

        if l > tr or r < tl:
            return 0

        if l <= tl and tr <= r:
            return self.tree[v]

        tm = (tl + tr) // 2

        return self._query(2 * v, tl, tm, l, r) + self._query(
            2 * v + 1, tm + 1, tr, l, r
        )

    def update(self, pos: int, value: float) -> None:
        """Обновляет значение элемента на позиции pos."""

        if self.n == 0 or pos < 0 or pos >= self.n:
            return

        self.a[pos] = value
        self._update(1, 0, self.n - 1, pos, value)

    def _update(self, v: int, tl: int, tr: int, pos: int, value: float) -> None:
        """Вспомогательный рекурсивный метод для обновления."""

        if tl == tr:
            self.tree[v] = value
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self._update(2 * v, tl, tm, pos, value)
            else:
                self._update(2 * v + 1, tm + 1, tr, pos, value)
            self.tree[v] = self.tree[2 * v] + self.tree[2 * v + 1]


class MaxHeap:
    """Бинарная куча на максимум.

    Позволяет быстро (за O(log n)) добавлять элементы и извлекать
    элемент с максимальным значением.
    Топ-10 самых дорогих заказов.
    """

    def __init__(self) -> None:
        """Инициализирует пустую кучу."""
        self.heap = []

    def _parent(self, i: int) -> int:
        return (i - 1) // 2

    def _left_child(self, i: int) -> int:
        return 2 * i + 1

    def _right_child(self, i: int) -> int:
        return 2 * i + 2

    def _swap(self, i: int, j: int) -> None:
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def push(self, value: tuple) -> None:
        """Добавляет элемент и 'поднимает' его на нужное место."""

        self.heap.append(value)
        current = len(self.heap) - 1

        while current > 0 and self.heap[current][0] > self.heap[self._parent(current)][0]:
            self._swap(current, self._parent(current))
            current = self._parent(current)

    def pop(self) -> tuple | None:
        """Извлекает элемент с максимальным значением."""

        if not self.heap:
            return None

        if len(self.heap) == 1:
            return self.heap.pop()

        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()

        self._sift_down(0)

        return max_val

    def _sift_down(self, i: int) -> None:
        max_index = i
        left = self._left_child(i)
        right = self._right_child(i)
        size = len(self.heap)

        if left < size and self.heap[left][0] > self.heap[max_index][0]:
            max_index = left

        if right < size and self.heap[right][0] > self.heap[max_index][0]:
            max_index = right

        if i != max_index:
            self._swap(i, max_index)
            self._sift_down(max_index)

    def __len__(self) -> int:
        return len(self.heap)


df_sorted = df.sort_values("OrderDate").reset_index(drop=True)
values = df_sorted["TotalAmount"].fillna(0).values.tolist()
seg_tree = SegmentTree(values)


def binary_search_first(data, check_fn, left: int, right: int) -> int | None:
    """Ищет минимальный X, при котором check_fn(X) возвращает True.

    Ожидает монотонность вида FFFF | TTTT.
    Находит минимальную сумму выручки, при которой планку преодолеют ровно N брендов.
    """

    if not check_fn(right, data):
        return None

    while left < right:
        mid = (left + right) // 2

        if check_fn(mid, data):
            right = mid
        else:
            left = mid + 1

    return left

# 2. ПОДГОТОВКА ДАННЫХ ДЛЯ АНАЛИЗА

brand_revenues = df.groupby("Brand")["TotalAmount"].sum().sort_values(ascending=False)
brand_names = brand_revenues.index.tolist()
brand_values = brand_revenues.values

df["DateOnly"] = df["OrderDate"].dt.date
daily_revenue = df.groupby("DateOnly")["TotalAmount"].sum().sort_index()
dates = daily_revenue.index.tolist()
revenues = daily_revenue.values.tolist()

prefix_revenue = [0]
for rev in revenues:
    prefix_revenue.append(prefix_revenue[-1] + rev)


# 3. ПРИЛОЖЕНИЕ И ЭНДПОИНТЫ


app = FastAPI(title="Amazon Analytics")


@app.get("/api/stats/{group_by}")
def get_stats(group_by: str) -> list:
    """Возвращает топ-20 групп по выручке (продукты, категории, бренды).

    Args:
        group_by: Строковый идентификатор группировки.
    """

    valid_columns = {
        "product": "ProductName",
        "category": "Category",
        "brand": "Brand",
        "payment": "PaymentMethod",
        "country": "Country",
    }

    if group_by not in valid_columns:
        raise HTTPException(status_code=400, detail="Неверная группировка")

    col = valid_columns[group_by]
    grouped = df.groupby(col)["TotalAmount"].sum().sort_values(ascending=False)
    total_sum = grouped.sum()

    result = []

    for name, amount in grouped.items():
        result.append(
            {
                "name": str(name),
                "sum": round(float(amount), 2),
                "percent": round(float((amount / total_sum) * 100), 2),
            }
        )

    return result[:20]


@app.get("/api/statuses")
def get_statuses() -> list:
    """Возвращает статистику заказов по их статусам (кол-во и сумма)."""

    grouped = df.groupby("OrderStatus").agg(count=("OrderID", "count"), sum=("TotalAmount", "sum")).reset_index()

    result = []

    for _, row in grouped.iterrows():
        result.append({
            "status": row["OrderStatus"],
            "count": int(row["count"]),
            "sum": round(float(row["sum"]), 2),
        })

    return result


@app.post("/api/query/hash")
def query_hash(req: dict) -> dict:
    """Ищет полную информацию о заказе по его OrderID через HashMap."""

    key = req.get("key")

    if not key:
        raise HTTPException(status_code=400, detail="Нужен ключ-айди заказа")

    row_id = hash_index.get(key)

    if row_id is None:
        raise HTTPException(status_code=404, detail="Заказ не найден")

    row_data = df.iloc[row_id].to_dict()
    row_data["OrderDate"] = str(row_data["OrderDate"])

    return row_data


@app.post("/api/query/segment")
def query_segment(req: dict) -> dict:
    """Вычисляет сумму заказов на диапазоне [L, R] через Segment Tree."""

    lower_bound = req.get("l")
    upper_bound = req.get("r")

    if lower_bound is None or upper_bound is None:
        raise HTTPException(status_code=400, detail="Нужны l и r границы")

    if lower_bound < 0 or upper_bound >= len(values) or lower_bound > upper_bound:
        raise HTTPException(status_code=400, detail="Неверный диапазон")

    d_start = df_sorted.iloc[lower_bound]["OrderDate"].strftime("%Y-%m-%d")
    d_end = df_sorted.iloc[upper_bound]["OrderDate"].strftime("%Y-%m-%d")

    return {
        "sum": round(seg_tree.query(lower_bound, upper_bound), 2),
        "start": d_start,
        "end": d_end,
    }


@app.post("/api/segment/update")
def segment_update(req: dict) -> dict:
    """Обновляет сумму заказа на конкретной позиции (возврат/скидка)."""

    pos = req.get("pos")
    value = req.get("value")

    if pos is None or value is None:
        raise HTTPException(status_code=400, detail="Нужны pos и value")

    if pos < 0 or pos >= len(values):
        raise HTTPException(status_code=400, detail="Неверная позиция")

    seg_tree.update(pos, value)
    values[pos] = value

    return {"status": "ok", "new_value": value}


@app.get("/api/prefix")
def get_prefix(l: int, r: int) -> dict:
    """Вычисляет выручку за период дней [L, R] через Prefix Sum."""

    if l < 0 or r >= len(revenues) or l > r:
        raise HTTPException(status_code=400, detail="Неверный диапазон дней")

    period_sum = prefix_revenue[r + 1] - prefix_revenue[l]
    return {
        "start_date": dates[l].strftime("%Y-%m-%d"),
        "end_date": dates[r].strftime("%Y-%m-%d"),
        "days_count": r - l + 1,
        "revenue": round(period_sum, 2),
    }


@app.post("/api/query/brand-plan")
def query_brand_plan(req: dict) -> dict:
    """Находит минимальный план по выручке для N лучших брендов.

    Использует бинарный поиск по ответу. Возвращает сумму плана и список
    брендов, которые его преодолели.
    """

    target = req.get("target")

    if target is None:
        raise HTTPException(status_code=400, detail="Нужен target")

    if target < 1 or target > len(brand_values):
        raise HTTPException(
            status_code=400,
            detail=f"target должен быть от 1 до {len(brand_values)}",
        )

    def check(x: float) -> bool:
        return (brand_values >= x).sum() <= target

    left_bound = 0
    right_bound = int(max(brand_values))

    answer = binary_search_first(
        brand_values, lambda x, d: check(x), left_bound, right_bound
    )

    if answer is None:
        raise HTTPException(status_code=404, detail="Решение не найдено")

    winners = []
    for name, revenue in zip(brand_names, brand_values):
        if revenue >= answer:
            winners.append({"name": name, "revenue": round(float(revenue), 2)})

    return {"plan": answer, "count": len(winners), "winners": winners}


@app.get("/api/pagerank")
def get_pagerank() -> dict:
    """Строит граф (Клиент <-> Бренд) и вычисляет PageRank.

    Возвращает топ-10 клиентов и топ-10 брендов.
    Находит самых «влиятельных» клиентов и брендов на основе графа покупок.
    """

    G = nx.DiGraph()

    # Берем топ-200 клиентов по количеству заказов
    top_customers = df["CustomerID"].value_counts().head(200).index
    sub_df = df[df["CustomerID"].isin(top_customers)]

    edges = sub_df.groupby(["CustomerID", "Brand"]).size().reset_index(name="weight")

    for _, row in edges.iterrows():
        G.add_edge(row["CustomerID"], row["Brand"], weight=row["weight"])
        G.add_edge(row["Brand"], row["CustomerID"], weight=row["weight"])

    scores = nx.pagerank(G, alpha=0.85, weight="weight")

    customers_scores = {k: v for k, v in scores.items() if k.startswith("CUST")}
    brands_scores = {k: v for k, v in scores.items() if not k.startswith("CUST")}

    top_cust = sorted(customers_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top_brands = sorted(brands_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "customers": [{"name": k, "score": round(v, 6)} for k, v in top_cust],
        "brands": [{"name": k, "score": round(v, 6)} for k, v in top_brands]
    }


@app.get("/api/top-orders")
def get_top_orders() -> list:
    """Использует бинарную кучу (MaxHeap) для поиска 10 самых дорогих заказов."""

    max_heap = MaxHeap()

    for _, row in df.iterrows():
        if row["TotalAmount"] > 0:
            max_heap.push((row["TotalAmount"], row["OrderID"]))

    result = []
    for _ in range(min(10, len(max_heap))):
        amount, order_id = max_heap.pop()
        result.append({"order_id": order_id, "amount": round(float(amount), 2)})

    return result


@app.get("/api/weekday-stats")
def get_weekday_stats() -> list:
    """Возвращает выручку по дням недели (Пн-Вс)."""

    temp = df.copy()
    temp["DayName"] = temp["OrderDate"].dt.day_name()

    daily = temp.groupby("DayName")["TotalAmount"].sum()

    ru_days = {
        "Monday": "Понедельник", "Tuesday": "Вторник", "Wednesday": "Среда",
        "Thursday": "Четверг", "Friday": "Пятница", "Saturday": "Суббота",
        "Sunday": "Воскресенье"
    }

    result = []
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for day in order:
        if day in daily.index:
            result.append({
                "day": ru_days[day],
                "sum": round(float(daily[day]), 2)
            })

    return result


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")