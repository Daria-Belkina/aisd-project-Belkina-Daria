import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles


print("Загрузка датасета...")
df = pd.read_excel("amazon.xlsx")
print(f"Загружено {len(df)} заказов")


class MyHashMap:
    """Реализация хеш-таблицы для быстрого поиска по ключу."""

    MAX_LOAD_FACTOR = 0.75

    def __init__(self, capacity=8):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _index(self, key):
        """Вычисляет номер корзины для ключа."""

        return hash(key) % self.capacity

    def load_factor(self):
        """Возвращает текущую заполненность таблицы."""

        return self.size / self.capacity

    def _resize(self):
        """Увеличивает размер таблицы в 2 раза и перераспределяет элементы."""

        first_buckets = self.buckets
        self.capacity = self.capacity * 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in first_buckets:
            for key, value in bucket:
                self.put(key, value)

    def put(self, key, value):
        """Добавляет пару ключ-значение или обновляет существующую."""

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

    def get(self, key):
        """Ищет значение по ключу. Возвращает None, если ключа нет."""

        for k, value in self.buckets[self._index(key)]:
            if k == key:
                return value
        return None

    def __len__(self):
        """Позволяет узнать количество элементов через len()."""

        return self.size

hash_index = MyHashMap()
for row_id, order_id in enumerate(df["OrderID"]):
    hash_index.put(order_id, row_id)
print(f"Hash Index: {len(hash_index)} ключей")


class SegmentTree:
    """Дерево отрезков для быстрого поиска суммы на диапазоне."""

    def __init__(self, a):
        self.n = len(a)
        self.a = a[:]
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(1, 0, self.n - 1)

    def _build(self, v, tl, tr):
        """Рекурсивно строит дерево."""

        if tl == tr:
            self.tree[v] = self.a[tl]
        else:
            tm = (tl + tr) // 2
            self._build(2 * v, tl, tm)
            self._build(2 * v + 1, tm + 1, tr)
            self.tree[v] = self.tree[2 * v] + self.tree[2 * v + 1]

    def query(self, l, r):
        """Возвращает сумму элементов на отрезке [l, r]."""

        if self.n == 0:
            return 0
        l = max(0, l)
        r = min(self.n - 1, r)
        if l > r:
            return 0
        return self._query(1, 0, self.n - 1, l, r)

    def _query(self, v, tl, tr, l, r):
        """Вспомогательная рекурсивная функция для query."""

        if l > tr or r < tl:
            return 0
        if l <= tl and tr <= r:
            return self.tree[v]
        tm = (tl + tr) // 2
        return self._query(2 * v, tl, tm, l, r) + self._query(
            2 * v + 1, tm + 1, tr, l, r
        )

    def update(self, pos, value):
        """Обновляет значение элемента на позиции pos."""

        if self.n == 0 or pos < 0 or pos >= self.n:
            return
        self.a[pos] = value
        self._update(1, 0, self.n - 1, pos, value)

    def _update(self, v, tl, tr, pos, value):
        """Вспомогательная рекурсивная функция для update."""

        if tl == tr:
            self.tree[v] = value
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self._update(2 * v, tl, tm, pos, value)
            else:
                self._update(2 * v + 1, tm + 1, tr, pos, value)
            self.tree[v] = self.tree[2 * v] + self.tree[2 * v + 1]

df_sorted = df.sort_values("OrderDate").reset_index(drop=True)
values = df_sorted["TotalAmount"].fillna(0).values.tolist()
seg_tree = SegmentTree(values)
print(f"Segment Tree: {len(values)} элементов")


def binary_search_first(data, check_fn, left, right):
    """Ищет первый True (FFFF|TTTT) — то есть минимальный X."""

    if not check_fn(right, data):
        return None
    while left < right:
        mid = (left + right) // 2
        if check_fn(mid, data):
            right = mid
        else:
            left = mid + 1
    return left


total_amounts = df["TotalAmount"].fillna(0).values


df["DateOnly"] = df["OrderDate"].dt.date
daily_revenue = df.groupby("DateOnly")["TotalAmount"].sum().sort_index()

dates = daily_revenue.index.tolist()
revenues = daily_revenue.values.tolist()


prefix_revenue = [0]
for r in revenues:
    prefix_revenue.append(prefix_revenue[-1] + r)

print(f"Выручка по дням: {len(revenues)} уникальных дат")


app = FastAPI(title="QueryLab")

@app.get("/api/health")
def health():
    """Проверка работоспособности сервера."""

    return {"status": "ok", "rows": len(df)}


@app.post("/api/query/hash")
def query_hash(req: dict):
    """
    Поиск строки (row_id) в базе по уникальному ключу OrderID.
    Принимает JSON: {"key": "ORD0050001"}
    """

    key = req.get("key")
    if not key:
        raise HTTPException(
            status_code=400, detail="Поле 'key' обязательно"
        )

    row_id = hash_index.get(key)
    if row_id is None:
        raise HTTPException(status_code=404, detail="Ключ не найден")
    return {"row_id": row_id}


@app.post("/api/query/segment")
def query_segment(req: dict):
    """
    Сумма TotalAmount на диапазоне [L..R] (отсортировано по дате).
    Принимает JSON: {"l": 0, "r": 5000}
    """

    lower_bound = req.get("l")
    upper_bound = req.get("r")

    if lower_bound is None or upper_bound is None:
        raise HTTPException(status_code=400, detail="Нужны l и r")

    if lower_bound < 0 or upper_bound >= len(values) or lower_bound > upper_bound:
        raise HTTPException(
            status_code=400,
            detail=f"L и R в [0, {len(values) - 1}], L ≤ R",
        )
    return {"sum": round(seg_tree.query(lower_bound, upper_bound), 2)}


@app.post("/api/segment/update")
def segment_update(req: dict):
    """
    Изменяет значение TotalAmount по индексу в отсортированном массиве.
    Принимает JSON: {"pos": 100, "value": 9999.99}
    """

    pos = req.get("pos")
    value = req.get("value")
    if pos is None or value is None:
        raise HTTPException(status_code=400, detail="Нужны pos и value")
    if pos < 0 or pos >= len(values):
        raise HTTPException(status_code=400, detail="pos вне диапазона")

    seg_tree.update(pos, value)
    values[pos] = value
    return {"status": "ok", "new_value": value}


@app.post("/api/query/binary")
def query_binary(req: dict):
    """
    Бинарный поиск по ответу: находит минимальный порог TotalAmount,
    при котором количество заказов с суммой >= порога не превышает target.
    Принимает JSON: {"target": 5000}
    """

    target = req.get("target")
    if target is None:
        raise HTTPException(status_code=400, detail="Нужен target")
    if target >= len(total_amounts):
        raise HTTPException(
            status_code=400,
            detail=f"target должен быть < {len(total_amounts)}",
        )

    def check(x):
        return (total_amounts >= x).sum() <= target

    left_bound = int(min(total_amounts))
    right_bound = int(max(total_amounts))

    answer = binary_search_first(
        total_amounts, lambda x, d: check(x), left_bound, right_bound
    )

    if answer is None:
        raise HTTPException(status_code=404, detail="Решение не найдено")
    return {"answer": answer}


@app.get("/api/prefix")
def get_prefix(l: int, r: int):
    """
    Сумма выручки за период дней [L..R].
    Параметры l и r передаются в URL (query parameters).
    """

    if l < 0 or r >= len(revenues) or l > r:
        raise HTTPException(
            status_code=400,
            detail=f"L и R в [0, {len(revenues) - 1}], L ≤ R",
        )

    period_sum = prefix_revenue[r + 1] - prefix_revenue[l]

    d_start = dates[l].strftime("%Y-%m-%d")
    d_end = dates[r].strftime("%Y-%m-%d")

    return {
        "start_date": d_start,
        "end_date": d_end,
        "days_count": r - l + 1,
        "revenue": round(period_sum, 2),
    }



app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")