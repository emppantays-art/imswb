"""
test_product_search.py — trigram fuzzy product search.

Run:  uv run python test_product_search.py
"""

import sys, time, random, string
sys.path.insert(0, ".")
from product_search import fuzzy_search, TrigramIndex

PASS = "\033[92mPASS\033[0m"; FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {PASS if cond else FAIL}  {name}" + (f"  — {detail}" if detail and not cond else ""))

PRODUCTS = [{"name": n} for n in [
    "Apple", "Apple Pie", "Pineapple", "Banana", "Orange",
    "Grape Juice", "Green Apple", "Mango", "Blueberry Muffin", "Almond Milk",
]]

# 1. exact match ranks first
hits = fuzzy_search("Apple", PRODUCTS)
check("exact match ranks first", hits and hits[0]["name"] == "Apple", str([h['name'] for h in hits[:3]]))

# 2. typo tolerance: 'aple' finds Apple
hits = fuzzy_search("aple", PRODUCTS)
names = [h["name"] for h in hits]
check("typo 'aple' finds Apple", "Apple" in names, str(names[:3]))

# 3. typo 'banan' finds Banana
hits = fuzzy_search("banan", PRODUCTS)
check("typo 'banan' finds Banana", hits and hits[0]["name"] == "Banana", str([h['name'] for h in hits[:3]]))

# 4. prefix match: 'gr' returns Grape/Green ahead of unrelated
hits = fuzzy_search("green", PRODUCTS)
check("'green' finds Green Apple", hits and hits[0]["name"] == "Green Apple", str([h['name'] for h in hits[:3]]))

# 5. substring: 'milk' finds Almond Milk
hits = fuzzy_search("milk", PRODUCTS)
check("substring 'milk' finds Almond Milk", any(h["name"] == "Almond Milk" for h in hits))

# 6. no match returns empty (not garbage)
hits = fuzzy_search("xyzzy", PRODUCTS)
check("nonsense query returns nothing", hits == [], str([h['name'] for h in hits]))

# 7. short query (<3 chars) still works via substring sweep
hits = fuzzy_search("ma", PRODUCTS)
check("short query 'ma' finds Mango/Muffin", any(h["name"] in ("Mango", "Blueberry Muffin") for h in hits),
      str([h['name'] for h in hits]))

# 8. limit respected
hits = fuzzy_search("a", PRODUCTS, limit=3)
check("limit respected", len(hits) <= 3, f"got {len(hits)}")

# 9. ranking: 'apple' puts 'Apple' before 'Pineapple'/'Apple Pie'
hits = fuzzy_search("apple", PRODUCTS)
order = [h["name"] for h in hits]
check("'apple' ranks 'Apple' first", order and order[0] == "Apple", str(order[:4]))

# 10. SPEED: large catalog, index reused — query stays fast
N = 20000
big = [{"name": "".join(random.choices(string.ascii_lowercase, k=8))} for _ in range(N)]
big.append({"name": "Special Widget"})
t0 = time.time(); idx = TrigramIndex(big, "name"); build_ms = (time.time()-t0)*1000
t0 = time.time()
for _ in range(100):
    hits = idx.search("widget")
query_ms = (time.time()-t0)*1000/100
found = any(h["name"] == "Special Widget" for h in hits)
check(f"speed: {N} items, query {query_ms:.2f} ms/search (build {build_ms:.0f} ms), found target",
      query_ms < 10 and found, f"{query_ms:.2f} ms, found={found}")
print(f"         → built index of {N} products in {build_ms:.0f} ms; ~{query_ms:.2f} ms per search")

print()
p = sum(1 for _, ok in results if ok)
print("="*58)
print(f"  \033[92m{p} passed\033[0m   \033[91m{len(results)-p} failed\033[0m   / {len(results)} total")
print("="*58)
sys.exit(len(results)-p)
