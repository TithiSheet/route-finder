"""
Q-Learning Route Finder — Delhi/Gurgaon Ride Bookings
======================================================
CHANGE ONLY THESE TWO LINES BELOW:
"""

START_CITY = "Samaypur Badli"   # <-- Change me!
GOAL_CITY  = "Cyber Hub"        # <-- Change me!

"""
Available cities:
  AIIMS, Ashram, Barakhamba Road, Basai Dhankot, Cyber Hub,
  Dwarka Sector 21, Gurgaon Railway Station, Hauz Khas,
  Kanhaiya Nagar, Khandsa, Lok Kalyan Marg, Madipur,
  Mehrauli, Moolchand, Narsinghpur, Nawada, Nehru Place,
  New Colony, Punjabi Bagh, Saket, Samaypur Badli,
  Sarai Kale Khan, Shastri Nagar, Tilak Nagar, Udyog Vihar
"""

# ══════════════════════════════════════════════════════════════
#  DO NOT EDIT BELOW THIS LINE
# ══════════════════════════════════════════════════════════════

import random, time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
from collections import Counter

random.seed(42)
np.random.seed(42)

# ── 1. LOAD DATA ──────────────────────────────────────────────
df_raw   = pd.read_csv("bookings__3_.csv")
top_locs = (pd.concat([df_raw["Pickup Location"], df_raw["Drop Location"]])
              .value_counts().head(20).index.tolist())

df_all = (df_raw[df_raw["Pickup Location"].isin(top_locs) &
                 df_raw["Drop Location"].isin(top_locs)]
          .groupby(["Pickup Location", "Drop Location"])["Ride Distance"]
          .mean().reset_index()
          .rename(columns={"Pickup Location": "Origin",
                            "Drop Location":   "Destination",
                            "Ride Distance":   "Distance"}))

# Sparse graph: keep edges ≤ median distance per origin
median_per_origin = df_all.groupby("Origin")["Distance"].median()
df = df_all[df_all.apply(
        lambda r: r["Distance"] <= median_per_origin[r["Origin"]], axis=1
    )].reset_index(drop=True)

cities      = sorted(set(df["Origin"]).union(set(df["Destination"])))
city_to_idx = {city: i for i, city in enumerate(cities)}
idx_to_city = {i: city for city, i in city_to_idx.items()}
n           = len(cities)

# Validate inputs
assert START_CITY in city_to_idx, f"'{START_CITY}' not found! Available: {cities}"
assert GOAL_CITY  in city_to_idx, f"'{GOAL_CITY}' not found! Available: {cities}"
assert START_CITY != GOAL_CITY,   "Start and Goal must be different!"

print(f"Locations loaded : {n}  |  Route pairs : {len(df)}")
print(f"Route            : {START_CITY}  →  {GOAL_CITY}")

# ── 2. DISTANCE MATRIX ────────────────────────────────────────
base_dm = np.full((n, n), np.inf)
np.fill_diagonal(base_dm, 0)
for _, row in df.iterrows():
    i, j = city_to_idx[row["Origin"]], city_to_idx[row["Destination"]]
    base_dm[i, j] = row["Distance"]
    base_dm[j, i] = row["Distance"]

# ── 3. DYNAMIC EVENTS ─────────────────────────────────────────
EVENTS = {
    "CLEAR":    {"color": "#A8D5A2", "cost_mult": 1.0,    "label": "Clear Route"},
    "WEATHER":  {"color": "#6BAED6", "cost_mult": 1.5,    "label": "Bad Weather (+50%)"},
    "BLOCKAGE": {"color": "#CC0000", "cost_mult": np.inf, "label": "Route Blocked"},
}

def generate_events(df, blockage_prob=0.15, event_prob=0.30, seed=42):
    random.seed(seed)
    emap = {}
    for _, row in df.iterrows():
        r    = random.random()
        edge = (row["Origin"], row["Destination"])
        if   r < blockage_prob:
            emap[edge] = "BLOCKAGE"
        elif r < blockage_prob + event_prob:
            emap[edge] = "WEATHER"
        else:
            emap[edge] = "CLEAR"
    return emap

event_map  = generate_events(df)
evt_counts = Counter(event_map.values())

dyn_dm = deepcopy(base_dm)
for (o, d), evt in event_map.items():
    if o in city_to_idx and d in city_to_idx:
        mult = EVENTS[evt]["cost_mult"]
        i, j = city_to_idx[o], city_to_idx[d]
        dyn_dm[i, j] = np.inf if mult == np.inf else base_dm[i, j] * mult
        dyn_dm[j, i] = np.inf if mult == np.inf else base_dm[j, i] * mult

# ── 4. Q-LEARNING ─────────────────────────────────────────────
def train_q_learning(sc, gc, dm, episodes=10000, alpha=0.8, gamma=0.95, eps0=0.4):
    s, g = city_to_idx[sc], city_to_idx[gc]
    Q    = np.zeros((n, n))

    def valid_actions(x):
        return [i for i in range(n) if dm[x, i] != np.inf and i != x]

    for ep in range(episodes):
        state   = s
        steps   = 0
        epsilon = max(0.01, eps0 * (1 - ep / episodes))
        while state != g and steps < 200:
            steps += 1
            acts  = valid_actions(state)
            if not acts:
                break
            action   = random.choice(acts) if random.random() < epsilon \
                       else max(acts, key=lambda a: Q[state, a])
            reward   = 5000 if action == g else -dm[state, action] / 50
            future_q = max([Q[action, a] for a in valid_actions(action)], default=0)
            Q[state, action] += alpha * (reward + gamma * future_q - Q[state, action])
            state = action
    return Q

def extract_path(Q, sc, gc, dm):
    s, g    = city_to_idx[sc], city_to_idx[gc]
    state   = s
    path    = [s]
    visited = {s}

    def valid_actions(x):
        return [i for i in range(n) if dm[x, i] != np.inf and i != x]

    steps = 0
    while state != g and steps < 200:
        steps += 1
        acts  = valid_actions(state)
        if not acts:
            break
        pool   = [a for a in acts if a not in visited] or acts
        action = max(pool, key=lambda a: Q[state, a])
        path.append(action)
        visited.add(action)
        state = action
    return [idx_to_city[i] for i in path]

# ── 5. TRAIN & FIND PATH ──────────────────────────────────────
print("\nTraining Q-Learning agent...")
t0      = time.time()
Q_agent = train_q_learning(START_CITY, GOAL_CITY, dyn_dm)
path    = extract_path(Q_agent, START_CITY, GOAL_CITY, dyn_dm)
km      = sum(base_dm[city_to_idx[path[i]], city_to_idx[path[i+1]]]
              for i in range(len(path)-1))
print(f"Trained in       : {time.time()-t0:.1f}s")
print(f"Q-Agent Path     : {' → '.join(path)}")
print(f"Total Distance   : {km:.2f} km")
print(f"Hops             : {len(path)-1}")
print(f"Event Summary    : {dict(evt_counts)}")

# ── 6. VISUALISE ──────────────────────────────────────────────
path_set = set(path)
G = nx.Graph()
for city in cities:
    G.add_node(city)
for _, row in df.iterrows():
    G.add_edge(row["Origin"], row["Destination"], weight=row["Distance"])

pos = nx.spring_layout(G, seed=170, k=2.5)
ped = list(zip(path, path[1:]))

fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor("#0d0d1a")
ax.set_facecolor("#0d0d1a")

# Draw all edges coloured by event
for u, v in G.edges():
    evt   = event_map.get((u, v), event_map.get((v, u), "CLEAR"))
    color = EVENTS[evt]["color"]
    width = 3.5   if evt == "BLOCKAGE" else 1.4
    style = "dashed" if evt == "BLOCKAGE" else "solid"
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                           width=width, edge_color=color,
                           style=style, alpha=0.80, ax=ax)

# Highlight optimal path in bright green
nx.draw_networkx_edges(G, pos, edgelist=ped,
                       width=7, edge_color="#00FF7F", ax=ax)

# Distance labels on path edges only
edge_labels = {(u, v): f"{base_dm[city_to_idx[u], city_to_idx[v]]:.1f} km"
               for u, v in ped}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                              font_color="#FFFFFF", font_size=7, ax=ax)

# Node colours & sizes
nc = ["#00FF00" if c == START_CITY else
      "#FF3333" if c == GOAL_CITY  else
      "#00BFFF" if c in path_set   else
      "#2d2d50" for c in G.nodes()]
ns = [900 if c in path_set else 450 for c in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=ns,
                       ax=ax, edgecolors="white", linewidths=0.8)
nx.draw_networkx_labels(G, pos, font_size=8,
                        font_color="white", font_weight="bold", ax=ax)

ax.set_title(
    f"DYNAMIC RIDE ENVIRONMENT  |  {START_CITY}  →  {GOAL_CITY}\n"
    f"Q-Agent Path: {' → '.join(path)}  |  {km:.2f} km  |  {len(path)-1} hops",
    fontsize=13, color="white", fontweight="bold", pad=16)
ax.axis("off")

ax.legend(handles=[
    mpatches.Patch(color="#CC0000", label=f"Route Blocked  ({evt_counts.get('BLOCKAGE', 0)}) – Impassable"),
    mpatches.Patch(color="#6BAED6", label=f"Bad Weather    ({evt_counts.get('WEATHER',  0)}) – +50% cost"),
    mpatches.Patch(color="#A8D5A2", label=f"Clear Route    ({evt_counts.get('CLEAR',    0)})"),
    mpatches.Patch(color="#00FF7F", label=f"Q-Agent Path   ({km:.2f} km)"),
    mpatches.Patch(color="#00FF00", label=f"Start: {START_CITY}"),
    mpatches.Patch(color="#FF3333", label=f"Goal:  {GOAL_CITY}"),
], loc="lower left", fontsize=9,
   facecolor="#1a1a3a", edgecolor="white", labelcolor="white")

plt.tight_layout()
outfile = f"ql_{START_CITY.replace(' ','_')}_to_{GOAL_CITY.replace(' ','_')}.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
plt.show()
print(f"\nChart saved → {outfile}")
