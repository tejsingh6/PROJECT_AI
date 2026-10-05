"""
Smart Parking Assistant - starter project
Stack: Streamlit + Pandas + NumPy + ChromaDB

Run:  streamlit run app.py
"""
import re
from datetime import datetime

import chromadb
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Parking Assistant", page_icon="🅿️", layout="wide")

ZONES = {"A": 1, "B": 2, "C": 3, "D": 4}  # zone -> distance rank from the gate
SLOTS_PER_ZONE = 10
BASE_RATE = 20  # price per hour (INR)


# ----------------------------------------------------------------------------
# 1. Synthetic data (pandas + numpy)
# ----------------------------------------------------------------------------
@st.cache_data
def generate_history(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Simulate hourly occupancy per zone with morning and evening peaks."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in pd.date_range(end=datetime.now().date(), periods=days):
        weekend = day.weekday() >= 5
        for hour in range(24):
            morning = np.exp(-((hour - 9) ** 2) / 8)
            evening = np.exp(-((hour - 18) ** 2) / 8)
            base = 0.25 + 0.6 * (morning + evening) * (0.6 if weekend else 1.0)
            for zone, rank in ZONES.items():
                # closer zones fill up faster
                occ = base * (1.15 - 0.1 * rank) + rng.normal(0, 0.05)
                rows.append((day, hour, zone, float(np.clip(occ, 0, 1))))
    return pd.DataFrame(rows, columns=["date", "hour", "zone", "occupancy"])


def predict_occupancy(history: pd.DataFrame) -> pd.DataFrame:
    """Naive forecast: average occupancy per zone and hour of day."""
    return history.groupby(["zone", "hour"], as_index=False)["occupancy"].mean()


def dynamic_price(occupancy: float) -> float:
    """Price rises with predicted demand (up to +100%)."""
    return round(BASE_RATE * (1 + occupancy), 2)


# ----------------------------------------------------------------------------
# 2. Slot state (kept in st.session_state so it survives Streamlit reruns)
# ----------------------------------------------------------------------------
def init_slots(forecast: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    now_hour = datetime.now().hour
    rows = []
    for zone, rank in ZONES.items():
        occ = forecast[(forecast.zone == zone) & (forecast.hour == now_hour)].occupancy.iloc[0]
        for i in range(1, SLOTS_PER_ZONE + 1):
            slot_type = "EV" if i <= 2 else ("Accessible" if i == 3 else "Regular")
            status = "occupied" if rng.random() < occ else "free"
            rows.append(
                {
                    "slot_id": f"{zone}{i}",
                    "zone": zone,
                    "type": slot_type,
                    "distance": rank * 50 + i * 3,  # metres from gate
                    "status": status,
                    "plate": "",
                }
            )
    return pd.DataFrame(rows)


def recommend_slot(slots: pd.DataFrame, need: str) -> pd.Series | None:
    """Closest free slot that matches the vehicle's needs."""
    free = slots[(slots.status == "free") & (slots.type == need)]
    if free.empty:
        return None
    return free.sort_values("distance").iloc[0]


# ----------------------------------------------------------------------------
# 3. ChromaDB assistant (semantic FAQ search)
# ----------------------------------------------------------------------------
FAQS = [
    "EV charging slots are in every zone at positions 1 and 2. They cost the same as regular slots.",
    "Zone A is closest to Gate 1 and fills up fastest, especially around 9 AM and 6 PM.",
    "Accessible parking is at position 3 in each zone and is reserved for permit holders.",
    "Parking is charged per hour. Prices go up when the lot is expected to be busy.",
    "To cancel a booking, open the Manage tab and release your slot using the slot ID.",
    "The lot is open 24 hours. The quietest time is usually between midnight and 6 AM.",
    "Overstaying your booked time is charged at 1.5x the current hourly rate.",
]


@st.cache_resource
def get_collection():
    client = chromadb.Client()  # in-memory; use PersistentClient("db/") to keep data
    col = client.get_or_create_collection("parking_faq")
    if col.count() == 0:
        col.add(documents=FAQS, ids=[f"faq{i}" for i in range(len(FAQS))])
    return col



# ----------------------------------------------------------------------------
# 3b. Assistant brain: live stats + memory of previous questions
# ----------------------------------------------------------------------------
INTENT_WORDS = {
    "price": ["price", "cost", "rate", "how much", "fee"],
    "ev": ["ev ", "ev?", "electric", "charging"],
    "free": ["free", "available", "empty", "vacant", "space left"],
    "booked": ["booked", "booking", "reserved"],
    "occupied": ["vehicle", "car", "parked", "occupied", "full", "how many"],
}


def detect_intent(ql: str):
    ql = ql + " "
    for intent, words in INTENT_WORDS.items():
        if any(w in ql for w in words):
            return intent
    return None


def answer_question(q: str, slots: pd.DataFrame, forecast: pd.DataFrame, hour: int) -> str:
    ql = q.lower().strip()
    chat = st.session_state.chat
    prev_questions = [m["content"] for m in chat if m["role"] == "user"][:-1]  # excludes current

    # Memory: recall earlier questions
    if any(p in ql for p in ["previous question", "earlier", "asked before", "what did i ask", "my questions", "last question"]):
        if not prev_questions:
            return "You haven't asked me anything before this."
        return "Here is what you asked me so far:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(prev_questions, 1))

    zm = re.search(r"\bzone\s*([a-d])\b", ql)
    zone = zm.group(1).upper() if zm else None
    intent = detect_intent(ql)
    ctx = st.session_state.chat_ctx

    # Memory: follow-up like "what about zone B?" reuses the last topic
    if intent is None and zone and ctx.get("intent"):
        intent = ctx["intent"]
    if zone is None and intent and intent == ctx.get("intent") and re.search(r"\b(there|that zone|same)\b", ql):
        zone = ctx.get("zone")

    if intent:
        st.session_state.chat_ctx = {"intent": intent, "zone": zone}
        df = slots if zone is None else slots[slots.zone == zone]
        scope = "in the whole lot" if zone is None else f"in Zone {zone}"
        if intent == "occupied":
            occ = int((df.status == "occupied").sum())
            bkd = int((df.status == "booked").sum())
            return f"**{occ + bkd} vehicles** are parked {scope} ({occ} walk-in, {bkd} booked) out of {len(df)} slots."
        if intent == "free":
            return f"**{int((df.status == 'free').sum())} free slots** {scope} out of {len(df)}."
        if intent == "booked":
            return f"**{int((df.status == 'booked').sum())} slots** are booked {scope}."
        if intent == "ev":
            n = int(((df.status == "free") & (df.type == "EV")).sum())
            return f"**{n} EV charging slots** are free {scope}."
        if intent == "price":
            zones = [zone] if zone else list(ZONES)
            parts = []
            for z in zones:
                o = forecast[(forecast.zone == z) & (forecast.hour == hour)].occupancy.iloc[0]
                parts.append(f"Zone {z}: ₹{dynamic_price(o)}/hr")
            return "Current rates - " + ", ".join(parts)

    # Fallback: semantic search; short follow-ups borrow context from the last question
    query = q
    if len(q.split()) <= 3 and prev_questions:
        query = prev_questions[-1] + " " + q
    res = get_collection().query(query_texts=[query], n_results=1)
    return res["documents"][0][0]

# ----------------------------------------------------------------------------
# 4. UI
# ----------------------------------------------------------------------------
history = generate_history()
forecast = predict_occupancy(history)

if "slots" not in st.session_state:
    st.session_state.slots = init_slots(forecast)
if "bookings" not in st.session_state:
    st.session_state.bookings = []
if "chat" not in st.session_state:
    st.session_state.chat = []  # [{"role": "user"/"assistant", "content": str}]
if "chat_ctx" not in st.session_state:
    st.session_state.chat_ctx = {}  # remembers the last topic for follow-ups

slots = st.session_state.slots
now_hour = datetime.now().hour

st.title("🅿️ Smart Parking Assistant")

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

free_count = int((slots.status == "free").sum())
parked_count = int((slots.status == "occupied").sum())
booked_count = int((slots.status == "booked").sum())
vehicles_total = parked_count + booked_count
avg_pred = forecast[forecast.hour == now_hour].occupancy.mean()

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total slots", len(slots))
m2.metric("Vehicles parked", vehicles_total)
m3.metric("Free slots", free_count)
m4.metric("Booked", booked_count)
m5.metric("Occupancy", f"{vehicles_total / len(slots):.0%}")
m6.metric("Price / hr", f"₹{dynamic_price(avg_pred)}")

tab_map, tab_book, tab_manage, tab_insights, tab_chat = st.tabs(
    ["Live map", "Book", "Manage", "Insights", "Assistant"]
)

# --- Live map ---------------------------------------------------------------
with tab_map:
    icons = {"free": "🟩", "occupied": "🟥", "booked": "🟦"}
    st.caption("🟩 free  🟥 occupied  🟦 booked")
    for zone in ZONES:
        z = slots[slots.zone == zone]
        st.markdown(f"**Zone {zone}**")
        cols = st.columns(SLOTS_PER_ZONE)
        for col, (_, s) in zip(cols, z.iterrows()):
            col.markdown(f"{icons[s.status]}<br><small>{s.slot_id}<br>{s.type[:3]}</small>",
                         unsafe_allow_html=True)

# --- Book ---------------------------------------------------------------------
with tab_book:
    with st.form("booking_form"):
        plate = st.text_input("Vehicle number", placeholder="MH12AB1234")
        need = st.selectbox("Slot type", ["Regular", "EV", "Accessible"])
        hours = st.slider("Duration (hours)", 1, 12, 2)
        submitted = st.form_submit_button("Find & book best slot")

    if submitted:
        if not plate.strip():
            st.error("Please enter a vehicle number.")
        else:
            best = recommend_slot(slots, need)
            if best is None:
                st.warning(f"No free {need} slots right now.")
            else:
                zone_occ = forecast[(forecast.zone == best.zone) & (forecast.hour == now_hour)].occupancy.iloc[0]
                rate = dynamic_price(zone_occ)
                total = rate * hours
                idx = slots.index[slots.slot_id == best.slot_id][0]
                st.session_state.slots.loc[idx, ["status", "plate"]] = ["booked", plate.upper()]
                st.session_state.bookings.append(
                    {"slot_id": best.slot_id, "plate": plate.upper(), "hours": hours,
                     "rate": rate, "total": total, "time": datetime.now().strftime("%H:%M")}
                )
                st.session_state.flash = (f"Booked slot **{best.slot_id}** ({best.distance} m from gate). "
                                          f"₹{rate}/hr x {hours} h = **₹{total:.2f}**")
                st.rerun()  # refresh so the live map shows the new booking

# --- Manage -------------------------------------------------------------------
with tab_manage:
    booked = slots[slots.status == "booked"]
    if booked.empty:
        st.info("No active bookings.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.bookings), use_container_width=True)
        to_release = st.selectbox("Release slot", booked.slot_id)
        if st.button("Release"):
            idx = slots.index[slots.slot_id == to_release][0]
            st.session_state.slots.loc[idx, ["status", "plate"]] = ["free", ""]
            st.session_state.bookings = [b for b in st.session_state.bookings if b["slot_id"] != to_release]
            st.rerun()

# --- Insights -----------------------------------------------------------------
with tab_insights:
    st.subheader("Predicted occupancy by hour")
    pivot = forecast.pivot(index="hour", columns="zone", values="occupancy")
    st.line_chart(pivot)

    st.subheader("Best time to arrive")
    quietest = forecast.groupby("hour").occupancy.mean().nsmallest(3)
    st.write("Quietest hours: " + ", ".join(f"{h}:00 ({v:.0%})" for h, v in quietest.items()))

    st.subheader("Raw history (last 5 rows)")
    st.dataframe(history.tail(), use_container_width=True)

# --- Assistant ----------------------------------------------------------------
with tab_chat:
    st.subheader("Ask the Parking Assistant")
    st.caption("Try: 'How many vehicles are parked?', then 'what about zone B?'. "
               "Ask 'what did I ask earlier?' to see your history.")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Type your question...")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        reply = answer_question(q, slots, forecast, now_hour)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.session_state.chat_ctx = {}
        st.rerun()
