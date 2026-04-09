import streamlit as st
import duckdb
import pandas as pd

# Streamlit : Python library for building interactive web apps, ideal for data science projects
# App setup

st.set_page_config(page_title = "NBA Draft Predictor", page_icon = "🏀", layout = "wide") 
# browser tab title and icon

@st.cache_resource # create the database connection once and reuse it across app interactions for efficiency
def get_connection():
    return duckdb.connect(database = 'curr_season.duckdb', read_only = True) 
con = get_connection()


# Search Bar

st.title("NBA Draft Predictor")
st.caption("2026 Draft Class")

# render text box and returns whatever user types as string
search = st.text_input(
    label = "Search for a player",
    placeholder = "e.g. Jeremy Fears Jr.",
    help = "Type a player's name to see their draft probability and stats"
)

#query logic
if search :
    query = """ 
    SELECT *
    FROM curr
    WHERE LOWER(Player) LIKE LOWER(?)
    ORDER BY Draft_Prob DESC
    """

    # ? : placeholder for parameterize query to pass user input to SQL safely without risk of SQL injection
    # Player LIKE ? : allows partial matching of player names, so users can type just a part of the name to find the player

    results = con.execute(query, [f"%{search}%"]).fetchdf() 

    if results.empty:
        st.warning("No player found. Please try again with a different name.")
    else :
        st.success(f"Found {len(results)} player(s)")
        selected_name = st.selectbox( # handles case where search matches multiple players, lets user pick
            "Select a player", 
            options = results['Player'].tolist(),
            )
        player = results[results['Player'] == selected_name].iloc[0] # get the row of the selected player

        col1, col2 = st.columns([2, 1])
        with col1 :
            st.subheader(player["Player"])
            st.write(f"**Team:** {player['Team']} | **Position:** {player['Pos']} | **Class:** {player['Class']}") 
        with col2 :
            draft_prob = float(player['Draft_Prob'])
            st.metric(
                label = "Draft Probability",
                value = f"{draft_prob:.1%}", # format as percentage with 1 decimal place
                delta = f"{(draft_prob - 0.5):+.1%}", # show how much above/below 50% the probability is
                delta_color = "normal" # green if negative (below 50%), red if positive (above 50%)
            )
        st.divider()

        # Stats section
        st.subheader("Season Stats")
        stat_cols = st.columns(5)
        stats = {
            "PPG": "PPG",
            "RPG": "RPG",
            "APG": "APG",
            "SPG": "SPG",
            "BPG": "BPG",
        }
        for i, (label, col_name) in enumerate(stats.items()):
            with stat_cols[i] :
                if col_name in player :
                    st.metric(label = label, value = f"{player[col_name]:.1f}")
        st.divider()

        #Shoooting section
        st.subheader("Shooting")
        shoot_cols = st.columns(3)
        shooting_stats = {
            "FG%": "FG%",
            "3P%": "3P%",
            "FT%": "FT%",
        }
        for i, (label, col_name) in enumerate(shooting_stats.items()):
            with shoot_cols[i] :
                if col_name in player and pd.notna(player[col_name]) :
                    st.metric(label = label, value = f"{player[col_name]:.1%}")
                else :
                    st.metric(label = label, value = "N/A")
        st.divider()

        # Games info
        st.subheader("Games Info")
        games_cols = st.columns(3)
        with games_cols[0]:
            st.metric(label="Games Played", value=int(player["GP"]))
        with games_cols[1]:
            st.metric(label="Games Started", value=int(player["GS"]))
        with games_cols[2]:
            st.metric(label="Start Rate", value=f"{float(player['GSPG']):.1%}")

        st.divider()

        # Full data expander
        with st.expander("View all stats") :
            st.dataframe(player.iloc[1 : -3].to_frame().T, hide_index = True, use_container_width = True) 
st.divider()

st.subheader("All Players")
all_players = con.execute("""
    SELECT Player, Team, Pos, Class, GP, GS,
           ROUND(PPG, 1) AS PPG, ROUND(RPG, 1) AS RPG, ROUND(APG, 1) AS APG,
           ROUND(SPG, 1) AS SPG, ROUND(BPG, 1) AS BPG,
           "FG%", "3P%", "FT%",
           ROUND(Draft_Prob, 3) AS Draft_Prob
            FROM curr
            ORDER BY Draft_Prob DESC
            """).fetchdf()
st.dataframe(all_players, hide_index = True, use_container_width = True)
st.divider()

