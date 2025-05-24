
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import seaborn as sns
from matplotlib import pyplot as plt

# Define file paths
DATA_DIR = r"C:\Users\Asus\Documents\Work\Transfermarket\archive"
team_year_market_value_file = os.path.join(DATA_DIR, "team_year_market_value.csv")
clubs_file = os.path.join(DATA_DIR, "clubs.csv")
players_file = os.path.join(DATA_DIR, "players_with_team_value.csv")


@st.cache_data
def load_merged_data():
    """
    Loads and merges team-year market value data with clubs data.
    Returns a DataFrame with columns: year, team_id, total_market_value,
    domestic_competition_id, and team name (from 'name').
    """
    team_year_df = pd.read_csv(team_year_market_value_file)
    clubs_df = pd.read_csv(clubs_file)

    # Ensure numeric conversion
    team_year_df['total_market_value'] = pd.to_numeric(team_year_df['total_market_value'], errors='coerce')
    team_year_df['team_id'] = pd.to_numeric(team_year_df['team_id'], errors='coerce')
    team_year_df['year'] = pd.to_numeric(team_year_df['year'], errors='coerce')
    clubs_df['club_id'] = pd.to_numeric(clubs_df['club_id'], errors='coerce')

    # Merge team-year data with clubs data to get league info and team names
    merged_df = team_year_df.merge(
        clubs_df[['club_id', 'domestic_competition_id', 'name']],
        left_on='team_id',
        right_on='club_id',
        how='left'
    )
    merged_df.drop(columns=['club_id'], inplace=True)

    return merged_df


def plot_market_value_box(merged_df):
    """
    Displays a box plot of total market value by league for the selected year.
    Allows filtering by year or showing all years.
    """
    # **YEAR FILTER** (Dropdown with "All Years" option)
    year_options = ["All Years"] + sorted(merged_df['year'].dropna().unique())
    selected_year = st.selectbox("📅 Select Year for Market Value Distribution", year_options, key="year_filter_box")

    if selected_year == "All Years":
        filtered_data = merged_df  # Show all years
    else:
        filtered_data = merged_df[merged_df['year'] == selected_year]

    if filtered_data.empty:
        st.warning(f"No data available for the selected year ({selected_year})")
        return None
    else:
        fig = px.box(
            filtered_data,
            x="domestic_competition_id",
            y="total_market_value",
            hover_data=["name"],  # Show team name on hover
            title=f"Club Market Value Distribution by League ({selected_year})",
            labels={"domestic_competition_id": "League", "total_market_value": "Market Value (EUR)"}
        )
        st.plotly_chart(fig, use_container_width=True)


def calculate_yearly_market_value_changes(merged_df):
    """
    Calculates the year-over-year percentage change in average market value per league.

    Returns:
        pd.DataFrame: Table showing yearly percentage changes.
    """
    avg_values = merged_df.groupby(['year', 'domestic_competition_id'])['total_market_value'].mean().reset_index()
    pivot_table = avg_values.pivot(index='domestic_competition_id', columns='year', values='total_market_value')

    # Compute percentage change year-over-year
    percentage_change = pivot_table.pct_change(axis=1) * 100
    percentage_change.columns = [f"Change_{col}" for col in percentage_change.columns]

    # Merge tables to keep only percentage changes
    final_table = pd.concat([percentage_change], axis=1).reset_index()

    return final_table



def plot_market_value_trend(merged_df):
    """
    Plots the percentage change in average market value over the years for selected leagues.
    Includes league selection inside the function, keeping main() empty.
    """

    st.subheader("📈 Market Value Trend Over the Years")

    # Select leagues for market value trend visualization
    league_options = sorted(merged_df['domestic_competition_id'].dropna().astype(str).unique())
    selected_leagues = st.multiselect(
        "🏆 Select Leagues to Compare Market Value Trend", league_options, default=league_options[:5]
    )

    if not selected_leagues:
        st.warning("⚠ Please select at least one league to visualize the trend.")
        return

    # Compute yearly market value changes
    change_df = calculate_yearly_market_value_changes(merged_df)
    change_df.reset_index(inplace=True, drop=True)

    # Convert wide format to long format for plotting
    melted_df = change_df.melt(id_vars=["domestic_competition_id"], var_name="Year", value_name="Percentage Change")

    # Filter only numeric years and remove 'Change_' prefix
    melted_df = melted_df[melted_df["Year"].str.startswith("Change_")]
    melted_df["Year"] = melted_df["Year"].str.replace("Change_", "").astype(int)

    # Remove NaN values from Percentage Change
    melted_df = melted_df.dropna(subset=["Percentage Change"])

    # Filter only selected leagues
    melted_df = melted_df[melted_df['domestic_competition_id'].isin(selected_leagues)]

    # Plot using Plotly
    fig = px.line(
        melted_df,
        x="Year",
        y="Percentage Change",
        color="domestic_competition_id",
        markers=True,
        title="📊 Yearly Market Value Percentage Change by League",
        labels={"domestic_competition_id": "League", "Percentage Change": "Market Value Change (%)"}
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def load_players_data():
        file_path = "C:\\Users\\Asus\\Documents\\Work\\Transfermarket\\archive\\players_with_team_value.csv"
        df = pd.read_csv(file_path)
        df['market_value_million'] = df['market_value_in_eur_y'] / 1_000_000  # Convert to million euros
        df['season'] = pd.to_numeric(df['season'], errors='coerce')
        return df


def plot_market_value_over_seasons(players_df):
    """
    Plots the market value of players over the seasons with options to:
    1. Show all players in a single scatter plot
    2. Filter by position and create 4 separate small graphs
    3. Ensure each season of a player is represented as a unique point
    """
    st.subheader("📈 Market Value Over Seasons")

    # Ensure one data point per player per season
    players_df = players_df.groupby(['player_name', 'season'], as_index=False).first()

    # Checkbox to filter by position
    use_position_filter = st.checkbox("Filter by Position", key="position_filter_mv")

    # Position filter (Only if selected)
    unique_positions = sorted(players_df['position'].dropna().unique())
    selected_positions = []

    if use_position_filter:
        selected_positions = st.multiselect(
            "🔍 Select Positions", unique_positions, default=unique_positions[:4], key="position_multiselect_mv"
        )

    # Checkbox for Top 10% Market Value Players
    top_10_percent = st.checkbox("Show only Top 10% Market Value Players", key="top_10_percent_mv")

    if top_10_percent:
        threshold = players_df["market_value_million"].quantile(0.9)
        players_df = players_df[players_df["market_value_million"] >= threshold]

    # Define color palette
    color_palette = ["#082a54", "#e02b35", "#59a89c", "#a559aa"]  # Dark Blue, Red, Teal, Purple

    if use_position_filter and selected_positions:
        # Create 4 smaller scatter plots for selected positions
        cols = st.columns(len(selected_positions))
        for i, position in enumerate(selected_positions):
            filtered_df = players_df[players_df['position'] == position]

            fig = px.scatter(
                filtered_df,
                x='season',
                y='market_value_million',
                opacity=0.6,
                color_discrete_sequence=[color_palette[i % len(color_palette)]],  # Cycle colors
                title=f"{position} Market Value Trend",
                labels={"season": "Season", "market_value_million": "Market Value (Million €)"},
                hover_data=['player_name'],  # Show player name on hover
            )
            with cols[i]:  # Assign each graph to a separate column
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Show a single scatter plot for all players
        fig = px.scatter(
            players_df,
            x='season',
            y='market_value_million',
            opacity=0.6,
            color_discrete_sequence=["#e02b35"],  # Red
            title="Market Value Trend of Players Over the Years (All Players)",
            labels={"season": "Season", "market_value_million": "Market Value (Million €)"},
            hover_data=['player_name'],  # Show player name on hover
        )
        st.plotly_chart(fig, use_container_width=True)




def plot_metric_vs_market_value_by_position(players_df):
    """
    Plots selected performance metrics vs market value with options to:
    1. Show all players in a single scatter plot
    2. Filter by position and create 4 separate small graphs, each with a different color
    3. Filter by year or show all years
    4. Ensure each season of a player is a unique point
    """
    st.subheader("🎯 Performance Impact on Market Value")

    # Add clean_sheet to the available metrics
    metric_options = [
        "yellow_cards", "red_cards", "goals", "assists",
        "minutes_played", "games_played", "clean_sheet"
    ]
    selected_metric = st.selectbox("📊 Select Metric", metric_options, key="metric_select")

    # Year Filter
    year_options = ["All Years"] + sorted(players_df['season'].dropna().unique())
    selected_year = st.selectbox("📅 Select Year", year_options, key="year_filter_metric")
    if selected_year != "All Years":
        players_df = players_df[players_df['season'] == selected_year]

    # Position Filter
    use_position_filter = st.checkbox("Filter by Position", key="position_filter_metric")
    unique_positions = sorted(players_df['position'].dropna().unique())
    selected_positions = []
    if use_position_filter:
        selected_positions = st.multiselect(
            "🔍 Select Positions", unique_positions, default=unique_positions[:4],
            key="position_multiselect_metric"
        )

    # Top 10% Market Value Filter
    top_10_percent = st.checkbox("Show only Top 10% Market Value Players", key="top_10_percent_metric")
    if top_10_percent:
        threshold = players_df["market_value_million"].quantile(0.9)
        players_df = players_df[players_df["market_value_million"] >= threshold]

    # Position-based colors
    position_colors = {
        "Attack": "#e02b35",     # Red
        "Defender": "#f0c571",   # Gold
        "Midfield": "#082a54",   # Dark Blue
        "Goalkeeper": "#59a89c", # Teal
    }

    # Metric-based color (fallback if not using position view)
    metric_colors = {
        "yellow_cards": "#f0c571",
        "red_cards": "#e02b35",
        "goals": "#082a54",
        "assists": "#a559aa",
        "minutes_played": "#59a89c",
        "games_played": "#cecece",
        "clean_sheet": "#59a89c",  # Same as Goalkeeper color
    }

    if use_position_filter and selected_positions:
        cols = st.columns(len(selected_positions))
        for i, position in enumerate(selected_positions):
            filtered_df = players_df[players_df['position'] == position]
            position_color = position_colors.get(position, "#082a54")

            fig = px.scatter(
                filtered_df,
                x=selected_metric,
                y='market_value_million',
                opacity=0.6,
                color_discrete_sequence=[position_color],
                title=f"{position} - Impact of {selected_metric.replace('_', ' ').title()} on Market Value",
                labels={
                    selected_metric: selected_metric.replace("_", " ").title(),
                    "market_value_million": "Market Value (Million €)"
                },
                hover_data=['player_name'],
            )
            with cols[i]:
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Plot for all players
        fig = px.scatter(
            players_df,
            x=selected_metric,
            y='market_value_million',
            opacity=0.6,
            color_discrete_sequence=[metric_colors.get(selected_metric, "#082a54")],
            title=f"Impact of {selected_metric.replace('_', ' ').title()} on Market Value ({selected_year})",
            labels={
                selected_metric: selected_metric.replace("_", " ").title(),
                "market_value_million": "Market Value (Million €)"
            },
            hover_data=['player_name'],
        )
        st.plotly_chart(fig, use_container_width=True)




def plot_player_attributes_bar(players_df):
    """
    Plots bar charts for categorical player attributes, showing the count of players
    in each category.

    Attributes:
    - foot (Left, Right, Both)
    - country_of_citizenship
    - age
    - height_in_cm
    - position
    - sub_position
    """

    st.subheader("📊 Player Attribute Distribution")

    attribute_options = ["foot", "country_of_citizenship", "age", "height_in_cm", "position", "sub_position"]
    selected_attribute = st.selectbox("📋 Select Attribute to Analyze", attribute_options, key="attribute_select")

    # **YEAR FILTER** (Dropdown with "All Years" option)
    year_options = ["All Years"] + sorted(players_df['season'].dropna().unique())
    selected_year = st.selectbox("📅 Select Year", year_options, key="year_filter_attribute")

    if selected_year != "All Years":
        players_df = players_df[players_df['season'] == selected_year]

    # **POSITION FILTER** (Checkbox to filter by position)
    use_position_filter = st.checkbox("Filter by Position", key="position_filter_attribute")

    unique_positions = sorted(players_df['position'].dropna().unique())
    selected_positions = []

    if use_position_filter:
        selected_positions = st.multiselect(
            "🔍 Select Positions", unique_positions, default=unique_positions[:4], key="position_multiselect_attribute"
        )

    # **FILTERING FOR TOP 10% MARKET VALUE PLAYERS**
    top_10_percent = st.checkbox("Show only Top 10% Market Value Players", key="top_10_percent_attribute")

    if top_10_percent:
        threshold = players_df["market_value_million"].quantile(0.9)
        players_df = players_df[players_df["market_value_million"] >= threshold]

    # **Position-Specific Colors**
    position_colors = {
        "Attack": "#e02b35",  # Red
        "Defender": "#f0c571",  # Gold
        "Midfield": "#082a54",  # Dark Blue
        "Goalkeeper": "#59a89c",  # Teal
    }

    # Group by selected attribute and count occurrences
    attribute_counts = players_df[selected_attribute].value_counts().reset_index()
    attribute_counts.columns = [selected_attribute, "count"]  # Rename columns properly

    if use_position_filter and selected_positions:
        # Show 4 smaller bar plots for selected positions (Each with a Different Color)
        cols = st.columns(len(selected_positions))
        for i, position in enumerate(selected_positions):
            filtered_df = players_df[players_df['position'] == position]

            # Group by selected attribute and count occurrences
            filtered_counts = filtered_df[selected_attribute].value_counts().reset_index()
            filtered_counts.columns = [selected_attribute, "count"]

            # Get position-specific color (default to blue if not found)
            position_color = position_colors.get(position, "#082a54")

            fig = px.bar(
                filtered_counts,
                x=selected_attribute,
                y="count",
                color_discrete_sequence=[position_color],  # Assign unique position color
                title=f"{position} - Distribution of {selected_attribute.replace('_', ' ').title()}",
                labels={selected_attribute: selected_attribute.replace("_", " ").title(), "count": "Number of Players"},
                text_auto=True
            )
            with cols[i]:  # Assign each graph to a separate column
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Show a single bar chart for all players
        fig = px.bar(
            attribute_counts,
            x=selected_attribute,
            y="count",
            color_discrete_sequence=["#082a54"],  # Dark Blue for all players
            title=f"Distribution of {selected_attribute.replace('_', ' ').title()} ({selected_year})",
            labels={selected_attribute: selected_attribute.replace("_", " ").title(), "count": "Number of Players"},
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_player_pairwise_relationships(players_df):
    """
    Generates pair plots (scatter matrix) for key player attributes to explore relationships.
    Includes filters for year and top 10% market value players.
    """
    st.subheader("🔗 Pairwise Player Attribute Relationships")

    # Relevant numeric columns for correlation
    numeric_cols = [
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played", "games_played",
        "market_value_in_eur_y", "highest_market_value_in_eur", "age", "height_in_cm", "total_team_market_value",
        "clean_sheet"
    ]

    # Filter: Year
    year_options = ["All Years"] + sorted(players_df['season'].dropna().unique())
    selected_year = st.selectbox("📅 Select Year", year_options, key="pairplot_year_filter")

    if selected_year != "All Years":
        players_df = players_df[players_df["season"] == selected_year]

    # Filter: Top 10% Market Value
    top_10_percent = st.checkbox("Show only Top 10% Market Value Players", key="pairplot_top10_filter")
    if top_10_percent:
        threshold = players_df["market_value_in_eur_y"].quantile(0.9)
        players_df = players_df[players_df["market_value_in_eur_y"] >= threshold]

    # Drop rows with missing values in selected columns
    filtered_df = players_df[numeric_cols].dropna()

    if filtered_df.empty:
        st.warning("⚠ No data available for the selected filters.")
        return

    # Plot pairwise relationships using seaborn
    st.info("🔄 Generating Pairplot... This may take a few seconds.")
    sns.set(style="whitegrid")
    fig = sns.pairplot(filtered_df, diag_kind="kde", corner=False, plot_kws={"s": 10, "alpha": 0.6})

    # Display in Streamlit
    st.pyplot(fig.fig)


def plot_correlation_heatmap(players_df):
    """
    Generates a Plotly heatmap of the Pearson correlation between player features.
    Color scale is set between -0.2 and 1 for better contrast.
    Includes filters for year, top 10% market value players, and position.
    """
    st.subheader("🔥 Correlation Heatmap of Player Features (-0.2 to 1 Scale)")

    # Relevant numeric columns
    numeric_cols = [
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played", "games_played",
        "market_value_in_eur_y", "highest_market_value_in_eur", "age", "height_in_cm",
        "total_team_market_value", "clean_sheet"
    ]

    # Year filter
    year_options = ["All Years"] + sorted(players_df["season"].dropna().unique())
    selected_year = st.selectbox("📅 Select Year", year_options, key="heatmap_year_filter")
    if selected_year != "All Years":
        players_df = players_df[players_df["season"] == selected_year]

    # Position filter
    use_position_filter = st.checkbox("Filter by Position", key="heatmap_position_filter")
    if use_position_filter:
        unique_positions = sorted(players_df["position"].dropna().unique())
        selected_positions = st.multiselect(
            "🔍 Select Positions", unique_positions, default=unique_positions[:4], key="heatmap_position_multiselect"
        )
        if selected_positions:
            players_df = players_df[players_df["position"].isin(selected_positions)]

    # Top 10% market value filter
    top_10_percent = st.checkbox("Show only Top 10% Market Value Players", key="heatmap_top10_filter")
    if top_10_percent:
        threshold = players_df["market_value_in_eur_y"].quantile(0.9)
        players_df = players_df[players_df["market_value_in_eur_y"] >= threshold]

    # Drop rows with missing values in selected columns
    filtered_df = players_df[numeric_cols].dropna()

    if filtered_df.empty:
        st.warning("⚠ No data available for the selected filters.")
        return

    # Pearson correlation matrix
    corr_matrix = filtered_df.corr().round(2)

    # Plot using Plotly
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-0.2,
        zmax=1,
        title="📊 Pearson Correlation Between Player Attributes",
        labels=dict(x="Feature", y="Feature", color="Correlation")
    )

    fig.update_layout(
        width=900,
        height=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
    """
    Main function to:
    1. Load data
    2. Display Market Value Distribution for a selected year (Box Plot)
    3. Display Market Value Trend Over the Years (Line Chart)
    4. Show Market Value Change Table (Only One Table)
    5. Analyze player market values over seasons and based on performance metrics
    6. Optionally filter by position for a more detailed breakdown
    """
    st.title("📊 Club & Player Market Value Trends")

    # Load merged data for teams and leagues
    merged_df = load_merged_data()

    # Load players data separately for player analysis
    players_df = load_players_data()
    st.write(players_df.columns.tolist())
    plot_market_value_box(merged_df)
    plot_market_value_trend(merged_df)


    # --- ONLY SHOW ONE TABLE ---
    st.subheader("📋 Yearly Market Value Percentage Change Table")
    change_df = calculate_yearly_market_value_changes(merged_df)

    if not change_df.empty:
        st.dataframe(change_df, use_container_width=True)
    else:
        st.warning("⚠ No market value data available for trend analysis.")

    ### --- PLAYER MARKET VALUE ANALYSIS --- ###
    st.header("⚽ Player Market Value Analysis")

    # Market Value over Seasons (Scatter Plot)
    plot_market_value_over_seasons(players_df)

    # Select a performance metric to analyze its impact on market value (With Position Filtering Option)
    plot_metric_vs_market_value_by_position(players_df)
    plot_player_attributes_bar(players_df)
    plot_player_pairwise_relationships(players_df)
    plot_correlation_heatmap(players_df)





if __name__ == "__main__":
    main()

